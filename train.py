# file: train.py
#
# V-parameterization diffusion (Salimans & Ho 2022).
#
# Key identities (all derived from the forward process):
#   x_t  = sqrt_a * x0 + sqrt_m * eps          (forward sample)
#   v    = sqrt_a * eps - sqrt_m * x0           (v-target definition)
#
#   Inverse relations:
#   x0   = sqrt_a * x_t - sqrt_m * v            (predict x0 from v)
#   eps  = sqrt_m * x_t + sqrt_a * v            (predict eps from v)  ← note order
#
# The paper (Diff4MMLiTS, eq.2-4) uses eps-parameterisation in the LDM,
# but we keep v-param here because it is better conditioned at low
# timesteps (TIMESTEPS=300).  The training loss (p_losses) and the
# reverse sampler (sample) are now fully consistent.

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.tensor(np.clip(betas, 1e-4, 0.9999), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

class Diffusion:
    def __init__(self, timesteps=1000, device="cuda", use_cosine=True):
        self.timesteps = timesteps
        self.device    = torch.device(device)

        betas = (cosine_beta_schedule(timesteps)
                 if use_cosine else
                 linear_beta_schedule(timesteps))

        self.betas    = betas.to(self.device)
        self.alphas   = 1.0 - self.betas
        self.alpha_cum = torch.cumprod(self.alphas, dim=0)

        # pre-computed convenience terms
        self.sqrt_alpha_cum          = torch.sqrt(self.alpha_cum)
        self.sqrt_one_minus_alpha_cum = torch.sqrt(1.0 - self.alpha_cum)

    # ------------------------------------------------------------------
    # helpers for indexing schedules at arbitrary batch timesteps t
    # ------------------------------------------------------------------
    def _get(self, arr, t):
        """Return arr[t] shaped (B,1,1,1) for broadcasting with image tensors."""
        return arr[t].view(-1, 1, 1, 1)

    # ------------------------------------------------------------------
    # Forward process  q(x_t | x_0)
    # ------------------------------------------------------------------
    def q_sample(self, x0, noise, t):
        """Sample x_t given x_0 and pre-drawn noise."""
        a = self._get(self.sqrt_alpha_cum,           t)
        b = self._get(self.sqrt_one_minus_alpha_cum, t)
        return a * x0 + b * noise

    # ------------------------------------------------------------------
    # V-parameterisation utilities
    # ------------------------------------------------------------------
    def get_v_target(self, x0, noise, t):
        """Compute the v-target:  v = sqrt_a * eps - sqrt_m * x0."""
        a = self._get(self.sqrt_alpha_cum,           t)
        b = self._get(self.sqrt_one_minus_alpha_cum, t)
        return a * noise - b * x0

    def predict_x0_from_v(self, xt, v_pred, t):
        """Recover x0 from x_t and predicted v:  x0 = sqrt_a * x_t - sqrt_m * v."""
        a = self._get(self.sqrt_alpha_cum,           t)
        b = self._get(self.sqrt_one_minus_alpha_cum, t)
        return a * xt - b * v_pred

    def predict_eps_from_v(self, xt, v_pred, t):
        """Recover eps from x_t and predicted v:  eps = sqrt_m * x_t + sqrt_a * v.

        Derivation:
            x_t = sqrt_a * x0  + sqrt_m * eps      ... (1)
            v   = sqrt_a * eps - sqrt_m * x0        ... (2)
            From (1): x0 = (x_t - sqrt_m * eps) / sqrt_a
            Substitute into (2):
            v = sqrt_a * eps - sqrt_m * (x_t - sqrt_m * eps) / sqrt_a
              = eps * (sqrt_a + sqrt_m^2 / sqrt_a) - sqrt_m / sqrt_a * x_t
              = eps / sqrt_a - sqrt_m / sqrt_a * x_t      [since sqrt_a^2 + sqrt_m^2 = 1]
            => eps = sqrt_a * v + sqrt_m * x_t            [multiply both sides by sqrt_a]

        NOTE: the coefficient on v is sqrt_a (=sqrt_alpha_cum) and on x_t is
        sqrt_m (=sqrt_one_minus_alpha_cum).  This is the fix for the previously
        swapped coefficients.
        """
        a = self._get(self.sqrt_alpha_cum,           t)
        b = self._get(self.sqrt_one_minus_alpha_cum, t)
        return b * xt + a * v_pred          # eps = sqrt_m * x_t + sqrt_a * v

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------
    def p_losses(self, model, x_start, cond, t, use_v=True, drop_cond=False):
        """
        Compute diffusion training loss.

        Args:
            model      : denoising network
            x_start    : clean image x0,  shape (B, C, H, W)
            cond       : conditioning tensor
            t          : integer timestep tensor, shape (B,)
            use_v      : True → v-parameterisation (recommended)
                         False → eps-parameterisation (as in Diff4MMLiTS eq.4)
            drop_cond  : classifier-free guidance dropout flag

        Returns:
            loss, pred, x0_hat, target
        """
        # --- normalise t ---
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x_start.device)
        t = t.long()
        if t.dim() == 0:
            t = t.unsqueeze(0)
        B = x_start.shape[0]
        if t.shape[0] != B:
            t = t.expand(B)

        noise = torch.randn_like(x_start)
        x_t   = self.q_sample(x_start, noise, t)

        pred = model(x_t, cond, t, drop_cond=drop_cond)

        if use_v:
            target = self.get_v_target(x_start, noise, t)
            loss   = F.mse_loss(pred, target)
            x0_hat = self.predict_x0_from_v(x_t, pred, t)
        else:
            # eps-parameterisation (matches Diff4MMLiTS eq.4)
            target = noise
            loss   = F.mse_loss(pred, noise)
            a      = self._get(self.sqrt_alpha_cum,           t)
            b      = self._get(self.sqrt_one_minus_alpha_cum, t)
            x0_hat = (x_t - b * pred) / a

        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
        return loss, pred, x0_hat, target

    # ------------------------------------------------------------------
    # Reverse sampler  (DDPM ancestral sampler with CFG)
    # ------------------------------------------------------------------
    def sample(
        self,
        model,
        cond,
        shape,
        timesteps=None,
        guidance_scale=2.0,
        use_v=True,
        device=None,
    ):
        """
        Ancestral DDPM sampler with classifier-free guidance.

        CFG is applied in the *prediction space* (v or eps), then a single
        consistent DDPM update is performed.  The two quantities are never
        mixed across parameterisations.

        For v-parameterisation the update is:
            x0_hat  = sqrt_a * x_t - sqrt_m * v_guided   (clipped)
            eps_hat = sqrt_m * x_t + sqrt_a * v_guided    (derived, not re-predicted)
            x_{t-1} = (x_t - beta/sqrt(1-alpha_cum) * eps_hat) / sqrt(alpha)
                      + sigma * z                          (z ~ N(0,I) if t > 0)

        This keeps everything in one coherent parameterisation throughout.
        """
        device    = torch.device(device) if device is not None else self.device
        timesteps = self.timesteps if timesteps is None else timesteps

        was_training = model.training
        model.eval()

        B = cond.shape[0]
        x = torch.randn(shape, device=device)

        # move all schedule tensors to the sampling device
        betas                  = self.betas.to(device)
        alphas                 = self.alphas.to(device)
        alpha_cum              = self.alpha_cum.to(device)
        sqrt_alpha_cum         = self.sqrt_alpha_cum.to(device)
        sqrt_one_minus_alpha_cum = self.sqrt_one_minus_alpha_cum.to(device)

        cond = cond.to(device)

        with torch.no_grad():
            for t_idx in reversed(range(timesteps)):
                t = torch.full((B,), t_idx, device=device, dtype=torch.long)

                # ── schedule values at this step ──────────────────────────
                sqrt_a = sqrt_alpha_cum[t].view(B, 1, 1, 1)
                sqrt_m = sqrt_one_minus_alpha_cum[t].view(B, 1, 1, 1)
                beta   = betas[t].view(B, 1, 1, 1)
                alpha  = alphas[t].view(B, 1, 1, 1)
                a_cum  = alpha_cum[t].view(B, 1, 1, 1)

                if use_v:
                    # ── v-parameterisation path ───────────────────────────
                    # 1. Get conditional and unconditional v-predictions
                    v_cond   = model(x, cond, t, drop_cond=False)
                    v_uncond = model(x, cond, t, drop_cond=True)

                    # 2. Apply CFG in v-space
                    v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)

                    # 3. Derive eps consistently from the guided v
                    #    eps = sqrt_m * x_t + sqrt_a * v   (see predict_eps_from_v)
                    eps_hat = sqrt_m * x + sqrt_a * v_guided

                    # 4. Optionally clip x0 for stability
                    # x0_hat = sqrt_a * x - sqrt_m * v_guided
                    # x0_hat = x0_hat.clamp(-1, 1)
                    # eps_hat = (x - sqrt_a * x0_hat) / sqrt_m  # re-derive if clipping

                else:
                    # ── eps-parameterisation path (matches Diff4MMLiTS eq.4) ──
                    eps_cond   = model(x, cond, t, drop_cond=False)
                    eps_uncond = model(x, cond, t, drop_cond=True)
                    eps_hat    = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

                # ── DDPM reverse step (same formula for both parameterisations) ──
                # x_{t-1} = 1/sqrt(alpha) * (x_t - beta/sqrt(1-alpha_cum) * eps)
                x_prev = (1.0 / torch.sqrt(alpha)) * (
                    x - (beta / torch.sqrt(1.0 - a_cum)) * eps_hat
                )

                if t_idx > 0:
                    sigma = torch.sqrt(beta)
                    x     = x_prev + sigma * torch.randn_like(x)
                else:
                    x = x_prev

        if was_training:
            model.train()
        else:
            model.eval()

        return x