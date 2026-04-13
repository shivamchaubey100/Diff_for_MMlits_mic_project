# file: train.py
import torch
import torch.nn.functional as F
import numpy as np


def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-4, 0.9999)


class Diffusion:
    def __init__(self, timesteps=1000, device="cuda", use_cosine=True):
        self.timesteps = timesteps
        self.device = torch.device(device)
        betas_src = cosine_beta_schedule(timesteps) if use_cosine else None
        if betas_src is None:
            betas = linear_beta_schedule(timesteps)
        else:
            betas = torch.tensor(betas_src, dtype=torch.float32)
        # ensure schedules live on the configured device
        self.betas = betas.to(self.device, dtype=torch.float32)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alpha_cum = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.sqrt_alpha_cum = torch.sqrt(self.alpha_cum).to(self.device)
        self.sqrt_one_minus_alpha_cum = torch.sqrt(1.0 - self.alpha_cum).to(self.device)

    def q_sample(self, x0, noise, t):
        a = self.sqrt_alpha_cum[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_cum[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise

    def get_v_target(self, x0, noise, t):
        a = self.sqrt_alpha_cum[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_cum[t].view(-1, 1, 1, 1)
        return a * noise - b * x0

    def predict_x0_from_v(self, xt, v_pred, t):
        a = self.sqrt_alpha_cum[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_cum[t].view(-1, 1, 1, 1)
        return a * xt - b * v_pred

    def p_losses(self, model, x_start, cond, t, use_v=False, drop_cond=False):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x_start.device)
        t = t.long()
        if t.dim() == 0:
            t = t.unsqueeze(0)
        B = x_start.shape[0]
        if t.shape[0] != B:
            t = t.expand(B)

        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, noise, t)

        pred = model(x_t, cond, t, drop_cond=drop_cond)

        if use_v:
            # v-parameterization
            target = self.get_v_target(x_start, noise, t)
            loss = F.mse_loss(pred, target)
            pred_v = pred
            x0_hat = self.predict_x0_from_v(x_t, pred_v, t)
            x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
        else:
            # eps-parameterization
            target = noise
            loss = F.mse_loss(pred, noise)
            sqrt_alpha_cum = self.sqrt_alpha_cum[t].view(B,1,1,1)
            sqrt_one_minus = self.sqrt_one_minus_alpha_cum[t].view(B,1,1,1)
            x0_hat = (x_t - sqrt_one_minus * pred) / sqrt_alpha_cum
            x0_hat = torch.clamp(x0_hat, -1, 1)

        return loss, pred, x0_hat, target


    def sample(self, model, cond, shape, timesteps=None, guidance_scale=2.0, device=None):
        # device selection and ensure tensors on correct device
        device = torch.device(device) if device is not None else self.device
        timesteps = self.timesteps if timesteps is None else timesteps

        # evaluation + no grad
        was_training = model.training
        model.eval()

        B = cond.shape[0]
        x = torch.randn(shape, device=device)

        # move schedules to sampling device (no-op if already there)
        betas = self.betas.to(device)
        alphas = self.alphas.to(device)
        alpha_cum = self.alpha_cum.to(device)
        sqrt_alpha_cum = self.sqrt_alpha_cum.to(device)
        sqrt_one_minus_alpha_cum = self.sqrt_one_minus_alpha_cum.to(device)

        # ensure cond on device
        cond = cond.to(device)

        with torch.no_grad():
            for t_idx in reversed(range(timesteps)):
                t = torch.full((B,), t_idx, device=device, dtype=torch.long)

                # v predictions: conditional & unconditional
                v_cond = model(x, cond, t, drop_cond=False)
                v_uncond = model(x, cond, t, drop_cond=True)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)

                # convert v -> eps
                sqrt_a = sqrt_alpha_cum[t].view(B,1,1,1)
                sqrt_m = sqrt_one_minus_alpha_cum[t].view(B,1,1,1)
                eps = sqrt_a * v + sqrt_m * x

                beta = betas[t].view(B,1,1,1)
                alpha = alphas[t].view(B,1,1,1)
                alpha_cum_t = alpha_cum[t].view(B,1,1,1)

                sigma = torch.sqrt(beta)

                x_prev = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cum_t)) * eps)

                if t_idx > 0:
                    x = x_prev + sigma * torch.randn_like(x)
                else:
                    x = x_prev

        # restore train/eval state
        if was_training:
            model.train()
        else:
            model.eval()

        return x
