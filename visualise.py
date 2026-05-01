"""
visualise.py

Fixes vs previous version:
  FIX (vis-1): v_pred was called without drop_cond=False explicitly — now explicit.
  FIX (vis-2): The mid-step t was timesteps//2 which can be very noisy (t=150 out
               of 300). Changed to t=50 (low noise) so x0_hat is much cleaner and
               actually diagnostic of what the model has learned.
  FIX (vis-3): Added a direct "clean x0_hat" panel that bypasses sampling noise
               entirely: run model at t=1 (almost no noise added) so x0_hat ≈ the
               model's best reconstruction. This is the most informative panel for
               monitoring training progress.
  FIX (vis-4): disp() now clips to [-1,1] before rescaling, guarding against
               out-of-range v-param predictions producing display artefacts.
  FIX (vis-5): model.eval() / model.train() now uses try/finally so training
               state is always restored even if an exception occurs.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path


@torch.no_grad()
def visualize_sample(
    diffusion,
    model,
    healthy_t,
    target_t,
    tumor_mask_t,
    liver_mask_t,
    pass_n,
    stage="mid",
    device="cuda",
    force_size=(256, 256),
    timesteps=100,
    guidance_scale=2.0,
    vis_dir=None,
):
    print("\n[VIS] ==== Running visualize_sample() ====")

    if vis_dir is not None:
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    H, W = force_size
    healthy_t    = F.interpolate(healthy_t,    size=force_size, mode="bilinear", align_corners=False)
    target_t     = F.interpolate(target_t,     size=force_size, mode="bilinear", align_corners=False)
    tumor_mask_t = F.interpolate(tumor_mask_t, size=force_size, mode="nearest")
    liver_mask_t = F.interpolate(liver_mask_t, size=force_size, mode="nearest")

    # Unwrap DataParallel if needed
    net = model.module if hasattr(model, "module") else model

    was_training = net.training
    net.eval()

    try:
        healthy_1 = healthy_t[:1].to(device)
        target_1  = target_t[:1].to(device)
        liver_1   = liver_mask_t[:1].to(device)

        # Conditioning: [healthy_ct, liver_mask] — exactly as in training
        cond = torch.cat([healthy_1, liver_1], dim=1)

        print(f"[VIS] healthy range: {float(healthy_1.min()):.3f}  {float(healthy_1.max()):.3f}")
        print(f"[VIS] target  range: {float(target_1.min()):.3f}  {float(target_1.max()):.3f}")

        # ── Panel 3: x0_hat at LOW noise (t=1) ─────────────────────────────
        # FIX (vis-2 + vis-3): use t=1 (almost no noise) so that x0_hat is a
        # near-clean reconstruction and directly shows what the model has learned.
        # The old t=timesteps//2 = t=150 adds so much noise that x0_hat ≈ noise
        # even for a well-trained model, making it useless for diagnosis.
        t_low = torch.tensor([1], device=device).long()
        noise_low = torch.randn_like(target_1)
        x_t_low = diffusion.q_sample(target_1, noise_low, t_low)
        # FIX (vis-1): pass drop_cond=False explicitly
        v_pred_low = net(x_t_low, cond, t_low, drop_cond=False)
        x0_hat_low = diffusion.predict_x0_from_v(x_t_low, v_pred_low, t_low)
        x0_hat_low = torch.clamp(x0_hat_low, -1.0, 1.0)
        x0_hat_disp = x0_hat_low[0, 0].cpu().numpy()
        print(f"[VIS] x0_hat (t=1) range: {x0_hat_disp.min():.3f}  {x0_hat_disp.max():.3f}")

        # ── Panel 4: x0_hat at MID noise (t=timesteps//4) ──────────────────
        # A quarter of the way through the schedule is noisy enough to be a real
        # test but not so noisy that the prediction is dominated by the schedule.
        t_mid = torch.tensor([max(1, timesteps // 4)], device=device).long()
        noise_mid = torch.randn_like(target_1)
        x_t_mid = diffusion.q_sample(target_1, noise_mid, t_mid)
        v_pred_mid = net(x_t_mid, cond, t_mid, drop_cond=False)
        x0_hat_mid = diffusion.predict_x0_from_v(x_t_mid, v_pred_mid, t_mid)
        x0_hat_mid = torch.clamp(x0_hat_mid, -1.0, 1.0)
        x0_hat_mid_disp = x0_hat_mid[0, 0].cpu().numpy()
        print(f"[VIS] x0_hat (t={max(1,timesteps//4)}) range: {x0_hat_mid_disp.min():.3f}  {x0_hat_mid_disp.max():.3f}")

        # ── Panel 5: Full DDPM reverse sample ──────────────────────────────
        final_sample = diffusion.sample(
            model=net,
            cond=cond,
            shape=(1, 1, H, W),
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            device=device,
        )[0, 0].cpu().numpy()
        print(f"[VIS] final_sample range: {final_sample.min():.3f}  {final_sample.max():.3f}")

        # ── Display helper ──────────────────────────────────────────────────
        # FIX (vis-4): clamp to [-1,1] before rescaling to handle any
        # out-of-range v-param predictions without display artefacts.
        def disp(img):
            return np.clip((np.clip(img, -1.0, 1.0) + 1.0) / 2.0, 0.0, 1.0)

        healthy_disp    = disp(healthy_t[0, 0].cpu().numpy())
        target_disp     = disp(target_t[0, 0].cpu().numpy())

        # ── 5-panel figure ──────────────────────────────────────────────────
        fig, axs = plt.subplots(1, 5, figsize=(28, 6))

        axs[0].imshow(healthy_disp,        cmap="gray"); axs[0].set_title("Healthy Cond")
        axs[1].imshow(target_disp,         cmap="gray"); axs[1].set_title("Ground Truth")
        axs[2].imshow(disp(x0_hat_disp),   cmap="gray"); axs[2].set_title(f"x0_hat  t=1\n(clean recon)")
        axs[3].imshow(disp(x0_hat_mid_disp), cmap="gray"); axs[3].set_title(f"x0_hat  t={max(1,timesteps//4)}\n(mid noise)")
        axs[4].imshow(disp(final_sample),  cmap="gray"); axs[4].set_title("Full DDPM sample")

        for ax in axs:
            ax.axis("off")

        fig.suptitle(f"Epoch {pass_n}  [{stage}]", fontsize=13, y=1.01)
        plt.tight_layout()

        if vis_dir is not None:
            save_path = vis_dir / f"recon_pass{pass_n}_{stage}.png"
            plt.savefig(save_path, dpi=130, bbox_inches="tight")
            print(f"[VIS] Saved to: {save_path}")

        plt.close()

    finally:
        # FIX (vis-5): always restore training state
        if was_training:
            net.train()

    print("[VIS] ==== visualize_sample() DONE ====\n")