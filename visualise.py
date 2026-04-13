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
    force_size=(256,256),
    timesteps=40,
    guidance_scale=2.0,
    vis_dir=None
):
    print("\n[VIS] ==== Running visualize_sample() ====")

    # -------------------------------------------------------
    # Ensure vis_dir exists
    # -------------------------------------------------------
    if vis_dir is not None:
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Resize to training resolution
    # -------------------------------------------------------
    H, W = force_size
    healthy_t = F.interpolate(healthy_t, size=force_size, mode="bilinear", align_corners=False)
    target_t  = F.interpolate(target_t,  size=force_size, mode="bilinear", align_corners=False)
    tumor_mask_t = F.interpolate(tumor_mask_t, size=force_size, mode="nearest")
    liver_mask_t = F.interpolate(liver_mask_t, size=force_size, mode="nearest")

    # -------------------------------------------------------
    # Unwrap DP
    # -------------------------------------------------------
    net = model.module if hasattr(model, "module") else model
    net.eval()

    # -------------------------------------------------------
    # Correct conditioning: MUST MATCH TRAINING (2-channels)
    # -------------------------------------------------------
    healthy_1 = healthy_t[:1].to(device)
    target_1  = target_t[:1].to(device)
    liver_1   = liver_mask_t[:1].to(device)

    # ONLY 2 channels — EXACTLY as in training
    cond = torch.cat([healthy_1, liver_1], dim=1)

    print("[VIS] healthy range:", float(healthy_1.min()), float(healthy_1.max()))
    print("[VIS] target  range:", float(target_1.min()),  float(target_1.max()))

    # -------------------------------------------------------
    # (1) Full reverse DDPM sampling
    # -------------------------------------------------------
    final_sample = diffusion.sample(
        model=net,
        cond=cond,
        shape=(1,1,H,W),
        timesteps=timesteps,
        guidance_scale=guidance_scale,
        device=device
    )[0,0].cpu().numpy()

    print("[VIS] final_sample range:", final_sample.min(), final_sample.max())

    # -------------------------------------------------------
    # (2) Training-style mid-step reconstruction
    # -------------------------------------------------------
    t = torch.tensor([timesteps//2], device=device).long()
    noise = torch.randn_like(target_1)
    x_t = diffusion.q_sample(target_1, noise, t)
    print("[VIS] x_t range:", float(x_t.min()), float(x_t.max()))

    v_pred = net(x_t, cond, t)
    print("[VIS] v_pred range:", float(v_pred.min()), float(v_pred.max()))

    x0_hat_tensor = diffusion.predict_x0_from_v(x_t, v_pred, t)
    x0_hat = x0_hat_tensor[0,0].cpu().numpy()
    print("[VIS] x0_hat range:", x0_hat.min(), x0_hat.max())

    # -------------------------------------------------------
    # Image conversion
    # -------------------------------------------------------
    def disp(img):
        return np.clip((img + 1) / 2, 0, 1)

    healthy_disp = disp(healthy_t[0,0].cpu().numpy())
    target_disp  = disp(target_t[0,0].cpu().numpy())
    x0_hat_disp  = disp(x0_hat)
    final_disp   = disp(final_sample)

    # -------------------------------------------------------
    # Plot
    # -------------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(22,6))

    axs[0].imshow(healthy_disp, cmap="gray"); axs[0].set_title("Healthy Cond")
    axs[1].imshow(target_disp, cmap="gray");  axs[1].set_title("Ground Truth")
    axs[2].imshow(x0_hat_disp, cmap="gray");  axs[2].set_title("x0_hat (mid-step)")
    axs[3].imshow(final_disp, cmap="gray");   axs[3].set_title("Final Reconstruction")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    if vis_dir is not None:
        save_path = vis_dir / f"recon_pass{pass_n}_{stage}.png"
        plt.savefig(save_path, dpi=130)
        print("[VIS] Saved to:", save_path)

    plt.close()
    print("[VIS] ==== visualize_sample() DONE ====\n")

