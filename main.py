"""
main.py

Full training entrypoint. Uses:
 - preprocess.py -> preprocess_dataset
 - inpainting.py -> inpaint_datasetainting
 - dataset.py -> CTNPZDataset or similar
 - models.py -> EnhancedUNet
 - train.py -> Diffusion
 - utils.py -> GPU & checkpoint helpers
 - visualise.py -> visualize_sample
"""

import os
from pathlib import Path
import random
import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt


# ---- Imports from your modules (ensure paths/names match) ----
from preprocess import preprocess_dataset, preprocess_volume_soft
from inpainting import inpaint_dataset
from dataset import CTNPZDataset, build_slice_entries_for_pairs
from models import EnhancedUNet
from train import Diffusion
from utils import enable_all_gpus, setup_model_for_device, save_checkpoint_epoch, load_latest_checkpoint
from visualise import visualize_sample
from augment import augment_medical
import warnings
warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# ---- Config ----
DATASET_IN = Path("/home/ie643_lambdaforce/Training_Batch2")
PROCESSED_IMG = Path("processed_volumes")
PROCESSED_MASKS = Path("processed_masks")
INPAINTED_DIR = Path("inpainted_volumes")
CKPT_DIR = Path("checkpoints")
VIS_DIR = Path("visualizations")

for p in [PROCESSED_IMG, PROCESSED_MASKS, INPAINTED_DIR, CKPT_DIR, VIS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# training hyperparams
TIMESTEPS = 300
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-4
FORCE_SIZE = (256, 256)
USE_ALL_GPUS = True
USE_EMA = True
EMA_DECAY = 0.9999
EMA_STEP = 10
MIXED_PRECISION = True

# loss weights (tweak if needed)
W_FULL_MAIN = 1.0
W_MASK_MAIN = 0.0
W_FULL_RECON = 4.0
W_MASK_RECON = 1.0
W_BG = 1.0
W_SSIM = 0.0
W_PERC = 0.0  # perceptual weight (you asked to include perceptual loss)

# ------------------------------------------------------------
# Visualization: Preprocessed vs Tumor Mask vs Inpainted
# ------------------------------------------------------------
def visualize_pre_and_inpaint(
    processed_vol_path,
    processed_mask_path,
    inpainted_vol_path,
    out_dir="visual_preview",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Load NPZ volumes
    ct = np.load(processed_vol_path)["image"]      # (Z,H,W)
    seg = np.load(processed_mask_path)["mask"]     # (Z,H,W)
    inpaint = np.load(inpainted_vol_path)["inpainted"]

    # Find slice with the largest tumor
    tumor_area = (seg > 0.5).sum(axis=(1, 2))
    idx = int(np.argmax(tumor_area))

    ct_slice = ct[idx]
    seg_slice = seg[idx]
    inpaint_slice = inpaint[idx]

    # Plot 3-panel visualization
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(ct_slice, cmap="gray")
    axs[0].set_title("Preprocessed CT")

    axs[1].imshow(seg_slice, cmap="hot")
    axs[1].set_title("Tumor Mask")

    axs[2].imshow(inpaint_slice, cmap="gray")
    axs[2].set_title("Inpainted CT")

    for ax in axs:
        ax.axis("off")

    save_path = out_dir / f"{Path(processed_vol_path).stem}_preview.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[VIS] Saved preview → {save_path}")

# ------------------------------------------------------------
# Perceptual (VGG) setup (pretrained)
# ------------------------------------------------------------
use_perceptual = False
vgg = None
if use_perceptual:
    try:
        from torchvision import models
        vgg = models.vgg19(pretrained=True).features[:16].eval().to(DEVICE)
        for p in vgg.parameters():
            p.requires_grad = False
    except Exception as e:
        print(f"[WARN] VGG perceptual init failed: {e}")
        vgg = None
# perceptual helper
def perceptual_loss(pred, target):
    if vgg is None:
        return torch.tensor(0.0, device=pred.device)
    # convert grayscale to 3 channels by repeating
    p = torch.clamp((pred + 1.0) / 2.0, 0, 1).repeat(1, 3, 1, 1)
    t = torch.clamp((target + 1.0) / 2.0, 0, 1).repeat(1, 3, 1, 1)
    with torch.no_grad():
        pf = vgg(p)
        tf = vgg(t)
    return F.mse_loss(pf, tf)

# ------------------------------------------------------------
# SSIM utility (same as master)
# ------------------------------------------------------------
def ssim_loss(pred, target, window_size=11):
    c1 = (0.01 * 2) ** 2
    c2 = (0.03 * 2) ** 2
    mu1 = F.avg_pool2d(pred, window_size, 1, window_size//2)
    mu2 = F.avg_pool2d(target, window_size, 1, window_size//2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(pred * pred, window_size, 1, window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, window_size, 1, window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, 1, window_size//2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return 1 - ssim_map.mean()

# ------------------------------------------------------------
# PSNR helper (inputs in [-1,1], data_range=2)
# ------------------------------------------------------------
def psnr(pred, target, data_range=2.0):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10((data_range**2) / mse)

# ------------------------------------------------------------
# EMA wrapper (lightweight)
# ------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.9999, update_every=10):
        self.model = model
        self.decay = decay
        self.update_every = update_every
        self.step = 0
        self.shadow = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

    def update(self, model):
        self.step += 1
        if self.step % self.update_every != 0:
            return
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(self.decay)
                    self.shadow[n].add_(p.detach().cpu(), alpha=1.0 - self.decay)

    def apply_shadow_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.device))


# ------------------------------------------------------------
# Training entrypoint
# ------------------------------------------------------------
def main():
    print("\n=== Starting training pipeline ===\n")
    enable_all_gpus(USE_ALL_GPUS)

    # # Preprocess if needed (commented to avoid re-running)
    # preprocess_dataset(DATASET_IN, PROCESSED_IMG, PROCESSED_MASKS)
    # --- Visualize N examples after preprocessing ---
    # try:
    #     N = 20  # <-- change number of examples here

    #     ct_files = sorted(PROCESSED_IMG.glob("*.npz"))
    #     mask_files = sorted(PROCESSED_MASKS.glob("*.npz"))

    #     total = min(N, len(ct_files))

    #     if total < 1:
    #         print("[WARN] No files found for visualization.")
    #     else:
    #         print(f"[INFO] Visualizing {total} example(s)...")

    #         # Pick random subset to avoid always showing the first volumes
    #         idxs = np.random.choice(len(ct_files), size=total, replace=False)

    #         for i in idxs:
    #             ct_path = ct_files[i]
    #             mask_path = mask_files[i]

    #             # Create temporary fake inpaint identical to CT
    #             tmp_fake_inpaint = PROCESSED_IMG / (ct_path.stem + "_tmp.npz")
    #             np.savez_compressed(tmp_fake_inpaint, inpainted=np.load(ct_path)["image"])

    #             try:
    #                 visualize_pre_and_inpaint(
    #                     ct_path,
    #                     mask_path,
    #                     tmp_fake_inpaint,
    #                     out_dir="visual_preprocess"
    #                 )
    #             except Exception as e:
    #                 print(f"[WARN] Visualization failed for {ct_path.name}: {e}")

    #             tmp_fake_inpaint.unlink(missing_ok=True)

    # except Exception as e:
    #     print(f"[WARN] Could not visualize preprocessing: {e}")

    # Run inpainting to produce inpainted npz files
    # inpaint_dataset(
    # processed_ct_dir=str(PROCESSED_IMG),
    # processed_mask_dir=str(PROCESSED_MASKS),
    # out_dir=str(INPAINTED_DIR),
    # force_binary=False
    # )


    # Build pairs (inpainted, original, mask)
    inpainted_files = sorted(list(INPAINTED_DIR.glob("*.npz")))
    orig_files = sorted(list(PROCESSED_IMG.glob("*.npz")))
    mask_files = sorted(list(PROCESSED_MASKS.glob("*.npz")))

    # --- Visualize first 5 examples after inpainting ---
    # try:
    #     ct_list = sorted(PROCESSED_IMG.glob("*.npz"))
    #     mask_list = sorted(PROCESSED_MASKS.glob("*.npz"))
    #     inpaint_list = sorted(INPAINTED_DIR.glob("*.npz"))

    #     N = 25
    #     for i in range(N):
    #         visualize_pre_and_inpaint(
    #             processed_vol_path=ct_list[i],
    #             processed_mask_path=mask_list[i],
    #             inpainted_vol_path=inpaint_list[i],
    #             out_dir="visual_inpainted"
    #         )
    # except Exception as e:
    #     print(f"[WARN] Could not visualize multiple inpainted samples: {e}")


    # Pairing logic: assume same ordering / same numbering
    pairs = list(zip(inpainted_files, orig_files, mask_files))
    entries = build_slice_entries_for_pairs(pairs)
    print(f"[INFO] Total dataset slice entries: {len(entries)}")

    # ===============================
    # Reduce dataset size per epoch
    # ===============================
    # MAX_SLICES_PER_EPOCH = 5000   # you can set 5000, 10000, etc.
    # if len(entries) > MAX_SLICES_PER_EPOCH:
    #     np.random.seed(epoch if 'epoch' in locals() else 42)
    #     entries = list(np.random.choice(entries, MAX_SLICES_PER_EPOCH, replace=False))
    #     print(f"[INFO] Using subset: {len(entries)} slices this epoch.")
    
    # # Dataset & DataLoader
    # ds = CTNPZDataset(entries=entries,
    #                   preprocess_fn=preprocess_volume_soft,
    #                   augment_fn=augment_medical,
    #                   aug_prob=0.2,
    #                   force_size=FORCE_SIZE)
    # dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # ------------------------------------------------------
    # VISUALISE RANDOM SUBSET OF THE COMPLETE DATASET
    # (BEFORE TRAINING)
    # ------------------------------------------------------
    def visualize_paired_dataset(entries, out_dir="vis_full_dataset", max_samples=30):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

        sample_entries = random.sample(entries, min(max_samples, len(entries)))

        for idx, entry in enumerate(sample_entries):

            inpaint_path = entry["inpaint"]
            orig_path    = entry["orig"]
            mask_path    = entry["mask"]
            zidx         = entry["slice"]

            # --- Load ---
            orig_npz = np.load(orig_path)
            img = orig_npz["image"][zidx].astype(np.float32)

            inpaint_npz = np.load(inpaint_path)
            img_inp = inpaint_npz["inpainted"][zidx].astype(np.float32)

            mask_npz = np.load(mask_path)
            mask = mask_npz["mask"][zidx].astype(np.float32)

            # --- Clip for visualization ---
            img_vis = np.clip(img, -200, 300)
            img_inp_vis = np.clip(img_inp, -200, 300)

            # --- Plot ---
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            axs[0].imshow(img_vis, cmap="gray")
            axs[0].set_title("Ground Truth Slice")
            axs[0].axis("off")

            axs[1].imshow(mask, cmap="hot")
            axs[1].set_title("Tumor Mask")
            axs[1].axis("off")

            axs[2].imshow(img_inp_vis, cmap="gray")
            axs[2].set_title("Inpainted Slice")
            axs[2].axis("off")

            plt.tight_layout()
            save_path = out_dir / f"paired_{idx}.png"
            plt.savefig(save_path, dpi=130)
            plt.close()

        print(f"[VIS] Saved {len(sample_entries)} paired dataset previews → {out_dir}")


    # ---- Call it immediately ----
    # visualize_paired_dataset(entries, out_dir="vis_full_dataset_pretrain", max_samples=30)


    # Model + diffusion + optimizer + scheduler
    model = EnhancedUNet(in_ch=1, cond_ch=2, base_ch=48, ch_mult=(1,2,4), num_res_blocks=2, time_dim=256).to(DEVICE)
    model = setup_model_for_device(model)
    diffusion = Diffusion(timesteps=TIMESTEPS, device=DEVICE, use_cosine=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS), eta_min=LR*0.01)

    scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION)

    # resume from checkpoint if available
    start_epoch, model, opt = load_latest_checkpoint(model, opt, CKPT_DIR, device=DEVICE)
    print(f"[INFO] Resuming training from epoch {start_epoch}")

    ema = EMA(model, decay=EMA_DECAY, update_every=EMA_STEP) if USE_EMA else None

    # metric logs
    all_train_losses = []
    all_psnrs = []
    all_ssims = []
    all_tumor_mses = []
    all_full_mses = []

    all_entries = entries  

    for epoch in range(start_epoch, EPOCHS):

        MAX_SLICES_PER_EPOCH = 5000

        if len(all_entries) > MAX_SLICES_PER_EPOCH:
            # deterministic seed per epoch
            np.random.seed(epoch)

            # choose indices, not entries directly
            sel_idx = np.random.choice(len(all_entries), MAX_SLICES_PER_EPOCH, replace=False)
            entries_epoch = [all_entries[i] for i in sel_idx]

            print(f"[INFO] Using subset: {len(entries_epoch)} slices this epoch.")
        else:
            entries_epoch = all_entries

        # Dataset & DataLoader
        ds = CTNPZDataset(
            entries=entries_epoch,
            preprocess_fn=preprocess_volume_soft,
            clip_min=-200,
            clip_max=300,
            force_size=FORCE_SIZE
        )


        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        psnr_epoch = []
        ssim_epoch = []
        tumor_mse_epoch = []
        full_mse_epoch = []

        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in pbar:
            # unpack now returns 5 elements
            healthy_t, target_t, tumor_mask_t, liver_mask_t = batch
            clean_target_t = target_t.clone()  # keep a clean copy for supervision

            # ---- FIX: force deterministic same spatial size ---
            healthy_t = F.interpolate(healthy_t, size=FORCE_SIZE, mode='bilinear', align_corners=False)
            # target_t is already clean and preprocessed; keep it as-is but also ensure size
            target_t  = F.interpolate(target_t,  size=FORCE_SIZE, mode='bilinear', align_corners=False)
            tumor_mask_t = F.interpolate(tumor_mask_t, size=FORCE_SIZE, mode='nearest')
            liver_mask_t = F.interpolate(liver_mask_t, size=FORCE_SIZE, mode='nearest')
            clean_target_t = F.interpolate(clean_target_t, size=FORCE_SIZE, mode='bilinear', align_corners=False)

            healthy_t = healthy_t.to(DEVICE)
            # Use the CLEAN target for supervision / visualization
            target_t  = target_t.to(DEVICE)
            clean_target_t = clean_target_t.to(DEVICE)
            tumor_mask_t = tumor_mask_t.to(DEVICE)
            liver_mask_t = liver_mask_t.to(DEVICE)


            # determine batch size (use target_t to be safe)
            B = int(target_t.shape[0])

            # create per-sample timesteps (guaranteed 1-D long tensor on the right device)
            t = torch.randint(0, TIMESTEPS, (B,), device=DEVICE).long()

            # defensive: if something turned t into a 0-d tensor later, fix it here
            if t.dim() == 0:
                t = t.unsqueeze(0)

            # final sanity: ensure t length matches batch
            if t.shape[0] != B:
                # if scalar was provided, expand it to whole batch
                if t.numel() == 1:
                    t = t.expand(B).contiguous()
                else:
                    # fall back to generate fresh
                    t = torch.randint(0, TIMESTEPS, (B,), device=DEVICE).long()


            # conditioning, with optional robustness augmentations
            # P_REPLACE_WITH_TARGET = 0.12
            # P_DROP_LIVER_MASK = 0.12
            healthy_cond = healthy_t
            liver_mask_cond = liver_mask_t
            # if random.random() < P_REPLACE_WITH_TARGET:
            #     healthy_cond = target_t.detach()
            # if random.random() < P_DROP_LIVER_MASK:
            #     liver_mask_cond = torch.zeros_like(liver_mask_t)
            cond = torch.cat([healthy_cond, liver_mask_cond], dim=1)



            drop_cond = (random.random() < 0.2)

            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                                # use clean_target_t as x_start (no augmentation)
                core_loss, pred, x0_hat, target_core = diffusion.p_losses(
                    model, clean_target_t, cond, t, use_v=True, drop_cond=drop_cond
                )

                # clamp reconstruction
                x0_hat = torch.clamp(x0_hat, -1.0, 1.0)

                # compute auxiliary losses (reconstruction + masked + bg consistency + ssim + perceptual)
                mask = tumor_mask_t.float()
                inv_mask = 1.0 - mask

                # Full recon L1 between reconstructed x0_hat and target_t
                full_recon_loss = F.l1_loss(x0_hat, target_t)

                # Masked recon (tumor region) L1
                masked_recon_loss = F.l1_loss(x0_hat * mask, target_t * mask)

                # Background consistency: keep non-liver areas consistent with healthy condition
                bg_consistency_loss = F.l1_loss(x0_hat * inv_mask, healthy_cond * inv_mask)

                # SSIM on tumor region (use entire image SSIM as proxy)
                ssim_mask_loss = ssim_loss(x0_hat, target_t)

                # perceptual
                perc_loss = perceptual_loss(x0_hat, target_t) if vgg is not None else torch.tensor(0.0, device=DEVICE)

                # Compose final loss (weights from master)
                main_loss = W_FULL_MAIN * core_loss + W_MASK_MAIN * F.mse_loss(pred * mask, target_core * mask)  # keep main core + masked v-loss
                recon_loss = W_FULL_RECON * full_recon_loss + W_MASK_RECON * masked_recon_loss

                loss = main_loss + 1.0 * recon_loss + 2.0 * W_BG * bg_consistency_loss + W_SSIM * ssim_mask_loss + W_PERC * perc_loss
                loss = loss / 1.0  # if you use grad accumulation, divide here
                print(f"Batch Loss: {loss.item():.4f} (Main: {main_loss.item():.4f}, Recon: {recon_loss.item():.4f}, BG: {bg_consistency_loss.item():.4f}, SSIM: {ssim_mask_loss.item():.4f}, Perc: {perc_loss.item():.4f})")
            # backward
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            if ema is not None:
                ema.update(model)

            # logging metrics for this batch
            epoch_loss_sum += loss.item()
            epoch_steps += 1

            # compute recon metrics using x0_hat
            with torch.no_grad():
                # PSNR returns a python float already, but ensure float()
                batch_psnr = float(psnr(x0_hat.detach(), target_t.detach()))

                # ssim_loss returns a torch tensor on device — convert to CPU python float
                ssim_val_tensor = ssim_loss(x0_hat.detach(), target_t.detach())    # tensor on GPU
                batch_ssim = float((1.0 - ssim_val_tensor).detach().cpu().item())

                # other metrics already returned as .item()
                batch_tumor_mse = float(F.mse_loss(x0_hat.detach() * mask, target_t.detach() * mask).item())
                batch_full_mse = float(F.mse_loss(x0_hat.detach(), target_t.detach()).item())

            psnr_epoch.append(batch_psnr)
            ssim_epoch.append(batch_ssim)
            tumor_mse_epoch.append(batch_tumor_mse)
            full_mse_epoch.append(batch_full_mse)


            pbar.set_postfix(loss=f"{(epoch_loss_sum/epoch_steps):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        scheduler.step()
        
        # epoch summaries
        avg_loss = epoch_loss_sum / max(1, epoch_steps)
        avg_psnr = float(np.mean(psnr_epoch)) if psnr_epoch else float("nan")
        avg_ssim = float(np.mean(ssim_epoch)) if ssim_epoch else float("nan")
        avg_tumor_mse = float(np.mean(tumor_mse_epoch)) if tumor_mse_epoch else float("nan")
        avg_full_mse = float(np.mean(full_mse_epoch)) if full_mse_epoch else float("nan")

        all_train_losses.append(avg_loss)
        all_psnrs.append(avg_psnr)
        all_ssims.append(avg_ssim)
        all_tumor_mses.append(avg_tumor_mse)
        all_full_mses.append(avg_full_mse)
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss={avg_loss:.6f}  PSNR={avg_psnr:.3f}  SSIM={avg_ssim:.4f}  TumorMSE={avg_tumor_mse:.6f}  FullMSE={avg_full_mse:.6f}")

        
        # Save checkpoints: full (model + opt + epoch + scheduler + ema + scaler)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_loss": avg_loss
        }
        if ema is not None:
            ckpt["ema"] = ema.shadow
        if MIXED_PRECISION:
            ckpt["scaler"] = scaler.state_dict()

        torch.save(ckpt, CKPT_DIR / "latest_full.pt")
        save_checkpoint_epoch(model, opt, epoch, CKPT_DIR)  # saves epoched model file
        os.makedirs(VIS_DIR, exist_ok=True)

        try:
            # pick visualisation model
            vis_model = ema.model if (ema is not None) else model

            visualize_sample(
                diffusion,
                vis_model,
                healthy_t,
                target_t,
                tumor_mask_t,
                liver_mask_t,
                pass_n=epoch,
                stage="mid",
                device=DEVICE,
                force_size=FORCE_SIZE,
                timesteps=300,              # <-- use more steps for clean sample
                vis_dir=VIS_DIR
            )


        except Exception as e:
            print("Visualization failed:", e)



    print("\n=== Training finished ===")
    # Save final metrics to disk (CSV)
    import pandas as pd
    df = pd.DataFrame({
        "train_loss": all_train_losses,
        "psnr": all_psnrs,
        "ssim": all_ssims,
        "tumor_mse": all_tumor_mses,
        "full_mse": all_full_mses
    })
    df.to_csv(CKPT_DIR / "train_metrics.csv", index=False)
    print(f"[INFO] Metrics saved to {CKPT_DIR/'train_metrics.csv'}")


if __name__ == "__main__":
    main()
