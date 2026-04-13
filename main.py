"""
main.py

Full pipeline — preprocessing → inpainting → segmentation → diffusion.

Stage gating
------------
Each stage is wrapped in  if True/False  so you can run exactly what
you need without touching the logic.  Currently:

    Stage 1 — Preprocessing   →  if False   (already done)
    Stage 2 — Inpainting      →  if False   (already done)
    Stage 3 — Segmentation    →  if True    (active)
    Stage 4 — Diffusion       →  if False

Visualisation vs training
--------------------------
Visualisation saves PNGs for a small subset (n_vis) of volumes/epochs
so you can sanity-check results without flooding disk.

Training always uses ALL 28 volumes and ALL their slices.
The MAX_SLICES_PER_EPOCH knob in Stage 4 is purely a speed trade-off:
set it to None to train on every slice every epoch, or set a number
(e.g. 5000) to randomly subsample per epoch for faster iteration.

Visualisation folders
---------------------
    vis/01_preprocess/   — raw vs processed, masks, overlay
    vis/02_inpaint/      — CT input, tumour mask, inpainted, diff
    vis/03_seg_train/    — CT, GT mask, pred mask, GT overlay, pred overlay
    vis/04_diffusion/    — healthy cond, GT, x0_hat, full sample
"""

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from preprocess  import preprocess_dataset, preprocess_volume_soft
from inpainting  import inpaint_dataset
from dataset     import CTNPZDataset, build_slice_entries_for_pairs
from models      import EnhancedUNet
from train       import Diffusion
from utils       import (enable_all_gpus, setup_model_for_device,
                         save_checkpoint_epoch, load_latest_checkpoint)
from visualise   import visualize_sample
from segmentation import (SegDataset, SegUNet, seg_loss, compute_seg_metrics,
                           save_seg_checkpoint, load_seg_checkpoint,
                           visualise_seg_batch, predict_volume)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

# ==============================================================================
# Paths
# ==============================================================================
DATASET_IN      = Path("Training_Batch_1")
PROCESSED_IMG   = Path("processed_volumes")
PROCESSED_MASKS = Path("processed_masks")
INPAINTED_DIR   = Path("inpainted_volumes")
SEG_CKPT_DIR    = Path("seg_checkpoints")
DIFF_CKPT_DIR   = Path("checkpoints")
VIS_ROOT        = Path("vis")

VIS_PREPROCESS  = VIS_ROOT / "01_preprocess"
VIS_INPAINT     = VIS_ROOT / "02_inpaint"
VIS_SEG         = VIS_ROOT / "03_seg_train"
VIS_DIFFUSION   = VIS_ROOT / "04_diffusion"

for p in [PROCESSED_IMG, PROCESSED_MASKS, INPAINTED_DIR,
          SEG_CKPT_DIR, DIFF_CKPT_DIR, VIS_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ==============================================================================
# Hyper-parameters
# ==============================================================================

# ── Segmentation ──────────────────────────────────────────────────────────────
SEG_EPOCHS        = 50
SEG_BATCH         = 8
SEG_LR            = 1e-3
SEG_BASE_CH       = 32
SEG_CLASS_WEIGHTS = (0.2, 1.0, 5.0)   # bg / liver / tumour
SEG_N_VIS         = 4                  # samples visualised per epoch

# ── Diffusion ─────────────────────────────────────────────────────────────────
TIMESTEPS          = 300
DIFF_EPOCHS        = 100
DIFF_BATCH         = 16
DIFF_LR            = 1e-4
FORCE_SIZE         = (256, 256)
USE_ALL_GPUS       = True
USE_EMA            = True
EMA_DECAY          = 0.9999
EMA_STEP           = 10
MIXED_PRECISION    = True
# Set to None to train on every slice every epoch (slow but complete).
# Set to e.g. 5000 to randomly subsample per epoch for faster iteration.
MAX_SLICES_PER_EPOCH = 5000

W_FULL_MAIN  = 1.0
W_MASK_MAIN  = 0.0
W_FULL_RECON = 4.0
W_MASK_RECON = 1.0
W_BG         = 1.0
W_SSIM       = 0.0
W_PERC       = 0.0


# ==============================================================================
# Shared utilities
# ==============================================================================

def ssim_loss(pred, target, window_size=11):
    c1  = (0.01 * 2) ** 2
    c2  = (0.03 * 2) ** 2
    mu1 = F.avg_pool2d(pred,   window_size, 1, window_size // 2)
    mu2 = F.avg_pool2d(target, window_size, 1, window_size // 2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    s1  = F.avg_pool2d(pred*pred,   window_size, 1, window_size//2) - mu1_sq
    s2  = F.avg_pool2d(target*target,window_size,1, window_size//2) - mu2_sq
    s12 = F.avg_pool2d(pred*target, window_size, 1, window_size//2) - mu1_mu2
    ssim_map = ((2*mu1_mu2+c1)*(2*s12+c2)) / ((mu1_sq+mu2_sq+c1)*(s1+s2+c2))
    return 1 - ssim_map.mean()


def psnr(pred, target, data_range=2.0):
    mse = F.mse_loss(pred, target).item()
    return float("inf") if mse == 0 else 10 * math.log10(data_range**2 / mse)


class EMA:
    def __init__(self, model, decay=0.9999, update_every=10):
        self.model, self.decay = model, decay
        self.update_every = update_every
        self.step   = 0
        self.shadow = {n: p.detach().cpu().clone()
                       for n, p in model.named_parameters()}

    def update(self, model):
        self.step += 1
        if self.step % self.update_every != 0:
            return
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(self.decay).add_(
                        p.detach().cpu(), alpha=1.0 - self.decay
                    )

    def apply_shadow_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.device))


# ==============================================================================
# STAGE 1 — Preprocessing
# ==============================================================================

def run_preprocessing():
    print("\n" + "="*60)
    print("STAGE 1 — Preprocessing")
    print("="*60)
    preprocess_dataset(
        dataset_in = str(DATASET_IN),
        img_dir    = str(PROCESSED_IMG),
        mask_dir   = str(PROCESSED_MASKS),
        vis_dir    = str(VIS_PREPROCESS),
        n_vis      = 5,      # save PNGs for 5 volumes; trains on all 28
    )


# ==============================================================================
# STAGE 2 — Inpainting
# ==============================================================================

def run_inpainting():
    print("\n" + "="*60)
    print("STAGE 2 — Inpainting")
    print("="*60)
    inpaint_dataset(
        processed_ct_dir   = str(PROCESSED_IMG),
        processed_mask_dir = str(PROCESSED_MASKS),
        out_dir            = str(INPAINTED_DIR),
        force_binary       = False,
        use_lama           = True,
        vis_dir            = str(VIS_INPAINT),
        n_vis              = 5,   # save PNGs for 5 volumes; processes all 28
    )


# ==============================================================================
# STAGE 3 — Segmentation training
# ==============================================================================

def run_segmentation_training():
    print("\n" + "="*60)
    print("STAGE 3 — Segmentation training")
    print("="*60)

    # ── Dataset: ALL slices from ALL 28 volumes ───────────────────────────────
    ds = SegDataset(
        img_dir  = str(PROCESSED_IMG),
        mask_dir = str(PROCESSED_MASKS),
        augment  = True,
    )
    dl = DataLoader(ds, batch_size=SEG_BATCH, shuffle=True,
                    num_workers=0, pin_memory=(DEVICE.type == "cuda"),
                    drop_last=True)

    print(f"[seg] Training on {len(ds)} slices from all volumes.")

    model  = SegUNet(in_ch=1, num_class=3, base_ch=SEG_BASE_CH).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=SEG_LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=SEG_EPOCHS, eta_min=SEG_LR * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    model, start_epoch = load_seg_checkpoint(
        model, str(SEG_CKPT_DIR), device=str(DEVICE), opt=opt
    )

    for epoch in range(start_epoch, SEG_EPOCHS):
        model.train()
        total_loss   = 0.0
        all_metrics  = []
        last_batch   = None

        pbar = tqdm(dl, desc=f"Seg {epoch+1}/{SEG_EPOCHS}", leave=False)
        for imgs, masks in pbar:
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(imgs)
                loss   = seg_loss(
                    logits, masks,
                    class_weights = SEG_CLASS_WEIGHTS,
                    device        = str(DEVICE),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            with torch.no_grad():
                m = compute_seg_metrics(logits, masks)
            all_metrics.append(m)
            total_loss += loss.item()
            last_batch  = (imgs.detach().cpu(),
                           masks.detach().cpu(),
                           logits.detach().cpu())
            pbar.set_postfix(
                loss  = f"{loss.item():.4f}",
                d_liv = f"{m['dice_cls1']:.3f}",
                d_tum = f"{m['dice_cls2']:.3f}",
            )

        sched.step()

        avg_loss  = total_loss / len(dl)
        avg_d_liv = float(np.mean([m["dice_cls1"] for m in all_metrics]))
        avg_d_tum = float(np.mean([m["dice_cls2"] for m in all_metrics]))
        avg_iou   = float(np.mean([m["mean_iou"]  for m in all_metrics]))
        print(f"  Epoch {epoch+1}/{SEG_EPOCHS}  "
              f"Loss={avg_loss:.4f}  "
              f"DiceLiver={avg_d_liv:.4f}  "
              f"DiceTumour={avg_d_tum:.4f}  "
              f"mIoU={avg_iou:.4f}")

        save_seg_checkpoint(
            model, opt, epoch,
            {"dice_liver": avg_d_liv, "dice_tumour": avg_d_tum,
             "mean_iou": avg_iou},
            str(SEG_CKPT_DIR),
        )

        # Visualise SEG_N_VIS examples from the last batch each epoch
        if last_batch is not None:
            imgs_cpu, masks_cpu, logits_cpu = last_batch
            visualise_seg_batch(
                imgs_cpu, masks_cpu, logits_cpu,
                epoch       = epoch + 1,
                vis_dir     = str(VIS_SEG),
                max_samples = SEG_N_VIS,
            )

    print("[seg] Training complete.")


# ==============================================================================
# STAGE 4 — Diffusion training
# ==============================================================================

def run_diffusion_training():
    print("\n" + "="*60)
    print("STAGE 4 — Diffusion training")
    print("="*60)

    enable_all_gpus(USE_ALL_GPUS)

    # ── Build full slice list from ALL 28 volumes ─────────────────────────────
    all_entries = build_slice_entries_for_pairs(
        inpainted_dir = str(INPAINTED_DIR),
        orig_dir      = str(PROCESSED_IMG),
        mask_dir      = str(PROCESSED_MASKS),
    )
    print(f"[diffusion] Total slices across all volumes: {len(all_entries)}")

    # Model + diffusion + optimiser
    model     = EnhancedUNet(in_ch=1, cond_ch=2, base_ch=48,
                              ch_mult=(1,2,4), num_res_blocks=2,
                              time_dim=256).to(DEVICE)
    model     = setup_model_for_device(model)
    diffusion = Diffusion(timesteps=TIMESTEPS, device=str(DEVICE),
                          use_cosine=True)
    opt       = torch.optim.AdamW(model.parameters(), lr=DIFF_LR,
                                   weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, DIFF_EPOCHS), eta_min=DIFF_LR * 0.01)
    scaler    = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION)

    start_epoch, model, opt = load_latest_checkpoint(
        model, opt, str(DIFF_CKPT_DIR), device=str(DEVICE)
    )
    print(f"[diffusion] Resuming from epoch {start_epoch}")

    ema = EMA(model, decay=EMA_DECAY, update_every=EMA_STEP) if USE_EMA else None

    for epoch in range(start_epoch, DIFF_EPOCHS):

        # ── Slice selection for this epoch ────────────────────────────────────
        # All entries are always available; MAX_SLICES_PER_EPOCH is purely a
        # speed knob. Set to None to use every slice every epoch.
        if MAX_SLICES_PER_EPOCH is not None and \
                len(all_entries) > MAX_SLICES_PER_EPOCH:
            np.random.seed(epoch)
            sel = np.random.choice(len(all_entries),
                                   MAX_SLICES_PER_EPOCH, replace=False)
            entries_epoch = [all_entries[i] for i in sel]
            print(f"[diffusion] Epoch {epoch+1}: using "
                  f"{len(entries_epoch)}/{len(all_entries)} slices "
                  f"(subsample for speed — all 28 volumes represented)")
        else:
            entries_epoch = all_entries
            print(f"[diffusion] Epoch {epoch+1}: using all "
                  f"{len(all_entries)} slices")

        ds = CTNPZDataset(
            entries       = entries_epoch,
            preprocess_fn = preprocess_volume_soft,
            clip_min      = -200,
            clip_max      = 300,
            force_size    = FORCE_SIZE,
        )
        dl = DataLoader(ds, batch_size=DIFF_BATCH, shuffle=True,
                        num_workers=0,
                        pin_memory=(DEVICE.type == "cuda"),
                        drop_last=True)

        model.train()
        epoch_loss     = 0.0
        epoch_steps    = 0
        psnr_epoch     = []
        ssim_epoch     = []
        last_vis_batch = None

        pbar = tqdm(dl, desc=f"Diff {epoch+1}/{DIFF_EPOCHS}", leave=False)
        for batch in pbar:
            healthy_t, target_t, tumor_mask_t, liver_mask_t = batch

            healthy_t    = F.interpolate(healthy_t,    FORCE_SIZE,
                                         mode="bilinear", align_corners=False)
            target_t     = F.interpolate(target_t,     FORCE_SIZE,
                                         mode="bilinear", align_corners=False)
            tumor_mask_t = F.interpolate(tumor_mask_t, FORCE_SIZE,
                                         mode="nearest")
            liver_mask_t = F.interpolate(liver_mask_t, FORCE_SIZE,
                                         mode="nearest")

            healthy_t    = healthy_t.to(DEVICE)
            target_t     = target_t.to(DEVICE)
            tumor_mask_t = tumor_mask_t.to(DEVICE)
            liver_mask_t = liver_mask_t.to(DEVICE)

            B    = target_t.shape[0]
            t    = torch.randint(0, TIMESTEPS, (B,), device=DEVICE).long()
            cond = torch.cat([healthy_t, liver_mask_t], dim=1)

            drop_cond = random.random() < 0.2

            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                core_loss, pred, x0_hat, target_core = diffusion.p_losses(
                    model, target_t, cond, t,
                    use_v=True, drop_cond=drop_cond,
                )
                x0_hat   = torch.clamp(x0_hat, -1.0, 1.0)
                mask     = tumor_mask_t.float()
                inv_mask = 1.0 - mask

                full_recon   = F.l1_loss(x0_hat, target_t)
                masked_recon = F.l1_loss(x0_hat * mask,     target_t * mask)
                bg_consist   = F.l1_loss(x0_hat * inv_mask, healthy_t * inv_mask)
                ssim_val     = ssim_loss(x0_hat, target_t)

                main_loss  = (W_FULL_MAIN * core_loss +
                              W_MASK_MAIN * F.mse_loss(pred * mask,
                                                       target_core * mask))
                recon_loss = W_FULL_RECON * full_recon + W_MASK_RECON * masked_recon
                loss       = (main_loss
                              + recon_loss
                              + 2.0 * W_BG  * bg_consist
                              + W_SSIM      * ssim_val)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            if ema:
                ema.update(model)

            epoch_loss  += loss.item()
            epoch_steps += 1

            with torch.no_grad():
                psnr_epoch.append(psnr(x0_hat, target_t))
                ssim_epoch.append(
                    float((1.0 - ssim_loss(x0_hat, target_t)).item())
                )

            last_vis_batch = (healthy_t.detach(), target_t.detach(),
                              tumor_mask_t.detach(), liver_mask_t.detach())

            pbar.set_postfix(
                loss = f"{epoch_loss/epoch_steps:.4f}",
                lr   = f"{sched.get_last_lr()[0]:.2e}",
            )

        sched.step()

        avg_loss = epoch_loss / max(1, epoch_steps)
        avg_psnr = float(np.mean(psnr_epoch)) if psnr_epoch else float("nan")
        avg_ssim = float(np.mean(ssim_epoch)) if ssim_epoch else float("nan")
        print(f"  Epoch {epoch+1}/{DIFF_EPOCHS}  "
              f"Loss={avg_loss:.5f}  PSNR={avg_psnr:.2f}  SSIM={avg_ssim:.4f}")

        # Save checkpoint
        ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "train_loss": avg_loss,
        }
        if ema:            ckpt["ema"]    = ema.shadow
        if MIXED_PRECISION:ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, DIFF_CKPT_DIR / "latest_full.pt")
        save_checkpoint_epoch(model, opt, epoch, str(DIFF_CKPT_DIR))

        # Visualise one batch per epoch (not every batch)
        if last_vis_batch is not None:
            try:
                vis_model = ema.model if ema else model
                ht, tt, tmt, lmt = last_vis_batch
                visualize_sample(
                    diffusion, vis_model, ht, tt, tmt, lmt,
                    pass_n         = epoch + 1,
                    stage          = "train",
                    device         = str(DEVICE),
                    force_size     = FORCE_SIZE,
                    timesteps      = 100,
                    guidance_scale = 2.0,
                    vis_dir        = str(VIS_DIFFUSION),
                )
            except Exception as e:
                print(f"[diffusion] Visualisation failed: {e}")

    print("[diffusion] Training complete.")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("\n=== CT Liver Tumour Pipeline ===")
    print(f"    Device : {DEVICE}")
    print(f"    Dataset: {DATASET_IN.resolve()}\n")

    # ------------------------------------------------------------------
    # STAGE 1 — Preprocessing
    # Reads raw NIfTI, writes processed_volumes/ and processed_masks/
    # Visualises 5 volumes → vis/01_preprocess/
    # ------------------------------------------------------------------
    if False:
        run_preprocessing()

    # ------------------------------------------------------------------
    # STAGE 2 — Inpainting
    # Processes all 28 volumes, writes inpainted_volumes/
    # Visualises 5 volumes → vis/02_inpaint/
    # ------------------------------------------------------------------
    if True:
        run_inpainting()

    # ------------------------------------------------------------------
    # STAGE 3 — Segmentation training
    # Trains on ALL slices from all 28 volumes
    # Visualises SEG_N_VIS samples per epoch → vis/03_seg_train/
    # ------------------------------------------------------------------
    if False:
        run_segmentation_training()

    # ------------------------------------------------------------------
    # STAGE 4 — Diffusion training
    # Trains on ALL slices from all 28 volumes
    # (MAX_SLICES_PER_EPOCH subsamples per epoch for speed if set)
    # Visualises 1 batch per epoch → vis/04_diffusion/
    # ------------------------------------------------------------------
    if False:
        run_diffusion_training()

    print("\n=== Pipeline finished ===")


if __name__ == "__main__":
    main()