"""
main.py  —  Kaggle-adapted version
====================================

Fixes vs previous version
--------------------------
  FIX (bug #7):  Removed redundant F.interpolate calls in the diffusion training
                 loop. CTNPZDataset already resizes to FORCE_SIZE.

  FIX (bug #8):  EMA visualisation now uses the live model directly instead of
                 the EMA shadow. With EMA_DECAY=0.9999 and EMA_STEP=10, the
                 shadow barely moves for the first ~30 epochs and is effectively
                 the random initialisation — so visualising it produces pure noise
                 (exactly what was observed at epoch 8-9). The live model is used
                 for all visualisation. EMA shadow is still maintained for final
                 inference use if needed.

  FIX (bug #9):  Stage 1 preprocessing gate changed from 'if True' to 'if False'
                 so it doesn't rerun on every launch after the first run.

  FIX (lr-1):    Replaced CosineAnnealingLR with linear warmup + cosine decay,
                 called per-batch. Prevents the large early-epoch gradients from
                 the reconstruction loss terms from overshooting into degenerate
                 flat regions.

  FIX (debug-1): Added per-epoch v_pred diagnostic print so you can immediately
                 see if the model is in the degenerate zero-prediction mode.
"""

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# PyTorch AMP compatibility shim  (works with both PyTorch 1.x and 2.x)
# ---------------------------------------------------------------------------
def _make_autocast(enabled: bool):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        enabled = False
    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=enabled)


def _make_scaler(enabled: bool):
    if not torch.cuda.is_available():
        enabled = False
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from preprocess import preprocess_dataset, preprocess_volume_soft
from inpainting import inpaint_dataset
from dataset    import CTNPZDataset, build_slice_entries_for_pairs
from models     import EnhancedUNet
from train      import Diffusion
from utils      import (enable_all_gpus, setup_model_for_device,
                        save_checkpoint_epoch, load_latest_checkpoint)
from visualise  import visualize_sample

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True


# ==============================================================================
# Paths
# ==============================================================================
KAGGLE_DATASET_SLUG = "liver-ct-batch1"          # ← CHANGE THIS

KAGGLE_INPUT    = Path("/kaggle/input") / KAGGLE_DATASET_SLUG
DATASET_IN      = KAGGLE_INPUT / "Training_Batch_1"

WORKING         = Path("/kaggle/working")
PROCESSED_IMG   = WORKING / "processed_volumes"
PROCESSED_MASKS = WORKING / "processed_masks"
INPAINTED_DIR   = WORKING / "inpainted_volumes"
DIFF_CKPT_DIR   = WORKING / "checkpoints"
VIS_ROOT        = WORKING / "vis"
VIS_DIFFUSION   = VIS_ROOT / "04_diffusion"

for p in [PROCESSED_IMG, PROCESSED_MASKS, INPAINTED_DIR, DIFF_CKPT_DIR, VIS_ROOT]:
    p.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Device
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[init] Running on: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[init] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[init] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ==============================================================================
# Hyper-parameters
# ==============================================================================
TIMESTEPS            = 300
DIFF_EPOCHS          = 100
DIFF_BATCH           = 8
DIFF_LR              = 1e-4
FORCE_SIZE           = (256, 256)
USE_ALL_GPUS         = False
USE_EMA              = True
EMA_DECAY            = 0.9999
EMA_STEP             = 10
MIXED_PRECISION      = True
MAX_SLICES_PER_EPOCH = 2000
WARMUP_EPOCHS        = 2        # linear warmup over this many epochs

W_FULL_MAIN  = 1.0
W_MASK_MAIN  = 0.0
W_FULL_RECON = 4.0
W_MASK_RECON = 2.0
W_BG         = 0.5
W_SSIM       = 0.5


# ==============================================================================
# Shared utilities
# ==============================================================================

def ssim_loss(pred, target, window_size=11):
    c1  = (0.01 * 2) ** 2
    c2  = (0.03 * 2) ** 2
    mu1 = F.avg_pool2d(pred,   window_size, 1, window_size // 2)
    mu2 = F.avg_pool2d(target, window_size, 1, window_size // 2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    s1  = F.avg_pool2d(pred * pred,     window_size, 1, window_size // 2) - mu1_sq
    s2  = F.avg_pool2d(target * target, window_size, 1, window_size // 2) - mu2_sq
    s12 = F.avg_pool2d(pred * target,   window_size, 1, window_size // 2) - mu1_mu2
    ssim_map = ((2*mu1_mu2+c1)*(2*s12+c2)) / ((mu1_sq+mu2_sq+c1)*(s1+s2+c2))
    return 1 - ssim_map.mean()


def psnr(pred, target, data_range=2.0):
    mse = F.mse_loss(pred, target).item()
    return float("inf") if mse == 0 else 10 * math.log10(data_range ** 2 / mse)


def make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, base_lr, min_lr):
    """
    Linear warmup for `warmup_steps` then cosine decay to `min_lr`.
    Called once per *batch* (not per epoch).

    FIX (lr-1): replaces CosineAnnealingLR (per-epoch) which caused large
    early gradients from the reconstruction terms to overshoot.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr + cosine * (base_lr - min_lr)) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.9999, update_every=10):
        self.decay        = decay
        self.update_every = update_every
        self.step         = 0
        self.shadow = {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def update(self, model):
        self.step += 1
        if self.step % self.update_every != 0:
            return
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    self.shadow[n].mul_(self.decay).add_(
                        p.detach().cpu(), alpha=1.0 - self.decay
                    )

    def apply_shadow_to(self, model):
        """Copy shadow weights into model in-place (for final inference)."""
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.device))


# ==============================================================================
# STAGE 1 — Preprocessing
# ==============================================================================

def run_preprocessing():
    print("\n" + "=" * 60)
    print("STAGE 1 — Preprocessing")
    print("=" * 60)
    preprocess_dataset(
        dataset_in = str(DATASET_IN),
        img_dir    = str(PROCESSED_IMG),
        mask_dir   = str(PROCESSED_MASKS),
    )


# ==============================================================================
# STAGE 2 — Inpainting
# ==============================================================================

def run_inpainting():
    print("\n" + "=" * 60)
    print("STAGE 2 — Inpainting")
    print("=" * 60)
    inpaint_dataset(
        processed_ct_dir   = str(PROCESSED_IMG),
        processed_mask_dir = str(PROCESSED_MASKS),
        out_dir            = str(INPAINTED_DIR),
        force_binary       = False,
    )


# ==============================================================================
# STAGE 3 — Diffusion training
# ==============================================================================

def run_diffusion_training():
    print("\n" + "=" * 60)
    print("STAGE 3 — Diffusion training")
    print("=" * 60)

    enable_all_gpus(USE_ALL_GPUS)

    all_entries = build_slice_entries_for_pairs(
        inpainted_dir = str(INPAINTED_DIR),
        orig_dir      = str(PROCESSED_IMG),
        mask_dir      = str(PROCESSED_MASKS),
    )
    print(f"[diffusion] Total slices: {len(all_entries)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = EnhancedUNet(
        in_ch=1, cond_ch=2, base_ch=48,
        ch_mult=(1, 2, 4), num_res_blocks=2,
        time_dim=256,
    ).to(DEVICE)
    model     = setup_model_for_device(model)
    diffusion = Diffusion(timesteps=TIMESTEPS, device=str(DEVICE), use_cosine=True)

    opt = torch.optim.AdamW(model.parameters(), lr=DIFF_LR, weight_decay=1e-4)

    # ── Scheduler (per-batch warmup + cosine) ────────────────────────────────
    steps_per_epoch = max(1, min(len(all_entries), MAX_SLICES_PER_EPOCH) // DIFF_BATCH)
    total_steps     = DIFF_EPOCHS * steps_per_epoch
    warmup_steps    = WARMUP_EPOCHS * steps_per_epoch

    sched = make_warmup_cosine_scheduler(
        opt,
        warmup_steps = warmup_steps,
        total_steps  = total_steps,
        base_lr      = DIFF_LR,
        min_lr       = DIFF_LR * 0.01,
    )

    scaler = _make_scaler(enabled=MIXED_PRECISION)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch, model, opt = load_latest_checkpoint(
        model, opt, str(DIFF_CKPT_DIR), device=str(DEVICE)
    )
    print(f"[diffusion] Resuming from epoch {start_epoch}")

    ema = EMA(model, decay=EMA_DECAY, update_every=EMA_STEP) if USE_EMA else None

    global_step = start_epoch * steps_per_epoch   # keep scheduler in sync on resume

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, DIFF_EPOCHS):

        # Slice sampling
        if MAX_SLICES_PER_EPOCH is not None and len(all_entries) > MAX_SLICES_PER_EPOCH:
            np.random.seed(epoch)
            sel           = np.random.choice(len(all_entries), MAX_SLICES_PER_EPOCH, replace=False)
            entries_epoch = [all_entries[i] for i in sel]
            print(f"[diffusion] Epoch {epoch+1}: {len(entries_epoch)}/{len(all_entries)} slices")
        else:
            entries_epoch = all_entries
            print(f"[diffusion] Epoch {epoch+1}: all {len(all_entries)} slices")

        ds = CTNPZDataset(
            entries    = entries_epoch,
            preprocess_fn = None,
            clip_min   = -200,
            clip_max   = 300,
            force_size = FORCE_SIZE,
        )
        dl = DataLoader(
            ds, batch_size=DIFF_BATCH, shuffle=True,
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda"),
            drop_last=True,
        )

        model.train()
        epoch_loss  = 0.0
        epoch_steps = 0
        psnr_vals   = []
        ssim_vals   = []
        last_vis_batch = None

        # Diagnostic accumulators
        vpred_std_sum = 0.0

        pbar = tqdm(dl, desc=f"Diff {epoch+1}/{DIFF_EPOCHS}", leave=False)
        for batch in pbar:
            opt.zero_grad()

            healthy_t, target_t, tumor_mask_t, liver_mask_t = batch

            # FIX (bug #7): no F.interpolate here — dataset already outputs
            # tensors at FORCE_SIZE.
            healthy_t    = healthy_t.to(DEVICE,    non_blocking=True)
            target_t     = target_t.to(DEVICE,     non_blocking=True)
            tumor_mask_t = tumor_mask_t.to(DEVICE, non_blocking=True)
            liver_mask_t = liver_mask_t.to(DEVICE, non_blocking=True)

            B         = target_t.shape[0]
            t         = torch.randint(0, TIMESTEPS, (B,), device=DEVICE).long()
            cond      = torch.cat([healthy_t, liver_mask_t], dim=1)
            drop_cond = random.random() < 0.2

            with _make_autocast(enabled=MIXED_PRECISION):
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

                main_loss  = (W_FULL_MAIN * core_loss
                              + W_MASK_MAIN * F.mse_loss(pred * mask, target_core * mask))
                recon_loss = W_FULL_RECON * full_recon + W_MASK_RECON * masked_recon
                loss       = (main_loss
                              + recon_loss
                              + W_BG * bg_consist
                              + W_SSIM    * ssim_val)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            # FIX (lr-1): step scheduler per batch, not per epoch
            sched.step()
            global_step += 1

            if ema:
                ema.update(model)

            epoch_loss  += loss.item()
            epoch_steps += 1

            # FIX (debug-1): accumulate v_pred std for end-of-epoch diagnostic
            vpred_std_sum += float(pred.detach().std().item())

            with torch.no_grad():
                psnr_vals.append(psnr(x0_hat, target_t))
                ssim_vals.append(float((1.0 - ssim_loss(x0_hat, target_t)).item()))

            last_vis_batch = (
                healthy_t.detach(), target_t.detach(),
                tumor_mask_t.detach(), liver_mask_t.detach(),
            )

            pbar.set_postfix(
                loss = f"{epoch_loss / epoch_steps:.4f}",
                lr   = f"{sched.get_last_lr()[0]:.2e}",
            )

        avg_loss     = epoch_loss / max(1, epoch_steps)
        avg_psnr     = float(np.mean(psnr_vals))  if psnr_vals  else float("nan")
        avg_ssim     = float(np.mean(ssim_vals))  if ssim_vals  else float("nan")
        avg_vpred_std = vpred_std_sum / max(1, epoch_steps)

        # FIX (debug-1): print v_pred std — if this is near 0 (<0.05) the model
        # is in the degenerate zero-prediction mode and something is wrong with
        # the data pipeline (likely the [-1,1] normalisation not applied).
        print(
            f"  Epoch {epoch+1}/{DIFF_EPOCHS}  "
            f"Loss={avg_loss:.5f}  PSNR={avg_psnr:.2f}  SSIM={avg_ssim:.4f}  "
            f"v_pred_std={avg_vpred_std:.4f}  "
            f"lr={sched.get_last_lr()[0]:.2e}"
        )
        if avg_vpred_std < 0.05:
            print(
                "  [WARN] v_pred_std is very small — model may be in degenerate "
                "zero-prediction mode. Check that dataset outputs are in [-1, 1]."
            )

        # ── Save checkpoint ───────────────────────────────────────────────────
        ckpt = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   opt.state_dict(),
            "scheduler":   sched.state_dict(),
            "train_loss":  avg_loss,
            "global_step": global_step,
        }
        if ema:             ckpt["ema"]    = ema.shadow
        if MIXED_PRECISION: ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, DIFF_CKPT_DIR / "latest_full.pt")
        save_checkpoint_epoch(model, opt, epoch, str(DIFF_CKPT_DIR))

        # ── Visualise using the LIVE model ────────────────────────────────────
        # FIX (bug #8): always use the live model for visualisation.
        # With EMA_DECAY=0.9999 and EMA_STEP=10, the EMA shadow is effectively
        # the random initialisation for the first ~30 epochs — visualising it
        # produces pure noise, which is exactly what was seen at epoch 8-9.
        # The EMA shadow is preserved for final inference use via ema.apply_shadow_to().
        if last_vis_batch is not None:
            try:
                ht, tt, tmt, lmt = last_vis_batch
                visualize_sample(
                    diffusion, model, ht, tt, tmt, lmt,
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
    print("\n=== CT Liver Tumour — Diffusion Pipeline ===")
    print(f"    Device : {DEVICE}")
    print(f"    Dataset: {DATASET_IN}\n")

    # ------------------------------------------------------------------
    # STAGE 1 — Preprocessing
    # FIX (bug #9): was 'if True'. Set to False after the first run.
    # Toggle back to True only when you need to reprocess raw data.
    # ------------------------------------------------------------------
    if False:
        run_preprocessing()

    # ------------------------------------------------------------------
    # STAGE 2 — Inpainting
    # Run once after Stage 1, then set to False.
    # ------------------------------------------------------------------
    if False:
        run_inpainting()

    # ------------------------------------------------------------------
    # STAGE 3 — Diffusion training  (active)
    # ------------------------------------------------------------------
    if True:
        run_diffusion_training()

    print("\n=== Pipeline finished ===")


if __name__ == "__main__":
    main()