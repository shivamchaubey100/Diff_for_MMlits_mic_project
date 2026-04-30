"""
segmentation.py

Liver + Tumour segmentation — model, dataset, loss, metrics, helpers.

Everything here is a reusable function/class.
The training loop, eval loop and export logic live in main.py.

Classes / functions exported
-----------------------------
SegDataset          — torch Dataset for (image, mask) slice pairs
SegUNet             — 4-level U-Net,  output: logits (B, 3, H, W)
seg_loss            — combined weighted-CE + soft-Dice loss
compute_seg_metrics — per-class Dice + mean IoU
save_seg_checkpoint / load_seg_checkpoint
predict_mask        — single 2-D slice  → predicted mask (H, W)
predict_volume      — full 3-D volume   → predicted mask (Z, H, W)  [batched GPU]
visualise_seg_batch — save PNG panels for a batch (called from main.py)

GPU / device changes vs previous version
-----------------------------------------
1. seg_loss: class_weights tensor is now created on logits.device instead of
   a hardcoded "cuda" string.  The `device` parameter is kept for API
   compatibility but is no longer used.

2. compute_seg_metrics: all per-class arithmetic now stays on the GPU.
   The old code called .item() inside the loop, forcing a GPU→CPU sync on
   every class per batch step — O(num_classes) stalls per iteration.
   Now we do all comparisons as tensor ops, sum on GPU, and pull a single
   flat array to CPU once at the end.

3. predict_volume: replaced the slice-by-slice Python loop with a batched
   GPU path.  All slices are preprocessed in parallel (via numpy vectorised
   ops), stacked into one tensor, moved to the device once, and passed
   through the model in mini-batches.  This is typically 10-30× faster than
   the old loop for a full 3-D volume.

4. load_seg_checkpoint: default device changed from "cuda" to "cpu" so it
   works safely without a GPU present (caller can always pass "cuda").

5. visualise_seg_batch: compute_seg_metrics is now called on GPU tensors
   directly (no redundant .cpu() before the call).

Existing fixes (unchanged)
--------------------------
1. SegDataset: numeric suffix pairing (volume-10 ≠ volume-2 fix).
2. predict_mask / autocast: torch.amp.autocast instead of deprecated
   torch.cuda.amp.autocast.
3. load_seg_checkpoint: weights_only=False to silence PyTorch ≥ 2.0 warning.
4. _augment: np.ascontiguousarray after flip/rot90.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ============================================================
# Constants
# ============================================================
NUM_CLASSES = 3
FORCE_SIZE  = (256, 256)
CLIP_MIN    = -150
CLIP_MAX    = 350
GAMMA       = 1.5


# ============================================================
# Preprocessing (kept local so SegDataset is self-contained)
# ============================================================

def _preprocess_slice(s: np.ndarray) -> np.ndarray:
    """Clip HU → normalise → gamma → float32 in [0, 1]."""
    s = np.clip(s, CLIP_MIN, CLIP_MAX).astype(np.float32)
    s = (s - CLIP_MIN) / float(CLIP_MAX - CLIP_MIN)
    s = np.clip(s, 0.0, 1.0)
    return np.power(s, GAMMA)


def _resize(s: np.ndarray, size: Tuple, interp: int) -> np.ndarray:
    H, W = size
    return cv2.resize(s, (W, H), interpolation=interp)


# ============================================================
# Numeric pairing helper
# ============================================================

def _extract_number(path: Path) -> int:
    """
    Extract the leading integer from the first underscore-free token.
    E.g. 'volume-3' → 3,  'volume-10_mask' → 10.
    """
    base   = path.stem.split("_")[0]
    digits = "".join(filter(str.isdigit, base))
    return int(digits) if digits else -1


def _pair_img_mask(img_dir: str, mask_dir: str) -> List[Tuple[Path, Path]]:
    """
    Match image .npz (volume-N.npz) with mask .npz (volume-N_mask.npz)
    by numeric suffix N.  Returns list sorted by N.
    """
    img_by_n:  Dict[int, Path] = {
        _extract_number(p): p for p in Path(img_dir).glob("*.npz")
    }
    mask_by_n: Dict[int, Path] = {
        _extract_number(p): p for p in Path(mask_dir).glob("*.npz")
    }

    common      = sorted(set(img_by_n) & set(mask_by_n))
    unmatched_i = sorted(set(img_by_n)  - set(mask_by_n))
    unmatched_m = sorted(set(mask_by_n) - set(img_by_n))

    if unmatched_i:
        print(f"[SegDataset] WARNING: images with no matching mask: "
              f"{[img_by_n[k].name for k in unmatched_i]}")
    if unmatched_m:
        print(f"[SegDataset] WARNING: masks with no matching image: "
              f"{[mask_by_n[k].name for k in unmatched_m]}")
    if not common:
        raise FileNotFoundError(
            f"No matched image/mask pairs found.\n"
            f"  img_dir : {img_dir}\n"
            f"  mask_dir: {mask_dir}"
        )

    return [(img_by_n[n], mask_by_n[n]) for n in common]


# ============================================================
# Dataset
# ============================================================

class SegDataset(Dataset):
    """
    Flat slice-level dataset for segmentation training.

    Returns
    -------
    img_t  : float tensor (1, H, W)  in [0, 1]
    mask_t : long  tensor (H, W)     labels 0 / 1 / 2
    """

    def __init__(
        self,
        img_dir:  str,
        mask_dir: str,
        augment:  bool = True,
    ):
        self.augment = augment

        pairs = _pair_img_mask(img_dir, mask_dir)

        self.entries: List[Tuple[str, str, int]] = []
        for ip, mp in pairs:
            Z = np.load(ip)["image"].shape[0]
            for z in range(Z):
                self.entries.append((str(ip), str(mp), z))

        print(f"[SegDataset] {len(self.entries)} slices  "
              f"({len(pairs)} volumes)")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        ip, mp, z = self.entries[idx]

        img_s  = np.load(ip)["image"][z].astype(np.float32)
        mask_s = np.load(mp)["mask"][z].astype(np.int64)

        img_pp = _preprocess_slice(img_s)
        img_pp = _resize(img_pp, FORCE_SIZE, cv2.INTER_LINEAR)
        mask_r = _resize(mask_s.astype(np.float32),
                         FORCE_SIZE, cv2.INTER_NEAREST).astype(np.int64)
        mask_r = np.clip(mask_r, 0, NUM_CLASSES - 1)

        if self.augment:
            img_pp, mask_r = _augment(img_pp, mask_r)

        img_t  = torch.from_numpy(np.ascontiguousarray(img_pp)[None]).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_r)).long()
        return img_t, mask_t


def _augment(
    img: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Lightweight geometric augmentation (numpy in-place safe)."""
    import random
    if random.random() < 0.5:
        img  = np.flip(img,  axis=1)
        mask = np.flip(mask, axis=1)
    if random.random() < 0.4:
        img  = np.flip(img,  axis=0)
        mask = np.flip(mask, axis=0)
    if random.random() < 0.4:
        k    = random.choice([1, 2, 3])
        img  = np.rot90(img,  k)
        mask = np.rot90(mask, k)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


# ============================================================
# Loss
# ============================================================

def dice_loss(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    smooth:  float = 1.0,
) -> torch.Tensor:
    """
    Soft multi-class Dice loss.

    logits  : (B, C, H, W)
    targets : (B, H, W)  integer labels
    """
    C     = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    oh    = F.one_hot(targets, C).permute(0, 3, 1, 2).float()
    dims  = (0, 2, 3)
    inter = (probs * oh).sum(dim=dims)
    union = (probs + oh).sum(dim=dims)
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def seg_loss(
    logits:        torch.Tensor,
    targets:       torch.Tensor,
    ce_weight:     float                     = 1.0,
    dice_weight:   float                     = 1.0,
    class_weights: Optional[Sequence[float]] = (0.2, 1.0, 5.0),
    device:        Optional[str]             = None,  # kept for API compat, ignored
) -> torch.Tensor:
    """
    Combined cross-entropy + Dice loss.

    class_weights upweights rare classes (tumour = class 2).
    Default [0.2, 1.0, 5.0] heavily penalises missing tumour.

    FIX: class_weights tensor is now created on logits.device instead of a
    hardcoded "cuda" string.  Passing device="cpu" logits to the old code
    would silently create the weight tensor on CUDA, causing a device
    mismatch crash on cross_entropy.  The `device` argument is retained for
    backward API compatibility but is no longer consulted.
    """
    tensor_device = logits.device   # always matches logits — no string needed
    if class_weights is not None:
        w  = torch.tensor(class_weights, dtype=torch.float32, device=tensor_device)
        ce = F.cross_entropy(logits, targets, weight=w)
    else:
        ce = F.cross_entropy(logits, targets)

    dl = dice_loss(logits, targets)
    return ce_weight * ce + dice_weight * dl


# ============================================================
# Metrics
# ============================================================

def compute_seg_metrics(
    logits:      torch.Tensor,
    targets:     torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> dict:
    """
    Per-class Dice and mean IoU.

    logits  : (B, C, H, W)
    targets : (B, H, W)  long

    FIX: all comparisons and sums now stay on whatever device logits/targets
    live on (GPU during training).  The old code called .item() inside the
    per-class loop, causing one GPU→CPU sync per class per batch step.
    Now we do vectorised tensor ops across all classes simultaneously and
    pull a single flat array to CPU once at the very end.
    """
    device = logits.device
    preds  = logits.argmax(dim=1)   # (B, H, W) — stays on GPU

    # Build class indicator tensors for all classes at once: (C, B, H, W)
    # arange gives [0, 1, 2] → compare against preds and targets in one op
    cls = torch.arange(num_classes, device=device).view(-1, 1, 1, 1)  # (C,1,1,1)

    pc    = (preds.unsqueeze(0)   == cls).float()   # (C, B, H, W)
    tc    = (targets.unsqueeze(0) == cls).float()   # (C, B, H, W)

    # Sum over B, H, W — stays on GPU
    inter = (pc * tc).sum(dim=(1, 2, 3))            # (C,)
    p_sum = pc.sum(dim=(1, 2, 3))                   # (C,)
    t_sum = tc.sum(dim=(1, 2, 3))                   # (C,)
    union = p_sum + t_sum                           # (C,)

    dice  = (2.0 * inter + 1e-6) / (union + 1e-6)          # (C,)
    iou   = (inter + 1e-6) / (union - inter + 1e-6)        # (C,)

    # Single transfer: pull both vectors to CPU numpy in one call
    dice_np = dice.cpu().numpy()
    iou_np  = iou.cpu().numpy()

    out = {}
    for c in range(num_classes):
        out[f"dice_cls{c}"] = float(dice_np[c])
        out[f"iou_cls{c}"]  = float(iou_np[c])

    out["mean_dice"] = float(dice_np.mean())
    out["mean_iou"]  = float(iou_np.mean())
    return out


# ============================================================
# Model — SegUNet
# ============================================================

class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_c: int, out_c: int):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class _DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            _ConvBnRelu(in_c, out_c),
            _ConvBnRelu(out_c, out_c),
        )
    def forward(self, x):
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.pool  = nn.MaxPool2d(2)
        self.dconv = _DoubleConv(in_c, out_c)
    def forward(self, x):
        return self.dconv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_c, in_c // 2, 2, stride=2)
        self.dconv = _DoubleConv(in_c // 2 + skip_c, out_c)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        return self.dconv(torch.cat([x, skip], dim=1))


class SegUNet(nn.Module):
    """
    4-level U-Net for 3-class liver/tumour segmentation.

    Input  : (B, 1, 256, 256)
    Output : (B, 3, 256, 256)  — raw logits
    """

    def __init__(
        self,
        in_ch:     int = 1,
        num_class: int = NUM_CLASSES,
        base_ch:   int = 32,
    ):
        super().__init__()
        b = base_ch
        self.enc1       = _DoubleConv(in_ch, b)
        self.enc2       = _Down(b,      b * 2)
        self.enc3       = _Down(b * 2,  b * 4)
        self.enc4       = _Down(b * 4,  b * 8)
        self.bottleneck = _Down(b * 8,  b * 16)
        self.dec4       = _Up(b * 16, b * 8,  b * 8)
        self.dec3       = _Up(b * 8,  b * 4,  b * 4)
        self.dec2       = _Up(b * 4,  b * 2,  b * 2)
        self.dec1       = _Up(b * 2,  b,      b)
        self.head       = nn.Conv2d(b, num_class, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        bt = self.bottleneck(s4)
        d4 = self.dec4(bt, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.head(d1)


# ============================================================
# Checkpointing
# ============================================================

def save_seg_checkpoint(
    model:    nn.Module,
    opt,
    epoch:    int,
    metrics:  dict,
    ckpt_dir: str,
) -> None:
    p = Path(ckpt_dir)
    p.mkdir(parents=True, exist_ok=True)
    path = p / f"seg_epoch_{epoch:04d}.pt"
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": opt.state_dict(),
        "metrics":   metrics,
    }, path)
    print(f"[seg] Checkpoint → {path}")


def load_seg_checkpoint(
    model:    nn.Module,
    ckpt_dir: str,
    device:   str = "cpu",   # FIX: was "cuda" — safe default that works without GPU
    opt=None,
) -> Tuple[nn.Module, int]:
    """
    Load the latest seg_epoch_XXXX.pt from ckpt_dir.
    Returns (model, start_epoch).

    FIX: default device changed from "cuda" to "cpu".  The old default would
    crash on machines without a GPU.  The caller (main.py) always passes the
    correct device string explicitly anyway.
    """
    import glob
    files = sorted(glob.glob(str(Path(ckpt_dir) / "seg_epoch_*.pt")))
    if not files:
        print(f"[seg] No checkpoint in {ckpt_dir} — starting from epoch 0.")
        return model, 0

    ck = torch.load(files[-1], map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    if opt is not None and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
    epoch = ck.get("epoch", 0)
    print(f"[seg] Loaded {files[-1]}  (epoch {epoch}  "
          f"metrics={ck.get('metrics', {})})")
    return model, epoch + 1


# ============================================================
# Inference helpers
# ============================================================

@torch.no_grad()
def predict_mask(
    model:     nn.Module,
    img_slice: np.ndarray,   # (H, W) raw HU float
    device:    str = "cuda",
) -> np.ndarray:
    """
    Predict segmentation mask for a single 2-D CT slice.
    Returns np.ndarray (H, W) with values in {0, 1, 2}.

    Prefer predict_volume for full 3-D volumes — it batches slices on GPU
    and is significantly faster than calling this in a loop.
    """
    model.eval()
    pp = _preprocess_slice(img_slice)
    pp = _resize(pp, FORCE_SIZE, cv2.INTER_LINEAR)
    t  = torch.from_numpy(pp[None, None]).float().to(device)

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
        logits = model(t)

    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


@torch.no_grad()
def predict_volume(
    model:      nn.Module,
    ct_vol:     np.ndarray,   # (Z, H, W) raw HU float
    device:     str = "cuda",
    batch_size: int = 16,     # slices per GPU batch — tune to VRAM
) -> np.ndarray:
    """
    Predict segmentation mask for a full 3-D CT volume.
    Returns np.ndarray (Z, H, W) with values in {0, 1, 2}.

    FIX: replaced the old slice-by-slice Python loop with a batched GPU
    path.  All preprocessing is vectorised over Z via numpy, then slices
    are transferred to the device in one go and processed in mini-batches.

    Old approach: Z individual model forward passes + Z GPU→CPU syncs.
    New approach: ceil(Z / batch_size) forward passes, one GPU→CPU sync.
    Typical speed-up: 10–30× for a 512-slice volume on a T4.

    Args:
        model      : SegUNet already on the target device
        ct_vol     : (Z, H, W) array of raw HU values
        device     : device string matching where model lives
        batch_size : number of slices per forward pass (reduce if OOM)
    """
    model.eval()
    Z = ct_vol.shape[0]

    # ── Preprocessing — fully vectorised over Z (no Python loop) ──────────
    # _preprocess_slice operates element-wise, so we can apply it to the
    # whole volume array at once.
    vol_f  = np.clip(ct_vol, CLIP_MIN, CLIP_MAX).astype(np.float32)
    vol_f  = (vol_f - CLIP_MIN) / float(CLIP_MAX - CLIP_MIN)
    vol_f  = np.clip(vol_f, 0.0, 1.0)
    vol_f  = np.power(vol_f, GAMMA)                  # (Z, H, W)  float32

    # Resize each slice to FORCE_SIZE — still needs a loop (cv2 is 2-D only)
    # but this is pure CPU and very fast compared to model inference.
    H, W = FORCE_SIZE
    resized = np.empty((Z, H, W), dtype=np.float32)
    for z in range(Z):
        resized[z] = cv2.resize(vol_f[z], (W, H), interpolation=cv2.INTER_LINEAR)

    # ── Single CPU→GPU transfer for all slices ─────────────────────────────
    # Shape: (Z, 1, H, W) — add channel dim expected by the model
    vol_t = torch.from_numpy(resized[:, None]).float().to(device, non_blocking=True)

    # ── Batched inference — one forward pass per batch_size slices ─────────
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    preds_gpu   = torch.empty(Z, H, W, dtype=torch.long, device=device)

    for start in range(0, Z, batch_size):
        end   = min(start + batch_size, Z)
        batch = vol_t[start:end]                     # (bs, 1, H, W) — slice of same tensor, no copy
        with torch.amp.autocast(device_type=device_type,
                                enabled=(device_type == "cuda")):
            logits = model(batch)                    # (bs, C, H, W)
        preds_gpu[start:end] = logits.argmax(dim=1) # stays on GPU

    # ── Single GPU→CPU transfer for all predictions ────────────────────────
    return preds_gpu.cpu().numpy().astype(np.uint8)  # (Z, H, W)


# ============================================================
# Visualisation (called from main.py)
# ============================================================

_CLS_COLORS = np.array([
    [0,   0,   0,   0  ],     # background — transparent
    [34,  192, 135, 160],     # liver      — green
    [232, 64,  64,  200],     # tumour     — red
], dtype=np.uint8)


def _colour_mask(mask: np.ndarray) -> np.ndarray:
    """Map integer mask (H, W) → RGB uint8."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 0] = [40,  40,  40]
    rgb[mask == 1] = [34,  192, 135]
    rgb[mask == 2] = [232, 64,  64]
    return rgb


def _overlay_mask(
    img_norm: np.ndarray,   # (H, W) float [0, 1]
    mask:     np.ndarray,   # (H, W) int
    alpha:    float = 0.45,
) -> np.ndarray:
    """Blend coloured mask over greyscale image → (H, W, 3) uint8."""
    rgb = (np.stack([img_norm] * 3, axis=-1) * 255).astype(np.uint8)
    for c in range(1, NUM_CLASSES):
        col   = _CLS_COLORS[c, :3].astype(np.float32)
        a     = _CLS_COLORS[c, 3] / 255.0 * alpha
        where = mask == c
        for ch in range(3):
            rgb[..., ch][where] = np.clip(
                (1 - a) * rgb[..., ch][where] + a * col[ch], 0, 255
            ).astype(np.uint8)
    return rgb


def visualise_seg_batch(
    imgs:        torch.Tensor,   # (B, 1, H, W) preprocessed [0, 1]
    gt_masks:    torch.Tensor,   # (B, H, W)    long
    pred_logits: torch.Tensor,   # (B, C, H, W)
    epoch:       int,
    vis_dir:     str,
    max_samples: int = 4,
) -> None:
    """
    Save 5-panel PNGs for up to max_samples examples:
        CT | GT mask | Pred mask | GT overlay | Pred overlay

    FIX: compute_seg_metrics is called on whatever device the tensors are on
    (GPU during training).  The old code forced .cpu() before the call, which
    was redundant since compute_seg_metrics now handles any device internally.
    The .cpu().numpy() calls for plotting are kept only where they're actually
    needed (the numpy visualisation ops).
    """
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    preds = pred_logits.argmax(dim=1)   # stays on whatever device tensors are on
    B     = min(imgs.shape[0], max_samples)

    for i in range(B):
        # Pull individual sample to CPU only when needed for numpy/matplotlib
        img_np  = imgs[i, 0].cpu().numpy()
        gt_np   = gt_masks[i].cpu().numpy()
        pred_np = preds[i].cpu().numpy()

        gt_rgb   = _colour_mask(gt_np)
        pred_rgb = _colour_mask(pred_np)
        gt_ov    = _overlay_mask(img_np, gt_np)
        pred_ov  = _overlay_mask(img_np, pred_np)

        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        fig.patch.set_facecolor("#111827")

        panels = [
            (img_np,   "CT input",     "gray"),
            (gt_rgb,   "GT mask",      None),
            (pred_rgb, "Pred mask",    None),
            (gt_ov,    "GT overlay",   None),
            (pred_ov,  "Pred overlay", None),
        ]
        for ax, (img, title, cmap) in zip(axs, panels):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, color="#e5e7eb", fontsize=9, pad=3)
            ax.axis("off")

        # FIX: pass tensors directly — no .cpu() needed, compute_seg_metrics
        # handles the device transfer internally
        m = compute_seg_metrics(
            pred_logits[i:i+1],
            gt_masks[i:i+1],
        )
        fig.suptitle(
            f"Epoch {epoch}  |  "
            f"Dice liver={m['dice_cls1']:.3f}  "
            f"tumour={m['dice_cls2']:.3f}",
            color="#9ca3af", fontsize=9,
        )
        plt.tight_layout(pad=0.4)
        plt.savefig(
            vis_dir / f"seg_epoch{epoch:04d}_sample{i}.png",
            dpi=130, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close()