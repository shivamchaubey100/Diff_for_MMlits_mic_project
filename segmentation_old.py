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
predict_volume      — full 3-D volume   → predicted mask (Z, H, W)
visualise_seg_batch — save PNG panels for a batch (called from main.py)
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
NUM_CLASSES = 3          # 0 = background, 1 = liver, 2 = tumour
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
# Dataset
# ============================================================

class SegDataset(Dataset):
    """
    Flat slice-level dataset for segmentation training.

    Scans img_dir for *.npz (key='image') and mask_dir for *.npz
    (key='mask').  Files are paired by sorted order — they must
    correspond 1-to-1 (same volume stem).

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

        img_files  = sorted(Path(img_dir).glob("*.npz"))
        mask_files = sorted(Path(mask_dir).glob("*.npz"))

        if not img_files:
            raise FileNotFoundError(f"No .npz files in {img_dir}")
        if len(img_files) != len(mask_files):
            raise ValueError(
                f"Mismatch: {len(img_files)} images vs "
                f"{len(mask_files)} masks in seg dataset."
            )

        self.entries: List[Tuple[str, str, int]] = []
        for ip, mp in zip(img_files, mask_files):
            Z = np.load(ip)["image"].shape[0]
            for z in range(Z):
                self.entries.append((str(ip), str(mp), z))

        print(f"[SegDataset] {len(self.entries)} slices  "
              f"({len(img_files)} volumes)")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        ip, mp, z = self.entries[idx]

        img_s  = np.load(ip)["image"][z].astype(np.float32)
        mask_s = np.load(mp)["mask"][z].astype(np.int64)

        # Preprocess
        img_pp = _preprocess_slice(img_s)
        img_pp = _resize(img_pp,  FORCE_SIZE, cv2.INTER_LINEAR)
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
    """Lightweight geometric augmentation (numpy)."""
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
    return img, mask


# ============================================================
# Loss
# ============================================================

def dice_loss(logits: torch.Tensor, targets: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
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
    device:        str                       = "cuda",
) -> torch.Tensor:
    """
    Combined cross-entropy + Dice loss.

    class_weights upweights rare classes (tumour = class 2).
    Default [0.2, 1.0, 5.0] heavily penalises missing tumour.
    """
    if class_weights is not None:
        w  = torch.tensor(class_weights, dtype=torch.float32,
                          device=device)
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
    Compute per-class Dice and mean IoU.

    logits  : (B, C, H, W)
    targets : (B, H, W)  long
    """
    preds = logits.argmax(dim=1)
    out   = {}
    dices = []
    ious  = []

    for c in range(num_classes):
        pc = (preds   == c).float()
        tc = (targets == c).float()
        inter = (pc * tc).sum().item()
        union = pc.sum().item() + tc.sum().item()
        dice  = (2 * inter + 1e-6) / (union + 1e-6)
        iou   = (inter + 1e-6) / (union - inter + 1e-6)
        out[f"dice_cls{c}"] = dice
        out[f"iou_cls{c}"]  = iou
        dices.append(dice)
        ious.append(iou)

    out["mean_dice"] = float(np.mean(dices))
    out["mean_iou"]  = float(np.mean(ious))
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

    base_ch : channel width at first encoder level (default 32)
    """

    def __init__(
        self,
        in_ch:     int = 1,
        num_class: int = NUM_CLASSES,
        base_ch:   int = 32,
    ):
        super().__init__()
        b = base_ch
        self.enc1      = _DoubleConv(in_ch, b)
        self.enc2      = _Down(b,     b * 2)
        self.enc3      = _Down(b * 2, b * 4)
        self.enc4      = _Down(b * 4, b * 8)
        self.bottleneck= _Down(b * 8, b * 16)
        self.dec4      = _Up(b * 16, b * 8,  b * 8)
        self.dec3      = _Up(b * 8,  b * 4,  b * 4)
        self.dec2      = _Up(b * 4,  b * 2,  b * 2)
        self.dec1      = _Up(b * 2,  b,      b)
        self.head      = nn.Conv2d(b, num_class, 1)

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
    device:   str = "cuda",
    opt=None,
) -> Tuple[nn.Module, int]:
    """
    Load the latest seg_epoch_XXXX.pt from ckpt_dir.
    Returns (model, start_epoch).
    """
    import glob
    files = sorted(glob.glob(str(Path(ckpt_dir) / "seg_epoch_*.pt")))
    if not files:
        print(f"[seg] No checkpoint in {ckpt_dir} — starting from epoch 0.")
        return model, 0
    ck = torch.load(files[-1], map_location=device)
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
    img_slice: np.ndarray,   # (H, W)  raw HU float
    device:    str = "cuda",
) -> np.ndarray:
    """
    Predict segmentation mask for a single 2-D CT slice.

    Returns np.ndarray (H, W) with values in {0, 1, 2}.
    """
    model.eval()
    pp = _preprocess_slice(img_slice)
    pp = _resize(pp, FORCE_SIZE, cv2.INTER_LINEAR)
    t  = torch.from_numpy(pp[None, None]).float().to(device)
    with torch.cuda.amp.autocast(enabled=(device != "cpu")):
        logits = model(t)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


@torch.no_grad()
def predict_volume(
    model:  nn.Module,
    ct_vol: np.ndarray,   # (Z, H, W)  raw HU float
    device: str = "cuda",
) -> np.ndarray:
    """
    Predict segmentation mask for a full 3-D CT volume.

    Returns np.ndarray (Z, H, W) with values in {0, 1, 2}.
    """
    Z     = ct_vol.shape[0]
    preds = np.zeros((Z, *FORCE_SIZE), dtype=np.uint8)
    for z in range(Z):
        preds[z] = predict_mask(model, ct_vol[z], device=device)
    return preds


# ============================================================
# Visualisation (called from main.py)
# ============================================================

# RGBA colour for each class: bg=transparent, liver=green, tumour=red
_CLS_COLORS = np.array([
    [0,   0,   0,   0  ],
    [34,  192, 135, 160],
    [232, 64,  64,  200],
], dtype=np.uint8)


def _colour_mask(mask: np.ndarray) -> np.ndarray:
    """Map integer mask (H, W) → RGB uint8 image."""
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
    """Blend coloured mask over grayscale image → (H, W, 3) uint8."""
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
    imgs:         torch.Tensor,    # (B, 1, H, W)  preprocessed, in [0,1]
    gt_masks:     torch.Tensor,    # (B, H, W)     long
    pred_logits:  torch.Tensor,    # (B, C, H, W)
    epoch:        int,
    vis_dir:      str,
    max_samples:  int = 4,
) -> None:
    """
    Save a grid of 5-panel PNGs for up to max_samples examples from a batch:
        CT | GT mask | Pred mask | GT overlay | Pred overlay
    """
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    preds = pred_logits.argmax(dim=1)   # (B, H, W)
    B     = min(imgs.shape[0], max_samples)

    for i in range(B):
        img_np  = imgs[i, 0].cpu().numpy()          # (H, W)
        gt_np   = gt_masks[i].cpu().numpy()          # (H, W)
        pred_np = preds[i].cpu().numpy()             # (H, W)

        gt_rgb   = _colour_mask(gt_np)
        pred_rgb = _colour_mask(pred_np)
        gt_ov    = _overlay_mask(img_np, gt_np)
        pred_ov  = _overlay_mask(img_np, pred_np)

        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        fig.patch.set_facecolor("#111827")

        panels = [
            (img_np,   "CT input",    "gray"),
            (gt_rgb,   "GT mask",     None),
            (pred_rgb, "Pred mask",   None),
            (gt_ov,    "GT overlay",  None),
            (pred_ov,  "Pred overlay",None),
        ]
        for ax, (img, title, cmap) in zip(axs, panels):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, color="#e5e7eb", fontsize=9, pad=3)
            ax.axis("off")

        # Per-sample Dice for liver and tumour
        m = compute_seg_metrics(
            pred_logits[i:i+1].cpu(),
            gt_masks[i:i+1].cpu(),
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