"""
preprocess.py

Preprocesses CT volumes and segmentation masks into consistent,
normalised NumPy arrays (.npz) for model training.

Pairing logic
-------------
volume-N.nii  <-->  segmentation-N.nii   (matched by numeric suffix N)

Key features
------------
- HU clipping: CLIP_MIN .. CLIP_MAX  (default -150 .. 350)
- Linear normalisation to [0, 1]
- Gamma correction  (default 1.5)
- Resize slices to OUT_SIZE  (default 256 x 256)
- Saves each volume  → .npz  key='image'
        each mask   → .npz  key='mask'
- Optional per-volume visualisation PNGs saved to vis_dir
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

# ============================================================
# Default configuration
# ============================================================
CLIP_MIN        = -150
CLIP_MAX        = 350
OUT_SIZE        = (256, 256)
GAMMA           = 1.5
MIN_MASK_VOXELS = 1000


# ============================================================
# Core slice-level preprocessing
# ============================================================

def preprocess_volume_soft(
    vol:      np.ndarray,
    clip_min: int   = CLIP_MIN,
    clip_max: int   = CLIP_MAX,
    out_size: Tuple = OUT_SIZE,
    gamma:    float = GAMMA,
) -> np.ndarray:
    """
    Preprocess a 3-D CT volume (Z, H, W) → float32 in [0, 1].

    Pipeline: clip HU → min-max normalise → gamma → resize (bilinear).
    """
    vol = np.clip(vol, clip_min, clip_max).astype(np.float32)
    vol = (vol - clip_min) / float(clip_max - clip_min)
    vol = np.clip(vol, 0.0, 1.0)
    vol = np.power(vol, float(gamma))

    out_h, out_w = out_size
    resized = [
        cv2.resize(s, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        for s in vol
    ]
    return np.stack(resized, axis=0).astype(np.float32)


def preprocess_mask_soft(
    seg:      np.ndarray,
    out_size: Tuple = OUT_SIZE,
) -> np.ndarray:
    """
    Resize segmentation mask (Z, H, W) preserving integer labels.
    Labels: 0 = background, 1 = liver, 2 = tumour.
    Uses nearest-neighbour interpolation to keep labels exact.
    """
    seg_int = seg.astype(np.int32)
    out_h, out_w = out_size
    resized = [
        cv2.resize(s, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        for s in seg_int
    ]
    return np.stack(resized, axis=0).astype(np.float32)


# ============================================================
# Pairing: match volume-N with segmentation-N by numeric suffix
# ============================================================

def _extract_number(path: Path) -> int:
    """Extract trailing integer from stem, e.g. 'volume-3' → 3."""
    digits = "".join(filter(str.isdigit, path.stem))
    return int(digits) if digits else -1


def _list_matched_pairs(dataset_in: Path):
    """
    Scan dataset_in recursively for volume-N.nii* and segmentation-N.nii*
    files.  Match them by the numeric suffix N.

    Returns sorted list of (vol_path, seg_path) tuples.
    Prints warnings for any unmatched files.
    """
    vol_by_n = {
        _extract_number(p): p
        for p in dataset_in.rglob("volume-*.nii*")
    }
    seg_by_n = {
        _extract_number(p): p
        for p in dataset_in.rglob("segmentation-*.nii*")
    }

    common       = sorted(set(vol_by_n) & set(seg_by_n))
    unmatched_v  = sorted(set(vol_by_n) - set(seg_by_n))
    unmatched_s  = sorted(set(seg_by_n) - set(vol_by_n))

    if unmatched_v:
        print(f"[preprocess] WARNING: volumes with no matching segmentation: "
              f"{[vol_by_n[k].name for k in unmatched_v]}")
    if unmatched_s:
        print(f"[preprocess] WARNING: segmentations with no matching volume: "
              f"{[seg_by_n[k].name for k in unmatched_s]}")

    pairs = [(vol_by_n[n], seg_by_n[n]) for n in common]
    print(f"[preprocess] Matched {len(pairs)} volume/segmentation pairs.")
    return pairs


# ============================================================
# Visualisation
# ============================================================

def _visualise_preprocessing(
    ct_proc:  np.ndarray,   # (Z, H, W) float32  [0, 1]
    seg_proc: np.ndarray,   # (Z, H, W) float32  labels 0/1/2
    stem:     str,
    vis_dir:  Path,
):
    """
    Save a 4-panel PNG for one volume:
        [ CT slice | liver mask | tumour mask | colour overlay ]

    Selects the slice with the most tumour pixels
    (or most liver pixels if no tumour present).
    """
    vis_dir.mkdir(parents=True, exist_ok=True)

    tumour_area = (seg_proc == 2).sum(axis=(1, 2))
    liver_area  = (seg_proc == 1).sum(axis=(1, 2))
    if tumour_area.max() > 0:
        idx = int(np.argmax(tumour_area))
    else:
        idx = int(np.argmax(liver_area))

    ct_sl     = ct_proc[idx]
    liver_m   = (seg_proc[idx] == 1).astype(np.float32)
    tumour_m  = (seg_proc[idx] == 2).astype(np.float32)

    # Colour overlay: green = liver, red = tumour
    overlay = np.stack([ct_sl, ct_sl, ct_sl], axis=-1).copy()
    overlay[..., 1] = np.clip(overlay[..., 1] + 0.35 * liver_m,  0, 1)
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.45 * tumour_m, 0, 1)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#111827")
    data   = [ct_sl,  liver_m,   tumour_m,  overlay]
    titles = ["CT (processed)", "Liver mask", "Tumour mask", "Overlay"]
    cmaps  = ["gray", "Greens",  "Reds",    None]

    for ax, img, title, cmap in zip(axs, data, titles, cmaps):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, color="#e5e7eb", fontsize=10, pad=4)
        ax.axis("off")

    pct_liver  = 100 * liver_m.mean()
    pct_tumour = 100 * tumour_m.mean()
    fig.suptitle(
        f"{stem}  |  slice {idx}  |  "
        f"liver {pct_liver:.1f}%  tumour {pct_tumour:.2f}%",
        color="#9ca3af", fontsize=9,
    )
    plt.tight_layout(pad=0.5)
    plt.savefig(vis_dir / f"{stem}_preprocess.png", dpi=130,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# ============================================================
# Dataset-level entry point
# ============================================================

def preprocess_dataset(
    dataset_in:      str,
    img_dir:         str,
    mask_dir:        str,
    clip_min:        int            = CLIP_MIN,
    clip_max:        int            = CLIP_MAX,
    out_size:        Tuple          = OUT_SIZE,
    gamma:           float          = GAMMA,
    min_mask_voxels: int            = MIN_MASK_VOXELS,
    vis_dir:         Optional[str]  = None,
    n_vis:           int            = 5,
) -> int:
    """
    Preprocess all matched volume-N / segmentation-N pairs and save
    as compressed .npz files.

    Parameters
    ----------
    dataset_in      : root folder with NIfTI files (searched recursively)
    img_dir         : output folder for preprocessed CT volumes
    mask_dir        : output folder for preprocessed masks
    vis_dir         : if given, save visualisation PNGs here
    n_vis           : number of volumes to visualise (first N processed)

    Returns
    -------
    int : number of volume pairs successfully saved
    """
    IMG_DIR  = Path(img_dir);  IMG_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR = Path(mask_dir); MASK_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR  = Path(vis_dir) if vis_dir else None

    pairs = _list_matched_pairs(Path(dataset_in))
    if not pairs:
        raise FileNotFoundError(
            f"No matched volume-N / segmentation-N pairs in {dataset_in}"
        )

    saved = 0
    for i, (vol_path, seg_path) in enumerate(
        tqdm(pairs, desc="Preprocessing")
    ):
        vol_nii = nib.load(str(vol_path)).get_fdata()
        seg_nii = nib.load(str(seg_path)).get_fdata()

        # Ensure (Z, H, W) — NIfTI often stores as (H, W, Z)
        if vol_nii.ndim == 3 and vol_nii.shape[2] < max(vol_nii.shape[:2]):
            vol_nii = np.transpose(vol_nii, (2, 0, 1))
            seg_nii = np.transpose(seg_nii, (2, 0, 1))

        mask_voxels = int((seg_nii > 0).sum())
        if mask_voxels < min_mask_voxels:
            print(f"[preprocess] Skipping {vol_path.name}: "
                  f"mask too small ({mask_voxels} voxels).")
            continue

        ct_proc  = preprocess_volume_soft(vol_nii, clip_min, clip_max,
                                          out_size, gamma)
        seg_proc = preprocess_mask_soft(seg_nii, out_size)

        # Use the volume stem as the shared key, e.g. "volume-3"
        stem = vol_path.stem
        np.savez_compressed(IMG_DIR  / f"{stem}.npz",      image=ct_proc)
        np.savez_compressed(MASK_DIR / f"{stem}_mask.npz", mask=seg_proc)

        if VIS_DIR is not None and i < n_vis:
            _visualise_preprocessing(ct_proc, seg_proc, stem, VIS_DIR)

        saved += 1

    print(f"[preprocess] Done. {saved} pairs saved.")
    if VIS_DIR:
        print(f"[preprocess] Visualisations → {VIS_DIR}")
    return saved