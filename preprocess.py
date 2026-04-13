"""
preprocess.py

Preprocesses CT volumes and segmentation masks into consistent,
normalized NumPy arrays (.npz) for model training.

Key features:
 - HU clipping: CLIP_MIN .. CLIP_MAX (default -100 .. 300)
 - Linear normalization to [0,1]
 - Gamma correction (default gamma=2.0)
 - Resize slices to OUT_SIZE (default 512x512)
 - Saves each volume and its segmentation as .npz files
 - No Gaussian blur (binary segmentation masks preserved)
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
import nibabel as nib
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
CLIP_MIN = -150          # HU lower bound
CLIP_MAX = 350           # HU upper bound
OUT_SIZE = (256, 256)    # Output (H, W)
GAMMA = 1.5              # Gamma correction factor
MIN_MASK_VOXELS = 1000   # Skip small/empty masks


# ============================================================
# Core Preprocessing Functions
# ============================================================

def preprocess_volume_soft(
    vol: np.ndarray,
    clip_min: int = CLIP_MIN,
    clip_max: int = CLIP_MAX,
    out_size: Tuple[int, int] = OUT_SIZE,
    gamma: float = GAMMA,
) -> np.ndarray:
    """
    Preprocess a 3D CT volume.

    Steps:
      1. Clip HU values to [clip_min, clip_max]
      2. Normalize to [0,1]
      3. Apply gamma correction (enhance mid-range contrast)
      4. Resize each slice to out_size (bilinear)
      5. Return float32 array (Z,H,W) in [0,1]

    Returns:
      np.ndarray (Z, H, W), dtype=float32
    """
    vol = np.clip(vol, clip_min, clip_max)
    vol = (vol - clip_min) / (clip_max - clip_min)
    vol = np.clip(vol, 0.0, 1.0)
    vol = np.power(vol, float(gamma))

    out_h, out_w = out_size
    resized_slices = [
        cv2.resize(s, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        for s in vol
    ]
    return np.stack(resized_slices, axis=0).astype(np.float32)


def preprocess_mask_soft(
    seg: np.ndarray,
    out_size: Tuple[int, int] = OUT_SIZE,
) -> np.ndarray:
    """
    Preprocess segmentation mask volume while preserving labels:
       0 = background
       1 = liver
       2 = tumour

    Steps:
      - Keep integer labels, no binarization
      - Resize with nearest-neighbor (keeps labels intact)
      - Return float32 array (Z,H,W)
    """
    # Ensure mask is integer (0,1,2,...)
    seg_lbl = seg.astype(np.int32)

    out_h, out_w = out_size
    resized_slices = [
        cv2.resize(s, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        for s in seg_lbl
    ]

    return np.stack(resized_slices, axis=0).astype(np.float32)



# ============================================================
# Dataset-Level Preprocessing
# ============================================================

def _list_volume_seg_pairs(dataset_in: Path):
    """Finds and numerically sorts volume-*.nii and segmentation-*.nii pairs."""
    vol_files = sorted(
        [p for p in dataset_in.glob('**/volume-*.nii')],
        key=lambda x: int(''.join(filter(str.isdigit, x.stem))) if any(c.isdigit() for c in x.stem) else 0,
    )
    seg_files = sorted(
        [p for p in dataset_in.glob('**/segmentation-*.nii')],
        key=lambda x: int(''.join(filter(str.isdigit, x.stem))) if any(c.isdigit() for c in x.stem) else 0,
    )
    return vol_files, seg_files


def preprocess_dataset(
    dataset_in: str,
    img_dir: str,
    mask_dir: str,
    clip_min: int = CLIP_MIN,
    clip_max: int = CLIP_MAX,
    out_size: Tuple[int, int] = OUT_SIZE,
    gamma: float = GAMMA,
    min_mask_voxels: int = MIN_MASK_VOXELS,
) -> int:
    """
    Process all NIfTI pairs (volume-*.nii, segmentation-*.nii)
    and save preprocessed 3D arrays as .npz files.

    Args:
        dataset_in: Input folder containing NIfTI files
        img_dir: Output folder for preprocessed CT volumes (.npz)
        mask_dir: Output folder for preprocessed masks (.npz)
        clip_min, clip_max, gamma, out_size: preprocessing parameters
        min_mask_voxels: skip volumes with very small/empty masks

    Returns:
        int: number of volume pairs processed and saved
    """
    DATASET_IN = Path(dataset_in)
    IMG_DIR = Path(img_dir)
    MASK_DIR = Path(mask_dir)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    vol_files, seg_files = _list_volume_seg_pairs(DATASET_IN)
    if not vol_files:
        raise FileNotFoundError(f"No 'volume-*.nii' found in {DATASET_IN}")

    if len(vol_files) != len(seg_files):
        print(f"[WARNING] Found {len(vol_files)} volumes and {len(seg_files)} masks — check pairing consistency.")

    saved = 0
    for vol_path, seg_path in tqdm(zip(vol_files, seg_files), total=len(vol_files), desc="Preprocessing to NPZ"):
        vol_nii = nib.load(str(vol_path)).get_fdata()
        seg_nii = nib.load(str(seg_path)).get_fdata()

        # Normalize orientation: ensure Z,H,W order
        if vol_nii.ndim == 3 and vol_nii.shape[2] < max(vol_nii.shape[0], vol_nii.shape[1]):
            vol_nii = np.transpose(vol_nii, (2, 0, 1))
            seg_nii = np.transpose(seg_nii, (2, 0, 1))

        # Skip small/empty masks
        mask_voxels = int(np.sum(seg_nii > 0))
        if mask_voxels < min_mask_voxels:
            print(f"[INFO] Skipping {vol_path.name}: mask too small ({mask_voxels} voxels).")
            continue

        # vol_proc = preprocess_volume_soft(vol_nii, clip_min, clip_max, out_size, gamma)
        seg_proc = preprocess_mask_soft(seg_nii, out_size)

        # Save as compressed .npz files
        # np.savez_compressed(IMG_DIR / f"{vol_path.stem}.npz", image=vol_proc)
        np.savez_compressed(MASK_DIR / f"{seg_path.stem}.npz", mask=seg_proc)
        saved += 1

    print(f"✅ Saved {saved} preprocessed volume pairs to {IMG_DIR} and {MASK_DIR}")
    return saved
