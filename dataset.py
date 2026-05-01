"""
dataset.py

Dataset for loading preprocessed CT volumes and segmentation masks
saved as .npz for the Enhanced Conditional Diffusion pipeline.

Pairing
-------
Files are matched by numeric suffix extracted from their stems:
    volume-3_inpainted.npz  ↔  volume-3.npz  ↔  volume-3_mask.npz

Each item returns:
    healthy_t    → inpainted slice (healthy CT)   tensor (1,H,W)  [-1,1]
    target_t     → original CT slice              tensor (1,H,W)  [-1,1]
    tumor_mask_t → tumour mask (0/1)              tensor (1,H,W)
    liver_mask_t → liver mask  (0/1)              tensor (1,H,W)

Fixes vs previous version
--------------------------
  FIX (bug #1): Output tensors were never scaled to [-1,1] despite the docstring
                claiming this. The model and diffusion forward process both expect
                [-1,1]. Replaced with _preprocess_slice() which explicitly applies
                the final * 2 - 1 step.

  FIX (bug #2): preprocess_volume_soft is designed for whole 3-D volumes and
                iterates over axis-0 as slices. Passing a fake (1,H,W) array
                worked by accident but was semantically wrong and fragile.
                _preprocess_slice() operates directly on 2-D arrays.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# Numeric suffix extraction
# ============================================================

def _extract_number(path: Path) -> int:
    """
    Return the integer suffix from the volume stem, ignoring trailing tags.
    Examples:
        volume-3.npz            → 3
        volume-3_inpainted.npz  → 3
        volume-3_mask.npz       → 3
    """
    base   = path.stem.split("_")[0]          # "volume-3"
    digits = "".join(filter(str.isdigit, base))
    return int(digits) if digits else -1


# ============================================================
# Build flat slice-entry list — paired by numeric suffix
# ============================================================

def build_slice_entries_for_pairs(
    inpainted_dir: str,
    orig_dir:      str,
    mask_dir:      str,
) -> List[dict]:
    """
    Scan three directories, match files by numeric volume index, and
    return a flat list of per-slice dicts ready for CTNPZDataset.

    Parameters
    ----------
    inpainted_dir : folder containing  volume-N_inpainted.npz  (key='inpainted')
    orig_dir      : folder containing  volume-N.npz             (key='image')
    mask_dir      : folder containing  volume-N_mask.npz        (key='mask')
    """
    inp_by_n:  Dict[int, Path] = {
        _extract_number(p): p for p in Path(inpainted_dir).glob("*.npz")
    }
    orig_by_n: Dict[int, Path] = {
        _extract_number(p): p for p in Path(orig_dir).glob("*.npz")
    }
    mask_by_n: Dict[int, Path] = {
        _extract_number(p): p for p in Path(mask_dir).glob("*.npz")
    }

    common = sorted(set(inp_by_n) & set(orig_by_n) & set(mask_by_n))

    missing_inp  = sorted(set(orig_by_n) - set(inp_by_n))
    missing_orig = sorted(set(inp_by_n)  - set(orig_by_n))
    missing_mask = sorted((set(inp_by_n) | set(orig_by_n)) - set(mask_by_n))
    if missing_inp:
        print(f"[dataset] WARNING: volumes with no inpainted file: {missing_inp}")
    if missing_orig:
        print(f"[dataset] WARNING: inpainted files with no orig CT: {missing_orig}")
    if missing_mask:
        print(f"[dataset] WARNING: volumes with no mask file: {missing_mask}")
    if not common:
        raise FileNotFoundError(
            "No matched inpainted/orig/mask triples found.\n"
            f"  inpainted_dir: {inpainted_dir}\n"
            f"  orig_dir     : {orig_dir}\n"
            f"  mask_dir     : {mask_dir}"
        )

    entries: List[dict] = []
    for n in common:
        inp_p  = str(inp_by_n[n])
        orig_p = str(orig_by_n[n])
        mask_p = str(mask_by_n[n])

        vol = np.load(inp_p)["inpainted"]
        Z   = vol.shape[0]

        for z in range(Z):
            entries.append({
                "inpaint": inp_p,
                "orig":    orig_p,
                "mask":    mask_p,
                "slice":   z,
            })

    print(f"[dataset] {len(common)} volumes  →  {len(entries)} total slices")
    return entries


# ============================================================
# Preprocessing helper — single 2-D slice → [-1, 1]
# ============================================================

def _preprocess_slice(
    slice_2d: np.ndarray,
    clip_min: int,
    clip_max: int,
    out_size: Tuple[int, int],
) -> np.ndarray:
    """
    Clip HU values, normalise to [0,1], resize, then scale to [-1,1].

    FIX (bug #1 + #2): The original code called preprocess_volume_soft
    (a 3-D volume function) with a fake (1,H,W) array and never applied
    the [-1,1] scaling, so tensors were in [0,1]. This function is a
    clean direct replacement that guarantees [-1,1] output.

    Args:
        slice_2d : 2-D float32 array (H, W) in raw HU
        clip_min : lower HU clip
        clip_max : upper HU clip
        out_size : (H_out, W_out)

    Returns:
        float32 array of shape out_size, values in [-1, 1]
    """
    s = np.clip(slice_2d.astype(np.float32), clip_min, clip_max)
    s = (s - clip_min) / float(clip_max - clip_min)      # → [0, 1]
    H, W = out_size
    if s.shape != (H, W):
        s = cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)
    s = s * 2.0 - 1.0                                    # → [-1, 1]
    return s


# ============================================================
# Tensor shape helper
# ============================================================

def ensure_channel_first(x: np.ndarray) -> np.ndarray:
    x = np.squeeze(np.asarray(x))
    if x.ndim == 2:
        x = x[None, ...]
    elif x.ndim == 3 and x.shape[0] != 1:
        x = np.moveaxis(x, -1, 0)[:1]
    return x.astype(np.float32)


# ============================================================
# Dataset
# ============================================================

class CTNPZDataset(Dataset):
    """
    Slice-level dataset for the diffusion model.

    Parameters
    ----------
    entries       : list of dicts from build_slice_entries_for_pairs()
    preprocess_fn : retained for API compatibility but no longer used internally.
                    Preprocessing is now handled by _preprocess_slice to
                    guarantee [-1,1] output range.
    clip_min/max  : HU window
    force_size    : (H, W) to resize every slice to
    """

    def __init__(
        self,
        entries:        List[dict],
        preprocess_fn   = None,     # kept for backwards-compatible call sites
        clip_min:   int             = -100,
        clip_max:   int             = 300,
        force_size: Tuple[int, int] = (256, 256),
    ):
        if not entries:
            raise RuntimeError("CTNPZDataset: entries list is empty.")
        self.entries    = entries
        self.clip_min   = clip_min
        self.clip_max   = clip_max
        self.force_size = force_size

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        z     = entry["slice"]

        inpaint_vol = np.load(entry["inpaint"])["inpainted"].astype(np.float32)
        orig_vol    = np.load(entry["orig"])["image"].astype(np.float32)
        seg_vol     = np.load(entry["mask"])["mask"].astype(np.float32)

        healthy_slice = inpaint_vol[z]   # (H, W)
        target_slice  = orig_vol[z]      # (H, W)
        seg_slice     = seg_vol[z]       # (H, W)

        # ── Masks ─────────────────────────────────────────────────────────────
        if np.max(seg_slice) >= 2:
            tumor_mask = (seg_slice == 2).astype(np.float32)
            liver_mask = (seg_slice >= 1).astype(np.float32)
        else:
            tumor_mask = (seg_slice > 0).astype(np.float32)
            liver_mask = (seg_slice > 0).astype(np.float32)

        H, W = self.force_size
        tumor_mask = cv2.resize(tumor_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        liver_mask = cv2.resize(liver_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        tumor_mask = ensure_channel_first(tumor_mask)   # (1, H, W)
        liver_mask = ensure_channel_first(liver_mask)   # (1, H, W)

        # ── CT slices → [-1, 1] ───────────────────────────────────────────────
        # FIX (bug #1 + #2): use _preprocess_slice, which is explicit about
        # clipping, normalising, resizing, and the final [-1,1] scaling.
        healthy_pp = _preprocess_slice(
            healthy_slice, self.clip_min, self.clip_max, self.force_size
        )
        target_pp = _preprocess_slice(
            target_slice,  self.clip_min, self.clip_max, self.force_size
        )

        healthy_pp = ensure_channel_first(healthy_pp)   # (1, H, W) in [-1,1]
        target_pp  = ensure_channel_first(target_pp)    # (1, H, W) in [-1,1]

        return (
            torch.from_numpy(healthy_pp),
            torch.from_numpy(target_pp),
            torch.from_numpy(tumor_mask),
            torch.from_numpy(liver_mask),
        )