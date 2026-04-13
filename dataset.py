"""
dataset.py (NO AUGMENTATION)

Dataset for loading preprocessed CT volumes and segmentation masks saved as .npz
for the Enhanced Conditional Diffusion pipeline.

Each item returns:
    healthy_t     → inpainted slice (healthy CT),   tensor (1,H,W), in [-1,1]
    target_t      → original CT slice,              tensor (1,H,W), in [-1,1]
    tumor_mask_t  → tumor mask (0/1),               tensor (1,H,W)
    liver_mask_t  → liver mask (0/1),               tensor (1,H,W)
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Tuple


# ---------------------------------------------------------
# Build list of slice entries
# ---------------------------------------------------------
def build_slice_entries_for_pairs(pairs):
    entries = []

    for inpaint_p, orig_p, mask_p in pairs:
        inpaint_p = str(inpaint_p)
        orig_p    = str(orig_p)
        mask_p    = str(mask_p)

        vol = np.load(inpaint_p)["inpainted"]
        Z = vol.shape[0]

        for s in range(Z):
            entries.append({
                "inpaint": inpaint_p,
                "orig": orig_p,
                "mask": mask_p,
                "slice": s
            })

    return entries


# ---------------------------------------------------------
# Ensure shape (1,H,W) (NO resizing)
# ---------------------------------------------------------
def ensure_channel_first(x):
    x = np.asarray(x)
    x = np.squeeze(x)

    if x.ndim == 2:         # (H,W)
        x = x[None, ...]    # (1,H,W)
    elif x.ndim == 3 and x.shape[0] != 1:
        x = np.moveaxis(x, -1, 0)
        x = x[:1]

    return x.astype(np.float32)


# ---------------------------------------------------------
# Dataset (NO AUGMENTATION)
# ---------------------------------------------------------
class CTNPZDataset(Dataset):
    def __init__(
        self,
        entries,
        preprocess_fn,
        clip_min=-100,
        clip_max=300,
        force_size=(256, 256),
    ):
        self.entries = entries
        self.preprocess_fn = preprocess_fn

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.force_size = force_size

        if len(self.entries) == 0:
            raise RuntimeError("CTNPZDataset: entries list is empty.")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):

        entry = self.entries[idx]
        s = entry["slice"]

        # -------------------------------------------------
        # Load volumes
        # -------------------------------------------------
        inpaint_vol = np.load(entry["inpaint"])["inpainted"].astype(np.float32)
        orig_vol    = np.load(entry["orig"])["image"].astype(np.float32)
        seg_vol     = np.load(entry["mask"])["mask"].astype(np.float32)

        healthy_slice = inpaint_vol[s]
        target_slice  = orig_vol[s]
        seg_slice     = seg_vol[s]

        # -------------------------------------------------
        # Build masks
        # -------------------------------------------------
        if np.max(seg_slice) >= 2:
            tumor_mask = (seg_slice == 2).astype(np.float32)
            liver_mask = (seg_slice >= 1).astype(np.float32)
        else:
            tumor_mask = (seg_slice > 0).astype(np.float32)
            liver_mask = (seg_slice > 0).astype(np.float32)

        H, W = self.force_size

        # Resize masks only once
        tumor_mask_r = cv2.resize(tumor_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        liver_mask_r = cv2.resize(liver_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        tumor_mask_r = ensure_channel_first(tumor_mask_r)
        liver_mask_r = ensure_channel_first(liver_mask_r)

        # -------------------------------------------------
        # Preprocess CT slices into [-1,1]
        # -------------------------------------------------
        healthy_pp = self.preprocess_fn(
            healthy_slice[None, ...], self.clip_min, self.clip_max, self.force_size
        )[0]

        target_pp = self.preprocess_fn(
            target_slice[None, ...], self.clip_min, self.clip_max, self.force_size
        )[0]

        healthy_pp = ensure_channel_first(healthy_pp)
        target_pp  = ensure_channel_first(target_pp)

        # -------------------------------------------------
        # Convert to tensors
        # -------------------------------------------------
        healthy_t    = torch.from_numpy(healthy_pp)
        target_t     = torch.from_numpy(target_pp)
        tumor_mask_t = torch.from_numpy(tumor_mask_r)
        liver_mask_t = torch.from_numpy(liver_mask_r)

        return healthy_t, target_t, tumor_mask_t, liver_mask_t
