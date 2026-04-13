"""
Improved medical inpainting:
 - Only modify tumour region
 - Replace tumour region using real healthy-liver pixels
 - No gaussian smoothing
 - No synthetic noise
 - No cv2.inpaint
"""

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm


def inpaint_using_liver_patches(ct_slice, liver_mask, tumor_mask):
    """Replace tumour pixels by random healthy-liver pixel samples."""

    out = ct_slice.copy()

    tumor_mask_bin = (tumor_mask == 1)
    liver_mask_bin = (liver_mask == 1)

    # healthy liver = liver - tumour
    healthy_mask = liver_mask_bin & (~tumor_mask_bin)
    healthy_pixels = ct_slice[healthy_mask]

    # Fallback: if no healthy region exists, use slice mean
    if healthy_pixels.size == 0:
        mean_val = float(ct_slice.mean())
        out[tumor_mask_bin] = mean_val
        return out

    # Sample real healthy intensities for each tumour pixel
    num_tumor = tumor_mask_bin.sum()
    rand_idx = np.random.randint(0, healthy_pixels.shape[0], size=num_tumor)
    out[tumor_mask_bin] = healthy_pixels[rand_idx]

    return out


def inpaint_volume(ct_vol, liver_vol, tumor_vol):
    """Process a 3D CT volume slice-by-slice."""

    Z, H, W = ct_vol.shape
    out = np.zeros_like(ct_vol)

    for z in range(Z):
        out[z] = inpaint_using_liver_patches(
            ct_vol[z], liver_vol[z], tumor_vol[z]
        )

    return out


def inpaint_dataset(
    processed_ct_dir="processed_volumes",
    processed_mask_dir="processed_masks",
    out_dir="inpainted_volumes",
    force_binary=False
):
    """Read dataset, separate masks, call inpainting."""

    CT_DIR = Path(processed_ct_dir)
    MASK_DIR = Path(processed_mask_dir)
    OUT = Path(out_dir)
    OUT.mkdir(exist_ok=True, parents=True)

    ct_files = sorted(CT_DIR.glob("*.npz"))
    mask_files = sorted(MASK_DIR.glob("*.npz"))

    print(f"[INFO] Running inpainting on {len(ct_files)} volumes...\n")

    for ct_path, mask_path in tqdm(zip(ct_files, mask_files), total=len(ct_files), desc="Inpainting"):

        ct = np.load(ct_path)["image"].astype(np.float32)
        seg = np.load(mask_path)["mask"].astype(np.float32)

        # seg: 0 = background, 1 = liver, 2 = tumor

        if not force_binary:
            # tumor_mask becomes empty if no tumor exists
            tumor_mask = (seg == 2).astype(np.float32)

            # liver_mask includes only healthy liver
            liver_mask = (seg == 1).astype(np.float32)

        else:
            # force_binary means you ignore class 2 entirely
            # treat any >0 as liver, and tumor is always empty
            tumor_mask = np.zeros_like(seg, dtype=np.float32)
            liver_mask = (seg > 0).astype(np.float32)


        # Inpaint
        inpainted = inpaint_volume(ct, liver_mask, tumor_mask)

        # Save
        out_name = ct_path.stem + "_inpainted.npz"
        np.savez_compressed(OUT / out_name, inpainted=inpainted)

    print(f"\n✅ Inpainting completed. Saved to {OUT}\n")
