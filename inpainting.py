"""
inpainting.py

Tumour-region inpainting using local-annular pixel sampling.

Strategy
--------
For each tumour slice:
  1. Build an annular ring of healthy liver pixels immediately surrounding
     the tumour (dilated tumour mask ∩ liver mask, excluding tumour itself).
     These are the most spatially relevant pixels — same organ, same depth,
     directly adjacent tissue.
  2. Randomly sample from that ring to fill the tumour region.
  3. Apply Gaussian smoothing only at the boundary (feathered blend) so the
     filled region transitions smoothly into the surrounding liver rather
     than having a hard edge.
  4. Fall back to the full healthy-liver pixel pool if the ring is too small,
     and to slice mean if no liver pixels exist at all.

Why this works well
-------------------
- Samples from spatially LOCAL tissue, so intensities match the surrounding
  parenchyma rather than a random region of the organ.
- Boundary feathering eliminates the hard edge artefact.
- No model weights, no internet, no GPU dependency.
- Deterministic (seeded) per slice for reproducibility.

Pairing
-------
volume-N.npz  ↔  volume-N_mask.npz   matched by numeric suffix.

Labels:  0 = background,  1 = liver,  2 = tumour
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# ============================================================
# Pairing helpers
# ============================================================

def _extract_number(path: Path) -> int:
    """
    Return the leading integer from the volume stem.
    E.g. 'volume-3' → 3,  'volume-10_mask' → 10.
    """
    base   = path.stem.split("_")[0]
    digits = "".join(filter(str.isdigit, base))
    return int(digits) if digits else -1


def _pair_ct_and_masks(
    ct_dir:   Path,
    mask_dir: Path,
) -> List[Tuple[Path, Path]]:
    """
    Match CT .npz (volume-N.npz) with mask .npz (volume-N_mask.npz)
    by numeric suffix.  Returns list of (ct_path, mask_path) sorted by N.
    """
    ct_by_n:   Dict[int, Path] = {_extract_number(p): p
                                   for p in ct_dir.glob("*.npz")}
    mask_by_n: Dict[int, Path] = {_extract_number(p): p
                                   for p in mask_dir.glob("*.npz")}

    common      = sorted(set(ct_by_n) & set(mask_by_n))
    unmatched_c = sorted(set(ct_by_n) - set(mask_by_n))
    unmatched_m = sorted(set(mask_by_n) - set(ct_by_n))

    if unmatched_c:
        print(f"[inpaint] WARNING: CT files with no matching mask: "
              f"{[ct_by_n[k].name for k in unmatched_c]}")
    if unmatched_m:
        print(f"[inpaint] WARNING: mask files with no matching CT: "
              f"{[mask_by_n[k].name for k in unmatched_m]}")
    if not common:
        raise FileNotFoundError(
            f"No matched CT/mask pairs found.\n"
            f"  CT dir  : {ct_dir}\n"
            f"  Mask dir: {mask_dir}\n"
            "Check filenames follow volume-N.npz / volume-N_mask.npz."
        )

    pairs = [(ct_by_n[n], mask_by_n[n]) for n in common]
    print(f"[inpaint] Matched {len(pairs)} CT/mask pairs.")
    return pairs


# ============================================================
# Core inpainting — random ring sampling + light smoothing
# ============================================================

def inpaint_slice(
    ct_slice:     np.ndarray,   # (H, W) float32
    liver_mask:   np.ndarray,   # (H, W) binary 0/1 — healthy liver (label 1)
    tumor_mask:   np.ndarray,   # (H, W) binary 0/1 — tumour (label 2)
    smooth_sigma: float = 0.4,  # sub-pixel Gaussian blur inside tumour only
) -> np.ndarray:
    """
    Inpaint the tumour region by sampling from the full healthy liver.

    Strategy
    --------
    1.  Collect ALL healthy liver pixels from the slice (label==1, not tumour).
        Sample from the entire liver — not just a ring — so every tumour
        pixel is guaranteed to be filled regardless of shape or position.

    2.  Randomly sample from the healthy liver pool to fill every tumour
        pixel completely, including the boundary shell.

    3.  Very light Gaussian smoothing (sigma=0.4) applied ONLY inside the
        tumour region — softens point discontinuities while preserving the
        natural grain and heterogeneity of liver parenchyma.

    4.  Boundary shell (outermost 1-px ring of tumour) blended 70% filled +
        30% healthy liver mean so the edge transitions smoothly — no tumour
        HU values are reintroduced.
    """
    tumor_bin = tumor_mask.astype(bool)
    if not tumor_bin.any():
        return ct_slice.copy()

    out = ct_slice.copy()
    rng = np.random.default_rng(seed=int(tumor_bin.sum()))

    # ── Step 1: collect healthy liver pixel pool ──────────────────────────────
    # All pixels labelled liver (1) that are not tumour (2).
    # Using the full liver (not a ring) guarantees coverage for every
    # tumour shape and position.
    healthy_bin = liver_mask.astype(bool) & ~tumor_bin
    if healthy_bin.sum() == 0:
        # No liver pixels at all on this slice — use slice mean as last resort
        out[tumor_bin] = float(ct_slice.mean())
        return out.astype(ct_slice.dtype)

    source_px  = ct_slice[healthy_bin]          # 1-D array of healthy HU values
    liver_mean = float(source_px.mean())

    # ── Step 2: random fill of every tumour pixel ─────────────────────────────
    n_tumor  = int(tumor_bin.sum())
    rand_idx = rng.integers(0, len(source_px), size=n_tumor)
    out[tumor_bin] = source_px[rand_idx]

    # ── Step 3: light smoothing inside tumour only ────────────────────────────
    if smooth_sigma > 0:
        blurred        = cv2.GaussianBlur(out, (0, 0), smooth_sigma)
        out[tumor_bin] = blurred[tumor_bin]

    # ── Step 4: boundary shell blend toward liver mean ────────────────────────
    erode_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    interior = cv2.erode(tumor_mask.astype(np.uint8), erode_k).astype(bool)
    shell    = tumor_bin & ~interior
    if shell.any():
        out[shell] = 0.7 * out[shell] + 0.3 * liver_mean

    return out.astype(ct_slice.dtype)


# ============================================================
# Volume-level processing
# ============================================================

def inpaint_volume(
    ct_vol:    np.ndarray,   # (Z, H, W)
    liver_vol: np.ndarray,   # (Z, H, W)  label=1 healthy liver
    tumor_vol: np.ndarray,   # (Z, H, W)  label=2 tumour
) -> np.ndarray:
    """Inpaint a 3-D CT volume slice-by-slice."""
    Z   = ct_vol.shape[0]
    out = np.zeros_like(ct_vol)
    for z in range(Z):
        out[z] = inpaint_slice(ct_vol[z], liver_vol[z], tumor_vol[z])
    return out


# ============================================================
# Visualisation
# ============================================================

def _visualise_inpainting(
    ct_proc:   np.ndarray,   # (Z, H, W)
    seg_proc:  np.ndarray,   # (Z, H, W)  labels 0/1/2
    inpainted: np.ndarray,   # (Z, H, W)
    stem:      str,
    vis_dir:   Path,
) -> None:
    """
    Save a 4-panel PNG at the most tumour-rich slice:

        Panel 1 — Original CT (with tumour)
        Panel 2 — Tumour mask
        Panel 3 — Inpainted CT (healthy liver) — same brightness window as
                  panel 1 so no blown-out white patches
        Panel 4 — Diff map (hot colourmap, original vs inpainted)
    """
    vis_dir.mkdir(parents=True, exist_ok=True)

    Z = ct_proc.shape[0]
    tumour_area = (seg_proc == 2).sum(axis=(1, 2))
    if tumour_area.max() == 0:
        return

    idx = min(int(np.argmax(tumour_area)), Z - 1)

    ct_sl     = ct_proc[idx]
    tumour_sl = (seg_proc[idx] == 2).astype(np.float32)
    inp_sl    = inpainted[idx]
    diff_sl   = np.abs(inp_sl - ct_sl)

    # Shared display window derived from original CT slice
    disp_vmin = ct_sl.min()
    disp_vmax = ct_sl.max() if ct_sl.max() > ct_sl.min() else ct_sl.min() + 1e-6

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#111827")

    axs[0].imshow(ct_sl,     cmap="gray", vmin=disp_vmin, vmax=disp_vmax)
    axs[0].set_title("Original CT (with tumour)",    color="#e5e7eb", fontsize=10)
    axs[1].imshow(tumour_sl, cmap="Reds")
    axs[1].set_title("Tumour mask",                  color="#e5e7eb", fontsize=10)
    axs[2].imshow(inp_sl,    cmap="gray", vmin=disp_vmin, vmax=disp_vmax)
    axs[2].set_title("Inpainted CT (healthy liver)", color="#e5e7eb", fontsize=10)
    axs[3].imshow(diff_sl,   cmap="hot")
    axs[3].set_title("Diff (original vs inpainted)", color="#e5e7eb", fontsize=10)

    for ax in axs:
        ax.axis("off")

    fig.suptitle(f"{stem}  |  slice {idx}  |  method: local-annular-sampling",
                 color="#9ca3af", fontsize=9)
    plt.tight_layout(pad=0.5)
    plt.savefig(vis_dir / f"{stem}_inpaint.png", dpi=130,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# ============================================================
# Dataset-level entry point
# ============================================================

def inpaint_dataset(
    processed_ct_dir:   str,
    processed_mask_dir: str,
    out_dir:            str,
    force_binary:       bool          = False,
    vis_dir:            Optional[str] = None,
    n_vis:              int           = 5,
) -> None:
    """
    Inpaint tumour regions across all preprocessed CT volumes.

    CT and mask files are paired by numeric suffix (volume-N.npz ↔
    volume-N_mask.npz) so multi-digit indices never mismatch.

    Parameters
    ----------
    processed_ct_dir   : folder with volume .npz  (key='image')
    processed_mask_dir : folder with mask   .npz  (key='mask')
    out_dir            : output folder for inpainted .npz  (key='inpainted')
    force_binary       : treat all >0 as liver, skip tumour inpainting
    vis_dir            : if set, save visualisation PNGs here
    n_vis              : number of volumes to visualise (all 28 still processed)
    """
    CT_DIR   = Path(processed_ct_dir)
    MASK_DIR = Path(processed_mask_dir)
    OUT      = Path(out_dir);  OUT.mkdir(parents=True, exist_ok=True)
    VIS_DIR  = Path(vis_dir) if vis_dir else None

    pairs = _pair_ct_and_masks(CT_DIR, MASK_DIR)
    print(f"[inpaint] Method : local-annular-sampling")
    print(f"[inpaint] Volumes: {len(pairs)}")

    for i, (ct_path, mask_path) in enumerate(
        tqdm(pairs, total=len(pairs), desc="Inpainting")
    ):
        ct  = np.load(ct_path)["image"].astype(np.float32)
        seg = np.load(mask_path)["mask"].astype(np.float32)

        if ct.shape[0] != seg.shape[0]:
            print(f"[inpaint] WARNING: Z-depth mismatch for {ct_path.name} "
                  f"(ct={ct.shape[0]}, seg={seg.shape[0]}) — skipping.")
            continue

        if force_binary:
            tumor_mask = np.zeros_like(seg, dtype=np.float32)
            liver_mask = (seg > 0).astype(np.float32)
        else:
            tumor_mask = (seg == 2).astype(np.float32)
            liver_mask = (seg == 1).astype(np.float32)

        inpainted = inpaint_volume(ct, liver_mask, tumor_mask)

        np.savez_compressed(
            OUT / (ct_path.stem + "_inpainted.npz"), inpainted=inpainted
        )

        if VIS_DIR is not None and i < n_vis:
            _visualise_inpainting(ct, seg, inpainted, ct_path.stem, VIS_DIR)

    print(f"[inpaint] Done → {OUT}")
    if VIS_DIR:
        print(f"[inpaint] Visualisations → {VIS_DIR}")