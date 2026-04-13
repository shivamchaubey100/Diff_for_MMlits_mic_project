"""
inpainting.py

Tumour-region inpainting using LaMa (Large Mask Inpainting).

Pairing fix
-----------
CT files are named  volume-N.npz  and mask files  volume-N_mask.npz.
Paired explicitly by numeric suffix so volume-10 always matches
volume-10_mask regardless of lexicographic sort order.

Install
-------
    pip install onnxruntime-gpu   # GPU (preferred)
    pip install onnxruntime       # CPU fallback
    pip install huggingface_hub   # one-time weight download (~208 MB)
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

# ── onnxruntime ───────────────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    print(
        "[inpaint] WARNING: onnxruntime not installed — falling back to "
        "local-annular pixel sampling.\n"
        "          Install:  pip install onnxruntime-gpu   (GPU)\n"
        "                or  pip install onnxruntime        (CPU)"
    )

_LAMA_SIZE = 512


# ============================================================
# Pairing helpers
# ============================================================

def _extract_number(path: Path) -> int:
    """Return the trailing integer from the first '_'-free token.
    E.g. 'volume-3' → 3,  'volume-3_mask' → 3,  'volume-10' → 10."""
    base   = path.stem.split("_")[0]
    digits = "".join(filter(str.isdigit, base))
    return int(digits) if digits else -1


def _pair_ct_and_masks(
    ct_dir:   Path,
    mask_dir: Path,
) -> List[Tuple[Path, Path]]:
    """
    Match CT .npz files (volume-N.npz) with mask .npz files
    (volume-N_mask.npz) by their numeric suffix N.

    Returns a list of (ct_path, mask_path) sorted by N.
    """
    ct_by_n:   Dict[int, Path] = {_extract_number(p): p for p in ct_dir.glob("*.npz")}
    mask_by_n: Dict[int, Path] = {_extract_number(p): p for p in mask_dir.glob("*.npz")}

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
# LaMa ONNX weight download + session
# ============================================================

def _get_lama_onnx_path() -> Path:
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(
            repo_id  = "Carve/LaMa-ONNX",
            filename = "lama_fp32.onnx",
        ))
    except Exception as e:
        raise RuntimeError(
            f"Could not download LaMa ONNX weights: {e}\n"
            "Install huggingface_hub:  pip install huggingface_hub"
        ) from e


_lama_session: Optional["ort.InferenceSession"] = None


def _get_lama_session() -> "ort.InferenceSession":
    global _lama_session
    if _lama_session is None:
        onnx_path = _get_lama_onnx_path()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        _lama_session = ort.InferenceSession(str(onnx_path), providers=providers)
        ep = _lama_session.get_providers()[0]
        print(f"[inpaint] LaMa ONNX loaded  (provider: {ep})")
    return _lama_session


# ============================================================
# LaMa inference
# ============================================================

def _to_lama_input(
    ct_slice:   np.ndarray,
    tumor_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    H, W      = ct_slice.shape
    orig_size = (H, W)

    if ct_slice.max() > 1.5:
        norm = (np.clip(ct_slice, -150.0, 350.0) + 150.0) / 500.0
    else:
        norm = np.clip(ct_slice, 0.0, 1.0)

    img_r    = cv2.resize(norm, (_LAMA_SIZE, _LAMA_SIZE), interpolation=cv2.INTER_LINEAR)
    bin_mask = (tumor_mask > 0).astype(np.uint8)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated  = cv2.dilate(bin_mask, kernel, iterations=1)
    mask_r   = cv2.resize(dilated.astype(np.float32),
                          (_LAMA_SIZE, _LAMA_SIZE), interpolation=cv2.INTER_NEAREST)
    mask_r   = (mask_r > 0.5).astype(np.float32)

    img_rgb = np.stack([img_r, img_r, img_r], axis=0)[None].astype(np.float32)
    mask_b  = mask_r[None, None].astype(np.float32)
    return img_rgb, mask_b, orig_size


def inpaint_slice_lama(
    ct_slice:   np.ndarray,
    tumor_mask: np.ndarray,
) -> np.ndarray:
    session                = _get_lama_session()
    img_b, mask_b, (H, W) = _to_lama_input(ct_slice, tumor_mask)

    inp_name  = session.get_inputs()[0].name
    mask_name = session.get_inputs()[1].name
    result    = session.run(None, {inp_name: img_b, mask_name: mask_b})[0]

    result_hw = cv2.resize(result[0, 0], (W, H), interpolation=cv2.INTER_LINEAR)

    if ct_slice.max() > 1.5:
        result_hw = result_hw * 500.0 - 150.0

    out = ct_slice.copy()
    out[tumor_mask > 0] = result_hw[tumor_mask > 0]
    return out


# ============================================================
# Fallback: local-annular pixel sampling
# ============================================================

def inpaint_slice_local_sampling(
    ct_slice:      np.ndarray,
    liver_mask:    np.ndarray,
    tumor_mask:    np.ndarray,
    ring_dilation: int = 12,
) -> np.ndarray:
    out       = ct_slice.copy()
    tumor_bin = tumor_mask.astype(bool)
    liver_bin = liver_mask.astype(bool)

    kernel    = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_dilation * 2 + 1, ring_dilation * 2 + 1)
    )
    dilated   = cv2.dilate(tumor_mask.astype(np.uint8), kernel).astype(bool)
    ring      = dilated & liver_bin & ~tumor_bin
    source_px = (ct_slice[ring] if ring.sum() > 0
                 else ct_slice[liver_bin & ~tumor_bin])

    if source_px.size == 0:
        out[tumor_bin] = float(ct_slice.mean())
        return out

    rand_idx = np.random.randint(0, source_px.shape[0], size=int(tumor_bin.sum()))
    out[tumor_bin] = source_px[rand_idx]
    return out


# ============================================================
# Unified slice dispatcher
# ============================================================

def inpaint_slice(
    ct_slice:   np.ndarray,
    liver_mask: np.ndarray,
    tumor_mask: np.ndarray,
    use_lama:   bool = True,
) -> np.ndarray:
    if not tumor_mask.astype(bool).any():
        return ct_slice.copy()
    if use_lama and _ORT_AVAILABLE:
        try:
            return inpaint_slice_lama(ct_slice, tumor_mask)
        except Exception as e:
            print(f"[inpaint] LaMa slice failed ({e}), falling back.")
    return inpaint_slice_local_sampling(ct_slice, liver_mask, tumor_mask)


# ============================================================
# Volume-level processing
# ============================================================

def inpaint_volume(
    ct_vol:    np.ndarray,
    liver_vol: np.ndarray,
    tumor_vol: np.ndarray,
    use_lama:  bool = True,
) -> np.ndarray:
    Z   = ct_vol.shape[0]
    out = np.zeros_like(ct_vol)
    for z in range(Z):
        out[z] = inpaint_slice(
            ct_vol[z], liver_vol[z], tumor_vol[z], use_lama=use_lama
        )
    return out


# ============================================================
# Visualisation
# ============================================================

def _visualise_inpainting(
    ct_proc:   np.ndarray,   # (Z, H, W)
    seg_proc:  np.ndarray,   # (Z, H, W) labels 0/1/2 — same Z as ct_proc
    inpainted: np.ndarray,   # (Z, H, W)
    stem:      str,
    vis_dir:   Path,
    method:    str,
) -> None:
    """
    Save a 5-panel PNG at the most tumour-rich slice:

        Panel 1 — CT input          original slice before any inpainting
        Panel 2 — Tumour mask       binary mask of the inpainted region
        Panel 3 — Inpainted CT      full slice after inpainting (greyscale)
        Panel 4 — Diff map          absolute per-pixel change (hot colourmap)
        Panel 5 — Context overlay   inpainted CT rendered as RGB with the
                                    inpainted region tinted cyan so you can
                                    see exactly where the edit sits within
                                    the full anatomy; amber border marks
                                    the region edge
    """
    vis_dir.mkdir(parents=True, exist_ok=True)

    Z = ct_proc.shape[0]

    tumour_area = (seg_proc == 2).sum(axis=(1, 2))
    if tumour_area.max() == 0:
        return
    idx = min(int(np.argmax(tumour_area)), Z - 1)   # safety clamp

    ct_sl     = ct_proc[idx]
    tumour_sl = (seg_proc[idx] == 2).astype(np.float32)
    inp_sl    = inpainted[idx]
    diff_sl   = np.abs(inp_sl - ct_sl)

    # ── Panel 5: context overlay ──────────────────────────────────────────────
    # Normalise the inpainted slice to [0,1] using the same window as ct_sl
    # so brightness is directly comparable to panel 1.
    vmin     = ct_sl.min()
    vmax     = ct_sl.max() if ct_sl.max() > ct_sl.min() else ct_sl.min() + 1e-6
    inp_norm = np.clip((inp_sl - vmin) / (vmax - vmin), 0.0, 1.0)
    overlay  = np.stack([inp_norm, inp_norm, inp_norm], axis=-1).copy()

    # Cyan tint over the inpainted region: dim red, lift green + blue
    region = tumour_sl.astype(bool)
    overlay[region, 0] = np.clip(overlay[region, 0] * 0.55,         0.0, 1.0)
    overlay[region, 1] = np.clip(overlay[region, 1] * 0.85 + 0.30,  0.0, 1.0)
    overlay[region, 2] = np.clip(overlay[region, 2] * 0.85 + 0.45,  0.0, 1.0)

    # Amber 1-px boundary around the inpainted region edge
    dilated  = cv2.dilate(tumour_sl.astype(np.uint8), np.ones((3, 3), np.uint8))
    boundary = (dilated - tumour_sl.astype(np.uint8)).astype(bool)
    overlay[boundary, 0] = 1.0
    overlay[boundary, 1] = 0.8
    overlay[boundary, 2] = 0.0

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    fig.patch.set_facecolor("#111827")

    axs[0].imshow(ct_sl,     cmap="gray"); axs[0].set_title("CT input",                            color="#e5e7eb", fontsize=10)
    axs[1].imshow(tumour_sl, cmap="Reds"); axs[1].set_title("Tumour mask",                         color="#e5e7eb", fontsize=10)
    axs[2].imshow(inp_sl,    cmap="gray"); axs[2].set_title("Inpainted CT",                        color="#e5e7eb", fontsize=10)
    axs[3].imshow(diff_sl,   cmap="hot");  axs[3].set_title("Diff (changed region)",               color="#e5e7eb", fontsize=10)
    axs[4].imshow(overlay);               axs[4].set_title("Overlay (cyan=inpainted, amber=edge)", color="#e5e7eb", fontsize=10)

    for ax in axs:
        ax.axis("off")

    fig.suptitle(f"{stem}  |  slice {idx}  |  method: {method}",
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
    use_lama:           bool          = True,
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
    use_lama           : use LaMa ONNX (True) or annular-sampling fallback
    vis_dir            : if set, save visualisation PNGs here
    n_vis              : number of volumes to visualise (all volumes processed)
    """
    CT_DIR   = Path(processed_ct_dir)
    MASK_DIR = Path(processed_mask_dir)
    OUT      = Path(out_dir);  OUT.mkdir(parents=True, exist_ok=True)
    VIS_DIR  = Path(vis_dir) if vis_dir else None

    pairs = _pair_ct_and_masks(CT_DIR, MASK_DIR)

    method = ("LaMa-ONNX" if (use_lama and _ORT_AVAILABLE)
              else "local-annular-sampling")
    print(f"[inpaint] Method : {method}")
    print(f"[inpaint] Volumes: {len(pairs)}")

    for i, (ct_path, mask_path) in enumerate(
        tqdm(pairs, total=len(pairs), desc="Inpainting")
    ):
        ct  = np.load(ct_path)["image"].astype(np.float32)
        seg = np.load(mask_path)["mask"].astype(np.float32)

        if ct.shape[0] != seg.shape[0]:
            print(
                f"[inpaint] WARNING: Z-depth mismatch for {ct_path.name} "
                f"(ct={ct.shape[0]}, seg={seg.shape[0]}) — skipping."
            )
            continue

        if force_binary:
            tumor_mask = np.zeros_like(seg, dtype=np.float32)
            liver_mask = (seg > 0).astype(np.float32)
        else:
            tumor_mask = (seg == 2).astype(np.float32)
            liver_mask = (seg == 1).astype(np.float32)

        inpainted = inpaint_volume(ct, liver_mask, tumor_mask, use_lama=use_lama)

        np.savez_compressed(
            OUT / (ct_path.stem + "_inpainted.npz"), inpainted=inpainted
        )

        # Save PNG for first n_vis volumes; all volumes are still processed
        if VIS_DIR is not None and i < n_vis:
            _visualise_inpainting(ct, seg, inpainted, ct_path.stem, VIS_DIR, method)

    print(f"[inpaint] Done → {OUT}")
    if VIS_DIR:
        print(f"[inpaint] Visualisations → {VIS_DIR}")