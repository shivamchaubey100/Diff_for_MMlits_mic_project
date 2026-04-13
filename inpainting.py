"""
inpainting.py

Tumour-region inpainting using LaMa (Large Mask Inpainting).

HOW LAMA IS LOADED HERE
-----------------------
We use the ONNX export of LaMa from  Carve/LaMa-ONNX  on HuggingFace.
This completely avoids the `simple-lama-inpainting` package and its
pinned `Pillow < 10` constraint that breaks on Python 3.12+/3.14.

Requirements (all Python-version-agnostic):
    pip install onnxruntime-gpu   # GPU inference  (preferred)
    pip install onnxruntime       # CPU-only fallback
    pip install huggingface_hub   # one-time weight download (~208 MB)

The ONNX model expects fixed 512×512 input. We tile/resize as needed.

FALLBACK
--------
If onnxruntime is not installed the code falls back to a local-annular
pixel-sampling method (samples healthy liver pixels from a ring
immediately around the tumour — far better than global random sampling).

Labels:  0 = background,  1 = liver,  2 = tumour
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ── Try to import onnxruntime ─────────────────────────────────────────────────
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    print(
        "[inpaint] WARNING: onnxruntime not installed — falling back to "
        "local-annular pixel sampling.\n"
        "          Install with:  pip install onnxruntime-gpu   (GPU)\n"
        "                     or  pip install onnxruntime        (CPU)"
    )

# LaMa ONNX fixed input resolution
_LAMA_SIZE = 512


# ============================================================
# Weight download (one-time, cached in ~/.cache/huggingface)
# ============================================================

def _get_lama_onnx_path() -> Path:
    """
    Download lama_fp32.onnx from Carve/LaMa-ONNX on HuggingFace Hub
    (only on first call; cached locally afterwards).
    Returns local path to the .onnx file.
    """
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id  = "Carve/LaMa-ONNX",
            filename = "lama_fp32.onnx",
        )
        return Path(path)
    except Exception as e:
        raise RuntimeError(
            f"Could not download LaMa ONNX weights: {e}\n"
            "Make sure huggingface_hub is installed:\n"
            "    pip install huggingface_hub\n"
            "and you have an internet connection on first run."
        ) from e


# ============================================================
# LaMa ONNX session (singleton)
# ============================================================

_lama_session: Optional["ort.InferenceSession"] = None


def _get_lama_session() -> "ort.InferenceSession":
    global _lama_session
    if _lama_session is None:
        onnx_path = _get_lama_onnx_path()
        # Prefer CUDA EP if available, fall back to CPU
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
# LaMa inference helpers
# ============================================================

def _to_lama_input(
    ct_slice:   np.ndarray,   # (H, W) float32 — raw HU or [0,1]
    tumor_mask: np.ndarray,   # (H, W) binary
) -> tuple[np.ndarray, np.ndarray, tuple[int,int]]:
    """
    Prepare a CT slice + tumour mask for LaMa inference.

    Returns
    -------
    img_512   : float32  (1, 3, 512, 512)  normalised to [0, 1]
    mask_512  : float32  (1, 1, 512, 512)  binary, dilated
    orig_size : (H, W)   of the input (for resizing result back)
    """
    H, W       = ct_slice.shape
    orig_size  = (H, W)

    # --- Normalise CT to [0, 1] ---
    if ct_slice.max() > 1.5:
        clip_min, clip_max = -150.0, 350.0
        norm = np.clip(ct_slice, clip_min, clip_max)
        norm = (norm - clip_min) / (clip_max - clip_min)
    else:
        norm = np.clip(ct_slice, 0.0, 1.0)

    # --- Resize to 512×512 ---
    img_r  = cv2.resize(norm, (_LAMA_SIZE, _LAMA_SIZE),
                        interpolation=cv2.INTER_LINEAR)

    # --- Dilate mask slightly to cover tumour boundaries ---
    bin_mask = (tumor_mask > 0).astype(np.uint8)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated  = cv2.dilate(bin_mask, kernel, iterations=1)
    mask_r   = cv2.resize(dilated.astype(np.float32),
                          (_LAMA_SIZE, _LAMA_SIZE),
                          interpolation=cv2.INTER_NEAREST)
    mask_r   = (mask_r > 0.5).astype(np.float32)

    # LaMa expects RGB — replicate grayscale to 3 channels
    img_rgb  = np.stack([img_r, img_r, img_r], axis=0)   # (3, 512, 512)
    img_b    = img_rgb[None].astype(np.float32)           # (1, 3, 512, 512)
    mask_b   = mask_r[None, None].astype(np.float32)      # (1, 1, 512, 512)

    return img_b, mask_b, orig_size


def inpaint_slice_lama(
    ct_slice:   np.ndarray,
    tumor_mask: np.ndarray,
) -> np.ndarray:
    """
    Inpaint the tumour region of one CT slice using LaMa ONNX.

    Returns float32 (H, W) in the same value range as the input.
    """
    session                = _get_lama_session()
    img_b, mask_b, (H, W) = _to_lama_input(ct_slice, tumor_mask)

    # LaMa input node names
    inp_name  = session.get_inputs()[0].name   # image
    mask_name = session.get_inputs()[1].name   # mask

    result = session.run(None, {inp_name: img_b, mask_name: mask_b})[0]
    # result: (1, 3, 512, 512) float32 in [0, 1]

    # Take the first (R) channel — all three are identical for grayscale input
    result_hw = result[0, 0]   # (512, 512)

    # Resize back to original resolution
    result_hw = cv2.resize(result_hw, (W, H), interpolation=cv2.INTER_LINEAR)

    # Restore original value range if HU input
    if ct_slice.max() > 1.5:
        clip_min, clip_max = -150.0, 350.0
        result_hw = result_hw * (clip_max - clip_min) + clip_min

    # Only overwrite the tumour region; leave everything else untouched
    out  = ct_slice.copy()
    mask = (tumor_mask > 0)
    out[mask] = result_hw[mask]
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
    """
    Improved pixel-sampling fallback.

    Samples healthy liver pixels from a LOCAL ANNULAR RING around the
    tumour (the immediately adjacent tissue) instead of random liver
    pixels from anywhere in the slice. Gives much better spatial
    coherence than the original global-random approach.
    """
    out       = ct_slice.copy()
    tumor_bin = tumor_mask.astype(bool)
    liver_bin = liver_mask.astype(bool)

    kernel  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_dilation * 2 + 1, ring_dilation * 2 + 1)
    )
    dilated = cv2.dilate(tumor_mask.astype(np.uint8), kernel).astype(bool)
    ring    = dilated & liver_bin & ~tumor_bin

    source_px = ct_slice[ring] if ring.sum() > 0 \
                else ct_slice[liver_bin & ~tumor_bin]
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
    """Inpaint one 2-D CT slice. Uses LaMa ONNX if available."""
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
    """Inpaint a 3-D CT volume (Z, H, W) slice-by-slice."""
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
    ct_proc:   np.ndarray,
    seg_proc:  np.ndarray,
    inpainted: np.ndarray,
    stem:      str,
    vis_dir:   Path,
    method:    str,
) -> None:
    """Save a 4-panel PNG: CT input | tumour mask | inpainted | diff map."""
    vis_dir.mkdir(parents=True, exist_ok=True)

    tumour_area = (seg_proc == 2).sum(axis=(1, 2))
    if tumour_area.max() == 0:
        return
    idx = int(np.argmax(tumour_area))

    ct_sl     = ct_proc[idx]
    tumour_sl = (seg_proc[idx] == 2).astype(np.float32)
    inp_sl    = inpainted[idx]
    diff_sl   = np.abs(inp_sl - ct_sl)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#111827")

    axs[0].imshow(ct_sl,     cmap="gray"); axs[0].set_title("CT input",              color="#e5e7eb", fontsize=10)
    axs[1].imshow(tumour_sl, cmap="Reds"); axs[1].set_title("Tumour mask",           color="#e5e7eb", fontsize=10)
    axs[2].imshow(inp_sl,    cmap="gray"); axs[2].set_title("Inpainted CT",          color="#e5e7eb", fontsize=10)
    axs[3].imshow(diff_sl,   cmap="hot");  axs[3].set_title("Diff (changed region)", color="#e5e7eb", fontsize=10)
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

    Parameters
    ----------
    processed_ct_dir   : folder with volume .npz  (key='image')
    processed_mask_dir : folder with mask   .npz  (key='mask')
    out_dir            : output folder for inpainted .npz  (key='inpainted')
    force_binary       : treat all >0 as liver; no tumour inpainting
    use_lama           : use LaMa ONNX (True) or annular-sampling fallback
    vis_dir            : if set, save visualisation PNGs here
    n_vis              : number of volumes to visualise
    """
    CT_DIR   = Path(processed_ct_dir)
    MASK_DIR = Path(processed_mask_dir)
    OUT      = Path(out_dir);  OUT.mkdir(parents=True, exist_ok=True)
    VIS_DIR  = Path(vis_dir) if vis_dir else None

    ct_files   = sorted(CT_DIR.glob("*.npz"))
    mask_files = sorted(MASK_DIR.glob("*.npz"))

    if not ct_files:
        raise FileNotFoundError(f"No .npz files in {processed_ct_dir}")
    if len(ct_files) != len(mask_files):
        print(f"[inpaint] WARNING: {len(ct_files)} CT vs "
              f"{len(mask_files)} mask files.")

    method = ("LaMa-ONNX" if (use_lama and _ORT_AVAILABLE)
              else "local-annular-sampling")
    print(f"[inpaint] Method : {method}")
    print(f"[inpaint] Volumes: {len(ct_files)}")

    for i, (ct_path, mask_path) in enumerate(
        tqdm(zip(ct_files, mask_files), total=len(ct_files), desc="Inpainting")
    ):
        ct  = np.load(ct_path)["image"].astype(np.float32)
        seg = np.load(mask_path)["mask"].astype(np.float32)

        if force_binary:
            tumor_mask = np.zeros_like(seg, dtype=np.float32)
            liver_mask = (seg > 0).astype(np.float32)
        else:
            tumor_mask = (seg == 2).astype(np.float32)
            liver_mask = (seg == 1).astype(np.float32)

        inpainted = inpaint_volume(ct, liver_mask, tumor_mask,
                                   use_lama=use_lama)

        np.savez_compressed(
            OUT / (ct_path.stem + "_inpainted.npz"), inpainted=inpainted
        )

        if VIS_DIR is not None and i < n_vis:
            _visualise_inpainting(
                ct, seg, inpainted, ct_path.stem, VIS_DIR, method
            )

    print(f"[inpaint] Done → {OUT}")
    if VIS_DIR:
        print(f"[inpaint] Visualisations → {VIS_DIR}")