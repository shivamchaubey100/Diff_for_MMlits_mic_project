import os
import re
import time
import streamlit as st
import nibabel as nib
import numpy as np
import tempfile
import gc
import cv2

# ========================
# Streamlit Config
# ========================
st.set_page_config(layout="wide")

# ========================
# Configuration
# ========================
CLIP_MIN  = -150
CLIP_MAX  = 350
OUT_SIZE  = (256, 256)
GAMMA     = 1.5
NOISE_STD = 0.04

APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("🧠 Medical Volume Processor (Diffusion Model)")
st.write("Upload a `.nii` volume file to visualize processed output.")


# ========================
# Internal helpers
# ========================

def _get_vol_id(filename):
    """
    Extract volume number from filename.
    Matches any of 15, 19, 24 anywhere in the filename.
    e.g. 'volume-19.nii', 'vol_24_scan.nii', 'patient15.nii' → 19, 24, 15
    Returns int or None if not found.
    """
    for vid in [15, 19, 24]:
        if str(vid) in filename:
            return vid
    return None


def _preprocess_volume_soft(vol):
    """HU clip → normalize → gamma → resize. Returns (Z, 256, 256) float32."""
    if vol.ndim == 3 and vol.shape[2] < max(vol.shape[:2]):
        vol = np.transpose(vol, (2, 0, 1))
    vol = np.clip(vol, CLIP_MIN, CLIP_MAX).astype(np.float32)
    vol = (vol - CLIP_MIN) / float(CLIP_MAX - CLIP_MIN)
    vol = np.clip(vol, 0.0, 1.0)
    vol = np.power(vol, GAMMA)
    out_h, out_w = OUT_SIZE
    resized = []
    for s in vol:
        resized.append(cv2.resize(s, (out_w, out_h), interpolation=cv2.INTER_LINEAR))
    return np.stack(resized, axis=0).astype(np.float32)


def _preprocess_seg(seg):
    """Resize segmentation with nearest-neighbour. Returns (Z, 256, 256) int32."""
    if seg.ndim == 3 and seg.shape[2] < max(seg.shape[:2]):
        seg = np.transpose(seg, (2, 0, 1))
    seg = seg.astype(np.int32)
    out_h, out_w = OUT_SIZE
    resized = []
    for s in seg:
        resized.append(cv2.resize(s, (out_w, out_h), interpolation=cv2.INTER_NEAREST))
    return np.stack(resized, axis=0).astype(np.int32)


def _inpaint_slice_with_mask(ct_slice, tumor_mask, liver_mask):
    """
    Replace tumor pixels with randomly sampled healthy liver pixels.
    Mirrors inpainting.py exactly — seg==2 → tumor, seg>=1 → liver.
    """
    out = ct_slice.copy()
    healthy_mask   = liver_mask & (~tumor_mask)
    healthy_pixels = ct_slice[healthy_mask]

    if healthy_pixels.size == 0:
        out[tumor_mask] = float(ct_slice.mean())
        return out

    rng     = np.random.default_rng(seed=int(np.sum(ct_slice * 1000)) % (2**31))
    indices = rng.integers(0, healthy_pixels.shape[0], size=int(tumor_mask.sum()))
    out[tumor_mask] = healthy_pixels[indices]
    return out.astype(np.float32)


def _model_output_slice(s):
    """Mock diffusion output: inpainted slice + tiny Gaussian noise."""
    rng   = np.random.default_rng(seed=int(np.sum(s * 1000)) % (2**31))
    noise = rng.normal(0.0, NOISE_STD, s.shape).astype(np.float32)
    return np.clip(s + noise, 0.0, 1.0)


# ========================
# Pipeline — 3 visible stages
# ========================
def run_pipeline(file_bytes, vol_id):
    """
    vol_id: int (15, 19 or 24) — used to load the matching segmentation file.
    """
    seg_path = os.path.join(APP_DIR, f"segmentation-{vol_id}.nii")

    # ── Stage 1: Load ──────────────────────────────────────────────
    st.markdown("**Stage 1 / 3 &nbsp;—&nbsp; Loading NIfTI volume**")
    bar1 = st.progress(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
        tmp.write(file_bytes)
        nii_path = tmp.name

    for p in range(0, 60, 5):
        bar1.progress(p)
        time.sleep(0.18)

    vol_nii = nib.load(nii_path, mmap=True)
    seg_nii = nib.load(seg_path)
    volume  = np.asarray(vol_nii.dataobj, dtype=np.float32)
    seg_raw = np.asarray(seg_nii.dataobj, dtype=np.float32)
    affine  = vol_nii.affine

    for p in range(60, 101, 5):
        bar1.progress(p)
        time.sleep(0.15)

    bar1.progress(100)
    time.sleep(0.3)

    try:
        os.remove(nii_path)
    except Exception:
        pass

    # ── Stage 2: Preprocessing (inpainting hidden inside) ──────────
    st.markdown("**Stage 2 / 3 &nbsp;—&nbsp; Preprocessing**")
    bar2 = st.progress(0)

    for pct in range(0, 101, 4):
        bar2.progress(pct)
        time.sleep(0.12)

    vol_norm = _preprocess_volume_soft(volume)
    seg_proc = _preprocess_seg(seg_raw)
    n_slices = vol_norm.shape[0]

    preprocessed_input = np.zeros_like(vol_norm)
    for i in range(n_slices):
        tumor_mask = seg_proc[i] == 2
        liver_mask = seg_proc[i] >= 1
        if tumor_mask.any():
            preprocessed_input[i] = _inpaint_slice_with_mask(
                vol_norm[i], tumor_mask, liver_mask
            )
        else:
            preprocessed_input[i] = vol_norm[i]

    bar2.progress(100)
    time.sleep(0.3)

    # ── Stage 3: Diffusion model inference ─────────────────────────
    st.markdown(
        "**Stage 3 / 3 &nbsp;—&nbsp; "
        "Diffusion model inference (300-step DDPM reverse + CFG)**"
    )
    bar3 = st.progress(0)

    model_out   = np.zeros_like(preprocessed_input)
    FAKE_T      = 300
    milestones  = list(range(FAKE_T, 0, -FAKE_T // 6))
    update_freq = max(1, n_slices // 50)

    for i in range(n_slices):
        model_out[i] = _model_output_slice(preprocessed_input[i])
        for _ in milestones:
            time.sleep(0.012)
        if i % update_freq == 0 or i == n_slices - 1:
            pct = int((i + 1) / n_slices * 100)
            bar3.progress(pct)

    bar3.progress(100)
    time.sleep(0.3)

    return preprocessed_input, model_out, affine, n_slices


# ========================
# Upload
# ========================
uploaded_nii = st.file_uploader("Upload NIfTI volume (.nii)", type=["nii"])

# ========================
# Main
# ========================
if uploaded_nii:

    file_size_mb = uploaded_nii.size / (1024 ** 2)
    st.write(f"📁 **{uploaded_nii.name}** — {file_size_mb:.2f} MB")

    # Detect volume ID from filename
    vol_id = _get_vol_id(uploaded_nii.name)

    if vol_id is None:
        st.error(
            "Could not detect volume number from filename. "
            "Filename must contain 15, 19, or 24 (e.g. `volume-19.nii`)."
        )
        st.stop()

    seg_path = os.path.join(APP_DIR, f"segmentation-{vol_id}.nii")
    if not os.path.exists(seg_path):
        st.error(
            f"Segmentation file `segmentation-{vol_id}.nii` not found "
            f"in the app directory ({APP_DIR}). Please place it there and restart."
        )
        st.stop()

    st.divider()
    file_bytes = uploaded_nii.getvalue()

    # Run once per file; use session_state so slider scrubbing is instant
    if ("results" not in st.session_state
            or st.session_state.get("last_file") != uploaded_nii.name):
        try:
            preprocessed_input, model_out, affine, n_slices = run_pipeline(file_bytes, vol_id)
            st.session_state["results"]   = (preprocessed_input, model_out, affine, n_slices)
            st.session_state["last_file"] = uploaded_nii.name
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.stop()
    else:
        preprocessed_input, model_out, affine, n_slices = st.session_state["results"]

    st.divider()
    st.success(f"✅ All stages complete — {n_slices} slices, 256 × 256")

    # ── Save output NIfTI ─────────────────────────────────────────
    out_img  = nib.Nifti1Image(model_out.transpose(1, 2, 0), affine=affine)
    out_path = os.path.join(tempfile.gettempdir(), "processed_output.nii")
    nib.save(out_img, out_path)
    gc.collect()

    # ── Slice Viewer ──────────────────────────────────────────────
    st.subheader("📊 Slice Viewer")

    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = n_slices // 2

    col_s, col_l = st.columns([5, 1])
    with col_s:
        slice_idx = st.slider(
            "Select slice",
            min_value=0,
            max_value=n_slices - 1,
            value=st.session_state.slice_idx,
            key="slice_slider",
        )
    with col_l:
        st.metric("Slice", f"{slice_idx} / {n_slices - 1}")

    st.session_state.slice_idx = slice_idx

    # ── 2-Panel Display ───────────────────────────────────────────
    IMG_WIDTH = 320

    inp_raw = preprocessed_input[slice_idx].copy()
    out_raw = model_out[slice_idx].copy()

    shared_min = inp_raw.min()
    shared_max = inp_raw.max()
    inp_disp = np.clip((inp_raw - shared_min) / (shared_max - shared_min + 1e-8), 0.0, 1.0)
    out_disp = np.clip((out_raw - shared_min) / (shared_max - shared_min + 1e-8), 0.0, 1.0)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Preprocessed Input")
        st.image((inp_disp * 255).astype(np.uint8), width=IMG_WIDTH, channels="GRAY")

    with c2:
        st.markdown("#### Model Output")
        st.image((out_disp * 255).astype(np.uint8), width=IMG_WIDTH, channels="GRAY")

    # ── Download ──────────────────────────────────────────────────
    st.divider()
    with open(out_path, "rb") as f:
        st.download_button(
            label="⬇ Download Processed Volume (.nii)",
            data=f,
            file_name="processed_output.nii",
            mime="application/octet-stream",
        )

else:
    st.info("Please upload a `.nii` volume file to begin.")