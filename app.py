import os
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1024"
import streamlit as st
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import gc

# ========================
# Configuration
# ========================
MODEL_WEIGHTS = "latest_ddpm.pt"  # your trained checkpoint path

st.title("🧠 Medical Volume Processor (Diffusion Model)")
st.write("Upload a `.nii` file (up to ~1 GB) to visualize processed output using your pre-trained model.")

# ========================
# Model definition
# ========================
class EnhancedUNet(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(EnhancedUNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, out_channels, 3, padding=1)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x


# ========================
# Upload + Model Loading
# ========================
uploaded_nii = st.file_uploader("Upload NIfTI file (.nii)", type=["nii"])

if uploaded_nii:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
        tmp.write(uploaded_nii.read())
        nii_path = tmp.name

    st.info("Loading NIfTI file (this may take a few seconds)...")
    nii_img = nib.load(nii_path, mmap=True)
    volume = nii_img.get_fdata(dtype=np.float32)
    st.write(f"Loaded volume shape: {volume.shape}, size: {volume.nbytes / (1024 ** 2):.1f} MB")

    # Normalize input to [0, 1]
    vmin, vmax = np.percentile(volume, (0.5, 99.5))
    vol_norm = np.clip((volume - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # Load model
    st.info("Loading model weights...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EnhancedUNet().to(device)

    if not os.path.exists(MODEL_WEIGHTS):
        st.error(f"❌ Model weights '{MODEL_WEIGHTS}' not found in {os.getcwd()}")
        st.stop()

    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    st.success("✅ Model loaded successfully!")

    # ========================
    # Process volume
    # ========================
    processed = np.zeros_like(vol_norm, dtype=np.float32)
    n_slices = vol_norm.shape[2]
    progress_bar = st.progress(0)

    st.info("Processing volume slices...")
    for i in range(n_slices):
        slice_ = vol_norm[:, :, i]
        inp = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp).cpu().squeeze().numpy().astype(np.float32)

        # --- Normalize output to [0, 1]
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)

        # --- Intensity matching (mean–std matching in normalized domain)
        mu_in, std_in = np.mean(slice_), np.std(slice_)
        mu_out, std_out = np.mean(out), np.std(out)
        if std_out > 1e-6:
            out = (out - mu_out) / std_out * std_in + mu_in
        else:
            out = out * std_in + mu_in

        # --- Clip to [0, 1] to preserve visualization contrast
        out = np.clip(out, 0, 1)

        processed[:, :, i] = out.astype(np.float32)

        if i % 5 == 0:
            progress_bar.progress((i + 1) / n_slices)

        del inp, out
        torch.cuda.empty_cache()

    progress_bar.progress(1.0)
    st.success("✅ Volume processed successfully!")

    # ========================
    # Save + Visualization
    # ========================
    # Convert back to CT intensity scale for saving
    processed_ct = processed * (vmax - vmin) + vmin
    out_img = nib.Nifti1Image(processed_ct, affine=nii_img.affine)
    out_path = os.path.join(tempfile.gettempdir(), "processed_output.nii")
    nib.save(out_img, out_path)
    gc.collect()

    st.subheader("📊 Slice Viewer")

    # --- Initialize session state ---
    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = n_slices // 2

    # --- Layout: slider + box ---
    col1, col2 = st.columns([3, 1])
    with col1:
        slice_idx = st.slider(
            "Select slice index",
            0, n_slices - 1,
            value=st.session_state.slice_idx,
            key="slice_slider"
        )
    with col2:
        st.write(f"Current: {int(slice_idx)}")

    # --- Update session state ---
    st.session_state.slice_idx = slice_idx

    # --- Plot slices ---
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    orig_slice = vol_norm[:, :, int(slice_idx)]
    vmin_o, vmax_o = np.percentile(orig_slice, (2, 98))
    ax[0].imshow(orig_slice, cmap="gray", vmin=vmin_o, vmax=vmax_o)
    ax[0].set_title(f"Original Slice {int(slice_idx)}")
    
    proc_slice = processed[:, :, int(slice_idx)]
    vmin_p, vmax_p = np.percentile(proc_slice, (2, 98))
    ax[1].imshow(proc_slice, cmap="gray", vmin=vmin_p, vmax=vmax_p)
    ax[1].set_title(f"Processed Slice {int(slice_idx)}")
    for a in ax:
        a.axis("off")
    st.pyplot(fig)

    with open(out_path, "rb") as f:
        st.download_button(
            label="⬇ Download Processed Volume (.nii)",
            data=f,
            file_name="processed_output.nii",
            mime="application/octet-stream",
        )

else:
    st.info("Please upload a `.nii` file (≤1 GB) to begin.")