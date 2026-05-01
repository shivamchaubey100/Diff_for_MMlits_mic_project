# Diff4MMLiTS — Conditional Diffusion Model for Liver Tumor CT Inpainting

> **MIC (Medical Image Computing) Course Project**  
> Inspired by: *Diff4MMLiTS: Advanced Multimodal Liver Tumor Segmentation via Diffusion-Based Image Synthesis and Alignment* — Chen et al., MLMI 2025  
> **Authors:** Rishith Gupta (23B1234) · Shivam Chaubey (23B1244)  
> **Dataset:** [LiTS — Liver Tumor Segmentation Benchmark](https://competitions.codalab.org/competitions/17094)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation & Problem Statement](#2-motivation--problem-statement)
3. [Repository Structure](#3-repository-structure)
4. [Pipeline: 4 Stages at a Glance](#4-pipeline-4-stages-at-a-glance)
5. [File-by-File Walkthrough](#5-file-by-file-walkthrough)
   - [preprocess.py](#51-preprocesspy)
   - [inpainting.py](#52-inpaintingpy)
   - [dataset.py](#53-datasetpy)
   - [models.py](#54-modelspy)
   - [train.py](#55-trainpy)
   - [augment.py](#56-augmentpy)
   - [visualise.py](#57-visualisepy)
   - [main.py](#58-mainpy)
6. [Diffusion Model — Key Design Decisions](#6-diffusion-model--key-design-decisions)
7. [Loss Function](#7-loss-function)
8. [Training Configuration](#8-training-configuration)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [How to Run](#10-how-to-run)
11. [Directory Layout After Running](#11-directory-layout-after-running)
12. [Differences from the Original Diff4MMLiTS Paper](#12-differences-from-the-original-diff4mmlits-paper)
13. [Dependencies](#13-dependencies)

---

## 1. Project Overview

This project implements a **conditional denoising diffusion probabilistic model (DDPM)** that learns to synthesize realistic, healthy liver CT tissue in place of tumor regions. Given a CT slice that contains a liver tumor, the model outputs a CT slice that looks as if the tumor was never there — a process called **CT inpainting**.

The synthesized healthy CTs can then be used as high-quality training data for downstream liver tumor segmentation models, addressing the chronic data scarcity problem in medical imaging.

The overall approach is a simplified, single-modality, 2D adaptation of the four-stage **Diff4MMLiTS** pipeline (Chen et al., MLMI 2025).

---

## 2. Motivation & Problem Statement

- **Liver tumors corrupt CT appearance.** A liver tumor changes the local HU (Hounsfield Unit) intensity distribution dramatically. Segmentation models trained on such corrupted data struggle to generalize.
- **Data augmentation requires healthy CTs.** To synthetically insert tumors in new positions, you first need a clean healthy-liver CT as a canvas. These are rare in annotated datasets.
- **Traditional inpainting fails.** Classical approaches like median filtering or OpenCV's `cv2.inpaint` produce blurry, anatomically implausible fills that do not respect the real intensity statistics of liver parenchyma.
- **GANs are unstable; VAEs are blurry.** Diffusion models offer a principled, stable training objective and produce diverse, sharp, HU-consistent outputs — making them the right tool for this task.

---

## 3. Repository Structure

```
project/
│
├── preprocess.py        # Stage 1 — load NIfTI, HU clip, normalize, resize, save .npz
├── inpainting.py        # Stage 2 — replace tumor pixels with healthy liver pixels (NCG)
├── dataset.py           # PyTorch Dataset — loads inpainted + original CT + masks per slice
├── models.py            # Enhanced conditional U-Net with time embedding + self-attention
├── train.py             # Diffusion class — forward process, v-parameterization, sampling
├── augment.py           # Geometric + intensity augmentation utilities
├── visualise.py         # Saves 4-panel reconstruction plots per epoch
├── main.py              # Full training entrypoint — wires all modules together
│
├── processed_volumes/   # [generated] Preprocessed CT .npz files
├── processed_masks/     # [generated] Preprocessed segmentation mask .npz files
├── inpainted_volumes/   # [generated] Inpainted (tumor-removed) CT .npz files
├── checkpoints/         # [generated] Model checkpoints + train_metrics.csv
└── visualizations/      # [generated] Per-epoch reconstruction plots
```

---

## 4. Pipeline: 4 Stages at a Glance

```
Raw NIfTI CTs + Segmentation Masks
          │
          ▼
  ┌─────────────────┐
  │  Stage 1        │  preprocess.py
  │  Preprocessing  │  HU clip → normalize → gamma correct → resize 256×256 → .npz
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Stage 2        │  inpainting.py
  │  Inpainting     │  Sample real healthy liver pixels to fill tumor region (NCG)
  │  (NCG)         │  Output: "healthy-looking" CT as the conditioning signal
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────────────┐
  │  Stage 3 — Diffusion Training           │  models.py + train.py + main.py
  │                                         │
  │  Input:  noisy CT (xₜ)                 │
  │          + inpainted CT (condition)     │
  │          + liver mask (condition)       │
  │  Model:  Enhanced conditional U-Net     │
  │  Loss:   v-pred MSE + L1 recon         │
  │          + masked L1 + BG consistency   │
  └────────┬────────────────────────────────┘
           │
           ▼
  ┌─────────────────┐
  │  Stage 4        │  train.py → Diffusion.sample()
  │  Inference      │  300-step DDPM reverse sampling
  │                 │  + Classifier-Free Guidance (scale=2.0)
  └─────────────────┘
           │
           ▼
  Synthesized healthy-liver CT slice
```

---

## 5. File-by-File Walkthrough

---

### 5.1 `preprocess.py`

**What it does:** Converts raw NIfTI (`.nii`) CT volumes and segmentation masks into normalized NumPy arrays saved as `.npz` files. This is a one-time offline step before training.

**Key functions:**

| Function | Description |
|---|---|
| `preprocess_volume_soft(vol, ...)` | HU clip → normalize to [0,1] → gamma correct → resize per slice |
| `preprocess_mask_soft(seg, ...)` | Resize segmentation mask with nearest-neighbour (preserves integer labels 0/1/2) |
| `preprocess_dataset(...)` | Finds all `volume-*.nii` / `segmentation-*.nii` pairs, processes them, saves `.npz` |

**Configuration constants at the top of the file:**

| Constant | Default | Meaning |
|---|---|---|
| `CLIP_MIN` | `-150` | Lower HU bound (air/fat boundary) |
| `CLIP_MAX` | `350` | Upper HU bound (bone excluded) |
| `OUT_SIZE` | `(256, 256)` | Spatial resolution per slice |
| `GAMMA` | `1.5` | Gamma correction exponent for soft tissue contrast |
| `MIN_MASK_VOXELS` | `1000` | Skip volumes with very small/empty masks |

**Preprocessing pipeline per slice:**
```
raw HU value
    → clip to [CLIP_MIN, CLIP_MAX]
    → normalize: (x - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)  → [0, 1]
    → gamma: pixel ^ GAMMA                                  → [0, 1], contrast enhanced
    → bilinear resize to 256×256
```

**Segmentation mask labels (preserved throughout):**
- `0` = background
- `1` = liver parenchyma
- `2` = tumor

**Output files:**
- `processed_volumes/volume-X.npz` → key: `"image"`, shape `(Z, 256, 256)`, dtype `float32`
- `processed_masks/segmentation-X.npz` → key: `"mask"`, shape `(Z, 256, 256)`, dtype `float32`

---

### 5.2 `inpainting.py`

**What it does:** Implements the **Normal CT Generator (NCG)** — the inpainting module that removes tumors from preprocessed CT volumes by replacing tumor pixels with real, randomly sampled healthy liver pixels from the same slice.

**Key functions:**

| Function | Description |
|---|---|
| `inpaint_using_liver_patches(ct_slice, liver_mask, tumor_mask)` | Core per-slice inpainting logic |
| `inpaint_volume(ct_vol, liver_vol, tumor_vol)` | Applies per-slice inpainting across entire 3D volume |
| `inpaint_dataset(...)` | Loads preprocessed `.npz` pairs, runs `inpaint_volume`, saves results |

**How the inpainting works:**

```python
healthy_pixels = ct_slice[liver_mask & ~tumor_mask]   # real healthy liver pixels
rand_idx = np.random.randint(0, len(healthy_pixels), size=num_tumor_pixels)
out[tumor_mask] = healthy_pixels[rand_idx]            # fill tumor with random healthy samples
```

**Why this approach over `cv2.inpaint` or Gaussian smoothing:**
- No blurring — pixel values are drawn from the real intensity distribution of that patient's liver
- No synthetic noise — no artificial statistics introduced
- Preserves HU value plausibility — the filled region is statistically indistinguishable from healthy liver
- Fallback: if a slice has no healthy liver pixels at all (rare edge case), fill with the slice mean

**Output files:**
- `inpainted_volumes/volume-X_inpainted.npz` → key: `"inpainted"`, shape `(Z, 256, 256)`

---

### 5.3 `dataset.py`

**What it does:** PyTorch `Dataset` that serves `(inpainted_slice, original_CT_slice, tumor_mask, liver_mask)` tuples to the training loop.

**Key functions:**

| Function | Description |
|---|---|
| `build_slice_entries_for_pairs(pairs)` | Expands (inpainted, original, mask) volume triplets into per-slice entry dicts |
| `ensure_channel_first(x)` | Normalizes numpy arrays to `(1, H, W)` shape |
| `CTNPZDataset.__getitem__` | Loads one slice, builds masks, preprocesses, returns 4 tensors |

**What each returned tensor represents:**

| Tensor | Shape | Range | Description |
|---|---|---|---|
| `healthy_t` | `(1, H, W)` | `[-1, 1]` | Inpainted CT — the diffusion conditioning input |
| `target_t` | `(1, H, W)` | `[-1, 1]` | Original CT with tumor — the reconstruction target |
| `tumor_mask_t` | `(1, H, W)` | `{0, 1}` | Binary mask: 1 inside tumor region |
| `liver_mask_t` | `(1, H, W)` | `{0, 1}` | Binary mask: 1 inside entire liver (including tumor) |

**Normalization to [-1, 1]:** `preprocess_volume_soft` is called on each slice, which maps `[0, 1]` → further processed. Final tensors are in `[-1, 1]` as required for diffusion training.

**Mask construction logic:**
```python
if max(seg) >= 2:
    tumor_mask = (seg == 2)   # class 2 = tumor
    liver_mask = (seg >= 1)   # classes 1 and 2 = full liver region
else:
    tumor_mask = (seg > 0)    # fallback for binary masks
    liver_mask = (seg > 0)
```

---

### 5.4 `models.py`

**What it does:** Defines the **Enhanced Conditional U-Net** — the neural network that forms the backbone of the diffusion model.

**Architecture at a glance:**

```
Input: [xₜ (1ch) | inpainted CT (1ch) | liver mask (1ch)] = 3 channels
                    │
            ┌───────┴────────┐
            │  input_conv    │  3 → 48 channels, 3×3 Conv
            └───────┬────────┘
                    │
         ┌──────────┼──────────┐
         │  ENCODER (3 levels) │
         │  L1: 48ch  (1×)     │◄── skip₁
         │  L2: 96ch  (2×)     │◄── skip₂   + strided Conv downsample
         │  L3: 192ch (4×)     │◄── skip₃
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │     BOTTLENECK      │
         │  ResBlock           │
         │  Self-Attention(4h) │  ← only attention in the network
         │  ResBlock           │
         └──────────┬──────────┘
                    │
         ┌──────────┼──────────┐
         │  DECODER (3 levels) │
         │  + skip₃ → ResBlocks│
         │  + skip₂ → ResBlocks│  + ConvTranspose2d upsample
         │  + skip₁ → ResBlocks│
         └──────────┬──────────┘
                    │
            ┌───────┴────────┐
            │  out_norm+conv  │  GroupNorm → SiLU → 1×1 Conv → 1 channel
            └───────┬────────┘
                    │
              Predicted v (or ε)
```

**Key components:**

| Component | Class | Role |
|---|---|---|
| Time embedding | `SinusoidalPosEmb` | Maps integer timestep `t` → continuous sinusoidal vector → 2-layer MLP |
| Building block | `ResBlock` | GroupNorm + SiLU + Conv2d × 2, with time MLP injected additively after first conv |
| Attention | `AttentionBlock` | Multi-head self-attention (4 heads) at bottleneck only — captures global context |
| Full model | `EnhancedUNet` | Assembles encoder, bottleneck, decoder with skip connections |

**Conditioning mechanism:** The inpainted CT and liver mask are simply concatenated with the noisy input `xₜ` along the channel dimension before the first convolution. This means conditioning is "baked in" from the very first layer — the model always sees the healthy reference alongside the noisy input.

**Classifier-free guidance support:** The `drop_cond=True` flag in `forward()` zeros out the conditioning channels, allowing the model to also learn the unconditional distribution during training.

**Constructor parameters:**

| Parameter | Default | Meaning |
|---|---|---|
| `in_ch` | `1` | Input CT channels |
| `cond_ch` | `2` | Conditioning channels (inpainted CT + liver mask) |
| `base_ch` | `48` | Base channel width |
| `ch_mult` | `(1, 2, 4)` | Channel multipliers per encoder level |
| `num_res_blocks` | `2` | ResBlocks per level |
| `time_dim` | `256` | Timestep embedding dimension |

---

### 5.5 `train.py`

**What it does:** Implements the **Diffusion** class — encapsulates the forward (noising) process, the v-parameterization training objective, and the reverse (denoising/sampling) process.

**Noise schedules:**

| Schedule | Formula | When to use |
|---|---|---|
| Linear | `β = linspace(1e-4, 0.02, T)` | Simpler, can cause instability at low noise levels |
| **Cosine** (used) | `αₜ = cos²(π/2 · (t/T + s)/(1+s))` | Smoother transitions, more stable for medical images |

**Key methods:**

| Method | Description |
|---|---|
| `q_sample(x0, noise, t)` | Forward process: `xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε` |
| `get_v_target(x0, noise, t)` | Compute v-target: `v = √ᾱₜ·ε - √(1-ᾱₜ)·x₀` |
| `predict_x0_from_v(xt, v_pred, t)` | Reconstruct x₀: `x̂₀ = √ᾱₜ·xₜ - √(1-ᾱₜ)·v_pred` |
| `p_losses(model, x_start, cond, t, ...)` | One training step — samples noise, noises x₀, runs model, computes loss |
| `sample(model, cond, shape, ...)` | Full reverse DDPM loop with classifier-free guidance |

**V-parameterization — why it matters:**

Standard diffusion models predict the added noise `ε`. At very high noise levels (large `t`), the noise signal dominates and the gradient signal from `x₀` nearly vanishes, making training unstable. V-parameterization predicts `v = √ᾱ·ε - √(1-ᾱ)·x₀` — a rotation in (ε, x₀) space that maintains a well-conditioned gradient at all timesteps.

**Classifier-Free Guidance at inference:**

```python
v_cond   = model(x, cond, t, drop_cond=False)   # conditioned prediction
v_uncond = model(x, cond, t, drop_cond=True)    # unconditioned prediction
v        = v_uncond + guidance_scale * (v_cond - v_uncond)  # guided prediction
```

This "steers" the output further toward what the conditioning improves, sharpening the result without needing a separately trained classifier.

---

### 5.6 `augment.py`

**What it does:** Lightweight geometric and intensity augmentation applied to `(image, mask)` pairs. Handles both single-sample `(C, H, W)` and batched `(B, C, H, W)` inputs transparently.

**Augmentations applied:**

| Augmentation | Probability | Notes |
|---|---|---|
| Horizontal flip | 50% | Applied identically to image and mask |
| Vertical flip | 50% | Applied identically to image and mask |
| 90°/180°/270° rotation | 50% | Random k chosen from {1, 2, 3} |
| Intensity jitter | 30% | Adds `N(0, 0.05)` noise, clamps to `[-1, 1]` |

**Note:** Augmentation was available in the codebase but was not used in the final training run (aug_prob effectively set to 0 in `CTNPZDataset`). The masks are always transformed identically to the image to maintain spatial correspondence.

---

### 5.7 `visualise.py`

**What it does:** Generates a 4-panel diagnostic plot after each training epoch, saved to `visualizations/`.

**The 4 panels:**

| Panel | Content | How generated |
|---|---|---|
| Healthy Cond | The inpainted CT conditioning input | Taken directly from batch |
| Ground Truth | The original CT with tumor | Taken directly from batch |
| x0_hat (mid-step) | Single-pass reconstruction at t=T/2 | `v_pred` → `predict_x0_from_v` |
| Final Reconstruction | Full 300-step DDPM sample | `diffusion.sample()` with CFG |

**Display conversion:** All tensors are in `[-1, 1]`. Display maps them to `[0, 1]` via `(x + 1) / 2`, then `np.clip(..., 0, 1)`.

**Saved to:** `visualizations/recon_pass{epoch}_mid.png`

---

### 5.8 `main.py`

**What it does:** The master training script. Wires all modules together, implements the training loop with all loss components, EMA, mixed precision, checkpointing, and per-epoch visualization.

**High-level flow:**

```
1. Build slice entries from (inpainted, original, mask) triplets
2. For each epoch:
   a. Sample up to 5000 slices randomly (epoch-seeded for reproducibility)
   b. Build CTNPZDataset + DataLoader
   c. For each batch:
      i.   Resize all tensors to FORCE_SIZE (256×256)
      ii.  Sample random timesteps t ~ Uniform(0, T)
      iii. Form conditioning: cat([inpainted_CT, liver_mask], dim=1)
      iv.  20% chance: drop conditioning (CFG training)
      v.   Forward pass → v_pred, x0_hat, loss
      vi.  Composite loss (see Section 7)
      vii. Backward + gradient clip + optimizer step
      viii.EMA update
   d. Save checkpoint
   e. Visualize
3. Save train_metrics.csv
```

**Loss weights defined in `main.py`:**

```python
W_FULL_MAIN  = 1.0   # core v-prediction MSE
W_MASK_MAIN  = 0.0   # masked v-loss (disabled)
W_FULL_RECON = 4.0   # full-image L1 reconstruction
W_MASK_RECON = 1.0   # tumor-region L1 reconstruction
W_BG         = 1.0   # background consistency (applied with factor 2.0 in loss line)
W_SSIM       = 0.0   # SSIM loss (disabled)
W_PERC       = 0.0   # VGG perceptual loss (disabled)
```

---

## 6. Diffusion Model — Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| **Noise schedule** | Cosine | Smoother noise transitions, more stable training than linear for medical images |
| **Parameterization** | V-prediction | Well-conditioned gradients at all noise levels; more stable than ε-prediction |
| **Timesteps** | T = 300 | Fewer than the standard 1000 — faster training while retaining generation quality |
| **Conditioning** | Channel concatenation | Simple, effective; model sees healthy reference at every layer |
| **Guidance** | Classifier-free (scale=2.0) | Sharpens outputs at inference without a separate classifier model |
| **Attention** | Bottleneck only | Balances global context capture with memory/compute efficiency |
| **2D vs 3D** | 2D slice-wise | Simpler, faster; 3D would require much more memory and data |

---

## 7. Loss Function

The total training loss combines four terms, each targeting a different failure mode:

```
L = 1.0 × L_core  +  4.0 × L_full_recon  +  1.0 × L_masked_recon  +  2.0 × L_bg
```

| Term | Formula | Purpose |
|---|---|---|
| **L_core** | `MSE(v_pred, v_target)` | Trains the denoising backbone — the fundamental diffusion objective |
| **L_full_recon** | `L1(x̂₀, target)` | Global pixel accuracy across the entire CT slice — highest weight (4.0) |
| **L_masked_recon** | `L1(x̂₀ · tumor_mask, target · tumor_mask)` | Extra supervision focused on the tumor region — where inpainting quality matters most |
| **L_bg** | `L1(x̂₀ · inv_mask, healthy_cond · inv_mask)` | Non-liver regions must stay identical to the healthy input — prevents the model from hallucinating outside the liver |

**Why these four terms together:**
- Without `L_full_recon`: model could satisfy `L_core` while producing globally inaccurate intensities
- Without `L_masked_recon`: the tumor region (the critical target area) might be under-supervised relative to easier background regions
- Without `L_bg`: the model might modify non-liver regions, introducing artifacts that could confuse downstream segmentation models

---

## 8. Training Configuration

| Hyperparameter | Value | Location |
|---|---|---|
| Epochs | 100 | `main.py` |
| Batch size | 16 | `main.py` |
| Max slices / epoch | 5000 | `main.py` |
| Learning rate | 1e-4 | `main.py` |
| Optimizer | AdamW (wd=1e-4) | `main.py` |
| LR schedule | CosineAnnealingLR (η_min = LR×0.01) | `main.py` |
| Mixed precision | FP16 via `torch.cuda.amp` | `main.py` |
| EMA decay | 0.9999 (update every 10 steps) | `main.py` |
| CFG dropout rate | 20% of batches | `main.py` |
| Gradient clipping | max norm = 1.0 | `main.py` |
| Image resolution | 256 × 256 | `main.py` |
| Timesteps T | 300 | `main.py` |
| Guidance scale | 2.0 (inference only) | `visualise.py`, `train.py` |

---

## 9. Evaluation Metrics

All metrics are computed per batch during training, averaged per epoch, and saved to `checkpoints/train_metrics.csv`.

| Metric | Range | Physical Interpretation |
|---|---|---|
| **PSNR** | `[0, ∞)` dB (higher better) | Signal-to-noise ratio in decibels. Every +3 dB halves the mean squared reconstruction error. A PSNR of 30 dB means the error is about 0.1% of the signal energy. |
| **SSIM** | `[0, 1]` (higher better) | Jointly measures luminance, contrast, and structural similarity. Closer to how the human visual system perceives image quality than pixel-wise metrics. A score of 1.0 is a perfect reconstruction. |
| **Tumor MSE** | `[0, ∞)` (lower better) | Mean squared error computed **only inside the tumor mask**. Directly measures inpainting quality in the region that matters most — the area the model was asked to synthesize. |
| **Full MSE** | `[0, ∞)` (lower better) | Mean squared error across the **entire CT slice**. Checks that the model did not improve the tumor region at the expense of breaking surrounding anatomy. |

**Visualization:** At the end of each epoch, `visualise.py` saves a 4-panel plot comparing the healthy conditioning input, ground truth, mid-step x0_hat reconstruction, and the full DDPM sample.

---

## 10. How to Run

### Prerequisites

```bash
pip install torch torchvision numpy nibabel opencv-python tqdm matplotlib pandas einops
```

### Step 1 — Preprocess the dataset

```python
# In main.py, uncomment this block:
preprocess_dataset(DATASET_IN, PROCESSED_IMG, PROCESSED_MASKS)
```

Or call directly:
```python
from preprocess import preprocess_dataset
preprocess_dataset(
    dataset_in="path/to/Training_Batch",
    img_dir="processed_volumes",
    mask_dir="processed_masks"
)
```

Expected input structure:
```
Training_Batch/
  volume-0.nii
  segmentation-0.nii
  volume-1.nii
  segmentation-1.nii
  ...
```

### Step 2 — Run inpainting

```python
# In main.py, uncomment:
inpaint_dataset(
    processed_ct_dir="processed_volumes",
    processed_mask_dir="processed_masks",
    out_dir="inpainted_volumes",
    force_binary=False
)
```

### Step 3 — Train

```bash
python main.py
```

Training will auto-resume from the latest checkpoint in `checkpoints/` if one exists.

### Step 4 — Monitor

- **Loss/metrics per epoch:** printed to stdout
- **Visual reconstructions:** `visualizations/recon_pass{epoch}_mid.png`
- **CSV of all metrics:** `checkpoints/train_metrics.csv`

---

## 11. Directory Layout After Running

```
project/
├── processed_volumes/
│   ├── volume-0.npz         # key: "image", shape (Z, 256, 256)
│   └── ...
├── processed_masks/
│   ├── segmentation-0.npz   # key: "mask", shape (Z, 256, 256), values in {0,1,2}
│   └── ...
├── inpainted_volumes/
│   ├── volume-0_inpainted.npz  # key: "inpainted", shape (Z, 256, 256)
│   └── ...
├── checkpoints/
│   ├── latest_full.pt          # full checkpoint (model + opt + scheduler + EMA + scaler)
│   ├── checkpoint_epoch_0000.pt
│   ├── checkpoint_epoch_0001.pt
│   └── train_metrics.csv       # columns: train_loss, psnr, ssim, tumor_mse, full_mse
└── visualizations/
    ├── recon_pass0_mid.png
    ├── recon_pass1_mid.png
    └── ...
```

---

## 12. Differences from the Original Diff4MMLiTS Paper

| Aspect | Original Diff4MMLiTS | This Implementation |
|---|---|---|
| **Dimensionality** | 3D volumes | 2D slice-wise (simpler, less memory) |
| **Diffusion space** | Latent space (LDM with KL-VAE) | Pixel space (direct DDPM) |
| **Modalities** | Multimodal (portal + venous CT phases) | Single modality CT + mask conditioning |
| **Parameterization** | ε-prediction | **V-prediction** (added for stability) |
| **Loss** | Standard diffusion loss | **Composite loss** with background consistency (novel) |
| **Guidance** | Not mentioned | **Classifier-free guidance** (added) |
| **EMA** | Not mentioned | **EMA averaging** (added) |
| **Inpainting (NCG)** | Dilated mask + inpainting | Random healthy pixel sampling — simpler but effective |

The simplifications are justified by the single-modality, course-project scope. The additions (v-parameterization, composite loss, CFG, EMA) are improvements over the baseline that address known failure modes.

---

## 13. Dependencies

| Package | Purpose |
|---|---|
| `torch` | Model training, GPU compute, autocast |
| `torchvision` | VGG perceptual model (optional, disabled by default) |
| `numpy` | Array operations throughout |
| `nibabel` | Loading NIfTI `.nii` CT volumes |
| `opencv-python` | Image resizing (`cv2.resize`) |
| `tqdm` | Progress bars |
| `matplotlib` | Visualization plots |
| `pandas` | Saving metrics CSV |
| `einops` | Tensor rearrangement in attention block |

---

*For questions about the implementation, refer to the inline comments in each file. For questions about the paper, see: Chen et al., "Diff4MMLiTS: Advanced Multimodal Liver Tumor Segmentation via Diffusion-Based Image Synthesis and Alignment", MLMI 2025, arXiv:2412.20418.*
