# augment.py (fixed)

"""
Lightweight medical image augmentation utilities.
Applies simple geometric transformations (flips, rotations)
to both CT slices and corresponding masks.

Now: DOES NOT add an extra batch dimension for single-sample inputs.
Accepts:
  - img_t: (C,H,W) or (B,C,H,W)
  - mask_t: (1,H,W) or (B,1,H,W)
Returns:
  - img_t, mask_t with same number of leading dims as input
"""

import torch
import random

def _is_batched(t: torch.Tensor) -> bool:
    return t.dim() == 4

def _ensure_batched(x: torch.Tensor):
    return x.unsqueeze(0) if x.dim() == 3 else x

def _unbatch_if_needed(x_out: torch.Tensor, original_was_batched: bool):
    return x_out if original_was_batched else x_out[0]

def augment_medical(img_t: torch.Tensor, mask_t: torch.Tensor, p: float = 0.5):
    """
    Apply simple geometric augmentations to (img, mask) tensors.

    Args:
        img_t (Tensor): (C,H,W) or (B,C,H,W)
        mask_t (Tensor): (1,H,W) or (B,1,H,W)
        p (float): probability of applying augmentation

    Returns:
        (Tensor, Tensor): augmented image and mask with the SAME
                          leading-dim structure as the inputs.
    """
    if random.random() > p:
        return img_t, mask_t

    # Record whether inputs are batched
    img_batched = _is_batched(img_t)
    mask_batched = _is_batched(mask_t)

    # Temporarily add batch dim if single-sample for vectorized ops
    img = _ensure_batched(img_t)    # shape -> (B, C, H, W)
    mask = _ensure_batched(mask_t)  # shape -> (B, 1, H, W)

    # ----- geometric augmentations -----
    # Random horizontal flip
    if random.random() < 0.5:
        img = torch.flip(img, dims=[-1])
        mask = torch.flip(mask, dims=[-1])

    # Random vertical flip
    if random.random() < 0.5:
        img = torch.flip(img, dims=[-2])
        mask = torch.flip(mask, dims=[-2])

    # Random 90-degree rotation
    if random.random() < 0.5:
        k = random.choice([1, 2, 3])  # rotate by 90, 180, or 270 deg
        img = torch.rot90(img, k, dims=[-2, -1])
        mask = torch.rot90(mask, k, dims=[-2, -1])

    # Intensity jitter
    if random.random() < 0.3:
        img = img + 0.05 * torch.randn_like(img)
        img = torch.clamp(img, -1.0, 1.0)

    # Remove batch dim if the original inputs were single-sample
    img_out = _unbatch_if_needed(img, img_batched)
    mask_out = _unbatch_if_needed(mask, mask_batched)

    return img_out, mask_out
