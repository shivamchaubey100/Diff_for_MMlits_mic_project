"""
utils.py

Utility functions for:
 - GPU management
 - Model/device setup
 - Checkpoint saving and auto-resume for epoch-based training

Designed for workflows where:
 • You train for N epochs (e.g. 200)
 • Each epoch uses a new random sample of volumes (e.g. 16)
 • Checkpoints save model + optimizer + current epoch
 • Training resumes automatically from the last saved epoch
"""

import os
import glob
import torch
from pathlib import Path


# ============================================================
# GPU MANAGEMENT
# ============================================================

_USE_ALL_GPUS = False  # internal flag


def enable_all_gpus(enable: bool = True):
    """
    Keep multi-GPU visible, but do NOT change CUDA_VISIBLE_DEVICES.
    PyTorch will automatically see all GPUs.
    """
    global _USE_ALL_GPUS
    _USE_ALL_GPUS = bool(enable)

    n_gpus = torch.cuda.device_count()
    if enable and n_gpus > 1:
        print(f"[utils] Multi-GPU enabled (PyTorch sees {n_gpus} GPUs).")
    else:
        print("[utils] Single-GPU mode.")

def setup_model_for_device(model: torch.nn.Module) -> torch.nn.Module:
    """
    Safe model placement + optional DataParallel.
    No CUDA_VISIBLE_DEVICES mutation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    n_gpus = torch.cuda.device_count()
    if _USE_ALL_GPUS and n_gpus > 1:
        print(f"[utils] Wrapping model with DataParallel ({n_gpus} GPUs).")
        model = torch.nn.DataParallel(model)
    else:
        print(f"[utils] Model on {device} (DataParallel OFF).")

    return model


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def save_checkpoint_epoch(model, optimizer, epoch: int, ckpt_dir: str, prefix: str = "checkpoint"):
    """
    Save model and optimizer state for a specific epoch.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer associated with model.
        epoch (int): Current epoch number to save.
        ckpt_dir (str): Directory to store checkpoints.
        prefix (str): Optional prefix for checkpoint filenames.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{prefix}_epoch_{epoch:04d}.pt"
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }

    torch.save(state, str(ckpt_path))
    print(f"[utils] Saved checkpoint: {ckpt_path}")


def load_latest_checkpoint(model, optimizer, ckpt_dir: str, device="cuda"):
    """
    Loads the latest checkpoint (by highest epoch number) from ckpt_dir.

    Args:
        model (torch.nn.Module): Model instance to load weights into.
        optimizer (torch.optim.Optimizer): Optimizer instance to restore.
        ckpt_dir (str): Directory containing checkpoints.
        device (str): Device to map loaded checkpoint to.

    Returns:
        tuple:
            start_epoch (int): Epoch index to resume from (last_saved_epoch + 1).
            model (torch.nn.Module): Model with loaded weights.
            optimizer (torch.optim.Optimizer): Optimizer with restored state.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_files = sorted(glob.glob(str(ckpt_dir / "checkpoint_epoch_*.pt")))

    if not ckpt_files:
        print(f"[utils] No checkpoints found in {ckpt_dir}. Starting training from epoch 0.")
        return 0, model, optimizer

    latest_ckpt = ckpt_files[-1]
    checkpoint = torch.load(latest_ckpt, map_location=device)

    model.load_state_dict(checkpoint["model_state"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    last_epoch = checkpoint.get("epoch", 0)

    print(f"[utils] Loaded checkpoint: {latest_ckpt} (epoch {last_epoch})")
    return last_epoch + 1, model, optimizer
