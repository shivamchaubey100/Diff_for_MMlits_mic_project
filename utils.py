"""
utils.py

Utility functions for:
 - GPU management
 - Model/device setup
 - Checkpoint saving with storage-efficient cleanup:
     * Always writes  latest_full.pt  (overwritten every epoch)
     * Keeps exactly ONE  checkpoint_epoch_NNNN.pt  (the most recent)
     * Deletes the previous epoch file after the new one is confirmed written
     * If latest_full.pt and the epoch file would be identical in content,
       only latest_full.pt is written (controlled by save_epoch_ckpt flag)
"""

import glob
import os
import torch
from pathlib import Path


# ============================================================
# GPU MANAGEMENT
# ============================================================

_USE_ALL_GPUS = False


def enable_all_gpus(enable: bool = True):
    global _USE_ALL_GPUS
    _USE_ALL_GPUS = bool(enable)
    n_gpus = torch.cuda.device_count()
    if enable and n_gpus > 1:
        print(f"[utils] Multi-GPU enabled (PyTorch sees {n_gpus} GPUs).")
    else:
        print("[utils] Single-GPU mode.")


def setup_model_for_device(model: torch.nn.Module) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
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

def save_checkpoint_epoch(
    model,
    optimizer,
    epoch:    int,
    ckpt_dir: str,
    prefix:   str  = "checkpoint",
    extra:    dict = None,
):
    """
    Save a checkpoint for the current epoch with automatic cleanup.

    Strategy
    --------
    1. Write  latest_full.pt  (always overwritten — single rolling file).
    2. Write  checkpoint_epoch_NNNN.pt  for this epoch.
    3. Delete all older  checkpoint_epoch_*.pt  files so at most one
       epoch file exists on disk at any time.

    This keeps storage bounded to two checkpoint files regardless of how
    many epochs you train.

    Args:
        model     : model to save (unwraps DataParallel automatically)
        optimizer : optimizer whose state to save
        epoch     : current epoch index (0-based)
        ckpt_dir  : directory to store checkpoints
        prefix    : filename prefix for epoch files (default 'checkpoint')
        extra     : optional dict of additional keys to include in the checkpoint
                    (e.g. {'scheduler': sched.state_dict(), 'ema': ema.shadow})
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap DataParallel so the state_dict is portable
    raw_model = model.module if hasattr(model, "module") else model

    state = {
        "epoch":           epoch,
        "model_state":     raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra:
        state.update(extra)

    # ── 1. Write latest_full.pt ───────────────────────────────────────────
    latest_path = ckpt_dir / "latest_full.pt"
    torch.save(state, str(latest_path))
    print(f"[utils] Saved latest_full.pt  (epoch {epoch})")

    # ── 2. Write epoch checkpoint ─────────────────────────────────────────
    epoch_path = ckpt_dir / f"{prefix}_epoch_{epoch:04d}.pt"
    torch.save(state, str(epoch_path))
    print(f"[utils] Saved {epoch_path.name}")

    # ── 3. Delete all previous epoch checkpoints ──────────────────────────
    pattern     = str(ckpt_dir / f"{prefix}_epoch_*.pt")
    all_epoch_ckpts = sorted(glob.glob(pattern))

    # Keep only the one we just wrote; delete everything else
    deleted = []
    for old_ckpt in all_epoch_ckpts:
        if Path(old_ckpt) != epoch_path:
            try:
                os.remove(old_ckpt)
                deleted.append(Path(old_ckpt).name)
            except OSError as e:
                print(f"[utils] WARNING: could not delete {old_ckpt}: {e}")

    if deleted:
        print(f"[utils] Deleted old epoch checkpoints: {deleted}")


def load_latest_checkpoint(
    model,
    optimizer,
    ckpt_dir: str,
    device:   str = "cuda",
):
    """
    Load the most recent checkpoint from ckpt_dir.

    Search order:
      1. latest_full.pt           (preferred — written every epoch)
      2. checkpoint_epoch_NNNN.pt (fallback — in case latest_full.pt is absent)

    Returns
    -------
    start_epoch : int                        — epoch to resume from
    model       : torch.nn.Module            — model with loaded weights
    optimizer   : torch.optim.Optimizer      — optimizer with restored state
    extra_keys  : dict                       — any keys beyond epoch/model/optimizer
                                               (e.g. scheduler, ema, scaler states)
    """
    ckpt_dir = Path(ckpt_dir)

    # Prefer latest_full.pt
    latest_path = ckpt_dir / "latest_full.pt"
    if latest_path.exists():
        ckpt_path = latest_path
    else:
        # Fall back to the highest-numbered epoch file
        epoch_files = sorted(glob.glob(str(ckpt_dir / "checkpoint_epoch_*.pt")))
        if not epoch_files:
            print(f"[utils] No checkpoints found in {ckpt_dir}. Starting from epoch 0.")
            return 0, model, optimizer, {}
        ckpt_path = Path(epoch_files[-1])

    checkpoint = torch.load(str(ckpt_path), map_location=device)

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model_state"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    last_epoch = int(checkpoint.get("epoch", 0))

    # Collect any extra saved keys (scheduler, ema, scaler, etc.)
    known_keys = {"epoch", "model_state", "optimizer_state"}
    extra_keys = {k: v for k, v in checkpoint.items() if k not in known_keys}

    print(f"[utils] Loaded checkpoint: {ckpt_path.name}  (epoch {last_epoch})")
    return last_epoch + 1, model, optimizer, extra_keys