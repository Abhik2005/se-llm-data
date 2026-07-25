"""
training/checkpoint.py — Save and auto-resume training checkpoints.

Critical for Kaggle training: Kaggle sessions disconnect after 12 hours.
This module saves state every N steps so training can resume exactly
where it left off in the next Kaggle session.

Checkpoint contains:
    - Model weights (state_dict)
    - Optimizer state (Adam momentum + variance)
    - Training step number
    - Total tokens processed
    - Best validation loss seen so far
    - Model config (so we can rebuild the model from checkpoint)
    - Training config
"""

import os
import glob
import torch
from pathlib import Path
from typing import Optional

# Import XLA model utilities for TPU-safe checkpoint saving.
# On a TPU, weights are sharded across 8 chips — xm.save gathers them
# all onto the master chip before writing, preventing corruption.
try:
    import torch_xla.core.xla_model as xm
    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens_processed: int,
    val_loss: float,
    model_config: dict,
    train_config: dict,
    keep_last_n: int = 3,
) -> str:
    """
    Save a training checkpoint to disk.

    Args:
        checkpoint_dir:    Directory to save checkpoints
        step:              Current training step
        model:             The model to save
        optimizer:         Optimizer with its state
        tokens_processed:  Total tokens trained on so far
        val_loss:          Latest validation loss
        model_config:      ModelConfig as dict
        train_config:      Training config as dict
        keep_last_n:       How many past checkpoints to keep

    Returns:
        Path to saved checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "step":             step,
        "tokens_processed": tokens_processed,
        "val_loss":         val_loss,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "model_config":     model_config,
        "train_config":     train_config,
    }

    # Save step-specific checkpoint
    ckpt_path   = os.path.join(checkpoint_dir, f"checkpoint_step_{step:07d}.pt")
    latest_path = os.path.join(checkpoint_dir, "latest.pt")

    if _XLA_AVAILABLE:
        # xm.save: gathers weights from all 8 TPU chips onto the master
        # chip before writing — prevents file corruption and hangs.
        xm.save(checkpoint, ckpt_path)
        xm.save(checkpoint, latest_path)
    else:
        torch.save(checkpoint, ckpt_path)
        torch.save(checkpoint, latest_path)

    # Only log and prune on the master process to avoid duplicate output
    if not _XLA_AVAILABLE or xm.is_master_ordinal():
        print(f"[Checkpoint] Saved step {step:,} → {ckpt_path}")
        _prune_old_checkpoints(checkpoint_dir, keep_last_n)

    return ckpt_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Load a checkpoint and restore model + optimizer state.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        model:           Model to load weights into
        optimizer:       Optimizer to restore state into (optional)
        device:          Device to load tensors onto

    Returns:
        Checkpoint dict with metadata (step, tokens_processed, val_loss, etc.)
    """
    print(f"[Checkpoint] Loading from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    step             = checkpoint.get("step", 0)
    tokens_processed = checkpoint.get("tokens_processed", 0)
    val_loss         = checkpoint.get("val_loss", float("inf"))

    print(
        f"[Checkpoint] Resumed from step {step:,} | "
        f"Tokens: {tokens_processed / 1e9:.3f}B | "
        f"Val loss: {val_loss:.4f}"
    )

    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Priority:
        1. 'latest.pt' if it exists (most recent)
        2. Highest step number in checkpoint_step_*.pt files

    Args:
        checkpoint_dir: Directory to search

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    # Check for latest.pt shortcut
    latest = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest):
        return latest

    # Fallback: find highest-numbered checkpoint
    pattern = os.path.join(checkpoint_dir, "checkpoint_step_*.pt")
    checkpoints = sorted(glob.glob(pattern))
    if checkpoints:
        return checkpoints[-1]

    return None


def auto_resume(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, float]:
    """
    Automatically detect and load the latest checkpoint.
    If no checkpoint exists, starts fresh from step 0.

    Args:
        checkpoint_dir: Directory to search for checkpoints
        model:          Model to restore
        optimizer:      Optimizer to restore
        device:         Device to load onto

    Returns:
        (start_step, tokens_processed, best_val_loss)
    """
    ckpt_path = find_latest_checkpoint(checkpoint_dir)

    if ckpt_path is None:
        print("[Checkpoint] No checkpoint found — starting fresh from step 0")
        return 0, 0, float("inf")

    ckpt = load_checkpoint(ckpt_path, model, optimizer, device)
    return (
        ckpt.get("step", 0),
        ckpt.get("tokens_processed", 0),
        ckpt.get("val_loss", float("inf")),
    )


def _prune_old_checkpoints(checkpoint_dir: str, keep_last_n: int) -> None:
    """Remove old step checkpoints, keeping only the most recent N."""
    pattern     = os.path.join(checkpoint_dir, "checkpoint_step_*.pt")
    checkpoints = sorted(glob.glob(pattern))

    # Keep the latest N, delete the rest
    to_delete = checkpoints[:-keep_last_n] if len(checkpoints) > keep_last_n else []
    for path in to_delete:
        os.remove(path)
        print(f"[Checkpoint] Removed old checkpoint: {os.path.basename(path)}")


if __name__ == "__main__":
    import tempfile
    from model.config import config_1m_test
    from model.transformer import Transformer

    cfg   = config_1m_test()
    model = Transformer(cfg)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save checkpoint
        save_checkpoint(
            checkpoint_dir=tmpdir,
            step=500,
            model=model,
            optimizer=opt,
            tokens_processed=2_048_000,
            val_loss=3.14,
            model_config=cfg.to_dict(),
            train_config={"lr": 1e-3},
            keep_last_n=3,
        )

        # Load checkpoint
        start_step, tokens, val_loss = auto_resume(tmpdir, model, opt, torch.device("cpu"))
        print(f"Resumed: step={start_step}, tokens={tokens:,}, val_loss={val_loss:.4f}")

    print("Checkpoint: OK")
