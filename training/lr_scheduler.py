"""
training/lr_scheduler.py — Cosine Learning Rate Scheduler with Linear Warmup.

Schedule:
    Phase 1 (warmup): Linear ramp from 0 → peak_lr over warmup_steps
    Phase 2 (decay):  Cosine decay from peak_lr → min_lr over total_steps

This is the standard schedule used by GPT-2, LLaMA, and all modern LLMs.
"""

import math


def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """
    Compute learning rate at a given training step.

    Args:
        step:         Current training step (0-indexed)
        warmup_steps: Steps for linear warmup phase
        total_steps:  Total training steps
        peak_lr:      Peak (maximum) learning rate
        min_lr:       Minimum (final) learning rate

    Returns:
        Learning rate for this step
    """
    # Phase 1: Linear warmup
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    # Phase 2: Past training — hold at min_lr
    if step >= total_steps:
        return min_lr

    # Phase 3: Cosine decay from peak_lr → min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    # progress goes from 0.0 (at warmup end) to 1.0 (at total_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine_decay * (peak_lr - min_lr)


def apply_lr(optimizer: "torch.optim.Optimizer", lr: float) -> None:
    """Apply a learning rate to all parameter groups in an optimizer."""
    for group in optimizer.param_groups:
        group["lr"] = lr


if __name__ == "__main__":
    import matplotlib
    # Verify schedule shape (print a few key points)
    warmup = 2000
    total  = 19073
    peak   = 3e-4
    min_lr = 3e-5

    checkpoints = [0, 100, 1000, 2000, 5000, 10000, 15000, 19073]
    print(f"{'Step':>8} | {'LR':>12}")
    print("-" * 25)
    for step in checkpoints:
        lr = get_lr(step, warmup, total, peak, min_lr)
        print(f"{step:>8,} | {lr:>12.2e}")

    # Verify boundaries
    assert abs(get_lr(0, warmup, total, peak, min_lr)) < 1e-10, "LR at step 0 should be ~0"
    assert abs(get_lr(warmup, warmup, total, peak, min_lr) - peak) < 1e-10, "LR at warmup end should be peak"
    assert abs(get_lr(total, warmup, total, peak, min_lr) - min_lr) < 1e-10, "LR at end should be min_lr"
    print("\nLR Scheduler: OK")
