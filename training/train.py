"""
training/train.py — Main Pre-training Loop.

This is the core training script. Designed for:
  - Kaggle P100 (16GB VRAM) — primary training platform
  - Auto-resume from checkpoint on every session restart
  - W&B logging (monitored from your local machine)
  - Mixed precision (bfloat16) for speed

Usage:
    # Start or resume training automatically:
    python training/train.py --config configs/350m.yaml

    # Force restart from scratch:
    python training/train.py --config configs/350m.yaml --resume none

    # Test with tiny model on CPU:
    python training/train.py --config configs/1m_test.yaml
"""

import os
import sys
import time
import math
import argparse
import yaml
from contextlib import nullcontext

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import build_model
from training.dataset import build_dataloader, estimate_loss
from training.checkpoint import save_checkpoint, auto_resume
from training.lr_scheduler import get_lr, apply_lr


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_full_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_wandb(cfg: dict, resume_step: int) -> bool:
    """Initialize W&B if project is set. Returns True if W&B is active."""
    project = cfg["training"].get("wandb_project")
    if not project:
        return False
    try:
        import wandb
        wandb.init(
            project=project,
            entity=cfg["training"].get("wandb_entity"),
            config=cfg,
            resume="allow",
            name=f"{cfg['model']['name']}_step{resume_step}",
        )
        return True
    except ImportError:
        print("[W&B] wandb not installed — skipping logging")
        return False
    except Exception as e:
        print(f"[W&B] Failed to initialize: {e} — skipping")
        return False


def get_device_and_dtype(dtype_str: str) -> tuple[torch.device, torch.dtype, any]:
    """Determine device, dtype, and autocast context."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype  = torch.float32  # CPU doesn't support bfloat16 training

    # autocast context for mixed precision
    if device.type == "cuda":
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        ctx = nullcontext()

    return device, dtype, ctx


# ── Main Training Loop ────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── 1. Load config ────────────────────────────────────────────
    cfg       = load_full_config(args.config)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    tcfg      = cfg["training"]

    print(f"\n{'='*60}")
    print(f"  SE-LLM Training: {model_cfg.name}")
    print(f"  Config: {args.config}")
    print(f"{'='*60}\n")

    # ── 2. Device setup ───────────────────────────────────────────
    device, dtype, autocast_ctx = get_device_and_dtype(tcfg.get("dtype", "bfloat16"))
    print(f"Device: {device} | Dtype: {dtype}")

    # ── 3. Build model ────────────────────────────────────────────
    model = build_model(model_cfg).to(device)

    # Compile model for speed if enabled in config (PyTorch 2.0+)
    use_compile = tcfg.get("compile", False)
    if use_compile and device.type == "cuda" and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # ── 4. Build optimizer ────────────────────────────────────────
    # Separate parameters into weight-decay and no-decay groups
    decay_params     = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": tcfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=tcfg["learning_rate"],
        betas=tuple(tcfg["betas"]),
        eps=tcfg.get("eps", 1e-8),
        fused=device.type == "cuda",   # Faster fused AdamW on GPU
    )

    # GradScaler for fp16 (not needed for bfloat16)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None

    # ── 5. Auto-resume from checkpoint ───────────────────────────
    checkpoint_dir = tcfg["checkpoint_dir"]
    resume_mode    = args.resume or tcfg.get("resume", "auto")

    if resume_mode == "auto":
        start_step, tokens_processed, best_val_loss = auto_resume(
            checkpoint_dir, model, optimizer, device
        )
    else:
        start_step, tokens_processed, best_val_loss = 0, 0, float("inf")
        print("[Checkpoint] Starting fresh from step 0")

    # ── 6. Build data loaders ─────────────────────────────────────
    data_cfg   = cfg.get("data", {})
    train_path = tcfg.get("train_path") or data_cfg.get("train_path", "data/processed/train.bin")
    val_path   = tcfg.get("val_path")   or data_cfg.get("val_path", "data/processed/val.bin")

    train_loader = build_dataloader(
        train_path,
        tcfg["sequence_length"],
        tcfg["batch_size"],
        shuffle=True,
    )
    val_loader = build_dataloader(
        val_path,
        tcfg["sequence_length"],
        tcfg["batch_size"],
        shuffle=False,
    )

    # ── 7. Compute training steps ─────────────────────────────────
    tokens_per_step = (
        tcfg["batch_size"]
        * tcfg["gradient_accumulation"]
        * tcfg["sequence_length"]
    )
    total_steps = tcfg["total_tokens"] // tokens_per_step
    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Total steps:     {total_steps:,}")
    print(f"Resume from:     step {start_step:,} ({tokens_processed/1e9:.3f}B tokens)")
    print(f"Remaining:       {total_steps - start_step:,} steps\n")

    if start_step >= total_steps:
        print("Training already complete!")
        return

    # ── 8. Initialize W&B ─────────────────────────────────────────
    use_wandb = setup_wandb(cfg, start_step)

    # ── 9. Training loop ──────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    data_iter     = iter(train_loader)
    step          = start_step
    t0            = time.time()
    running_loss  = 0.0

    print("Starting training...\n")

    try:
        while step < total_steps:

            # ── Apply learning rate schedule ──────────────────────
            lr = get_lr(
                step,
                tcfg["warmup_steps"],
                total_steps,
                tcfg["learning_rate"],
                tcfg["min_lr"],
            )
            apply_lr(optimizer, lr)

            # ── Gradient accumulation loop ────────────────────────
            accum_loss = 0.0
            for micro_step in range(tcfg["gradient_accumulation"]):
                # Get next batch (cycle through dataset)
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)

                x, y = x.to(device), y.to(device)

                # Forward pass with mixed precision
                with autocast_ctx:
                    _, loss = model(x, y)
                    # Scale loss by accumulation steps
                    loss = loss / tcfg["gradient_accumulation"]

                accum_loss += loss.item()

                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # ── Optimizer step ────────────────────────────────────
            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            step              += 1
            tokens_processed  += tokens_per_step
            running_loss      += accum_loss

            # ── Logging ───────────────────────────────────────────
            if step % tcfg["log_every"] == 0:
                t1       = time.time()
                dt       = t1 - t0
                tok_sec  = (tcfg["log_every"] * tokens_per_step) / dt
                avg_loss = running_loss / tcfg["log_every"]

                progress_pct = 100 * tokens_processed / tcfg["total_tokens"]
                eta_hours    = (total_steps - step) * (dt / tcfg["log_every"]) / 3600

                print(
                    f"step {step:6,}/{total_steps:,} "
                    f"| loss {avg_loss:.4f} "
                    f"| lr {lr:.2e} "
                    f"| {tok_sec/1000:.1f}K tok/s "
                    f"| {progress_pct:.1f}% done "
                    f"| ETA {eta_hours:.1f}h"
                )

                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr":   lr,
                        "train/tokens_processed": tokens_processed,
                        "train/tokens_per_sec":   tok_sec,
                        "train/step": step,
                    }, step=step)

                running_loss = 0.0
                t0 = t1

            # ── Validation ────────────────────────────────────────
            if step % tcfg["eval_every"] == 0:
                losses = estimate_loss(model, train_loader, val_loader, eval_batches=20, device=device)
                val_l  = losses["val"]
                print(f"\n[Eval] step {step:,} | train_loss {losses['train']:.4f} | val_loss {val_l:.4f}\n")

                if val_l < best_val_loss:
                    best_val_loss = val_l
                    # Save best model separately
                    save_checkpoint(
                        checkpoint_dir, step, model, optimizer,
                        tokens_processed, val_l,
                        model_cfg.to_dict(), tcfg, keep_last_n=1,
                    )
                    best_path = os.path.join(checkpoint_dir, "best.pt")
                    import shutil
                    shutil.copy(
                        os.path.join(checkpoint_dir, "latest.pt"),
                        best_path,
                    )
                    print(f"[Best] New best val_loss: {val_l:.4f} → saved to {best_path}")

                if use_wandb:
                    import wandb
                    wandb.log({"val/loss": val_l, "val/step": step}, step=step)

                model.train()

            # ── Sample generation ─────────────────────────────────
            if step % tcfg["sample_every"] == 0:
                tok_p = data_cfg.get("tokenizer_path", "tokenizer/tokenizer.json")
                _generate_sample(model, model_cfg, tcfg, device, tokenizer_path=tok_p)

            # ── Checkpoint ────────────────────────────────────────
            if step % tcfg["save_every"] == 0:
                save_checkpoint(
                    checkpoint_dir, step, model, optimizer,
                    tokens_processed, best_val_loss,
                    model_cfg.to_dict(), tcfg,
                    keep_last_n=tcfg.get("keep_last_n", 3),
                )

    except KeyboardInterrupt:
        print("\n[Interrupted] Saving checkpoint before exit...")
        save_checkpoint(
            checkpoint_dir, step, model, optimizer,
            tokens_processed, best_val_loss,
            model_cfg.to_dict(), tcfg,
        )
        print("Checkpoint saved. Training can be resumed.")

    # ── Final checkpoint ──────────────────────────────────────────
    if step >= total_steps:
        save_checkpoint(
            checkpoint_dir, step, model, optimizer,
            tokens_processed, best_val_loss,
            model_cfg.to_dict(), tcfg,
        )
        print(f"\n{'='*60}")
        print(f"  Training COMPLETE!")
        print(f"  Total steps: {step:,}")
        print(f"  Total tokens: {tokens_processed/1e9:.3f}B")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Checkpoints: {checkpoint_dir}/")
        print(f"{'='*60}\n")

    if use_wandb:
        import wandb
        wandb.finish()


# ── Sample generation helper ──────────────────────────────────────────────────

def _generate_sample(model, model_cfg, tcfg, device, tokenizer_path: str = "tokenizer/tokenizer.json") -> None:
    """Generate a sample to qualitatively check model quality."""
    model.eval()
    prompt_text = tcfg.get("sample_prompt", "def hello_world():")

    try:
        if os.path.exists(tokenizer_path):
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(tokenizer_path)
            prompt_ids = torch.tensor([tokenizer.encode(prompt_text).ids], dtype=torch.long, device=device)
            with torch.no_grad():
                output_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=40,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=tokenizer.token_to_id("<|endoftext|>") or 1,
                )
            generated_text = tokenizer.decode(output_ids[0].tolist())
            print(f"\n{'─'*50}")
            print(f"[Sample Output]\n{generated_text}")
            print(f"{'─'*50}\n")
        else:
            print(f"\n[Sample Prompt] {prompt_text}\n")
    except Exception as e:
        print(f"\n[Sample Generation Warning]: {e}\n")
    finally:
        model.train()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SE-LLM Pre-training")
    parser.add_argument(
        "--config", type=str, default="configs/350m.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        choices=["auto", "none"],
        help="Resume strategy: 'auto' (default) or 'none' (fresh start)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
