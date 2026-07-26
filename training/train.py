"""
training/train.py — Main Pre-training Loop.

This is the core training script. Designed for:
  - Kaggle TPU v5e-8 (128GB HBM, 8 chips) — primary training platform
  - Falls back to CUDA (GPU) or CPU automatically when TPU is not available
  - Auto-resume from checkpoint on every session restart
  - W&B logging (monitored from your local machine)
  - bfloat16 precision (native TPU dtype)

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
from torch.utils.data import DistributedSampler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import build_model
from training.dataset import build_dataloader, estimate_loss
from training.checkpoint import save_checkpoint, auto_resume
from training.lr_scheduler import get_lr, apply_lr

# ── XLA / TPU detection ───────────────────────────────────────────────────────
# We try to import PyTorch/XLA. If it is available we run on TPU, otherwise
# we fall back to CUDA/CPU gracefully.
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr
    import torch_xla
    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_full_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _is_master() -> bool:
    """True only on the master process (rank 0 / ordinal 0)."""
    if _XLA_AVAILABLE:
        return xm.is_master_ordinal()
    return True


def setup_wandb(cfg: dict, resume_step: int) -> bool:
    """Initialize W&B if project is set. Returns True if W&B is active."""
    if not _is_master():
        return False
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


def get_device_and_dtype(dtype_str: str):
    """
    Determine device, dtype, and autocast context.
    Priority: TPU (XLA) > CUDA > CPU
    """
    if _XLA_AVAILABLE:
        device = torch_xla.device()
        # TPU always uses bfloat16 — its dedicated hardware unit
        dtype  = torch.bfloat16
        # XLA handles mixed precision natively; no autocast context needed
        ctx    = nullcontext()
        return device, dtype, ctx

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        device = torch.device("cpu")
        dtype  = torch.float32
        ctx    = nullcontext()

    return device, dtype, ctx


# ── Core training function (runs on EACH TPU chip / GPU) ─────────────────────

def train_worker(index: int, args: argparse.Namespace) -> None:
    """
    This function is spawned once per accelerator core:
      - On TPU v5e-8: runs 8 parallel copies (one per chip)
      - On GPU:       runs once
      - On CPU:       runs once

    Args:
        index: Ordinal index of this worker (0–7 on TPU v5e-8)
        args:  Parsed command-line arguments
    """
    # ── 1. Load config ────────────────────────────────────────────
    cfg       = load_full_config(args.config)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    tcfg      = cfg["training"]

    if _is_master():
        print(f"\n{'='*60}")
        print(f"  SE-LLM Training: {model_cfg.name}")
        print(f"  Config: {args.config}")
        backend = f"TPU v5e-8 (8 chips via XLA)" if _XLA_AVAILABLE else \
                  ("CUDA" if torch.cuda.is_available() else "CPU")
        print(f"  Backend: {backend}")
        print(f"{'='*60}\n", flush=True)

    # ── 2. Device setup ───────────────────────────────────────────
    device, dtype, autocast_ctx = get_device_and_dtype(tcfg.get("dtype", "bfloat16"))

    if _is_master():
        print(f"Device: {device} | Dtype: {dtype}")

    # ── 3. Build model ────────────────────────────────────────────
    model = build_model(model_cfg).to(device)

    # ── 4. Build optimizer ────────────────────────────────────────
    decay_params    = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() <  2]

    # fused AdamW is CUDA-only — disabled on TPU
    use_fused = (not _XLA_AVAILABLE) and torch.cuda.is_available()

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": tcfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=tcfg["learning_rate"],
        betas=tuple(tcfg["betas"]),
        eps=tcfg.get("eps", 1e-8),
        fused=use_fused,
    )

    # GradScaler is only needed for float16 on CUDA (not bfloat16 / TPU)
    scaler = (
        torch.amp.GradScaler("cuda")
        if (not _XLA_AVAILABLE and torch.cuda.is_available() and dtype == torch.float16)
        else None
    )

    # ── 5. Auto-resume from checkpoint ───────────────────────────
    checkpoint_dir = tcfg["checkpoint_dir"]
    resume_mode    = args.resume or tcfg.get("resume", "auto")

    if resume_mode == "auto":
        start_step, tokens_processed, best_val_loss = auto_resume(
            checkpoint_dir, model, optimizer, device
        )
    else:
        start_step, tokens_processed, best_val_loss = 0, 0, float("inf")
        if _is_master():
            print("[Checkpoint] Starting fresh from step 0")

    # ── 6. Build data loaders ─────────────────────────────────────
    data_cfg   = cfg.get("data", {})
    train_path = tcfg.get("train_path") or data_cfg.get("train_path", "data/processed/train.bin")
    val_path   = tcfg.get("val_path")   or data_cfg.get("val_path",   "data/processed/val.bin")

    # DistributedSampler: assigns a unique non-overlapping slice of the
    # dataset to each TPU chip. chip 0 gets samples 0,8,16...
    #                              chip 1 gets samples 1,9,17...  etc.
    # This is what gives us the 8x data throughput on TPU v5e-8.
    if _XLA_AVAILABLE:
        world_size    = xr.world_size()   # 8 on TPU v5e-8
        train_sampler = DistributedSampler(
            # We pass a dummy dataset just to compute length —
            # the real dataset is inside build_dataloader
            torch.zeros(1),  # placeholder; length set below
            num_replicas=world_size,
            rank=index,
            shuffle=True,
            drop_last=True,
        )
    else:
        world_size    = 1
        train_sampler = None

    train_loader = build_dataloader(
        train_path,
        tcfg["sequence_length"],
        tcfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    val_loader = build_dataloader(
        val_path,
        tcfg["sequence_length"],
        tcfg["batch_size"],
        shuffle=False,
    )

    # Wrap DataLoaders with XLA's MpDeviceLoader so data is streamed to
    # the TPU chips in the background while the previous step is running.
    if _XLA_AVAILABLE:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader   = pl.MpDeviceLoader(val_loader,   device)

    # ── 7. Compute training steps ─────────────────────────────────
    # Effective batch per step = batch_size × world_size × seq_len
    tokens_per_step = (
        tcfg["batch_size"]
        * tcfg["gradient_accumulation"]
        * tcfg["sequence_length"]
        * world_size        # ← multiply by number of TPU chips
    )
    total_steps = tcfg["total_tokens"] // tokens_per_step

    if _is_master():
        print(f"World size:      {world_size} chips")
        print(f"Tokens per step: {tokens_per_step:,}")
        print(f"Total steps:     {total_steps:,}")
        print(f"Resume from:     step {start_step:,} ({tokens_processed/1e9:.3f}B tokens)")
        print(f"Remaining:       {total_steps - start_step:,} steps\n", flush=True)

    if start_step >= total_steps:
        if _is_master():
            print("Training already complete!")
        return

    # ── 8. Initialize W&B (master only) ──────────────────────────
    use_wandb = setup_wandb(cfg, start_step)

    # ── 9. Training loop ──────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    data_iter    = iter(train_loader)
    step         = start_step
    t0           = time.time()
    running_loss = 0.0

    if _is_master():
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
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)

                # MpDeviceLoader already moves data to the XLA device,
                # but for GPU/CPU we still need to move it manually.
                if not _XLA_AVAILABLE:
                    x, y = x.to(device), y.to(device)

                # Forward pass
                with autocast_ctx:
                    _, loss = model(x, y)
                    loss = loss / tcfg["gradient_accumulation"]

                accum_loss += loss.item()

                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # ── Optimizer step ────────────────────────────────────
            if scaler is not None:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])

            if _XLA_AVAILABLE:
                # xm.optimizer_step: all-reduces gradients across all 8
                # TPU chips before updating weights, keeping them in sync.
                xm.optimizer_step(optimizer)
            elif scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            step             += 1
            tokens_processed += tokens_per_step
            running_loss     += accum_loss

            # ── Logging (master only) ─────────────────────────────
            if step % tcfg["log_every"] == 0 and _is_master():
                t1      = time.time()
                dt      = t1 - t0
                tok_sec = (tcfg["log_every"] * tokens_per_step) / dt
                avg_loss = running_loss / tcfg["log_every"]

                progress_pct = 100 * tokens_processed / tcfg["total_tokens"]
                eta_hours    = (total_steps - step) * (dt / tcfg["log_every"]) / 3600

                print(
                    f"step {step:6,}/{total_steps:,} "
                    f"| loss {avg_loss:.4f} "
                    f"| lr {lr:.2e} "
                    f"| {tok_sec/1000:.1f}K tok/s "
                    f"| {progress_pct:.1f}% done "
                    f"| ETA {eta_hours:.1f}h",
                    flush=True
                )

                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss":            avg_loss,
                        "train/lr":              lr,
                        "train/tokens_processed": tokens_processed,
                        "train/tokens_per_sec":  tok_sec,
                        "train/step":            step,
                    }, step=step)

                running_loss = 0.0
                t0 = t1

            # ── Validation (master only) ──────────────────────────
            if step % tcfg["eval_every"] == 0 and _is_master():
                losses = estimate_loss(
                    model, train_loader, val_loader,
                    eval_batches=20, device=device,
                )
                val_l = losses["val"]
                print(f"\n[Eval] step {step:,} | train_loss {losses['train']:.4f} | val_loss {val_l:.4f}\n")

                if val_l < best_val_loss:
                    best_val_loss = val_l
                    save_checkpoint(
                        checkpoint_dir, step, model, optimizer,
                        tokens_processed, val_l,
                        model_cfg.to_dict(), tcfg, keep_last_n=1,
                    )
                    import shutil
                    best_path = os.path.join(checkpoint_dir, "best.pt")
                    shutil.copy(os.path.join(checkpoint_dir, "latest.pt"), best_path)
                    print(f"[Best] New best val_loss: {val_l:.4f} → saved to {best_path}")

                if use_wandb:
                    import wandb
                    wandb.log({"val/loss": val_l, "val/step": step}, step=step)

                model.train()

            # ── Sample generation (master only) ───────────────────
            if step % tcfg["sample_every"] == 0 and _is_master():
                tok_p = data_cfg.get("tokenizer_path", "tokenizer/tokenizer.json")
                _generate_sample(model, model_cfg, tcfg, device, tokenizer_path=tok_p)

            # ── Checkpoint ────────────────────────────────────────
            if step % tcfg["save_every"] == 0:
                # xm.save (inside save_checkpoint) is called on all ranks
                # but only the master rank writes to disk.
                save_checkpoint(
                    checkpoint_dir, step, model, optimizer,
                    tokens_processed, best_val_loss,
                    model_cfg.to_dict(), tcfg,
                    keep_last_n=tcfg.get("keep_last_n", 3),
                )

    except KeyboardInterrupt:
        if _is_master():
            print("\n[Interrupted] Saving checkpoint before exit...")
        save_checkpoint(
            checkpoint_dir, step, model, optimizer,
            tokens_processed, best_val_loss,
            model_cfg.to_dict(), tcfg,
        )
        if _is_master():
            print("Checkpoint saved. Training can be resumed.")

    # ── Final checkpoint ──────────────────────────────────────────
    if step >= total_steps:
        save_checkpoint(
            checkpoint_dir, step, model, optimizer,
            tokens_processed, best_val_loss,
            model_cfg.to_dict(), tcfg,
        )
        if _is_master():
            print(f"\n{'='*60}")
            print(f"  Training COMPLETE!")
            print(f"  Total steps:  {step:,}")
            print(f"  Total tokens: {tokens_processed/1e9:.3f}B")
            print(f"  Best val loss: {best_val_loss:.4f}")
            print(f"  Checkpoints:  {checkpoint_dir}/")
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
            tokenizer   = Tokenizer.from_file(tokenizer_path)
            prompt_ids  = torch.tensor([tokenizer.encode(prompt_text).ids], dtype=torch.long, device=device)
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

    if _XLA_AVAILABLE:
        # On TPU v5e-8: spawn 8 parallel worker processes (one per chip).
        # xmp.spawn automatically assigns each process to its own TPU chip.
        xmp.spawn(train_worker, args=(args,), nprocs=None, start_method="fork")
    else:
        # On GPU or CPU: run a single worker directly.
        train_worker(0, args)
