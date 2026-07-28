"""
training/sft.py — Supervised Fine-Tuning (Instruction Fine-Tuning) on TPU v5e-8.

Runs AFTER pre-training. Takes the pre-trained Aarohan-350M base model and
fine-tunes it on (instruction, response) pairs using xmp.spawn across all
8 TPU chips — the same multi-chip pattern used by train.py.

Key design decisions:
  - xmp.spawn with nprocs=None (auto-detects 8 chips on v5e-8)
  - gradient_accumulation: 1 (prevents XLA graph unrolling OOM — lesson learned!)
  - Loss computed ONLY on assistant tokens (loss_mask=-1 for user/system tokens)
  - Sequences padded to SFT_MAX_LEN=1024 (SFT samples are shorter than pre-training)
  - bfloat16 precision (native TPU dtype)
  - All-rank checkpoint saving (master writes to disk) — same as train.py

Fixes vs old GPU-only sft.py:
  BUG-1 GRADIENT LOOP — old sft.py called next(iter(loader)) inside the
                         accumulation loop, creating a NEW iterator each micro-step.
                         It re-sampled random batches instead of advancing the dataset.
                         Fix: removed gradient accumulation (TPU must use 1 anyway).
  BUG-2 DEVICE MISMATCH — old sft.py used torch.device("cuda"/"cpu"), missing TPU.
                           Fix: uses same get_device_and_dtype() pattern as train.py.
  BUG-3 FIXED PAD LEN — old sft.py used dynamic padding (variable-length batches).
                          XLA requires static shapes for graph compilation.
                          Fix: pad all sequences to a fixed SFT_MAX_LEN.

Usage (Kaggle TPU v5e-8):
    python training/sft.py --config configs/350m.yaml \\
                           --base-checkpoint checkpoints/best.pt
"""

import os
import sys
import time
import json
import argparse
import shutil
import yaml
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import build_model
from training.checkpoint import save_checkpoint, load_checkpoint
from training.lr_scheduler import get_lr, apply_lr

# ── XLA / TPU detection ───────────────────────────────────────────────────────
# Same environment fix as train.py — clears broken Kaggle TPU env vars.
os.environ.pop('CLOUD_TPU_TASK_ID', None)
os.environ.pop('TPU_PROCESS_ADDRESSES', None)

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr
    import torch_xla
    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False


# ── SFT sequence length ───────────────────────────────────────────────────────
# Instruction samples are shorter than pre-training (avg ~300-600 tokens).
# Using 1024 instead of 2048 halves memory usage and doubles throughput.
SFT_MAX_LEN = 1024


# ── Helpers (mirrors train.py) ────────────────────────────────────────────────

def _is_master() -> bool:
    if _XLA_AVAILABLE:
        return xm.is_master_ordinal()
    return True


def get_device_and_dtype():
    """
    Determine device and dtype.
    Priority: TPU (XLA) > CUDA > CPU
    """
    if _XLA_AVAILABLE:
        device = torch_xla.device()
        dtype  = torch.bfloat16
        ctx    = nullcontext()
        return device, dtype, ctx

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx    = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        device = torch.device("cpu")
        dtype  = torch.float32
        ctx    = nullcontext()

    return device, dtype, ctx


# ── SFT Dataset ───────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning dataset from a JSONL file of ChatML conversations.

    Key design for XLA:
    - All sequences are padded to exactly SFT_MAX_LEN tokens (static shape).
    - Loss mask: targets are -1 for system/user tokens (ignored by cross_entropy).
    - Loss is computed ONLY on assistant response tokens.

    Each JSONL line must have the format:
        {"messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = SFT_MAX_LEN) -> None:
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.samples     = []

        # Special token IDs
        self.im_start_id  = tokenizer.token_to_id("<|im_start|>") or 0
        self.im_end_id    = tokenizer.token_to_id("<|im_end|>")   or 0
        assistant_enc     = tokenizer.encode("assistant").ids
        self.assistant_id = assistant_enc[0] if assistant_enc else -1

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        messages  = self.samples[idx].get("messages", [])
        text      = self._format_chatml(messages)
        token_ids = self.tokenizer.encode(text).ids[:self.max_seq_len]

        # Build loss mask — 1 for assistant tokens only
        loss_mask = self._build_loss_mask(token_ids)

        # Pad to fixed length (XLA needs static shapes)
        pad_len   = self.max_seq_len - len(token_ids)
        token_ids = token_ids + [0] * pad_len
        loss_mask = loss_mask + [0] * pad_len

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        targets   = torch.tensor(token_ids[1:],  dtype=torch.long)
        mask      = torch.tensor(loss_mask[1:],  dtype=torch.bool)

        # Set non-assistant positions to -1 (ignored by F.cross_entropy)
        targets[~mask] = -1

        return {"input_ids": input_ids, "targets": targets}

    def _format_chatml(self, messages: list) -> str:
        text = ""
        for msg in messages:
            role    = msg.get("role", "user")
            content = msg.get("content", "")
            text   += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return text

    def _build_loss_mask(self, token_ids: list) -> list:
        """Return binary list: 1 = compute loss (assistant tokens only)."""
        mask      = [0] * len(token_ids)
        in_assist = False

        i = 0
        while i < len(token_ids):
            if token_ids[i] == self.im_start_id:
                in_assist = False
                # Peek ahead to check if role == "assistant"
                for j in range(i + 1, min(i + 5, len(token_ids))):
                    if token_ids[j] == self.assistant_id:
                        in_assist = True
                        break
            elif token_ids[i] == self.im_end_id:
                in_assist = False
            elif in_assist:
                mask[i] = 1
            i += 1

        return mask


def sft_collate_fn(batch: list) -> dict:
    """Stack pre-padded fixed-length tensors — no dynamic padding needed."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "targets":   torch.stack([b["targets"]   for b in batch]),
    }


# ── Per-chip SFT worker ───────────────────────────────────────────────────────

def sft_worker(index: int, args: argparse.Namespace) -> None:
    """
    One copy of this function runs on EACH TPU chip (indices 0-7).
    Mirrors the structure of train_worker() in train.py exactly.
    """
    # ── 1. Load config ────────────────────────────────────────────
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg_dict = cfg["model"]
    sft_cfg        = cfg.get("sft", {})
    data_cfg       = cfg.get("data", {})

    learning_rate  = sft_cfg.get("learning_rate",  1e-5)
    min_lr         = sft_cfg.get("min_lr",          1e-6)
    warmup_steps   = sft_cfg.get("warmup_steps",    100)
    epochs         = sft_cfg.get("epochs",          3)
    batch_size     = sft_cfg.get("batch_size",      4)
    grad_clip      = sft_cfg.get("grad_clip",       1.0)
    checkpoint_dir = sft_cfg.get("checkpoint_dir",  "checkpoints_sft")
    data_path      = sft_cfg.get("dataset",         "data/sft/sft_data.jsonl")
    log_every      = sft_cfg.get("log_every",       10)
    tok_path       = data_cfg.get("tokenizer_path", "tokenizer/tokenizer.json")
    base_ckpt      = args.base_checkpoint

    model_cfg  = ModelConfig.from_dict(model_cfg_dict)

    # ── 2. Device and dtype ───────────────────────────────────────
    device, dtype, autocast_ctx = get_device_and_dtype()

    if _is_master():
        backend = "TPU v5e-8 (8 chips via XLA)" if _XLA_AVAILABLE else \
                  ("CUDA" if torch.cuda.is_available() else "CPU")
        print(f"\n{'='*55}")
        print(f"  Aarohan-350M — Instruction Fine-Tuning (SFT)")
        print(f"  Config:  {args.config}")
        print(f"  Backend: {backend}")
        print(f"  Dtype:   {dtype}")
        print(f"{'='*55}\n")

    # ── 3. Tokenizer ──────────────────────────────────────────────
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tok_path)

    # ── 4. Dataset — shard across chips ──────────────────────────
    # Each chip loads the full dataset but only trains on its shard.
    # Chip k processes samples k, k+world_size, k+2*world_size, ...
    full_dataset = SFTDataset(data_path, tokenizer, SFT_MAX_LEN)

    if _XLA_AVAILABLE:
        world_size = xr.world_size()  # 8 on v5e-8
    else:
        world_size = 1

    shard_indices = list(range(index, len(full_dataset), world_size))
    shard         = torch.utils.data.Subset(full_dataset, shard_indices)

    loader = DataLoader(
        shard,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sft_collate_fn,
        num_workers=0,
        drop_last=True,
    )

    # Wrap with MpDeviceLoader for background prefetch to TPU
    if _XLA_AVAILABLE:
        loader = pl.MpDeviceLoader(loader, device)

    steps_per_epoch = len(shard) // batch_size
    total_steps     = steps_per_epoch * epochs
    tokens_per_step = batch_size * world_size * SFT_MAX_LEN

    if _is_master():
        print(f"  Chips:           {world_size}")
        print(f"  Batch / chip:    {batch_size}")
        print(f"  Samples / chip:  {len(shard):,}")
        print(f"  Steps / epoch:   {steps_per_epoch:,}")
        print(f"  Total steps:     {total_steps:,}")
        print(f"  Total epochs:    {epochs}")
        print(f"  LR:              {learning_rate} → {min_lr} (cosine)")
        print(f"  Warmup steps:    {warmup_steps}\n")

    # ── 5. Build model in bfloat16 ────────────────────────────────
    model = build_model(model_cfg).to(device=device, dtype=dtype)

    # ── 6. Load pre-trained weights ───────────────────────────────
    if base_ckpt and os.path.exists(base_ckpt):
        if _is_master():
            print(f"Loading base model from: {base_ckpt}")
        ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
        # Load state dict — cast to bfloat16 to match model dtype
        state = {k: v.to(dtype) for k, v in ckpt["model_state"].items()}
        model.load_state_dict(state)
        if _is_master():
            print(f"Loaded pre-trained weights from step {ckpt.get('step', 0):,}")
            print(f"Pre-train val_loss was {ckpt.get('val_loss', 0):.4f}\n")
    elif _is_master():
        print("WARNING: No base checkpoint provided — SFT from random weights!\n")

    # ── 7. Optimizer ──────────────────────────────────────────────
    # Lower LR than pre-training to avoid catastrophic forgetting.
    # fused AdamW is CUDA-only, disabled on TPU.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=False,
    )

    # ── 8. Training loop ──────────────────────────────────────────
    model.train()
    step         = 0
    data_iter    = iter(loader)
    t0           = time.time()
    running_loss = 0.0

    if _is_master():
        print("Starting SFT training...\n")

    for epoch in range(epochs):
        if _is_master():
            print(f"\n─── Epoch {epoch + 1}/{epochs} ───")

        data_iter = iter(loader)

        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                break  # epoch done

            # ── LR schedule ───────────────────────────────────────
            lr = get_lr(step, warmup_steps, total_steps, learning_rate, min_lr)
            apply_lr(optimizer, lr)

            # MpDeviceLoader already moves to TPU; manual move for GPU/CPU
            if not _XLA_AVAILABLE:
                batch = {k: v.to(device) for k, v in batch.items()}

            x = batch["input_ids"]  # [B, SFT_MAX_LEN - 1]
            y = batch["targets"]    # [B, SFT_MAX_LEN - 1]  (-1 = ignored)

            optimizer.zero_grad(set_to_none=True)

            # ── Forward ───────────────────────────────────────────
            # Pass targets directly to the model — it computes full-sequence
            # logits internally and uses ignore_index=-1 to skip non-assistant
            # tokens (system/user). This avoids the [B,1,V] vs [B,T,V] mismatch
            # that happens when calling model(x) without targets (inference mode).
            with autocast_ctx:
                _, loss = model(x, y)   # model handles ignore_index=-1 internally

            # ── Backward ──────────────────────────────────────────
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # ── Optimizer step (all-reduces grads across all chips) ─
            if _XLA_AVAILABLE:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()

            step         += 1
            running_loss += loss.detach().item()

            # ── Logging (master only) ─────────────────────────────
            if step % log_every == 0 and _is_master():
                t1       = time.time()
                dt       = t1 - t0
                avg_loss = running_loss / log_every
                tok_s    = (log_every * tokens_per_step) / dt
                pct      = 100.0 * step / total_steps

                print(
                    f"step {step:5,}/{total_steps:,} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"{tok_s/1e3:.1f}K tok/s | "
                    f"{pct:.1f}% done",
                    flush=True,
                )

                running_loss = 0.0
                t0 = t1

    # ── 9. Save SFT model (all ranks call — master writes to disk) ─
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_checkpoint(
        checkpoint_dir, step, model, optimizer,
        step * tokens_per_step, 0.0,
        model_cfg.to_dict(), sft_cfg,
    )

    if _is_master():
        # Copy latest.pt → sft_final.pt for easy identification
        src = os.path.join(checkpoint_dir, "latest.pt")
        dst = os.path.join(checkpoint_dir, "sft_final.pt")
        if os.path.exists(src):
            shutil.copy(src, dst)

        print(f"\n{'='*55}")
        print(f"  ✅ Aarohan-350M SFT COMPLETE!")
        print(f"  Total steps:  {step:,}")
        print(f"  Checkpoints:  {checkpoint_dir}/")
        print(f"  Chat model:   {checkpoint_dir}/sft_final.pt")
        print(f"  Test locally:")
        print(f"    python evaluation/generate.py \\")
        print(f"      --checkpoint {checkpoint_dir}/sft_final.pt \\")
        print(f"      --mode chat")
        print(f"{'='*55}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aarohan-350M Instruction Fine-Tuning (SFT)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/350m.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--base-checkpoint", type=str, default=None,
        help="Path to pre-trained checkpoint (best.pt from pre-training)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if _XLA_AVAILABLE:
        # TPU v5e-8: spawn one worker per chip (nprocs=None = auto-detect)
        # start_method="spawn" required on Kaggle PJRT to prevent C++ crashes
        xmp.spawn(sft_worker, args=(args,), nprocs=None, start_method="spawn")
    else:
        # GPU or CPU: single worker
        sft_worker(0, args)
