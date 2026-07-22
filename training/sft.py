"""
training/sft.py — Supervised Fine-Tuning (Instruction Fine-Tuning).

Runs AFTER pre-training. Takes the pre-trained base model and fine-tunes
it on (instruction, response) pairs so it learns to follow commands.

Key difference from pre-training:
  - Loss is computed ONLY on assistant responses (not user messages)
  - Much shorter: 3 epochs over ~255K pairs (~4-6 hours on Kaggle P100)
  - Lower learning rate (1e-5 vs 3e-4 for pre-training)

Input format (ChatML):
    <|im_start|>system
    You are SE-LLM, an expert software engineering assistant.<|im_end|>
    <|im_start|>user
    Write a Python function to reverse a string.<|im_end|>
    <|im_start|>assistant
    def reverse_string(s: str) -> str:
        return s[::-1]<|im_end|>

Usage:
    python training/sft.py --config configs/350m.yaml --base-checkpoint checkpoints/pretrain_final.pt
"""

import os
import sys
import time
import json
import argparse
import yaml
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import Transformer
from training.checkpoint import save_checkpoint, load_checkpoint
from training.lr_scheduler import get_lr, apply_lr


# ── SFT Dataset ───────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning.
    Loads JSONL file of ChatML conversations.
    Computes loss mask so loss is only on assistant tokens.

    Args:
        data_path:      Path to JSONL file (one conversation per line)
        tokenizer:      Trained tokenizer with encode() method
        max_seq_len:    Maximum sequence length (truncate if longer)
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int) -> None:
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.samples     = []

        # Special token IDs
        self.im_start_id   = tokenizer.token_to_id("<|im_start|>")
        self.im_end_id     = tokenizer.token_to_id("<|im_end|>")
        self.assistant_id  = tokenizer.encode("assistant").ids[0]

        print(f"Loading SFT dataset from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    self.samples.append(item)
                except json.JSONDecodeError:
                    continue

        print(f"Loaded {len(self.samples):,} SFT samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        messages = sample.get("messages", [])

        # Build the full ChatML string
        text = self._format_chatml(messages)

        # Tokenize
        token_ids = self.tokenizer.encode(text).ids
        token_ids = token_ids[:self.max_seq_len]

        # Build loss mask (1 = compute loss, 0 = ignore)
        loss_mask = self._build_loss_mask(token_ids)

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        targets   = torch.tensor(token_ids[1:],  dtype=torch.long)
        mask      = torch.tensor(loss_mask[1:],  dtype=torch.bool)

        # Apply mask: set ignored positions to -1 (ignored by cross_entropy)
        targets[~mask] = -1

        return {"input_ids": input_ids, "targets": targets}

    def _format_chatml(self, messages: list) -> str:
        """Convert messages list to ChatML format string."""
        text = ""
        for msg in messages:
            role    = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return text

    def _build_loss_mask(self, token_ids: list) -> list:
        """
        Build a binary mask where 1 = compute loss (assistant tokens only).
        Scans for <|im_start|>assistant ... <|im_end|> spans.
        """
        mask       = [0] * len(token_ids)
        in_assist  = False

        i = 0
        while i < len(token_ids):
            if token_ids[i] == self.im_start_id:
                # Check if next meaningful token indicates "assistant"
                # Skip to see what role follows
                if i + 1 < len(token_ids):
                    # Check next few tokens for 'assistant' role marker
                    in_assist = False
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
    """Pad sequences to equal length within a batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids_list = []
    targets_list   = []

    for item in batch:
        seq_len  = item["input_ids"].shape[0]
        pad_len  = max_len - seq_len

        input_ids = torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        targets   = torch.cat([item["targets"],   torch.full((pad_len,), -1, dtype=torch.long)])

        input_ids_list.append(input_ids)
        targets_list.append(targets)

    return {
        "input_ids": torch.stack(input_ids_list),
        "targets":   torch.stack(targets_list),
    }


# ── SFT Training ──────────────────────────────────────────────────────────────

def run_sft(args: argparse.Namespace) -> None:
    # ── Load config ───────────────────────────────────────────────
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = ModelConfig.from_dict(cfg["model"])
    sft_cfg   = cfg.get("sft", {})

    # SFT-specific defaults
    learning_rate   = sft_cfg.get("learning_rate",      1e-5)
    min_lr          = sft_cfg.get("min_lr",              1e-6)
    warmup_steps    = sft_cfg.get("warmup_steps",        100)
    epochs          = sft_cfg.get("epochs",              3)
    batch_size      = sft_cfg.get("batch_size",          4)
    grad_accum      = sft_cfg.get("gradient_accumulation", 8)
    grad_clip       = sft_cfg.get("grad_clip",           1.0)
    checkpoint_dir  = sft_cfg.get("checkpoint_dir",      "checkpoints_sft")
    data_path       = sft_cfg.get("dataset",             "data/sft/sft_data.jsonl")
    tokenizer_path  = cfg["data"]["tokenizer_path"]

    # ── Device setup ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SFT Device: {device}")

    # ── Load tokenizer ────────────────────────────────────────────
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path.replace(".model", ".json"))

    # ── Build model and load pre-trained weights ──────────────────
    model = Transformer(model_cfg).to(device)

    base_ckpt = args.base_checkpoint or sft_cfg.get("base_model")
    if base_ckpt and os.path.exists(base_ckpt):
        print(f"Loading base model from: {base_ckpt}")
        load_checkpoint(base_ckpt, model, device=device)
    else:
        print("WARNING: No base checkpoint provided — SFT on random weights")

    # ── Optimizer (lower LR than pre-training) ────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        fused=device.type == "cuda",
    )

    # ── Dataset ───────────────────────────────────────────────────
    dataset = SFTDataset(data_path, tokenizer, model_cfg.max_seq_len)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sft_collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    total_steps   = (len(dataset) * epochs) // (batch_size * grad_accum)
    tokens_done   = 0

    print(f"\nSFT Training")
    print(f"  Samples:     {len(dataset):,}")
    print(f"  Epochs:      {epochs}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  LR:          {learning_rate}\n")

    # ── SFT training loop ─────────────────────────────────────────
    model.train()
    step = 0

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        for batch in loader:
            # LR schedule
            lr = get_lr(step, warmup_steps, total_steps, learning_rate, min_lr)
            apply_lr(optimizer, lr)

            optimizer.zero_grad()
            accum_loss = 0.0

            for micro in range(grad_accum):
                try:
                    b = next(iter(loader))
                except StopIteration:
                    break

                x = b["input_ids"].to(device)
                y = b["targets"].to(device)

                with autocast_ctx:
                    logits, _ = model(x)
                    # Compute loss manually so we can use the mask (-1 targets)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, model_cfg.vocab_size),
                        y.view(-1),
                        ignore_index=-1,
                    ) / grad_accum

                loss.backward()
                accum_loss += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1

            if step % 10 == 0:
                print(f"  SFT step {step:4,}/{total_steps:,} | loss {accum_loss:.4f} | lr {lr:.2e}")

    # ── Save final SFT model ──────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_path = os.path.join(checkpoint_dir, "sft_final.pt")
    save_checkpoint(
        checkpoint_dir, step, model, optimizer,
        tokens_done, 0.0,
        model_cfg.to_dict(), sft_cfg,
    )
    print(f"\nSFT Complete! Model saved to {checkpoint_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SE-LLM Instruction Fine-Tuning")
    parser.add_argument("--config",           type=str, default="configs/350m.yaml")
    parser.add_argument("--base-checkpoint",  type=str, default=None,
                        help="Path to pre-trained checkpoint to start SFT from")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sft(args)
