import os
import sys
import time
import json
import argparse
import shutil
import math
from contextlib import nullcontext

try:
    import wandb
except ImportError:
    wandb = None

import yaml
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import Transformer

SFT_MAX_LEN = 1024

class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning dataset from a JSONL file of ChatML conversations.
    """
    def __init__(self, data_path: str, tokenizer, max_seq_len: int = SFT_MAX_LEN) -> None:
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.samples     = []

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

        loss_mask = self._build_loss_mask(token_ids)

        pad_len   = self.max_seq_len - len(token_ids)
        token_ids = token_ids + [0] * pad_len
        loss_mask = loss_mask + [0] * pad_len

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        targets   = torch.tensor(token_ids[1:],  dtype=torch.long)
        mask      = torch.tensor(loss_mask[1:],  dtype=torch.bool)

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
        im_end_id   = self.tokenizer.token_to_id("<|im_end|>")
        asst_prefix = self.tokenizer.encode("<|im_start|>assistant\n").ids
        prefix_len  = len(asst_prefix)

        mask = [0] * len(token_ids)
        in_assistant = False

        i = 0
        while i < len(token_ids):
            if not in_assistant:
                if (i + prefix_len <= len(token_ids) and
                        token_ids[i:i + prefix_len] == asst_prefix):
                    in_assistant = True
                    i += prefix_len
                    continue
                i += 1
            else:
                mask[i] = 1
                if token_ids[i] == im_end_id:
                    in_assistant = False
                i += 1

        if sum(mask) == 0:
            return [1] * len(token_ids)

        return mask

def sft_collate_fn(batch: list) -> dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "targets":   torch.stack([b["targets"]   for b in batch]),
    }

def get_cosine_lr(step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def ddp_main(rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup(rank, world_size)
    is_master = (rank == 0)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg_dict = cfg["model"]
    sft_cfg        = cfg.get("sft", {})
    data_cfg       = cfg.get("data", {})

    learning_rate  = sft_cfg.get("learning_rate",  3e-5)
    min_lr         = sft_cfg.get("min_lr",          1e-6)
    warmup_steps   = sft_cfg.get("warmup_steps",    200)
    epochs         = sft_cfg.get("epochs",          12)
    batch_size     = sft_cfg.get("batch_size",      4)
    # T4 x2 means we need gradient accumulation to match the TPU's effective batch size of 32
    # 2 GPUs * 4 batch size * 4 accum = 32 effective batch size
    grad_accum_steps = args.gradient_accumulation
    grad_clip      = sft_cfg.get("grad_clip",       1.0)
    checkpoint_dir = sft_cfg.get("checkpoint_dir",  "checkpoints_sft")
    data_path      = sft_cfg.get("dataset",         "data/sft/sft_data.jsonl")
    log_every      = sft_cfg.get("log_every",       10)
    tok_path       = data_cfg.get("tokenizer_path", "tokenizer/tokenizer.json")
    
    if is_master:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if wandb is not None:
            wandb.init(project="aarohan-350m-sft-gpu", config=cfg)

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    tokenizer = Tokenizer.from_file(tok_path)
    dataset = SFTDataset(data_path, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=sft_collate_fn,
        num_workers=2,
        pin_memory=True
    )

    model_cfg = ModelConfig.from_dict(model_cfg_dict)
    model = Transformer(model_cfg)

    # ── Load Checkpoint ───────────────────────────────────────────────────────
    start_epoch = 0
    start_step = 0
    tokens_processed = 0

    if args.resume and os.path.exists(args.resume):
        if is_master: print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        start_step = ckpt.get("step", 0)
        tokens_processed = ckpt.get("tokens_processed", 0)
    elif args.base_checkpoint and os.path.exists(args.base_checkpoint):
        if is_master: print(f"Loading base model from {args.base_checkpoint}")
        ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    else:
        if is_master: print("WARNING: Starting from random weights!")

    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    if args.resume and os.path.exists(args.resume) and "ckpt" in dir() and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        # ensure optimizer tensors are on correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    # only delete ckpt if it was actually loaded
    if 'ckpt' in dir():
        del ckpt

    scaler = torch.amp.GradScaler('cuda')

    total_steps_per_epoch = len(loader) // grad_accum_steps
    total_steps = epochs * total_steps_per_epoch

    if is_master:
        print(f"\\n{'='*50}")
        print(f"Starting GPU SFT: {world_size}x T4")
        print(f"Epochs: {epochs} | Batch/GPU: {batch_size} | Accum: {grad_accum_steps}")
        print(f"Effective Batch: {batch_size * world_size * grad_accum_steps}")
        print(f"Steps per epoch: {total_steps_per_epoch} | Total: {total_steps}")
        print(f"Resuming from Epoch {start_epoch}, Step {start_step}")
        print(f"{'='*50}\\n")

    model.train()
    step = start_step

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        t0 = time.time()
        
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for i, batch in enumerate(loader):
            # Calculate absolute step to safely skip when resuming mid-epoch
            absolute_batch_idx = (epoch * len(loader)) + i
            if absolute_batch_idx < start_step * grad_accum_steps:
                continue

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-1
                )
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader):
                # Unscale for gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                lr = get_cosine_lr(step, total_steps, warmup_steps, learning_rate, min_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Sync loss across GPUs for logging
                loss_t = torch.tensor(accum_loss, device=device)
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
                avg_loss = loss_t.item()

                tokens_processed += (input_ids.numel() * world_size)
                
                if step % log_every == 0 and is_master:
                    dt = time.time() - t0
                    tok_sec = (input_ids.numel() * world_size * grad_accum_steps) / dt
                    print(f"Epoch {epoch+1}/{epochs} | Step {step:5d}/{total_steps} | Loss {avg_loss:5.4f} | LR {lr:.2e} | {tok_sec/1000:.1f}K tok/s")
                    
                    if wandb is not None:
                        wandb.log({
                            "loss": avg_loss,
                            "lr": lr,
                            "tokens_processed": tokens_processed,
                            "epoch": epoch
                        }, step=step)

                # Save checkpoint periodically
                if step > 0 and step % 1000 == 0 and is_master:
                    save_path = os.path.join(checkpoint_dir, f"sft_gpu_step_{step}.pt")
                    torch.save({
                        "step": step,
                        "epoch": epoch,
                        "model_state": model.module.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "model_config": model_cfg_dict,
                        "tokens_processed": tokens_processed,
                        "loss": avg_loss
                    }, save_path)
                    print(f"Saved {save_path}")
                
                step += 1
                accum_loss = 0.0
                t0 = time.time()

        if is_master:
            save_path = os.path.join(checkpoint_dir, f"sft_gpu_epoch_{epoch+1}.pt")
            torch.save({
                "step": step,
                "epoch": epoch + 1,
                "model_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": model_cfg_dict,
                "tokens_processed": tokens_processed
            }, save_path)
            
            # Update final pointer
            final_path = os.path.join(checkpoint_dir, "sft_final.pt")
            shutil.copy2(save_path, final_path)
            print(f"Epoch {epoch+1} complete. Checkpoint saved.")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base-checkpoint", type=str, default=None, help="Start from pre-trained weights")
    parser.add_argument("--resume", type=str, default=None, help="Resume from an SFT checkpoint")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Steps to accumulate gradients")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("Error: No CUDA GPUs found!")
        sys.exit(1)
        
    print(f"Found {world_size} GPUs. Launching DDP...")
    mp.spawn(ddp_main, args=(world_size, args), nprocs=world_size, join=True)
