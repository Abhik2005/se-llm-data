"""
training/dataset.py — Memory-mapped binary dataset loader.

Loads pre-tokenized binary files (train.bin, val.bin) using
numpy memory-mapping so the OS handles paging — no need to
load the entire dataset into RAM.

File format:
    Binary file of uint16 token IDs packed sequentially.
    Each sequence is sequence_length tokens.
    Dataset provides (inputs, targets) pairs where:
        inputs  = tokens[i : i + seq_len]
        targets = tokens[i+1 : i + seq_len + 1]  (next-token prediction)
"""

import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    """
    Memory-mapped dataset of tokenized code sequences.

    Args:
        data_path:       Path to .bin file of uint16 token IDs
        sequence_length: Number of tokens per training sample
    """

    def __init__(self, data_path: str, sequence_length: int) -> None:
        assert os.path.exists(data_path), f"Data file not found: {data_path}"

        self.seq_len = sequence_length
        self.data_path = data_path

        # Memory-map the file — OS handles paging automatically
        # dtype=uint16: supports vocab sizes up to 65535
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")

        # Number of non-overlapping sequences
        # We need seq_len + 1 tokens per sample (for input + target)
        self.n_samples = (len(self.data) - 1) // sequence_length

        print(
            f"Dataset '{os.path.basename(data_path)}': "
            f"{len(self.data):,} tokens → {self.n_samples:,} samples"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Returns:
            (input_ids, targets) both of shape [sequence_length]
            targets is input_ids shifted by 1 (next-token prediction)
        """
        start = idx * self.seq_len
        # Read seq_len + 1 tokens
        chunk = torch.from_numpy(
            self.data[start : start + self.seq_len + 1].astype(np.int64)
        )
        input_ids = chunk[:-1]   # tokens 0..seq_len-1
        targets   = chunk[1:]    # tokens 1..seq_len   (shifted by 1)
        return input_ids, targets


def build_dataloader(
    data_path: str,
    sequence_length: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    sampler=None,
) -> DataLoader:
    """
    Build a DataLoader for the tokenized dataset.

    Args:
        data_path:       Path to .bin file
        sequence_length: Tokens per sequence
        batch_size:      Sequences per batch
        shuffle:         Whether to shuffle samples (ignored when sampler is set)
        num_workers:     Parallel data loading workers
        sampler:         Optional sampler (e.g. DistributedSampler for TPU)

    Returns:
        PyTorch DataLoader
    """
    dataset = TokenDataset(data_path, sequence_length)

    # sampler and shuffle are mutually exclusive in PyTorch
    if sampler is not None:
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,  # pin_memory is GPU-only — disabled for TPU/XLA
        drop_last=True,    # Consistent batch sizes across all chips
    )


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    eval_batches: int = 20,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """
    Estimate train and validation loss by averaging over multiple batches.

    Args:
        model:        The language model
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader
        eval_batches: Number of batches to average over
        device:       Compute device

    Returns:
        {"train": float, "val": float}
    """
    model.eval()
    losses = {}

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total_loss = 0.0
        for i, (x, y) in enumerate(loader):
            if i >= eval_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
        losses[split] = total_loss / min(eval_batches, len(loader))

    model.train()
    return losses


if __name__ == "__main__":
    import tempfile

    # Create a dummy binary file for testing
    print("Creating dummy dataset for testing...")
    n_tokens = 100_000
    seq_len  = 512

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        dummy = np.random.randint(0, 32000, size=n_tokens, dtype=np.uint16)
        dummy.tofile(f)
        tmp_path = f.name

    dataset = TokenDataset(tmp_path, seq_len)
    loader  = build_dataloader(tmp_path, seq_len, batch_size=4)

    x, y = next(iter(loader))
    print(f"Batch input shape:  {x.shape}")   # [4, 512]
    print(f"Batch target shape: {y.shape}")   # [4, 512]
    print(f"Targets = inputs shifted by 1: {torch.all(y[:, :-1] == x[:, 1:]).item()}")

    os.unlink(tmp_path)
    print("TokenDataset: OK")
