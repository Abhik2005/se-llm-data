"""
model/rope.py — Rotary Position Embeddings (RoPE).

RoPE encodes position by rotating Q and K vectors using
complex-valued frequencies. Benefits over learned positions:
  - Better generalization to longer sequences
  - No extra parameters
  - Used in: LLaMA, Mistral, Qwen, all modern models
"""

import torch
from torch import Tensor


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    base: int = 10000,
    device: torch.device = torch.device("cpu"),
) -> tuple[Tensor, Tensor]:
    """
    Precompute RoPE cosine and sine frequency matrices.

    Args:
        head_dim:    Dimension per attention head (must be even)
        max_seq_len: Maximum sequence length to precompute for
        base:        RoPE theta base (default 10000)
        device:      Target device

    Returns:
        cos, sin: Both shape [max_seq_len, head_dim]
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

    # Compute inverse frequencies: θ_i = 1 / (base^(2i / head_dim))
    # Shape: [head_dim // 2]
    half_dim = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim)
    )

    # Position indices: [max_seq_len]
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)

    # Outer product → [max_seq_len, head_dim // 2]
    freqs = torch.outer(positions, inv_freq)

    # Concatenate to get [max_seq_len, head_dim]
    freqs = torch.cat([freqs, freqs], dim=-1)

    return freqs.cos(), freqs.sin()


def rotate_half(x: Tensor) -> Tensor:
    """
    Rotate the second half of the last dimension to the first half (negated).
    Used to apply the rotation in apply_rope.

    Input:  [..., head_dim]  where head_dim = 2 * half_dim
    Output: [..., head_dim]
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]          # first half
    x2 = x[..., half:]          # second half
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.

    Args:
        q:   Query tensor  [batch, n_heads, seq_len, head_dim]
        k:   Key tensor    [batch, n_heads, seq_len, head_dim]
        cos: Precomputed cosines [seq_len, head_dim]
        sin: Precomputed sines   [seq_len, head_dim]

    Returns:
        q_rot, k_rot: Rotated Q and K, same shapes as input
    """
    seq_len = q.shape[2]

    # Slice to match current sequence length and add batch + head dims
    # cos/sin: [1, 1, seq_len, head_dim]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # Apply rotation: q_rot = q * cos + rotate_half(q) * sin
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot


if __name__ == "__main__":
    # Quick sanity check
    head_dim = 64
    max_seq = 2048
    cos, sin = precompute_rope_freqs(head_dim, max_seq)
    print(f"cos shape: {cos.shape}")  # [2048, 64]
    print(f"sin shape: {sin.shape}")  # [2048, 64]

    # Test apply_rope
    B, H, T, D = 2, 12, 128, 64
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    print(f"q_rot shape: {q_rot.shape}")  # [2, 12, 128, 64]
    print("RoPE: OK")
