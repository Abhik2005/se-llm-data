"""
model/attention.py — Causal Multi-Head Self-Attention with RoPE.

Features:
  - Causal masking (decoder-only: each token sees only past tokens)
  - Rotary Position Embeddings (RoPE) on Q and K
  - No bias terms (modern practice)
  - Flash Attention support via F.scaled_dot_product_attention
    (automatically uses Flash Attention if available on GPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.config import ModelConfig
from model.rope import precompute_rope_freqs, apply_rope


class CausalSelfAttention(nn.Module):
    """
    Causal multi-head self-attention with RoPE position embeddings.

    Uses torch's scaled_dot_product_attention which automatically
    dispatches to Flash Attention when running on compatible GPU.

    Args:
        config: ModelConfig instance
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config    = config
        self.n_heads   = config.n_heads
        self.head_dim  = config.head_dim
        self.d_model   = config.d_model

        # Single fused QKV projection (3 × d_model output)
        # No bias — standard modern practice
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # Output projection
        self.o_proj   = nn.Linear(config.d_model, config.d_model, bias=False)

        # Dropout (applied to attention weights if dropout > 0)
        self.dropout  = config.dropout

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """
        Args:
            x:   Input tensor [batch, seq_len, d_model]
            cos: RoPE cosines [max_seq_len, head_dim]
            sin: RoPE sines   [max_seq_len, head_dim]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        B, T, C = x.shape  # batch, seq_len, d_model

        # ── 1. Compute Q, K, V via fused projection ───────────────
        # [B, T, 3*d_model] → split into 3 × [B, T, d_model]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # ── 2. Reshape to multi-head format ───────────────────────
        # [B, T, d_model] → [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # ── 3. Apply RoPE to Q and K ──────────────────────────────
        q, k = apply_rope(q, k, cos, sin)

        # ── 4. Scaled dot-product attention (causal) ──────────────
        # is_causal=True adds the causal mask automatically
        # Uses Flash Attention on GPU if torch >= 2.0 + CUDA available
        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )
        # attn_out: [B, n_heads, T, head_dim]

        # ── 5. Merge heads and project output ─────────────────────
        # [B, n_heads, T, head_dim] → [B, T, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_out)


if __name__ == "__main__":
    from model.config import config_1m_test
    cfg = config_1m_test()
    attn = CausalSelfAttention(cfg)

    cos, sin = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len)
    x = torch.randn(2, 64, cfg.d_model)
    out = attn(x, cos, sin)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in attn.parameters()):,}")
    print("CausalSelfAttention: OK")
