"""
model/block.py — Single Transformer Block.

Structure (Pre-norm, used in all modern LLMs):
    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))

Pre-norm (norm before sublayer) is more stable than post-norm
(used in original Transformer). LLaMA, Mistral, etc. all use pre-norm.
"""

import torch
import torch.nn as nn
from torch import Tensor

from model.config import ModelConfig
from model.norm import RMSNorm
from model.attention import CausalSelfAttention
from model.ffn import SwiGLUFFN


class TransformerBlock(nn.Module):
    """
    Single decoder transformer block with pre-norm architecture.

    Components:
        - RMSNorm before attention
        - Causal self-attention with RoPE
        - RMSNorm before FFN
        - SwiGLU feed-forward network
        - Residual connections around both sublayers

    Args:
        config: ModelConfig instance
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Pre-attention norm
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)

        # Causal self-attention
        self.attn  = CausalSelfAttention(config)

        # Pre-FFN norm
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)

        # SwiGLU FFN
        self.ffn   = SwiGLUFFN(config)

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
        # Attention sublayer with residual connection
        x = x + self.attn(self.norm1(x), cos, sin)

        # FFN sublayer with residual connection
        x = x + self.ffn(self.norm2(x))

        return x


if __name__ == "__main__":
    from model.config import config_1m_test
    from model.rope import precompute_rope_freqs

    cfg = config_1m_test()
    block = TransformerBlock(cfg)

    cos, sin = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len)
    x = torch.randn(2, 64, cfg.d_model)
    out = block(x, cos, sin)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in block.parameters()):,}")
    print("TransformerBlock: OK")
