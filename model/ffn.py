"""
model/ffn.py — SwiGLU Feed-Forward Network.

SwiGLU is a gated linear unit variant that outperforms standard FFN.
Formula: FFN(x) = (SiLU(W_gate * x) * (W_up * x)) @ W_down

Used in: LLaMA, PaLM, Mistral, Qwen, Gemma — all modern LLMs.
No bias terms anywhere (modern practice).

Note on d_ff sizing:
  Standard FFN:  d_ff = 4 × d_model (e.g. 768 → 3072)
  SwiGLU FFN:    uses TWO matrices of size d_model × d_ff
                 gate projection + up projection
  So total params ≈ same as standard but with gating → better quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.config import ModelConfig


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network block.

    Architecture:
        gate = SiLU(x @ W_gate)      # Gating signal
        up   = x @ W_up              # Value signal
        out  = (gate * up) @ W_down  # Gated output

    Args:
        config: ModelConfig instance
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Gate projection: d_model → d_ff
        self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=False)

        # Up projection: d_model → d_ff
        self.w_up   = nn.Linear(config.d_model, config.d_ff, bias=False)

        # Down projection: d_ff → d_model
        self.w_down = nn.Linear(config.d_ff, config.d_model, bias=False)

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
        Returns:
            Output tensor  [batch, seq_len, d_model]
        """
        # Gate: apply SiLU activation
        gate = F.silu(self.w_gate(x))

        # Up: linear projection
        up = self.w_up(x)

        # Element-wise product (gating mechanism)
        hidden = gate * up

        # Down projection back to d_model
        return self.w_down(self.dropout(hidden))


if __name__ == "__main__":
    from model.config import config_1m_test
    cfg = config_1m_test()
    ffn = SwiGLUFFN(cfg)

    x = torch.randn(2, 64, cfg.d_model)
    out = ffn(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in ffn.parameters()):,}")
    print("SwiGLUFFN: OK")
