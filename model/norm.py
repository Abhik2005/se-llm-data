"""
model/norm.py — RMSNorm (Root Mean Square Normalization).

Faster than LayerNorm: skips mean subtraction step.
Used in: LLaMA, Mistral, Qwen, Gemma — all modern LLMs.

Formula: RMSNorm(x) = x / RMS(x) * weight
         RMS(x) = sqrt(mean(x^2) + eps)
"""

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Args:
        dim: Feature dimension to normalize over
        eps: Small value for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (no bias — that's the point)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        """Compute the RMS normalization without the scale."""
        # x: [..., dim]
        # Compute RMS over the last dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor [..., dim]
        Returns:
            Normalized tensor [..., dim]
        """
        # Cast to float32 for stable norm, then back to input dtype
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


if __name__ == "__main__":
    norm = RMSNorm(768)
    x = torch.randn(2, 16, 768)
    out = norm(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Params:       {sum(p.numel() for p in norm.parameters())}")
    print("RMSNorm: OK")
