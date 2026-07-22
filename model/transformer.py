"""
model/transformer.py — Full SE-LLM Transformer Model.

Architecture:
    Token Embedding
    ↓
    N × TransformerBlock (with RoPE passed in)
    ↓
    RMSNorm (final)
    ↓
    Output projection → logits [vocab_size]

Features:
    - Weight tying: embedding and output head share weights
    - RoPE frequencies precomputed once, reused every forward pass
    - generate() method with temperature + top-k + top-p sampling
    - torch.compile() compatible
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from model.config import ModelConfig
from model.norm import RMSNorm
from model.block import TransformerBlock
from model.rope import precompute_rope_freqs


class Transformer(nn.Module):
    """
    Full decoder-only transformer language model.

    Args:
        config: ModelConfig instance defining all hyperparameters
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # ── Token embedding ───────────────────────────────────────
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # ── Transformer blocks ────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # ── Final normalization ───────────────────────────────────
        self.norm_out = RMSNorm(config.d_model, eps=config.norm_eps)

        # ── Output projection (logits) ────────────────────────────
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ── Weight tying ──────────────────────────────────────────
        # Share weights between embedding and output head
        # Reduces parameters by ~24M and improves training stability
        if config.tie_weights:
            self.lm_head.weight = self.embed.weight

        # ── Precompute RoPE frequencies ───────────────────────────
        # Register as buffer so it moves to GPU automatically with .to(device)
        cos, sin = precompute_rope_freqs(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # ── Initialize weights ────────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using scaled normal distribution."""
        std = 0.02

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Scale residual projections by 1/sqrt(2 * n_layers)
        # This prevents gradient explosion in deep networks (GPT-2 trick)
        scale = (2 * self.config.n_layers) ** -0.5
        for name, param in self.named_parameters():
            if "o_proj" in name or "w_down" in name:
                nn.init.normal_(param, mean=0.0, std=std * scale)

    def forward(
        self,
        input_ids: Tensor,
        targets: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Token indices [batch, seq_len]
            targets:   Target token indices [batch, seq_len] for loss computation
                       If None, only logits are returned (inference mode)

        Returns:
            (logits, loss)
            logits: [batch, seq_len, vocab_size]
            loss:   Scalar cross-entropy loss (None if targets not provided)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # ── Token embeddings ──────────────────────────────────────
        x = self.embed(input_ids)  # [B, T, d_model]

        # ── Pass through all transformer blocks ───────────────────
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)

        # ── Final norm ────────────────────────────────────────────
        x = self.norm_out(x)  # [B, T, d_model]

        # ── Compute logits ────────────────────────────────────────
        if targets is not None:
            # Training: compute logits over all positions
            logits = self.lm_head(x)  # [B, T, vocab_size]

            # Cross-entropy loss: flatten batch+seq dims
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),  # [B*T, vocab_size]
                targets.view(-1),                          # [B*T]
                ignore_index=-1,                           # ignore padding
            )
        else:
            # Inference: only compute logits for the last token (efficiency)
            logits = self.lm_head(x[:, -1:, :])  # [B, 1, vocab_size]
            loss = None

        return logits, loss

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Autoregressive token generation with temperature + top-k + top-p sampling.

        Args:
            input_ids:      Prompt token ids [1, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature:    Sampling temperature (lower = more deterministic)
            top_k:          Keep only top-k logits (0 = disabled)
            top_p:          Nucleus sampling threshold (1.0 = disabled)
            eos_token_id:   Stop generation when this token is produced

        Returns:
            Generated token ids [1, seq_len + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Truncate context if it exceeds max_seq_len
            ids = input_ids[:, -self.config.max_seq_len:]

            # Forward pass (inference mode — only last token logits)
            logits, _ = self(ids)           # [1, 1, vocab_size]
            logits = logits[:, -1, :]       # [1, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                kth_val   = torch.topk(logits, top_k_val).values[:, -1, None]
                logits    = logits.masked_fill(logits < kth_val, float("-inf"))

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens beyond nucleus
                sorted_logits[cumprobs - F.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
                # Scatter back to original order
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            # Sample from distribution
            probs     = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop at EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids


def build_model(config: ModelConfig) -> Transformer:
    """Build and return a new model from config."""
    model = Transformer(config)
    print(f"Model '{config.name}' built: {model.n_params / 1e6:.2f}M parameters")
    return model


if __name__ == "__main__":
    from model.config import config_1m_test, config_350m

    # Test tiny model on CPU
    print("=== Testing 1M test model ===")
    cfg = config_1m_test()
    model = build_model(cfg)

    B, T = 2, 64
    ids     = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(ids, targets)
    print(f"Input:   {ids.shape}")
    print(f"Logits:  {logits.shape}")
    print(f"Loss:    {loss.item():.4f} (should be ~log(32000) ≈ 10.37 at init)")

    # Test generation
    prompt = torch.randint(0, cfg.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")

    print("\n=== 350M model parameter count ===")
    cfg_350m = config_350m()
    model_350m = build_model(cfg_350m)
    print(f"Total: {model_350m.n_params / 1e6:.2f}M parameters")
