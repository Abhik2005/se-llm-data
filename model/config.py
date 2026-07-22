"""
model/config.py — Model configuration dataclass.
All hyperparameters for SE-LLM-350M are defined here.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    # ── Architecture ──────────────────────────────────────
    name: str = "se-llm-350m"
    n_layers: int = 12           # Number of transformer blocks
    d_model: int = 768           # Hidden / embedding dimension
    n_heads: int = 12            # Number of attention heads
    d_ff: int = 3072             # FFN intermediate size (4 × d_model)
    vocab_size: int = 32000      # Tokenizer vocabulary size
    max_seq_len: int = 2048      # Maximum context window

    # ── Modern components ────────────────────────────────
    rope_base: int = 10000       # RoPE theta base frequency
    norm_eps: float = 1e-5       # RMSNorm epsilon
    dropout: float = 0.0         # Dropout (0 = disabled, modern practice)
    tie_weights: bool = True     # Tie input embedding & output projection

    # ── Special tokens ───────────────────────────────────
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 1

    # ── Derived properties ───────────────────────────────
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        return self.d_model // self.n_heads

    @property
    def n_params(self) -> int:
        """Approximate parameter count (excluding tied weights double-count)."""
        embed   = self.vocab_size * self.d_model
        attn    = self.n_layers * (4 * self.d_model * self.d_model)  # Q, K, V, O
        # SwiGLU FFN has 3 matrices: gate, up, down
        ffn     = self.n_layers * (3 * self.d_model * self.d_ff)
        norms   = self.n_layers * 2 * self.d_model                  # pre-attn + pre-ffn
        out_norm = self.d_model
        # Output head shares weights with embed if tie_weights=True
        out_head = 0 if self.tie_weights else self.vocab_size * self.d_model
        return embed + attn + ffn + norms + out_norm + out_head

    def __post_init__(self):
        _ = self.head_dim  # Validate divisibility at init

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        model_data = data.get("model", data)  # support both nested and flat
        # Only pass keys that exist in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in model_data.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Serialize config to dictionary (for saving with checkpoint)."""
        return {
            "name": self.name,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "rope_base": self.rope_base,
            "norm_eps": self.norm_eps,
            "dropout": self.dropout,
            "tie_weights": self.tie_weights,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Deserialize config from dictionary (for loading from checkpoint)."""
        valid_keys = {f for f in cls.__dataclass_fields__}  # type: ignore
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

# ── Preset configs ────────────────────────────────────────────

def config_350m() -> ModelConfig:
    """350M model — primary target for Kaggle P100 training."""
    return ModelConfig(
        name="se-llm-350m",
        n_layers=24, d_model=1024, n_heads=16,
        d_ff=3072, vocab_size=32000, max_seq_len=2048,
    )


def config_125m() -> ModelConfig:
    """125M model — smaller/faster option."""
    return ModelConfig(
        name="se-llm-125m",
        n_layers=12, d_model=768, n_heads=12,
        d_ff=3072, vocab_size=32000, max_seq_len=2048,
    )


def config_1m_test() -> ModelConfig:
    """Tiny 1M model — for local CPU pipeline testing."""
    return ModelConfig(
        name="se-llm-1m-test",
        n_layers=4, d_model=128, n_heads=4,
        d_ff=512, vocab_size=32000, max_seq_len=512,
    )


if __name__ == "__main__":
    for cfg_fn in [config_350m, config_125m, config_1m_test]:
        cfg = cfg_fn()
        print(f"{cfg.name:<20} {cfg.n_params/1e6:.1f}M params | "
              f"layers={cfg.n_layers} d_model={cfg.d_model}")
    print(f"Config:     {cfg.to_dict()}")
