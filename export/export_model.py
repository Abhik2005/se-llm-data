"""
export/export_model.py — Export the trained model for API deployment.

Converts the training checkpoint into a clean inference-only format:
  - Strips optimizer state (not needed for inference)
  - Keeps only: model weights + config + tokenizer
  - Saves a compact bundle ready for uploading to your server

Usage:
    python export/export_model.py --checkpoint checkpoints_sft/sft_final.pt
    python export/export_model.py --checkpoint checkpoints/best.pt --output export/model_v1.0
"""

import os
import sys
import argparse
import shutil
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import Transformer


def export_model(checkpoint_path: str, output_dir: str, tokenizer_path: str) -> None:
    """
    Export model to a clean inference bundle.

    Output structure:
        output_dir/
        ├── model_weights.pt     ← weights only (no optimizer state)
        ├── config.json          ← model architecture config
        ├── tokenizer.json       ← tokenizer
        └── model_info.json      ← metadata (params, version, etc.)
    """
    print(f"\n{'='*55}")
    print(f"  SE-LLM Model Export")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output:     {output_dir}/")
    print(f"{'='*55}\n")

    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    os.makedirs(output_dir, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────
    print("Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_cfg_dict = ckpt.get("model_config")
    assert model_cfg_dict, "Checkpoint does not contain model_config — cannot export"

    model_cfg = ModelConfig.from_dict(model_cfg_dict)

    # ── Rebuild model and load weights ────────────────────────────
    model = Transformer(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_params = model.n_params
    print(f"Model: {model_cfg.name} | {n_params/1e6:.2f}M parameters")

    # ── Save inference-only weights ───────────────────────────────
    weights_path = os.path.join(output_dir, "model_weights.pt")
    torch.save({"model_state": ckpt["model_state"]}, weights_path)
    size_mb = os.path.getsize(weights_path) / 1e6
    print(f"Weights saved: {weights_path} ({size_mb:.0f} MB)")

    # ── Save config ───────────────────────────────────────────────
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(model_cfg_dict, f, indent=2)
    print(f"Config saved:  {config_path}")

    # ── Copy tokenizer ────────────────────────────────────────────
    tok_dest = os.path.join(output_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        shutil.copy(tokenizer_path, tok_dest)
        print(f"Tokenizer saved: {tok_dest}")
    else:
        print(f"Warning: Tokenizer not found at {tokenizer_path}")

    # ── Save model info ───────────────────────────────────────────
    info = {
        "model_name":        model_cfg.name,
        "n_params":          n_params,
        "n_params_millions": round(n_params / 1e6, 2),
        "architecture":      "decoder-only-transformer",
        "n_layers":          model_cfg.n_layers,
        "d_model":           model_cfg.d_model,
        "n_heads":           model_cfg.n_heads,
        "d_ff":              model_cfg.d_ff,
        "vocab_size":        model_cfg.vocab_size,
        "max_seq_len":       model_cfg.max_seq_len,
        "training_tokens":   ckpt.get("tokens_processed", 0),
        "training_steps":    ckpt.get("step", 0),
        "val_loss":          ckpt.get("val_loss", None),
        "source_checkpoint": checkpoint_path,
    }
    info_path = os.path.join(output_dir, "model_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Info saved:    {info_path}")

    # ── Summary ───────────────────────────────────────────────────
    total_size_mb = sum(
        os.path.getsize(os.path.join(output_dir, f)) / 1e6
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )

    print(f"\n{'='*55}")
    print(f"  Export COMPLETE")
    print(f"  Output dir:   {output_dir}/")
    print(f"  Total size:   {total_size_mb:.0f} MB")
    print(f"  Parameters:   {n_params/1e6:.2f}M")
    print(f"\n  Upload '{output_dir}/' to your inference server")
    print(f"  Set env: MODEL_CHECKPOINT_PATH={output_dir}/model_weights.pt")
    print(f"           TOKENIZER_PATH={output_dir}/tokenizer.json")
    print(f"{'='*55}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SE-LLM for deployment")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to training checkpoint (.pt file)")
    parser.add_argument("--output",      type=str, default="export/model_v1",
                        help="Output directory (default: export/model_v1)")
    parser.add_argument("--tokenizer",   type=str, default="tokenizer/tokenizer.json",
                        help="Path to tokenizer.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_model(args.checkpoint, args.output, args.tokenizer)
