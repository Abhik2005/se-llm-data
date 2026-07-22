# SE-LLM-350M

A 350M parameter instruction-following language model specialized for **software engineering tasks** — built from scratch.

---

## Project Structure

```
se-llm-350m/
├── api/                 # FastAPI REST API serving layer
│   ├── auth.py          # JWT & API key authentication
│   ├── database.py      # Database connection & session management
│   ├── main.py          # FastAPI application & REST endpoints
│   └── rate_limit.py    # Rate-limiting middleware
├── configs/             # Training & model configurations
│   ├── 125m.yaml        # 125M preset config
│   ├── 1m_test.yaml     # Tiny model for local CPU testing
│   └── 350m.yaml        # Full 350M model configuration
├── data/                # Data preparation & processing pipeline
│   ├── prepare_data.py  # Download, tokenize & save pre-training data
│   ├── sft_data.py      # Prepare ChatML instruction dataset
│   └── processed/       # Memory-mapped binary dataset files (.bin)
├── evaluation/          # Benchmarking & generation CLI
│   ├── generate.py      # Interactive chat & completion interface
│   └── humaneval.py     # HumanEval benchmark evaluation suite
├── export/              # Model exporting utilities
│   └── export_model.py  # Export weights & config for serving/inference
├── model/               # Model architecture (PyTorch)
│   ├── config.py        # ModelConfig dataclass
│   ├── rope.py          # Rotary Position Embeddings (RoPE)
│   ├── norm.py          # RMSNorm
│   ├── attention.py     # Causal Multi-Head Attention with KV-cache
│   ├── ffn.py           # SwiGLU Feed-Forward Network
│   ├── block.py         # Transformer Block
│   └── transformer.py   # Full transformer model & generate() loop
├── notebooks/           # Kaggle training & evaluation notebooks
│   ├── kaggle_eval.ipynb
│   ├── kaggle_pretrain.ipynb
│   └── kaggle_sft.ipynb
├── tokenizer/           # BPE Tokenizer training & testing
│   ├── test_tokenizer.py
│   ├── tokenizer.json
│   └── train_tokenizer.py
├── training/            # Pre-training & fine-tuning infrastructure
│   ├── checkpoint.py    # Checkpoint saving, loading & auto-resume
│   ├── dataset.py       # Memory-mapped data loader
│   ├── lr_scheduler.py  # Cosine LR scheduler with warmup
│   ├── sft.py           # Supervised Fine-Tuning execution script
│   └── train.py         # Pre-training loop (main)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the tokenizer (runs on CPU, ~2 hours)
```bash
python tokenizer/train_tokenizer.py
```

### 3. Prepare training data (runs on CPU overnight)
```bash
python data/prepare_data.py
```

### 4. Test the pipeline locally with a tiny model
```bash
python training/train.py --config configs/1m_test.yaml
```

### 5. Train the full 350M model on Kaggle
- Upload `data/processed/train.bin` and `val.bin` to a Kaggle Dataset
- Open `notebooks/kaggle_pretrain.ipynb` on Kaggle
- Select `--config configs/350m.yaml`
- Enable P100 GPU, run all cells

### 6. Test the trained model
```bash
# Interactive chat:
python evaluation/generate.py --checkpoint checkpoints_sft/sft_final.pt --mode chat

# Code completion:
python evaluation/generate.py --checkpoint checkpoints/best.pt --mode completion --prompt "def quicksort"
```

---

## Model Architecture

| Component | Choice |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | ~350M (~360M) |
| Layers | 24 |
| Hidden dim | 1024 |
| Attention heads | 16 |
| FFN size | 3072 (SwiGLU) |
| Context window | 2048 tokens |
| Positional encoding | RoPE |
| Normalization | RMSNorm (pre-norm) |
| Activation | SwiGLU |

---

## Training

- **Pre-training**: 2.5B tokens from The Stack v2 (MIT/Apache licensed code)
- **Fine-tuning**: 255K instruction-response pairs (ChatML format)
- **Hardware**: Kaggle P100 16GB (free tier)
- **Duration**: ~3 weeks of Kaggle sessions (auto-resume)

---

## License

Proprietary — All rights reserved.
