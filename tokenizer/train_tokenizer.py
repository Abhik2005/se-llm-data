"""
tokenizer/train_tokenizer.py — Train a Code + English BPE Tokenizer.

Now with HF token support — unlocks bigcode/the-stack for real
code files in 33 languages. Much better BPE merges for C++, Rust, Go, SQL, etc.

Data mix (50M tokens):
  60% — Actual code files (bigcode/the-stack, 33 languages)  ← needs HF token
  20% — Code instruction pairs (Magicoder, evol-instruct, CodeAlpaca)
  10% — English educational text (fineweb-edu)
  10% — SE knowledge Q&A (Stack Exchange: security, devops, architecture...)

HF Token setup (one-time):
  1. huggingface.co/settings/tokens → create Read token
  2. Accept dataset terms: huggingface.co/datasets/bigcode/the-stack
  3. Add to .env:  HF_TOKEN=hf_xxxxxxxxxxxx
     OR set in PowerShell: $env:HF_TOKEN = "hf_xxxxxxxxxxxx"

Runtime: 30-60 minutes
Output:  tokenizer/tokenizer.json  (32,000 vocab, ByteLevel BPE)

Usage:
    python tokenizer/train_tokenizer.py
    python tokenizer/train_tokenizer.py --sample-size 100000000
"""

import os
import sys
import argparse
from typing import Iterator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Load HF token ─────────────────────────────────────────────────────────────

def _load_hf_token() -> str | None:
    """Load HF_TOKEN from env or .env file."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token

    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    t = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if t and t != "your_token_here":
                        os.environ["HF_TOKEN"] = t
                        return t
    return None

HF_TOKEN = _load_hf_token()


# ── Special tokens ────────────────────────────────────────────────────────────

SPECIAL_TOKENS = [
    "<|pad|>",        # 0
    "<|endoftext|>",  # 1
    "<|im_start|>",   # 2
    "<|im_end|>",     # 3
    "<|fim_prefix|>", # 4
    "<|fim_suffix|>", # 5
    "<|fim_middle|>", # 6
    "<|unk|>",        # 7
]

VOCAB_SIZE  = 32_000
SAMPLE_SIZE = 50_000_000   # 50M tokens = ~200MB text (bigger = better merges)

# All SE languages — same list as prepare_data.py
STACK_LANGUAGES = [
    "c", "c++", "rust", "assembly", "cuda",
    "python", "java", "go", "kotlin", "scala",
    "swift", "dart", "r", "julia", "haskell",
    "ocaml", "elixir", "erlang", "clojure",
    "javascript", "typescript", "ruby", "php", "perl",
    "lua", "shell", "powershell", "batchfile",
    "html", "css", "vue",
    "sql",
    "dockerfile", "makefile", "cmake",
    "c-sharp", "f#", "zig",
]


# ── Data streaming ─────────────────────────────────────────────────────────────

def stream_the_stack(budget: int) -> Iterator[str]:
    """
    Stream actual code files from bigcode/the-stack (full).
    Best quality — filtered, deduplicated, permissive licenses only.
    Requires HF_TOKEN + accepted terms: huggingface.co/datasets/bigcode/the-stack
    """
    from datasets import load_dataset

    if not HF_TOKEN:
        print("  [the-stack] No HF_TOKEN — skipping. Set it in .env to unlock.", flush=True)
        return

    print(f"  [the-stack] {len(STACK_LANGUAGES)} languages (HF token: ✅)...", flush=True)
    per_lang = budget // len(STACK_LANGUAGES)
    total    = 0

    for lang in STACK_LANGUAGES:
        lang_chars = 0
        try:
            ds = load_dataset(
                "bigcode/the-stack",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True,
                token=HF_TOKEN,
            )
            for sample in ds:
                if lang_chars >= per_lang: break
                content = sample.get("content", "") or ""
                if len(content) < 100: continue
                yield content
                lang_chars += len(content)
                total      += len(content)
            if lang_chars > 0:
                print(f"    → {lang}: {lang_chars/1e6:.1f}M chars")
        except Exception as e:
            # Language not in dataset or other error — skip silently
            if "gated" in str(e).lower():
                print(f"    ⚠ {lang}: access denied — accept terms at huggingface.co/datasets/bigcode/the-stack")
                break  # All languages will fail if terms not accepted
            # else: language just not available in smol subset

    print(f"    Total from Stack: {total/1e6:.1f}M chars")


def stream_instruction_code(budget: int) -> Iterator[str]:
    """Stream code instruction pairs: Magicoder + evol-instruct + CodeAlpaca."""
    from datasets import load_dataset
    per_source = budget // 3

    # Magicoder
    print("  [Magicoder-OSS-Instruct] ...", flush=True)
    chars = 0
    try:
        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True)
        for s in ds:
            if chars >= per_source: break
            text = f"{s.get('problem','')}\n\n{s.get('solution','')}".strip()
            if len(text) < 50: continue
            yield text; chars += len(text)
        print(f"    → {chars/1e6:.1f}M chars")
    except Exception as e:
        print(f"    Warning: {e}")

    # evol-codealpaca
    print("  [evol-codealpaca-v1] ...", flush=True)
    chars = 0
    try:
        ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train", streaming=True)
        for s in ds:
            if chars >= per_source: break
            text = f"{s.get('instruction','')}\n\n{s.get('output','')}".strip()
            if len(text) < 50: continue
            yield text; chars += len(text)
        print(f"    → {chars/1e6:.1f}M chars")
    except Exception as e:
        print(f"    Warning: {e}")

    # CodeAlpaca
    print("  [CodeAlpaca-20k] ...", flush=True)
    chars = 0
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
        for s in ds:
            if chars >= per_source: break
            text = f"{s.get('instruction','')}\n{s.get('input','')}\n{s.get('output','')}".strip()
            if len(text) < 50: continue
            yield text; chars += len(text)
        print(f"    → {chars/1e6:.1f}M chars")
    except Exception as e:
        print(f"    Warning: {e}")


def stream_english(budget: int) -> Iterator[str]:
    """Stream fineweb-edu — high quality educational English text."""
    from datasets import load_dataset
    print("  [fineweb-edu] ...", flush=True)
    chars = 0
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        for s in ds:
            if chars >= budget: break
            text = s.get("text", "") or ""
            if len(text) < 100 or len(text) > 50_000: continue
            yield text; chars += len(text)
        print(f"    → {chars/1e6:.1f}M chars")
    except Exception as e:
        print(f"    Warning: {e}")


def stream_se_knowledge(budget: int) -> Iterator[str]:
    """
    Stream Stack Exchange Q&A — covers all SE knowledge domains:
    system design, cybersecurity, DevOps, architecture, DB design...
    """
    from datasets import load_dataset
    print("  [stack-exchange-instruction] SE knowledge ...", flush=True)
    chars = 0
    try:
        ds = load_dataset("ArmelR/stack-exchange-instruction", split="test", streaming=True)
        for s in ds:
            if chars >= budget: break
            text = f"{s.get('question','')}\n\n{s.get('response','')}".strip()
            if len(text) < 100: continue
            yield text; chars += len(text)
        print(f"    → {chars/1e6:.1f}M chars")
    except Exception as e:
        print(f"    Warning: {e}")


def all_text_iterator(sample_size: int) -> Iterator[str]:
    """Combined iterator with 60/20/10/10 code/instruct/english/SE mix."""
    total  = sample_size * 4   # chars

    code_b    = int(total * 0.60)
    inst_b    = int(total * 0.20)
    eng_b     = int(total * 0.10)
    seknow_b  = int(total * 0.10)

    print(f"\n  Budget breakdown (~{total/1e6:.0f}MB text):")
    print(f"    Actual code (Stack):  {code_b/1e6:.0f}M chars  (60%)")
    print(f"    Instruction code:     {inst_b/1e6:.0f}M chars  (20%)")
    print(f"    English:              {eng_b/1e6:.0f}M chars  (10%)")
    print(f"    SE knowledge:         {seknow_b/1e6:.0f}M chars  (10%)\n")

    print("── ACTUAL CODE (The Stack smol) ──────────────────────────")
    yield from stream_the_stack(code_b)

    print("\n── INSTRUCTION CODE ──────────────────────────────────────")
    yield from stream_instruction_code(inst_b)

    print("\n── ENGLISH ───────────────────────────────────────────────")
    yield from stream_english(eng_b)

    print("\n── SE KNOWLEDGE (security, devops, architecture...) ──────")
    yield from stream_se_knowledge(seknow_b)


# ── Train tokenizer ────────────────────────────────────────────────────────────

def train_tokenizer(sample_size: int, output_dir: str) -> None:
    """Train a BPE tokenizer on code + English data."""
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    except ImportError:
        raise RuntimeError("pip install tokenizers")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training BPE Tokenizer (Code + English + SE Knowledge)")
    print(f"  Vocab:       {VOCAB_SIZE:,}")
    print(f"  Sample:      {sample_size:,} tokens (~{sample_size*4/1e6:.0f}MB text)")
    print(f"  HF Token:    {'✅ set — The Stack unlocked' if HF_TOKEN else '❌ not set — Stack will be skipped'}")
    print(f"  Mix:         60% code | 20% instruct | 10% English | 10% SE Q&A")
    print(f"{'='*60}")

    tokenizer               = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder       = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(all_text_iterator(sample_size), trainer=trainer)

    json_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(json_path)
    print(f"\nTokenizer saved → {json_path}")

    print("\nSpecial token IDs:")
    for token in SPECIAL_TOKENS:
        print(f"  {token:<22} → {tokenizer.token_to_id(token)}")

    # Quality check across languages
    tests = [
        ("Python",  'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr'),
        ("Rust",    'fn main() {\n    let x: i32 = 42;\n    println!("{}", x);\n}'),
        ("C++",     '#include <iostream>\nint main() {\n    std::cout << "Hello";\n}'),
        ("SQL",     'SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.name;'),
        ("English", "Let's design a microservices architecture with load balancing."),
        ("SE Q&A",  "Q: How do I prevent SQL injection?\nA: Use parameterized queries."),
    ]

    print("\nEncoding quality:")
    for label, text in tests:
        enc = tokenizer.encode(text)
        cpt = len(text) / max(len(enc.ids), 1)
        print(f"  {label:<10} {len(text):>4} chars → {len(enc.ids):>3} tokens ({cpt:.1f} chars/tok)")

    print(f"\n✅ Tokenizer training COMPLETE!")
    print(f"Next step: python data/prepare_data.py")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SE-LLM BPE tokenizer")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--output-dir",  type=str, default="tokenizer")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(args.sample_size, args.output_dir)
