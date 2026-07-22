"""
data/sft_data.py — Build the Instruction Fine-Tuning Dataset.

Downloads and formats instruction-response pairs in ChatML format.
All data sources are permissively licensed.

Sources:
  - Code Alpaca (20K)       — MIT
  - Evol-Instruct-Code (110K) — Apache 2.0
  - OSS-Instruct (75K)      — Apache 2.0
  - Stack Overflow CS (50K) — CC BY-SA (formatted, transformed)

Output: data/sft/sft_data.jsonl  (~255K examples)

Usage:
    python data/sft_data.py
    python data/sft_data.py --max-samples 10000  # quick test
"""

import os
import sys
import json
import argparse
import random
from typing import Iterator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_PROMPT = (
    "You are SE-LLM, an expert software engineering assistant. "
    "Write clean, correct, well-documented code. "
    "Explain your reasoning when helpful."
)

OUTPUT_PATH = "data/sft/sft_data.jsonl"


# ── Formatters ────────────────────────────────────────────────────────────────

def make_sample(user: str, assistant: str, system: str = SYSTEM_PROMPT) -> dict:
    """Build a single ChatML sample dict."""
    return {
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def is_good_sample(user: str, assistant: str) -> bool:
    """Filter out low-quality samples."""
    if len(user) < 10 or len(assistant) < 20:
        return False
    if len(user) > 4000 or len(assistant) > 8000:
        return False
    return True


# ── Data source loaders ───────────────────────────────────────────────────────

def load_code_alpaca(max_samples: int) -> Iterator[dict]:
    """
    Load Code Alpaca — 20K coding instruction-response pairs.
    https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k
    """
    from datasets import load_dataset
    print("  Loading Code Alpaca (20K)...")
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        count = 0
        for item in ds:
            if count >= max_samples:
                break
            instruction = item.get("instruction", "")
            output      = item.get("output", "")
            context     = item.get("input", "")

            if context:
                user = f"{instruction}\n\n```\n{context}\n```"
            else:
                user = instruction

            if not is_good_sample(user, output):
                continue

            yield make_sample(user, output)
            count += 1
        print(f"    → {count} samples from Code Alpaca")
    except Exception as e:
        print(f"    Warning: Could not load Code Alpaca: {e}")


def load_evol_instruct(max_samples: int) -> Iterator[dict]:
    """
    Load Evol-Instruct-Code — complex coding tasks.
    https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1
    """
    from datasets import load_dataset
    print("  Loading Evol-Instruct-Code (~110K)...")
    try:
        ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
        count = 0
        for item in ds:
            if count >= max_samples:
                break
            user   = item.get("instruction", "") or item.get("input", "")
            answer = item.get("output", "")
            if not is_good_sample(user, answer):
                continue
            yield make_sample(user, answer)
            count += 1
        print(f"    → {count} samples from Evol-Instruct")
    except Exception as e:
        print(f"    Warning: Could not load Evol-Instruct: {e}")


def load_oss_instruct(max_samples: int) -> Iterator[dict]:
    """
    Load OSS-Instruct — real-world SE tasks from open-source code.
    https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K
    """
    from datasets import load_dataset
    print("  Loading OSS-Instruct (75K)...")
    try:
        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
        count = 0
        for item in ds:
            if count >= max_samples:
                break
            user   = item.get("problem", "") or item.get("instruction", "")
            answer = item.get("solution", "") or item.get("output", "")
            if not is_good_sample(user, answer):
                continue
            yield make_sample(user, answer)
            count += 1
        print(f"    → {count} samples from OSS-Instruct")
    except Exception as e:
        print(f"    Warning: Could not load OSS-Instruct: {e}")


def load_codesearchnet_qa(max_samples: int) -> Iterator[dict]:
    """
    Generate instruction pairs from CodeSearchNet docstrings.
    Turns (docstring, function) into ("Write a function that...", code).
    """
    from datasets import load_dataset
    print("  Generating Q&A from CodeSearchNet docstrings...")
    langs = ["python", "javascript", "java", "go"]
    count = 0
    per_lang = max_samples // len(langs)

    for lang in langs:
        try:
            ds = load_dataset("code_search_net", lang, split="train", streaming=True)
            lang_count = 0
            for item in ds:
                if lang_count >= per_lang:
                    break
                docstring = item.get("func_documentation_string", "").strip()
                code      = item.get("whole_func_string", "").strip()

                if not docstring or not code or len(docstring) < 20:
                    continue

                # Format as instruction
                user   = f"Write a {lang} function that: {docstring}"
                answer = f"```{lang}\n{code}\n```"

                if not is_good_sample(user, answer):
                    continue

                yield make_sample(user, answer)
                count += 1
                lang_count += 1
        except Exception as e:
            print(f"    Warning: CodeSearchNet {lang}: {e}")

    print(f"    → {count} samples from CodeSearchNet Q&A")


def generate_synthetic_samples() -> Iterator[dict]:
    """
    A small set of hand-crafted high-quality SE instruction pairs.
    These ensure the model handles common SE tasks well.
    """
    examples = [
        {
            "user": "What is the difference between `==` and `===` in JavaScript?",
            "assistant": (
                "In JavaScript:\n\n"
                "- `==` (loose equality): compares values after **type coercion**\n"
                "- `===` (strict equality): compares both **value AND type** — no coercion\n\n"
                "```javascript\n"
                "// == with type coercion\n"
                "0 == '0'    // true  (string '0' coerced to number)\n"
                "0 == false  // true  (false coerced to 0)\n"
                "null == undefined // true\n\n"
                "// === strict\n"
                "0 === '0'   // false (different types)\n"
                "0 === false // false (different types)\n"
                "```\n\n"
                "**Best practice**: Always use `===` unless you specifically need type coercion."
            ),
        },
        {
            "user": "Write a Python function to check if a string is a palindrome.",
            "assistant": (
                "```python\n"
                "def is_palindrome(s: str) -> bool:\n"
                "    \"\"\"\n"
                "    Check if a string is a palindrome (reads same forwards and backwards).\n"
                "    Case-insensitive, ignores spaces and punctuation.\n\n"
                "    Args:\n"
                "        s: Input string\n\n"
                "    Returns:\n"
                "        True if palindrome, False otherwise\n\n"
                "    Examples:\n"
                "        >>> is_palindrome('racecar')  # True\n"
                "        >>> is_palindrome('A man a plan a canal Panama')  # True\n"
                "        >>> is_palindrome('hello')  # False\n"
                "    \"\"\"\n"
                "    # Normalize: lowercase, keep only alphanumeric\n"
                "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n"
                "    return cleaned == cleaned[::-1]\n"
                "```"
            ),
        },
        {
            "user": "Explain the difference between SQL INNER JOIN and LEFT JOIN with examples.",
            "assistant": (
                "## INNER JOIN vs LEFT JOIN\n\n"
                "**INNER JOIN**: Returns only rows where there's a match in BOTH tables.\n"
                "**LEFT JOIN**: Returns ALL rows from the left table, and matched rows from the right (NULL if no match).\n\n"
                "```sql\n"
                "-- Sample tables\n"
                "-- users: id, name\n"
                "-- orders: id, user_id, amount\n\n"
                "-- INNER JOIN: only users who have orders\n"
                "SELECT u.name, o.amount\n"
                "FROM users u\n"
                "INNER JOIN orders o ON u.id = o.user_id;\n\n"
                "-- LEFT JOIN: ALL users, even those without orders\n"
                "SELECT u.name, o.amount  -- o.amount is NULL for users with no orders\n"
                "FROM users u\n"
                "LEFT JOIN orders o ON u.id = o.user_id;\n"
                "```\n\n"
                "**When to use:**\n"
                "- `INNER JOIN` → you only care about records that exist in both tables\n"
                "- `LEFT JOIN` → you want all records from the left table regardless of matches"
            ),
        },
    ]

    for ex in examples:
        yield make_sample(ex["user"], ex["assistant"])


# ── Main ──────────────────────────────────────────────────────────────────────

def build_sft_dataset(max_samples: int, output_path: str) -> None:
    """
    Download and merge all SFT data sources into a single JSONL file.
    """
    print(f"\n{'='*55}")
    print(f"  Building SFT Dataset")
    print(f"  Max samples: {max_samples:,}")
    print(f"  Output: {output_path}")
    print(f"{'='*55}\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    per_source = max_samples // 4
    all_samples = []

    # Collect from all sources
    for sample in load_code_alpaca(per_source):
        all_samples.append(sample)

    for sample in load_evol_instruct(per_source):
        all_samples.append(sample)

    for sample in load_oss_instruct(per_source):
        all_samples.append(sample)

    for sample in load_codesearchnet_qa(per_source):
        all_samples.append(sample)

    for sample in generate_synthetic_samples():
        all_samples.append(sample)

    # Shuffle for better training distribution
    random.seed(42)
    random.shuffle(all_samples)

    # Trim to max
    all_samples = all_samples[:max_samples]

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n{'='*55}")
    print(f"  SFT Dataset COMPLETE")
    print(f"  Total samples: {len(all_samples):,}")
    print(f"  Output: {output_path}")
    print(f"  Size: {os.path.getsize(output_path)/1e6:.1f} MB")
    print(f"{'='*55}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT instruction dataset")
    parser.add_argument("--max-samples", type=int, default=255_000)
    parser.add_argument("--output",      type=str, default=OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_sft_dataset(args.max_samples, args.output)
