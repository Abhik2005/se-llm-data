"""
evaluation/humaneval.py — Run HumanEval Benchmark on SE-LLM.

HumanEval is the standard benchmark for code generation models.
  - 164 Python programming problems
  - Each problem: function signature + docstring → generate the body
  - Metric: pass@1 (does the generated code pass all unit tests?)

Usage:
    python evaluation/humaneval.py --checkpoint checkpoints/best.pt
    python evaluation/humaneval.py --checkpoint checkpoints_sft/sft_final.pt
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import Transformer
from evaluation.generate import load_model_from_checkpoint, load_tokenizer, generate_completion


# ── HumanEval loader ──────────────────────────────────────────────────────────

def load_humaneval() -> list[dict]:
    """
    Load the HumanEval dataset.
    Returns list of problems with: task_id, prompt, entry_point, test
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
        problems = list(ds)
        print(f"Loaded {len(problems)} HumanEval problems")
        return problems
    except Exception as e:
        print(f"Could not load HumanEval from HuggingFace: {e}")
        print("Install: pip install datasets")
        return []


# ── Code execution sandbox ────────────────────────────────────────────────────

def execute_code_with_tests(code: str, test: str, timeout: int = 10) -> bool:
    """
    Execute generated code + test suite in a subprocess sandbox.
    Returns True if all tests pass, False otherwise.

    Args:
        code:    Generated function code
        test:    HumanEval test string (assert statements)
        timeout: Maximum execution time in seconds
    """
    # Build the full test script
    full_code = f"{code}\n\n{test}\n\ncheck({{}})".format(
        # Extract entry point from test string if possible
        "solution"
    )

    # Better: use the standard HumanEval format
    full_script = f"""
{code}

# Tests
{test}
"""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(full_script)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Generation for HumanEval ──────────────────────────────────────────────────

def generate_for_problem(
    model: Transformer,
    tokenizer,
    problem: dict,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate a completion for a single HumanEval problem.
    The prompt is the function signature + docstring.
    The model must complete the function body.
    """
    prompt = problem["prompt"]  # function signature + docstring

    completion = generate_completion(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=0,       # No top-k for evaluation
        top_p=1.0,     # No top-p for evaluation
        device=device,
    )

    # Return prompt + completion as full function
    return prompt + completion


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_humaneval(
    checkpoint: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    n_samples: int = 1,       # pass@k: number of samples per problem
    max_problems: Optional[int] = None,
    output_file: str = "evaluation/humaneval_results.jsonl",
) -> dict:
    """
    Run HumanEval benchmark and report pass@k scores.

    Args:
        checkpoint:    Path to model checkpoint
        tokenizer_path: Path to tokenizer.json
        temperature:   Sampling temperature
        n_samples:     Number of completions per problem (for pass@k)
        max_problems:  Limit number of problems (None = all 164)
        output_file:   Where to save detailed results

    Returns:
        {"pass@1": float, "total": int, "passed": int}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nHumanEval Benchmark")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Temperature: {temperature}\n")

    # Load model
    model, model_cfg = load_model_from_checkpoint(checkpoint, device)
    tokenizer        = load_tokenizer(tokenizer_path)

    # Load problems
    problems = load_humaneval()
    if not problems:
        print("ERROR: Could not load HumanEval. Install: pip install datasets")
        return {}

    if max_problems:
        problems = problems[:max_problems]

    total  = len(problems)
    passed = 0
    results = []

    print(f"Running {total} problems...\n")

    for i, problem in enumerate(problems):
        task_id     = problem["task_id"]
        test        = problem["test"]
        entry_point = problem["entry_point"]

        problem_results = []

        for sample_idx in range(n_samples):
            code = generate_for_problem(
                model, tokenizer, problem,
                temperature=temperature,
                device=device,
            )

            # Try to execute and pass tests
            # Build full test with entry point
            test_code = f"{test}\ncheck({entry_point})"
            ok = execute_code_with_tests(code, test_code)
            problem_results.append(ok)

        # pass@1: did at least one sample pass?
        any_passed = any(problem_results)
        if any_passed:
            passed += 1

        status = "✅" if any_passed else "❌"
        print(f"  [{i+1:3d}/{total}] {task_id:<30} {status}")

        results.append({
            "task_id":    task_id,
            "passed":     any_passed,
            "samples":    problem_results,
        })

    # Save detailed results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    pass_at_1 = passed / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"  HumanEval Results")
    print(f"  pass@1: {pass_at_1*100:.1f}%  ({passed}/{total} passed)")
    print(f"  Results saved: {output_file}")
    print(f"{'='*50}\n")

    return {"pass@1": pass_at_1, "total": total, "passed": passed}


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HumanEval benchmark")
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--tokenizer",       type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--temperature",     type=float, default=0.2)
    parser.add_argument("--max-problems",    type=int, default=None,
                        help="Limit to N problems (None = all 164)")
    parser.add_argument("--output",          type=str, default="evaluation/humaneval_results.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_humaneval(
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        temperature=args.temperature,
        max_problems=args.max_problems,
        output_file=args.output,
    )
