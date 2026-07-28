"""
evaluation/generate.py — Interactive text generation for model testing.

Use this to manually test the model after training.
Works with both the pre-trained base model and the SFT fine-tuned model.

Usage:
    # Interactive chat mode (SFT model):
    python evaluation/generate.py --checkpoint checkpoints_sft/sft_final.pt --mode chat

    # Code completion mode (base model):
    python evaluation/generate.py --checkpoint checkpoints/best.pt --mode completion --prompt "def binary_search"
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import Transformer
from training.checkpoint import load_checkpoint


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load model and config from a checkpoint file."""
    print(f"Loading model from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Rebuild model from saved config
    model_cfg = ModelConfig.from_dict(ckpt["model_config"])
    model     = Transformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    # TPU checkpoints are saved in bfloat16.
    # bfloat16 has NO hardware acceleration on CPU → cast to float32 for local inference.
    # On CUDA, keep bfloat16 (supported natively and faster).
    if device.type == "cpu":
        model = model.to(torch.float32)

    model.eval()

    print(f"Model loaded: {model_cfg.name} ({model.n_params/1e6:.1f}M params)")
    return model, model_cfg



def load_tokenizer(tokenizer_path: str):
    """Load the trained tokenizer."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    return tok


def generate_completion(
    model: Transformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_token_id: int = None,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate a code completion from a prompt."""
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Default EOS to <|endoftext|>; chat mode passes <|im_end|> instead
    if eos_token_id is None:
        eos_token_id = tokenizer.token_to_id("<|endoftext|>")

    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

    generated_ids = output[0, len(input_ids):].tolist()
    return tokenizer.decode(generated_ids)


def generate_streaming(
    model: Transformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_token_id: int = None,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate text token-by-token, printing each token immediately.
    This makes local CPU inference feel responsive instead of hanging silently.
    """
    import torch.nn.functional as F

    if eos_token_id is None:
        eos_token_id = tokenizer.token_to_id("<|endoftext|>")

    input_ids = tokenizer.encode(prompt).ids
    ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = ids[:, -model.config.max_seq_len:]
            logits, _ = model(ctx)          # [1, 1, vocab_size]
            logits = logits[:, -1, :]       # [1, vocab_size]

            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p < 1.0:
                s_logits, s_idx = torch.sort(logits, descending=True)
                cump = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
                s_logits[cump - F.softmax(s_logits, dim=-1) > top_p] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, s_idx, s_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            if token_id == eos_token_id:
                break

            generated_ids.append(token_id)
            ids = torch.cat([ids, next_token], dim=1)

            # ── Stream token to screen immediately ────────────────
            token_text = tokenizer.decode([token_id])
            print(token_text, end="", flush=True)

    print()  # newline after response
    return tokenizer.decode(generated_ids)


def chat_turn(
    model: Transformer,
    tokenizer,
    user_message: str,
    system_prompt: str = "You are Aarohan, an expert software engineering assistant. Write clean, correct, well-documented code.",
    history: list = None,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    device: torch.device = torch.device("cpu"),
    stream: bool = False,
) -> str:
    """
    Generate a chat response in ChatML format.
    history: list of {"role": ..., "content": ...} dicts
    stream:  if True, print each token as it's generated (best for CPU)
    """
    # Build ChatML prompt
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

    if history:
        for turn in history:
            prompt += f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"

    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    im_end_id = tokenizer.token_to_id("<|im_end|>")

    if stream:
        response = generate_streaming(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=im_end_id,
            device=device,
        )
    else:
        response = generate_completion(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=im_end_id,
            device=device,
        )

    # Strip trailing <|im_end|> if present
    response = response.split("<|im_end|>")[0].strip()
    return response


def interactive_chat(model, tokenizer, device):
    """Run an interactive chat session in the terminal."""
    system  = "You are Aarohan, an expert software engineering assistant."
    history = []
    # Use streaming on CPU (tokens appear one by one as generated)
    # On GPU it's fast enough that streaming isn't needed, but doesn't hurt
    streaming = True

    print("\n" + "="*55)
    print("  Aarohan-350M Interactive Chat")
    print("  Type 'quit' to exit | 'clear' to reset history")
    print("  (streaming mode — tokens appear as generated)")
    print("="*55 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            history = []
            print("[History cleared]\n")
            continue

        print("\nAarohan: ", end="", flush=True)
        response = chat_turn(
            model, tokenizer, user_input, system, history,
            device=device, stream=streaming,
        )
        print()

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant",  "content": response})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SE-LLM Generation")
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--tokenizer",       type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--mode",            type=str, default="chat",
                        choices=["chat", "completion"])
    parser.add_argument("--prompt",          type=str, default="def binary_search(arr, target):")
    parser.add_argument("--max-new-tokens",  type=int, default=256)
    parser.add_argument("--temperature",     type=float, default=0.8)
    parser.add_argument("--top-k",           type=int,   default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_cfg = load_model_from_checkpoint(args.checkpoint, device)
    tokenizer        = load_tokenizer(args.tokenizer)

    if args.mode == "chat":
        interactive_chat(model, tokenizer, device)
    else:
        print(f"\nPrompt: {args.prompt}\n")
        result = generate_completion(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(f"Completion:\n{result}")
