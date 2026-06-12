#!/usr/bin/env python3
"""Generate text from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import ModelConfig, TransformerLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/tinystories/best.pt"),
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run: python3 -m pip install -r requirements.txt"
        ) from exc

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TransformerLM(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenizer = tiktoken.get_encoding(checkpoint.get("tokenizer", "gpt2"))
    prompt_tokens = tokenizer.encode_ordinary(args.prompt)
    tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = model.generate(
        tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    eos_token_id = checkpoint.get("eos_token_id", 50_256)
    if eos_token_id in generated:
        generated = generated[: generated.index(eos_token_id)]
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
