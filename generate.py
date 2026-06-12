#!/usr/bin/env python3
"""Generate text from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import ModelConfig, TransformerLM

EOS_TOKEN_ID = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--prompt", default="The history of")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TransformerLM(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prompt_tokens = list(args.prompt.encode("utf-8"))
    tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = model.generate(
        tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    generated_bytes = bytes(token for token in generated if token != EOS_TOKEN_ID)
    print(generated_bytes.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()

