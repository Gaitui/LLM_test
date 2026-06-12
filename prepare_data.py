#!/usr/bin/env python3
"""Download WikiText-2 and encode its official splits as byte tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

EOS_TOKEN_ID = 256
VOCAB_SIZE = 257


def encode_rows(rows):
    tokens = []
    for text in rows:
        if not text:
            continue
        tokens.extend(text.encode("utf-8"))
        tokens.append(EOS_TOKEN_ID)
    return torch.tensor(tokens, dtype=torch.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/wikitext2"))
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run: python3 -m pip install -r requirements.txt"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    metadata = {
        "dataset": "Salesforce/wikitext",
        "subset": "wikitext-2-raw-v1",
        "tokenizer": "UTF-8 bytes plus EOS token",
        "vocab_size": VOCAB_SIZE,
        "eos_token_id": EOS_TOKEN_ID,
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        encoded = encode_rows(dataset[split]["text"])
        output_path = args.output_dir / f"{split}.pt"
        torch.save(encoded, output_path)
        metadata["splits"][split] = {
            "rows": len(dataset[split]),
            "tokens": encoded.numel(),
            "file": output_path.name,
        }
        print(f"{split:10s}: {len(dataset[split]):6,d} rows, {encoded.numel():10,d} tokens")

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved encoded dataset to {args.output_dir}")


if __name__ == "__main__":
    main()

