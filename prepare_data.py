#!/usr/bin/env python3
"""Stream TinyStories and encode it into memory-mapped GPT-2 token files."""

from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path
from typing import Iterable

TOKENIZER_NAME = "gpt2"
VOCAB_SIZE = 50_257
EOS_TOKEN_ID = 50_256


def write_split(documents, output_path, max_tokens, tokenizer):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0
    document_count = 0
    buffer = array("H")

    with output_path.open("wb") as output:
        for text in documents:
            if not text:
                continue
            tokens = tokenizer.encode_ordinary(text)
            tokens.append(EOS_TOKEN_ID)
            if max_tokens > 0:
                remaining = max_tokens - token_count
                if remaining <= 0:
                    break
                tokens = tokens[:remaining]

            buffer.extend(tokens)
            token_count += len(tokens)
            document_count += 1

            if len(buffer) >= 1_000_000:
                buffer.tofile(output)
                buffer = array("H")

        if buffer:
            buffer.tofile(output)

    return document_count, token_count


def validation_documents(dataset, parity):
    for index, row in enumerate(dataset):
        if index % 2 == parity:
            yield row["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("data/tinystories"))
    parser.add_argument("--max_train_tokens", type=int, default=100_000_000, help="Use 0 for the complete training split.")
    parser.add_argument("--max_eval_tokens", type=int, default=2_000_000)
    args = parser.parse_args()

    try:
        import tiktoken
        from datasets import load_dataset
    
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: python3 -m pip install -r requirements.txt") from exc

    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "roneneldan/TinyStories",
        "tokenizer": TOKENIZER_NAME,
        "dtype": "uint16",
        "vocab_size": VOCAB_SIZE,
        "eos_token_id": EOS_TOKEN_ID,
        "splits": {},
    }

    print("Streaming TinyStories train split...")
    train_dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    rows, tokens = write_split((row["text"] for row in train_dataset), args.output_dir / "train.bin", args.max_train_tokens, tokenizer)
    metadata["splits"]["train"] = {
        "documents": rows,
        "tokens": tokens,
        "file": "train.bin",
    }
    print(f"train     : {rows:9,d} stories, {tokens:12,d} tokens")

    # TinyStories has train/validation splits. Divide validation deterministically
    # so test data is never used for checkpoint selection.
    for split, parity in (("validation", 0), ("test", 1)):
        print(f"Streaming TinyStories validation split for {split}...")
        source = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
        rows, tokens = write_split(validation_documents(source, parity), args.output_dir / f"{split}.bin", args.max_eval_tokens, tokenizer)
        metadata["splits"][split] = {
            "documents": rows,
            "tokens": tokens,
            "file": f"{split}.bin",
        }
        print(f"{split:10s}: {rows:9,d} stories, {tokens:12,d} tokens")

    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved encoded dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
