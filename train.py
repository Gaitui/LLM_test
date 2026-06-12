#!/usr/bin/env python3
"""Train and evaluate a GPT-style model on memory-mapped token data."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from model import ModelConfig, TransformerLM


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(data_dir: Path, split: str) -> np.memmap:
    path = data_dir / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run: python3 prepare_data.py"
        )
    return np.memmap(path, dtype=np.uint16, mode="r")


def sample_batch(
    data: np.ndarray | torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    data_length = len(data)
    if data_length <= context_length:
        raise ValueError("Dataset split is shorter than context_length")
    starts = torch.randint(0, data_length - context_length, (batch_size,)).tolist()
    if isinstance(data, torch.Tensor):
        x = torch.stack([data[i : i + context_length] for i in starts])
        y = torch.stack([data[i + 1 : i + context_length + 1] for i in starts])
    else:
        sequences = np.stack(
            [data[i : i + context_length + 1] for i in starts]
        ).astype(np.int64)
        batch = torch.from_numpy(sequences)
        x, y = batch[:, :-1], batch[:, 1:]
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    data: np.ndarray | torch.Tensor,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = sample_batch(
            data, batch_size, model.config.context_length, device
        )
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(
    path: Path,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    validation_loss: float,
    tokenizer: str,
    eos_token_id: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "validation_loss": validation_loss,
            "tokenizer": tokenizer,
            "eos_token_id": eos_token_id,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/tinystories"))
    parser.add_argument(
        "--output_dir", type=Path, default=Path("checkpoints/tinystories")
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    metadata_path = args.data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} does not exist. Run: python3 prepare_data.py"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_data = load_split(args.data_dir, "train")
    validation_data = load_split(args.data_dir, "validation")
    test_data = load_split(args.data_dir, "test")

    config = ModelConfig(
        vocab_size=metadata["vocab_size"],
        context_length=args.context_length,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )
    model = TransformerLM(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameter_count:,}")
    print(
        f"Training tokens: {len(train_data):,} | "
        f"validation tokens: {len(validation_data):,} | "
        f"test tokens: {len(test_data):,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_validation_loss = float("inf")
    started_at = time.time()

    model.train()
    for step in range(1, args.max_steps + 1):
        x, y = sample_batch(
            train_data, args.batch_size, args.context_length, device
        )
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0:
            validation_loss = evaluate(
                model,
                validation_data,
                args.batch_size,
                args.eval_batches,
                device,
            )
            elapsed = time.time() - started_at
            print(
                f"step {step:5d} | train loss {loss.item():.4f} | "
                f"val loss {validation_loss:.4f} | "
                f"val ppl {math.exp(min(validation_loss, 20)):.2f} | "
                f"{elapsed:.1f}s"
            )
            save_checkpoint(
                args.output_dir / "last.pt",
                model,
                optimizer,
                step,
                validation_loss,
                metadata["tokenizer"],
                metadata["eos_token_id"],
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer,
                    step,
                    validation_loss,
                    metadata["tokenizer"],
                    metadata["eos_token_id"],
                )

    best = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(best["model_state"])
    test_loss = evaluate(
        model, test_data, args.batch_size, args.eval_batches, device
    )
    print(
        f"Final test loss: {test_loss:.4f} | "
        f"test perplexity: {math.exp(min(test_loss, 20)):.2f}"
    )


if __name__ == "__main__":
    main()
