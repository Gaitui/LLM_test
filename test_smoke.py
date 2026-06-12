"""Fast checks that do not require downloading WikiText."""

import torch

from model import ModelConfig, TransformerLM
from train import sample_batch


def test_forward_backward_and_generation():
    config = ModelConfig(
        vocab_size=257,
        context_length=16,
        n_layers=2,
        n_heads=2,
        d_model=32,
        dropout=0.0,
    )
    model = TransformerLM(config)
    data = torch.randint(0, config.vocab_size, (200,))
    x, y = sample_batch(data, batch_size=4, context_length=16, device=torch.device("cpu"))
    logits, loss = model(x, y)

    assert logits.shape == (4, 16, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()

    generated = model.generate(x[:1, :4], max_new_tokens=5, top_k=10)
    assert generated.shape == (1, 9)


if __name__ == "__main__":
    test_forward_backward_and_generation()
    print("Smoke test passed")

