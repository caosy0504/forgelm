from __future__ import annotations

import dataclasses
import unittest

import torch

from forgelm.config import ModelConfig
from forgelm.model import TransformerLM


def tiny_config(attention_impl: str = "eager") -> ModelConfig:
    return ModelConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
        attention_impl=attention_impl,
        tie_embeddings=True,
        vocab_size=270,
    )


class ModelTests(unittest.TestCase):
    def test_forward_shape_and_finite_loss(self) -> None:
        model = TransformerLM(tiny_config())
        inputs = torch.randint(0, 270, (3, 12))
        logits, loss = model(inputs, inputs)
        self.assertEqual(tuple(logits.shape), (3, 12, 270))
        self.assertIsNotNone(loss)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_causal_mask_blocks_future_tokens(self) -> None:
        torch.manual_seed(0)
        model = TransformerLM(tiny_config()).eval()
        left = torch.tensor([[1, 2, 3, 4, 5]])
        right = torch.tensor([[1, 2, 9, 8, 7]])
        with torch.inference_mode():
            left_logits, _ = model(left)
            right_logits, _ = model(right)
        torch.testing.assert_close(left_logits[:, :2], right_logits[:, :2], atol=1e-6, rtol=1e-6)

    def test_eager_matches_sdpa(self) -> None:
        torch.manual_seed(0)
        eager = TransformerLM(tiny_config("eager")).eval()
        sdpa = TransformerLM(dataclasses.replace(tiny_config(), attention_impl="sdpa")).eval()
        sdpa.load_state_dict(eager.state_dict())
        inputs = torch.randint(0, 270, (2, 10))
        with torch.inference_mode():
            eager_logits, _ = eager(inputs)
            sdpa_logits, _ = sdpa(inputs)
        torch.testing.assert_close(eager_logits, sdpa_logits, atol=2e-5, rtol=2e-5)

    def test_qk_norm_and_z_loss_are_finite(self) -> None:
        config = dataclasses.replace(tiny_config(), qk_norm=True, z_loss_weight=1e-4, rope_theta=500_000.0)
        model = TransformerLM(config)
        inputs = torch.randint(0, 270, (2, 10))
        _, loss = model(inputs, inputs)
        self.assertIsNotNone(loss)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_activation_checkpointing_backward(self) -> None:
        model = TransformerLM(dataclasses.replace(tiny_config(), qk_norm=True))
        model.set_gradient_checkpointing(True)
        inputs = torch.randint(0, 270, (2, 10))
        _, loss = model(inputs, inputs)
        assert loss is not None
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
