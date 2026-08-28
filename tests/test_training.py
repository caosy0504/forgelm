from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from forgelm.config import ModelConfig, TrainingConfig
from forgelm.model import TransformerLM
from forgelm.training import Trainer, learning_rate_for_step, sample_batch


class TrainingTests(unittest.TestCase):
    def test_wsd_schedule_has_stable_and_decay_phases(self) -> None:
        config = TrainingConfig(
            steps=10,
            batch_size=1,
            seq_len=4,
            learning_rate=1.0,
            min_learning_rate_ratio=0.1,
            warmup_steps=2,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            lr_schedule="wsd",
            wsd_decay_steps=3,
        )
        self.assertEqual(learning_rate_for_step(0, config), 0.5)
        self.assertEqual(learning_rate_for_step(3, config), 1.0)
        self.assertAlmostEqual(learning_rate_for_step(9, config), 0.1)

    def test_training_checkpoint_round_trip(self) -> None:
        model_config = ModelConfig(
            d_model=24,
            n_layers=1,
            n_heads=3,
            d_ff=48,
            max_seq_len=12,
            attention_impl="eager",
            vocab_size=270,
        )
        training_config = TrainingConfig(
            steps=2,
            batch_size=2,
            seq_len=8,
            learning_rate=1e-3,
            warmup_steps=0,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=2,
            device="cpu",
        )
        tokens = torch.arange(400, dtype=torch.long) % 270
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                TransformerLM(model_config),
                tokens,
                tokens.clone(),
                training_config,
                seed=7,
                artifact_dir=directory,
            )
            summary = trainer.train()
            checkpoint = Path(summary["checkpoint"])
            self.assertTrue(checkpoint.exists())
            restored = Trainer(
                TransformerLM(model_config),
                tokens,
                tokens.clone(),
                training_config,
                seed=7,
                artifact_dir=Path(directory) / "restored",
            )
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.step, 2)
            for expected, actual in zip(trainer.model.parameters(), restored.model.parameters(), strict=True):
                torch.testing.assert_close(expected, actual)

    def test_resume_continues_steps_and_appends_metrics(self) -> None:
        model_config = ModelConfig(
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            max_seq_len=8,
            attention_impl="eager",
            vocab_size=260,
        )
        first_config = TrainingConfig(
            steps=1,
            batch_size=2,
            seq_len=6,
            learning_rate=1e-3,
            warmup_steps=0,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            device="cpu",
        )
        resumed_config = TrainingConfig(**{**first_config.__dict__, "steps": 2})
        tokens = torch.arange(200, dtype=torch.long) % 260
        with tempfile.TemporaryDirectory() as directory:
            first = Trainer(
                TransformerLM(model_config), tokens, tokens, first_config, seed=11, artifact_dir=directory
            )
            checkpoint = first.train()["checkpoint"]
            original_metric_count = len((Path(directory) / "metrics.jsonl").read_text().splitlines())
            resumed = Trainer(
                TransformerLM(model_config),
                tokens,
                tokens,
                resumed_config,
                seed=11,
                artifact_dir=directory,
                reset_metrics=False,
            )
            resumed.load_checkpoint(checkpoint)
            resumed.train()
            self.assertEqual(resumed.step, 2)
            new_metric_count = len((Path(directory) / "metrics.jsonl").read_text().splitlines())
            self.assertGreater(new_metric_count, original_metric_count)

    def test_resume_rejects_fingerprint_mismatch(self) -> None:
        model_config = ModelConfig(
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            max_seq_len=8,
            vocab_size=260,
        )
        config = TrainingConfig(
            steps=1,
            batch_size=1,
            seq_len=4,
            warmup_steps=0,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            device="cpu",
        )
        tokens = torch.arange(100, dtype=torch.long) % 260
        with tempfile.TemporaryDirectory() as directory:
            first = Trainer(
                TransformerLM(model_config),
                tokens,
                tokens,
                config,
                seed=3,
                artifact_dir=directory,
                run_fingerprint={"fingerprint_id": "first"},
            )
            checkpoint = first.train()["checkpoint"]
            incompatible = Trainer(
                TransformerLM(model_config),
                tokens,
                tokens,
                config,
                seed=3,
                artifact_dir=Path(directory) / "incompatible",
                run_fingerprint={"fingerprint_id": "second"},
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                incompatible.load_checkpoint(checkpoint)

    def test_validation_reports_cross_entropy_not_z_regularized_objective(self) -> None:
        model_config = ModelConfig(
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            max_seq_len=8,
            vocab_size=260,
            z_loss_weight=0.1,
        )
        config = TrainingConfig(
            steps=1,
            batch_size=2,
            seq_len=6,
            warmup_steps=0,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            device="cpu",
        )
        tokens = torch.arange(200, dtype=torch.long) % 260
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                TransformerLM(model_config), tokens, tokens, config, seed=5, artifact_dir=directory
            )
            reported = trainer.evaluate()
            generator = torch.Generator(device="cpu").manual_seed(10_005)
            inputs, targets = sample_batch(
                tokens,
                batch_size=2,
                seq_len=6,
                device=torch.device("cpu"),
                generator=generator,
            )
            with torch.inference_mode():
                logits, objective = trainer.model(inputs, targets)
            cross_entropy = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            self.assertAlmostEqual(reported, float(cross_entropy), places=6)
            self.assertGreater(float(objective), reported)


if __name__ == "__main__":
    unittest.main()
