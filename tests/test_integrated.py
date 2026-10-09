from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import torch

from forgelm.config import ModelConfig, TrainingConfig
from forgelm.data_pipeline import Document, write_jsonl
from forgelm.evaluation import evaluate_checkpoint
from forgelm.fingerprint import sha256_file
from forgelm.model import TransformerLM
from forgelm.pipeline import run_pipeline
from forgelm.sampling import DocumentSpan, DocumentWindowSampler, encode_documents_with_spans
from forgelm.tokenizer import BytePairTokenizer
from forgelm.training import Trainer
from forgelm.wikitext import CONFIG, DATASET, REVISION, iter_articles, prepare_wikitext


def model_config() -> ModelConfig:
    return ModelConfig(
        d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=8,
        vocab_size=270, attention_impl="sdpa", qk_norm=True,
    )


class IntegratedTests(unittest.TestCase):
    def test_preparation_is_reused_and_detects_split_modification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("train", "validation", "test"):
                (root / f"{name}.txt").write_text(
                    f"{name} document has enough words for a short model window.\n\n"
                    f"Another {name} document provides a distinct example for this experiment.\n",
                    encoding="utf-8",
                )
            config_path = root / "experiment.toml"
            config_path.write_text(
                f'[project]\nname="prepare-test"\noutput_dir="{root / "artifacts"}"\n'
                f'[data]\ninput_path="{root / "train.txt"}"\n'
                f'validation_path="{root / "validation.txt"}"\n'
                f'test_path="{root / "test.txt"}"\nmin_words=2\n'
                'enable_quality_rules=false\nenable_near_dedup=false\nenable_decontamination=false\n'
                '[tokenizer]\nvocab_size=270\n'
                '[model]\nd_model=16\nn_layers=1\nn_heads=2\nd_ff=32\nmax_seq_len=8\n'
                '[training]\nsteps=1\nbatch_size=1\nseq_len=4\nwarmup_steps=0\n'
                'eval_interval=1\neval_batches=1\ncheckpoint_interval=1\ndevice="cpu"\n',
                encoding="utf-8",
            )
            first = run_pipeline(config_path, prepare_only=True)
            second = run_pipeline(config_path, prepare_only=True)
            self.assertEqual(first["run_fingerprint_id"], second["run_fingerprint_id"])
            prepared_test = root / "artifacts" / "dataset" / "test.jsonl"
            prepared_test.write_text(prepared_test.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "prepared data inputs changed"):
                run_pipeline(config_path, prepare_only=True)

    def test_static_cache_matches_full_forward_and_reuses_storage(self) -> None:
        torch.manual_seed(4)
        model = TransformerLM(model_config()).eval()
        ids = torch.randint(0, 270, (1, 7))
        with torch.inference_mode():
            full, _ = model(ids)
            caches = model.allocate_kv_cache(1, torch.device("cpu"))
            pointer = caches[0].key.data_ptr()
            prefill = model.forward_cached(ids[:, :4], caches)
            next_logits = model.forward_cached(ids[:, 4:6], caches)
            final_logits = model.forward_cached(ids[:, 6:], caches)
            torch.testing.assert_close(prefill, full[:, :4], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(next_logits, full[:, 4:6], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(final_logits, full[:, 6:], atol=2e-5, rtol=2e-5)
            self.assertEqual(caches[0].key.data_ptr(), pointer)
            self.assertEqual(caches[0].length, 7)
            uncached = model.generate(ids[:, :6], max_new_tokens=6, temperature=0, use_cache=False)
            cached = model.generate(ids[:, :6], max_new_tokens=6, temperature=0, use_cache=True)
            torch.testing.assert_close(cached, uncached)

    def test_document_sampler_never_crosses_boundaries(self) -> None:
        tokens = torch.tensor([1] * 10 + [2] * 10 + [3] * 10)
        spans = [DocumentSpan(i * 10, (i + 1) * 10, i / 2, i) for i in range(3)]
        sampler = DocumentWindowSampler(
            tokens, spans, 4, policy="quality_tier", tier_probabilities=(0.0, 0.0, 1.0)
        )
        inputs, targets = sampler.sample_batch(20, torch.device("cpu"), torch.Generator().manual_seed(7))
        self.assertTrue(bool(torch.all(inputs == 3)))
        self.assertTrue(bool(torch.all(targets == 3)))

    def test_wikitext_articles_preserve_heading_boundaries(self) -> None:
        source = ["= First =\n", "A @-@ B is here.\n", "\n", "= Second =\n", "Other text.\n"]
        self.assertEqual(list(iter_articles(source)), ["First A-B is here.", "Second Other text."])

    def test_wikitext_preparation_writes_hashed_article_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            for split in ("train", "validation", "test"):
                (raw / f"{split}.txt").write_text(
                    f"= {split} one =\nA first article.\n= {split} two =\nA second article.\n",
                    encoding="utf-8",
                )
            (raw / "download_manifest.json").write_text(json.dumps({
                "dataset": DATASET, "config": CONFIG, "revision": REVISION,
                "splits": {split: {"sha256": sha256_file(raw / f"{split}.txt")}
                           for split in ("train", "validation", "test")},
            }), encoding="utf-8")
            output = Path(directory) / "prepared"
            report = prepare_wikitext(raw, output, min_document_chars=8)
            self.assertEqual(report["splits"]["train"]["documents"], 2)
            self.assertTrue(report["splits"]["test"]["prepared_sha256"])
            self.assertIn("train one", (output / "train.txt").read_text())

    def test_fork_only_changes_sampling_policy(self) -> None:
        config = TrainingConfig(
            steps=2, batch_size=1, seq_len=4, warmup_steps=0, eval_interval=1,
            eval_batches=1, checkpoint_interval=1, device="cpu", sampling_policy="document_uniform",
        )
        tokens = torch.arange(90, dtype=torch.long) % 270
        spans = [DocumentSpan(i * 30, (i + 1) * 30, float(i), i) for i in range(3)]
        base_contract = {
            "dataset": {"train_sha256": "same"}, "tokenizer_sha256": "same",
            "quality_model_sha256": None, "model_config": dataclasses.asdict(model_config()),
            "seed": 7, "source_sha256": "same", "training_config": dataclasses.asdict(config),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = Trainer(
                TransformerLM(model_config()), tokens, tokens, config, seed=7,
                artifact_dir=root / "parent", train_spans=spans,
                run_fingerprint={"fingerprint_id": "parent", "contract": base_contract},
            )
            checkpoint = parent.train(stop_step=1)["checkpoint"]
            branch_config = dataclasses.replace(config, sampling_policy="quality_tier")
            branch_contract = {**base_contract, "training_config": dataclasses.asdict(branch_config)}
            branch = Trainer(
                TransformerLM(model_config()), tokens, tokens, branch_config, seed=7,
                artifact_dir=root / "branch", train_spans=spans,
                run_fingerprint={"fingerprint_id": "branch", "contract": branch_contract},
            )
            branch.fork_from_checkpoint(checkpoint)
            self.assertEqual(branch.step, 1)
            self.assertEqual(branch.parent_checkpoint["step"], 1)
            branch.train()
            self.assertEqual(branch.step, 2)
            invalid_config = dataclasses.replace(branch_config, learning_rate=2e-3)
            invalid_contract = {**branch_contract, "training_config": dataclasses.asdict(invalid_config)}
            invalid = Trainer(
                TransformerLM(model_config()), tokens, tokens, invalid_config, seed=7,
                artifact_dir=root / "invalid", train_spans=spans,
                run_fingerprint={"fingerprint_id": "invalid", "contract": invalid_contract},
            )
            with self.assertRaisesRegex(ValueError, "only change sampling_policy"):
                invalid.fork_from_checkpoint(checkpoint)

    def test_full_corpus_evaluates_each_target_once(self) -> None:
        documents = [Document("a", "a story about a careful experiment", "test", {})]
        tokenizer = BytePairTokenizer.train((doc.text for doc in documents), 270)
        tokens, spans = encode_documents_with_spans((doc.text for doc in documents), tokenizer, 4)
        config = dataclasses.replace(model_config(), vocab_size=tokenizer.vocab_size)
        training = TrainingConfig(
            steps=1, batch_size=1, seq_len=4, warmup_steps=0, eval_interval=1,
            eval_batches=1, checkpoint_interval=1, device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer.save(root / "tokenizer.json")
            write_jsonl(root / "validation.jsonl", documents)
            trainer = Trainer(TransformerLM(config), tokens, tokens, training, seed=4, artifact_dir=root)
            checkpoint = trainer.train()["checkpoint"]
            report = evaluate_checkpoint(
                checkpoint_path=checkpoint, tokenizer_path=root / "tokenizer.json",
                validation_jsonl=root / "validation.jsonl", device_name="cpu", batch_size=2,
                seq_len=4, eval_batches=1, seed=4, precision="fp32", full_corpus=True,
            )
            self.assertEqual(report["evaluated_tokens"], len(tokens))
            self.assertEqual(report["evaluation_mode"], "full_corpus")
            self.assertTrue(report["document_bounded"])
            self.assertGreater(report["bits_per_byte"], 0)

    def test_full_corpus_scores_each_document_from_eos(self) -> None:
        documents = [
            Document("a", "first short story", "test", {}),
            Document("b", "second short story", "test", {}),
        ]
        tokenizer = BytePairTokenizer.train((doc.text for doc in documents), 270)
        config = dataclasses.replace(model_config(), vocab_size=tokenizer.vocab_size, max_seq_len=64)
        training = TrainingConfig(
            steps=1, batch_size=1, seq_len=4, warmup_steps=0, eval_interval=1,
            eval_batches=1, checkpoint_interval=1, device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer.save(root / "tokenizer.json")
            write_jsonl(root / "validation.jsonl", documents)
            tokens, _ = encode_documents_with_spans((doc.text for doc in documents), tokenizer, 4)
            trainer = Trainer(TransformerLM(config), tokens, tokens, training, seed=4, artifact_dir=root)
            checkpoint = trainer.train()["checkpoint"]
            report = evaluate_checkpoint(
                checkpoint_path=checkpoint, tokenizer_path=root / "tokenizer.json",
                validation_jsonl=root / "validation.jsonl", device_name="cpu", batch_size=2,
                seq_len=64, eval_batches=1, seed=4, precision="fp32", full_corpus=True,
            )
            self.assertEqual(report["evaluated_tokens"], len(tokens))
            model = trainer.model.eval()
            manual_sum = 0.0
            with torch.inference_mode():
                for document in documents:
                    ids = tokenizer.encode(document.text, add_eos=True)
                    inputs = torch.tensor([[tokenizer.EOS_ID, *ids[:-1]]])
                    targets = torch.tensor(ids)
                    logits, _ = model(inputs)
                    manual_sum += float(torch.nn.functional.cross_entropy(
                        logits[0], targets, reduction="sum"
                    ))
            self.assertAlmostEqual(report["cross_entropy_loss"], manual_sum / len(tokens), places=5)


if __name__ == "__main__":
    unittest.main()
