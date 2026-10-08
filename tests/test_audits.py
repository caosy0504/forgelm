from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from forgelm.data_pipeline import Document, write_jsonl
from forgelm.comparison import summarize_seeds
from forgelm.fingerprint import sha256_file
from forgelm.quality_audit import audit_quality
from forgelm.sampling import encode_documents_with_spans
import forgelm.sampling as sampling_module
from forgelm.tokenizer import BytePairTokenizer
from forgelm.tokenizer_audit import audit_tokenizers


class AuditTests(unittest.TestCase):
    def test_multiseed_summary_requires_distinct_seeds_and_equal_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {
                "train_dataset_sha256": "train", "validation_dataset_sha256": "validation",
                "tokenizer_sha256": "tokenizer", "model_config_sha256": "model",
                "branch_steps": 200, "sampled_tokens_per_branch": 1000,
            }
            paths = []
            for seed, change in ((1, -0.01), (2, 0.02)):
                path = root / f"{seed}.json"
                path.write_text(json.dumps({**common, "seed": seed,
                                            "quality_relative_perplexity_change": change}))
                paths.append(path)
            report = summarize_seeds(paths, root / "summary.json")
            self.assertEqual(report["quality_ppl_improvement_runs"], 1)
            self.assertEqual(report["runs"], 2)
            with self.assertRaisesRegex(ValueError, "repeated training seed"):
                summarize_seeds([paths[0], paths[0]], root / "bad.json")

    def test_quality_labels_and_tokenizer_sweep_use_prepared_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifacts"
            data = artifact / "dataset"
            data.mkdir(parents=True)
            documents = [
                Document("a", "A careful scientific explanation contains many clear sentences. " * 3, "sample", {}),
                Document("b", "A simple story has a beginning and an ending. " * 3, "sample", {}),
                Document("c", "Another document provides a different example and useful details. " * 3, "sample", {}),
            ]
            write_jsonl(data / "train.jsonl", documents)
            write_jsonl(data / "validation.jsonl", documents[:1])
            tokenizer = BytePairTokenizer.train((doc.text for doc in documents), 270)
            tokenizer.save(artifact / "tokenizer.json")
            tokens, spans = encode_documents_with_spans((doc.text for doc in documents), tokenizer, 4)
            torch.save({"train_tokens": tokens, "validation_tokens": tokens[:40],
                        "spans": [(s.start, s.end, s.score, s.tier) for s in spans],
                        "validation_spans": []}, artifact / "token_cache.pt")
            (artifact / "token_cache_manifest.json").write_text(json.dumps({
                "seq_len": 4,
                "train_sha256": sha256_file(data / "train.jsonl"),
                "tokenizer_sha256": sha256_file(artifact / "tokenizer.json"),
                "cache_sha256": sha256_file(artifact / "token_cache.pt"),
                "sampling_source_sha256": sha256_file(sampling_module.__file__),
            }))
            config = root / "config.toml"
            config.write_text(
                f'[project]\nname="audit"\noutput_dir="{artifact}"\n'
                f'[data]\ninput_path="{root / "source.txt"}"\n'
                '[tokenizer]\nvocab_size=270\nmax_training_characters=1000\n'
                '[training]\nseq_len=4\n', encoding="utf-8",
            )
            report = audit_quality(config, sample_per_tier=1)
            self.assertEqual(report["eligible_documents"], 3)
            self.assertEqual(sum(tier["documents"] for tier in report["tiers"].values()), 3)
            sample_path = artifact / "quality_audit_samples.csv"
            with sample_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["human_label"] = "2"
            with sample_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            labeled = audit_quality(config, sample_per_tier=1, labels_path=sample_path)
            self.assertEqual(sum(value["labeled"] for value in labeled["human_labels"].values()), 1)
            token_report = audit_tokenizers(config, vocab_sizes=[270, 280])
            self.assertEqual(len(token_report["rows"]), 2)
            self.assertTrue(all(row["tokens_per_byte"] > 0 for row in token_report["rows"]))


if __name__ == "__main__":
    unittest.main()
