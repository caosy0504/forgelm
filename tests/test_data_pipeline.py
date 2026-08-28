from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from forgelm.config import DataConfig
from forgelm.data_pipeline import (
    Document,
    mask_pii,
    near_deduplicate,
    normalized_words,
    prepare_dataset,
)


class DataPipelineTests(unittest.TestCase):
    def test_test_path_requires_explicit_validation_path(self) -> None:
        config = DataConfig(input_path="train.txt", test_path="test.txt")
        with self.assertRaisesRegex(ValueError, "requires an explicit"):
            config.validate()

    def test_masks_supported_pii(self) -> None:
        masked, counts = mask_pii("Email a.person@example.com, call (415) 555-0199, or visit 10.0.0.7.")
        self.assertNotIn("a.person@example.com", masked)
        self.assertNotIn("415", masked)
        self.assertNotIn("10.0.0.7", masked)
        self.assertEqual(counts, {"email": 1, "phone": 1, "ipv4": 1})

    def test_near_deduplication_is_deterministic(self) -> None:
        documents = [
            Document("a", "the quick brown fox jumps over the quiet dog", "test", {}),
            Document("b", "the quick brown fox jumps over a quiet dog", "test", {}),
            Document("c", "a telescope maps distant stars above the mountain", "test", {}),
        ]
        kept, dropped = near_deduplicate(
            documents,
            threshold=0.45,
            ngram_size=2,
            num_hashes=32,
            num_bands=32,
        )
        self.assertEqual([document.doc_id for document in kept], ["a", "c"])
        self.assertEqual(dropped, 1)

    def test_prepare_dataset_writes_manifest_and_disjoint_splits(self) -> None:
        paragraphs = [
            "alpha beta gamma delta epsilon zeta eta theta",
            "one two three four five six seven eight",
            "red orange yellow green blue indigo violet color",
            "north south east west compass road travel map",
            "small medium large narrow wide shape object",
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            output = Path(directory) / "prepared"
            source.write_text("\n\n".join(paragraphs), encoding="utf-8")
            config = DataConfig(
                input_path=str(source),
                min_words=3,
                min_alpha_ratio=0.5,
                validation_fraction=0.2,
                near_dedup_threshold=1.0,
                ngram_size=2,
                num_hashes=8,
                num_bands=2,
            )
            report = prepare_dataset(source, output, config)
            self.assertEqual(report.kept_documents, 5)
            self.assertTrue((output / "dataset_manifest.json").exists())
            train_text = (output / "train.jsonl").read_text(encoding="utf-8")
            validation_text = (output / "validation.jsonl").read_text(encoding="utf-8")
            self.assertNotEqual(train_text, validation_text)

    def test_unicode_dedup_units_preserve_cjk_content(self) -> None:
        first = normalized_words("机器学习模型能够处理中文文本")
        second = normalized_words("天气预报说明明天可能下雨")
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertNotEqual(first, second)

    def test_external_validation_is_fixed_across_filter_variants(self) -> None:
        train_paragraphs = [
            "useful scientific explanation with measurements and a clear conclusion",
            "CLICK HERE BUY NOW FREE OFFER sponsored links advertisement",
            "another careful technical document with examples and limitations",
        ]
        validation_paragraphs = [
            "fixed evaluation document about experiments and repeatable measurements",
            "固定的中文验证文档用于比较不同的数据过滤方法",
        ]
        test_paragraphs = [
            "held out final document about evidence and careful conclusions",
            "最终测试文档在所有模型配置确定之前保持封存",
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.txt"
            validation = Path(directory) / "validation.txt"
            test = Path(directory) / "test.txt"
            source.write_text("\n\n".join(train_paragraphs), encoding="utf-8")
            validation.write_text("\n\n".join(validation_paragraphs), encoding="utf-8")
            test.write_text("\n\n".join(test_paragraphs), encoding="utf-8")
            base = DataConfig(
                input_path=str(source),
                validation_path=str(validation),
                min_words=2,
                min_alpha_ratio=0.0,
                ngram_size=2,
                num_hashes=8,
                num_bands=2,
                enable_decontamination=False,
            )
            first_output = Path(directory) / "first"
            second_output = Path(directory) / "second"
            first_report = prepare_dataset(source, first_output, base, validation_path=validation, test_path=test)
            second_report = prepare_dataset(
                source,
                second_output,
                DataConfig(**{**base.__dict__, "enable_quality_rules": False, "enable_exact_dedup": False}),
                validation_path=validation,
                test_path=test,
            )
            self.assertEqual(
                (first_output / "validation.jsonl").read_text(),
                (second_output / "validation.jsonl").read_text(),
            )
            self.assertEqual((first_output / "test.jsonl").read_text(), (second_output / "test.jsonl").read_text())
            self.assertEqual(first_report.test_documents, 2)
            self.assertEqual(second_report.test_documents, 2)


if __name__ == "__main__":
    unittest.main()
