from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.audit_sampling_quality import Record, _human_label_summary, _length_stratified_tiers


class LengthBalancedAuditTests(unittest.TestCase):
    def test_candidate_removes_length_tier_imbalance_without_changing_scores_or_windows(self) -> None:
        # A globally ranked score would concentrate high-tier documents in long articles.
        records = [
            Record(
                doc_id=f"doc-{index:03d}",
                score=0.1 * (index // 30) + (index % 30) / 1000,
                current_tier=0 if index < 40 else 1 if index < 80 else 2,
                valid_windows=100 * (1 + index // 30),
                tokens=1024 + 100 * index,
                utf8_bytes=2000 + 120 * index,
            )
            for index in range(120)
        ]
        candidate = _length_stratified_tiers(records)
        self.assertEqual(
            [(r.doc_id, r.score, r.valid_windows) for r in records],
            [(r.doc_id, r.score, r.valid_windows) for r in candidate],
        )
        for stratum in range(4):
            members = [r for r in candidate if r.length_stratum == stratum]
            self.assertEqual(len(members), 30)
            total = sum(r.valid_windows for r in members)
            for tier in range(3):
                fraction = sum(r.valid_windows for r in members if r.candidate_tier == tier) / total
                self.assertAlmostEqual(fraction, 1 / 3, places=6)

    def test_not_enough_documents_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "eligible document"):
            _length_stratified_tiers([], strata=4)

    def test_partial_human_labels_are_separate_from_hidden_tier_key(self) -> None:
        lookup = {
            "a": Record("a", 0.2, 0, 3, 1200, 2000, 0, 2),
            "b": Record("b", 0.8, 2, 4, 1300, 2200, 1, 0),
        }
        with tempfile.TemporaryDirectory() as directory:
            sheet = Path(directory) / "review.csv"
            with sheet.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("doc_id", "human_label"))
                writer.writeheader()
                writer.writerows(({"doc_id": "a", "human_label": "2"}, {"doc_id": "b", "human_label": ""}))
            report = _human_label_summary(sheet, lookup, set(lookup))
            self.assertEqual(report["labeled"], 1)
            self.assertEqual(report["by_policy"]["current_tier"]["low"]["mean_human_label"], 2)
            self.assertEqual(report["by_policy"]["candidate_tier"]["high"]["mean_human_label"], 2)
            self.assertIsNone(report["by_policy"]["current_tier"]["high"]["mean_human_label"])


if __name__ == "__main__":
    unittest.main()
