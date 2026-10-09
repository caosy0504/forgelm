from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from forgelm.dataset_split import split_tinystories_file


class DatasetSplitTests(unittest.TestCase):
    def test_tinystories_split_is_deterministic_and_disjoint(self) -> None:
        complete = [f"Once upon a time story number {index} had a unique ending." for index in range(12)]
        complete[3] = "A story starts here.\n   \nIt continues after an internal blank line."
        source_text = "partial beginning<|endoftext|>" + "<|endoftext|>".join(complete)
        source_text += "<|endoftext|>partial trailing story"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text(source_text, encoding="utf-8")
            first = split_tinystories_file(source, root / "first", seed=17)
            second = split_tinystories_file(source, root / "second", seed=17)
            self.assertEqual(first["unique_complete_stories"], 12)
            written_document_count = 0
            for split in ("train", "validation", "test"):
                written_document_count += len(
                    [part for part in (root / "first" / f"{split}.txt").read_text().split("\n\n") if part.strip()]
                )
            self.assertEqual(written_document_count, 12)
            self.assertEqual(first["exact_cross_split_overlap"], {
                "train_validation": 0,
                "train_test": 0,
                "validation_test": 0,
            })
            for split in ("train", "validation", "test"):
                self.assertEqual(first["splits"][split]["sha256"], second["splits"][split]["sha256"])


if __name__ == "__main__":
    unittest.main()
