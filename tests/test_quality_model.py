from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from forgelm.quality_model import HashedNgramQualityClassifier, QualityExample


class QualityModelTests(unittest.TestCase):
    def test_classifier_separates_transparent_seed_examples(self) -> None:
        examples = [
            QualityExample("A careful report explains the method, measurements, and limitations.", 1),
            QualityExample("A tutorial provides definitions, examples, and a reproducible procedure.", 1),
            QualityExample("CLICK BUY FREE FREE advertisement navigation navigation", 0),
            QualityExample("login cookie footer footer sidebar sponsored download", 0),
        ]
        classifier = HashedNgramQualityClassifier(256)
        classifier.fit(examples, epochs=120, learning_rate=0.3)
        high = classifier.score("A technical explanation reports measurements and limitations.")
        low = classifier.score("BUY FREE sponsored advertisement footer navigation")
        self.assertGreater(high, low)

    def test_serialization_preserves_score(self) -> None:
        classifier = HashedNgramQualityClassifier(64)
        classifier.bias = 0.25
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.json"
            classifier.save(path)
            restored = HashedNgramQualityClassifier.load(path)
            self.assertAlmostEqual(classifier.score("sample text"), restored.score("sample text"))


if __name__ == "__main__":
    unittest.main()
