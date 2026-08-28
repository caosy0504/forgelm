from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from forgelm.tokenizer import BytePairTokenizer


class TokenizerTests(unittest.TestCase):
    def test_unicode_round_trip(self) -> None:
        texts = ["hello hello world", "你好，世界", "naïve café", "hello tokenizer"]
        tokenizer = BytePairTokenizer.train(texts, vocab_size=280)
        sample = "hello，世界! café\n"
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)
        self.assertLessEqual(tokenizer.vocab_size, 280)

    def test_save_and_load_preserves_encoding(self) -> None:
        tokenizer = BytePairTokenizer.train(["repeat repeat repeated", "another repeated token"], vocab_size=275)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = BytePairTokenizer.load(path)
            self.assertEqual(restored.encode("repeated token"), tokenizer.encode("repeated token"))
            self.assertEqual(restored.decode(restored.encode("repeated token")), "repeated token")


if __name__ == "__main__":
    unittest.main()

