from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


PRETOKEN_RE = re.compile(r"\s+|[\w]+|[^\w\s]+", re.UNICODE)


class BytePairTokenizer:
    """A deterministic byte-level BPE tokenizer with explicit PAD and EOS tokens."""

    PAD_ID = 256
    EOS_ID = 257
    FIRST_MERGE_ID = 258

    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges = list(merges or [])
        self._id_to_bytes: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
        for token_id, (left, right) in enumerate(self.merges, start=self.FIRST_MERGE_ID):
            self._id_to_bytes[token_id] = self._id_to_bytes[left] + self._id_to_bytes[right]

    @property
    def vocab_size(self) -> int:
        return self.FIRST_MERGE_ID + len(self.merges)

    @staticmethod
    def _pretokenize(text: str) -> list[bytes]:
        return [match.group(0).encode("utf-8") for match in PRETOKEN_RE.finditer(text)]

    @staticmethod
    def _merge_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        merged: list[int] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
                merged.append(new_id)
                index += 2
            else:
                merged.append(tokens[index])
                index += 1
        return merged

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "BytePairTokenizer":
        if vocab_size < cls.FIRST_MERGE_ID:
            raise ValueError("vocab_size must be at least 258")
        piece_counts: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            for piece in cls._pretokenize(text):
                if piece:
                    piece_counts[tuple(piece)] += 1

        merges: list[tuple[int, int]] = []
        mutable_pieces = {piece: count for piece, count in piece_counts.items()}
        next_id = cls.FIRST_MERGE_ID
        while next_id < vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for piece, frequency in mutable_pieces.items():
                for index in range(len(piece) - 1):
                    pair_counts[(piece[index], piece[index + 1])] += frequency
            if not pair_counts:
                break
            best_frequency = max(pair_counts.values())
            best_pair = min(pair for pair, count in pair_counts.items() if count == best_frequency)
            merges.append(best_pair)
            mutable_pieces = {
                tuple(cls._merge_pair(list(piece), best_pair, next_id)): frequency
                for piece, frequency in mutable_pieces.items()
            }
            next_id += 1
        return cls(merges)

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        encoded: list[int] = []
        for piece in self._pretokenize(text):
            tokens = list(piece)
            for merge_index, pair in enumerate(self.merges, start=self.FIRST_MERGE_ID):
                tokens = self._merge_pair(tokens, pair, merge_index)
            encoded.extend(tokens)
        if add_eos:
            encoded.append(self.EOS_ID)
        return encoded

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        chunks: list[bytes] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in {self.PAD_ID, self.EOS_ID}:
                if skip_special_tokens:
                    continue
                chunks.append(b"<|pad|>" if token_id == self.PAD_ID else b"<|eos|>")
                continue
            if token_id not in self._id_to_bytes:
                raise ValueError(f"unknown token id: {token_id}")
            chunks.append(self._id_to_bytes[token_id])
        return b"".join(chunks).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "type": "byte_bpe",
            "pad_id": self.PAD_ID,
            "eos_id": self.EOS_ID,
            "merges": [list(pair) for pair in self.merges],
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BytePairTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "byte_bpe" or payload.get("version") != 1:
            raise ValueError("unsupported tokenizer format")
        return cls([tuple(map(int, pair)) for pair in payload["merges"]])

