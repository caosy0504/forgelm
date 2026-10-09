"""Document-bounded training windows and transparent quality-tier sampling."""

from __future__ import annotations

import bisect
import math
import re
from array import array
from dataclasses import dataclass
from typing import Iterable

import torch

from .tokenizer import BytePairTokenizer


WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)


@dataclass(frozen=True)
class DocumentSpan:
    start: int
    end: int
    score: float
    tier: int = -1

    def valid_starts(self, seq_len: int) -> int:
        return max(0, self.end - self.start - seq_len)


def prose_quality_features(text: str) -> dict[str, float | int]:
    """Transparent prose-shape features; none measures factual correctness."""
    words = WORD_RE.findall(text)
    visible = [char for char in text if not char.isspace()]
    alpha = sum(char.isalpha() for char in visible) / max(1, len(visible))
    sentence_count = max(1, sum(text.count(mark) for mark in ".!?"))
    words_per_sentence = len(words) / sentence_count
    sentence_score = 1.0 / (1.0 + abs(math.log(max(words_per_sentence, 1.0) / 25.0)))
    length = min(1.0, len(words) / 256.0)
    alpha_score = min(1.0, max(0.0, (alpha - 0.55) / 0.40))
    folded = [word.casefold() for word in words]
    window_size = 256
    starts = [0] if len(folded) <= window_size else [0, (len(folded) - window_size) // 2, len(folded) - window_size]
    windows = [folded[start : start + window_size] for start in starts]
    unique_ratio = sum(len(set(window)) / max(1, len(window)) for window in windows) / len(windows)
    trigram_uniqueness = sum(
        len(set(zip(window, window[1:], window[2:]))) / max(1, len(window) - 2)
        for window in windows
    ) / len(windows)
    score = round(
        0.15 * length + 0.15 * alpha_score + 0.25 * sentence_score
        + 0.30 * unique_ratio + 0.15 * trigram_uniqueness,
        8,
    )
    return {
        "score": score,
        "word_count": len(words),
        "utf8_bytes": len(text.encode("utf-8")),
        "length_score": round(length, 8),
        "alphabetic_score": round(alpha_score, 8),
        "sentence_density": round(sentence_score, 8),
        "unique_word_ratio": round(unique_ratio, 8),
        "trigram_uniqueness": round(trigram_uniqueness, 8),
    }


def prose_quality_score(text: str) -> float:
    return float(prose_quality_features(text)["score"])


def encode_documents_with_spans(
    texts: Iterable[str], tokenizer: BytePairTokenizer, seq_len: int
) -> tuple[torch.Tensor, list[DocumentSpan]]:
    token_ids = array("q")
    spans: list[DocumentSpan] = []
    for text in texts:
        start = len(token_ids)
        token_ids.extend(tokenizer.encode(text, add_eos=True))
        spans.append(DocumentSpan(start, len(token_ids), prose_quality_score(text)))
    if not token_ids:
        raise ValueError("no tokens were produced")
    return torch.frombuffer(token_ids, dtype=torch.int64), assign_quality_tiers(spans, seq_len)


def assign_quality_tiers(spans: list[DocumentSpan], seq_len: int) -> list[DocumentSpan]:
    """Assign thirds by the mass of valid token-window starts."""
    eligible = sorted(
        (index for index, span in enumerate(spans) if span.valid_starts(seq_len) > 0),
        key=lambda index: (spans[index].score, index),
    )
    total = sum(spans[index].valid_starts(seq_len) for index in eligible)
    cumulative = 0
    tiers = [-1] * len(spans)
    for index in eligible:
        weight = spans[index].valid_starts(seq_len)
        midpoint = (cumulative + weight / 2) / total
        tiers[index] = 0 if midpoint <= 1 / 3 else 1 if midpoint <= 2 / 3 else 2
        cumulative += weight
    return [
        DocumentSpan(span.start, span.end, span.score, tiers[index]) for index, span in enumerate(spans)
    ]


def rescore_document_spans(
    texts: Iterable[str], spans: list[DocumentSpan], seq_len: int
) -> list[DocumentSpan]:
    """Recompute quality tiers without changing already encoded token boundaries."""
    records = list(texts)
    if len(records) != len(spans):
        raise ValueError("document count differs from encoded token spans")
    rescored = [
        DocumentSpan(span.start, span.end, prose_quality_score(text))
        for text, span in zip(records, spans, strict=True)
    ]
    return assign_quality_tiers(rescored, seq_len)


class DocumentWindowSampler:
    def __init__(
        self,
        tokens: torch.Tensor,
        spans: list[DocumentSpan],
        seq_len: int,
        *,
        policy: str,
        tier_probabilities: tuple[float, float, float],
    ) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        self.policy = policy
        self.tier_probabilities = tier_probabilities
        self.tier_draw_counts = [0, 0, 0]
        self.eligible = [span for span in spans if span.valid_starts(seq_len) > 0]
        if not self.eligible:
            raise ValueError("no document has enough tokens for a training window")
        self.cumulative: dict[int, list[int]] = {}
        self.members: dict[int, list[DocumentSpan]] = {}
        for tier in (-1, 0, 1, 2):
            members = self.eligible if tier == -1 else [span for span in self.eligible if span.tier == tier]
            running = 0
            cumulative: list[int] = []
            for span in members:
                running += span.valid_starts(seq_len)
                cumulative.append(running)
            self.members[tier] = members
            self.cumulative[tier] = cumulative
        if policy == "quality_tier":
            for tier, probability in enumerate(tier_probabilities):
                if probability > 0 and not self.members[tier]:
                    raise ValueError(f"quality tier {tier} has no eligible documents")

    def _pick_span(self, tier: int, generator: torch.Generator) -> DocumentSpan:
        cumulative = self.cumulative[tier]
        ticket = int(torch.randint(cumulative[-1], (1,), generator=generator).item())
        return self.members[tier][bisect.bisect_right(cumulative, ticket)]

    def sample_batch(
        self, batch_size: int, device: torch.device, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        starts: list[int] = []
        probabilities = torch.tensor(self.tier_probabilities, dtype=torch.float64)
        for _ in range(batch_size):
            tier = (
                int(torch.multinomial(probabilities, 1, generator=generator).item())
                if self.policy == "quality_tier"
                else -1
            )
            span = self._pick_span(tier, generator)
            if 0 <= span.tier < 3:
                self.tier_draw_counts[span.tier] += 1
            offset = int(torch.randint(span.valid_starts(self.seq_len), (1,), generator=generator).item())
            starts.append(span.start + offset)
        inputs = torch.stack([self.tokens[start : start + self.seq_len] for start in starts])
        targets = torch.stack([self.tokens[start + 1 : start + self.seq_len + 1] for start in starts])
        return inputs.to(device), targets.to(device)
