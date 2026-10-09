from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class QualityExample:
    text: str
    label: int


@dataclass(frozen=True)
class QualityModelReport:
    train_examples: int
    validation_examples: int
    validation_accuracy: float
    validation_precision: float
    validation_recall: float
    positive_rate: float
    threshold: float
    feature_dimension: int


def load_quality_examples(path: str | Path) -> list[QualityExample]:
    examples: list[QualityExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            label = int(payload["label"])
            if label not in {0, 1}:
                raise ValueError(f"quality label on line {line_number} must be 0 or 1")
            text = str(payload["text"]).strip()
            if not text:
                raise ValueError(f"quality text on line {line_number} is empty")
            examples.append(QualityExample(text=text, label=label))
    if len(examples) < 8 or {example.label for example in examples} != {0, 1}:
        raise ValueError("quality seed data needs at least eight examples and both labels")
    label_counts = {label: sum(example.label == label for example in examples) for label in (0, 1)}
    if min(label_counts.values()) < 4:
        raise ValueError("quality seed data needs at least four examples per class")
    return examples


def _stable_bucket(text: str, modulo: int) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little") % modulo


def split_quality_examples(
    examples: Iterable[QualityExample], validation_fraction: float = 0.25
) -> tuple[list[QualityExample], list[QualityExample]]:
    by_label = {0: [], 1: []}
    for example in examples:
        by_label[example.label].append(example)
    train: list[QualityExample] = []
    validation: list[QualityExample] = []
    for label, members in by_label.items():
        ordered = sorted(members, key=lambda item: _stable_bucket(item.text, 2**31 - 1))
        validation_count = max(1, round(len(ordered) * validation_fraction))
        validation.extend(ordered[:validation_count])
        train.extend(ordered[validation_count:])
    return train, validation


class HashedNgramQualityClassifier:
    """Transparent logistic baseline over dense document statistics and hashed character n-grams."""

    FORMAT_VERSION = 1
    DENSE_FEATURES = 8

    def __init__(self, feature_dimension: int = 2048) -> None:
        if feature_dimension < 32:
            raise ValueError("feature_dimension must be at least 32")
        self.feature_dimension = feature_dimension
        self.weights = np.zeros(feature_dimension, dtype=np.float64)
        self.bias = 0.0

    def _features(self, text: str) -> np.ndarray:
        normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()
        vector = np.zeros(self.feature_dimension, dtype=np.float64)
        visible = [char for char in normalized if not char.isspace()]
        words = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        unique_ratio = len(set(words)) / max(1, len(words))
        vector[: self.DENSE_FEATURES] = (
            math.log1p(len(normalized)) / 10.0,
            sum(char.isalpha() for char in visible) / max(1, len(visible)),
            sum(char.isdigit() for char in visible) / max(1, len(visible)),
            sum(unicodedata.category(char).startswith("P") for char in visible) / max(1, len(visible)),
            unique_ratio,
            sum(len(word) for word in words) / max(1, len(words)) / 20.0,
            min(normalized.count("http") + normalized.count("www."), 10) / 10.0,
            min(max((normalized.count(char * 4) for char in set(normalized)), default=0), 10) / 10.0,
        )
        padded = f"  {normalized}  "
        for ngram_size in (3, 4, 5):
            for index in range(max(0, len(padded) - ngram_size + 1)):
                ngram = padded[index : index + ngram_size]
                digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest()
                bucket = self.DENSE_FEATURES + int.from_bytes(digest[:4], "little") % (
                    self.feature_dimension - self.DENSE_FEATURES
                )
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[bucket] += sign
        tail_norm = np.linalg.norm(vector[self.DENSE_FEATURES :])
        if tail_norm > 0:
            vector[self.DENSE_FEATURES :] /= tail_norm
        return vector

    @staticmethod
    def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
        clipped = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def fit(
        self,
        examples: Iterable[QualityExample],
        *,
        epochs: int = 40,
        learning_rate: float = 0.1,
        l2: float = 1e-4,
    ) -> None:
        examples = list(examples)
        matrix = np.stack([self._features(example.text) for example in examples])
        labels = np.array([example.label for example in examples], dtype=np.float64)
        positive_weight = len(labels) / max(1.0, 2.0 * labels.sum())
        negative_weight = len(labels) / max(1.0, 2.0 * (len(labels) - labels.sum()))
        sample_weights = np.where(labels == 1.0, positive_weight, negative_weight)
        for epoch in range(epochs):
            probabilities = self._sigmoid(matrix @ self.weights + self.bias)
            errors = (probabilities - labels) * sample_weights
            rate = learning_rate / math.sqrt(1.0 + epoch / 10.0)
            self.weights -= rate * (matrix.T @ errors / len(labels) + l2 * self.weights)
            self.bias -= rate * float(errors.mean())

    def score(self, text: str) -> float:
        return float(self._sigmoid(float(self._features(text) @ self.weights + self.bias)))

    def evaluate(self, examples: Iterable[QualityExample], threshold: float) -> dict[str, float]:
        examples = list(examples)
        labels = np.array([example.label for example in examples], dtype=np.int64)
        predictions = np.array([self.score(example.text) >= threshold for example in examples], dtype=np.int64)
        true_positive = int(((predictions == 1) & (labels == 1)).sum())
        false_positive = int(((predictions == 1) & (labels == 0)).sum())
        false_negative = int(((predictions == 0) & (labels == 1)).sum())
        return {
            "accuracy": float((predictions == labels).mean()),
            "precision": true_positive / max(1, true_positive + false_positive),
            "recall": true_positive / max(1, true_positive + false_negative),
        }

    def save(self, path: str | Path, report: QualityModelReport | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "model_type": "hashed_ngram_logistic_regression",
            "feature_dimension": self.feature_dimension,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "report": asdict(report) if report is not None else None,
        }
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HashedNgramQualityClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported quality-model format")
        model = cls(int(payload["feature_dimension"]))
        model.weights = np.asarray(payload["weights"], dtype=np.float64)
        model.bias = float(payload["bias"])
        return model


def train_quality_classifier(
    seed_path: str | Path,
    *,
    feature_dimension: int,
    epochs: int,
    learning_rate: float,
    threshold: float,
    output_path: str | Path | None = None,
) -> tuple[HashedNgramQualityClassifier, QualityModelReport]:
    examples = load_quality_examples(seed_path)
    train, validation = split_quality_examples(examples)
    classifier = HashedNgramQualityClassifier(feature_dimension)
    classifier.fit(train, epochs=epochs, learning_rate=learning_rate)
    metrics = classifier.evaluate(validation, threshold)
    report = QualityModelReport(
        train_examples=len(train),
        validation_examples=len(validation),
        validation_accuracy=metrics["accuracy"],
        validation_precision=metrics["precision"],
        validation_recall=metrics["recall"],
        positive_rate=sum(example.label for example in examples) / len(examples),
        threshold=threshold,
        feature_dimension=feature_dimension,
    )
    if output_path is not None:
        classifier.save(output_path, report)
    return classifier, report
