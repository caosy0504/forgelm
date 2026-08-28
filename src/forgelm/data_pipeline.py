from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from .config import DataConfig


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)")
IPV4_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")


class QualityScorer(Protocol):
    def score(self, text: str) -> float: ...


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    source: str
    pii_replacements: dict[str, int]
    quality_score: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class PipelineReport:
    input_path: str
    raw_documents: int
    raw_train_documents: int
    quality_dropped: int
    quality_drop_reasons: dict[str, int]
    model_quality_dropped: int
    exact_duplicates_dropped: int
    near_duplicates_dropped: int
    decontaminated_documents_dropped: int
    kept_documents: int
    train_documents: int
    validation_documents: int
    test_documents: int
    pii_replacements: dict[str, int]
    model_quality_scores: dict[str, float] | None
    config: dict[str, object]


def normalize_text(text: str) -> str:
    """Apply explicit compatibility normalization and stable whitespace handling."""
    text = unicodedata.normalize("NFKC", text).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    counts: Counter[str] = Counter()

    def replace_email(_: re.Match[str]) -> str:
        counts["email"] += 1
        return "|||EMAIL_ADDRESS|||"

    def replace_phone(_: re.Match[str]) -> str:
        counts["phone"] += 1
        return "|||PHONE_NUMBER|||"

    def replace_ip(match: re.Match[str]) -> str:
        candidate = match.group(0)
        if all(0 <= int(part) <= 255 for part in candidate.split(".")):
            counts["ipv4"] += 1
            return "|||IP_ADDRESS|||"
        return candidate

    text = EMAIL_RE.sub(replace_email, text)
    text = PHONE_RE.sub(replace_phone, text)
    text = IPV4_RE.sub(replace_ip, text)
    return text, {key: counts.get(key, 0) for key in ("email", "phone", "ipv4")}


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def normalized_words(text: str) -> list[str]:
    """Produce Unicode-aware dedup units; CJK characters become individual units."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    units: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            units.append("".join(buffer))
            buffer.clear()

    for character in normalized:
        if _is_cjk(character):
            flush()
            units.append(character)
        elif character.isalnum() or unicodedata.category(character).startswith("M"):
            buffer.append(character)
        else:
            flush()
    flush()
    return units


def canonical_dedup_text(text: str) -> str:
    units = normalized_words(text)
    return " ".join(units) if units else normalize_text(text).casefold()


def quality_check(text: str, config: DataConfig) -> tuple[bool, str | None]:
    words = normalized_words(text)
    if len(words) < config.min_words:
        return False, "too_few_words"
    visible = [char for char in text if not char.isspace()]
    alpha_ratio = sum(char.isalpha() for char in visible) / max(1, len(visible))
    if alpha_ratio < config.min_alpha_ratio:
        return False, "low_alpha_ratio"
    if len(set(words)) == 1 and len(words) > 4:
        return False, "repetitive_text"
    return True, None


def iter_raw_documents(path: str | Path) -> Iterator[str]:
    """Stream blank-line-separated documents without reading the full corpus into memory."""
    paragraph: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                paragraph.append(line.rstrip("\n"))
            elif paragraph:
                yield "\n".join(paragraph).strip()
                paragraph.clear()
        if paragraph:
            yield "\n".join(paragraph).strip()


def read_raw_documents(path: str | Path) -> list[str]:
    return list(iter_raw_documents(path))


def stable_digest(text: str, digest_size: int = 12) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=digest_size).hexdigest()


def word_ngrams(text: str, n: int) -> set[str]:
    words = normalized_words(text)
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[index : index + n]) for index in range(len(words) - n + 1)}


def _seeded_hash(value: str, seed: int) -> int:
    payload = seed.to_bytes(4, "little", signed=False) + value.encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def minhash_signature(shingles: set[str], num_hashes: int) -> tuple[int, ...]:
    if not shingles:
        return tuple(0 for _ in range(num_hashes))
    return tuple(min(_seeded_hash(shingle, seed) for shingle in shingles) for seed in range(num_hashes))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0 if left == right else 0.0
    return len(left & right) / len(union)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def near_deduplicate(
    documents: list[Document],
    *,
    threshold: float,
    ngram_size: int,
    num_hashes: int,
    num_bands: int,
) -> tuple[list[Document], int]:
    """Deterministically retain one document per MinHash/LSH duplicate cluster."""
    if len(documents) < 2:
        return documents, 0
    rows_per_band = num_hashes // num_bands
    shingles = [word_ngrams(document.text, ngram_size) for document in documents]
    signatures = [minhash_signature(items, num_hashes) for items in shingles]
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for doc_index, signature in enumerate(signatures):
        for band in range(num_bands):
            start = band * rows_per_band
            buckets[(band, signature[start : start + rows_per_band])].append(doc_index)

    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        for left_offset, left in enumerate(members):
            for right in members[left_offset + 1 :]:
                candidates.add((left, right))

    groups = _UnionFind(len(documents))
    for left, right in sorted(candidates):
        if shingles[left] and shingles[right] and jaccard(shingles[left], shingles[right]) >= threshold:
            groups.union(left, right)

    kept: list[Document] = []
    seen_roots: set[int] = set()
    for index, document in enumerate(documents):
        root = groups.find(index)
        if root not in seen_roots:
            kept.append(document)
            seen_roots.add(root)
    return kept, len(documents) - len(kept)


def split_documents(documents: list[Document], validation_fraction: float) -> tuple[list[Document], list[Document]]:
    if len(documents) < 2:
        raise ValueError("at least two documents are required for a leakage-safe train/validation split")
    ordered = sorted(documents, key=lambda item: stable_digest(item.doc_id, digest_size=8))
    validation_count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    validation_ids = {document.doc_id for document in ordered[:validation_count]}
    train = [document for document in documents if document.doc_id not in validation_ids]
    validation = [document for document in documents if document.doc_id in validation_ids]
    return train, validation


def decontaminate_documents(
    train: list[Document], validation: list[Document], *, threshold: float, ngram_size: int
) -> tuple[list[Document], int]:
    validation_shingles = [word_ngrams(document.text, ngram_size) for document in validation]
    inverted: dict[str, set[int]] = defaultdict(set)
    for validation_index, shingles in enumerate(validation_shingles):
        for shingle in shingles:
            inverted[shingle].add(validation_index)
    kept: list[Document] = []
    dropped = 0
    for document in train:
        shingles = word_ngrams(document.text, ngram_size)
        candidates: set[int] = set()
        for shingle in shingles:
            candidates.update(inverted.get(shingle, ()))
        contaminated = any(
            jaccard(shingles, validation_shingles[index]) >= threshold for index in candidates if shingles
        )
        if contaminated:
            dropped += 1
        else:
            kept.append(document)
    return kept, dropped


def write_jsonl(path: Path, documents: Iterable[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(document.to_json() + "\n")


def read_jsonl(path: str | Path) -> Iterator[Document]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield Document(**json.loads(line))


def _config_manifest(config: DataConfig) -> dict[str, object]:
    return asdict(config)


def prepare_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    config: DataConfig,
    quality_classifier: QualityScorer | None = None,
    validation_path: str | Path | None = None,
    test_path: str | Path | None = None,
) -> PipelineReport:
    pii_totals: Counter[str] = Counter()
    def ingest(path: str | Path, prefix: str) -> list[Document]:
        documents: list[Document] = []
        for index, raw in enumerate(iter_raw_documents(path)):
            normalized = normalize_text(raw)
            if config.enable_pii_masking:
                cleaned, counts = mask_pii(normalized)
            else:
                cleaned, counts = normalized, {"email": 0, "phone": 0, "ipv4": 0}
            pii_totals.update(counts)
            documents.append(
                Document(
                    doc_id=f"{prefix}-{index:06d}-{stable_digest(normalized, digest_size=6)}",
                    text=cleaned,
                    source=str(Path(path).name),
                    pii_replacements=counts,
                )
            )
        return documents

    input_documents = ingest(input_path, "train-source")
    if validation_path is None:
        raw_train, validation = split_documents(input_documents, config.validation_fraction)
        test: list[Document] = []
        raw_count = len(input_documents)
    else:
        raw_train = input_documents
        validation = ingest(validation_path, "validation-source")
        if not validation:
            raise ValueError("external validation corpus contains no documents")
        test = ingest(test_path, "test-source") if test_path is not None else []
        if test_path is not None and not test:
            raise ValueError("external test corpus contains no documents")
        raw_count = len(raw_train) + len(validation) + len(test)
    train = raw_train
    quality_reasons: Counter[str] = Counter()
    if config.enable_quality_rules:
        quality_kept: list[Document] = []
        for document in train:
            passes, reason = quality_check(document.text, config)
            if passes:
                quality_kept.append(document)
            else:
                quality_reasons[reason or "unknown"] += 1
        train = quality_kept

    quality_scores: list[float] = []
    model_quality_dropped = 0
    if config.enable_model_quality:
        if quality_classifier is None:
            raise ValueError("model-quality filtering is enabled but no classifier was supplied")
        model_kept: list[Document] = []
        for document in train:
            score = quality_classifier.score(document.text)
            quality_scores.append(score)
            scored_document = replace(document, quality_score=score)
            if score >= config.quality_threshold:
                model_kept.append(scored_document)
            else:
                model_quality_dropped += 1
        train = model_kept

    exact_dropped = 0
    if config.enable_exact_dedup:
        exact_kept: list[Document] = []
        seen_exact: set[str] = set()
        for document in train:
            key = stable_digest(canonical_dedup_text(document.text))
            if key in seen_exact:
                exact_dropped += 1
            else:
                seen_exact.add(key)
                exact_kept.append(document)
        train = exact_kept

    near_dropped = 0
    if config.enable_near_dedup:
        train, near_dropped = near_deduplicate(
            train,
            threshold=config.near_dedup_threshold,
            ngram_size=config.ngram_size,
            num_hashes=config.num_hashes,
            num_bands=config.num_bands,
        )

    decontaminated_dropped = 0
    if config.enable_decontamination:
        train, decontaminated_dropped = decontaminate_documents(
            train,
            validation + test,
            threshold=config.decontamination_threshold,
            ngram_size=config.ngram_size,
        )

    destination = Path(output_dir)
    write_jsonl(destination / "train.jsonl", train)
    write_jsonl(destination / "validation.jsonl", validation)
    if test:
        write_jsonl(destination / "test.jsonl", test)
    score_summary = None
    if quality_scores:
        score_summary = {
            "min": min(quality_scores),
            "mean": sum(quality_scores) / len(quality_scores),
            "max": max(quality_scores),
        }
    report = PipelineReport(
        input_path=str(Path(input_path).resolve()),
        raw_documents=raw_count,
        raw_train_documents=len(raw_train),
        quality_dropped=sum(quality_reasons.values()),
        quality_drop_reasons=dict(sorted(quality_reasons.items())),
        model_quality_dropped=model_quality_dropped,
        exact_duplicates_dropped=exact_dropped,
        near_duplicates_dropped=near_dropped,
        decontaminated_documents_dropped=decontaminated_dropped,
        kept_documents=len(train) + len(validation) + len(test),
        train_documents=len(train),
        validation_documents=len(validation),
        test_documents=len(test),
        pii_replacements={key: pii_totals.get(key, 0) for key in ("email", "phone", "ipv4")},
        model_quality_scores=score_summary,
        config=_config_manifest(config),
    )
    (destination / "dataset_manifest.json").write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
