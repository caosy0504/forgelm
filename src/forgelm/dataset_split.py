from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


TINYSTORIES_SOURCE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories"
TINYSTORIES_LICENSE = "CDLA-Sharing-1.0"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_story(story: str) -> str:
    normalized = unicodedata.normalize("NFKC", story).strip()
    # ForgeLM uses blank lines as document boundaries, so internal blank lines
    # (including whitespace-only lines) must be collapsed inside each story.
    normalized = re.sub(r"\n[ \t]*\n+", "\n", normalized)
    return normalized


def _stable_order_key(story: str, seed: int) -> str:
    payload = f"{seed}\0{story}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_documents(path: Path, documents: list[str]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(documents) + "\n", encoding="utf-8")
    return {
        "path": path.name,
        "documents": len(documents),
        "characters": sum(len(document) for document in documents),
        "words": sum(len(document.split()) for document in documents),
        "sha256": sha256_file(path),
    }


def split_tinystories_file(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    train_fraction: float = 0.90,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 42,
    discard_first_fragment: bool = True,
) -> dict[str, Any]:
    fractions = train_fraction + validation_fraction + test_fraction
    if abs(fractions - 1.0) > 1e-9:
        raise ValueError("train/validation/test fractions must sum to 1")
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("all split fractions must be positive")

    source = Path(source_path).resolve()
    text = source.read_text(encoding="utf-8")
    raw_segments = text.split("<|endoftext|>")
    trailing_fragment = bool(raw_segments[-1].strip())
    candidates = raw_segments[1:] if discard_first_fragment else raw_segments
    if candidates and not candidates[-1].strip():
        candidates = candidates[:-1]
    elif trailing_fragment:
        candidates = candidates[:-1]

    exact_seen: set[str] = set()
    stories: list[str] = []
    exact_duplicates = 0
    empty_segments = 0
    for segment in candidates:
        story = canonical_story(segment)
        if not story:
            empty_segments += 1
            continue
        digest = hashlib.sha256(story.encode("utf-8")).hexdigest()
        if digest in exact_seen:
            exact_duplicates += 1
            continue
        exact_seen.add(digest)
        stories.append(story)

    ordered = sorted(stories, key=lambda story: _stable_order_key(story, seed))
    validation_count = max(1, round(len(ordered) * validation_fraction))
    test_count = max(1, round(len(ordered) * test_fraction))
    train_count = len(ordered) - validation_count - test_count
    if train_count < 1:
        raise ValueError("not enough stories for three non-empty splits")
    test = ordered[:test_count]
    validation = ordered[test_count : test_count + validation_count]
    train = ordered[test_count + validation_count :]

    destination = Path(output_dir).resolve()
    split_reports = {
        "train": _write_documents(destination / "train.txt", train),
        "validation": _write_documents(destination / "validation.txt", validation),
        "test": _write_documents(destination / "test.txt", test),
    }
    split_hash_sets = {
        name: {hashlib.sha256(document.encode("utf-8")).hexdigest() for document in documents}
        for name, documents in (("train", train), ("validation", validation), ("test", test))
    }
    overlap = {
        "train_validation": len(split_hash_sets["train"] & split_hash_sets["validation"]),
        "train_test": len(split_hash_sets["train"] & split_hash_sets["test"]),
        "validation_test": len(split_hash_sets["validation"] & split_hash_sets["test"]),
    }
    manifest = {
        "format_version": 1,
        "dataset": "TinyStories 5M-character course sample",
        "source_filename": source.name,
        "source_sha256": sha256_file(source),
        "source_url": TINYSTORIES_SOURCE_URL,
        "license": TINYSTORIES_LICENSE,
        "paper": "https://arxiv.org/abs/2305.07759",
        "delimiter": "<|endoftext|>",
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "source_segments": len(raw_segments),
        "discarded_first_fragment": discard_first_fragment,
        "discarded_trailing_fragment": trailing_fragment,
        "empty_segments": empty_segments,
        "exact_duplicates_removed_before_split": exact_duplicates,
        "unique_complete_stories": len(stories),
        "exact_cross_split_overlap": overlap,
        "splits": split_reports,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "DATASET_CARD.md").write_text(
        "# TinyStories 5M Split\n\n"
        "This directory is a deterministic 90/5/5 document-level split of the 5M-character "
        "TinyStories sample bundled with the Stanford CS336 assignment fixtures.\n\n"
        f"- Source: {TINYSTORIES_SOURCE_URL}\n"
        f"- Paper: https://arxiv.org/abs/2305.07759\n"
        f"- License: {TINYSTORIES_LICENSE}\n"
        "- Split unit: complete story delimited by `<|endoftext|>`\n"
        "- Exact duplicates are removed globally before splitting.\n"
        "- The first and trailing partial fragments in the course sample are discarded.\n\n"
        "See `manifest.json` for counts and SHA-256 fingerprints.\n",
        encoding="utf-8",
    )
    return manifest
