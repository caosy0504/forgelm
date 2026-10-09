"""Pinned WikiText-103 preparation with article-level document boundaries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .fingerprint import sha256_file


DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-103-raw-v1"
REVISION = "5fddba447aa4e75996922ea0d6b18b42f0a81cc4"
PREPARATION_VERSION = "forgelm-wikitext-articles-v2"
HEADING_RE = re.compile(r"^=\s*([^=]+?)\s*=$")
ARTIFACTS = (
    (re.compile(r"\s*@-@\s*"), "-"),
    (re.compile(r"(?<=\d)\s*@,@\s*(?=\d)"), ","),
    (re.compile(r"\s*@,@\s*"), ", "),
    (re.compile(r"(?<=\d)\s*@\.@\s*(?=\d)"), "."),
    (re.compile(r"\s*@\.@\s*"), ". "),
)


def _load_split(split: str):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError('Install the WikiText data extra: pip install -e ".[data]"') from error
    try:
        return load_dataset(DATASET, CONFIG, split=split, streaming=True, revision=REVISION)
    except RuntimeError as error:
        if "Dataset scripts are no longer supported" not in str(error):
            raise
        pattern = f"hf://datasets/{DATASET}@{REVISION}/{CONFIG}/{split}-*.parquet"
        return load_dataset("parquet", data_files={split: pattern}, split=split, streaming=True)


def download_wikitext(raw_dir: str | Path, *, max_train_characters: int = 200_000_000) -> dict[str, object]:
    if max_train_characters <= 0:
        raise ValueError("max_train_characters must be positive")
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "download_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (existing.get("dataset"), existing.get("config"), existing.get("revision")) != (DATASET, CONFIG, REVISION):
            raise ValueError("existing WikiText download manifest has different provenance")
        if existing.get("requested_max_train_characters") != max_train_characters:
            raise ValueError("existing WikiText train prefix uses a different character limit")
        for split in ("train", "validation", "test"):
            source = raw_dir / f"{split}.txt"
            if not source.exists() or sha256_file(source) != existing["splits"][split]["sha256"]:
                raise ValueError(f"existing raw {split} does not match the download manifest")
        return existing
    if any((raw_dir / f"{split}.txt").exists() for split in ("train", "validation", "test")):
        raise FileExistsError("raw WikiText directory already contains a split; refusing to overwrite")
    details: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        destination = raw_dir / f"{split}.txt"
        count = 0
        characters = 0
        with destination.open("w", encoding="utf-8", newline="\n") as output:
            for record in _load_split(split):
                value = record.get("text")
                if not isinstance(value, str):
                    raise ValueError("WikiText record has no text string")
                output.write(value)
                if not value.endswith("\n"):
                    output.write("\n")
                count += 1
                characters += len(value)
                if split == "train" and characters >= max_train_characters:
                    break
        if count == 0 or (split == "train" and characters < max_train_characters):
            raise ValueError(f"WikiText {split} stream ended before the requested data was available")
        details[split] = {"rows": count, "characters": characters, "sha256": sha256_file(destination)}
    manifest = {
        "dataset": DATASET, "config": CONFIG, "revision": REVISION,
        "requested_max_train_characters": max_train_characters, "splits": details,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def iter_articles(lines: Iterable[str]) -> Iterable[str]:
    article: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        heading = HEADING_RE.fullmatch(line)
        if heading and article:
            yield " ".join(article)
            article = []
        if heading:
            line = heading.group(1).strip()
        elif line.startswith("=") and line.endswith("="):
            line = line.strip("= ").strip()
        if not line:
            continue
        for pattern, replacement in ARTIFACTS:
            line = pattern.sub(replacement, line)
        article.append(" ".join(line.split()))
    if article:
        yield " ".join(article)


def prepare_wikitext(raw_dir: str | Path, output_dir: str | Path, *, min_document_chars: int = 200) -> dict[str, object]:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_manifest_path = raw_dir / "download_manifest.json"
    if not download_manifest_path.exists():
        raise FileNotFoundError("raw WikiText download manifest is missing; use --download for pinned provenance")
    download_manifest = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    if (download_manifest.get("dataset"), download_manifest.get("config"), download_manifest.get("revision")) != (
        DATASET, CONFIG, REVISION
    ):
        raise ValueError("raw WikiText manifest does not match the pinned source")
    splits: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        source = raw_dir / f"{split}.txt"
        if not source.exists():
            raise FileNotFoundError(source)
        source_hash = sha256_file(source)
        if source_hash != download_manifest["splits"][split]["sha256"]:
            raise ValueError(f"raw {split} hash differs from its download manifest")
        destination = output_dir / f"{split}.txt"
        count = 0
        with source.open(encoding="utf-8") as input_file, destination.open("w", encoding="utf-8") as output_file:
            for article in iter_articles(input_file):
                if len(article) < min_document_chars:
                    continue
                output_file.write(article + "\n\n")
                count += 1
        if count == 0:
            raise ValueError(f"no eligible WikiText articles in {split}")
        splits[split] = {
            "source_sha256": source_hash,
            "prepared_sha256": sha256_file(destination),
            "documents": count,
            "bytes": destination.stat().st_size,
        }
    manifest = {
        "format": PREPARATION_VERSION,
        "dataset": DATASET,
        "config": CONFIG,
        "revision": REVISION,
        "min_document_chars": min_document_chars,
        "splits": splits,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
