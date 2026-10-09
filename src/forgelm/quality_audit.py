"""Audit heuristic quality tiers and prepare a small blinded human-label sample."""

from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

import torch

from .config import ProjectConfig
from .data_pipeline import read_jsonl
from .fingerprint import sha256_file
from .sampling import DocumentSpan, prose_quality_features, rescore_document_spans


TIER_NAMES = ("low", "medium", "high")


def retier_quality_cache(config_path: str | Path) -> dict[str, Any]:
    """Rescore existing token spans before training, preserving token IDs and boundaries."""
    config = ProjectConfig.from_toml(config_path)
    artifact_dir = config.artifact_dir
    if any(artifact_dir.glob("checkpoint*.pt")):
        raise ValueError("quality tiers cannot be changed after a training checkpoint exists")
    dataset_path = artifact_dir / "dataset" / "train.jsonl"
    tokenizer_path = artifact_dir / "tokenizer.json"
    cache_path = artifact_dir / "token_cache.pt"
    manifest_path = artifact_dir / "token_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, actual in (
        ("train_sha256", sha256_file(dataset_path)),
        ("tokenizer_sha256", sha256_file(tokenizer_path)),
        ("cache_sha256", sha256_file(cache_path)),
    ):
        if manifest.get(key) != actual:
            raise ValueError(f"cannot retier mismatched cache: {key}")
    if manifest.get("seq_len") != config.training.seq_len:
        raise ValueError("cannot retier cache with a different sequence length")
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    old_spans = [DocumentSpan(*row) for row in payload["spans"]]
    documents = list(read_jsonl(dataset_path))
    if not old_spans or old_spans[-1].end != int(payload["train_tokens"].numel()):
        raise ValueError("cached document spans do not cover the token stream")
    new_spans = rescore_document_spans((document.text for document in documents), old_spans, config.training.seq_len)
    payload["spans"] = [(span.start, span.end, span.score, span.tier) for span in new_spans]
    for name in ("quality_audit.json", "quality_audit_samples.csv"):
        old_path = artifact_dir / name
        if old_path.exists():
            archive = artifact_dir / f"{name}.before_retier"
            if archive.exists():
                raise FileExistsError(f"quality audit archive already exists: {archive}")
            old_path.rename(archive)
    old_hash = manifest["cache_sha256"]
    temporary_cache = artifact_dir / "token_cache.pt.retier"
    torch.save(payload, temporary_cache)
    manifest["cache_sha256"] = sha256_file(temporary_cache)
    manifest["sampling_source_sha256"] = sha256_file(Path(__file__).with_name("sampling.py"))
    manifest["retiered_from_cache_sha256"] = old_hash
    temporary_manifest = artifact_dir / "token_cache_manifest.json.retier"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_cache.replace(cache_path)
    temporary_manifest.replace(manifest_path)
    return {
        "documents": len(new_spans),
        "train_tokens": int(payload["train_tokens"].numel()),
        "old_cache_sha256": old_hash,
        "new_cache_sha256": manifest["cache_sha256"],
        "tier_documents": {name: sum(span.tier == index for span in new_spans)
                           for index, name in enumerate(TIER_NAMES)},
        "status": "rescored_before_training",
    }


def audit_quality(
    config_path: str | Path,
    *,
    sample_per_tier: int = 20,
    seed: int = 42,
    labels_path: str | Path | None = None,
) -> dict[str, Any]:
    if sample_per_tier < 1:
        raise ValueError("sample_per_tier must be positive")
    config = ProjectConfig.from_toml(config_path)
    artifact_dir = config.artifact_dir
    dataset_path = artifact_dir / "dataset" / "train.jsonl"
    tokenizer_path = artifact_dir / "tokenizer.json"
    cache_path = artifact_dir / "token_cache.pt"
    cache_manifest = json.loads((artifact_dir / "token_cache_manifest.json").read_text(encoding="utf-8"))
    for key, actual in (
        ("train_sha256", sha256_file(dataset_path)),
        ("tokenizer_sha256", sha256_file(tokenizer_path)),
        ("cache_sha256", sha256_file(cache_path)),
    ):
        if cache_manifest.get(key) != actual:
            raise ValueError(f"quality audit input mismatch: {key}")
    if cache_manifest.get("seq_len") != config.training.seq_len:
        raise ValueError("quality audit sequence length differs from the token cache")
    if cache_manifest.get("sampling_source_sha256") != sha256_file(Path(__file__).with_name("sampling.py")):
        raise ValueError("quality audit sampling code differs from the token cache")
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    spans = [DocumentSpan(*row) for row in payload["spans"]]
    documents = list(read_jsonl(dataset_path))
    if len(documents) != len(spans):
        raise ValueError("quality audit document and token-span counts differ")
    eligible = [(document, span) for document, span in zip(documents, spans, strict=True) if span.tier >= 0]
    total_windows = sum(span.valid_starts(config.training.seq_len) for _, span in eligible)
    tiers: dict[str, dict[str, Any]] = {}
    rng = random.Random(seed)
    sampled: list[tuple[str, str]] = []
    for tier, name in enumerate(TIER_NAMES):
        members = [(document, span) for document, span in eligible if span.tier == tier]
        windows = sum(span.valid_starts(config.training.seq_len) for _, span in members)
        scores = [span.score for _, span in members]
        features = [prose_quality_features(document.text) for document, _ in members]
        tiers[name] = {
            "documents": len(members),
            "eligible_windows": windows,
            "natural_window_fraction": windows / total_windows if total_windows else 0.0,
            "configured_sampling_probability": float(config.training.quality_tier_probabilities[tier]),
            "mean_heuristic_score": statistics.mean(scores) if scores else None,
            "median_heuristic_score": statistics.median(scores) if scores else None,
            "mean_utf8_bytes": statistics.mean(len(document.text.encode("utf-8")) for document, _ in members)
            if members else None,
            "mean_features": {
                key: statistics.mean(float(item[key]) for item in features) if features else None
                for key in (
                    "length_score", "alphabetic_score", "sentence_density",
                    "unique_word_ratio", "trigram_uniqueness",
                )
            },
        }
        for document, _ in rng.sample(members, min(sample_per_tier, len(members))):
            sampled.append((document.doc_id, document.text[:500].replace("\n", " ")))
    rng.shuffle(sampled)
    samples_path = artifact_dir / "quality_audit_samples.csv"
    if samples_path.exists():
        with samples_path.open(encoding="utf-8", newline="") as handle:
            existing_ids = [row["doc_id"] for row in csv.DictReader(handle)]
        if existing_ids != [doc_id for doc_id, _ in sampled]:
            raise ValueError("existing quality audit sample does not match current data or seed")
    else:
        with samples_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("doc_id", "text_preview", "human_label"))
            writer.writeheader()
            for doc_id, preview in sampled:
                writer.writerow({"doc_id": doc_id, "text_preview": preview, "human_label": ""})
    label_report = None
    if labels_path is not None:
        lookup = {document.doc_id: span.tier for document, span in eligible}
        scored: dict[str, list[int]] = {name: [] for name in TIER_NAMES}
        seen_labels: set[str] = set()
        with Path(labels_path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                doc_id = row.get("doc_id", "")
                value = row.get("human_label", "").strip()
                if not value:
                    continue
                if doc_id not in lookup or value not in {"0", "1", "2"}:
                    raise ValueError("human labels require known doc_id and a 0, 1, or 2 label")
                if doc_id in seen_labels:
                    raise ValueError("duplicate human label for one document")
                seen_labels.add(doc_id)
                scored[TIER_NAMES[lookup[doc_id]]].append(int(value))
        label_report = {
            name: {"labeled": len(values), "mean_human_label": statistics.mean(values) if values else None}
            for name, values in scored.items()
        }
    report = {
        "format": "forgelm-quality-audit-v1",
        "train_sha256": cache_manifest["train_sha256"],
        "tokenizer_sha256": cache_manifest["tokenizer_sha256"],
        "eligible_documents": len(eligible),
        "eligible_windows": total_windows,
        "tiers": tiers,
        "sample_path": str(samples_path),
        "sample_seed": seed,
        "human_label_scale": "0=low prose utility, 1=medium, 2=high; use content judgment, not the hidden tier",
        "human_labels": label_report,
        "scope": "heuristic prose form, not factuality, safety, or learned educational quality",
    }
    (artifact_dir / "quality_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return report
