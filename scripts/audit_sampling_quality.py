"""CPU-only diagnostics and blinded review sheet for a future quality-sampling experiment.

This script never modifies ForgeLM's prepared dataset, token cache, or checkpoints.
The length-balanced policy is implemented separately, but has no GPU result yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from forgelm.data_pipeline import read_jsonl
from forgelm.fingerprint import sha256_file
from forgelm.sampling import DocumentSpan, assign_length_balanced_quality_tiers, prose_quality_score


@dataclass(frozen=True)
class Record:
    doc_id: str
    score: float
    current_tier: int
    valid_windows: int
    tokens: int
    utf8_bytes: int
    length_stratum: int = -1
    candidate_tier: int = -1


def _length_stratified_tiers(records: list[Record], seq_len: int, strata: int = 4) -> list[Record]:
    """Describe the same length-balanced tiers that the training sampler will use."""
    if strata < 1 or len(records) < strata:
        raise ValueError("need at least one eligible document per length stratum")
    if any(record.valid_windows != max(0, record.tokens - seq_len) for record in records):
        raise ValueError("record windows do not match the training sequence length")
    order = sorted(range(len(records)), key=lambda index: (records[index].tokens, index))
    stratum_for = {index: min(strata - 1, rank * strata // len(order)) for rank, index in enumerate(order)}
    starts = 0
    spans = []
    for record in records:
        spans.append(DocumentSpan(starts, starts + record.tokens, record.score, record.current_tier))
        starts += record.tokens
    assigned = assign_length_balanced_quality_tiers(spans, seq_len, length_strata=strata)
    return [
        Record(**{**record.__dict__, "length_stratum": stratum_for[index], "candidate_tier": assigned[index].tier})
        for index, record in enumerate(records)
    ]


def _summarize_tiers(records: list[Record], attribute: str) -> dict[str, object]:
    total_windows = sum(record.valid_windows for record in records)
    result: dict[str, object] = {}
    for tier, name in enumerate(("low", "medium", "high")):
        members = [record for record in records if getattr(record, attribute) == tier]
        windows = sum(record.valid_windows for record in members)
        result[name] = {
            "documents": len(members),
            "valid_windows": windows,
            "window_fraction": windows / total_windows,
            "mean_utf8_bytes": sum(record.utf8_bytes for record in members) / len(members),
            "mean_score": sum(record.score for record in members) / len(members),
        }
    return result


def _length_table(records: list[Record], attribute: str) -> list[dict[str, object]]:
    table = []
    for stratum in range(4):
        members = [record for record in records if record.length_stratum == stratum]
        table.append({
            "length_stratum": stratum,
            "documents": len(members),
            "mean_tokens": sum(record.tokens for record in members) / len(members),
            "tier_document_fraction": [
                sum(getattr(record, attribute) == tier for record in members) / len(members) for tier in range(3)
            ],
            "tier_window_fraction": [
                sum(record.valid_windows for record in members if getattr(record, attribute) == tier)
                / sum(record.valid_windows for record in members)
                for tier in range(3)
            ],
        })
    return table


def _excerpt(text: str, start: int, width: int = 380) -> str:
    return re.sub(r"\s+", " ", text[start : start + width]).strip()


def _write_review_sheet(path: Path, source: Path, ordered_ids: list[str]) -> None:
    wanted = set(ordered_ids)
    excerpts = {}
    for document in read_jsonl(source):
        if document.doc_id not in wanted:
            continue
        text = document.text
        width = 380
        middle = max(0, (len(text) - width) // 2)
        tail = max(0, len(text) - width)
        excerpts[document.doc_id] = (
            _excerpt(text, 0, width), _excerpt(text, middle, width), _excerpt(text, tail, width)
        )
    if set(excerpts) != wanted:
        raise ValueError("sampled document IDs are missing from the prepared dataset")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("doc_id", "excerpt_start", "excerpt_middle", "excerpt_end", "human_label", "reason"))
        for doc_id in ordered_ids:
            writer.writerow((doc_id, *excerpts[doc_id], "", ""))


def _human_label_summary(path: Path, lookup: dict[str, Record], expected_ids: set[str]) -> dict[str, object]:
    grouped: dict[str, dict[int, list[int]]] = {
        "current_tier": defaultdict(list),
        "candidate_tier": defaultdict(list),
    }
    seen = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            doc_id = row.get("doc_id", "")
            if doc_id not in expected_ids or doc_id in seen:
                raise ValueError(f"unknown or duplicate review document: {doc_id}")
            seen.add(doc_id)
            label = row.get("human_label", "").strip()
            if not label:
                continue
            if label not in {"0", "1", "2"}:
                raise ValueError(f"human_label must be 0, 1, or 2 for {doc_id}")
            record = lookup[doc_id]
            for name in grouped:
                grouped[name][getattr(record, name)].append(int(label))
    if seen != expected_ids:
        raise ValueError("review sheet must retain exactly the sampled document IDs")
    return {
        "labeled": sum(len(labels) for labels in grouped["current_tier"].values()),
        "expected": len(expected_ids),
        "by_policy": {
            name: {
                ("low", "medium", "high")[tier]: {
                    "labeled": len(groups[tier]),
                    "mean_human_label": sum(groups[tier]) / len(groups[tier]) if groups[tier] else None,
                }
                for tier in range(3)
            }
            for name, groups in grouped.items()
        },
        "note": "Human labels assess sampled prose utility; they do not establish downstream model gains.",
    }


def run(base_dir: Path, output_dir: Path, seed: int, per_cell: int, labels: Path | None) -> dict[str, object]:
    if per_cell < 1:
        raise ValueError("per-cell must be positive")
    dataset = base_dir / "dataset" / "train.jsonl"
    validation = base_dir / "dataset" / "validation.jsonl"
    tokenizer = base_dir / "tokenizer.json"
    cache = base_dir / "token_cache.pt"
    manifest = json.loads((base_dir / "token_cache_manifest.json").read_text())
    for key, path in (
        ("train_sha256", dataset),
        ("validation_sha256", validation),
        ("tokenizer_sha256", tokenizer),
        ("cache_sha256", cache),
    ):
        if sha256_file(path) != manifest[key]:
            raise ValueError(f"prepared artifact checksum mismatch: {key}")
    payload = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    spans = [DocumentSpan(*row) for row in payload["spans"]]
    records = []
    count = 0
    for document, span in zip(read_jsonl(dataset), spans, strict=True):
        count += 1
        if span.tier < 0:
            continue
        records.append(Record(
            doc_id=document.doc_id,
            score=span.score,
            current_tier=span.tier,
            valid_windows=span.valid_starts(int(manifest["seq_len"])),
            tokens=span.end - span.start,
            utf8_bytes=len(document.text.encode("utf-8")),
        ))
    if count != len(spans):
        raise ValueError("prepared document count differs from token-cache spans")
    records = _length_stratified_tiers(records, int(manifest["seq_len"]))
    scores = np.asarray([record.score for record in records])
    lengths = np.asarray([record.utf8_bytes for record in records])
    score_ranks = np.argsort(np.argsort(scores))
    length_ranks = np.argsort(np.argsort(lengths))
    score_length_corr = float(np.corrcoef(score_ranks, length_ranks)[0, 1])

    rng = random.Random(seed)
    sample_ids = []
    for stratum in range(4):
        for tier in range(3):
            members = [r.doc_id for r in records if r.length_stratum == stratum and r.candidate_tier == tier]
            if len(members) < per_cell:
                raise ValueError(f"not enough documents in length stratum {stratum}, tier {tier}")
            sample_ids.extend(rng.sample(members, per_cell))
    rng.shuffle(sample_ids)
    lookup = {record.doc_id: record for record in records}
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_tiers.json"
    candidate = {
        "format": "forgelm-length-balanced-tier-map-v1",
        "status": "audit snapshot; experimental trainer recomputes tiers and does not read this file",
        "train_sha256": manifest["train_sha256"],
        "token_cache_sha256": manifest["cache_sha256"],
        "seq_len": manifest["seq_len"],
        "length_strata": 4,
        "tiers_by_doc_id": {record.doc_id: record.candidate_tier for record in records},
    }
    if candidate_path.exists() and json.loads(candidate_path.read_text()) != candidate:
        raise ValueError("existing candidate tier map differs; use a new output directory")
    candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    key_path = output_dir / "annotation_key.json"
    key = {
        "format": "forgelm-length-balanced-quality-audit-v1",
        "seed": seed,
        "per_cell": per_cell,
        "train_sha256": manifest["train_sha256"],
        "cache_sha256": manifest["cache_sha256"],
        "rows": {
            doc_id: {
                "current_tier": lookup[doc_id].current_tier,
                "candidate_tier": lookup[doc_id].candidate_tier,
                "length_stratum": lookup[doc_id].length_stratum,
                "heuristic_score": lookup[doc_id].score,
            }
            for doc_id in sample_ids
        },
    }
    if key_path.exists() and json.loads(key_path.read_text()) != key:
        raise ValueError("existing annotation key differs; use a new output directory")
    key_path.write_text(json.dumps(key, indent=2, sort_keys=True) + "\n")
    sheet = output_dir / "quality_review_blind.csv"
    if not sheet.exists():
        _write_review_sheet(sheet, dataset, sample_ids)
    with sheet.open(encoding="utf-8", newline="") as handle:
        observed = [row["doc_id"] for row in csv.DictReader(handle)]
    if observed != sample_ids:
        raise ValueError("existing review sheet has different document IDs or order")

    low_cut = max(record.score for record in records if record.current_tier == 0)
    mid_cut = max(record.score for record in records if record.current_tier == 1)
    validation_scores = [prose_quality_score(doc.text) for doc in read_jsonl(validation)]
    report: dict[str, object] = {
        "format": "forgelm-length-balanced-quality-audit-v1",
        "status": "cpu-validated experimental policy; no GPU result or human labels",
        "train_sha256": manifest["train_sha256"],
        "token_cache_sha256": manifest["cache_sha256"],
        "eligible_documents": len(records),
        "score_length_rank_correlation_approx": score_length_corr,
        "current_tiers": _summarize_tiers(records, "current_tier"),
        "length_balanced_candidate": _summarize_tiers(records, "candidate_tier"),
        "current_by_length": _length_table(records, "current_tier"),
        "candidate_by_length": _length_table(records, "candidate_tier"),
        "validation_documents_by_current_score_band": [
            sum(value <= low_cut for value in validation_scores),
            sum(low_cut < value <= mid_cut for value in validation_scores),
            sum(value > mid_cut for value in validation_scores),
        ],
        "validation_warning": "Exploratory distribution check only; do not tune the policy on official validation.",
        "review_sheet": str(sheet),
        "candidate_tier_map": str(candidate_path),
        "candidate_tier_map_sha256": sha256_file(candidate_path),
        "review_sample_design": f"{per_cell} documents per candidate tier per token-length quartile; randomized and tier-blinded",
    }
    if labels is not None:
        report["human_label_summary"] = _human_label_summary(labels, lookup, set(sample_ids))
    report_path = output_dir / "diagnostics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("artifacts/wikitext4090/base"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/wikitext4090/quality_mac_audit"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-cell", type=int, default=5)
    parser.add_argument("--labels", type=Path, default=None)
    args = parser.parse_args()
    report = run(args.base_dir, args.output_dir, args.seed, args.per_cell, args.labels)
    print(json.dumps({
        "eligible_documents": report["eligible_documents"],
        "score_length_rank_correlation_approx": report["score_length_rank_correlation_approx"],
        "review_sheet": report["review_sheet"],
        "status": report["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
