"""Compare tokenizer compression and cost without touching the production tokenizer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .data_pipeline import read_jsonl
from .fingerprint import sha256_file
from .tokenizer import BytePairTokenizer


def audit_tokenizers(
    config_path: str | Path,
    *,
    vocab_sizes: list[int],
    max_training_characters: int | None = None,
) -> dict[str, Any]:
    if not vocab_sizes or any(size < 258 for size in vocab_sizes):
        raise ValueError("vocab sizes must be at least 258")
    config = ProjectConfig.from_toml(config_path)
    train_path = config.artifact_dir / "dataset" / "train.jsonl"
    validation_path = config.artifact_dir / "dataset" / "validation.jsonl"
    train_documents = list(read_jsonl(train_path))
    validation_documents = list(read_jsonl(validation_path))
    limit = max_training_characters or config.tokenizer.max_training_characters
    if limit <= 0:
        raise ValueError("set a positive tokenizer training character limit for the audit")
    prefix: list[str] = []
    consumed = 0
    for document in train_documents:
        if consumed >= limit:
            break
        piece = document.text[:limit - consumed]
        prefix.append(piece)
        consumed += len(piece)
    if consumed == 0:
        raise ValueError("no training text available for tokenizer audit")
    validation_bytes = sum(len(document.text.encode("utf-8")) for document in validation_documents)
    rows = []
    for requested_size in vocab_sizes:
        start = time.perf_counter()
        tokenizer = BytePairTokenizer.train(prefix, requested_size)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        token_count = sum(len(tokenizer.encode(document.text, add_eos=True)) for document in validation_documents)
        encode_seconds = time.perf_counter() - start
        rows.append({
            "requested_vocab_size": requested_size,
            "actual_vocab_size": tokenizer.vocab_size,
            "validation_tokens_including_eos": token_count,
            "validation_utf8_bytes": validation_bytes,
            "tokens_per_byte": token_count / validation_bytes,
            "bytes_per_token": validation_bytes / token_count,
            "tied_embedding_parameters": tokenizer.vocab_size * config.model.d_model,
            "tokenizer_train_seconds": train_seconds,
            "validation_encode_seconds": encode_seconds,
        })
    report = {
        "format": "forgelm-tokenizer-audit-v1",
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "training_characters": consumed,
        "rows": rows,
        "scope": "compression and preprocessing cost only; different tokenizers require separate model training",
    }
    output_path = config.artifact_dir / "tokenizer_audit.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
