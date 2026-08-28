from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import ModelConfig, ProjectConfig


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_source_tree(source_dir: str | Path) -> str:
    source_dir = Path(source_dir)
    digest = hashlib.sha256()
    for path in sorted(source_dir.rglob("*.py")):
        digest.update(path.relative_to(source_dir).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def build_run_fingerprint(
    config: ProjectConfig,
    *,
    model_config: ModelConfig,
    dataset_dir: str | Path,
    tokenizer_path: str | Path,
    quality_model_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    contract = {
        "dataset": {
            "manifest_sha256": sha256_file(dataset_dir / "dataset_manifest.json"),
            "train_sha256": sha256_file(dataset_dir / "train.jsonl"),
            "validation_sha256": sha256_file(dataset_dir / "validation.jsonl"),
            "test_sha256": sha256_file(dataset_dir / "test.jsonl") if (dataset_dir / "test.jsonl").exists() else None,
        },
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "quality_model_sha256": sha256_file(quality_model_path) if quality_model_path is not None else None,
        "model_config": asdict(model_config),
        "training_config": asdict(config.training),
        "seed": config.seed,
        "source_sha256": sha256_source_tree(config.root / "src" / "forgelm"),
    }
    return {"format_version": 1, "fingerprint_id": sha256_json(contract), "contract": contract}


def write_fingerprint(path: str | Path, fingerprint: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(fingerprint, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
