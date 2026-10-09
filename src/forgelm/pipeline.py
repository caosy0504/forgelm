from __future__ import annotations

import dataclasses
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import DataConfig, ProjectConfig
from .data_pipeline import PipelineReport, prepare_dataset, read_jsonl
from .fingerprint import build_run_fingerprint, sha256_file, write_fingerprint
from .model import TransformerLM, with_vocab_size
from .quality_model import HashedNgramQualityClassifier, QualityModelReport, train_quality_classifier
from .reproducibility import resolve_device, set_seed
from .tokenizer import BytePairTokenizer
from .training import Trainer
from .sampling import DocumentSpan, encode_documents_with_spans


def build_quality_model(
    config: ProjectConfig, artifact_dir: str | Path
) -> tuple[HashedNgramQualityClassifier | None, QualityModelReport | None]:
    if not config.data.enable_model_quality:
        return None, None
    seed_path = config.quality_seed_path
    if seed_path is None:
        raise ValueError("model-quality filtering requires a seed dataset")
    return train_quality_classifier(
        seed_path,
        feature_dimension=config.data.quality_hash_dim,
        epochs=config.data.quality_epochs,
        learning_rate=config.data.quality_learning_rate,
        threshold=config.data.quality_threshold,
        output_path=Path(artifact_dir) / "quality_model.json",
    )


def _tokenizer_texts(documents: list[Any], max_characters: int):
    consumed = 0
    for document in documents:
        text = document.text
        if max_characters:
            remaining = max_characters - consumed
            if remaining <= 0:
                break
            text = text[:remaining]
        consumed += len(text)
        yield text


def _token_cache_contract(prepared_dir: Path, tokenizer_path: Path, seq_len: int) -> dict[str, Any]:
    return {
        "format": "forgelm-token-cache-v2",
        "train_sha256": sha256_file(prepared_dir / "train.jsonl"),
        "validation_sha256": sha256_file(prepared_dir / "validation.jsonl"),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "seq_len": seq_len,
        "sampling_source_sha256": sha256_file(Path(__file__).with_name("sampling.py")),
    }


def _preparation_inputs(config: ProjectConfig, prepared_dir: Path | None = None) -> dict[str, Any]:
    sources = {}
    for name, path in (
        ("train", config.input_path),
        ("validation", config.validation_path),
        ("test", config.test_path),
    ):
        sources[name] = {"path": str(path), "sha256": sha256_file(path)} if path is not None else None
    prepared = None
    if prepared_dir is not None:
        prepared = {
            name: sha256_file(prepared_dir / filename)
            for name, filename in (
                ("train", "train.jsonl"), ("validation", "validation.jsonl"),
                ("test", "test.jsonl"), ("manifest", "dataset_manifest.json"),
            ) if (prepared_dir / filename).exists()
        }
    return {"data_config": asdict(config.data), "sources": sources, "prepared_sha256": prepared}


def _load_or_build_token_cache(
    artifact_dir: Path,
    prepared_dir: Path,
    tokenizer_path: Path,
    tokenizer: BytePairTokenizer,
    train_documents: list[Any],
    validation_documents: list[Any],
    seq_len: int,
    *,
    require_existing: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[DocumentSpan], list[DocumentSpan]]:
    cache_path = artifact_dir / "token_cache.pt"
    manifest_path = artifact_dir / "token_cache_manifest.json"
    contract = _token_cache_contract(prepared_dir, tokenizer_path, seq_len)
    if cache_path.exists() and manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(recorded.get(key) != value for key, value in contract.items()):
            raise ValueError("token cache does not match the prepared data, tokenizer, or sequence length")
        if recorded.get("cache_sha256") != sha256_file(cache_path):
            raise ValueError("token cache checksum mismatch")
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        spans = [DocumentSpan(*row) for row in payload["spans"]]
        validation_spans = [DocumentSpan(*row) for row in payload["validation_spans"]]
        return payload["train_tokens"], payload["validation_tokens"], spans, validation_spans
    if require_existing:
        raise FileNotFoundError("fork parent has no verified token cache")
    train_tokens, spans = encode_documents_with_spans(
        (document.text for document in train_documents), tokenizer, seq_len
    )
    validation_tokens, validation_spans = encode_documents_with_spans(
        (document.text for document in validation_documents), tokenizer, seq_len
    )
    torch.save(
        {
            "train_tokens": train_tokens,
            "validation_tokens": validation_tokens,
            "spans": [(span.start, span.end, span.score, span.tier) for span in spans],
            "validation_spans": [
                (span.start, span.end, span.score, span.tier) for span in validation_spans
            ],
        },
        cache_path,
    )
    manifest_path.write_text(
        json.dumps({**contract, "cache_sha256": sha256_file(cache_path)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return train_tokens, validation_tokens, spans, validation_spans


def run_pipeline_config(
    config: ProjectConfig,
    *,
    resume_path: str | Path | None = None,
    fork_path: str | Path | None = None,
    stop_step: int | None = None,
    prepare_only: bool = False,
    tokenizer_override: BytePairTokenizer | None = None,
) -> dict[str, Any]:
    if resume_path is not None and fork_path is not None:
        raise ValueError("resume and fork are mutually exclusive")
    if prepare_only and (resume_path is not None or fork_path is not None or stop_step is not None):
        raise ValueError("prepare_only cannot resume, fork, or stop training")
    config.validate()
    artifact_dir = config.artifact_dir
    prepared_dir = artifact_dir / "dataset"
    resolved_config_path = artifact_dir / "resolved_config.json"
    previous_config = (
        json.loads(resolved_config_path.read_text(encoding="utf-8"))
        if resolved_config_path.exists() else None
    )
    if fork_path is not None and artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError("fork output directory must be new and empty")
    parent_dir = None
    if fork_path is not None:
        if config.data.enable_model_quality:
            raise ValueError("controlled sampling fork currently requires model-quality filtering disabled")
        parent_dir = Path(fork_path).resolve().parent
        parent_config = json.loads((parent_dir / "resolved_config.json").read_text(encoding="utf-8"))
        current_config = config.as_dict()
        for key in ("seed", "data", "tokenizer", "model"):
            if parent_config[key] != current_config[key]:
                raise ValueError(f"fork config mismatch: {key}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path.write_text(
        json.dumps(config.as_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    if fork_path is not None:
        assert parent_dir is not None
        shutil.copytree(parent_dir / "dataset", prepared_dir)
        shutil.copy2(parent_dir / "tokenizer.json", artifact_dir / "tokenizer.json")
        shutil.copy2(parent_dir / "token_cache.pt", artifact_dir / "token_cache.pt")
        shutil.copy2(parent_dir / "token_cache_manifest.json", artifact_dir / "token_cache_manifest.json")
        data_report = PipelineReport(**json.loads((prepared_dir / "dataset_manifest.json").read_text()))
        quality_report = None
    else:
        input_contract = _preparation_inputs(config, prepared_dir)
        input_manifest_path = prepared_dir / "preparation_inputs.json"
        if input_manifest_path.exists() and (prepared_dir / "dataset_manifest.json").exists():
            if json.loads(input_manifest_path.read_text(encoding="utf-8")) != input_contract:
                raise ValueError("prepared data inputs changed; use a new output directory")
            data_report = PipelineReport(**json.loads((prepared_dir / "dataset_manifest.json").read_text()))
            quality_report = None
            if config.data.enable_model_quality and not (artifact_dir / "quality_model.json").exists():
                raise FileNotFoundError("prepared data requires its original quality model")
        else:
            quality_classifier, quality_report = build_quality_model(config, artifact_dir)
            data_report = prepare_dataset(
                config.input_path,
                prepared_dir,
                config.data,
                quality_classifier,
                validation_path=config.validation_path,
                test_path=config.test_path,
            )
            input_manifest_path.write_text(
                json.dumps(_preparation_inputs(config, prepared_dir), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8"
            )
    tokenizer_path = artifact_dir / "tokenizer.json"
    reusable_cache = (
        tokenizer_path.exists() and (artifact_dir / "token_cache.pt").exists()
        and (artifact_dir / "token_cache_manifest.json").exists()
    )
    if reusable_cache and fork_path is None and tokenizer_override is None:
        if previous_config is None or previous_config.get("tokenizer") != asdict(config.tokenizer):
            raise ValueError("existing tokenizer cache uses a different tokenizer configuration")
    if (fork_path is not None or reusable_cache) and tokenizer_override is None:
        tokenizer = BytePairTokenizer.load(tokenizer_path)
        train_documents: list[Any] = []
        validation_documents: list[Any] = []
    else:
        train_documents = list(read_jsonl(prepared_dir / "train.jsonl"))
        validation_documents = list(read_jsonl(prepared_dir / "validation.jsonl"))
        tokenizer = tokenizer_override or BytePairTokenizer.train(
            _tokenizer_texts(train_documents, config.tokenizer.max_training_characters),
            config.tokenizer.vocab_size,
        )
    tokenizer.save(tokenizer_path)
    train_tokens, validation_tokens, train_spans, validation_spans = _load_or_build_token_cache(
        artifact_dir,
        prepared_dir,
        tokenizer_path,
        tokenizer,
        train_documents,
        validation_documents,
        config.training.seq_len,
        require_existing=fork_path is not None,
    )

    set_seed(config.seed)
    model_config = with_vocab_size(config.model, tokenizer.vocab_size)
    run_fingerprint = build_run_fingerprint(
        config,
        model_config=model_config,
        dataset_dir=prepared_dir,
        tokenizer_path=tokenizer_path,
        quality_model_path=(artifact_dir / "quality_model.json") if config.data.enable_model_quality else None,
    )
    write_fingerprint(artifact_dir / "run_fingerprint.json", run_fingerprint)
    if prepare_only:
        summary = {
            "project": config.name,
            "data": asdict(data_report),
            "train_tokens": int(train_tokens.numel()),
            "validation_tokens": int(validation_tokens.numel()),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "run_fingerprint_id": run_fingerprint["fingerprint_id"],
            "status": "prepared_no_training",
        }
        (artifact_dir / "preparation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary
    model = TransformerLM(model_config)
    trainer = Trainer(
        model,
        train_tokens,
        validation_tokens,
        config.training,
        seed=config.seed,
        artifact_dir=artifact_dir,
        reset_metrics=resume_path is None,
        run_fingerprint=run_fingerprint,
        train_spans=train_spans,
        validation_spans=validation_spans,
    )
    if resume_path is not None:
        trainer.load_checkpoint(resume_path)
    elif fork_path is not None:
        trainer.fork_from_checkpoint(fork_path)
    training_summary = trainer.train(stop_step=stop_step)

    prompt_ids = tokenizer.encode(config.generation.prompt)
    if not prompt_ids:
        raise ValueError("generation prompt produced no tokens")
    device = resolve_device(config.training.device)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generator = (
        torch.Generator(device=device.type).manual_seed(config.seed + 1) if device.type in {"cpu", "cuda"} else None
    )
    generated_ids = trainer.model.generate(
        prompt,
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        top_k=config.generation.top_k,
        eos_id=tokenizer.EOS_ID,
        generator=generator,
    )[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    (artifact_dir / "sample.txt").write_text(generated_text + "\n", encoding="utf-8")
    summary = {
        "project": config.name,
        "data": asdict(data_report),
        "quality_model": asdict(quality_report) if quality_report is not None else None,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "training": training_summary,
        "generation": {"prompt": config.generation.prompt, "text": generated_text},
        "run_fingerprint_id": run_fingerprint["fingerprint_id"],
    }
    (artifact_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_pipeline(
    config_path: str | Path,
    *,
    resume_path: str | Path | None = None,
    fork_path: str | Path | None = None,
    stop_step: int | None = None,
    prepare_only: bool = False,
) -> dict[str, Any]:
    return run_pipeline_config(
        ProjectConfig.from_toml(config_path), resume_path=resume_path, fork_path=fork_path,
        stop_step=stop_step, prepare_only=prepare_only,
    )


def _ablation_data_config(base: DataConfig, variant: str) -> DataConfig:
    shared = {
        "enable_pii_masking": True,
        "enable_decontamination": True,
    }
    if variant == "raw":
        return dataclasses.replace(
            base,
            **shared,
            enable_quality_rules=False,
            enable_exact_dedup=False,
            enable_near_dedup=False,
            enable_model_quality=False,
        )
    if variant == "heuristic":
        return dataclasses.replace(
            base,
            **shared,
            enable_quality_rules=True,
            enable_exact_dedup=False,
            enable_near_dedup=False,
            enable_model_quality=False,
        )
    if variant == "dedup":
        return dataclasses.replace(
            base,
            **shared,
            enable_quality_rules=True,
            enable_exact_dedup=True,
            enable_near_dedup=True,
            enable_model_quality=False,
        )
    if variant == "model_quality":
        return dataclasses.replace(
            base,
            **shared,
            enable_quality_rules=True,
            enable_exact_dedup=True,
            enable_near_dedup=True,
            enable_model_quality=True,
        )
    raise ValueError(f"unknown ablation variant: {variant}")


def run_data_ablation(
    config_path: str | Path,
    *,
    variants: tuple[str, ...] = ("raw", "heuristic", "dedup", "model_quality"),
) -> dict[str, Any]:
    base = ProjectConfig.from_toml(config_path)
    ablation_root_relative = str(Path(base.output_dir).parent / "data_ablation")
    shared_tokenizer: BytePairTokenizer | None = None
    validation_hash: str | None = None
    test_hash: str | None = None
    rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_config = dataclasses.replace(
            base,
            name=f"{base.name}-{variant}",
            output_dir=str(Path(ablation_root_relative) / variant),
            data=_ablation_data_config(base.data, variant),
        )
        result = run_pipeline_config(variant_config, tokenizer_override=shared_tokenizer)
        if shared_tokenizer is None:
            shared_tokenizer = BytePairTokenizer.load(variant_config.artifact_dir / "tokenizer.json")
        current_validation_hash = result["training"]["run_fingerprint"]["contract"]["dataset"][
            "validation_sha256"
        ]
        if validation_hash is None:
            validation_hash = current_validation_hash
        elif current_validation_hash != validation_hash:
            raise RuntimeError("controlled ablation produced different validation splits")
        current_test_hash = result["training"]["run_fingerprint"]["contract"]["dataset"].get("test_sha256")
        if test_hash is None:
            test_hash = current_test_hash
        elif current_test_hash != test_hash:
            raise RuntimeError("controlled ablation produced different test splits")
        rows.append(
            {
                "variant": variant,
                "train_documents": result["data"]["train_documents"],
                "train_tokens": result["training"]["train_tokens"],
                "initial_validation_loss": result["training"]["initial_validation_loss"],
                "final_validation_loss": result["training"]["final_validation_loss"],
                "final_validation_perplexity": result["training"]["final_validation_perplexity"],
                "wall_time_seconds": result["training"]["wall_time_seconds"],
                "model_quality_dropped": result["data"]["model_quality_dropped"],
                "exact_duplicates_dropped": result["data"]["exact_duplicates_dropped"],
                "near_duplicates_dropped": result["data"]["near_duplicates_dropped"],
                "decontaminated_documents_dropped": result["data"]["decontaminated_documents_dropped"],
                "tokenizer_sha256": result["tokenizer_sha256"],
            }
        )
    summary = {
        "experiment": "controlled_data_ablation",
        "config": str(Path(config_path).resolve()),
        "shared_validation_sha256": validation_hash,
        "shared_test_sha256": test_hash,
        "shared_tokenizer_sha256": rows[0]["tokenizer_sha256"],
        "equal_training_step_budget": base.training.steps,
        "variants": rows,
        "interpretation_note": "Smoke-scale results validate causal controls; use a real corpus and multiple seeds for claims.",
    }
    output_path = (base.root / ablation_root_relative / "ablation_summary.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
