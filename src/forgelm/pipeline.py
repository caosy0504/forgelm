from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import DataConfig, ProjectConfig
from .data_pipeline import prepare_dataset, read_jsonl
from .fingerprint import build_run_fingerprint, sha256_file, write_fingerprint
from .model import TransformerLM, with_vocab_size
from .quality_model import HashedNgramQualityClassifier, QualityModelReport, train_quality_classifier
from .reproducibility import resolve_device, set_seed
from .tokenizer import BytePairTokenizer
from .training import Trainer, encode_documents


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


def run_pipeline_config(
    config: ProjectConfig,
    *,
    resume_path: str | Path | None = None,
    tokenizer_override: BytePairTokenizer | None = None,
) -> dict[str, Any]:
    config.validate()
    artifact_dir = config.artifact_dir
    prepared_dir = artifact_dir / "dataset"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "resolved_config.json").write_text(
        json.dumps(config.as_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    quality_classifier, quality_report = build_quality_model(config, artifact_dir)
    data_report = prepare_dataset(
        config.input_path,
        prepared_dir,
        config.data,
        quality_classifier,
        validation_path=config.validation_path,
        test_path=config.test_path,
    )
    train_documents = list(read_jsonl(prepared_dir / "train.jsonl"))
    validation_documents = list(read_jsonl(prepared_dir / "validation.jsonl"))
    tokenizer = tokenizer_override or BytePairTokenizer.train(
        (document.text for document in train_documents), config.tokenizer.vocab_size
    )
    tokenizer_path = artifact_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    train_tokens = encode_documents((document.text for document in train_documents), tokenizer)
    validation_tokens = encode_documents((document.text for document in validation_documents), tokenizer)

    set_seed(config.seed)
    model_config = with_vocab_size(config.model, tokenizer.vocab_size)
    run_fingerprint = build_run_fingerprint(
        config,
        model_config=model_config,
        dataset_dir=prepared_dir,
        tokenizer_path=tokenizer_path,
        quality_model_path=(artifact_dir / "quality_model.json") if quality_report is not None else None,
    )
    write_fingerprint(artifact_dir / "run_fingerprint.json", run_fingerprint)
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
    )
    if resume_path is not None:
        trainer.load_checkpoint(resume_path)
    training_summary = trainer.train()

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


def run_pipeline(config_path: str | Path, *, resume_path: str | Path | None = None) -> dict[str, Any]:
    return run_pipeline_config(ProjectConfig.from_toml(config_path), resume_path=resume_path)


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
