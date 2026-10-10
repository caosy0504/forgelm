"""Verify controlled sampling branches before comparing validation results."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .fingerprint import sha256_file, sha256_json


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_sampling(
    uniform_dir: str | Path, quality_dir: str | Path, output_path: str | Path
) -> dict[str, Any]:
    uniform_dir = Path(uniform_dir)
    quality_dir = Path(quality_dir)
    summaries = [
        _read(directory / "training_summary.json") for directory in (uniform_dir, quality_dir)
    ]
    evaluations = [
        _read(directory / "evaluation_summary.json") for directory in (uniform_dir, quality_dir)
    ]
    left, right = summaries
    left_eval, right_eval = evaluations
    if left["parent_checkpoint"] is None or left["parent_checkpoint"] != right["parent_checkpoint"]:
        raise ValueError("sampling branches do not share the same parent checkpoint")
    left_contract = left["run_fingerprint"]["contract"]
    right_contract = right["run_fingerprint"]["contract"]
    for key in ("dataset", "tokenizer_sha256", "quality_model_sha256", "model_config", "seed", "source_sha256"):
        if left_contract[key] != right_contract[key]:
            raise ValueError(f"sampling branch mismatch: {key}")
    left_training = dict(left_contract["training_config"])
    right_training = dict(right_contract["training_config"])
    left_policy = left_training.pop("sampling_policy")
    right_policy = right_training.pop("sampling_policy")
    if left_policy != "document_uniform" or right_policy not in {
        "quality_tier", "quality_tier_length_balanced"
    }:
        raise ValueError("expected document_uniform and a supported quality-tier branch")
    if left_training != right_training:
        raise ValueError("sampling branches differ in training settings beyond policy")
    if left["steps"] != right["steps"] or left["sampled_tokens_this_run"] != right["sampled_tokens_this_run"]:
        raise ValueError("sampling branches have unequal training budgets")
    if left_eval["evaluation_mode"] != "full_corpus" or right_eval["evaluation_mode"] != "full_corpus":
        raise ValueError("sampling comparison requires full-corpus validation")
    if (
        left_eval.get("full_corpus_stride") != right_eval.get("full_corpus_stride")
        or left_eval.get("document_bounded") is not True
        or right_eval.get("document_bounded") is not True
    ):
        raise ValueError("sampling branches require matching document-aware evaluation settings")
    for directory, evaluation in zip((uniform_dir, quality_dir), evaluations, strict=True):
        if evaluation["split"] != "validation":
            raise ValueError("sampling comparison requires validation reports")
        if evaluation["dataset_sha256"] != left_contract["dataset"]["validation_sha256"]:
            raise ValueError("evaluation corpus does not match the training contract")
        if evaluation["tokenizer_sha256"] != left_contract["tokenizer_sha256"]:
            raise ValueError("evaluation tokenizer does not match the training contract")
        if evaluation["checkpoint_sha256"] != sha256_file(directory / "checkpoint_last.pt"):
            raise ValueError("evaluation checkpoint differs from the final branch checkpoint")
    if (
        left_eval["dataset_stream_tokens"] != right_eval["dataset_stream_tokens"]
        or left_eval["evaluated_tokens"] != right_eval["evaluated_tokens"]
    ):
        raise ValueError("sampling branches evaluated different token counts")
    selected = "quality" if right_eval["cross_entropy_loss"] < left_eval["cross_entropy_loss"] else "uniform"
    result = {
        "comparison": f"document_uniform_vs_{right_policy}",
        "parent_checkpoint": left["parent_checkpoint"],
        "train_dataset_sha256": left_contract["dataset"]["train_sha256"],
        "validation_dataset_sha256": left_contract["dataset"]["validation_sha256"],
        "tokenizer_sha256": left_contract["tokenizer_sha256"],
        "model_config_sha256": sha256_json(left_contract["model_config"]),
        "seed": left_contract["seed"],
        "branch_steps": left["steps"] - left["parent_checkpoint"]["step"],
        "sampled_tokens_per_branch": left["sampled_tokens_this_run"],
        "sampled_tokens_total": left.get("sampled_tokens_total"),
        "sampled_to_unique_token_ratio": left.get("sampled_to_unique_token_ratio"),
        "tier_draw_counts": {
            "uniform": left.get("tier_draw_counts_this_run"),
            "quality": right.get("tier_draw_counts_this_run"),
        },
        "branch_end_to_end_tokens_per_second": {
            "uniform": left.get("end_to_end_sampled_tokens_per_second"),
            "quality": right.get("end_to_end_sampled_tokens_per_second"),
        },
        "evaluated_tokens": left_eval["evaluated_tokens"],
        "uniform_validation_loss": left_eval["cross_entropy_loss"],
        "quality_validation_loss": right_eval["cross_entropy_loss"],
        "uniform_validation_perplexity": left_eval["perplexity"],
        "quality_validation_perplexity": right_eval["perplexity"],
        "quality_relative_perplexity_change": right_eval["perplexity"] / left_eval["perplexity"] - 1.0,
        "selected_on_validation": selected,
        "selected_checkpoint": str((quality_dir if selected == "quality" else uniform_dir) / "checkpoint_last.pt"),
        "scope": "one training seed; validation comparison does not establish significance across runs",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def summarize_seeds(comparison_paths: list[str | Path], output_path: str | Path) -> dict[str, Any]:
    if len(comparison_paths) < 2:
        raise ValueError("multi-seed summary requires at least two independent comparison reports")
    reports = [_read(Path(path)) for path in comparison_paths]
    controls = ("train_dataset_sha256", "validation_dataset_sha256", "tokenizer_sha256",
                "model_config_sha256", "branch_steps", "sampled_tokens_per_branch")
    for report in reports[1:]:
        if any(report[key] != reports[0][key] for key in controls):
            raise ValueError("multi-seed reports differ in data, tokenizer, model, or branch budget")
    seeds = [report["seed"] for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("multi-seed reports contain a repeated training seed")
    changes = [report["quality_relative_perplexity_change"] for report in reports]
    result = {
        "format": "forgelm-multiseed-summary-v1",
        "seeds": sorted(seeds),
        "runs": len(reports),
        "quality_ppl_improvement_runs": sum(change < 0 for change in changes),
        "mean_relative_perplexity_change": statistics.mean(changes),
        "median_relative_perplexity_change": statistics.median(changes),
        "sample_std_relative_perplexity_change": statistics.stdev(changes),
        "min_relative_perplexity_change": min(changes),
        "max_relative_perplexity_change": max(changes),
        "controls": {key: reports[0][key] for key in controls},
        "scope": "descriptive across independent training seeds; no significance claim is computed",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
