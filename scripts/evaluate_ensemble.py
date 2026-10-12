"""Document-aware evaluation of a fixed, equal-probability two-seed ensemble.

This script lives outside ``src/forgelm`` so adding the analysis does not change
the training-source fingerprint embedded in existing checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from forgelm.config import ModelConfig
from forgelm.data_pipeline import read_jsonl
from forgelm.fingerprint import sha256_file
from forgelm.model import TransformerLM
from forgelm.reproducibility import environment_metadata, resolve_device
from forgelm.tokenizer import BytePairTokenizer


POLICIES = ("document_uniform", "document_uniform")


def gold_log_probabilities(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the gold-token log probability at each scored position."""
    return F.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).squeeze(1)


def equal_probability_mixture(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Mix probabilities, not logits, using a fixed 0.5/0.5 weight."""
    if first.shape != second.shape:
        raise ValueError("model score shapes differ")
    return torch.logaddexp(first, second) - math.log(2.0)


def load_checked_models(
    checkpoint_paths: tuple[Path, Path], tokenizer_path: Path, split: str, corpus_path: Path,
    device: torch.device,
) -> tuple[list[TransformerLM], list[dict[str, object]], dict[str, object]]:
    tokenizer_sha = sha256_file(tokenizer_path)
    corpus_sha = sha256_file(corpus_path)
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        for path in checkpoint_paths
    ]
    contracts = [payload["run_fingerprint"]["contract"] for payload in payloads]
    if payloads[0]["step"] != payloads[1]["step"]:
        raise ValueError("checkpoints have different training steps")
    if payloads[0]["model_config"] != payloads[1]["model_config"]:
        raise ValueError("checkpoints have different model configurations")
    if contracts[0]["seed"] == contracts[1]["seed"]:
        raise ValueError("expected independently seeded checkpoints")
    for index, contract in enumerate(contracts):
        if contract["training_config"]["sampling_policy"] != POLICIES[index]:
            raise ValueError("the measured ensemble requires two document-uniform checkpoints")
        if contract["tokenizer_sha256"] != tokenizer_sha:
            raise ValueError("checkpoint tokenizer does not match the supplied tokenizer")
        if contract["dataset"][f"{split}_sha256"] != corpus_sha:
            raise ValueError(f"checkpoint {index} does not match the supplied {split} corpus")
    content_hashes = ("train_sha256", "validation_sha256", "test_sha256")
    if any(contracts[0]["dataset"][key] != contracts[1]["dataset"][key] for key in content_hashes):
        raise ValueError("checkpoints used different prepared corpus contents")
    if contracts[0]["training_config"] != contracts[1]["training_config"]:
        raise ValueError("checkpoints used different training configurations")

    models = []
    for payload in payloads:
        model = TransformerLM(ModelConfig(**payload["model_config"]))
        model.load_state_dict(payload["model_state"])
        models.append(model.to(device).eval())
    metadata = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "seed": contract["seed"],
            "step": payload["step"],
            "source_sha256": contract["source_sha256"],
        }
        for path, contract, payload in zip(checkpoint_paths, contracts, payloads, strict=True)
    ]
    controls = {
        "dataset_content_sha256": {key: contracts[0]["dataset"][key] for key in content_hashes},
        "dataset_manifest_sha256_by_checkpoint": [
            contract["dataset"]["manifest_sha256"] for contract in contracts
        ],
        "tokenizer_sha256": tokenizer_sha,
        "model_config": payloads[0]["model_config"],
        "training_config": contracts[0]["training_config"],
    }
    return models, metadata, controls


@torch.inference_mode()
def evaluate(
    models: list[TransformerLM], tokenizer: BytePairTokenizer, corpus_path: Path,
    *, device: torch.device, seq_len: int, stride: int, progress_every: int = 0,
) -> dict[str, object]:
    if not 1 <= stride <= seq_len:
        raise ValueError("stride must be between 1 and seq_len")
    if seq_len > models[0].config.max_seq_len:
        raise ValueError("evaluation context exceeds the model context")
    loss_sums = [0.0, 0.0, 0.0]
    documents: list[dict[str, object]] = []
    scored_tokens = 0
    corpus_bytes = 0
    windows = 0
    start_time = time.perf_counter()
    for index, document in enumerate(read_jsonl(corpus_path)):
        ids = tokenizer.encode(document.text, add_eos=True)
        stream = torch.tensor([tokenizer.EOS_ID, *ids], dtype=torch.long)
        document_losses = [0.0, 0.0, 0.0]
        for target_start in range(0, len(ids), stride):
            target_end = min(len(ids), target_start + stride)
            context_start = max(0, target_end - seq_len)
            inputs = stream[context_start:target_end].unsqueeze(0).to(device)
            targets = stream[target_start + 1:target_end + 1].to(device)
            gold_scores = []
            for model in models:
                logits, _ = model(inputs)
                selected = logits[0, target_start - context_start:target_end - context_start]
                gold_scores.append(gold_log_probabilities(selected, targets))
            mixed = equal_probability_mixture(*gold_scores)
            for position, scores in enumerate((*gold_scores, mixed)):
                value = -float(scores.sum())
                loss_sums[position] += value
                document_losses[position] += value
            windows += 1
        documents.append({
            "index": index,
            "tokens": len(ids),
            "negative_log_likelihood": {
                name: value for name, value in zip(("first", "second", "ensemble"), document_losses, strict=True)
            },
        })
        scored_tokens += len(ids)
        corpus_bytes += len(document.text.encode("utf-8"))
        if progress_every and (index + 1) % progress_every == 0:
            print(f"scored {index + 1} documents, {scored_tokens} tokens", file=sys.stderr, flush=True)
    if not scored_tokens or not corpus_bytes:
        raise ValueError("empty evaluation corpus")
    names = ("first", "second", "ensemble")
    losses = {name: value / scored_tokens for name, value in zip(names, loss_sums, strict=True)}
    perplexities = {name: math.exp(loss) for name, loss in losses.items()}
    bpb = {name: value / (math.log(2.0) * corpus_bytes) for name, value in zip(names, loss_sums, strict=True)}
    best_single = "first" if perplexities["first"] <= perplexities["second"] else "second"
    return {
        "documents": len(documents),
        "evaluated_tokens": scored_tokens,
        "corpus_text_bytes": corpus_bytes,
        "forward_windows": windows,
        "model_forward_calls": 2 * windows,
        "cross_entropy_loss": losses,
        "perplexity": perplexities,
        "bits_per_byte": bpb,
        "relative_ppl_change_vs_best_single": (
            perplexities["ensemble"] / min(perplexities["first"], perplexities["second"]) - 1
        ),
        "best_single": best_single,
        "improved_documents_vs_best_single": sum(
            item["negative_log_likelihood"]["ensemble"]
            < item["negative_log_likelihood"][best_single]
            for item in documents
        ),
        "document_results": documents,
        "wall_time_seconds": time.perf_counter() - start_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True,
                        help="provide exactly two independently seeded document-uniform checkpoints")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.checkpoint) != 2:
        parser.error("exactly two --checkpoint arguments are required")
    device = resolve_device(args.device)
    models, checkpoints, controls = load_checked_models(
        tuple(args.checkpoint), args.tokenizer, args.split, args.jsonl, device
    )
    tokenizer = BytePairTokenizer.load(args.tokenizer)
    result = {
        "format": "forgelm-two-seed-probability-ensemble-v1",
        "method": "fixed_equal_probability_mixture",
        "mixture_weights": [0.5, 0.5],
        "split": args.split,
        "corpus_path": str(args.jsonl.resolve()),
        "corpus_sha256": sha256_file(args.jsonl),
        "checkpoints": checkpoints,
        "controls": controls,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "device": str(device),
        "precision": "fp32",
        "environment": environment_metadata(device),
        "scope": (
            "retrospective test; the component models' test scores were previously known"
            if args.split == "test" else "validation-only model comparison"
        ),
        **evaluate(models, tokenizer, args.jsonl, device=device, seq_len=args.seq_len,
                   stride=args.stride, progress_every=args.progress_every),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "split", "evaluated_tokens", "perplexity", "relative_ppl_change_vs_best_single",
        "improved_documents_vs_best_single", "wall_time_seconds"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
