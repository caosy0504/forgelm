from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .data_pipeline import read_jsonl
from .reproducibility import environment_metadata, resolve_device
from .tokenizer import BytePairTokenizer
from .training import encode_documents, load_model_from_checkpoint, sample_batch


@torch.inference_mode()
def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    validation_jsonl: str | Path,
    device_name: str,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    seed: int,
    precision: str,
    split_name: str = "validation",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA GPU/PyTorch 组合不支持 BF16，请将配置中的 precision 改为 fp32")
    tokenizer = BytePairTokenizer.load(tokenizer_path)
    model = load_model_from_checkpoint(checkpoint_path, device)
    if seq_len > model.config.max_seq_len:
        raise ValueError("evaluation seq_len exceeds the checkpoint model max_seq_len")
    documents = list(read_jsonl(validation_jsonl))
    tokens = encode_documents((document.text for document in documents), tokenizer)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    autocast = (
        (lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16))
        if precision == "bf16"
        else nullcontext
    )
    losses: list[float] = []
    evaluated_tokens = 0
    start = time.perf_counter()
    for _ in range(eval_batches):
        inputs, targets = sample_batch(
            tokens,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            generator=generator,
        )
        with autocast():
            logits, _ = model(inputs)
        cross_entropy = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
        losses.append(float(cross_entropy))
        evaluated_tokens += targets.numel()
    elapsed = max(time.perf_counter() - start, 1e-9)
    loss = sum(losses) / len(losses)
    report = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "split": split_name,
        "dataset_jsonl": str(Path(validation_jsonl).resolve()),
        "dataset_documents": len(documents),
        "dataset_stream_tokens": int(tokens.numel()),
        "evaluated_tokens": evaluated_tokens,
        "cross_entropy_loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "tokens_per_second": evaluated_tokens / elapsed,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "eval_batches": eval_batches,
        "precision": precision,
        "environment": environment_metadata(device),
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return report
