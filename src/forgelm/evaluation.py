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
from .fingerprint import sha256_file
from .reproducibility import environment_metadata, resolve_device
from .sampling import DocumentWindowSampler, encode_documents_with_spans
from .tokenizer import BytePairTokenizer
from .training import load_model_from_checkpoint


def rolling_document_loss(
    model: torch.nn.Module,
    document_tokens: list[int],
    *,
    eos_id: int,
    seq_len: int,
    stride: int,
    device: torch.device,
    autocast: Any,
) -> tuple[float, int, int]:
    """Score every document token once, conditioned only on that document."""
    if not document_tokens:
        return 0.0, 0, 0
    stream = torch.tensor([eos_id, *document_tokens], dtype=torch.long)
    loss_sum = 0.0
    windows = 0
    for target_start in range(0, len(document_tokens), stride):
        target_end = min(len(document_tokens), target_start + stride)
        context_start = max(0, target_end - seq_len)
        inputs = stream[context_start:target_end].unsqueeze(0).to(device)
        targets = stream[target_start + 1 : target_end + 1].to(device)
        with autocast():
            logits, _ = model(inputs)
        selected = logits[0, target_start - context_start : target_end - context_start]
        loss_sum += float(F.cross_entropy(selected.float(), targets, reduction="sum"))
        windows += 1
    return loss_sum, len(document_tokens), windows


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
    full_corpus: bool = False,
    stride: int | None = None,
) -> dict[str, Any]:
    if batch_size < 1 or seq_len < 1 or eval_batches < 1:
        raise ValueError("batch_size, seq_len, and eval_batches must be positive")
    stride = max(1, seq_len // 2) if stride is None else stride
    if not 1 <= stride <= seq_len:
        raise ValueError("evaluation stride must be between 1 and seq_len")
    device = resolve_device(device_name)
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA GPU/PyTorch 组合不支持 BF16，请将配置中的 precision 改为 fp32")
    tokenizer = BytePairTokenizer.load(tokenizer_path)
    model = load_model_from_checkpoint(checkpoint_path, device)
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("evaluation tokenizer vocabulary differs from the checkpoint")
    if seq_len > model.config.max_seq_len:
        raise ValueError("evaluation seq_len exceeds the checkpoint model max_seq_len")
    documents = list(read_jsonl(validation_jsonl))
    tokens, spans = encode_documents_with_spans((document.text for document in documents), tokenizer, seq_len)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    autocast = (
        (lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16))
        if precision == "bf16"
        else nullcontext
    )
    losses: list[float] = []
    evaluated_tokens = 0
    loss_sum = 0.0
    forward_windows = 0
    start = time.perf_counter()
    if full_corpus:
        for document in documents:
            document_tokens = tokenizer.encode(document.text, add_eos=True)
            document_loss, count, windows = rolling_document_loss(
                model, document_tokens, eos_id=tokenizer.EOS_ID, seq_len=seq_len,
                stride=stride, device=device, autocast=autocast,
            )
            loss_sum += document_loss
            evaluated_tokens += count
            forward_windows += windows
        if evaluated_tokens != int(tokens.numel()):
            raise AssertionError("full-corpus evaluation did not cover every next-token target")
        loss = loss_sum / evaluated_tokens
    else:
        sampler = DocumentWindowSampler(
            tokens, spans, seq_len, policy="document_uniform", tier_probabilities=(1.0, 0.0, 0.0)
        )
        for _ in range(eval_batches):
            inputs, targets = sampler.sample_batch(batch_size, device, generator)
            with autocast():
                logits, _ = model(inputs)
            cross_entropy = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
            losses.append(float(cross_entropy))
            evaluated_tokens += targets.numel()
            forward_windows += batch_size
        loss = sum(losses) / len(losses)
    elapsed = max(time.perf_counter() - start, 1e-9)
    corpus_bytes = sum(len(document.text.encode("utf-8")) for document in documents)
    if corpus_bytes == 0:
        raise ValueError("evaluation corpus has no UTF-8 text bytes")
    report = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split": split_name,
        "dataset_jsonl": str(Path(validation_jsonl).resolve()),
        "dataset_sha256": sha256_file(validation_jsonl),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "dataset_documents": len(documents),
        "dataset_stream_tokens": int(tokens.numel()),
        "dataset_text_bytes": corpus_bytes,
        "evaluated_tokens": evaluated_tokens,
        "forward_windows": forward_windows,
        "evaluation_mode": "full_corpus" if full_corpus else "random_windows",
        "document_bounded": True,
        "full_corpus_stride": stride if full_corpus else None,
        "first_token_conditioning": "EOS" if full_corpus else None,
        "cross_entropy_loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "bits_per_byte": (
            loss * int(tokens.numel()) / (math.log(2.0) * corpus_bytes) if full_corpus else None
        ),
        "tokens_per_second": evaluated_tokens / elapsed,
        "batch_size": 1 if full_corpus else batch_size,
        "requested_batch_size": batch_size,
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
