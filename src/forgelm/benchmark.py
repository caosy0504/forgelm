from __future__ import annotations

import dataclasses
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .config import ModelConfig
from .model import TransformerLM
from .reproducibility import resolve_device, set_seed


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_model(
    model: TransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    precision: str = "fp32",
) -> dict[str, float | str | int]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    autocast = (
        (lambda: torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16))
        if precision == "bf16"
        else nullcontext
    )
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
    if inputs.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(inputs.device)
    synchronize(inputs.device)
    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
    synchronize(inputs.device)
    elapsed = time.perf_counter() - start
    tokens = inputs.numel() * iterations
    result = {
        "iterations": float(iterations),
        "elapsed_seconds": elapsed,
        "step_time_ms": 1000.0 * elapsed / iterations,
        "tokens_per_second": tokens / elapsed,
        "precision": precision,
    }
    if inputs.device.type == "cuda":
        result["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(inputs.device))
    return result


def probe_flash_attention(
    config: ModelConfig, *, batch_size: int, seq_len: int, device: torch.device, precision: str
) -> dict[str, Any]:
    if device.type != "cuda":
        return {"status": "not_cuda"}
    from torch.nn.attention import SDPBackend, sdpa_kernel

    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    head_dim = config.d_model // config.n_heads
    query = torch.randn(batch_size, config.n_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    try:
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
            output.float().sum().backward()
        return {"status": "flash_forward_backward_available", "dtype": str(dtype), "head_dim": head_dim}
    except (RuntimeError, ValueError, NotImplementedError) as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}


def benchmark_attention_implementations(
    config: ModelConfig,
    *,
    batch_size: int,
    seq_len: int,
    device_name: str,
    warmup: int = 2,
    iterations: int = 5,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if config.vocab_size < 258:
        config = dataclasses.replace(config, vocab_size=320)
    if seq_len > config.max_seq_len:
        raise ValueError("benchmark sequence exceeds model.max_seq_len")
    set_seed(1234)
    device = resolve_device(device_name)
    eager = TransformerLM(dataclasses.replace(config, attention_impl="eager")).to(device)
    sdpa = TransformerLM(dataclasses.replace(config, attention_impl="sdpa")).to(device)
    sdpa.load_state_dict(eager.state_dict())
    inputs = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    eager.eval()
    sdpa.eval()
    with torch.inference_mode():
        eager_logits, _ = eager(inputs)
        sdpa_logits, _ = sdpa(inputs)
    max_error = float((eager_logits - sdpa_logits).abs().max())
    eager.train()
    sdpa.train()
    result = {
        "device": str(device),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "parameter_count": eager.parameter_count(),
        "max_forward_absolute_error": max_error,
        "eager": _time_model(eager, inputs, targets, warmup=warmup, iterations=iterations),
        "sdpa": _time_model(sdpa, inputs, targets, warmup=warmup, iterations=iterations),
    }
    result["sdpa_speedup"] = result["sdpa"]["tokens_per_second"] / result["eager"]["tokens_per_second"]
    if device.type == "cuda":
        result["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def benchmark_modern_training_stack(
    config: ModelConfig,
    *,
    batch_size: int,
    seq_len: int,
    device_name: str,
    precision: str,
    compile_model: bool,
    activation_checkpointing: bool,
    warmup: int = 2,
    iterations: int = 5,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if config.vocab_size < 258:
        config = dataclasses.replace(config, vocab_size=320)
    set_seed(2026)
    device = resolve_device(device_name)
    baseline = TransformerLM(dataclasses.replace(config, attention_impl="eager")).to(device)
    optimized = TransformerLM(dataclasses.replace(config, attention_impl="sdpa")).to(device)
    optimized.load_state_dict(baseline.state_dict())
    optimized.set_gradient_checkpointing(activation_checkpointing)
    inputs = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    baseline.eval()
    optimized.eval()
    with torch.inference_mode():
        baseline_logits, _ = baseline(inputs)
        optimized_logits, _ = optimized(inputs)
    max_error = float((baseline_logits - optimized_logits).abs().max())
    baseline.train()
    optimized.train()
    execution_model = torch.compile(optimized) if compile_model else optimized
    result: dict[str, Any] = {
        "device": str(device),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "parameter_count": baseline.parameter_count(),
        "vocab_size": config.vocab_size,
        "flash_attention_probe": probe_flash_attention(
            config, batch_size=batch_size, seq_len=seq_len, device=device, precision=precision
        ),
        "max_forward_absolute_error": max_error,
        "baseline": _time_model(
            baseline, inputs, targets, warmup=warmup, iterations=iterations, precision="fp32"
        ),
        "optimized": _time_model(
            execution_model,
            inputs,
            targets,
            warmup=warmup,
            iterations=iterations,
            precision=precision,
        ),
        "features": {
            "attention": "sdpa",
            "compile_model": compile_model,
            "activation_checkpointing": activation_checkpointing,
            "precision": precision,
        },
    }
    result["optimized_speedup"] = (
        result["optimized"]["tokens_per_second"] / result["baseline"]["tokens_per_second"]
    )
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
