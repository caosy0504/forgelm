"""Correctness-first cached versus full-forward decoding benchmark."""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from .fingerprint import sha256_file
from .reproducibility import resolve_device
from .tokenizer import BytePairTokenizer
from .training import load_model_from_checkpoint


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def benchmark_decode(
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    prompt: str,
    new_tokens: int,
    trials: int,
    device_name: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if new_tokens < 1 or trials < 1:
        raise ValueError("new_tokens and trials must be positive")
    device = resolve_device(device_name)
    tokenizer = BytePairTokenizer.load(tokenizer_path)
    model = load_model_from_checkpoint(checkpoint_path, device)
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt produced no tokens")
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    uncached = model.generate(inputs, max_new_tokens=new_tokens, temperature=0, use_cache=False)
    cached = model.generate(inputs, max_new_tokens=new_tokens, temperature=0, use_cache=True)
    if not torch.equal(uncached, cached):
        raise AssertionError("cached and uncached greedy output differ")
    timings: dict[str, list[float]] = {"full_forward": [], "static_kv_cache": []}
    first_token_timings: dict[str, list[float]] = {"full_forward": [], "static_kv_cache": []}
    peak_memory: dict[str, int] = {"full_forward": 0, "static_kv_cache": 0}
    for trial in range(trials):
        for name, use_cache in (
            (("full_forward", False), ("static_kv_cache", True))
            if trial % 2 == 0 else (("static_kv_cache", True), ("full_forward", False))
        ):
            _synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            model.generate(inputs, max_new_tokens=new_tokens, temperature=0, use_cache=use_cache)
            _synchronize(device)
            timings[name].append(time.perf_counter() - start)
            if device.type == "cuda":
                peak_memory[name] = max(peak_memory[name], int(torch.cuda.max_memory_allocated(device)))
            _synchronize(device)
            start = time.perf_counter()
            model.generate(inputs, max_new_tokens=1, temperature=0, use_cache=use_cache)
            _synchronize(device)
            first_token_timings[name].append(time.perf_counter() - start)
    full_rate = statistics.median(new_tokens / seconds for seconds in timings["full_forward"])
    cached_rate = statistics.median(new_tokens / seconds for seconds in timings["static_kv_cache"])
    result = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "device": str(device),
        "model_parameters": model.parameter_count(),
        "prompt_tokens": len(prompt_ids),
        "new_tokens": new_tokens,
        "trials": trials,
        "greedy_output_equal": True,
        "full_forward_tokens_per_second_median": full_rate,
        "static_kv_cache_tokens_per_second_median": cached_rate,
        "speedup": cached_rate / full_rate,
        "first_token_latency_ms_median": {
            name: statistics.median(seconds * 1000 for seconds in values)
            for name, values in first_token_timings.items()
        },
        "total_latency_ms_median": {
            name: statistics.median(seconds * 1000 for seconds in values)
            for name, values in timings.items()
        },
        "total_latency_ms_p95": {
            name: sorted(seconds * 1000 for seconds in values)[math.ceil(0.95 * len(values)) - 1]
            for name, values in timings.items()
        },
        "peak_memory_bytes": peak_memory if device.type == "cuda" else None,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
