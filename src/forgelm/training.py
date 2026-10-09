from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn import functional as F

from .config import ModelConfig, TrainingConfig
from .model import TransformerLM
from .reproducibility import environment_metadata, resolve_device, set_seed
from .sampling import DocumentSpan, DocumentWindowSampler
from .fingerprint import sha256_file


def encode_documents(texts: Iterable[str], tokenizer: Any) -> torch.Tensor:
    token_ids: list[int] = []
    for text in texts:
        token_ids.extend(tokenizer.encode(text, add_eos=True))
    if not token_ids:
        raise ValueError("no tokens were produced")
    return torch.tensor(token_ids, dtype=torch.long)


def sample_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 1 or len(tokens) <= seq_len:
        raise ValueError(f"token stream must contain more than {seq_len} tokens")
    starts = torch.randint(0, len(tokens) - seq_len, (batch_size,), generator=generator)
    inputs = torch.stack([tokens[start : start + seq_len] for start in starts.tolist()])
    targets = torch.stack([tokens[start + 1 : start + seq_len + 1] for start in starts.tolist()])
    return inputs.to(device), targets.to(device)


def cosine_learning_rate(step: int, config: TrainingConfig) -> float:
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if config.steps == config.warmup_steps:
        return config.learning_rate
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps - 1)
    progress = min(1.0, max(0.0, progress))
    minimum = config.learning_rate * config.min_learning_rate_ratio
    return minimum + 0.5 * (config.learning_rate - minimum) * (1.0 + math.cos(math.pi * progress))


def wsd_learning_rate(step: int, config: TrainingConfig) -> float:
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    decay_start = config.steps - config.wsd_decay_steps
    if step < decay_start:
        return config.learning_rate
    progress = (step - decay_start) / max(1, config.wsd_decay_steps - 1)
    progress = min(1.0, max(0.0, progress))
    minimum = config.learning_rate * config.min_learning_rate_ratio
    return minimum + 0.5 * (config.learning_rate - minimum) * (1.0 + math.cos(math.pi * progress))


def learning_rate_for_step(step: int, config: TrainingConfig) -> float:
    return wsd_learning_rate(step, config) if config.lr_schedule == "wsd" else cosine_learning_rate(step, config)


class Trainer:
    def __init__(
        self,
        model: TransformerLM,
        train_tokens: torch.Tensor,
        validation_tokens: torch.Tensor,
        config: TrainingConfig,
        *,
        seed: int,
        artifact_dir: str | Path,
        reset_metrics: bool = True,
        run_fingerprint: dict[str, Any] | None = None,
        train_spans: list[DocumentSpan] | None = None,
        validation_spans: list[DocumentSpan] | None = None,
    ) -> None:
        self.config = config
        self.seed = seed
        set_seed(seed)
        self.device = resolve_device(config.device)
        if config.precision == "bf16" and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA GPU/PyTorch 组合不支持 BF16，请将 precision 改为 fp32")
        self.model = model.to(self.device)
        self.model.set_gradient_checkpointing(config.activation_checkpointing)
        self.execution_model = torch.compile(self.model) if config.compile_model else self.model
        self.train_tokens = train_tokens.cpu()
        self.validation_tokens = validation_tokens.cpu()
        self.validation_sampler = (
            DocumentWindowSampler(
                self.validation_tokens,
                validation_spans,
                config.seq_len,
                policy="document_uniform",
                tier_probabilities=(1.0, 0.0, 0.0),
            )
            if validation_spans is not None else None
        )
        self.document_sampler = None
        if config.sampling_policy != "stream_uniform":
            if train_spans is None:
                raise ValueError("document sampling requires encoded document spans")
            self.document_sampler = DocumentWindowSampler(
                self.train_tokens,
                train_spans,
                config.seq_len,
                policy=config.sampling_policy,
                tier_probabilities=tuple(config.quality_tier_probabilities),
            )
        if config.fused_optimizer and self.device.type != "cuda":
            raise RuntimeError("fused AdamW is only enabled for CUDA in ForgeLM")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
            fused=config.fused_optimizer,
        )
        self.step = 0
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.artifact_dir / "metrics.jsonl"
        if reset_metrics or not self.metrics_path.exists():
            self.metrics_path.write_text("", encoding="utf-8")
        self.train_generator = torch.Generator(device="cpu").manual_seed(seed)
        self.run_fingerprint = run_fingerprint
        self.parent_checkpoint: dict[str, Any] | None = None

    def _autocast_context(self):
        if self.config.precision == "bf16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def _peak_memory_bytes(self) -> int | None:
        if self.device.type == "cuda":
            return int(torch.cuda.max_memory_allocated(self.device))
        if self.device.type == "mps":
            return int(torch.mps.current_allocated_memory())
        return None

    def _write_metric(self, metric: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, sort_keys=True) + "\n")

    @torch.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        generator = torch.Generator(device="cpu").manual_seed(self.seed + 10_000)
        losses: list[float] = []
        for _ in range(self.config.eval_batches):
            if self.validation_sampler is None:
                inputs, targets = sample_batch(
                    self.validation_tokens,
                    batch_size=self.config.batch_size,
                    seq_len=self.config.seq_len,
                    device=self.device,
                    generator=generator,
                )
            else:
                inputs, targets = self.validation_sampler.sample_batch(
                    self.config.batch_size, self.device, generator
                )
            with self._autocast_context():
                logits, loss = self.execution_model(inputs, targets)
            assert loss is not None
            cross_entropy = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
            losses.append(float(cross_entropy))
        return sum(losses) / len(losses)

    def save_checkpoint(self, filename: str = "checkpoint_last.pt") -> Path:
        path = self.artifact_dir / filename
        torch.save(
            {
                "format_version": 1,
                "step": self.step,
                "seed": self.seed,
                "model_config": asdict(self.model.config),
                "training_config": asdict(self.config),
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "train_generator_state": self.train_generator.get_state(),
                "train_tier_draw_counts": self.document_sampler.tier_draw_counts
                if self.document_sampler is not None else None,
                "run_fingerprint": self.run_fingerprint,
                "parent_checkpoint": self.parent_checkpoint,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        checkpoint_fingerprint = checkpoint.get("run_fingerprint")
        if self.run_fingerprint is not None:
            if checkpoint_fingerprint is None:
                raise ValueError("checkpoint has no run fingerprint; refusing strict resume")
            expected = self.run_fingerprint.get("fingerprint_id")
            actual = checkpoint_fingerprint.get("fingerprint_id")
            if expected != actual:
                raise ValueError(f"resume fingerprint mismatch: expected {expected}, checkpoint has {actual}")
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.train_generator.set_state(checkpoint["train_generator_state"].cpu())
        if self.document_sampler is not None and checkpoint.get("train_tier_draw_counts") is not None:
            self.document_sampler.tier_draw_counts = list(checkpoint["train_tier_draw_counts"])
        self.step = int(checkpoint["step"])
        self.parent_checkpoint = checkpoint.get("parent_checkpoint")

    def fork_from_checkpoint(self, path: str | Path) -> None:
        """Start a new run from a common prefix while changing only sampling policy."""
        source = Path(path)
        checkpoint = torch.load(source, map_location=self.device, weights_only=False)
        if checkpoint.get("format_version") != 1 or self.run_fingerprint is None:
            raise ValueError("fork requires a versioned checkpoint and a current run fingerprint")
        parent = checkpoint.get("run_fingerprint")
        if not isinstance(parent, dict) or "contract" not in parent:
            raise ValueError("fork parent lacks a run contract")
        expected = self.run_fingerprint["contract"]
        actual = parent["contract"]
        for key in ("dataset", "tokenizer_sha256", "quality_model_sha256", "model_config", "seed", "source_sha256"):
            if actual.get(key) != expected.get(key):
                raise ValueError(f"fork contract mismatch: {key}")
        old_training = dict(actual["training_config"])
        new_training = dict(expected["training_config"])
        old_training.pop("sampling_policy", None)
        new_training.pop("sampling_policy", None)
        if old_training != new_training:
            raise ValueError("fork training config may only change sampling_policy")
        if int(checkpoint["step"]) >= self.config.steps:
            raise ValueError("fork checkpoint must precede the configured final step")
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.train_generator.set_state(checkpoint["train_generator_state"].cpu())
        self.step = int(checkpoint["step"])
        expected_lr = learning_rate_for_step(self.step - 1, self.config)
        if any(abs(group["lr"] - expected_lr) > max(1e-12, abs(expected_lr) * 1e-6) for group in self.optimizer.param_groups):
            raise ValueError("fork optimizer learning rate differs from the shared schedule")
        self.parent_checkpoint = {
            "sha256": sha256_file(source),
            "step": self.step,
            "fingerprint_id": parent["fingerprint_id"],
            "sampling_policy": actual["training_config"]["sampling_policy"],
        }

    def train(self, *, stop_step: int | None = None) -> dict[str, Any]:
        target_step = self.config.steps if stop_step is None else stop_step
        if not self.step < target_step <= self.config.steps:
            raise ValueError("stop_step must be after the current step and no later than training.steps")
        start_wall_time = time.perf_counter()
        starting_step = self.step
        maximum_peak_memory = 0
        starting_tier_counts = (
            list(self.document_sampler.tier_draw_counts) if self.document_sampler is not None else None
        )
        initial_validation_loss = self.evaluate()
        self._write_metric({"step": self.step, "split": "validation", "loss": initial_validation_loss})
        last_validation_loss = initial_validation_loss

        while self.step < target_step:
            step_start = time.perf_counter()
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            tokens_this_step = 0
            learning_rate = learning_rate_for_step(self.step, self.config)
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            for _ in range(self.config.gradient_accumulation_steps):
                if self.document_sampler is None:
                    inputs, targets = sample_batch(
                        self.train_tokens,
                        batch_size=self.config.batch_size,
                        seq_len=self.config.seq_len,
                        device=self.device,
                        generator=self.train_generator,
                    )
                else:
                    inputs, targets = self.document_sampler.sample_batch(
                        self.config.batch_size, self.device, self.train_generator
                    )
                with self._autocast_context():
                    logits, loss = self.execution_model(inputs, targets)
                assert loss is not None
                (loss / self.config.gradient_accumulation_steps).backward()
                accumulated_loss += float(loss.detach()) / self.config.gradient_accumulation_steps
                tokens_this_step += inputs.numel()
                if self.model.config.z_loss_weight:
                    z_value = torch.logsumexp(logits.detach().float(), dim=-1).square().mean()
                else:
                    z_value = None

            gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {self.step}")
            self.optimizer.step()
            self.step += 1
            elapsed = max(time.perf_counter() - step_start, 1e-9)
            estimated_tflops = (
                6.0 * self.model.parameter_count(non_embedding=True) * tokens_this_step / elapsed / 1e12
            )
            peak_memory = self._peak_memory_bytes()
            if peak_memory is not None:
                maximum_peak_memory = max(maximum_peak_memory, peak_memory)
            train_metric: dict[str, Any] = {
                "step": self.step,
                "split": "train",
                "loss": accumulated_loss,
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm),
                "tokens_per_second": tokens_this_step / elapsed,
                "estimated_tflops": estimated_tflops,
            }
            if peak_memory is not None:
                train_metric["peak_memory_bytes"] = peak_memory
            if self.config.peak_tflops > 0:
                train_metric["mfu"] = estimated_tflops / self.config.peak_tflops
            if z_value is not None:
                train_metric["z_loss_unweighted"] = float(z_value)
            self._write_metric(
                train_metric
            )

            if self.step % self.config.eval_interval == 0 or self.step == target_step:
                last_validation_loss = self.evaluate()
                self._write_metric({"step": self.step, "split": "validation", "loss": last_validation_loss})
            if self.step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.step:06d}.pt")

        checkpoint_path = self.save_checkpoint()
        wall_time = max(time.perf_counter() - start_wall_time, 1e-9)
        tokens_per_step = self.config.batch_size * self.config.seq_len * self.config.gradient_accumulation_steps
        sampled_tokens_this_run = (self.step - starting_step) * tokens_per_step
        sampled_tokens_total = self.step * tokens_per_step
        tier_draw_counts = (
            [current - previous for current, previous in zip(
                self.document_sampler.tier_draw_counts, starting_tier_counts, strict=True
            )]
            if self.document_sampler is not None and starting_tier_counts is not None else None
        )
        summary = {
            "initial_validation_loss": initial_validation_loss,
            "final_validation_loss": last_validation_loss,
            "final_validation_perplexity": math.exp(min(last_validation_loss, 20.0)),
            "steps": self.step,
            "configured_steps": self.config.steps,
            "sampled_tokens_this_run": sampled_tokens_this_run,
            "sampled_tokens_total": sampled_tokens_total,
            "sampled_to_unique_token_ratio": sampled_tokens_total / int(self.train_tokens.numel()),
            "tier_draw_counts_this_run": tier_draw_counts,
            "end_to_end_sampled_tokens_per_second": sampled_tokens_this_run / wall_time,
            "max_peak_memory_bytes": maximum_peak_memory if maximum_peak_memory else None,
            "parent_checkpoint": self.parent_checkpoint,
            "wall_time_seconds": wall_time,
            "parameter_count": self.model.parameter_count(),
            "non_embedding_parameter_count": self.model.parameter_count(non_embedding=True),
            "train_tokens": int(self.train_tokens.numel()),
            "validation_tokens": int(self.validation_tokens.numel()),
            "checkpoint": str(checkpoint_path),
            "environment": environment_metadata(self.device),
            "system": {
                "precision": self.config.precision,
                "compile_model": self.config.compile_model,
                "activation_checkpointing": self.config.activation_checkpointing,
                "fused_optimizer": self.config.fused_optimizer,
                "lr_schedule": self.config.lr_schedule,
            },
            "run_fingerprint": self.run_fingerprint,
        }
        (self.artifact_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary


def load_model_from_checkpoint(path: str | Path, device: torch.device) -> TransformerLM:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model = TransformerLM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()
