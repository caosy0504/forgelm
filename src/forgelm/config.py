from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    input_path: str
    validation_path: str | None = None
    test_path: str | None = None
    min_words: int = 20
    min_alpha_ratio: float = 0.6
    validation_fraction: float = 0.1
    near_dedup_threshold: float = 0.9
    ngram_size: int = 5
    num_hashes: int = 64
    num_bands: int = 8
    enable_pii_masking: bool = True
    enable_quality_rules: bool = True
    enable_exact_dedup: bool = True
    enable_near_dedup: bool = True
    enable_decontamination: bool = True
    decontamination_threshold: float = 0.85
    enable_model_quality: bool = False
    quality_seed_path: str | None = None
    quality_threshold: float = 0.5
    quality_hash_dim: int = 2048
    quality_epochs: int = 40
    quality_learning_rate: float = 0.1

    def validate(self) -> None:
        if self.test_path is not None and self.validation_path is None:
            raise ValueError("data.test_path requires an explicit data.validation_path")
        if self.min_words < 1:
            raise ValueError("data.min_words must be positive")
        if not 0.0 <= self.min_alpha_ratio <= 1.0:
            raise ValueError("data.min_alpha_ratio must be in [0, 1]")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("data.validation_fraction must be in (0, 1)")
        if not 0.0 <= self.near_dedup_threshold <= 1.0:
            raise ValueError("data.near_dedup_threshold must be in [0, 1]")
        if self.ngram_size < 1 or self.num_hashes < 1 or self.num_bands < 1:
            raise ValueError("ngram_size, num_hashes, and num_bands must be positive")
        if self.num_hashes % self.num_bands != 0:
            raise ValueError("data.num_hashes must be divisible by data.num_bands")
        if not 0.0 <= self.decontamination_threshold <= 1.0:
            raise ValueError("data.decontamination_threshold must be in [0, 1]")
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError("data.quality_threshold must be in [0, 1]")
        if self.quality_hash_dim < 32 or self.quality_epochs < 1 or self.quality_learning_rate <= 0:
            raise ValueError("invalid model-quality classifier settings")
        if self.enable_model_quality and not self.quality_seed_path:
            raise ValueError("data.quality_seed_path is required when model-quality filtering is enabled")


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 512

    def validate(self) -> None:
        if self.vocab_size < 258:
            raise ValueError("tokenizer.vocab_size must leave room for 256 bytes plus special tokens")


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 768
    max_seq_len: int = 256
    dropout: float = 0.0
    attention_impl: str = "sdpa"
    tie_embeddings: bool = True
    qk_norm: bool = False
    z_loss_weight: float = 0.0
    rope_theta: float = 10_000.0
    residual_scaled_init: bool = False
    vocab_size: int = 0

    def validate(self) -> None:
        if min(self.d_model, self.n_layers, self.n_heads, self.d_ff, self.max_seq_len) < 1:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.attention_impl not in {"eager", "sdpa"}:
            raise ValueError("model.attention_impl must be 'eager' or 'sdpa'")
        if self.z_loss_weight < 0 or self.rope_theta <= 0:
            raise ValueError("model.z_loss_weight must be non-negative and rope_theta must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 500
    batch_size: int = 16
    seq_len: int = 128
    gradient_accumulation_steps: int = 1
    learning_rate: float = 6e-4
    min_learning_rate_ratio: float = 0.1
    weight_decay: float = 0.1
    warmup_steps: int = 20
    gradient_clip: float = 1.0
    eval_interval: int = 25
    eval_batches: int = 10
    checkpoint_interval: int = 100
    device: str = "auto"
    lr_schedule: str = "cosine"
    wsd_decay_steps: int = 0
    precision: str = "fp32"
    compile_model: bool = False
    activation_checkpointing: bool = False
    fused_optimizer: bool = False
    peak_tflops: float = 0.0

    def validate(self, model: ModelConfig) -> None:
        integer_fields = (
            self.steps,
            self.batch_size,
            self.seq_len,
            self.gradient_accumulation_steps,
            self.eval_interval,
            self.eval_batches,
            self.checkpoint_interval,
        )
        if min(integer_fields) < 1:
            raise ValueError("training integer fields must be positive")
        if self.seq_len > model.max_seq_len:
            raise ValueError("training.seq_len cannot exceed model.max_seq_len")
        if self.warmup_steps < 0 or self.warmup_steps > self.steps:
            raise ValueError("training.warmup_steps must be between 0 and steps")
        if self.learning_rate <= 0 or not 0.0 <= self.min_learning_rate_ratio <= 1.0:
            raise ValueError("invalid learning-rate settings")
        if self.gradient_clip <= 0:
            raise ValueError("training.gradient_clip must be positive")
        if self.lr_schedule not in {"cosine", "wsd"}:
            raise ValueError("training.lr_schedule must be 'cosine' or 'wsd'")
        if self.wsd_decay_steps < 0 or self.wsd_decay_steps > self.steps - self.warmup_steps:
            raise ValueError("training.wsd_decay_steps must fit after warmup")
        if self.lr_schedule == "wsd" and self.wsd_decay_steps < 1:
            raise ValueError("training.wsd_decay_steps must be positive for WSD")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("training.precision must be 'fp32' or 'bf16'")
        if self.peak_tflops < 0:
            raise ValueError("training.peak_tflops must be non-negative")


@dataclass(frozen=True)
class GenerationConfig:
    prompt: str = "Once upon a time"
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_k: int = 40

    def validate(self) -> None:
        if self.max_new_tokens < 1 or self.temperature <= 0 or self.top_k < 0:
            raise ValueError("invalid generation settings")


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    output_dir: str
    data: DataConfig
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    root: Path = field(default_factory=Path, compare=False, repr=False)

    @classmethod
    def from_toml(cls, path: str | Path) -> "ProjectConfig":
        config_path = Path(path).resolve()
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        project_raw = raw.get("project", {})
        root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        config = cls(
            name=str(project_raw.get("name", "forgelm")),
            seed=int(project_raw.get("seed", 42)),
            output_dir=str(project_raw.get("output_dir", "artifacts/default")),
            data=DataConfig(**raw["data"]),
            tokenizer=TokenizerConfig(**raw.get("tokenizer", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            generation=GenerationConfig(**raw.get("generation", {})),
            root=root,
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.data.validate()
        self.tokenizer.validate()
        self.model.validate()
        self.training.validate(self.model)
        self.generation.validate()

    @property
    def input_path(self) -> Path:
        return (self.root / self.data.input_path).resolve()

    @property
    def artifact_dir(self) -> Path:
        return (self.root / self.output_dir).resolve()

    @property
    def quality_seed_path(self) -> Path | None:
        if self.data.quality_seed_path is None:
            return None
        return (self.root / self.data.quality_seed_path).resolve()

    @property
    def validation_path(self) -> Path | None:
        if self.data.validation_path is None:
            return None
        return (self.root / self.data.validation_path).resolve()

    @property
    def test_path(self) -> Path | None:
        if self.data.test_path is None:
            return None
        return (self.root / self.data.test_path).resolve()

    def as_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["root"] = str(self.root)
        return result
