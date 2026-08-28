from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_dtype = inputs.dtype
        normalized = inputs.float() * torch.rsqrt(inputs.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(original_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
        inverse_frequency = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.shape[-2]
        cos = self.cos[:sequence_length].to(device=inputs.device, dtype=inputs.dtype)[None, None, :, :]
        sin = self.sin[:sequence_length].to(device=inputs.device, dtype=inputs.dtype)[None, None, :, :]
        even, odd = inputs[..., 0::2], inputs[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.attention_impl = config.attention_impl
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else None
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, theta=config.rope_theta)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, model_dim = inputs.shape
        qkv = self.qkv(inputs).view(batch_size, sequence_length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = self.rope(query)
        key = self.rope(key)
        value = value.transpose(1, 2)

        if self.attention_impl == "sdpa":
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = self.head_dim**-0.5
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            causal_mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=inputs.device).tril()
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
            probabilities = F.softmax(scores.float(), dim=-1).to(scores.dtype)
            probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
            attended = torch.matmul(probabilities, value)

        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, model_dim)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.value = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.output = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.gate(inputs)) * self.value(inputs))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(config.d_model)
        self.feed_forward = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.residual_dropout(self.attention(self.attention_norm(inputs)))
        return inputs + self.residual_dropout(self.feed_forward(self.feed_forward_norm(inputs)))


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        if config.vocab_size < 258:
            raise ValueError("model.vocab_size must be supplied by the trained tokenizer")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.gradient_checkpointing = False
        self.apply(self._initialize_weights)
        if config.residual_scaled_init:
            residual_std = 0.02 / math.sqrt(2 * config.n_layers)
            for block in self.blocks:
                nn.init.normal_(block.attention.output.weight, mean=0.0, std=residual_std)
                nn.init.normal_(block.feed_forward.output.weight, mean=0.0, std=residual_std)

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds model.max_seq_len")
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            if self.config.z_loss_weight:
                z_loss = torch.logsumexp(logits.float(), dim=-1).square().mean()
                loss = loss + self.config.z_loss_weight * z_loss
        return logits, loss

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                k = min(top_k, next_logits.shape[-1])
                cutoff = torch.topk(next_logits, k).values[:, [-1]]
                next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_id is not None and bool(torch.all(next_token == eos_id)):
                break
        return input_ids

    def parameter_count(self, *, non_embedding: bool = False) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            total -= self.token_embedding.weight.numel()
        return total


def with_vocab_size(config: ModelConfig, vocab_size: int) -> ModelConfig:
    return replace(config, vocab_size=vocab_size)
