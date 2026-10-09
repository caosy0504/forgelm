from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


@dataclass
class StaticKVCache:
    key: torch.Tensor
    value: torch.Tensor
    length: int = 0

    @classmethod
    def allocate(
        cls, batch_size: int, heads: int, capacity: int, head_dim: int, *, device: torch.device, dtype: torch.dtype
    ) -> "StaticKVCache":
        shape = (batch_size, heads, capacity, head_dim)
        return cls(torch.empty(shape, device=device, dtype=dtype), torch.empty(shape, device=device, dtype=dtype))

    def reset(self) -> None:
        self.length = 0


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

    def forward(self, inputs: torch.Tensor, start_position: int = 0) -> torch.Tensor:
        sequence_length = inputs.shape[-2]
        end = start_position + sequence_length
        if end > self.cos.shape[0]:
            raise ValueError("RoPE position exceeds model context")
        cos = self.cos[start_position:end].to(device=inputs.device, dtype=inputs.dtype)[None, None, :, :]
        sin = self.sin[start_position:end].to(device=inputs.device, dtype=inputs.dtype)[None, None, :, :]
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

    def forward_cached(self, inputs: torch.Tensor, cache: StaticKVCache) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("KV cache is for inference only")
        batch_size, sequence_length, model_dim = inputs.shape
        past = cache.length
        end = past + sequence_length
        if end > cache.key.shape[-2]:
            raise ValueError("KV cache capacity exceeded")
        qkv = self.qkv(inputs).view(batch_size, sequence_length, 3, self.n_heads, self.head_dim)
        query, key, value = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = self.rope(query, past)
        key = self.rope(key, past)
        cache.key[:, :, past:end].copy_(key)
        cache.value[:, :, past:end].copy_(value)
        cache.length = end
        keys = cache.key[:, :, :end]
        values = cache.value[:, :, :end]
        if past == 0:
            attended = F.scaled_dot_product_attention(query, keys, values, is_causal=True)
        elif sequence_length == 1:
            attended = F.scaled_dot_product_attention(query, keys, values, is_causal=False)
        else:
            query_positions = past + torch.arange(sequence_length, device=inputs.device)
            key_positions = torch.arange(end, device=inputs.device)
            mask = key_positions[None, :] <= query_positions[:, None]
            attended = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
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

    def forward_cached(self, inputs: torch.Tensor, cache: StaticKVCache) -> torch.Tensor:
        inputs = inputs + self.attention.forward_cached(self.attention_norm(inputs), cache)
        return inputs + self.feed_forward(self.feed_forward_norm(inputs))


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

    def allocate_kv_cache(self, batch_size: int, device: torch.device) -> list[StaticKVCache]:
        dtype = (
            torch.get_autocast_dtype(device.type)
            if torch.is_autocast_enabled(device.type)
            else self.token_embedding.weight.dtype
        )
        return [
            StaticKVCache.allocate(
                batch_size,
                self.config.n_heads,
                self.config.max_seq_len,
                self.config.d_model // self.config.n_heads,
                device=device,
                dtype=dtype,
            )
            for _ in self.blocks
        ]

    def forward_cached(self, input_ids: torch.Tensor, caches: list[StaticKVCache]) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("cached forward requires eval and inference mode")
        if len(caches) != len(self.blocks) or not caches:
            raise ValueError("one KV cache per Transformer block is required")
        lengths = {cache.length for cache in caches}
        if len(lengths) != 1:
            raise ValueError("KV cache layers have inconsistent lengths")
        hidden = self.token_embedding(input_ids)
        for block, cache in zip(self.blocks, caches, strict=True):
            hidden = block.forward_cached(hidden, cache)
        return self.lm_head(self.final_norm(hidden))

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
        use_cache: bool = True,
    ) -> torch.Tensor:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.eval()
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty batch of token ids")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return input_ids
        prompt_length = input_ids.shape[1]
        generated = input_ids.new_empty((input_ids.shape[0], prompt_length + max_new_tokens))
        generated[:, :prompt_length].copy_(input_ids)
        generated_length = prompt_length
        caches = self.allocate_kv_cache(input_ids.shape[0], input_ids.device) if use_cache else None
        for _ in range(max_new_tokens):
            context = generated[:, max(0, generated_length - self.config.max_seq_len) : generated_length]
            if caches is None:
                logits, _ = self(context)
            else:
                cached_length = caches[0].length
                if cached_length >= self.config.max_seq_len:
                    for cache in caches:
                        cache.reset()
                    cached_length = 0
                model_input = context if cached_length == 0 else generated[:, generated_length - 1 : generated_length]
                logits = self.forward_cached(model_input, caches)
            next_logits = logits[:, -1, :]
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated[:, generated_length].copy_(next_token.squeeze(1))
                generated_length += 1
                if eos_id is not None and bool(torch.all(next_token == eos_id)):
                    break
                continue
            next_logits = next_logits / temperature
            if top_k > 0:
                k = min(top_k, next_logits.shape[-1])
                cutoff = torch.topk(next_logits, k).values[:, [-1]]
                next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            generated[:, generated_length].copy_(next_token.squeeze(1))
            generated_length += 1
            if eos_id is not None and bool(torch.all(next_token == eos_id)):
                break
        return generated[:, :generated_length]

    def parameter_count(self, *, non_embedding: bool = False) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            total -= self.token_embedding.weight.numel()
        return total


def with_vocab_size(config: ModelConfig, vocab_size: int) -> ModelConfig:
    return replace(config, vocab_size=vocab_size)
