from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import TransformerLM
from .tokenizer import BytePairTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "The following is a conversation between a user and an assistant. "
    "The assistant tries to answer clearly and honestly."
)


@dataclass(frozen=True)
class ChatTurn:
    user: str
    assistant: str


def build_chat_prompt(system_prompt: str, history: list[ChatTurn], user_message: str) -> str:
    parts = [system_prompt.strip()]
    for turn in history:
        parts.append(f"User: {turn.user.strip()}\nAssistant: {turn.assistant.strip()}")
    parts.append(f"User: {user_message.strip()}\nAssistant:")
    return "\n\n".join(parts)


@torch.inference_mode()
def generate_chat_response(
    model: TransformerLM,
    tokenizer: BytePairTokenizer,
    *,
    system_prompt: str,
    history: list[ChatTurn],
    user_message: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    generator: torch.Generator | None,
) -> str:
    prompt = build_chat_prompt(system_prompt, history, user_message)
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("chat prompt produced no tokens")
    generated = model.generate(
        torch.tensor([prompt_ids], dtype=torch.long, device=device),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.EOS_ID,
        generator=generator,
    )[0].tolist()
    response = tokenizer.decode(generated[len(prompt_ids) :]).strip()
    for stop_text in ("\nUser:", "\n\nUser:"):
        if stop_text in response:
            response = response.split(stop_text, maxsplit=1)[0].strip()
    return response

