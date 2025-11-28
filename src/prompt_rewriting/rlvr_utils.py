from __future__ import annotations

from typing import Iterable, Sequence


def extract_generated_responses(
    tokenizer,
    query_token_ids: Iterable[Sequence[int]],
    response_token_ids: Iterable[Sequence[int]],
) -> list[str]:
    """Strip the original prompt tokens from model generations.

    TRL returns sequences that prepend the original prompt to the generated
    continuation. The reward model should only see the newly generated text, so
    we drop the prompt prefix on a best-effort basis.
    """

    prompt_texts = tokenizer.batch_decode(query_token_ids, skip_special_tokens=True)
    response_texts = tokenizer.batch_decode(response_token_ids, skip_special_tokens=True)
    rewrites: list[str] = []

    for prompt_text, response_text in zip(prompt_texts, response_texts):
        if response_text.startswith(prompt_text):
            rewritten_segment = response_text[len(prompt_text) :]
            rewrite = rewritten_segment.strip() or response_text.strip()
        else:
            rewrite = response_text.strip()
        rewrites.append(rewrite)

    return rewrites
