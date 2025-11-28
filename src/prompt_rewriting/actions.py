from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List


@dataclass(frozen=True)
class RewriteAction:
    """Encapsulates a single rewriting strategy."""

    name: str
    description: str
    transform: Callable[[str], str]

    def apply(self, sentence: str) -> str:
        return self.transform(sentence)


def _lower_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def _ensure_sentence_punctuation(text: str) -> str:
    stripped = text.rstrip()
    if not stripped:
        return stripped
    return stripped if stripped[-1] in ".!?" else f"{stripped}."


NEGATIVE_REPLACEMENTS = {
    "annoying": "a bit challenging",
    "awful": "difficult",
    "bad": "not ideal",
    "can't": "am having trouble",
    "fix": "take a look at",
    "hate": "really dislike",
    "immediately": "as soon as you can",
    "now": "when you have a moment",
    "problem": "issue",
    "terrible": "not ideal",
}


def polite_prefix(sentence: str) -> str:
    stripped = sentence.strip()
    if not stripped:
        return stripped
    lowered = stripped.lower()
    if lowered.startswith(("please", "could you", "would you", "kindly")):
        return stripped
    body = _lower_first(stripped)
    return _ensure_sentence_punctuation(f"Please {body}")


def supportive_suffix(sentence: str) -> str:
    stripped = sentence.strip()
    if not stripped:
        return "Thank you for your help!"
    if re.search(r"thanks|thank you", stripped, re.IGNORECASE):
        return _ensure_sentence_punctuation(stripped)
    base = stripped.rstrip(".!? ")
    return f"{base}. Thanks so much for your help!"


def soften_imperative(sentence: str) -> str:
    stripped = sentence.strip()
    if not stripped:
        return stripped
    lower = stripped.lower()
    if lower.startswith(("could you", "would you", "can you", "please")):
        return _ensure_sentence_punctuation(stripped)
    core = stripped.rstrip(".!? ")
    core = _lower_first(core)
    return f"Could you please {core}?"


def replace_negative_tones(sentence: str) -> str:
    def substitute(match: re.Match[str]) -> str:
        word = match.group(0)
        replacement = NEGATIVE_REPLACEMENTS[word.lower()]
        if word[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(NEGATIVE_REPLACEMENTS, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )
    updated = pattern.sub(substitute, sentence)
    return _ensure_sentence_punctuation(updated.strip())


def add_empathy(sentence: str) -> str:
    stripped = sentence.strip()
    empathy = " I understand this might take extra time, and I appreciate your effort."
    if not stripped:
        return empathy.strip()
    if re.search(r"appreciate", stripped, re.IGNORECASE):
        return _ensure_sentence_punctuation(stripped)
    return _ensure_sentence_punctuation(stripped.rstrip(".!?")) + empathy


DEFAULT_ACTIONS: List[RewriteAction] = [
    RewriteAction(
        name="polite_prefix",
        description="Adds a polite opening like 'Please'.",
        transform=polite_prefix,
    ),
    RewriteAction(
        name="soften_imperative",
        description="Turns direct commands into courteous questions.",
        transform=soften_imperative,
    ),
    RewriteAction(
        name="supportive_suffix",
        description="Adds appreciation and gratitude to the end of the sentence.",
        transform=supportive_suffix,
    ),
    RewriteAction(
        name="replace_negative_tones",
        description="Replaces sharp or negative words with softer alternatives.",
        transform=replace_negative_tones,
    ),
    RewriteAction(
        name="add_empathy",
        description="Acknowledges effort and shows understanding.",
        transform=add_empathy,
    ),
]
