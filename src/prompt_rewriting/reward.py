from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class RewardModel(ABC):
    """Common interface for reward models."""

    @abstractmethod
    def score(self, original: str, rewritten: str) -> float:
        """Return reward in the range [0, 1]."""


@dataclass
class HeuristicRewardModel(RewardModel):
    """Rule-based proxy for friendliness to support offline experimentation."""

    question_bonus: float = 0.1
    gratitude_bonus: float = 0.35
    empathy_bonus: float = 0.25
    courtesy_bonus: float = 0.3
    harshness_penalty: float = 0.4

    polite_patterns = (
        r"\bplease",
        r"\bkindly",
        r"\bwould you",
        r"\bcould you",
    )
    gratitude_patterns = (
        r"thank you",
        r"thanks",
        r"appreciate",
        r"grateful",
    )
    empathy_patterns = (
        r"i understand",
        r"i know",
        r"i realize",
        r"i appreciate",
    )
    harsh_patterns = (
        r"\bnow\b",
        r"\bimmediately\b",
        r"\bawful\b",
        r"\bterrible\b",
        r"\bannoying\b",
        r"\bdumb\b",
        r"\bhate\b",
        r"\bstupid\b",
    )

    def _tone_score(self, text: str) -> float:
        lowered = text.lower()
        score = 0.15

        if any(re.search(p, lowered) for p in self.polite_patterns):
            score += self.courtesy_bonus
        if any(re.search(p, lowered) for p in self.gratitude_patterns):
            score += self.gratitude_bonus
        if any(re.search(p, lowered) for p in self.empathy_patterns):
            score += self.empathy_bonus
        if "?" in text:
            score += self.question_bonus
        if any(re.search(p, lowered) for p in self.harsh_patterns):
            score -= self.harshness_penalty

        return max(0.0, min(1.0, score))

    def score(self, original: str, rewritten: str) -> float:
        original_score = self._tone_score(original)
        rewritten_score = self._tone_score(rewritten)
        improvement = rewritten_score - original_score
        if improvement <= 0:
            return max(0.0, rewritten_score * 0.1)
        return max(0.0, min(1.0, improvement))


def _parse_reward_score(text: str) -> float:
    """Extract a score between 0 and 1 from a model response."""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"([0-1](?:\.\d+)?)", text)
        return float(match.group(1)) if match else 0.0
    try:
        score = float(data.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return score


class OpenAIRewardModel(RewardModel):
    """LLM-based reward using the OpenAI Responses API."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_retries: int = 2,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The openai package is required for OpenAIRewardModel.") from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("An OpenAI API key must be provided via argument or OPENAI_API_KEY.")
        self.client = OpenAI(api_key=key)
        self.model = model
        self.max_retries = max_retries

    system_prompt = (
        "You are a reward model that evaluates whether a rewritten sentence sounds friendlier. "
        "Respond with JSON: {\"score\": <number between 0 and 1>, \"explanation\": <short reason>}"
    )

    def score(self, original: str, rewritten: str) -> float:  # pragma: no cover - network call
        attempts = 0
        message = (
            "Original sentence: "
            + original.strip()
            + "\nRewritten sentence: "
            + rewritten.strip()
            + "\nRate how much friendlier the rewrite is compared to the original."
        )

        while True:
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": message},
                    ],
                )
            except Exception as exc:  # noqa: BLE001
                attempts += 1
                if attempts > self.max_retries:
                    raise RuntimeError("Failed to obtain reward from OpenAI API") from exc
                continue

            output = response.output[0].content[0].text  # type: ignore[attr-defined]
            score = _parse_reward_score(output)
            return max(0.0, min(1.0, score))


class AzureOpenAIRewardModel(RewardModel):
    """LLM-based reward that targets Azure OpenAI deployments."""

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        max_retries: int = 2,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The openai package is required for AzureOpenAIRewardModel.") from exc

        endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise RuntimeError(
                "Azure endpoint missing. Provide via argument or AZURE_OPENAI_ENDPOINT environment variable."
            )
        deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise RuntimeError(
                "Azure deployment name missing. Provide via argument or AZURE_OPENAI_DEPLOYMENT environment variable."
            )
        key = api_key or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "Azure API key missing. Provide via argument or AZURE_OPENAI_API_KEY/OPENAI_API_KEY environment variable."
            )

        self.client = AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=key)
        self.deployment = deployment
        self.max_retries = max_retries

    system_prompt = (
        "You are a reward model that evaluates whether a rewritten sentence sounds friendlier. "
        "Respond with JSON: {\"score\": <number between 0 and 1>, \"explanation\": <short reason>}"
    )

    def score(self, original: str, rewritten: str) -> float:  # pragma: no cover - network call
        attempts = 0
        user_prompt = (
            "Original sentence: "
            + original.strip()
            + "\nRewritten sentence: "
            + rewritten.strip()
            + "\nRate how much friendlier the rewrite is compared to the original."
        )

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=200,
                    top_p=1.0,
                    n=1,
                )
            except Exception as exc:  # noqa: BLE001
                attempts += 1
                if attempts > self.max_retries:
                    raise RuntimeError("Failed to obtain reward from Azure OpenAI API") from exc
                continue

            message = response.choices[0].message
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
            if isinstance(content, list):
                text = " ".join(part.get("text", "") for part in content)
            else:
                text = str(content)
            score = _parse_reward_score(text)
            return max(0.0, min(1.0, score))
