from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SampledAction:
    index: int
    log_prob: float


class CategoricalPolicy:
    """Softmax policy updated via REINFORCE."""

    def __init__(
        self,
        num_actions: int,
        *,
        learning_rate: float = 0.1,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.logits = np.zeros(num_actions, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    @property
    def num_actions(self) -> int:
        return self.logits.shape[0]

    def probabilities(self) -> np.ndarray:
        scaled_logits = self.logits / max(1e-6, self.temperature)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / exp_logits.sum()
        return probs

    def sample(self) -> SampledAction:
        probs = self.probabilities()
        index = int(self.rng.choice(self.num_actions, p=probs))
        log_prob = float(np.log(probs[index] + 1e-12))
        return SampledAction(index=index, log_prob=log_prob)

    def best_action(self) -> int:
        probs = self.probabilities()
        return int(np.argmax(probs))

    def update(self, action_index: int, reward: float, baseline: float = 0.0) -> None:
        if action_index < 0 or action_index >= self.num_actions:
            raise IndexError("action_index out of range")
        advantage = reward - baseline
        if advantage == 0:
            return
        probs = self.probabilities()
        one_hot = np.zeros_like(probs)
        one_hot[action_index] = 1.0
        grad = one_hot - probs
        self.logits += self.learning_rate * advantage * grad

    def entropy(self) -> float:
        probs = self.probabilities()
        safe_probs = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(safe_probs * np.log(safe_probs)))

    def temperature_anneal(self, factor: float) -> None:
        if factor <= 0:
            raise ValueError("factor must be positive")
        self.temperature = max(0.05, self.temperature * factor)
