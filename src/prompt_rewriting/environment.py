from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .actions import DEFAULT_ACTIONS, RewriteAction
from .reward import RewardModel


@dataclass
class StepResult:
    rewritten: str
    reward: float
    done: bool
    info: dict


class PromptRewriteEnv:
    """Single-step environment that rewards friendlier rewrites."""

    def __init__(
        self,
        sentences: Iterable[str],
        reward_model: RewardModel,
        *,
        actions: Optional[Sequence[RewriteAction]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.sentences: List[str] = [s for s in sentences if s.strip()]
        if not self.sentences:
            raise ValueError("The environment requires at least one non-empty sentence.")
        self.reward_model = reward_model
        self.actions = list(actions or DEFAULT_ACTIONS)
        if not self.actions:
            raise ValueError("At least one rewrite action must be provided.")
        self.rng = np.random.default_rng(seed)
        self._current_sentence: Optional[str] = None

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def reset(self) -> str:
        self._current_sentence = self.rng.choice(self.sentences)
        return self._current_sentence

    def step(self, action_index: int) -> StepResult:
        if self._current_sentence is None:
            raise RuntimeError("Call reset() before step().")
        if action_index < 0 or action_index >= len(self.actions):
            raise IndexError(f"Action index {action_index} is out of range.")

        action = self.actions[action_index]
        rewritten = action.apply(self._current_sentence)
        reward = float(self.reward_model.score(self._current_sentence, rewritten))
        info = {
            "action": action.name,
            "original": self._current_sentence,
            "rewritten": rewritten,
        }
        self._current_sentence = None
        return StepResult(rewritten=rewritten, reward=reward, done=True, info=info)

    def sample_episode(self, action_index: int) -> Tuple[str, StepResult]:
        """Convenience helper for REINFORCE style updates."""

        original = self.reset()
        result = self.step(action_index)
        return original, result
