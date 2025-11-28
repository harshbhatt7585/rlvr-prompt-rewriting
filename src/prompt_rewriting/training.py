from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .environment import PromptRewriteEnv
from .policy import CategoricalPolicy


@dataclass
class EpisodeLog:
    episode: int
    reward: float
    action: str
    original: str
    rewritten: str
    baseline: float


@dataclass
class TrainingStats:
    history: List[EpisodeLog]
    baseline: float


def train(
    env: PromptRewriteEnv,
    policy: CategoricalPolicy,
    *,
    episodes: int = 400,
    baseline_momentum: float = 0.9,
    log_every: int = 50,
    temperature_anneal: Optional[float] = 0.98,
) -> TrainingStats:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    baseline = 0.0
    history: List[EpisodeLog] = []

    for episode in range(1, episodes + 1):
        sampled = policy.sample()
        original, step = env.sample_episode(sampled.index)
        reward = step.reward
        policy.update(sampled.index, reward, baseline)
        baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * reward

        history.append(
            EpisodeLog(
                episode=episode,
                reward=reward,
                action=step.info["action"],
                original=original,
                rewritten=step.rewritten,
                baseline=baseline,
            )
        )


        print("Episode", episode, "reward", reward, "baseline", baseline)
        print("Step", step.info["action"], "original", original, "rewritten", step.rewritten)
        print("Action", step.info["action"], "original", original, "rewritten", step.rewritten)
        print("Action", step.info["action"], "original", original, "rewritten", step.rewritten)
        if temperature_anneal and episode % max(1, log_every) == 0:
            policy.temperature_anneal(temperature_anneal)

        if log_every and episode % log_every == 0:
            print(
                f"Episode {episode}: reward={reward:.3f}, action={step.info['action']}, baseline={baseline:.3f}"
            )

    return TrainingStats(history=history, baseline=baseline)


def generate_rewrites(
    policy: CategoricalPolicy,
    sentences: Iterable[str],
    *,
    actions=None,
) -> List[str]:
    outputs: List[str] = []
    action_idx = policy.best_action()
    if actions is None:
        from .actions import DEFAULT_ACTIONS  # Local import to avoid circular deps

        action = DEFAULT_ACTIONS[action_idx]
    else:
        action = list(actions)[action_idx]

    for sentence in sentences:
        outputs.append(action.apply(sentence))
    return outputs
