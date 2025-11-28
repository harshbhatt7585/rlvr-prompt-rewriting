"""Convenience script to train the prompt rewriting policy without the CLI.

Edit the CONFIG dictionary to switch datasets, reward backends, or training
hyperparameters. By default, the script expects the heuristic reward.

For Azure OpenAI, set CONFIG["reward_backend"] = "azure-openai" and provide the
endpoint, deployment, and API key values via the config itself or environment
variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_KEY).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prompt_rewriting import (  # noqa: E402
    AzureOpenAIRewardModel,
    CategoricalPolicy,
    DEFAULT_ACTIONS,
    PromptRewriteEnv,
    train,
)
from dotenv import load_dotenv

load_dotenv('.env')

print("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT"))


DATA_DIR = ROOT_DIR / "data"
DEFAULT_DATASET = DATA_DIR / "rude_sentences.json"

CONFIG = {
    "dataset": DEFAULT_DATASET,
    "episodes": 400,
    "seed": 13,
    "reward_backend": "azure-openai",  # "heuristic", "openai", or "azure-openai"
    "openai_model": "gpt-4.1",
    "azure": {
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "deployment": "gpt-4",
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_version": "2024-12-01-preview",
    },
    "policy": {
        "learning_rate": 0.1,
        "temperature": 1.0,
    },
    "training": {
        "baseline_momentum": 0.9,
        "log_every": 50,
        "temperature_anneal": 0.98,
    },
}


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _load_sentences(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Sentence dataset not found: {path}")
    if path.suffix.lower() == ".json":
        import json

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "sentences" in data:
            sentences = data["sentences"]
        else:
            sentences = data
        if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
            raise ValueError("JSON dataset must be a list of strings or {\"sentences\": [...]}")
        return sentences
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _select_reward() -> AzureOpenAIRewardModel:
    backend = CONFIG["reward_backend"].lower()
    azure_cfg = CONFIG["azure"]
    return AzureOpenAIRewardModel(
        endpoint=azure_cfg.get("endpoint"),
        deployment=azure_cfg.get("deployment"),
        api_key=azure_cfg.get("api_key"),
        api_version=azure_cfg.get("api_version", "2024-12-01-preview"),
    )


def run_training() -> None:
    _load_env_file(ROOT_DIR / ".env")
    sentences = _load_sentences(Path(CONFIG["dataset"]))
    reward_model = _select_reward()

    policy_cfg = CONFIG["policy"]
    policy = CategoricalPolicy(
        len(DEFAULT_ACTIONS),
        seed=CONFIG["seed"],
        learning_rate=policy_cfg.get("learning_rate", 0.1),
        temperature=policy_cfg.get("temperature", 1.0),
    )
    env = PromptRewriteEnv(sentences, reward_model, seed=CONFIG["seed"], actions=DEFAULT_ACTIONS)

    training_cfg = CONFIG["training"]
    stats = train(
        env,
        policy,
        episodes=CONFIG["episodes"],
        baseline_momentum=training_cfg.get("baseline_momentum", 0.9),
        log_every=training_cfg.get("log_every", 0),
        temperature_anneal=training_cfg.get("temperature_anneal"),
    )

    best_action = DEFAULT_ACTIONS[policy.best_action()]
    print(f"Best action learned: {best_action.name}\n")

    for entry in stats.history[-10:]:
        print(
            f"Episode {entry.episode:4d}: reward={entry.reward:.3f} action={entry.action} baseline={entry.baseline:.3f}"
        )

    print("\nSample rewrites:\n")
    for sentence in sentences:
        rewrite = best_action.apply(sentence)
        print("Original:", sentence)
        print("Rewrite:", rewrite)
        print()


if __name__ == "__main__":
    run_training()
