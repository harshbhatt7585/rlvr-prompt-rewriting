from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .actions import DEFAULT_ACTIONS
from .environment import PromptRewriteEnv
from .policy import CategoricalPolicy
from .reward import AzureOpenAIRewardModel, HeuristicRewardModel, OpenAIRewardModel
from .training import generate_rewrites, train


def _load_sentences(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Sentence dataset not found: {path}")
    if path.suffix.lower() == ".json":
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a prompt rewriting policy with an LLM reward")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/rude_sentences.json"),
        help="Path to a JSON file with sentences to soften.",
    )
    parser.add_argument("--episodes", type=int, default=400, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument(
        "--reward-model",
        choices=["heuristic", "openai", "azure-openai"],
        default="heuristic",
        help="Which reward model to use.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="Model to use when reward-model=openai",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=None,
        help="Azure OpenAI endpoint URL (e.g. https://<resource>.cognitiveservices.azure.com/)",
    )
    parser.add_argument(
        "--azure-deployment",
        default=None,
        help="Azure OpenAI deployment name to query.",
    )
    parser.add_argument(
        "--azure-api-version",
        default="2024-12-01-preview",
        help="Azure OpenAI API version.",
    )
    parser.add_argument(
        "--azure-api-key",
        default=None,
        help="Azure OpenAI API key (falls back to AZURE_OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--baseline-momentum",
        type=float,
        default=0.9,
        help="Exponential moving average factor for the REINFORCE baseline",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="How frequently to print progress",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    sentences = _load_sentences(args.data)

    if args.reward_model == "heuristic":
        reward_model = HeuristicRewardModel()
    elif args.reward_model == "openai":
        reward_model = OpenAIRewardModel(model=args.openai_model)
    else:
        reward_model = AzureOpenAIRewardModel(
            endpoint=args.azure_endpoint,
            deployment=args.azure_deployment,
            api_key=args.azure_api_key,
            api_version=args.azure_api_version,
        )

    env = PromptRewriteEnv(sentences, reward_model, seed=args.seed)
    policy = CategoricalPolicy(env.num_actions, seed=args.seed)

    train(
        env,
        policy,
        episodes=args.episodes,
        baseline_momentum=args.baseline_momentum,
        log_every=args.log_every,
    )

    best_action = DEFAULT_ACTIONS[policy.best_action()]
    print(f"Best action learned: {best_action.name}\n")

    friendly_sentences = generate_rewrites(policy, sentences, actions=DEFAULT_ACTIONS)
    for original, rewrite in zip(sentences, friendly_sentences):
        print("Original:  ", original)
        print("Rewrite:   ", rewrite)
        print()


if __name__ == "__main__":
    main()
