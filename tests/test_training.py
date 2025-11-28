from prompt_rewriting.actions import DEFAULT_ACTIONS
from prompt_rewriting.environment import PromptRewriteEnv
from prompt_rewriting.policy import CategoricalPolicy
from prompt_rewriting.reward import HeuristicRewardModel
from prompt_rewriting.training import train


def test_training_prefers_friendliest_action():
    sentences = [
        "Send me the report now.",
        "Fix this problem immediately.",
        "Stop ignoring my emails.",
    ]
    model = HeuristicRewardModel()
    env = PromptRewriteEnv(sentences, model, seed=42)
    policy = CategoricalPolicy(env.num_actions, seed=42, learning_rate=0.25)

    train(
        env,
        policy,
        episodes=200,
        baseline_momentum=0.8,
        log_every=0,
        temperature_anneal=None,
    )

    expected_best = max(
        range(len(DEFAULT_ACTIONS)),
        key=lambda idx: sum(
            model.score(sentence, DEFAULT_ACTIONS[idx].apply(sentence)) for sentence in sentences
        ),
    )
    assert policy.best_action() == expected_best
