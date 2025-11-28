import sys
import types

import pytest

from prompt_rewriting.reward import (
    AzureOpenAIRewardModel,
    HeuristicRewardModel,
    _parse_reward_score,
)


def test_heuristic_reward_prefers_friendlier_text():
    model = HeuristicRewardModel()
    original = "Send me that file now."
    friendly = "Please send me that file when you have a moment. Thanks for your help!"
    unfriendly = "Send me that file immediately."

    friendly_score = model.score(original, friendly)
    unfriendly_score = model.score(original, unfriendly)

    assert friendly_score > 5 * unfriendly_score
    assert friendly_score > 0.1


def test_heuristic_reward_penalises_no_improvement():
    model = HeuristicRewardModel()
    sentence = "You messed this up."  # identical rewrite
    reward = model.score(sentence, sentence)
    assert 0.0 <= reward <= 0.1


def test_parse_reward_score_handles_json_and_numbers():
    assert _parse_reward_score('{"score": 0.8, "explanation": "good"}') == 0.8
    assert _parse_reward_score("Rewritten score: 0.42 (friendlier)") == 0.42
    assert _parse_reward_score("no score here") == 0.0


def _install_openai_stub(monkeypatch, client_cls):
    module = types.SimpleNamespace(AzureOpenAI=client_cls)
    monkeypatch.setitem(sys.modules, "openai", module)


def test_azure_reward_requires_endpoint(monkeypatch):
    class DummyClient:
        def __init__(self, **_kwargs):
            pass

    _install_openai_stub(monkeypatch, DummyClient)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        AzureOpenAIRewardModel(deployment="my-deploy", api_key="key")


def test_azure_reward_uses_env_defaults(monkeypatch):
    captured_kwargs = {}

    class DummyClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    _install_openai_stub(monkeypatch, DummyClient)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example-endpoint/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "friendly-model")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")

    model = AzureOpenAIRewardModel()
    assert model.deployment == "friendly-model"
    assert captured_kwargs["azure_endpoint"] == "https://example-endpoint/"
    assert captured_kwargs["api_key"] == "secret"
