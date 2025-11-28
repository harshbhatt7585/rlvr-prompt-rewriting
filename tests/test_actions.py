from prompt_rewriting.actions import (
    DEFAULT_ACTIONS,
    RewriteAction,
    add_empathy,
    polite_prefix,
    replace_negative_tones,
    soften_imperative,
    supportive_suffix,
)


def get_action(name: str) -> RewriteAction:
    for action in DEFAULT_ACTIONS:
        if action.name == name:
            return action
    raise KeyError(name)


def test_polite_prefix_adds_please():
    result = polite_prefix("Send me the report")
    assert result.startswith("Please ")


def test_soften_imperative_turns_into_question():
    result = soften_imperative("Fix this issue")
    assert result.lower().startswith("could you please")
    assert result.endswith("?")


def test_supportive_suffix_adds_gratitude():
    result = supportive_suffix("You messed this up")
    assert "Thanks" in result


def test_replace_negative_words_softens_language():
    result = replace_negative_tones("This is terrible")
    assert "not ideal" in result.lower()


def test_empathy_action_adds_appreciation():
    result = add_empathy("Stop ignoring me")
    assert "appreciate" in result.lower()


def test_default_actions_apply_without_errors():
    sentence = "Send me the report now"
    for action in DEFAULT_ACTIONS:
        output = action.apply(sentence)
        assert isinstance(output, str)
        assert output
