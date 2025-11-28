from prompt_rewriting.rlvr_utils import extract_generated_responses


class DummyTokenizer:
    def batch_decode(self, sequences, *, skip_special_tokens=True):  # noqa: ARG003 - keep signature aligned
        # Join integers as characters to keep ordering predictable.
        outputs = []
        for seq in sequences:
            outputs.append("".join(str(token) for token in seq))
        return outputs


def test_extract_generated_responses_removes_prompt_prefix():
    tokenizer = DummyTokenizer()
    prompts = [[0, 0, 1, 2]]  # Left padded prompt tokens
    responses = [[0, 0, 1, 2, 3, 4]]  # TRL returns prompt + continuation

    rewrites = extract_generated_responses(tokenizer, prompts, responses)

    assert rewrites == ["34"]


def test_extract_generated_responses_handles_non_matching_prefix():
    tokenizer = DummyTokenizer()
    prompts = [[0, 5, 6], [1, 2]]
    responses = [[0, 5, 6, 7, 8], [3, 4]]

    rewrites = extract_generated_responses(tokenizer, prompts, responses)

    assert rewrites == ["78", "34"]
