import pytest

from harness.tokenization import token_stats


class FakeTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text)}  # one token per character


def test_token_stats_counts():
    s = token_stats(FakeTok(), ["ab cd", "efgh"])
    assert s["n"] == 2
    assert s["tokens_per_text"] == 4.5          # (5 + 4) / 2
    assert s["tokens_per_word"] == 3.0          # 9 tokens / 3 words
    assert s["chars_per_token"] == pytest.approx(1.0)