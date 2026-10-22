from harness.data import clean_text


def test_clean_text_removes_noise():
    raw = "Great movie!!\u200b\n\n  Visit https://example.com <br> now"
    assert clean_text(raw) == "Great movie!! Visit now"


def test_clean_text_detokenize_sst2_style():
    raw = "  it 's a charming   and often affecting journey .  "
    assert clean_text(raw, detokenize=True) == "it's a charming and often affecting journey."


def test_clean_text_detokenize_brackets_and_contractions():
    raw = "this does n't work at all , -lrb- really -rrb- ."
    assert clean_text(raw, detokenize=True) == "this doesn't work at all, (really)."


def test_clean_text_leaves_spacing_alone_without_detokenize():
    assert clean_text("a  b .") == "a b ."