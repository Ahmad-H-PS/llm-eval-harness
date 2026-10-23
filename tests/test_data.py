import pytest

from harness.data import clean_text, normalize_arabic


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
    
def test_arabic_diacritics_and_tatweel_removed():
    assert normalize_arabic("\u0643\u064E\u062A\u064E\u0628\u064E") == "\u0643\u062A\u0628"   # kataba
    assert normalize_arabic("\u0645\u0640\u0640\u0640\u062F") == "\u0645\u062F"                # tatweel


def test_arabic_alef_variants_normalized():
    raw = "\u0623\u062D\u0645\u062F \u0625\u0628\u0631\u0627\u0647\u064A\u0645 \u0622\u0645\u0646"
    assert normalize_arabic(raw) == "\u0627\u062D\u0645\u062F \u0627\u0628\u0631\u0627\u0647\u064A\u0645 \u0627\u0645\u0646"


def test_arabic_ya_and_ta_marbuta_are_opt_in():
    ya, tm = "\u0639\u0644\u0649", "\u0645\u062F\u0631\u0633\u0629"
    assert normalize_arabic(ya + " " + tm) == ya + " " + tm            # default leaves both
    assert normalize_arabic(ya, steps=("ya",)) == "\u0639\u0644\u064A"
    assert normalize_arabic(tm, steps=("ta_marbuta",)) == "\u0645\u062F\u0631\u0633\u0647"


def test_arabic_unknown_step_raises():
    with pytest.raises(ValueError):
        normalize_arabic("\u0627", steps=("typo",))


def test_clean_text_arabic_only_when_language_ar():
    raw = "  \u0643\u064E\u062A\u064E\u0628\u064E   \u0623\u062D\u0645\u062F "
    assert clean_text(raw, language="ar") == "\u0643\u062A\u0628 \u0627\u062D\u0645\u062F"
    assert clean_text("\u0623", language="en") == "\u0623"                 # en leaves Arabic alone