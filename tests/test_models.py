import pytest

from harness.models import parse_label, parse_label_ar


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("positive", 1),
        ("Negative\n", 0),
        ("I think it is positive", 1),
        ("not positive", None),
        ("great movie", None),
        ("", None),
        ("one two three four five six positive", None),
    ],
)
def test_parse_label(raw, expected):
    assert parse_label(raw) == expected


AR_POS = "\u0625\u064A\u062C\u0627\u0628\u064A"   # positive
AR_NEG = "\u0633\u0644\u0628\u064A"               # negative


@pytest.mark.parametrize(
    "raw, expected",
    [
        (AR_POS, 1),
        (AR_NEG, 0),
        ("\u0633\u064E\u0644\u0652\u0628\u0650\u064A", 0),            # with tashkeel
        ("\u0627\u0644\u0625\u062C\u0627\u0628\u0629: " +
         AR_NEG, 0),  # "answer: negative"
        # negated label
        ("\u063A\u064A\u0631 " + AR_POS, None),
        # English fallback
        ("Positive", 1),
        # no label word
        ("\u062C\u064A\u062F", None),
        ("", None),
    ],
)
def test_parse_label_ar(raw, expected):
    assert parse_label_ar(raw) == expected
