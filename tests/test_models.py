import pytest

from harness.models import parse_label


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("positive", 1),
        ("Negative\n", 0),
        ("I think it is positive", 1),
        ("not positive", None),                              # negated label is unparseable
        ("great movie", None),                               # no label word
        ("", None),
        ("one two three four five six positive", None),      # label beyond first 6 words
    ],
)
def test_parse_label(raw, expected):
    assert parse_label(raw) == expected