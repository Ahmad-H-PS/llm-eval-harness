import pytest

from harness.metrics import (
    exact_match,
    normalize_answer,
    score_predictions,
    score_qa,
    token_f1,
)


def test_perfect_predictions():
    rows = [{"gold": 1, "pred": 1}, {"gold": 0, "pred": 0}]
    s = score_predictions(rows)
    assert s["accuracy"] == 1.0
    assert s["macro_f1"] == 1.0
    assert s["unparseable_rate"] == 0.0


def test_unparseable_counts_as_wrong_not_dropped():
    rows = [
        {"gold": 1, "pred": 1},
        {"gold": 0, "pred": 0},
        {"gold": 1, "pred": None},   # flipped to 0, so wrong
        {"gold": 0, "pred": 0},
    ]
    s = score_predictions(rows)
    assert s["n"] == 4                                   # nothing dropped
    assert s["accuracy"] == 0.75
    assert s["macro_f1"] == pytest.approx(0.733, abs=1e-3)
    assert s["unparseable_rate"] == 0.25
    
def test_normalize_answer():
    assert normalize_answer("The Eiffel Tower.") == "eiffel tower"


def test_exact_match_uses_best_gold():
    assert exact_match("The Eiffel Tower.", ["Louvre", "eiffel tower"]) == 1.0
    assert exact_match("Paris", ["Louvre"]) == 0.0


def test_token_f1_partial_overlap():
    assert token_f1("the Eiffel Tower in Paris", "Eiffel Tower") == pytest.approx(0.667, abs=1e-3)


def test_score_qa_empty_output_counts_as_wrong():
    rows = [{"gold": ["a b"], "pred": "a b"}, {"gold": ["c"], "pred": ""}]
    s = score_qa(rows)
    assert s["n"] == 2 and s["exact_match"] == 0.5 and s["empty_rate"] == 0.5