import pytest

from harness.metrics import score_predictions


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