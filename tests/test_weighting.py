import math

import pytest

from epf_spf.weighting import NEUTRAL_SCORE, aggregate, log_weight


def test_log_transform_gives_weight_proportional_to_score():
    # softmax(log s) == s / sum(s): the paper's rule.
    scores = [0.6, 0.9]
    weights = [math.exp(log_weight(s, "log", 1.0)) for s in scores]
    assert weights[1] / weights[0] == pytest.approx(0.9 / 0.6)


def test_logit_transform_gives_weight_proportional_to_odds():
    scores = [0.6, 0.9]
    weights = [math.exp(log_weight(s, "logit", 1.0)) for s in scores]
    assert weights[1] / weights[0] == pytest.approx((0.9 / 0.1) / (0.6 / 0.4))


def test_logit_and_log_differ_by_minus_log_one_minus_s():
    s = 0.83
    assert log_weight(s, "logit", 1.0) - log_weight(s, "log", 1.0) == pytest.approx(
        -math.log(1 - s)
    )


@pytest.mark.parametrize("transform", ["log", "logit"])
def test_temperature_divides_the_log_weight(transform):
    assert log_weight(0.9, transform, 3.0) == pytest.approx(log_weight(0.9, transform, 1.0) / 3)


def test_certain_scores_are_clamped_to_finite_weights():
    assert math.isfinite(log_weight(1.0, "logit", 1.0))
    assert math.isfinite(log_weight(0.0, "logit", 1.0))
    assert math.isfinite(log_weight(0.0, "log", 1.0))
    assert log_weight(1.0, "log", 1.0) == 0.0


def test_neutral_score_has_zero_logit_weight():
    assert log_weight(NEUTRAL_SCORE, "logit", 3.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "how, expected",
    [("last", 0.5), ("prod", 0.9 * 0.6 * 0.5), ("min", 0.5), ("mean", (0.9 + 0.6 + 0.5) / 3)],
)
def test_aggregation(how, expected):
    assert aggregate([0.9, 0.6, 0.5], how) == pytest.approx(expected)


def test_missing_scores_count_as_neutral():
    assert aggregate([0.9, None], "prod") == pytest.approx(0.9 * NEUTRAL_SCORE)
    assert aggregate([None], "last") == NEUTRAL_SCORE


def test_unknown_options_are_rejected():
    with pytest.raises(ValueError):
        aggregate([0.5], "median")
    with pytest.raises(ValueError):
        log_weight(0.5, "probit", 1.0)
