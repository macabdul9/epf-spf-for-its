"""From a particle's per-step scores to its log-weight.

Two choices, both recorded with every result:

aggregation  which number summarises the steps scored so far
  last   the current step's score               (paper's algorithm block; the project's arm)
  prod   product of all step scores so far      (its_hub's documented ePF setup)
  min    the weakest step so far
  mean   the average step score so far

transform    how that number becomes a log-weight, divided by the weight temperature T
  log    log(s) / T     -> resampling probability proportional to s^(1/T)       (paper)
  logit  logit(s) / T   -> resampling probability proportional to odds^(1/T)    (its_hub)

At T = 1 these are exactly the paper's and its_hub's rules. See docs/weight-temperature.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

AGGREGATIONS = ("last", "prod", "min", "mean")
TRANSFORMS = ("log", "logit")

# A step with no usable score (a judge call that failed under --on-score-failure=neutral,
# or a probe that found no option-letter mass) counts as s = 0.5: logit(0.5) = 0, the
# neutral weight the previous implementation used.
NEUTRAL_SCORE = 0.5

_EPS = 1e-6


def aggregate(scores: Sequence[float | None], how: str) -> float:
    values = [NEUTRAL_SCORE if s is None else s for s in scores]
    if not values:
        raise ValueError("no scores to aggregate")
    if how == "last":
        return values[-1]
    if how == "prod":
        return math.prod(values)
    if how == "min":
        return min(values)
    if how == "mean":
        return sum(values) / len(values)
    raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {how!r}")


def log_weight(score: float, transform: str, temperature: float) -> float:
    """Scores are clamped to [1e-6, 1 - 1e-6] (logit) or [1e-6, 1] (log) so a judge that
    is certain cannot produce an infinite weight."""
    if transform == "logit":
        s = min(max(score, _EPS), 1 - _EPS)
        return math.log(s / (1 - s)) / temperature
    if transform == "log":
        return math.log(min(max(score, _EPS), 1.0)) / temperature
    raise ValueError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
