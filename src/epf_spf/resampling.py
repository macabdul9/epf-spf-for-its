"""Resampling distribution, entropic annealing, the SPF gate, and the draws.

The annealing schedule is ePF's:

    p        = softmax(w)
    ESS_n    = 1 / (N * sum(p^2))
    if ESS_n < threshold and t / horizon < window:
        beta^-1 = max(1, (1 - t/horizon) / ESS_n)
        p        = softmax(w * beta)

`threshold` and `window` are the only things the variants change (paper and its_hub
0.5 and 0.5, the project and SPF 0.6 and 0.7).
"""

from __future__ import annotations

import numpy as np


def softmax(log_weights) -> np.ndarray:
    x = np.asarray(log_weights, dtype=float)
    e = np.exp(x - np.max(x))
    return e / e.sum()


def ess_ratio(probabilities, clip: float = 1e-7) -> float:
    """Normalised effective sample size, 1/(N * sum(p^2)), in (0, 1].

    The 1e-7 clip matches its_hub's annealing computation; the SPF gate uses 1e-12,
    as the previous implementation did.
    """
    p = np.clip(np.asarray(probabilities, dtype=float), clip, 1 - clip)
    return float(1.0 / np.sum(p**2) / len(p))


def annealing_temperature(
    probabilities, step: int, horizon: int, threshold: float, window: float
) -> float:
    """beta^-1 for this step; 1.0 means no flattening."""
    n = len(probabilities)
    if n <= 1:
        return 1.0
    progress = step / horizon
    ratio = ess_ratio(probabilities)
    if ratio < threshold and progress < window:
        if ratio <= 0:
            return 1.0
        return max(1.0, (1.0 / ratio) * (1 - progress))
    return 1.0


def resampling_distribution(
    log_weights, step: int, horizon: int, annealing: bool, threshold: float, window: float
) -> tuple[np.ndarray, float]:
    """The distribution resampling draws from, and the annealing temperature applied."""
    probabilities = softmax(log_weights)
    if not annealing:
        return probabilities, 1.0
    temperature = annealing_temperature(probabilities, step, horizon, threshold, window)
    if temperature == 1.0:
        return probabilities, 1.0
    return softmax(np.asarray(log_weights, dtype=float) / temperature), temperature


def gate_allows_resampling(probabilities, gate_threshold: float) -> bool:
    """SPF: resample only while the (already annealed) distribution has ESS_n below the gate."""
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1)
    return float(1.0 / np.sum(p**2)) / len(p) < gate_threshold


def systematic_indices(probabilities, rng: np.random.Generator) -> list[int]:
    """One uniform draw, N evenly spaced positions along the cumulative distribution."""
    n = len(probabilities)
    positions = (np.arange(n) + rng.uniform(0, 1)) / n
    cumulative = np.cumsum(probabilities)
    # Floating-point cumsum can end just below 1, which would walk past the last index.
    cumulative[-1] = max(cumulative[-1], 1.0)
    indices = []
    j = 0
    for position in positions:
        while position >= cumulative[j]:
            j += 1
        indices.append(j)
    return indices


def multinomial_indices(probabilities, rng: np.random.Generator) -> list[int]:
    n = len(probabilities)
    return [
        int(i) for i in rng.choice(n, size=n, p=np.asarray(probabilities) / np.sum(probabilities))
    ]
