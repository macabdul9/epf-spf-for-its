import numpy as np
import pytest

from epf_spf.resampling import (
    annealing_temperature,
    ess_ratio,
    gate_allows_resampling,
    resampling_distribution,
    softmax,
    systematic_indices,
)


def test_ess_ratio_is_one_for_uniform_and_one_over_n_for_a_spike():
    assert ess_ratio(np.full(8, 1 / 8)) == pytest.approx(1.0)
    spike = np.zeros(8)
    spike[0] = 1.0
    assert ess_ratio(spike) == pytest.approx(1 / 8, rel=1e-4)


def test_annealing_follows_the_epf_formula():
    log_weights = np.array([5.0, 0.0, 0.0, 0.0])
    p = softmax(log_weights)
    ratio = ess_ratio(p)
    assert ratio < 0.5
    expected = max(1.0, (1 / ratio) * (1 - 2 / 10))
    assert annealing_temperature(p, step=2, horizon=10, threshold=0.5, window=1.0) == pytest.approx(
        expected
    )


def test_annealing_does_not_fire_above_the_threshold():
    p = softmax([0.1, 0.0, 0.0, 0.0])
    assert annealing_temperature(p, step=1, horizon=10, threshold=0.5, window=1.0) == 1.0


def test_annealing_stops_outside_the_window():
    p = softmax([5.0, 0.0, 0.0, 0.0])
    assert annealing_temperature(p, step=4, horizon=10, threshold=0.6, window=0.7) > 1.0
    assert annealing_temperature(p, step=7, horizon=10, threshold=0.6, window=0.7) == 1.0
    assert annealing_temperature(p, step=5, horizon=10, threshold=0.5, window=0.5) == 1.0


def test_annealed_distribution_is_flatter():
    log_weights = [5.0, 0.0, 0.0, 0.0]
    raw, _ = resampling_distribution(log_weights, 1, 10, False, 0.6, 0.7)
    annealed, temperature = resampling_distribution(log_weights, 1, 10, True, 0.6, 0.7)
    assert temperature > 1.0
    assert ess_ratio(annealed) > ess_ratio(raw)


def test_gate_blocks_resampling_when_weights_are_spread():
    assert not gate_allows_resampling(np.full(16, 1 / 16), 0.5)
    spike = softmax([8.0] + [0.0] * 15)
    assert gate_allows_resampling(spike, 0.5)


def test_systematic_resampling_gives_each_particle_floor_or_ceil_of_n_p():
    rng = np.random.default_rng(0)
    p = np.array([0.5, 0.25, 0.125, 0.125])
    for _ in range(50):
        counts = np.bincount(systematic_indices(p, rng), minlength=4)
        assert np.all(counts >= np.floor(4 * p)) and np.all(counts <= np.ceil(4 * p))


def test_systematic_resampling_is_reproducible_from_the_seed():
    p = softmax(np.arange(8.0))
    assert systematic_indices(p, np.random.default_rng(7)) == systematic_indices(
        p, np.random.default_rng(7)
    )
