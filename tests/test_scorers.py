import asyncio

from epf_spf.config import build_config
from epf_spf.records import MCQRecord
from epf_spf.scorers import RandomScorer

RECORD = MCQRecord("item-1", "Which?", ["x", "y"], "x", [], "sound", "le10s", 0)
PARTICLES = [(["Step 1: a"], False), (["Step 1: b", " c"], False), (["Step 1: d"], True)]


def scores(seed, record=RECORD):
    return [s.score for s in asyncio.run(RandomScorer(seed).score(record, [], PARTICLES))]


def test_random_scores_are_uniform_draws_reproducible_from_the_seed():
    first = scores(7)
    assert all(0 <= s < 1 for s in first)
    assert scores(7) == first
    assert scores(8) != first


def test_random_scores_differ_between_items():
    other = MCQRecord("item-2", "Which?", ["x", "y"], "x", [], "sound", "le10s", 0)
    assert scores(7, other) != scores(7)


def test_random_scorer_is_a_valid_config_choice():
    assert build_config("epf-paper", scorer="random").scorer == "random"
