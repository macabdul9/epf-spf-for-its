import asyncio
import math
import re

import numpy as np
import pytest

from epf_spf.config import build_config
from epf_spf.filter import ScoreFailure, run_filter
from epf_spf.records import MCQRecord
from epf_spf.scorers import StepScore
from epf_spf.steps import StepFormat

RECORD = MCQRecord("item-1", "Which?", ["x", "y", "z", "w"], "x", [], "sound", "le10s", 0)


class ScriptedGenerator:
    """Returns `script(steps_so_far, call_index)` as each particle's next step."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    async def next_steps(self, base_messages, steps_per_particle, fmt):
        out = []
        for steps in steps_per_particle:
            out.append(self.script(steps, self.calls))
            self.calls += 1
        return out


class TextScorer:
    """Score = the q=<value> written in the particle's latest step."""

    def __init__(self, fail=False):
        self.fail = fail
        self.scored_per_round = []

    async def score(self, record, base_messages, particles):
        self.scored_per_round.append(len(particles))
        if self.fail:
            return [StepScore(score=None, failed=True) for _ in particles]
        return [
            StepScore(score=float(re.findall(r"q=([0-9.]+)", steps[-1])[-1]))
            for steps, _ in particles
        ]


def run(config, generator, scorer, budget=4, max_steps=10):
    return asyncio.run(
        run_filter(
            RECORD,
            budget,
            config,
            generator,
            scorer,
            StepFormat(max_steps=max_steps),
            [],
            np.random.default_rng(0),
        )
    )


def three_steps(steps, _):
    return ["Step 1: q=0.9", " q=0.8", " q=0.7 Answer: A"][len(steps)]


def test_paper_variant_does_not_resample_at_step_one():
    result = run(build_config("epf-paper"), ScriptedGenerator(three_steps), TextScorer())
    assert result.trace[0]["resample"]["applied"] is False
    assert result.trace[1]["resample"]["applied"] is True


def test_itshub_variant_resamples_from_step_one():
    result = run(build_config("epf-itshub"), ScriptedGenerator(three_steps), TextScorer())
    assert result.trace[0]["resample"]["applied"] is True


def test_spf_gate_holds_back_resampling_when_scores_are_flat():
    flat = ScriptedGenerator(lambda steps, _: ["Step 1: q=0.5", " q=0.5 Answer: A"][len(steps)])
    result = run(build_config("spf"), flat, TextScorer())
    assert all(r["resample"]["applied"] is False for r in result.trace)
    assert {r["resample"]["reason"] for r in result.trace} == {"ESS gate"}


def test_spf_gate_resamples_when_one_particle_dominates():
    first = iter(["0.999999", "0.000001", "0.000001", "0.000001"])
    skewed = ScriptedGenerator(
        lambda steps, _: f"Step 1: q={next(first)}" if not steps else " q=0.5 Answer: A"
    )
    result = run(build_config("spf", weight_temperature=1.0), skewed, TextScorer())
    assert result.trace[0]["resample"]["applied"] is True


def test_a_particle_can_take_max_steps_plus_one_steps():
    endless = ScriptedGenerator(lambda steps, _: "Step 1: q=0.5" if not steps else " q=0.5")
    result = run(build_config("epf-project"), endless, TextScorer(), budget=2, max_steps=3)
    assert [len(p.steps) for p in result.particles] == [4, 4]


def test_finished_particles_are_not_rescored():
    # Particle 0 finishes at step 1; the others take three steps.
    def script(steps, call):
        if not steps:
            return "Step 1: q=0.9 Answer: A" if call == 0 else "Step 1: q=0.6"
        return [" q=0.6", " q=0.6 Answer: B"][len(steps) - 1]

    scorer = TextScorer()
    config = build_config("epf-project", resample_from_step=99)  # keep particle identities
    result = run(config, ScriptedGenerator(script), scorer)
    assert scorer.scored_per_round == [4, 3, 3]
    assert result.particles[0].scores == [0.9]


def test_score_failure_raises_by_default():
    with pytest.raises(ScoreFailure):
        run(build_config("epf-project"), ScriptedGenerator(three_steps), TextScorer(fail=True))


def test_score_failure_counts_as_neutral_when_allowed():
    config = build_config("epf-project", on_score_failure="neutral")
    result = run(config, ScriptedGenerator(three_steps), TextScorer(fail=True))
    assert result.score_failures == 4 * 3
    assert all(p.log_weight == pytest.approx(0.0) for p in result.particles)


def test_weight_uses_the_configured_aggregation_and_transform():
    config = build_config("epf-itshub", resample_from_step=99)  # prod, logit, T = 1
    result = run(config, ScriptedGenerator(three_steps), TextScorer())
    expected = math.log(0.9 * 0.8 * 0.7 / (1 - 0.9 * 0.8 * 0.7))
    assert all(p.log_weight == pytest.approx(expected) for p in result.particles)
