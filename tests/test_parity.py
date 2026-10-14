"""Behavioural parity with the previous implementation (entropic-particle-filter/).

Each test feeds the same input to the new code and the old code and requires the same
output. Skipped when the old code or the data is not on this machine.
"""

import asyncio
import itertools
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from epf_spf import prompts
from epf_spf.answers import predicted_index
from epf_spf.data import BENCHMARKS, load_benchmark
from epf_spf.records import MCQRecord
from epf_spf.resampling import annealing_temperature, softmax, systematic_indices
from epf_spf.scorers import JudgeScorer, ProbeScorer, letter_masses, yes_probability
from epf_spf.selection import select_answers
from epf_spf.steps import StepFormat
from epf_spf.weighting import log_weight

pytestmark = pytest.mark.legacy

PROJECT = Path(__file__).resolve().parents[2]
RECORD = MCQRecord(
    "id-1", "Which instrument plays first?", ["Piano", "Violin", "Drums", "Flute"],
    "Piano", ["/tmp/a.wav"], "music", "le10s", 0,
)  # fmt: skip
STEP_SETS = [
    ["Step 1: The audio opens with a piano."],
    ["Step 1: A piano.", " Then strings enter.", " Drums last.\nAnswer: A"],
    ["Step 1: Speech.", "  trailing spaces   ", " Answer: C"],
]


def old_header_steps(legacy):
    return legacy["probe_epf"].build_arm(
        "judge_epf_prm", 0.8, 10, 4, seed_step_headers=True, header_prefix="Step"
    )


def test_generator_prompt_is_identical(legacy, monkeypatch):
    monkeypatch.delenv("EPF_AUDIO_UUID", raising=False)
    old_messages, _ = legacy["prompt"].build(12, RECORD, audio_mode="local-path")
    assert prompts.generator_messages(RECORD) == [m.to_dict() for m in old_messages]
    old_trigger = legacy["prompt"].add_think_trigger(old_messages, RECORD)
    assert prompts.generator_messages(RECORD, think_trigger=True) == [
        m.to_dict() for m in old_trigger
    ]


def test_judge_templates_are_identical(legacy):
    old = legacy["judge_client"]
    assert prompts.JUDGE_STEP_TEMPLATE == old.JUDGE_PROMPT_TEMPLATE_STEP
    assert prompts.JUDGE_OUTCOME_TEMPLATE == old.JUDGE_PROMPT_TEMPLATE


def test_judge_sees_the_same_prompt_at_every_step(legacy):
    old = legacy["judge_client"]
    old_arm = old_header_steps(legacy)
    question = f"Question: {RECORD.question}\n\nOptions:\n{old.format_choices(RECORD.choices)}"
    scorer = JudgeScorer("http://unused/v1", "judge", StepFormat())
    Particle = legacy["particle_filtering"].Particle
    for steps, finished in itertools.product(STEP_SETS, [False, True]):
        item = old_arm._judge_item(
            Particle(steps=list(steps), is_stopped=finished, partial_log_weights=[])
        )
        if isinstance(item, tuple):
            expected = old.JUDGE_PROMPT_TEMPLATE_STEP.format(
                question=question,
                context=item[0] or "(none — this is the first step)",
                latest_step=item[1],
            )
        else:
            expected = old.JUDGE_PROMPT_TEMPLATE.format(question=question, candidate=item)
        assert scorer.prompt_for(RECORD, list(steps), finished) == expected


def test_step_rendering_and_stop_rule_are_identical(legacy):
    old_sg = old_header_steps(legacy).sg
    fmt = StepFormat(max_steps=10)
    for steps in STEP_SETS:
        assert fmt.continuation(steps) == old_sg._post_process(steps, stopped=None)
        assert fmt.trajectory(steps) == old_sg._post_process(steps, stopped=False)
        assert fmt.trajectory(steps) == old_sg._post_process(steps, stopped=True)
    for before, new in [([], "x"), (["a"] * 9, "x"), (["a"] * 10, "x"), (["a"], "Answer: B")]:
        old_final = len(before) >= old_sg.max_steps or old_sg._step_is_final(new, before)
        assert fmt.is_final(before, new) == old_final


def test_probe_sends_the_same_prefix(legacy):
    old_arm = legacy["probe_epf"].build_arm(
        "probe_adaptive_t3", 0.8, 10, 4, seed_step_headers=True, header_prefix="Step"
    )
    captured = []

    class FakeLM:
        async def agenerate(self, messages_lst, **kwargs):
            captured.extend(m[-1]["content"] for m in messages_lst)
            return [{"_logprobs": None} for _ in messages_lst]

    Particle = legacy["particle_filtering"].Particle
    particles = [
        Particle(steps=list(s), is_stopped=False, partial_log_weights=[]) for s in STEP_SETS
    ]
    asyncio.run(old_arm._probe(FakeLM(), particles, []))
    scorer = ProbeScorer(generator=None, fmt=StepFormat())
    assert [scorer.prefix_for(list(s)) for s in STEP_SETS] == captured


def test_yes_probability_is_identical(legacy):
    rng = random.Random(0)
    tokens = ["Yes", " yes", "No", "no.", '"Yes', "Maybe", "The", " NO"]
    for _ in range(300):
        top = [(rng.choice(tokens), -rng.random() * 6) for _ in range(rng.randint(0, 20))]
        old_entries = [SimpleNamespace(token=t, logprob=lp) for t, lp in top]
        assert yes_probability(top) == pytest.approx(
            legacy["judge_client"].yes_probability(old_entries)
        )


def test_letter_masses_are_identical(legacy):
    rng = random.Random(1)
    tokens = ["A", " B", "C)", "(D", "E", "a", "The", " b.", "AB"]
    for _ in range(300):
        positions = [
            [(rng.choice(tokens), -rng.random() * 6) for _ in range(rng.randint(0, 8))]
            for _ in range(rng.randint(1, 3))
        ]
        old_obj = {
            "content": [
                {"top_logprobs": [{"token": t, "logprob": lp} for t, lp in pos]}
                for pos in positions
            ]
        }
        assert letter_masses(positions, "ABCD") == legacy["probe_epf"].letter_masses(
            old_obj, "ABCD"
        )


@pytest.mark.parametrize("temperature", [1.0, 3.0])
def test_judge_weight_is_identical(legacy, temperature):
    old_inv_sigmoid = legacy["particle_filtering"]._inv_sigmoid
    for s in np.linspace(0, 1, 401):
        expected = old_inv_sigmoid(min(max(s, 1e-6), 1 - 1e-6)) / temperature
        assert log_weight(s, "logit", temperature) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("threshold, window", [(0.6, 0.7), (0.5, 0.5), (0.5, 1.0)])
def test_annealing_is_identical_to_its_hub(legacy, threshold, window):
    pf = legacy["particle_filtering"]
    old = object.__new__(pf.EntropicParticleFiltering)
    old.max_steps, old.ess_threshold, old.early_phase = 10, threshold, window
    old.temperature_method = pf.TemperatureMethod.ESS
    rng = np.random.default_rng(2)
    for _ in range(300):
        n = int(rng.integers(2, 65))
        p = softmax(rng.normal(0, rng.uniform(0.1, 6), n))
        step = int(rng.integers(1, 12))
        assert annealing_temperature(p, step, 10, threshold, window) == pytest.approx(
            old._temperature_annealing(p, step, n)
        )


def test_systematic_resampling_is_identical_to_its_hub(legacy, monkeypatch):
    old = object.__new__(legacy["particle_filtering"].EntropicParticleFiltering)
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(1, 65))
        p = softmax(rng.normal(0, 3, n))
        u = float(rng.uniform())
        monkeypatch.setattr(np.random, "uniform", lambda a, b, u=u: u)
        stub = SimpleNamespace(uniform=lambda a, b, u=u: u)
        assert systematic_indices(p, stub) == old._resampling_systematic_indices(p, n)


STORED_RESULTS = [
    PROJECT / "epf-with-different-prms" / "qwen-omni__qwen3b.jsonl",
    PROJECT / "epf_results" / "prm_multi" / "phi4mm__md06__mmau_csv_orig.jsonl",
]


def _stored_rows(limit=300):
    rows = []
    for path in STORED_RESULTS:
        if not path.exists():
            continue
        with open(path) as handle:
            for line in itertools.islice(handle, limit):
                row = json.loads(line)
                if not row.get("error"):
                    rows.append(row)
    p12 = sorted((PROJECT / "epf_results" / "p12_gated" / "out").glob("mmar_ext_*_b128g0.jsonl"))
    for path in p12[:1]:
        with open(path) as handle:
            rows += [json.loads(line) for line in itertools.islice(handle, 100) if line.strip()]
    return rows


def test_selectors_are_identical_on_stored_rows(legacy):
    rows = _stored_rows()
    if not rows:
        pytest.skip("no stored result rows on this machine")
    for row in rows:
        assert select_answers(row) == legacy["probe_epf"].selectors_from_row(row)


def test_answer_extraction_is_identical_on_stored_texts(legacy):
    rows = [r for r in _stored_rows(100) if r.get("texts")]
    if not rows:
        pytest.skip("no stored texts on this machine")
    choices = ["Piano", "Violin", "Drums", "Flute", "Guitar"]
    for row in rows:
        for text in row["texts"][:16]:
            assert predicted_index(text, choices) == legacy["scoring"].predicted_index(
                text, choices
            )


@pytest.mark.data
@pytest.mark.parametrize("bench", BENCHMARKS)
def test_loaders_are_identical(legacy, data_paths, monkeypatch, bench):
    ext = data_paths.extended_dir
    for var, value in {
        "BABEL_MMAR_ROOT": data_paths.mmar_root,
        "BABEL_MMAU_ROOT": data_paths.mmau_root,
        "BABEL_D1K_ROOT": data_paths.d1k_root,
        "BABEL_MEDIA_ROOT": data_paths.mmar_root.parent,
        "BABEL_MMAR_EXT": ext / "mmar_augmented.jsonl",
        "BABEL_MMAU_EXT": ext / "mmau_augmented.jsonl",
        "BABEL_D1K_EXT": ext / "mmau-pro-d1k_augmented.jsonl",
        "BABEL_EXT_CSV_DIR": ext,
    }.items():
        monkeypatch.setenv(var, str(value))
    monkeypatch.delenv("BABEL_AUDIO_OVERRIDE_DIR", raising=False)
    old = legacy["probe_epf"].load_records(bench, None)
    new = load_benchmark(bench, data_paths)

    def key(r):
        return (
            r.unique_id,
            r.question,
            list(r.choices),
            r.answer,
            r.answer_index,
            list(r.audio_paths),
            r.category,
        )

    assert len(new) == len(old)
    assert [key(r) for r in new] == [key(r) for r in old]


def test_no_audio_uuid_env_leaks_into_prompts():
    assert os.environ.get("EPF_AUDIO_UUID") != "1"
