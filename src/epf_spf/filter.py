"""The particle filter over reasoning steps.

One loop serves every method; FilterConfig decides how scores become weights and when
and how particles are resampled. Each round:

  1. every running particle generates its next step
  2. the scorer scores the particles that ran this round
  3. each such particle's log-weight is recomputed from its scores so far
  4. finished particles keep the weight they finished with and stay in the pool
     (its_hub's behaviour; the paper does not say)
  5. unless this step is before `resample_from_step`, the resampling distribution is
     built (with entropic annealing) and, unless the SPF gate holds it back, N particles
     are drawn with replacement
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from epf_spf.config import FilterConfig
from epf_spf.records import MCQRecord
from epf_spf.resampling import (
    ess_ratio,
    gate_allows_resampling,
    multinomial_indices,
    resampling_distribution,
    softmax,
    systematic_indices,
)
from epf_spf.steps import StepFormat
from epf_spf.weighting import aggregate, log_weight


class ScoreFailure(RuntimeError):
    """A judge call failed and the config says not to substitute a neutral score."""


@dataclass
class Particle:
    steps: list[str] = field(default_factory=list)
    finished: bool = False
    scores: list[float | None] = field(default_factory=list)
    letter_masses: list[dict | None] = field(default_factory=list)
    log_weight: float = 0.0

    def copy(self) -> Particle:
        return dataclasses.replace(
            self,
            steps=list(self.steps),
            scores=list(self.scores),
            letter_masses=[dict(m) if m else None for m in self.letter_masses],
        )


@dataclass
class FilterResult:
    particles: list[Particle]
    trace: list[dict]
    score_failures: int = 0


async def run_filter(
    record: MCQRecord,
    budget: int,
    config: FilterConfig,
    generator,
    scorer,
    fmt: StepFormat,
    base_messages: list[dict],
    rng: np.random.Generator,
) -> FilterResult:
    particles = [Particle() for _ in range(budget)]
    trace: list[dict] = []
    failures = 0
    step = 0
    while not all(p.finished for p in particles):
        running = [i for i, p in enumerate(particles) if not p.finished]
        ran = set(running)
        new_steps = await generator.next_steps(
            base_messages, [particles[i].steps for i in running], fmt
        )
        for i, text in zip(running, new_steps, strict=True):
            particle = particles[i]
            particle.finished = fmt.is_final(particle.steps, text)
            particle.steps.append(text)
        step += 1

        scored = await scorer.score(
            record, base_messages, [(particles[i].steps, particles[i].finished) for i in running]
        )
        for i, result in zip(running, scored, strict=True):
            if result.failed:
                failures += 1
                if config.on_score_failure == "fail":
                    raise ScoreFailure(f"judge call failed at step {step}")
            particle = particles[i]
            particle.scores.append(result.score)
            particle.letter_masses.append(result.letter_masses)
            particle.log_weight = log_weight(
                aggregate(particle.scores, config.score_aggregation),
                config.weight_transform,
                config.weight_temperature,
            )

        log_weights = [p.log_weight for p in particles]
        snapshot = [
            {
                "new_text": particles[i].steps[-1] if i in ran else None,
                "finished": particles[i].finished,
                "score": particles[i].scores[-1] if i in ran else None,
                "log_weight": particles[i].log_weight,
            }
            for i in range(len(particles))
        ]
        round_trace = {"step": step, "particles": snapshot}

        if step < config.resample_from_step:
            round_trace["resample"] = {"applied": False, "reason": "before resample_from_step"}
            trace.append(round_trace)
            continue

        probabilities, annealing_temperature = resampling_distribution(
            log_weights,
            step,
            fmt.max_steps,
            config.annealing,
            config.anneal_ess_threshold,
            config.anneal_window,
        )
        resample = {
            "annealing_temperature": annealing_temperature,
            "ess_ratio_untempered": ess_ratio(softmax(log_weights)),
            "ess_ratio": ess_ratio(probabilities),
            "probabilities": [float(x) for x in probabilities],
        }
        if config.resample_rule == "ess_gate" and not gate_allows_resampling(
            probabilities, config.gate_threshold
        ):
            resample |= {"applied": False, "reason": "ESS gate"}
            round_trace["resample"] = resample
            trace.append(round_trace)
            continue

        if config.resampling_method == "systematic":
            parents = systematic_indices(probabilities, rng)
        else:
            parents = multinomial_indices(probabilities, rng)
        resample |= {"applied": True, "parents": parents}
        round_trace["resample"] = resample
        trace.append(round_trace)
        particles = [particles[j].copy() for j in parents]

    return FilterResult(particles=particles, trace=trace, score_failures=failures)
