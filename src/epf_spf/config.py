"""Filter configuration and the named presets.

Every knob that changes how particles are weighted or resampled lives in FilterConfig,
and the full resolved config is written into every result row. A preset only fills in
defaults; any field can be overridden from the command line.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from epf_spf.weighting import AGGREGATIONS, TRANSFORMS

SCORERS = ("judge", "probe", "random")
RESAMPLE_RULES = ("every_step", "ess_gate")
RESAMPLING_METHODS = ("systematic", "multinomial")
FAILURE_POLICIES = ("fail", "neutral")


@dataclass(frozen=True)
class FilterConfig:
    method: str
    scorer: str
    weight_transform: str
    weight_temperature: float
    score_aggregation: str
    resample_from_step: int
    resample_rule: str
    gate_threshold: float | None
    annealing: bool
    anneal_ess_threshold: float
    anneal_window: float
    resampling_method: str
    on_score_failure: str

    def __post_init__(self) -> None:
        if self.scorer not in SCORERS:
            raise ValueError(f"scorer must be one of {SCORERS}, got {self.scorer!r}")
        if self.weight_transform not in TRANSFORMS:
            raise ValueError(f"weight_transform must be one of {TRANSFORMS}")
        if self.weight_temperature <= 0:
            raise ValueError("weight_temperature must be > 0")
        if self.score_aggregation not in AGGREGATIONS:
            raise ValueError(f"score_aggregation must be one of {AGGREGATIONS}")
        if self.resample_from_step < 1:
            raise ValueError("resample_from_step must be >= 1")
        if self.resample_rule not in RESAMPLE_RULES:
            raise ValueError(f"resample_rule must be one of {RESAMPLE_RULES}")
        if self.resample_rule == "ess_gate" and not self.gate_threshold:
            raise ValueError("resample_rule='ess_gate' needs gate_threshold")
        if not 0 < self.anneal_ess_threshold <= 1:
            raise ValueError("anneal_ess_threshold must be in (0, 1]")
        if not 0 < self.anneal_window <= 1:
            raise ValueError("anneal_window must be in (0, 1]")
        if self.resampling_method not in RESAMPLING_METHODS:
            raise ValueError(f"resampling_method must be one of {RESAMPLING_METHODS}")
        if self.on_score_failure not in FAILURE_POLICIES:
            raise ValueError(f"on_score_failure must be one of {FAILURE_POLICIES}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


PRESETS: dict[str, dict] = {
    # The method as the ePF paper describes it (Algorithm "ePF", Sec. 4 setup, Table
    # of hyper-parameters): log-reward, no fixed temperature, step 1 left
    # unresampled, annealing when ESS_n < 0.5 during the first 50% of the steps.
    "epf-paper": dict(
        scorer="judge",
        weight_transform="log",
        weight_temperature=1.0,
        score_aggregation="last",
        resample_from_step=2,
        resample_rule="every_step",
        gate_threshold=None,
        annealing=True,
        anneal_ess_threshold=0.5,
        anneal_window=0.5,
        resampling_method="systematic",
    ),
    # The method as upstream its_hub implements it (EntropicParticleFiltering,
    # particle_gibbs.py at commit 17f7813): logit of the PRM score, product
    # aggregation in its documented setup, annealing only in the first half.
    "epf-itshub": dict(
        scorer="judge",
        weight_transform="logit",
        weight_temperature=1.0,
        score_aggregation="prod",
        resample_from_step=1,
        resample_rule="every_step",
        gate_threshold=None,
        annealing=True,
        anneal_ess_threshold=0.5,
        anneal_window=0.5,
        resampling_method="systematic",
    ),
    # The project's judge arm (judge_epf_prm) at the weight temperature its
    # documentation specifies, T = 3.
    "epf-project": dict(
        scorer="judge",
        weight_transform="logit",
        weight_temperature=3.0,
        score_aggregation="last",
        resample_from_step=1,
        resample_rule="every_step",
        gate_threshold=None,
        annealing=True,
        anneal_ess_threshold=0.6,
        anneal_window=0.7,
        resampling_method="systematic",
    ),
    # Self-Probing Particle Filtering (probe_adaptive_t3): the generator's own forced
    # answer as the score, T = 3, resampling only when the annealed ESS_n < 0.5.
    "spf": dict(
        scorer="probe",
        weight_transform="logit",
        weight_temperature=3.0,
        score_aggregation="last",
        resample_from_step=1,
        resample_rule="ess_gate",
        gate_threshold=0.5,
        annealing=True,
        anneal_ess_threshold=0.6,
        anneal_window=0.7,
        resampling_method="systematic",
    ),
}


def build_config(method: str, on_score_failure: str = "fail", **overrides) -> FilterConfig:
    """The preset for `method` with every non-None override applied."""
    if method not in PRESETS:
        raise ValueError(f"method must be one of {', '.join(PRESETS)}, got {method!r}")
    fields = {**PRESETS[method], **{k: v for k, v in overrides.items() if v is not None}}
    return FilterConfig(method=method, on_score_failure=on_score_failure, **fields)
