# ePF as its_hub implements it (`epf-itshub`)

Source: [`Red-Hat-AI-Innovation-Team/its_hub`](https://github.com/Red-Hat-AI-Innovation-Team/its_hub),
class `EntropicParticleFiltering` in `its_hub/core/algorithms/particle_gibbs.py`. Line
numbers below are at commit **17f7813**. `EntropicParticleFiltering` is a
`ParticleGibbs` with one iteration and no reference particles, so the loop it runs is
`ParticleGibbs`'s.

The previous project code (`entropic-particle-filter/particle_filtering.py`) is a copy
of this class with project additions. `tests/test_parity.py` checks the annealing
temperature and the systematic draw of this repository against that copy.

## The loop, with line references

| what | its_hub | line |
|---|---|---|
| constructor defaults | `resampling_method=SYSTEMATIC`, `temperature_method=ESS`, `ess_threshold=0.5`, `early_phase=0.5`, `final_response_selection=ARGMAX` | 560–571 (and 115–116 on the base) |
| propagate | every unstopped particle generates one step | 403 |
| step counter | `current_step += 1  # Increment after propagation` | 411 |
| score | the PRM scores each unstopped particle's steps so far, post-processed as a finished response | 185–191 |
| weight | `p.partial_log_weights.append(_inv_sigmoid(scores[i]))` | 198 |
| logit | `x = np.clip(x, 1e-7, 1 - 1e-7); return np.log(x / (1 - x))` | 66–70 |
| weights used for resampling | unstopped: `partial_log_weights[current_step - 1]`; stopped: `# Stopped particles use their final weight` | 414–422 |
| annealing | `if self.does_entropic_annealing: temperature = self._temperature_annealing(...)`, then `softmax(log_weights * (1 / temperature))` | 427–433 |
| annealing condition | `ess_ratio < self.ess_threshold and progress < self.early_phase`, with `progress = current_step / self.max_steps` | 290, 298 |
| ESS temperature | `max(1.0, 1.0 / ess_ratio * (1 - progress))` | 264–278 |
| resample | `self._resampling(particles, probabilities, ...)`, then deep copies | 436–447 |
| look-ahead | `raise NotImplementedError("Lookahead modulation is not implemented")` | 444 |
| final answer | `np.argmax(log_weights)` over the last iteration's weights | 490 |

## How it differs from the paper

| | paper | its_hub |
|---|---|---|
| reward → log-weight | r = log s | **logit(s)** = log(s / (1 − s)) |
| resampling probability ∝ | s | **s / (1 − s)**, the odds |
| first resampling | after step 2 (loop t = 2 … T, w₁ = 1/N) | **after step 1**: `current_step` is 1 at the first resample |
| annealing window | first 50% of the steps (Sec. 4) | `early_phase = 0.5` |
| τ | 0.5 | `ess_threshold = 0.5` |
| what s is | the PRM's score of z_{1:t} | the PRM's score of the steps so far, **aggregated by the reward model** |
| finished particles | not stated | keep their final weight and stay in the pool |
| LaM | part of the paper's full method | not implemented |

### The score aggregation

its_hub passes the whole trajectory to `prm.ascore`. The reward model returns one
number per trajectory, and how the per-step PRM scores are combined is a setting of the
reward model, not of the algorithm. Every ePF/PF example in its_hub's documentation
(`docs/algorithms.md`, `docs/quick-start.md`) constructs the PRM with
`aggregation_method="prod"`. The preset therefore uses `score_aggregation="prod"`: the
weight of a particle at step t is logit(s₁ · s₂ · … · s_t).

The judge in this project scores one step per call (see [scorers.md](scorers.md)), so
the product is taken here over the judge's per-step answers. Use
`--score-aggregation last` to weight by the newest step only, as the project's arm did.

## Mapping to the code

| its_hub | `epf-itshub` preset / code |
|---|---|
| `_inv_sigmoid` | `weight_transform="logit"`, `weight_temperature=1.0` |
| `aggregation_method="prod"` | `score_aggregation="prod"` |
| resample from step 1 | `resample_from_step=1` |
| `ess_threshold=0.5`, `early_phase=0.5` | `anneal_ess_threshold=0.5`, `anneal_window=0.5` |
| `TemperatureMethod.ESS` | `resampling.annealing_temperature` (parity-tested) |
| `ResamplingMethod.SYSTEMATIC` | `resampling.systematic_indices` (parity-tested) |
| stopped particles keep `p.log_weight` | finished particles keep their `log_weight` |
| `StepGeneration.max_steps` | `--max-steps` |
| `SelectionMethod.ARGMAX` | the `argmax_w` selector |

## Known small differences

- **Clamp.** its_hub clips s to [1e-7, 1 − 1e-7] before the logit. This code clips to
  [1e-6, 1 − 1e-6], as the previous project code did. They differ only for scores within
  1e-6 of 0 or 1. Such scores do occur with a saturating judge: renormalised P(Yes) is
  exactly 1.0 when "No" is not among the top 20 tokens. In that case the log-weight is
  log(1e6) ≈ 13.8 here and log(1e7) ≈ 16.1 in its_hub.
- **Generator stopping.** its_hub's `StepGeneration` stops a particle on its
  `stop_token` or at `max_steps`. Here a particle stops on `Answer:` or at `max_steps`.
  This is the same rule the previous code used through its_hub's `StepGeneration` (see
  [parity-and-differences.md](parity-and-differences.md)).
- **No `ParticleGibbs` iterations.** ePF runs one iteration, so the reference-particle
  machinery is not needed and not implemented.
