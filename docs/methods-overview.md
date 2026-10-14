# Methods overview

All four methods share one loop ([`filter.py`](../src/epf_spf/filter.py)). They differ only
in the values of the fields of `FilterConfig` ([`config.py`](../src/epf_spf/config.py)).
A method name selects a preset. Any field can then be overridden with a flag of the same
name (`weight_temperature` becomes `--weight-temperature`).

## The loop

For one question and a budget of N particles:

1. **Propagate.** Every particle that has not finished generates its next reasoning step
   (`Step k: ...`). A particle finishes when the step contains `Answer:`, or when it
   already had `max_steps` steps before generating.
2. **Score.** The scorer returns a score s in [0, 1] for each particle that just
   generated a step (see [scorers.md](scorers.md)).
3. **Weight.** The particle's scores so far are aggregated into one number (`last`,
   `prod`, `min` or `mean`), then turned into a log-weight: `log(s)/T` or `logit(s)/T`.
   A finished particle keeps the weight it finished with.
4. **Resample**, unless the step is before `resample_from_step`:
   - p = softmax(log-weights)
   - entropic annealing: when ESS_n(p) < `anneal_ess_threshold` and
     step / `max_steps` < `anneal_window`, flatten with
     β⁻¹ = max(1, (1 − step/max_steps) / ESS_n) and p = softmax(log-weights / β⁻¹)
   - SPF only: skip the draw unless the annealed ESS_n < `gate_threshold`
   - draw N parent indices from p (`systematic` or `multinomial`) and copy those
     particles. Finished particles are drawn like any other.
5. Repeat until every particle has finished.

The final answer is then read off the N particles by the selectors in
[answer-selection.md](answer-selection.md). All of them are computed and stored for every
row, so the selector is chosen at analysis time and is not a run setting.

## The presets

| field | `epf-paper` | `epf-itshub` | `epf-project` | `spf` |
|---|---|---|---|---|
| `scorer` | judge | judge | judge | **probe** |
| `weight_transform` | **log** | logit | logit | logit |
| `weight_temperature` (T) | 1 | 1 | **3** | **3** |
| `score_aggregation` | last | **prod** | last | last |
| `resample_from_step` | **2** | 1 | 1 | 1 |
| `resample_rule` | every_step | every_step | every_step | **ess_gate** |
| `gate_threshold` | none | none | none | 0.5 |
| `annealing` | on | on | on | on |
| `anneal_ess_threshold` | 0.5 | 0.5 | **0.6** | **0.6** |
| `anneal_window` | 0.5 | 0.5 | **0.7** | **0.7** |
| `resampling_method` | systematic | systematic | systematic | systematic |

Where each column comes from:

| preset | source | details |
|---|---|---|
| `epf-paper` | the ePF paper: Algorithm "ePF", Sec. 4 "Baselines and Setup", hyper-parameter table | [epf-paper.md](epf-paper.md) |
| `epf-itshub` | `its_hub` `EntropicParticleFiltering`, `particle_gibbs.py` at commit 17f7813, with the `aggregation_method="prod"` reward setup from its docs | [epf-its-hub.md](epf-its-hub.md) |
| `epf-project` | the project's `judge_epf_prm` arm in `entropic-particle-filter/babel-ship/probe_epf.py`, at the T = 3 its documentation specifies | [weight-temperature.md](weight-temperature.md) |
| `spf` | the project's `probe_adaptive_t3` arm in the same file | [spf.md](spf.md) |

## What each difference does

| knob | effect |
|---|---|
| `log` vs `logit` | `log`: the resampling probability is proportional to s, so a particle with s = 0.99 gets at most ~2x one with s = 0.5. `logit`: the probability is proportional to the odds s/(1−s), so the same pair differs by ~99x. Near s = 1 the logit keeps growing while `log` saturates. |
| T | divides the log-weight. Under `logit`, the resampling probability ∝ odds^(1/T). T = 3 takes the cube root of the odds: 99:1 becomes ~4.6:1. |
| `last` vs `prod` | `last`: a particle is judged on its newest step only. `prod`: one bad step lowers the particle for the rest of its life, and the product shrinks as steps accumulate. |
| `resample_from_step` 2 vs 1 | the paper keeps all N first steps (w₁ = 1/N, its loop starts at t = 2); its_hub and the project already resample after step 1. |
| `anneal_window` | annealing only fires during the first `window` fraction of `max_steps`; later, resampling uses p unflattened. |
| `anneal_ess_threshold` | annealing fires only when ESS_n is below it; higher = fires more often. |
| `ess_gate` | SPF resamples only when the weights are already concentrated; otherwise particles keep their own trajectories. |

## Combining freely

Scorer and method are independent. Some combinations worth knowing:

```bash
# the paper's rule with the project's horizon handling and T = 3 log-weights
--method epf-paper --weight-temperature 3

# its_hub's weighting, but judged on the newest step only
--method epf-itshub --score-aggregation last

# the project's judge arm at T = 1 (what the Fir extension runs actually used)
--method epf-project --weight-temperature 1

# ePF-style resampling (every step) with the self-probe score
--method epf-project --scorer probe

# SPF with the external judge instead of the probe
--method spf --scorer judge

# controls: random scores, and no resampling at all
--method epf-paper --scorer random
--method epf-paper --resample-from-step 1000
```

The method name stays in the row as `method`, and the overridden values are in `config`.
A row therefore always says what actually ran, whatever the name suggests.

## Things that are not method settings but change results

These are run settings. They are written into every row and into the run fingerprint
([output-format.md](output-format.md)):

| setting | default | note |
|---|---|---|
| `--max-steps` | 10 | also the annealing horizon T in t/T. A particle can end with `max_steps + 1` steps. The paper used 300. |
| `--sampling-temperature` | 0.8 | the generator's sampling temperature. The paper used 0.7 with top-k 20 and top-p 0.8; this code sends no top-k/top-p, as the previous code did. |
| `--max-step-tokens` | 300 | the paper used 512 |
| judge model, generator model, benchmark, audio override, seed | | |

## Choosing

- To reproduce the paper's algorithm on audio, use `epf-paper`.
- To match what its_hub would do out of the box, use `epf-itshub`.
- To be comparable with the project's earlier judge-PRM results, use `epf-project` with
  the T those results were run at. See [weight-temperature.md](weight-temperature.md):
  this was not the same T everywhere.
- For a method that needs no reward model, use `spf`.
