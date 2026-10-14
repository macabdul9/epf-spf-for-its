# Weight transform and weight temperature

Two flags decide how a score s in [0, 1] becomes a log-weight:

```text
--weight-transform log     log-weight = log(s) / T      resampling probability ∝ s^(1/T)
--weight-transform logit   log-weight = logit(s) / T    resampling probability ∝ (s / (1 − s))^(1/T)
--weight-temperature T     T > 0, default per preset
```

At T = 1 these are exactly the paper's rule (`log`) and its_hub's rule (`logit`). At
T = 1, `logit(s)/T` and `logit(s)` are the same thing, so `epf-itshub` and
`epf-project --weight-temperature 1` weight particles identically. Those two presets
still differ in aggregation, annealing settings, and `resample_from_step`.

T is not the annealing temperature β⁻¹. T is fixed for a run and applied to every
weight. β⁻¹ is computed at every step from the ESS and divides the already-tempered
log-weights a second time when annealing fires. The trace records β⁻¹ as
`annealing_temperature` ([output-format.md](output-format.md)).

## What the transforms do to a pair of particles

Ratio of resampling probabilities between a particle with score a and one with score b:

| a vs b | `log`, T=1 (paper) | `logit`, T=1 (its_hub) | `logit`, T=2 | `logit`, T=3 |
|---|---|---|---|---|
| 0.6 vs 0.4 | 1.5 | 2.25 | 1.5 | 1.31 |
| 0.9 vs 0.5 | 1.8 | 9 | 3 | 2.08 |
| 0.99 vs 0.5 | 1.98 | 99 | 9.95 | 4.63 |
| 0.999 vs 0.9 | 1.11 | 111 | 10.5 | 4.81 |
| 0.5 vs 0.001 | 500 | 999 | 31.6 | 9.99 |

Two things follow:

- `log` is bounded above: no particle can get more than 1/s_min times another, and two
  high scores are nearly equal (0.999 vs 0.9 is 1.11). `logit` is unbounded at both
  ends, so it separates high scores that `log` treats as the same.
- T compresses `logit`. Raising T makes `logit` closer to `log` near the top of the
  range. Low scores are still penalised less than under `log` (0.5 vs 0.001 is 9.99 at
  T = 3 vs 500 under `log`).

Which behaviour helps depends on how calibrated the scorer is. A judge whose P(Yes) sits
near 1 for most steps gives `logit` at T = 1 large weight gaps from small score
differences. A judge whose scores sit near 0.5 gives nearly uniform weights under any
setting.

## Measured: how much each setting concentrates the particles

Step-1 resampling distribution computed offline from recorded judge scores. MMAR,
qwen-omni (Qwen2.5-Omni-7B) generator, 64 particles, 150 items per judge. ESS_n is the
effective fraction of particles kept (1.0 = uniform). "With annealing" applies the
project's annealing (threshold 0.6, window 0.7) at step 1 of 10.

| judge | P(Yes) 10th / 50th / 90th pct | `log` T=1 | `logit` T=1 | `logit` T=2 | `logit` T=3 | with annealing: `log` T=1 / `logit` T=1 / T=2 / T=3 |
|---|---|---|---|---|---|---|
| midashenglm-0.6B | 0.62 / 0.82 / 0.95 | 0.996 | 0.917 | 0.978 | 0.990 | 0.996 / 0.917 / 0.978 / 0.990 |
| Qwen2.5-Omni-3B | 0.13 / 0.38 / 0.62 | 0.922 | 0.833 | 0.950 | 0.977 | 0.924 / 0.850 / 0.950 / 0.977 |
| Qwen2.5-Omni-7B | 0.16 / 0.48 / 0.87 | 0.914 | 0.723 | 0.914 | 0.961 | 0.924 / 0.805 / 0.914 / 0.961 |
| Qwen3-Omni-30B-A3B | 0.29 / 0.99 / 1.00 | 0.904 | 0.332 | 0.587 | 0.728 | 0.912 / 0.756 / 0.738 / 0.766 |

Reading the table:

- For the three small and mid-size judges, every setting keeps most particles. `logit`
  at T = 1 is the most concentrating. T = 3 keeps more than the paper's `log` for the
  3B and 7B judges, and slightly fewer for midashenglm (0.990 vs 0.996).
- The 30B judge saturates (median P(Yes) 0.99). Here `logit` at T = 1 keeps a third of
  the particles before annealing. Annealing pulls it back to about three quarters at
  step 1, but annealing stops after 70% of the steps.
- The `log` column ignores that the paper does not resample at step 1. It is shown
  for comparison of the weight rule alone.

One further measurement of trajectory diversity: on a 100-item midashenglm-0.6B, b = 64
check, 88% of the final trajectories were distinct at T = 3, against 67% at T = 1.

## Why T is a flag here

In the previous code, the weight temperature of the judge arm was a constant inside
`build_arm` in `entropic-particle-filter/babel-ship/probe_epf.py`. There was no flag for
it, and result rows did not record it. That file existed in more than one copy, and the
copies did not agree. The audit in `../audit/` (see `../audit/REPORT.md`) established
from the code copies, launch scripts and logs which runs used which value:

- The original different-PRMs MMAR runs (b = 1–64) and all Qwen3-Omni-30B-A3B judge
  runs used T = 3.
- The Fir extension runs (32 output files, 27 settings) used T = 1.0, the value in the
  Fir copy of the file.

The project's documentation specifies T = 3 for the judge arm, which is the
`epf-project` default.

In this repository T is part of `FilterConfig`. It is printed at startup, stored in
every row as `config.weight_temperature`, and covered by the run fingerprint. A run at
another T cannot be resumed into the same output file ([output-format.md](output-format.md)).

## Recipes

```bash
--method epf-project                          # logit / 3, the documented project setting
--method epf-project --weight-temperature 1   # logit / 1, as the Fir extension runs ran
--method epf-itshub                           # logit, prod, its_hub defaults
--method epf-paper                            # log, the paper
--method epf-paper --weight-temperature 2     # s^(1/2): flatter than the paper
```
