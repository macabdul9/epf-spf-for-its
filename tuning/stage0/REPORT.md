# Stage 0: score signal on MMAU-Pro D1K, standard options

Script: `score_signal.py`. It only reads stored rows (no requests, no GPU). Output files:
`discrimination.csv`, `selection.csv`, `concentration.csv`.

## Inputs

| generator | scorer | file(s) | budgets | weight T (checked) |
|---|---|---|---|---|
| qwen-omni (Omni-7B) | Qwen3-Omni-30B-A3B judge | `epf-with-qwen3-30ba3b-prm/original/mmau_pro_d1k_qwen-omni_judgeprm_p12.jsonl` + `extended-source-b128-queue/original/..._t3_p12.jsonl` | 1–128 | 3 |
| qwen-omni-3b | same judge | same layout | 1–128 | 3 |
| qwen-omni | self-probe (SPF, `probe_adaptive_t3`) | `epf_results/p12_gated/out/mmau_pro_d1k_sub_qwen-omni_probe_adaptive_t3_p12_b128g{0..3}.jsonl` | 128 | 3 |
| qwen-omni-3b | self-probe | `..._qwen-omni-3b_..._b128g{0,1}.jsonl` | 128 | 3 |

- Items are the 583 `mmau_pro_d1k_sub` ids (the extended dataset's D1K ids with the
  original options). The judge files cover all 991 D1K items and were filtered to these.
- Every cell is present once, there are no error rows, and the stored gold letters
  match the new loader for every row.
- T = 3 was recovered from the data: logit(last judge score) / final weight has a
  median of 3.000 (5–95%: 2.99–3.01).
- The b128 judge files in `extended-source-b128-queue/` and `re-reused-from-rorqual/`
  are byte-identical (md5).

## 1. Discrimination: within an item, do correct trajectories score higher?

This is the AUROC of a trajectory's score for correct vs wrong distinct final
trajectories, computed per item (only items with both kinds) and then averaged. 0.5
means no signal. Each score summary corresponds to an aggregation choice. 95% CIs come
from a bootstrap over items.

| generator | scorer | b | items used | first step | last | mean | min | prod |
|---|---|---|---|---|---|---|---|---|
| Omni-7B | 30B judge | 16 | 225 | 0.51 | **0.55** (0.50–0.59) | 0.54 | 0.53 | 0.54 |
| Omni-7B | 30B judge | 64 | 333 | 0.51 | **0.56** (0.52–0.59) | 0.55 | 0.55 | 0.55 |
| Omni-3B | 30B judge | 16 | 253 | 0.55 | **0.62** (0.57–0.66) | 0.56 | 0.58 | 0.58 |
| Omni-3B | 30B judge | 64 | 356 | 0.52 | **0.56** (0.52–0.59) | 0.55 | 0.55 | 0.55 |
| Omni-7B | probe | 128 | 558 | 0.49 | **0.55** (0.52–0.57) | 0.53 | 0.52 | 0.53 |
| Omni-3B | probe | 128 | 575 | 0.51 | **0.54** (0.52–0.56) | 0.53 | 0.53 | 0.53 |

Pooled over all items, the AUROC is about 0.60–0.64. That pooled figure mostly reflects
item difficulty (easy items get both high scores and right answers), which cannot help
resampling, because resampling only compares particles within one item.

Caveat: these are the trajectories that survived resampling at T = 3, so the
population is already selected by the judge.

## 2. Selection: accuracy by budget

| generator | scorer | b | argmax_w | dedup_majority | majority | judge_marginal | random distinct trajectory | oracle (any distinct correct) | distinct fraction |
|---|---|---|---|---|---|---|---|---|---|
| Omni-7B | 30B judge | 1 | 0.472 | | | | 0.472 | 0.472 | 1.00 |
| Omni-7B | 30B judge | 16 | 0.496 | 0.515 | 0.515 | 0.518 | 0.502 | 0.703 | 0.80 |
| Omni-7B | 30B judge | 64 | 0.532 | 0.540 | 0.551 | 0.547 | 0.524 | 0.810 | 0.76 |
| Omni-7B | 30B judge | 128 | 0.523 | 0.539 | 0.537 | 0.540 | 0.517 | 0.863 | 0.74 |
| Omni-3B | 30B judge | 1 | 0.407 | | | | 0.407 | 0.407 | 1.00 |
| Omni-3B | 30B judge | 16 | 0.521 | 0.496 | 0.511 | 0.506 | 0.487 | 0.717 | 0.76 |
| Omni-3B | 30B judge | 64 | 0.506 | 0.516 | 0.509 | 0.516 | 0.494 | 0.808 | 0.71 |
| Omni-3B | 30B judge | 128 | 0.492 | 0.506 | 0.511 | 0.509 | 0.495 | 0.858 | 0.67 |
| Omni-7B | probe (SPF) | 128 | 0.477 | 0.496 | 0.492 | probe_marginal 0.489 | 0.457 | 0.988 | 0.95 |
| Omni-3B | probe (SPF) | 128 | 0.451 | 0.494 | 0.499 | probe_marginal 0.501 | 0.420 | 0.997 | 0.98 |

"Random distinct trajectory" is the expected accuracy of one trajectory drawn uniformly
from the final distinct trajectories. Without resampling it would equal b = 1 accuracy,
so the difference between them is what resampling did to the population.

The SPF rows have no b = 1 of their own. The b = 1 figure from the judge files is a
plain sample from the same generator and prompt, which is why it serves as the
reference.

## 3. Concentration: ESS_n at step 1 under each weight rule

This is the median over items of ESS_n of the step-1 resampling distribution, using
all N particles' step-1 scores. Those scores are not affected by selection.

| generator | scorer | b | median step-1 score | log T1 (paper) | logit T1 | logit T2 | logit T3 | logit T1 + annealing (paper / project) | logit T3 + annealing (paper / project) |
|---|---|---|---|---|---|---|---|---|---|
| Omni-7B | 30B judge | 64 | 0.987 | 0.94 | 0.29 | 0.59 | 0.74 | 0.75 / 0.75 | 0.75 / 0.76 |
| Omni-3B | 30B judge | 64 | 0.932 | 0.85 | 0.20 | 0.45 | 0.63 | 0.78 / 0.78 | 0.66 / 0.73 |
| Omni-7B | probe | 128 | 0.579 | 0.94 | 0.11 | 0.51 | 0.79 | 0.97 / 0.97 | 0.83 / 0.85 |
| Omni-3B | probe | 128 | 0.660 | 0.93 | 0.13 | 0.46 | 0.72 | 0.95 / 0.95 | 0.73 / 0.77 |

## Findings

1. **Both scores carry weak within-item signal**, with AUROC around 0.54–0.56 at the
   last step. For the 30B judge on Omni-3B at b16 it reaches 0.62.
2. **The step-1 score carries no usable signal** (0.49–0.55, most CIs include 0.5), yet
   under `logit` at T = 1 the step-1 weights are the most concentrated: ESS_n
   0.11–0.29 before annealing. Resampling at step 1 therefore mostly copies particles
   at random.
3. **The `last` score separates best**; `prod`, `min` and `mean` add nothing over it.
4. **The 30B-guided filter does improve the particle population**: a random final
   trajectory is right 52% of the time at b64 vs 47% at b1 (Omni-7B), and 49% vs 41%
   (Omni-3B). The judge-based selectors then add little or nothing over majority vote.
5. **The stored SPF runs barely resampled** (95–98% of final trajectories distinct),
   and the random-trajectory accuracy did not rise above b1 (0.457 vs 0.472 and
   0.420 vs 0.407). At these settings SPF behaved like self-consistency, with
   majority ≈ 0.49–0.50 at b128.
6. **The headroom is large.** Some distinct trajectory is correct in 81–86% (judge, b64–128)
   or 99% (SPF, b128) of items, against about 50–55% selected.

## Statistical resolution

With 583 items, one accuracy has a standard error of about 2.1 points. Between two
settings on the same items the paired SE is smaller, but sampling noise between runs
adds to it: expect about 1.5–2.5 points. On a 300-item split, about 2.5–3.5 points.
Differences in the tables above of under about 3 points (e.g. argmax_w vs majority) are
within noise.
