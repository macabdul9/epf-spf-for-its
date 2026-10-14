# ePF as the paper describes it (`epf-paper`)

Source: `papers/epf-paper/` in the parent tree, "Entropic Particle Filtering". Three
places define the method:

- Sec. 3, "Entropic Particle Filtering (ePF)": `sec/3_method.tex`
- the algorithm block "Particle Filtering with Entropic Annealing (ePF)":
  `sec/appendix/algorithms.tex`
- the experimental setup, Sec. 4 "Baselines and Setup" (`sec/4_experiment.tex`), and the
  hyper-parameter table (`sec/appendix/details.tex`)

The paper's experiments are on math with a trained PRM (Qwen2.5-Math-PRM-7B). Here the
same algorithm runs on audio multiple-choice questions, and the PRM is replaced by a
zero-shot audio judge or the self-probe.

## The algorithm, step by step

Notation: N particles; T the number of steps (the horizon); s the PRM score in [0, 1].

**Initialisation**

1. Sample N first steps z₁ ~ p(z₁ | c).
2. Set w₁ = 1/N. The first step is not weighted: the posterior factorisation sets
   p(o₁ | z₁, c) = 1.

**For t = 2 … T**

3. Propagate: each particle samples its next step z_t ~ p(z_t | z_{t−1}, c).
4. Score: s_t = g(z_{1:t}, c), the PRM's score of the trajectory so far.
5. Log-reward: r_t = log s_t.
6. Weights: w_t = softmax(r_t), i.e. w_t ∝ s_t.
7. ESS_n(t) = (1/N) / Σ w².
8. If ESS_n(t) < τ: β⁻¹ = max(1, (1 − t/T) / ESS_n(t)) and w_t = softmax(r_t · β_t).
9. Resample N indices with **systematic** resampling and copy those particles.

**Output**

10. The answer of the selected particle. In the experiments: "The final solution from
    each run is selected using the same PRM that provides intermediate guidance". This
    is the Argmax rows of the result tables.

**Hyper-parameters used in the paper's experiments**

| | value | source |
|---|---|---|
| τ | 0.5 | table; Sec. 4: "threshold of ESS_n(t) ≤ 0.5" |
| annealing window | first 50% of steps | Sec. 4: "We use ePF for the first 50 % of the steps" |
| T (max steps) | 300 | table |
| tokens per step | 512 | table |
| sampling | temperature 0.7, top-k 20, top-p 0.8 | table |
| budgets | 2, 4, 8, 16, 32 | table |

## Mapping to the code

| paper | `epf-paper` preset / code |
|---|---|
| r = log s | `weight_transform="log"`, `weight_temperature=1.0` → `log_weight = log(s)` |
| weight from the current step's reward | `score_aggregation="last"`. The PRM scores z_{1:t}, so the score already sees the whole trajectory, but no product over steps is taken. |
| loop starts at t = 2 | `resample_from_step=2`: step 1 is generated and scored (the score is stored) but not resampled |
| ESS_n < τ, τ = 0.5 | `anneal_ess_threshold=0.5` |
| first 50% of the steps | `anneal_window=0.5` |
| β⁻¹ = max(1, (1 − t/T)/ESS_n) | `resampling.annealing_temperature` |
| systematic resampling | `resampling_method="systematic"`, `resampling.systematic_indices` |
| T | `--max-steps` (default 10, not the paper's 300; see below) |
| PRM selection of the final answer | the `argmax_w` selector (highest final log-weight). Other selectors are stored too. |

With `log` at T = 1, softmax(log s) = s / Σ s: resampling probability is exactly
proportional to the PRM score, as in the paper.

## What the paper leaves open, and what this code does

| question | the paper | here |
|---|---|---|
| what happens to a particle that has finished while others continue | not stated (the algorithm runs a fixed t = 2 … T) | finished particles keep their last weight and stay in the resampling pool, as in its_hub |
| `<` or `≤` τ | the algorithm uses `<`, the text says `≤` | `<`, as in the algorithm block and its_hub |
| what T is in t/T when trajectories are shorter than the cap | T = max steps | `--max-steps` |
| clamping s = 0 | not stated (log 0 = −∞) | s is clamped to [1e-6, 1] |
| top-k / top-p | 20 / 0.8 | not sent; only `--sampling-temperature` (default 0.8, the project's setting). Use `--sampling-temperature 0.7` for the paper's temperature. |

## The horizon

The paper's T = 300 is a step cap that math trajectories approach on AIME. Here the
reasoning format asks for at most 10 steps, and `--max-steps 10` is the cap. t/T
therefore runs from 0.1 to 1.0 within one trajectory, and the 50% window covers steps
1–4 (t/T < 0.5). With T = 300 the window would never close in practice. Keep
`--max-steps 10` to stay comparable with the project's other runs.

## Not implemented

- **Look-ahead Modulation (LaM)**, the paper's second component. It is not implemented
  here, and its_hub does not implement it either (`raise NotImplementedError("Lookahead
  modulation is not implemented")`).
- Linear and entropy temperature schedules (paper appendix ablation). Only the ESS
  schedule is implemented, which is the paper's main method.
