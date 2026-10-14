# Self-Probing Particle Filtering (`spf`)

SPF is the project's reward-model-free particle filter. It is the `probe_adaptive_t3`
arm of the previous implementation (`AnswerProbeEPF` in
`entropic-particle-filter/babel-ship/probe_epf.py`).

The idea: after each reasoning step, interrupt the generator and force it to answer
now. How sure it is of its forced answer is the particle's score. No second model is
needed, and the probe costs one short greedy call to the same server per particle per
step.

## The score

For a particle with steps z₁ … z_t:

1. Build the assistant prefix: the trajectory so far, then `\nAnswer:`. If the particle
   already wrote `Answer:` itself, cut the text right after its own `Answer:` instead.
2. Ask the generator to continue that prefix greedily (temperature 0, up to 3 tokens,
   top-20 log-probabilities per position, `continue_final_message`).
3. At the first generated position whose top-20 contains any option letter, sum the
   probability mass per letter over its surface forms (`B`, ` B`, `B)`, `(B`, `b.`, ...).
   Only letters of the question's options count (A–D for 4 options, A–J for 10).
4. s = the largest letter mass, p_max.

If no position has any letter mass, the score is missing and counts as s = 0.5 (logit
0, a neutral weight). The per-letter masses are stored per step as `probe_dists_all`;
the answer selectors use them ([answer-selection.md](answer-selection.md)).

p_max is the generator's confidence in whatever it would answer now, not the
probability that the answer is right. SPF bets that, on average, trajectories that make
the model more decided are the better ones.

## The resampling rule

| field | value | meaning |
|---|---|---|
| `weight_transform`, `weight_temperature` | logit, 3 | log-weight = logit(p_max) / 3 |
| `score_aggregation` | last | the newest probe only |
| `resample_from_step` | 1 | |
| `annealing` | on, threshold 0.6, window 0.7 | ePF's entropic annealing, the project's settings |
| `resample_rule` | `ess_gate` | resample only when the weights are concentrated |
| `gate_threshold` | 0.5 | ... i.e. when ESS_n of the annealed distribution < 0.5 |
| `resampling_method` | systematic | |

Each step:

1. p = softmax(logit(p_max) / 3).
2. Annealing, as in ePF: if ESS_n(p) < 0.6 and step/max_steps < 0.7, flatten p with
   β⁻¹ = max(1, (1 − t/T) / ESS_n).
3. **Gate:** compute ESS_n of the (annealed) p. If it is ≥ 0.5, do not resample: every
   particle keeps its own trajectory. The trace records `"reason": "ESS gate"`.
4. Otherwise draw N particles systematically from p.

The gate is the main difference from ePF, which resamples every step. While the
particles' confidences are similar, SPF behaves like N independent samples. It resamples
only when a few particles are much more confident than the rest.

Annealing and the gate interact. Annealing flattens p whenever ESS_n < 0.6 during the
first 70% of the steps, which pushes ESS_n up, often above the 0.5 gate. In that period
SPF resamples only when the weights are concentrated enough that ESS_n stays below 0.5
even after flattening. The gate's ESS uses a 1e-12 clip and annealing's uses 1e-7, both
as in the previous implementation.

## Running SPF

```bash
epf-spf --method spf --bench mmar \
    --generator-endpoint http://127.0.0.1:9401/v1 --generator-model qwen-omni \
    --budgets 1,2,4,8,16,32,64 --output runs/mmar_qwen-omni_spf.jsonl
```

No judge flags are needed. The probe uses the generator endpoint.

## Variants

```bash
--method spf --weight-temperature 1        # sharper: logit(p_max)
--method spf --resample-rule every_step    # ePF-style, resample every step with the probe score
--method spf --scorer judge --judge-...    # the SPF gate with the external judge score
--method spf --no-annealing                # gate on the raw softmax
```

## What the probe costs

One extra request per running particle per step: a greedy continuation of at most 3
tokens over a prompt that shares the generator's prefix (served from vLLM's prefix
cache when the server has prefix caching on). The judge arms instead send one request per particle per step to a second
model, with the audio re-encoded by that model.
