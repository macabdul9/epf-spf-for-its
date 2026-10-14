# Scorers

`--scorer judge`, `--scorer probe` or `--scorer random` decides where s comes from. Each preset has a
default; any preset accepts either. Both live in
[`scorers.py`](../src/epf_spf/scorers.py).

## `judge`: a zero-shot audio language model as the PRM

The judge is a separate audio LM served by vLLM. It is not trained as a reward model.
It is asked a Yes/No question, and its first-token probabilities are read.

### Which prompt, when

| particle state | template | what the judge sees |
|---|---|---|
| running, first step | step template | context `(none — this is the first step)`, latest step = step 1 |
| running, step k > 1 | step template | context = steps 1 … k−1 (with headers), latest step = step k |
| finished this round | outcome template | the whole trajectory as the candidate answer |

A finished particle is not scored again in later rounds. Its last score stays its
score.

Both templates are in [`prompts.py`](../src/epf_spf/prompts.py), byte-identical to
`judge_client.py` of the previous code (parity-tested). The question block is
`Question: …\n\nOptions:\nA. …\nB. …`.

The step template tells the judge not to penalise incomplete reasoning, to judge only
the latest step, and to answer "Yes" if the step is correct and advances toward the
right answer.

### The request

```text
messages     one user turn: [text prompt, audio clip(s)]   text first, then audio
temperature  1.0
max_tokens   1
logprobs     top 20
```

Audio is passed as a `file://` URL of the clip's **realpath**. vLLM resolves symlinks
before checking `--allowed-local-media-path`, so a symlinked path would be rejected.

### The score

```text
yes_mass = Σ exp(logprob) over top-20 tokens that normalise to "yes"
no_mass  = Σ exp(logprob) over top-20 tokens that normalise to "no"
s        = yes_mass / (yes_mass + no_mass)        or 0.5 if neither appears
```

A token normalises by stripping whitespace, `.`, `"` and curly quotes, then lowercasing
(`" Yes"`, `"yes."`, `"\"Yes"` all count).

Renormalising over Yes+No means the scale is set by the Yes:No ratio only. Mass the
judge puts on other tokens is ignored. When "No" is not in the top 20, s is exactly 1.0
(clamped to 1 − 1e-6 before the logit).

### Failures

A judge request that still fails after the client's retries (3 retries, 300 s timeout
each) is a **failed score**, distinct from a judge that answered neither Yes nor No
(which is a valid 0.5).

| `--on-score-failure` | behaviour |
|---|---|
| `fail` (default) | the item is aborted and written as an error row; it is retried on resume |
| `neutral` | the step counts as s = 0.5 and the run continues; the row's `score_failures` counts how many |

The previous code silently used 0.5 for failed calls. `fail` is the default here so
that an overloaded or crashed judge cannot quietly turn a run into an unguided one.

The run log ends with `judge calls N (M failed)`.

### Judges used in the project

| short name | model | served name (per `env-setup.md`) | vLLM env |
|---|---|---|---|
| 30B | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | the full repo id | legacy 0.22.1 |
| `md06` | `mispeech/midashenglm-0.6b-fp32` | `midashenglm-0.6b` | legacy 0.22.1, patched |
| `qwen3b` | `Qwen/Qwen2.5-Omni-3B` | `qwen-omni-3b-judge` | modern |
| `qwen7b_omni` | `Qwen/Qwen2.5-Omni-7B` | `qwen-omni-7b-judge` | modern |

Serving commands for each are in the parent tree's `env-setup.md`. The served model name
is whatever you pass to `--judge-model`; this code does not check it.

## `probe`: the generator's own forced answer

Described in full in [spf.md](spf.md#the-score). In short: append `\nAnswer:` to the
trajectory (or cut after the particle's own `Answer:`), continue greedily for up to 3
tokens with top-20 logprobs, sum letter masses at the first position that has any, and
take s = the largest mass. A probe with no letter mass is a missing score (s = 0.5),
not a failure. A probe request that errors fails the item like any other generator
error.

## `random`: the control

Every score is an independent Uniform(0, 1) draw that ignores the text. Resampling then
copies particles at random, and the answer still comes from voting over the particles.
Run it beside a real scorer with the same method, generator and budgets. The difference
between the two is what the scorer's signal is worth. Without it, gains over b = 1 cannot
be told apart from what resampling and majority voting do on their own.

Draws are seeded by `(--seed, unique_id, round)`, so a resumed item gets the same
scores. The random scores are not stored in the row (`judge_scores_all` is judge-only);
the per-round values are in the trace.

The other control, no resampling at all (N independent samples), needs no scorer
change: `--resample-from-step 1000` never resamples.

## Scores and the weight

The scorer only produces s. How s becomes a weight (`last`/`prod`/`min`/`mean`, then
`log`/`logit`, then / T) is the method's business: [weight-temperature.md](weight-temperature.md).
Every per-step judge score is stored in the row as `judge_scores_all`, and every probe
distribution as `probe_dists_all`, so weights under another aggregation can be
recomputed offline. The resampling outcome cannot be recomputed this way, because it
depended on the weights used at run time.
