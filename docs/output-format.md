# Output format

A run appends one JSON object per line to `--output`, one line per (item, budget) cell.
Next to it, `<output>.manifest.json` records how the run was started.

## A result row

| field | type | meaning |
|---|---|---|
| `unique_id` | str | item id from the benchmark |
| `budget` | int | N, the number of particles |
| `gold` | str | gold letter |
| `n` | int | number of particles at the end (= budget) |
| `pred_letters` | list[str] | per particle, the letter read from its text; `""` if none |
| `final_weights` | list[float] | per particle, the final log-weight (4 decimals) |
| `steps_used` | list[int] | per particle, number of steps (up to `max_steps + 1`) |
| `texts_hash` | list[str] | per particle, first 12 hex characters of sha1(text); equal hashes = copies of one trajectory |
| `probe_dists` | list[dict \| null] | per particle, the last non-empty probe letter distribution (probe scorer) |
| `probe_dists_all` | list[list[dict \| null]] | per particle, per step, the probe letter distribution |
| `judge_scores_all` | list[list[float \| null]] | judge scorer only: per particle, per step, the judge's P(Yes) (4 decimals); `null` for a failed call under `neutral` |
| `score_failures` | int | judge calls that failed in this cell (non-zero only under `--on-score-failure neutral`) |
| `audio_paths` | list[str] | the clips the generator heard (after the 30-second override) |
| `latency_s` | float | wall time for the cell |
| `error` | null | `null` on a result row |
| `texts` | list[str] | with `--store-texts`: each particle's full trajectory |
| `trace` | list[dict] | with `--store-texts`: one entry per round, below |
| provenance fields | | below |

Particles are stored in their final order, after the last resampling. A particle's
history (`scores`, `probe_dists_all`, `judge_scores_all`) is the history of its lineage:
when a particle is copied by resampling, the copy inherits the parent's history.

## Provenance (every row, including error rows)

| field | meaning |
|---|---|
| `method` | the preset name passed to `--method` |
| `config` | every `FilterConfig` field after overrides: `scorer`, `weight_transform`, `weight_temperature`, `score_aggregation`, `resample_from_step`, `resample_rule`, `gate_threshold`, `annealing`, `anneal_ess_threshold`, `anneal_window`, `resampling_method`, `on_score_failure` |
| `benchmark` | `--bench` |
| `generator_model`, `judge_model` | served names (`judge_model` is null for the probe) |
| `sampling_temperature`, `max_steps`, `max_step_tokens`, `min_tokens`, `inline_step_stop`, `think_trigger` | generation settings |
| `audio_override_dir` | the override folder used, or null |
| `seed` | |
| `run_fingerprint` | first 12 hex characters of sha256 over `config` and all the run settings above |
| `code_version` | `{package, git_commit, git_dirty}` |

`run_fingerprint` covers everything that changes results and nothing that does not.
`code_version` is excluded from it, so a run can be resumed after a fix that does not
change behaviour. `git_dirty: true` means the code had uncommitted changes when the row
was written.

**One file, one fingerprint.** Starting a run whose fingerprint differs from rows
already in the output file stops with an error before any request is sent. A file
written by this code therefore never mixes configurations. To find the setting of any
row, read its `config`; nothing has to be inferred from the file name.

## An error row

```json
{"unique_id": "...", "budget": 8, "error": "ScoreFailure: judge call failed at step 3",
 "method": "...", "config": {...}, "run_fingerprint": "...", ...}
```

Error rows carry provenance but no results. They are retried on resume and must be
dropped before analysis.

## The trace

One entry per round (step) of the filter:

```json
{
  "step": 3,
  "particles": [
    {"new_text": " the second tone is higher ...", "finished": false, "score": 0.81, "log_weight": 0.4834},
    {"new_text": null, "finished": true, "score": null, "log_weight": 1.2001}
  ],
  "resample": {
    "annealing_temperature": 1.0,
    "ess_ratio_untempered": 0.912,
    "ess_ratio": 0.912,
    "probabilities": [0.031, 0.029, "..."],
    "applied": true,
    "parents": [0, 0, 2, 3, "..."]
  }
}
```

| key | meaning |
|---|---|
| `particles[i].new_text` | the step particle i generated this round; `null` if it had already finished |
| `particles[i].score` | this round's score; `null` if it did not run, or the score was missing |
| `particles[i].log_weight` | the log-weight that entered resampling |
| `resample.annealing_temperature` | β⁻¹; 1.0 when annealing did not fire |
| `resample.ess_ratio_untempered` | ESS_n of softmax(log-weights), before annealing |
| `resample.ess_ratio` | ESS_n of the distribution actually drawn from |
| `resample.probabilities` | that distribution |
| `resample.applied` | whether particles were resampled this round |
| `resample.parents` | when applied: new particle j is a copy of old particle `parents[j]` |
| `resample.reason` | when not applied: `before resample_from_step` or `ESS gate` |

Particle indices refer to the order **before** this round's resampling. The next round's
indices refer to the order after it. Following `parents` backwards from the last round
recovers each final particle's ancestry.

A round before `resample_from_step` has only `applied` and `reason` in `resample`.

## The manifest

`<output>.manifest.json` is rewritten at every start (including resumes) with the
provenance fields, the full argument list, and the start time. It is a convenience for
humans; the rows are the record.

## Reading results

```python
import json
from epf_spf.selection import select_answers

rows = [json.loads(line) for line in open("runs/x.jsonl")]
rows = [r for r in rows if r.get("error") is None]
assert len({r["run_fingerprint"] for r in rows}) == 1
assert len({(r["unique_id"], r["budget"]) for r in rows}) == len(rows)

acc = {}
for r in rows:
    picked = select_answers(r)["argmax_w"]
    acc.setdefault(r["budget"], []).append(picked == r["gold"])
print({b: sum(v) / len(v) for b, v in sorted(acc.items())})
```
