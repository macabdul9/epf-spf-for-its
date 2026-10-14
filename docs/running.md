# Running

## Install

```bash
cd epf-spf-its-reimplementation
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install -e ".[dev]"            # add ,legacy to run the parity tests
```

Runtime dependencies: numpy, openai, pandas, pyarrow, soundfile. No GPU code runs in this
package. It talks to vLLM servers over the OpenAI-compatible API. How to serve each
generator and judge is in the parent tree's `env-setup.md`.

## Before a real run

1. Serve the generator (and the judge, for `--scorer judge`).
2. Run the exact command with `--dry-run`. It prints the resolved config, the run
   settings, the fingerprint and the record count, then stops without sending any
   request.
3. Check the printed `config` line, especially `weight_temperature`,
   `weight_transform` and `score_aggregation`.
4. Try `--limit 2 --budgets 2` into a scratch output file and look at one row.

## Command-line reference

### Method

| flag | values | default |
|---|---|---|
| `--method` | `epf-paper`, `epf-itshub`, `epf-project`, `spf` | required |
| `--scorer` | `judge`, `probe`, `random` (control) | preset |
| `--weight-transform` | `log`, `logit` | preset |
| `--weight-temperature` | float > 0 | preset |
| `--score-aggregation` | `last`, `prod`, `min`, `mean` | preset |
| `--resample-from-step` | int ≥ 1 | preset |
| `--resample-rule` | `every_step`, `ess_gate` | preset |
| `--gate-threshold` | float, needed by `ess_gate` | preset |
| `--annealing` / `--no-annealing` | | preset (on in all four) |
| `--anneal-ess-threshold` | (0, 1] | preset |
| `--anneal-window` | (0, 1], fraction of `--max-steps` | preset |
| `--resampling-method` | `systematic`, `multinomial` | preset |
| `--on-score-failure` | `fail`, `neutral` | `fail` |

Presets and their values: [methods-overview.md](methods-overview.md). An invalid
combination (for example `--resample-rule ess_gate` without a gate threshold) fails
before anything loads.

### Data

| flag | meaning | default |
|---|---|---|
| `--bench` | one of the 13 benchmarks in [data.md](data.md) | required |
| `--data-root` | folder holding `mmar/`, `mmau_testmini/`, `mmau_pro_d1k/` | `$EPF_DATA_ROOT` |
| `--extended-dir` | `extended_dataset/distractor_results` | `$EPF_EXTENDED_DIR` |
| `--no-audio-override` | use the original clips, not `<data-root>/_audio_override_30s` | override on if the folder exists |
| `--limit` | first k records (after offset) | all |
| `--item-offset`, `--item-count` | a contiguous slice, for sharding | 0, all |

### Generator

| flag | meaning | default |
|---|---|---|
| `--generator-endpoint` | e.g. `http://127.0.0.1:9401/v1` | required |
| `--generator-model` | served model name; Phi-4-multimodal must be `speech` (its LoRA) | required |
| `--sampling-temperature` | | 0.8 |
| `--max-steps` | step cap and annealing horizon | 10 |
| `--max-step-tokens` | tokens per step | 300 |
| `--min-tokens` | vLLM `min_tokens` on continuation steps | none |
| `--inline-step-stop` | also stop at `" Step "`, for models that chain steps on one line | off |
| `--think-trigger` | append "Please think and reason about the input {music,speech,sound,audio} before you respond." | off |

### Judge (scorer `judge` only)

| flag | default |
|---|---|
| `--judge-endpoint` | required with `--scorer judge` |
| `--judge-model` | required with `--scorer judge` |
| `--judge-concurrency` | 48 concurrent judge requests |

### Run

| flag | meaning | default |
|---|---|---|
| `--budgets` | comma-separated particle counts, run in the order given | required |
| `--output` | JSONL file; a `.manifest.json` is written next to it | required |
| `--max-inflight` | particles in flight; items run `max(1, max_inflight // budget)` at a time | 192 |
| `--seed` | | 1234 |
| `--store-texts` / `--no-store-texts` | store full trajectories and the per-step trace | on |
| `--dry-run` | resolve and load, then stop | |

## Examples

Shared variables used below:

```bash
export EPF_DATA_ROOT=/path/to/epf_data
export EPF_EXTENDED_DIR=/path/to/extended_dataset/distractor_results
GEN="--generator-endpoint http://127.0.0.1:9401/v1 --generator-model qwen-omni"
JUDGE="--judge-endpoint http://127.0.0.1:9311/v1 --judge-model qwen-omni-3b-judge"
B=1,2,4,8,16,32,64
```

```bash
# the paper's ePF
epf-spf --method epf-paper   --bench mmar $GEN $JUDGE --budgets $B --output runs/mmar__qwen-omni__qwen3b__epf-paper.jsonl

# its_hub's ePF
epf-spf --method epf-itshub  --bench mmar $GEN $JUDGE --budgets $B --output runs/mmar__qwen-omni__qwen3b__epf-itshub.jsonl

# the project's judge arm at T = 3, and at T = 1
epf-spf --method epf-project --bench mmar $GEN $JUDGE --budgets $B --output runs/mmar__qwen-omni__qwen3b__epf-project-T3.jsonl
epf-spf --method epf-project --weight-temperature 1 \
                             --bench mmar $GEN $JUDGE --budgets $B --output runs/mmar__qwen-omni__qwen3b__epf-project-T1.jsonl

# SPF (no judge)
epf-spf --method spf         --bench mmar $GEN        --budgets $B --output runs/mmar__qwen-omni__spf.jsonl
```

Put every setting that differs between runs in the file name. The file itself also
records it, but the name is what people read.

## Resuming

Re-run the same command. Cells (`unique_id`, `budget`) that already have a non-error
row are skipped; error rows are retried. Each item's random numbers depend only on
`(seed, unique_id, budget)`, so a resumed item draws exactly what it would have drawn in
an uninterrupted run. Generation itself is sampled by the server, so a re-run item does
not reproduce the same text.

Resuming with any setting that changes results (any method field, models, sampling,
step format, benchmark, audio override, seed) stops immediately:

```text
runs/x.jsonl contains rows from a different configuration (fingerprints [...], this run ...).
Write to a new file; one output file holds one configuration.
```

Settings that do not change results are not in the fingerprint and may differ on
resume: `--limit`, `--item-offset`/`--item-count`, `--max-inflight`,
`--judge-concurrency`, `--store-texts`, endpoints.

Error rows stay in the file after a successful retry. When reading results, drop rows
with `error` set. What remains has at most one row per cell, because a cell with a good
row is skipped.

## Sharding

Split the items across processes with `--item-offset`/`--item-count`, **one output file
per shard**. Two processes must never append to the same file: the skip-done check reads
the file once at start, so two writers would both run the same cells.

```bash
epf-spf ... --item-offset 0   --item-count 250 --output runs/mmar__...__shard0.jsonl
epf-spf ... --item-offset 250 --item-count 250 --output runs/mmar__...__shard1.jsonl
```

All shards of one configuration have the same fingerprint, so they can be concatenated
afterwards.

## Concurrency and load

At budget b, `max_inflight // b` items run at once, and each has up to b particles
generating together. Judge requests are capped separately by `--judge-concurrency`. If
the judge falls behind, its requests time out (300 s, 3 retries). With the default
`--on-score-failure fail`, the item becomes an error row to be retried, not a silently
unguided result. Lower `--max-inflight` or `--judge-concurrency` if `judge calls N (M
failed)` at the end of a run shows failures.

## Logs

The shape of the log (values in angle brackets vary):

```text
method=epf-project fingerprint=<12 hex chars>
config {<every FilterConfig field>}
run {<benchmark, models, sampling, step format, audio override, seed>}
<records> records (mmar), budgets [1, 8, 64]
[epf-project b8] <to do> to do, 24 concurrent items
  done in <seconds>s (<errors> errors)
judge calls <calls> (<failed> failed)
budget 8 (n=<rows>): <selector>=<accuracy> ... (every selector, alphabetical)
```
