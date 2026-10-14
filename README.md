# epf-spf-its-reimplementation

Particle filtering over the reasoning steps of an audio-language model answering
multiple-choice questions. This repository implements, with one shared loop and one
configuration object:

- **ePF as the paper describes it**: Entropic Particle Filtering, from the algorithm
  block, experimental setup and hyper-parameter table of the ePF paper
- **ePF as `its_hub` implements it**: the authors' reference code,
  `EntropicParticleFiltering` in `its_hub` at commit 17f7813
- **the project's judge-PRM arm** with the weight temperature T as a flag (T = 3 by
  default, any value from the command line)
- **SPF**: Self-Probing Particle Filtering, which needs no reward model. The
  generator's own forced answer is the score.

The score source is independent of the method. Any method runs with the external judge
(`--scorer judge`), the self-probe (`--scorer probe`) or random scores as a control
(`--scorer random`).

Every setting that changes results is printed at startup, written into every result
row, and covered by a run fingerprint. An output file can only hold rows of one
configuration.

The repository is standalone. It reads data only from paths you pass in, and it talks to
models only through the OpenAI-compatible endpoints you give it.

## Contents

- [The methods](#the-methods)
- [Install](#install)
- [Data](#data)
- [Model servers](#model-servers)
- [Running](#running)
- [Outputs](#outputs)
- [Analysing results](#analysing-results)
- [The tuning experiment](#the-tuning-experiment)
- [Tests](#tests)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)

## The methods

A run handles one question at a time with N particles (the budget). Each round:

1. every unfinished particle generates its next step (`Step k: ...`). A particle
   finishes when its step contains `Answer:`, or when it already had `--max-steps`
   steps.
2. the scorer gives each particle that just stepped a score s in [0, 1]
3. the scores so far become a log-weight: an aggregation (`last`, `prod`, `min`,
   `mean`), then `log(s) / T` or `logit(s) / T`
4. resampling, unless the step is before `resample_from_step`:
   - p = softmax(log-weights)
   - entropic annealing: if ESS_n(p) < threshold and step/max_steps < window, flatten
     with β⁻¹ = max(1, (1 − t/T) / ESS_n)
   - SPF only: skip the draw unless the annealed ESS_n < the gate
   - draw N particles (systematic or multinomial)

The four presets (`--method`):

| field | `epf-paper` | `epf-itshub` | `epf-project` | `spf` |
|---|---|---|---|---|
| scorer | judge | judge | judge | **probe** |
| weight transform | **log** | logit | logit | logit |
| weight temperature T | 1 | 1 | **3** | **3** |
| score aggregation | last | **prod** | last | last |
| first resampling after step | **2** | 1 | 1 | 1 |
| resampling rule | every step | every step | every step | **ESS gate 0.5** |
| annealing threshold / window | 0.5 / 0.5 | 0.5 / 0.5 | **0.6 / 0.7** | **0.6 / 0.7** |
| resampling | systematic | systematic | systematic | systematic |

Where each value comes from, and what each one does: [docs/methods-overview.md](docs/methods-overview.md).

Any field can be overridden with a flag of the same name:

```bash
--method epf-project --weight-temperature 1        # the judge arm at T = 1
--method epf-paper --weight-temperature 3          # the paper's log rule, flattened
--method epf-itshub --score-aggregation last       # its_hub weighting, newest step only
--method spf --resample-rule every_step            # SPF without the gate
--method spf --scorer judge                        # the SPF rule with the judge's score
--method epf-paper --scorer random                 # control: random scores
--method epf-paper --resample-from-step 1000       # control: no resampling at all
```

The row records both the preset name (`method`) and the resolved values (`config`).

### Scorers

| `--scorer` | score s | needs |
|---|---|---|
| `judge` | a zero-shot audio LM is asked whether the latest step is correct; s = P(Yes) / (P(Yes) + P(No)) from the top-20 logprobs of its first token. A finished particle is judged on its whole trajectory instead. | a judge endpoint |
| `probe` | the generator is interrupted with `Answer:` and continued greedily; s = the probability mass on its most likely option letter | nothing extra; uses the generator endpoint |
| `random` | an independent Uniform(0, 1) draw per particle and step, seeded per item | nothing |

A judge call that fails after retries stops the item by default
(`--on-score-failure fail`). `--on-score-failure neutral` scores it 0.5 and counts it.
Details: [docs/scorers.md](docs/scorers.md).

## Install

Python 3.10 or newer. The environment is about 350 MB. Put it wherever there is space;
it does not have to be inside the repo.

```bash
ENV_DIR=/path/to/envs/epf_spf
uv venv "$ENV_DIR" --python 3.12
VIRTUAL_ENV="$ENV_DIR" uv pip install -e ".[dev]"     # run from the repo root
```

uv keeps a download cache in `~/.cache/uv` by default. Set `UV_CACHE_DIR` if your home
is small. The command-line entry point is `$ENV_DIR/bin/epf-spf`.

Runtime dependencies: numpy, openai, pandas, pyarrow, soundfile. No GPU code runs in
this package.

## Data

Paths are passed by flag or environment variable. There are no defaults.

| | flag | environment variable | holds |
|---|---|---|---|
| data root | `--data-root` | `EPF_DATA_ROOT` | `mmar/`, `mmau_testmini/`, `mmau_pro_d1k/` (only the folders of the benchmarks you run) |
| extended dataset | `--extended-dir` | `EPF_EXTENDED_DIR` | the `distractor_results/` folder: `mmar_augmented.jsonl`, `mmau_augmented.jsonl`, `mmau-pro-d1k_augmented.jsonl` and the CSVs (`mmau-pro-d1k.csv`, ...) |

If `<data root>/_audio_override_30s/` exists, clips with the same basename replace the
originals (for servers that reject audio over 30 s). This is recorded in every row.
`--no-audio-override` turns it off.

Benchmarks (`--bench`):

| benchmark | items | options |
|---|---|---|
| `mmar`, `mmau`, `mmau_pro_d1k` | 996, 1000, 991 | original |
| `mmar_ext`, `mmau_ext`, `mmau_ext_fixed`, `mmau_pro_d1k_ext` | 833, 869, 869, 583 | 10 (distractors added) |
| `mmar_sub`, `mmau_sub`, `mmau_pro_d1k_sub` | 832, 869, 583 | original, restricted to the extended items |
| `mmar_csv_orig`, `mmau_csv_orig`, `d1k_csv_orig` | 832, 869, 583 | original options and gold from the extended CSVs, per-item shuffled order |

Details: [docs/data.md](docs/data.md).

## Model servers

The package talks to vLLM (or any server with the same OpenAI-compatible extensions).
Requirements:

- **Generator:** chat completions with audio input; `continue_final_message` and
  `add_generation_prompt` in `extra_body` (vLLM); top-20 logprobs (needed by the
  probe). `--generator-model` is the served model name. Phi-4-multimodal must be
  addressed by its LoRA name, `speech`.
- **Judge:** chat completions with audio input and top-20 logprobs. Audio is sent as
  `file://` URLs of the clips' real paths (symlinks resolved), so the server's
  `--allowed-local-media-path` must cover them.

Models used in the project:

| role | model | served name used so far |
|---|---|---|
| generator | Qwen/Qwen2.5-Omni-7B | `qwen-omni` |
| generator | Qwen/Qwen2.5-Omni-3B | `qwen-omni-3b` |
| judge | Qwen/Qwen3-Omni-30B-A3B-Instruct | `Qwen/Qwen3-Omni-30B-A3B-Instruct` (one 80 GB GPU) |
| judge | Qwen/Qwen2.5-Omni-3B, Qwen/Qwen2.5-Omni-7B, mispeech/midashenglm-0.6b | `qwen-omni-3b-judge`, `qwen-omni-7b-judge`, `midashenglm-0.6b` |

## Running

Always dry-run first. A dry run prints the resolved configuration, run settings,
fingerprint and record count, then stops without sending any request.

```bash
export EPF_DATA_ROOT=/path/to/epf_data
export EPF_EXTENDED_DIR=/path/to/extended_dataset/distractor_results

epf-spf --method epf-paper --bench d1k_csv_orig \
    --generator-endpoint http://127.0.0.1:9401/v1 --generator-model qwen-omni \
    --judge-endpoint http://127.0.0.1:8801/v1 --judge-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --budgets 1,2,4,8,16,32,64,128 \
    --output /path/to/results/d1k__qwen-omni__epf-paper.jsonl \
    --dry-run
```

Drop `--dry-run` to run. SPF needs no judge:

```bash
epf-spf --method spf --bench d1k_csv_orig \
    --generator-endpoint http://127.0.0.1:9401/v1 --generator-model qwen-omni \
    --budgets 1,2,4,8,16,32,64,128 \
    --output /path/to/results/d1k__qwen-omni__spf.jsonl
```

Commonly used run flags:

| flag | default | |
|---|---|---|
| `--budgets` | required | comma-separated particle counts |
| `--output` | required | JSONL file; `<output>.manifest.json` is written next to it |
| `--max-steps` | 10 | step cap, and the horizon of the annealing schedule |
| `--sampling-temperature` | 0.8 | generator sampling |
| `--max-step-tokens` | 300 | |
| `--max-inflight` | 192 | particles in flight; items run `max(1, max_inflight // budget)` at a time |
| `--judge-concurrency` | 48 | concurrent judge requests |
| `--seed` | 1234 | resampling and random-score draws, per item |
| `--limit`, `--item-offset`, `--item-count` | | subsets and shards |
| `--store-texts` / `--no-store-texts` | on | full trajectories and the per-step trace |

Every flag: [docs/running.md](docs/running.md).

**Resuming:** run the same command again. Finished (item, budget) cells are skipped and
error rows are retried. Each item's random numbers depend only on (seed, item, budget),
so a resumed item draws the same numbers.

**One file, one configuration:** if the output file holds rows written under any other
setting that changes results, the run stops before sending anything. Write to a new
file instead.

**Sharding:** split items with `--item-offset` / `--item-count`, one output file per
shard. Never point two processes at the same output file.

## Outputs

One JSON line per (item, budget):

| field | |
|---|---|
| `unique_id`, `budget`, `gold`, `n` | the cell |
| `pred_letters`, `final_weights`, `steps_used`, `texts_hash` | per particle |
| `judge_scores_all` / `probe_dists_all` | per particle, per step (judge / probe) |
| `texts`, `trace` | with `--store-texts`: full trajectories, and per round the scores, log-weights, annealing temperature, ESS before and after annealing, and the resampling parents |
| `method`, `config`, `run_fingerprint`, run settings, `code_version` | provenance |
| `error` | `null`, or the error for a failed cell (retried on resume) |

With texts stored, expect about 230 MB per 583 items for budgets 1–64 with a judge
(measured on earlier runs).

Details: [docs/output-format.md](docs/output-format.md).

## Analysing results

Every answer-selection rule is computed from a stored row, so the rule is chosen at
analysis time:

```python
import json
from epf_spf.selection import select_answers

rows = [json.loads(line) for line in open("results.jsonl")]
rows = [r for r in rows if r.get("error") is None]
assert len({r["run_fingerprint"] for r in rows}) == 1

for r in rows[:3]:
    picked = select_answers(r)              # {"argmax_w": "B", "dedup_majority": "C", ...}
    print(r["budget"], r["gold"], picked["dedup_majority"])
```

Main selectors: `argmax_w` (highest final weight, as the paper and its_hub select),
`majority`, `dedup_majority` (one vote per distinct trajectory), `judge_marginal`,
`probe_marginal`. All of them: [docs/answer-selection.md](docs/answer-selection.md).

At the end of a run, `epf-spf` logs the accuracy of every selector per budget.

## The tuning experiment

`tuning/` holds the experiment that picks the ePF and SPF settings before the full
re-run: Qwen2.5-Omni-7B and -3B generators, the Qwen3-Omni-30B judge, the 583 D1K items
of `mmau-pro-d1k.csv`, budgets 1–128.

| file | |
|---|---|
| [tuning/PLAN.md](tuning/PLAN.md) | the plan: the 28 settings, servers, setup, running, checking, choosing the winner, cost |
| [tuning/run_tuning.sh](tuning/run_tuning.sh) | runs settings for one generator: `tuning/run_tuning.sh qwen-omni E1 E2 E3` (`all`, `list`; `DRY_RUN=1` to check) |
| [tuning/check_runs.py](tuning/check_runs.py) | checks result files (one fingerprint, no duplicates, rows per budget) and prints accuracy |
| [tuning/stage0/](tuning/stage0/REPORT.md) | the analysis of stored runs that motivated the settings |
| [tuning/sync_to_babel.sh](tuning/sync_to_babel.sh) | copies this repo to Babel (`DRY_RUN=1` first) |

## Tests

```bash
pytest                       # unit and behaviour tests; parity tests are skipped
ruff check . && ruff format --check .
```

Run from the repo root inside the environment. No test sends a request or needs a
GPU.

The parity tests compare this code with the previous implementation (prompts, judge
prompts, step format, score extraction, annealing, systematic resampling, answer
extraction, selectors, all 13 loaders). They are **opt-in**:

```bash
VIRTUAL_ENV="$ENV_DIR" uv pip install -e ".[dev,legacy]"
LEGACY_EPF_ROOT=/path/to/entropic-particle-filter \
EPF_DATA_ROOT=/path/to/epf_data \
EPF_EXTENDED_DIR=/path/to/extended_dataset/distractor_results \
pytest
```

What each test file guards: [docs/testing.md](docs/testing.md).

## Repository layout

```text
src/epf_spf/
  config.py       FilterConfig, the four presets, validation
  filter.py       the particle filter loop (one loop for every method)
  weighting.py    score -> log-weight: aggregation, log/logit, temperature
  resampling.py   softmax, ESS, entropic annealing, the SPF gate, systematic/multinomial
  scorers.py      JudgeScorer, ProbeScorer, RandomScorer
  steps.py        the "Step N:" format and the stop rule
  llm.py          OpenAI-compatible generator client
  prompts.py      generator and judge prompts
  data.py         benchmark loaders
  answers.py      answer extraction
  selection.py    answer selectors
  results.py      result rows, resume, provenance, manifest
  run.py          the epf-spf command
tests/            unit, behaviour and (opt-in) parity tests
docs/             method and usage documentation
tuning/           the tuning experiment
```

## Documentation

| | |
|---|---|
| [docs/index.md](docs/index.md) | start here |
| [docs/methods-overview.md](docs/methods-overview.md) | the four methods side by side |
| [docs/epf-paper.md](docs/epf-paper.md) | ePF as the paper describes it |
| [docs/epf-its-hub.md](docs/epf-its-hub.md) | ePF as its_hub implements it, with line references |
| [docs/weight-temperature.md](docs/weight-temperature.md) | `log` vs `logit`, the weight temperature, measured effects |
| [docs/spf.md](docs/spf.md) | Self-Probing Particle Filtering |
| [docs/scorers.md](docs/scorers.md) | judge, probe, random |
| [docs/answer-selection.md](docs/answer-selection.md) | the selectors |
| [docs/running.md](docs/running.md) | every flag, resuming, sharding |
| [docs/output-format.md](docs/output-format.md) | rows, trace, provenance |
| [docs/data.md](docs/data.md) | benchmarks and the audio override |
| [docs/parity-and-differences.md](docs/parity-and-differences.md) | what matches the previous code, what changed, what is not implemented |
| [docs/testing.md](docs/testing.md) | the tests |

Not implemented: the paper's Look-ahead Modulation (its_hub does not implement it
either), linear and entropy annealing schedules, and prompt methods other than the
stepwise "method 12" prompt.
