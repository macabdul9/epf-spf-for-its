# Tuning experiments: ePF and SPF on MMAU-Pro D1K

Goal: find the best ePF setting (with the Qwen3-Omni-30B-A3B judge) and the best SPF
setting, for the Qwen2.5-Omni-7B and Qwen2.5-Omni-3B generators, before the full
re-run.

Everything here runs with the `epf-spf` package of this repo. Paths below are relative
to the repo root. Background on the methods: `docs/`. Why these settings:
`tuning/stage0/REPORT.md`.

## 1. What is fixed

| | value |
|---|---|
| items | the 583 items of `extended_dataset/distractor_results/mmau-pro-d1k.csv` |
| options | the CSV's original options (`mcq_original_options`) and gold (`ground_truth`), per-item shuffled order (`--bench d1k_csv_orig`); audio from the D1K data root |
| budgets | 1, 2, 4, 8, 16, 32, 64, 128 |
| generators (solvers) | `qwen-omni` (Qwen2.5-Omni-7B), `qwen-omni-3b` (Qwen2.5-Omni-3B) |
| judge (verifier, ePF only) | `Qwen/Qwen3-Omni-30B-A3B-Instruct`, revision `26291f79` |
| prompt, step format | method 12, `--max-steps 10`, `--max-step-tokens 300` (package defaults) |
| generator sampling | temperature 0.8 (package default) |
| audio | the 30-second override clips are used (package default); none affect D1K |
| seed | 1234 (package default) |

The items, options and gold come from
`extended_dataset/distractor_results/mmau-pro-d1k.csv` (`d1k_csv_orig`). The CSV
holds no audio; the clips come from `<data root>/mmau_pro_d1k/`. The option order
is shuffled per item, as in the earlier small-judge D1K runs. The stored 30B D1K
runs used the original order instead (548 of 583 items differ in order, and 1 item
has a different option set). Accuracy is comparable per item; predicted letters
are not.

## 2. The 28 runs

14 settings per generator. Each setting is one output file. The ids are what
`tuning/run_tuning.sh` takes.

| id | group | flags (on top of the fixed settings) | needs 30B judge | what it tests |
|---|---|---|---|---|
| C1 | control | `--method epf-project --scorer random` | no | resampling with random scores (logit, T 3, the project's annealing) |
| C2 | control | `--method epf-project --scorer random --resample-from-step 1000` | no | no resampling: N independent samples |
| E1 | ePF | `--method epf-paper` | yes | the paper: log s, step 1 not resampled, annealing 0.5 / first 50% |
| E2 | ePF | `--method epf-itshub` | yes | its_hub: logit s, product over steps, annealing 0.5 / 50% |
| E3 | ePF | `--method epf-project` | yes | the project's arm: logit s / 3, annealing 0.6 / 70% (the setting of the stored 30B runs) |
| E4 | ePF | `--method epf-project --resample-from-step 2` | yes | E3, first resampling after step 2 |
| E5 | ePF | `--method epf-project --resample-from-step 3` | yes | E3, first resampling after step 3 |
| E6 | ePF | `--method epf-project --weight-temperature 1` | yes | E3 at T = 1 |
| S1 | SPF | `--method spf` | no | the SPF preset: T 3, gate 0.5 (the setting of the stored SPF runs) |
| S2 | SPF | `--method spf --gate-threshold 0.7` | no | resample more often |
| S3 | SPF | `--method spf --resample-rule every_step` | no | no gate |
| S4 | SPF | `--method spf --weight-temperature 1` | no | sharper weights, gate 0.5 |
| S5 | SPF | `--method spf --weight-temperature 1 --gate-threshold 0.7` | no | sharper, gate 0.7 |
| S6 | SPF | `--method spf --weight-temperature 1 --resample-rule every_step` | no | sharper, no gate |

Why these (from Stage 0): the 30B judge's step-1 score carries no signal, so E4 and E5
delay resampling. The stored SPF runs barely resampled, so S2–S6 make SPF resample
more. The latest-step score separated best, so aggregation is not swept except as part
of E2.

At budget 1 there is nothing to resample, so b1 is a plain sample in every setting.
It is kept in every file anyway so each file is complete on its own; it costs 583
trajectories.

## 3. Servers

| server | model | served name | GPU memory | notes |
|---|---|---|---|---|
| generator | Qwen2.5-Omni-7B | `qwen-omni` | one GPU | |
| generator | Qwen2.5-Omni-3B | `qwen-omni-3b` | one GPU | |
| judge | Qwen3-Omni-30B-A3B | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | one 80 GB GPU, `--gpu-memory-utilization 0.92` | vLLM 0.22.1 needs `--mm-encoder-attn-backend TORCH_SDPA`; first start on a new GPU type spends about 25 min compiling |

Serving commands, pinned revisions and environments: `env-setup.md` in the parent project
tree on the original machine (not part of this repo)
(section 4.1 for the two generators, 5.1 for the 30B judge, 6 for GPU memory, 8 for checking a server). Generators must be served with
logprobs available (top 20), which SPF's probe needs. The judge's
`--allowed-local-media-path` must cover the **realpath** of the audio files.

A layout that fits one 4-GPU node: GPU 0 judge for the Omni-7B chain, GPU 1 judge for
the Omni-3B chain, GPU 2 `qwen-omni`, GPU 3 `qwen-omni-3b`. The C and S settings use
only the generator servers.

## 4. Setup on a new machine

To Babel: `tuning/sync_to_babel.sh` copies this repo (run it with `DRY_RUN=1` first). It
does not copy data: point `EPF_DATA_ROOT` and `EPF_EXTENDED_DIR` at Babel's own copy.
Then continue below on Babel.

```bash
REPO=/path/to/epf-spf-its-reimplementation          # this repo
ENV_DIR=/path/outside/home/epf_spf                  # ~350 MB; keep it off a small home
export UV_CACHE_DIR=/path/outside/home/uv-cache     # uv's cache also defaults to home
uv venv "$ENV_DIR" --python 3.12
VIRTUAL_ENV="$ENV_DIR" uv pip install -e "$REPO"
cd "$REPO"
```

Data the runs need: a data root holding `mmau_pro_d1k/` (parquet and audio), and the
extended dataset's `distractor_results/` folder (holding `mmau-pro-d1k.csv`). Check that it loads before
starting any server:

```bash
export EPF_SPF="$ENV_DIR/bin/epf-spf"
export EPF_DATA_ROOT=/path/to/epf_data                       # holds mmau_pro_d1k/
export EPF_EXTENDED_DIR=/path/to/extended_dataset/distractor_results
export OUT_ROOT=/path/with/space/for/results                  # about 13 GB for all 28 runs (estimate)
export GEN_ENDPOINT=http://127.0.0.1:9401/v1       # anything; not contacted in a dry run
export JUDGE_ENDPOINT=http://127.0.0.1:8801/v1

DRY_RUN=1 tuning/run_tuning.sh qwen-omni all
```

Every setting must print `583 records (d1k_csv_orig), budgets [1, 2, 4, 8, 16, 32,
64, 128]` and a `config` line matching the table above. A dry run writes no result file.

## 5. Running

`tuning/run_tuning.sh <generator> <ids...>` runs the given settings one after another,
against `GEN_ENDPOINT` (and `JUDGE_ENDPOINT` for E settings). Outputs:

```text
$OUT_ROOT/d1k/<generator>/<id>.jsonl                 results, one row per (item, budget)
$OUT_ROOT/d1k/<generator>/<id>.jsonl.manifest.json   how the run was started
$OUT_ROOT/d1k/<generator>/logs/<id>.log              log
```

Example for one node with the layout above (four chains, one per line, each in its own
shell or `tmux` window):

```bash
GEN_ENDPOINT=http://127.0.0.1:9401/v1 JUDGE_ENDPOINT=http://127.0.0.1:8801/v1 tuning/run_tuning.sh qwen-omni    E1 E2 E3 E4 E5 E6
GEN_ENDPOINT=http://127.0.0.1:9403/v1 JUDGE_ENDPOINT=http://127.0.0.1:8802/v1 tuning/run_tuning.sh qwen-omni-3b E1 E2 E3 E4 E5 E6
GEN_ENDPOINT=http://127.0.0.1:9401/v1 tuning/run_tuning.sh qwen-omni    C1 C2 S1 S2 S3 S4 S5 S6
GEN_ENDPOINT=http://127.0.0.1:9403/v1 tuning/run_tuning.sh qwen-omni-3b C1 C2 S1 S2 S3 S4 S5 S6
```

The C/S chains share the generator servers with the E chains, which slows both. To
keep the E chains at full speed, run the C/S chains after them or on other GPUs.

Rules:

- **One process per output file.** Never start the same `<generator> <id>` in two
  places at once. Settings can be split across machines freely, because each id is its
  own file.
- **Resume = run the same command again.** Finished (item, budget) cells are skipped;
  error rows are retried.
- **Never change a setting into an existing file.** A file holds exactly one
  configuration. If the settings differ, `epf-spf` stops before sending anything.
- Results from several machines are combined by copying the `<id>.jsonl` files into
  one folder. Don't concatenate two copies of the same id that both ran.

## 6. Checking results

```bash
"$ENV_DIR/bin/python" tuning/check_runs.py \
    "$OUT_ROOT/d1k/qwen-omni"
```

For each setting this prints whether the file has a single fingerprint and no
duplicate cells, how many error rows there are, and, per budget, either the accuracy
(when all 583 items are done) or the count done so far. Before using any number:

- every budget shows an accuracy, not `n/583`
- `ok`, i.e. one fingerprint and no duplicate cells
- error rows are only a problem if their cells never got a good row; they show up as
  missing counts
- the run log's last lines say `judge calls N (0 failed)` for E settings

## 7. Choosing the winner

Decide these before looking at the numbers:

- **Selector:** `dedup_majority` (majority vote over distinct trajectories) is the main
  number for every setting. `argmax_w` is reported alongside for E1 and E2, because the
  paper and its_hub select by weight. All selectors are stored, so others can be
  computed later.
- **Comparison:** each setting against C1 and C2 for the same generator, at every
  budget. A setting that does not beat C2 (no resampling) has no effect.
- **Noise:** one accuracy on 583 items has a standard error of about 2.1 points. Treat
  differences under about 3 points as ties. Among tied settings, prefer the one closest
  to its source (E1 for ePF, S1 for SPF).

## 8. Cost

Per setting: 583 items × (1 + 2 + … + 128) = 583 × 255 ≈ 149k trajectories.

| | basis | estimate |
|---|---|---|
| one E setting | the stored 30B D1K sweep processed about 10 trajectories/s per generator + judge pair (991 items: b16 in 1613 s, b64 in 6318 s) | ≈ 4.1 h per generator + judge pair |
| all 12 E settings | | ≈ 50 h of generator + judge time; ≈ 12.5 h wall on two nodes laid out as in §3 |
| 16 C and S settings | not measured: no judge, but SPF adds one short probe request per particle per step | expected less than an E setting; time the first one |

These are estimates from one earlier sweep on similar hardware. Time the first setting
on the new machine and re-estimate.
