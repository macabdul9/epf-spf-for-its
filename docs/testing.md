# Testing

```bash
.venv/bin/pip install -e ".[dev,legacy]"     # legacy: what the old code needs to import
.venv/bin/pytest                             # everything
.venv/bin/pytest -m "not legacy"             # without the previous implementation
.venv/bin/pytest -m "not data"               # without the benchmark data
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

No test sends a network request or needs a GPU.

## Where the parity tests look

| what | default | override |
|---|---|---|
| previous implementation | not used unless set (opt-in) | `LEGACY_EPF_ROOT` |
| benchmark data | `../epf_data` | `EPF_DATA_ROOT` |
| extended dataset | `../extended_dataset/distractor_results` | `EPF_EXTENDED_DIR` |
| stored result rows (selector and extraction parity) | a few fixed files under `../epf-with-different-prms/` and `../epf_results/` | |

The tests read those result files and never write to them. If any input is missing, the
dependent tests are skipped, not failed. A green run on another machine may therefore
have skipped them; check the `skipped` count.

## What each file guards

| file | tests | guards |
|---|---|---|
| `test_weighting.py` | 13 | log ∝ s, logit ∝ odds, T divides the log-weight, clamps keep weights finite, missing scores are neutral, aggregations |
| `test_resampling.py` | 8 | ESS bounds, the annealing formula, its threshold and window, annealing flattens, the SPF gate, systematic resampling gives each particle ⌊Np⌋ or ⌈Np⌉ copies and is reproducible from the seed |
| `test_config.py` | 8 | every preset builds; presets encode the documented differences; overrides; invalid configs are rejected |
| `test_filter.py` | 9 | the loop with a scripted generator and scorer: paper skips step 1, its_hub resamples at step 1, the SPF gate both ways, max_steps + 1 steps, finished particles are not re-scored, failure policy, weight from the configured aggregation and transform |
| `test_results.py` | 4 | the fingerprint changes with every result-relevant setting and ignores the code version; resume skips done cells and retries errors; a file cannot mix configurations |
| `test_scorers.py` | 3 | the random control scorer is reproducible from the seed and differs between items |
| `test_parity.py` | 29 | equality with the previous implementation; see [parity-and-differences.md](parity-and-differences.md) |

74 tests in total. On this machine, with the old code and data present, all 74 pass.

## Adding a method

1. Add a preset to `PRESETS` in `config.py`, with a comment naming its source.
2. If it needs behaviour the loop does not have, add a `FilterConfig` field (validated
   in `__post_init__`) and use it in `filter.py`. New fields enter the fingerprint
   automatically.
3. Add its distinguishing values to `test_presets_encode_the_documented_differences`,
   and a `test_filter.py` case for the new behaviour.
4. Document it in `docs/methods-overview.md` and its own page.
