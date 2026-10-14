# Data

Loaders are in [`data.py`](../src/epf_spf/data.py). For every benchmark they produce the
same items, item order, option order and gold index as the previous implementation
(`probe_epf.load_records` and `benchmarking/*/loader.py`). `tests/test_parity.py`
checks all 13 benchmarks field by field.

## Layout

```text
$EPF_DATA_ROOT/
  mmar/                     MMAR-meta.json and audio
  mmau_testmini/            MMAU test-mini metadata and audio
  mmau_pro_d1k/             data/test.parquet and audio
  _audio_override_30s/      optional: trimmed clips, matched by basename
$EPF_EXTENDED_DIR/          extended_dataset/distractor_results
  mmar_augmented.jsonl, mmau_augmented.jsonl, mmau-pro-d1k_augmented.jsonl, *.csv
```

## Benchmarks

Counts are as loaded on this machine with the override on.

| `--bench` | items | options | what it is | override clips used |
|---|---|---|---|---|
| `mmar` | 996 | original | MMAR | 16 |
| `mmau` | 1000 | original | MMAU test-mini | 63 |
| `mmau_pro_d1k` | 991 | original | the MMAU-Pro 1k subset ("D1K") | 0 |
| `mmar_ext` | 833 | 10 | MMAR with distractors added to 10 options | 16 |
| `mmau_ext` | 869 | 10 | MMAU with 10 options | 63 |
| `mmau_ext_fixed` | 869 | 10 | `mmau_ext` with stale `(A) ` prefixes removed from option text | 63 |
| `mmau_pro_d1k_ext` | 583 | 10 | D1K with 10 options | 0 |
| `mmar_sub` | 832 | original | MMAR, restricted to the extended ids | 16 |
| `mmau_sub` | 869 | original | MMAU, restricted to the extended ids | 63 |
| `mmau_pro_d1k_sub` | 583 | original | D1K, restricted to the extended ids | 0 |
| `mmar_csv_orig` | 832 | original | the extended CSVs' `mcq_original_options` | 16 |
| `mmau_csv_orig` | 869 | original | same, MMAU | 63 |
| `d1k_csv_orig` | 583 | original | same, D1K | 0 |

The `*_sub` and `*_csv_orig` sets have the same items. The difference is where the
original options come from: the source benchmark (`*_sub`) or the extended CSV
(`*_csv_orig`). When a CSV row's options cannot be parsed, the loader falls back to the
source benchmark's options for that item, as the previous code did.

### The extended (10-option) sets

- The option order of each item is a fixed shuffle seeded by
  `random.Random(f"{unique_id}-extended")`, identical to the previous code.
- The gold option is found by fuzzy text match (threshold 0.85). Two items whose gold
  text matches no option are corrected by explicit overrides from the extended
  dataset's README (`MMAR_EXT_ANSWER_OVERRIDES`, `D1K_EXT_ANSWER_OVERRIDES`).

## The 30-second audio override

Some servers reject audio longer than 30 seconds. `<data-root>/_audio_override_30s`
holds 79 trimmed clips. When a record's clip has the same basename as a file there, the
trimmed clip is used.

- **The previous code applied this silently** whenever the folder existed. This code
  applies it by default too, so results stay comparable, but the choice is visible: the
  folder used is recorded as `audio_override_dir` in every row and is part of the
  fingerprint.
- `--no-audio-override` uses the original clips. That changes the run fingerprint.
- `audio_paths` in each row is the list of clips actually sent.

Items affected: 16 in MMAR-based sets, 63 in MMAU-based sets, none in D1K.

## Records

```python
MCQRecord(unique_id, question, choices, answer, audio_paths, category, length_type, answer_index)
```

`category` (`sound`, `music`, `speech`, ...) selects the think-trigger word.
`length_type` is a duration bucket. `answer_index` indexes `choices`, and `gold` in the
output is `"ABCDEFGHIJK"[answer_index]`.
