"""Benchmark loaders.

Item sets, item order, option order and gold indices match the previous
implementation (`probe_epf.load_records` and the `benchmarking/*/loader.py` modules);
tests/test_parity.py checks every benchmark against it.

Benchmarks:
  mmar, mmau, mmau_pro_d1k                     original option sets
  mmar_ext, mmau_ext, mmau_pro_d1k_ext         10-option extended sets
  mmau_ext_fixed                               mmau_ext with stale "(A) " prefixes removed
  mmar_sub, mmau_sub, mmau_pro_d1k_sub         original options, restricted to extended ids
  mmar_csv_orig, mmau_csv_orig, d1k_csv_orig   original options taken from the CSVs
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

from epf_spf.answers import match_answer_index
from epf_spf.records import MCQRecord

BENCHMARKS = (
    "mmar", "mmau", "mmau_pro_d1k",
    "mmar_ext", "mmau_ext", "mmau_ext_fixed", "mmau_pro_d1k_ext",
    "mmar_sub", "mmau_sub", "mmau_pro_d1k_sub",
    "mmar_csv_orig", "mmau_csv_orig", "d1k_csv_orig",
)  # fmt: skip

# Gold texts that match no original option even fuzzily; the correct option per the
# extended dataset's README.
MMAR_EXT_ANSWER_OVERRIDES = {
    "BV1P4411677K_0-00_0-20": "Clarinet player may be at ARSM diploma or professional level",
}
D1K_EXT_ANSWER_OVERRIDES = {
    "f34a9281-bb97-4570-bb90-48120aed2a7b": "neutralization reaction",
}

_OPTION_LABEL = re.compile(r"^\([A-Z]+\)\s*")
_STALE_PREFIX = re.compile(r"^\([A-Z]\)\s*")


@dataclass(frozen=True)
class DataPaths:
    """Where the benchmark data lives.

    `audio_override_dir`: clips with the same basename replace the originals. The
    previous implementation applied `<media_root>/_audio_override_30s` implicitly
    whenever it existed (79 trimmed clips for servers that reject audio over 30 s);
    keep it to stay comparable with earlier results, set None to hear the originals.
    """

    mmar_root: Path
    mmau_root: Path
    d1k_root: Path
    extended_dir: Path
    audio_override_dir: Path | None = None

    @classmethod
    def from_layout(cls, data_root: Path, extended_dir: Path, audio_override: bool = True):
        override = data_root / "_audio_override_30s"
        return cls(
            mmar_root=data_root / "mmar",
            mmau_root=data_root / "mmau_testmini",
            d1k_root=data_root / "mmau_pro_d1k",
            extended_dir=extended_dir,
            audio_override_dir=override if audio_override and override.is_dir() else None,
        )


def _bucket_length(duration_s: float | None) -> str:
    if duration_s is None:
        return "unknown"
    if duration_s <= 10:
        return "le10s"
    if duration_s <= 30:
        return "10to30s"
    return "gt30s"


def _timestamp_duration(timestamp) -> float | None:
    """'00:11:10,00:11:30' -> 20.0; None when malformed or non-positive."""
    if not isinstance(timestamp, str) or "," not in timestamp:
        return None

    def seconds(t: str) -> float:
        h, m, s = (float(x) for x in t.strip().split(":"))
        return h * 3600 + m * 60 + s

    try:
        start, end = timestamp.split(",", 1)
        duration = seconds(end) - seconds(start)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def _measured_length(record: MCQRecord) -> str | None:
    try:
        import soundfile
    except ImportError:
        return None
    if not all(os.path.exists(p) for p in record.audio_paths):
        return None
    with contextlib.suppress(RuntimeError):
        return _bucket_length(sum(soundfile.info(p).duration for p in record.audio_paths))
    return None


def _load_mmar(root: Path) -> list[MCQRecord]:
    rows = json.loads((root / "MMAR-meta.json").read_text())
    records = []
    for row in rows:
        choices = list(row.get("choices") or [])
        if not choices:
            continue
        audio_rel = row.get("audio_path")
        audio_paths = [os.path.abspath(os.path.join(root, str(audio_rel)))] if audio_rel else []
        answer = row.get("answer")
        # Verbatim index first: normalize() strips non-latin text, so CJK options can
        # collapse to the same string and the fuzzy match would pick the wrong one.
        answer_index = (
            choices.index(answer) if answer in choices else match_answer_index(answer, choices)
        )
        record = MCQRecord(
            unique_id=str(row.get("id")),
            question=str(row.get("question")),
            choices=choices,
            answer=str(answer),
            audio_paths=audio_paths,
            category=str(row.get("modality")),
            length_type=_bucket_length(_timestamp_duration(row.get("timestamp"))),
            answer_index=answer_index,
        )
        if not all(os.path.exists(p) for p in record.audio_paths):
            continue
        record.length_type = _measured_length(record) or record.length_type
        records.append(record)
    return records


def _load_mmau(root: Path) -> list[MCQRecord]:
    rows = json.loads((root / "MMAU-meta.json").read_text())
    records = []
    for row in rows:
        if not row.get("choices"):
            continue
        choices = [str(c) for c in row["choices"]]
        audio_rel = row.get("audio_path")
        audio_paths = [os.path.abspath(os.path.join(root, str(audio_rel)))] if audio_rel else []
        answer = row.get("answer")
        choices_raw, answer_raw = row.get("choices_raw"), row.get("answer_raw")
        # The shipped "(A) ..." strings are unique even where the stripped options repeat.
        if choices_raw and answer_raw in choices_raw:
            answer_index = choices_raw.index(answer_raw)
        elif answer in choices:
            answer_index = choices.index(answer)
        else:
            answer_index = match_answer_index(answer, choices)
        record = MCQRecord(
            unique_id=str(row.get("id")),
            question=str(row.get("question")),
            choices=choices,
            answer=str(answer),
            audio_paths=audio_paths,
            category=str(row.get("task")),
            length_type="unknown",
            answer_index=answer_index,
        )
        if not all(os.path.exists(p) for p in record.audio_paths):
            continue
        record.length_type = _measured_length(record) or record.length_type
        records.append(record)
    return records


def _load_mmau_pro_d1k(root: Path) -> list[MCQRecord]:
    import pandas as pd

    frame = pd.read_parquet(root / "data" / "test.parquet")
    records = []
    for _, row in frame.iterrows():
        row = row.to_dict()
        choices = [] if row.get("choices") is None else list(row["choices"])
        if not choices:
            continue
        audio_rel = [] if row.get("audio_path") is None else list(row["audio_path"])
        answer = row.get("answer")
        record = MCQRecord(
            unique_id=str(row.get("id")),
            question=str(row.get("question")),
            choices=choices,
            answer=str(answer),
            audio_paths=[os.path.join(root, p) for p in audio_rel],
            category=str(row.get("category")),
            length_type=str(row.get("length_type")),
            answer_index=match_answer_index(answer, choices),
        )
        if all(os.path.exists(p) for p in record.audio_paths):
            records.append(record)
    return records


def _extend(base: list[MCQRecord], augmented_jsonl: Path, overrides: dict[str, str]):
    """Overlay the 10-option `augmented_choices`, shuffled with the per-id seed the
    dataset's own export uses. Gold is resolved against the original options only."""
    by_id = {r.unique_id: r for r in base}
    records = []
    for line in augmented_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record = by_id.get(str(row["id"]))
        if record is None:
            continue
        gold = overrides.get(str(row["id"]), row["answer"])
        original = list(row["choices"])
        index = original.index(gold) if gold in original else match_answer_index(gold, original)
        if index is None:
            continue
        gold_text = original[index]
        choices = list(row["augmented_choices"])
        random.Random(f"{row['id']}-extended").shuffle(choices)
        records.append(
            dataclasses.replace(
                record, choices=choices, answer=gold_text, answer_index=choices.index(gold_text)
            )
        )
    return records


def _split_csv_options(raw: str) -> list[str]:
    # MMAU labels every original option, so split on the labels there. MMAR and d1k
    # are bare ";"-joined, and a few option texts contain ";" themselves; those rows
    # are caught by the caller and taken from the base benchmark instead.
    parts = re.split(r";\s*(?=\([A-Z]+\)\s)", raw.strip())
    if len(parts) > 1:
        return [_OPTION_LABEL.sub("", x.strip()).strip() for x in parts if x.strip()]
    return [x.strip() for x in raw.split(";") if x.strip()]


def _csv_key(text: str) -> str:
    return _OPTION_LABEL.sub("", text.strip()).strip().rstrip(".").lower()


def _from_csv(base: list[MCQRecord], csv_path: Path) -> list[MCQRecord]:
    """Original options from `mcq_original_options`, gold from `ground_truth`, audio from
    the base benchmark, option order from the same per-id shuffle as the ext sets."""
    csv.field_size_limit(10**9)
    by_id = {r.unique_id: r for r in base}
    records = []
    with open(csv_path) as handle:
        for row in csv.DictReader(handle):
            record = by_id.get(row["id"])
            if record is None:
                continue
            choices = _split_csv_options(row["mcq_original_options"])
            gold = _OPTION_LABEL.sub("", row["ground_truth"]).strip()
            base_keys = {_csv_key(c) for c in record.choices}
            if (
                len(choices) != len(record.choices)
                or not {_csv_key(c) for c in choices} <= base_keys
            ):
                choices, gold = list(record.choices), record.answer
            keys = [_csv_key(c) for c in choices]
            index = (
                keys.index(_csv_key(gold))
                if _csv_key(gold) in keys
                else match_answer_index(gold, choices)
            )
            if index is None:
                continue
            gold_text = choices[index]
            random.Random(f"{row['id']}-extended").shuffle(choices)
            records.append(
                dataclasses.replace(
                    record,
                    choices=choices,
                    answer=gold_text,
                    answer_index=choices.index(gold_text),
                )
            )
    return records


def _apply_audio_override(records: list[MCQRecord], override_dir: Path | None):
    if override_dir is None:
        return records
    available = set(os.listdir(override_dir))
    return [
        dataclasses.replace(
            r,
            audio_paths=[
                os.path.join(override_dir, os.path.basename(p))
                if os.path.basename(p) in available
                else p
                for p in r.audio_paths
            ],
        )
        for r in records
    ]


def _load(name: str, paths: DataPaths) -> list[MCQRecord]:
    ext = paths.extended_dir
    if name == "mmar":
        records = _load_mmar(paths.mmar_root)
    elif name == "mmau":
        records = _load_mmau(paths.mmau_root)
    elif name == "mmau_pro_d1k":
        records = _load_mmau_pro_d1k(paths.d1k_root)
    elif name == "mmar_ext":
        records = _extend(
            _load_mmar(paths.mmar_root), ext / "mmar_augmented.jsonl", MMAR_EXT_ANSWER_OVERRIDES
        )
    elif name == "mmau_ext":
        records = _extend(_load_mmau(paths.mmau_root), ext / "mmau_augmented.jsonl", {})
    elif name == "mmau_ext_fixed":
        records = [
            dataclasses.replace(
                r,
                choices=[_STALE_PREFIX.sub("", c) for c in r.choices],
                answer=_STALE_PREFIX.sub("", r.answer),
            )
            for r in _load("mmau_ext", paths)
        ]
    elif name == "mmau_pro_d1k_ext":
        records = _extend(
            _load_mmau_pro_d1k(paths.d1k_root),
            ext / "mmau-pro-d1k_augmented.jsonl",
            D1K_EXT_ANSWER_OVERRIDES,
        )
    elif name in ("mmar_sub", "mmau_sub", "mmau_pro_d1k_sub"):
        base_name, fname = {
            "mmar_sub": ("mmar", "mmar_augmented.jsonl"),
            "mmau_sub": ("mmau", "mmau_augmented.jsonl"),
            "mmau_pro_d1k_sub": ("mmau_pro_d1k", "mmau-pro-d1k_augmented.jsonl"),
        }[name]
        keep = {
            str(json.loads(line)["id"])
            for line in (ext / fname).read_text().splitlines()
            if line.strip()
        }
        records = [r for r in _load(base_name, paths) if r.unique_id in keep]
    elif name in ("mmar_csv_orig", "mmau_csv_orig", "d1k_csv_orig"):
        base_name, fname = {
            "mmar_csv_orig": ("mmar", "mmar.csv"),
            "mmau_csv_orig": ("mmau", "mmau.csv"),
            "d1k_csv_orig": ("mmau_pro_d1k", "mmau-pro-d1k.csv"),
        }[name]
        records = _from_csv(_load(base_name, paths), ext / fname)
    else:
        raise ValueError(f"unknown benchmark {name!r}; choose from {', '.join(BENCHMARKS)}")
    records = [r for r in records if r.answer_index is not None]
    return _apply_audio_override(records, paths.audio_override_dir)


def load_benchmark(name: str, paths: DataPaths, limit: int | None = None) -> list[MCQRecord]:
    """Gradeable records of `name`, in the dataset's own order, truncated to `limit`."""
    return _load(name, paths)[:limit]
