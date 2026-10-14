"""Result rows, resume, and the run manifest.

Every row carries the full resolved configuration and a run fingerprint covering
everything that changes results: the method config plus models, sampling, step format
and benchmark (not the code version, so a bug-fix commit can resume a run). An output
file may hold rows of exactly one fingerprint: resuming into a file written under
different settings is refused, so runs can never be mixed inside one file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from epf_spf.answers import LETTERS, predicted_index
from epf_spf.config import FilterConfig
from epf_spf.filter import FilterResult
from epf_spf.records import MCQRecord
from epf_spf.steps import StepFormat


def _text_hash(text: str) -> str:
    # Stable across processes (Python's hash() is salted per process).
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def item_row(
    record: MCQRecord,
    budget: int,
    result: FilterResult,
    fmt: StepFormat,
    provenance: dict,
    store_texts: bool,
) -> dict:
    texts = [fmt.trajectory(p.steps) for p in result.particles]
    predicted = [predicted_index(t, record.choices) for t in texts]
    rounded = [
        [({k: round(v, 4) for k, v in m.items()} if m else None) for m in p.letter_masses]
        for p in result.particles
    ]
    row = {
        "unique_id": record.unique_id,
        "budget": budget,
        "gold": LETTERS[record.answer_index],
        "n": len(result.particles),
        "pred_letters": [LETTERS[i] if i is not None else "" for i in predicted],
        "final_weights": [round(p.log_weight, 4) for p in result.particles],
        "steps_used": [len(p.steps) for p in result.particles],
        "probe_dists": [next((m for m in reversed(masses) if m), None) for masses in rounded],
        "probe_dists_all": rounded,
        "texts_hash": [_text_hash(t) for t in texts],
        "score_failures": result.score_failures,
        "audio_paths": record.audio_paths,
        "error": None,
        **provenance,
    }
    if provenance["config"]["scorer"] == "judge":
        row["judge_scores_all"] = [
            [round(s, 4) if s is not None else None for s in p.scores] for p in result.particles
        ]
    if store_texts:
        row["texts"] = texts
        row["trace"] = result.trace
    return row


def error_row(record: MCQRecord, budget: int, error: BaseException, provenance: dict) -> dict:
    return {
        "unique_id": record.unique_id,
        "budget": budget,
        "error": f"{type(error).__name__}: {error}",
        **provenance,
    }


def completed_cells(output: Path, fingerprint: str) -> set[tuple[str, int]]:
    """(unique_id, budget) cells already done in `output`.

    Raises if the file holds rows from another configuration, or rows without one.
    Error rows are not counted as done, so they are retried.
    """
    if not output.exists():
        return set()
    done, foreign = set(), set()
    with open(output) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_fingerprint") != fingerprint:
                foreign.add(row.get("run_fingerprint"))
                continue
            if row.get("error") is None:
                done.add((row["unique_id"], row["budget"]))
    if foreign:
        raise SystemExit(
            f"{output} contains rows from a different configuration "
            f"(fingerprints {sorted(map(str, foreign))}, this run {fingerprint}). "
            "Write to a new file; one output file holds one configuration."
        )
    return done


def code_version() -> dict:
    """Package version and, when run from the git checkout, the commit and dirty state."""
    from epf_spf import __version__

    root = Path(__file__).resolve().parents[2]
    version = {"package": __version__}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        version |= {"git_commit": commit, "git_dirty": bool(dirty)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        version |= {"git_commit": None, "git_dirty": None}
    return version  # fmt: skip


def provenance(config: FilterConfig, run_settings: dict, version: dict) -> dict:
    """Fields written into every row: the method config, the run settings that also
    change results (models, sampling, step format, benchmark), the code version, and the
    run fingerprint over all of it except the code version."""
    identity = {"config": config.to_dict(), **run_settings}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return {
        "method": config.method,
        "config": config.to_dict(),
        "run_fingerprint": fingerprint,
        **run_settings,
        "code_version": version,
    }


def write_manifest(output: Path, manifest: dict) -> Path:
    path = output.with_suffix(output.suffix + ".manifest.json")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path
