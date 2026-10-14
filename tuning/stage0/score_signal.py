"""Stage 0: does the score carry signal on MMAU-Pro D1K (standard options, 583 items)?

Reads stored result rows only; sends no requests. For each generator and scorer:

  discrimination   within one item, do correct trajectories score higher than wrong
                   ones? AUROC over the item's distinct final trajectories, averaged
                   over items that have both kinds. Also pooled over all items.
  selection        accuracy of answer selectors vs. a random distinct trajectory
                   and vs. the oracle (any distinct trajectory correct).
  concentration    ESS_n of the step-1 resampling distribution over all N particles
                   under each weight rule.

Final trajectories are the survivors of resampling at T = 3, so discrimination is
measured on a population the stored run already selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from epf_spf.data import DataPaths, load_benchmark
from epf_spf.resampling import annealing_temperature, ess_ratio, softmax
from epf_spf.selection import select_answers

PROJECT = Path(__file__).resolve().parents[3]
JUDGE_DIR = PROJECT / "epf-with-qwen3-30ba3b-prm"
PROBE_DIR = PROJECT / "epf_results" / "p12_gated" / "out"
STORED_T = 3.0

B128 = JUDGE_DIR / "extended-source-b128-queue" / "original"
SOURCES = {
    ("qwen-omni", "judge-30B"): [
        JUDGE_DIR / "original" / "mmau_pro_d1k_qwen-omni_judgeprm_p12.jsonl",
        B128 / "mmau_pro_d1k_qwen-omni_judgeprm_t3_p12.jsonl",
    ],
    ("qwen-omni-3b", "judge-30B"): [
        JUDGE_DIR / "original" / "mmau_pro_d1k_qwen-omni-3b_judgeprm_p12.jsonl",
        B128 / "mmau_pro_d1k_qwen-omni-3b_judgeprm_t3_p12.jsonl",
    ],
    ("qwen-omni", "probe"): sorted(
        PROBE_DIR.glob("mmau_pro_d1k_sub_qwen-omni_probe_adaptive_t3_p12_b128g*.jsonl")
    ),
    ("qwen-omni-3b", "probe"): sorted(
        PROBE_DIR.glob("mmau_pro_d1k_sub_qwen-omni-3b_probe_adaptive_t3_p12_b128g*.jsonl")
    ),
}

WEIGHT_RULES = {
    "log_T1": lambda s: math.log(max(s, 1e-6)),
    "logit_T1": lambda s: _logit(s),
    "logit_T2": lambda s: _logit(s) / 2,
    "logit_T3": lambda s: _logit(s) / 3,
}


def _logit(s: float) -> float:
    s = min(max(s, 1e-6), 1 - 1e-6)
    return math.log(s / (1 - s))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def load_rows(paths: list[Path], keep_ids: set[str]) -> list[dict]:
    rows, seen = [], set()
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("error") or row["unique_id"] not in keep_ids:
                    continue
                cell = (row["unique_id"], row["budget"])
                if cell in seen:
                    raise SystemExit(f"duplicate cell {cell} in {path}")
                seen.add(cell)
                rows.append(row)
    return rows


def trajectory_scores(row: dict, scorer: str, i: int) -> dict[str, float] | None:
    """Per-trajectory score summaries; each corresponds to an aggregation choice."""
    if scorer == "probe":
        steps = [max(d.values()) for d in row["probe_dists_all"][i] if d]
    else:
        steps = [s for s in row["judge_scores_all"][i] if s is not None]
    if not steps:
        return None
    return {
        "first": steps[0],
        "last": steps[-1],
        "mean": sum(steps) / len(steps),
        "min": min(steps),
        "prod": math.prod(steps),
    }


def auroc(positives: list[float], negatives: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def distinct_trajectories(row: dict) -> list[int]:
    seen, keep = set(), []
    for i, h in enumerate(row["texts_hash"]):
        if h not in seen and row["pred_letters"][i]:
            seen.add(h)
            keep.append(i)
    return keep


def bootstrap_ci(values: list[float], rng: np.random.Generator, n: int = 2000):
    x = np.asarray(values)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def discrimination(rows: list[dict], scorer: str, rng) -> list[dict]:
    out = []
    by_budget = defaultdict(list)
    for row in rows:
        by_budget[row["budget"]].append(row)
    for budget in sorted(by_budget):
        if budget < 4:
            continue
        per_item = defaultdict(list)
        pooled = defaultdict(lambda: ([], []))
        mixed = 0
        for row in by_budget[budget]:
            labelled = []
            for i in distinct_trajectories(row):
                scores = trajectory_scores(row, scorer, i)
                if scores is not None:
                    labelled.append((row["pred_letters"][i] == row["gold"], scores))
            for correct, scores in labelled:
                for name, value in scores.items():
                    pooled[name][0 if correct else 1].append(value)
            pos = [s for c, s in labelled if c]
            neg = [s for c, s in labelled if not c]
            if not pos or not neg:
                continue
            mixed += 1
            for name in pos[0]:
                per_item[name].append(auroc([s[name] for s in pos], [s[name] for s in neg]))
        for name, values in per_item.items():
            low, high = bootstrap_ci(values, rng)
            out.append(
                {
                    "budget": budget,
                    "score": name,
                    "items": len(by_budget[budget]),
                    "mixed_items": mixed,
                    "within_item_auroc": round(float(np.mean(values)), 3),
                    "ci95_low": round(low, 3),
                    "ci95_high": round(high, 3),
                    "pooled_auroc": round(rank_auroc(*pooled[name]), 3),
                }
            )
    return out


def rank_auroc(positives: list[float], negatives: list[float]) -> float:
    values = np.concatenate([positives, negatives])
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = len(positives), len(negatives)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def selection(rows: list[dict], scorer: str) -> list[dict]:
    selectors = ["argmax_w", "dedup_majority", "majority"]
    selectors += ["argmax_judge", "judge_marginal", "judge_marg_mean"] if scorer != "probe" else [
        "probe_marginal", "ent_marg"]  # fmt: skip
    by_budget = defaultdict(list)
    for row in rows:
        by_budget[row["budget"]].append(row)
    out = []
    for budget in sorted(by_budget):
        hits = defaultdict(float)
        for row in by_budget[budget]:
            picked = select_answers(row)
            for name in selectors:
                hits[name] += picked.get(name) == row["gold"]
            distinct = distinct_trajectories(row)
            labels = [row["pred_letters"][i] == row["gold"] for i in distinct]
            hits["random_distinct"] += sum(labels) / len(labels) if labels else 0.0
            hits["oracle_any_distinct"] += any(labels)
            hits["distinct_fraction"] += len(set(row["texts_hash"])) / row["n"]
        n = len(by_budget[budget])
        out.append({"budget": budget, "items": n} | {k: round(v / n, 4) for k, v in hits.items()})
    return out


def step1_scores(row: dict, scorer: str) -> list[float]:
    """All N particles' step-1 scores, recovered from the stored step-1 log-weights."""
    first = row["trace"]["iterations"][0]
    return [_sigmoid(STORED_T * p["cum_log_weight"]) for p in first["particles"]]


def concentration(rows: list[dict], scorer: str) -> list[dict]:
    out = []
    by_budget = defaultdict(list)
    for row in rows:
        if row.get("trace") and row["budget"] >= 16:
            by_budget[row["budget"]].append(row)
    for budget in sorted(by_budget):
        stats = defaultdict(list)
        for row in by_budget[budget]:
            scores = step1_scores(row, scorer)
            stats["score_median"].append(float(np.median(scores)))
            for rule, fn in WEIGHT_RULES.items():
                weights = np.array([fn(s) for s in scores])
                p = softmax(weights)
                stats[f"ess_{rule}"].append(ess_ratio(p))
                for label, (threshold, window) in {
                    "paper": (0.5, 0.5),
                    "proj": (0.6, 0.7),
                }.items():
                    beta_inv = annealing_temperature(p, 1, 10, threshold, window)
                    stats[f"ess_{rule}_anneal_{label}"].append(
                        ess_ratio(softmax(weights / beta_inv))
                    )
        row_out = {"budget": budget, "items": len(by_budget[budget])}
        row_out |= {k: round(float(np.median(v)), 3) for k, v in stats.items()}
        out.append(row_out)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=PROJECT / "epf_data")
    parser.add_argument("--extended-dir", type=Path,
                        default=PROJECT / "extended_dataset" / "distractor_results")  # fmt: skip
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    records = load_benchmark(
        "mmau_pro_d1k_sub", DataPaths.from_layout(args.data_root, args.extended_dir)
    )
    gold = {r.unique_id: "ABCDEFGHIJK"[r.answer_index] for r in records}
    rng = np.random.default_rng(0)

    tables = defaultdict(list)
    for (generator, scorer), paths in SOURCES.items():
        rows = load_rows(paths, set(gold))
        mismatched = sum(row["gold"] != gold[row["unique_id"]] for row in rows)
        ids = {row["unique_id"] for row in rows}
        budgets = sorted({r["budget"] for r in rows})
        print(
            f"{generator} {scorer}: {len(rows)} rows, {len(ids)} items, "
            f"budgets {budgets}, gold mismatches {mismatched}"
        )
        if mismatched:
            raise SystemExit("stored gold letters disagree with the loader")
        tag = {"generator": generator, "scorer": scorer}
        tables["discrimination"] += [tag | r for r in discrimination(rows, scorer, rng)]
        tables["selection"] += [tag | r for r in selection(rows, scorer)]
        tables["concentration"] += [tag | r for r in concentration(rows, scorer)]

    for name, rows in tables.items():
        write_csv(args.out_dir / f"{name}.csv", rows)
        print(f"wrote {args.out_dir / f'{name}.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
