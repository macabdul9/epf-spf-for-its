"""Check tuning result files and print accuracy per setting and budget.

    python tuning/check_runs.py <OUT_ROOT>/d1k/qwen-omni [--selector dedup_majority]

For every <setting>.jsonl: one fingerprint, no duplicate cells, how many good rows each
budget has (583 when complete), and error rows. Accuracy is printed only for complete
budgets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from epf_spf.selection import select_answers

EXPECTED_ITEMS = 583
BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128)


def check(path: Path, selector: str) -> None:
    good, errors, fingerprints = {}, 0, set()
    duplicates = 0
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            fingerprints.add(row.get("run_fingerprint"))
            if row.get("error"):
                errors += 1
                continue
            cell = (row["unique_id"], row["budget"])
            duplicates += cell in good
            good[cell] = row
    per_budget = Counter(budget for _, budget in good)
    hits = defaultdict(int)
    for (_, budget), row in good.items():
        hits[budget] += select_answers(row)[selector] == row["gold"]

    problems = []
    if len(fingerprints) != 1:
        problems.append(f"{len(fingerprints)} fingerprints")
    if duplicates:
        problems.append(f"{duplicates} duplicate cells")
    status = "; ".join(problems) or "ok"
    print(f"{path.stem}: {status}, {errors} error rows")
    cells = []
    for budget in BUDGETS:
        n = per_budget.get(budget, 0)
        if n == EXPECTED_ITEMS:
            cells.append(f"b{budget}={hits[budget] / n:.3f}")
        else:
            cells.append(f"b{budget}={n}/{EXPECTED_ITEMS}")
    print("   " + "  ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--selector", default="dedup_majority")
    args = parser.parse_args()
    files = sorted(args.run_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no .jsonl files in {args.run_dir}")
    print(f"selector: {args.selector}")
    for path in files:
        check(path, args.selector)


if __name__ == "__main__":
    main()
