"""Final-answer selectors, computed from a stored result row.

Identical to `selectors_from_row` in the previous implementation (tests/test_parity.py
replays stored rows through both). All selectors except `argmax_w` and `majority` work
over deduplicated trajectories, so a trajectory copied by resampling votes once.

  argmax_w            letter of the highest final log-weight
  majority            plain majority over all particles
  dedup_majority      majority over distinct trajectories
  probe_*/ent_*       probe-based marginals (SPF); fall back to dedup_majority without probe data
  judge_marg_mean     per letter, sum of each trajectory's mean judge score
  judge_marginal      per letter, sum of each trajectory's final judge score
  argmax_judge        letter of the trajectory with the highest final judge score
"""

from __future__ import annotations

import math
from collections import defaultdict


def _argmax(tally: dict[str, float], fallback: str) -> str:
    return max(tally, key=tally.get) if tally else fallback


def select_answers(row: dict) -> dict[str, str]:
    n = row["n"]
    letters = row["pred_letters"]
    final_weights = row["final_weights"]
    dists = row["probe_dists"]
    hashes = row["texts_hash"]
    all_dists = row.get("probe_dists_all") or [[] for _ in range(n)]
    out: dict[str, str] = {}

    out["argmax_w"] = letters[max(range(n), key=lambda i: final_weights[i])]

    tally = defaultdict(float)
    for letter in letters:
        if letter:
            tally[letter] += 1
    out["majority"] = _argmax(tally, "")

    seen, unique = set(), []
    for i in range(n):
        if hashes[i] not in seen:
            seen.add(hashes[i])
            unique.append(i)

    tally = defaultdict(float)
    for i in unique:
        if letters[i]:
            tally[letters[i]] += 1
    out["dedup_majority"] = _argmax(tally, "")
    dedup = out["dedup_majority"]

    tally = defaultdict(float)
    for i in unique:
        for letter, mass in (dists[i] or {}).items():
            tally[letter] += mass
    out["probe_marginal"] = _argmax(tally, dedup)

    tally = defaultdict(float)
    for i in range(n):
        for letter, mass in (dists[i] or {}).items():
            tally[letter] += mass
    out["probe_marginal_all"] = _argmax(tally, out["majority"])

    prestop, mean = defaultdict(float), defaultdict(float)
    for i in unique:
        steps = [d for d in (all_dists[i] if i < len(all_dists) else []) if d]
        if not steps:
            continue
        before_stop = steps[-2] if len(steps) >= 2 else steps[-1]
        for letter, mass in before_stop.items():
            prestop[letter] += mass
        for d in steps:
            for letter, mass in d.items():
                mean[letter] += mass / len(steps)
    out["probe_marg_prestop"] = _argmax(prestop, dedup)
    out["probe_marg_mean"] = _argmax(mean, dedup)

    def certainty_weighted(d: dict[str, float]) -> dict[str, float]:
        # Damp a confused distribution: c = 1 - H/log(K).
        total = sum(d.values())
        p = {k: v / total for k, v in d.items()}
        entropy = -sum(v * math.log(v) for v in p.values() if v > 1e-12)
        certainty = 1.0 - entropy / math.log(max(2, len(p)))
        return {k: certainty * v for k, v in p.items()}

    final_cert, mean_cert = defaultdict(float), defaultdict(float)
    for i in unique:
        steps = [
            d for d in (all_dists[i] if i < len(all_dists) else []) if d and sum(d.values()) > 0
        ]
        if not steps:
            if letters[i]:
                final_cert[letters[i]] += 1.0
                mean_cert[letters[i]] += 1.0
            continue
        for letter, value in certainty_weighted(steps[-1]).items():
            final_cert[letter] += value
        for d in steps:
            for letter, value in certainty_weighted(d).items():
                mean_cert[letter] += value / len(steps)
    out["ent_marg"] = _argmax(final_cert, dedup)
    out["ent_marg_mean"] = _argmax(mean_cert, dedup)

    confidence = [max(d.values()) if d else -1 for d in dists]
    if max(confidence) > -1:
        j = max(range(n), key=lambda i: confidence[i])
        out["probe_argmax_text"] = letters[j]
        out["probe_argmax_letter"] = max(dists[j], key=dists[j].get)
    else:
        out["probe_argmax_text"] = out["argmax_w"]
        out["probe_argmax_letter"] = out["argmax_w"]

    judge_scores = row.get("judge_scores_all")
    if judge_scores:
        final, mean_score = [], []
        for history in judge_scores:
            values = [s for s in (history or []) if s is not None]
            final.append(values[-1] if values else None)
            mean_score.append(sum(values) / len(values) if values else None)
        by_mean, by_final = defaultdict(float), defaultdict(float)
        for i in unique:
            if not letters[i]:
                continue
            if mean_score[i] is not None:
                by_mean[letters[i]] += mean_score[i]
            if final[i] is not None:
                by_final[letters[i]] += final[i]
        out["judge_marg_mean"] = _argmax(by_mean, dedup)
        out["judge_marginal"] = _argmax(by_final, dedup)
        candidates = [i for i in unique if letters[i] and final[i] is not None]
        out["argmax_judge"] = (
            letters[max(candidates, key=lambda i: final[i])] if candidates else dedup
        )
    return out
