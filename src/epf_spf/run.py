"""Command-line entry point: run one method on one benchmark at one or more budgets.

    epf-spf --method epf-paper --bench mmar \\
        --generator-endpoint http://127.0.0.1:9401/v1 --generator-model qwen-omni \\
        --judge-endpoint http://127.0.0.1:9311/v1 --judge-model qwen-omni-3b-judge \\
        --budgets 1,8,64 --output runs/mmar_qwen-omni_epf-paper.jsonl

See docs/running.md for every flag and docs/index.md for the methods.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from epf_spf.config import (
    FAILURE_POLICIES,
    PRESETS,
    RESAMPLE_RULES,
    RESAMPLING_METHODS,
    SCORERS,
    build_config,
)
from epf_spf.data import BENCHMARKS, DataPaths, load_benchmark
from epf_spf.filter import run_filter
from epf_spf.llm import GenerationConfig, GeneratorClient
from epf_spf.prompts import generator_messages
from epf_spf.results import (
    code_version,
    completed_cells,
    error_row,
    item_row,
    provenance,
    write_manifest,
)
from epf_spf.scorers import JudgeScorer, ProbeScorer, RandomScorer
from epf_spf.selection import select_answers
from epf_spf.steps import StepFormat
from epf_spf.weighting import AGGREGATIONS, TRANSFORMS

logger = logging.getLogger("epf_spf")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    method = parser.add_argument_group("method (a preset; every field below overrides it)")
    method.add_argument("--method", required=True, choices=list(PRESETS))
    method.add_argument("--scorer", choices=SCORERS)
    method.add_argument("--weight-transform", choices=TRANSFORMS)
    method.add_argument("--weight-temperature", type=float, help="T in log(s)/T or logit(s)/T")
    method.add_argument("--score-aggregation", choices=AGGREGATIONS)
    method.add_argument("--resample-from-step", type=int)
    method.add_argument("--resample-rule", choices=RESAMPLE_RULES)
    method.add_argument("--gate-threshold", type=float)
    method.add_argument("--annealing", action=argparse.BooleanOptionalAction, default=None)
    method.add_argument("--anneal-ess-threshold", type=float)
    method.add_argument(
        "--anneal-window",
        type=float,
        help="fraction of the step horizon during which annealing may fire",
    )
    method.add_argument("--resampling-method", choices=RESAMPLING_METHODS)
    method.add_argument("--on-score-failure", choices=FAILURE_POLICIES, default="fail")

    data = parser.add_argument_group("data")
    data.add_argument("--bench", required=True, choices=BENCHMARKS)
    data.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("EPF_DATA_ROOT"),
        help="folder holding mmar/, mmau_testmini/, mmau_pro_d1k/",
    )
    data.add_argument(
        "--extended-dir",
        type=Path,
        default=os.environ.get("EPF_EXTENDED_DIR"),
        help="extended_dataset/distractor_results",
    )
    data.add_argument(
        "--no-audio-override",
        action="store_true",
        help="do not substitute <data-root>/_audio_override_30s clips",
    )
    data.add_argument("--limit", type=int)
    data.add_argument("--item-offset", type=int, default=0)
    data.add_argument("--item-count", type=int)

    gen = parser.add_argument_group("generator")
    gen.add_argument("--generator-endpoint", required=True)
    gen.add_argument(
        "--generator-model",
        required=True,
        help="served model name; Phi-4-multimodal must be 'speech' (its LoRA)",
    )
    gen.add_argument("--sampling-temperature", type=float, default=0.8)
    gen.add_argument("--max-steps", type=int, default=10)
    gen.add_argument("--max-step-tokens", type=int, default=300)
    gen.add_argument("--min-tokens", type=int)
    gen.add_argument("--inline-step-stop", action="store_true")
    gen.add_argument("--think-trigger", action="store_true")

    judge = parser.add_argument_group("judge (scorer=judge)")
    judge.add_argument("--judge-endpoint")
    judge.add_argument("--judge-model")
    judge.add_argument("--judge-concurrency", type=int, default=48)

    run = parser.add_argument_group("run")
    run.add_argument("--budgets", required=True, help="comma-separated particle counts")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--max-inflight",
        type=int,
        default=192,
        help="particles in flight; items run max(1, max_inflight // budget) at a time",
    )
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--store-texts", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument(
        "--dry-run", action="store_true", help="resolve the config and load the data, then stop"
    )
    return parser.parse_args(argv)


def item_rng(seed: int, unique_id: str, budget: int) -> np.random.Generator:
    """Independent of how items interleave, so a resumed run draws the same numbers."""
    return np.random.default_rng([seed, zlib.crc32(unique_id.encode()), budget])


def summarize(output: Path, fingerprint: str) -> None:
    hits = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    with open(output) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("run_fingerprint") != fingerprint or row.get("error"):
                continue
            totals[row["budget"]] += 1
            for name, letter in select_answers(row).items():
                hits[row["budget"]][name] += letter == row["gold"]
    for budget in sorted(totals):
        scores = " ".join(f"{k}={v / totals[budget]:.4f}" for k, v in sorted(hits[budget].items()))
        logger.info("budget %d (n=%d): %s", budget, totals[budget], scores)


async def run(args: argparse.Namespace) -> None:
    config = build_config(
        args.method,
        on_score_failure=args.on_score_failure,
        scorer=args.scorer,
        weight_transform=args.weight_transform,
        weight_temperature=args.weight_temperature,
        score_aggregation=args.score_aggregation,
        resample_from_step=args.resample_from_step,
        resample_rule=args.resample_rule,
        gate_threshold=args.gate_threshold,
        annealing=args.annealing,
        anneal_ess_threshold=args.anneal_ess_threshold,
        anneal_window=args.anneal_window,
        resampling_method=args.resampling_method,
    )
    if config.scorer == "judge" and not (args.judge_endpoint and args.judge_model):
        raise SystemExit("scorer=judge needs --judge-endpoint and --judge-model")
    if args.data_root is None or args.extended_dir is None:
        raise SystemExit("set --data-root and --extended-dir (or EPF_DATA_ROOT, EPF_EXTENDED_DIR)")

    paths = DataPaths.from_layout(args.data_root, args.extended_dir, not args.no_audio_override)
    records = load_benchmark(args.bench, paths)
    end = args.item_offset + args.item_count if args.item_count else None
    records = records[args.item_offset : end][: args.limit]
    budgets = [int(b) for b in args.budgets.split(",")]

    fmt = StepFormat(max_steps=args.max_steps, inline_step_stop=args.inline_step_stop)
    gen_config = GenerationConfig(
        sampling_temperature=args.sampling_temperature,
        max_step_tokens=args.max_step_tokens,
        min_tokens=args.min_tokens,
        think_trigger=args.think_trigger,
    )
    run_settings = {
        "benchmark": args.bench,
        "generator_model": args.generator_model,
        "judge_model": args.judge_model if config.scorer == "judge" else None,
        "sampling_temperature": gen_config.sampling_temperature,
        "max_steps": fmt.max_steps,
        "max_step_tokens": gen_config.max_step_tokens,
        "min_tokens": gen_config.min_tokens,
        "inline_step_stop": fmt.inline_step_stop,
        "think_trigger": gen_config.think_trigger,
        "audio_override_dir": (
            str(paths.audio_override_dir.resolve()) if paths.audio_override_dir else None
        ),
        "seed": args.seed,
    }
    row_provenance = provenance(config, run_settings, code_version())
    logger.info("method=%s fingerprint=%s", config.method, row_provenance["run_fingerprint"])
    logger.info("config %s", json.dumps(config.to_dict(), sort_keys=True))
    logger.info("run %s", json.dumps(run_settings, sort_keys=True))
    logger.info("%d records (%s), budgets %s", len(records), args.bench, budgets)
    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_cells(args.output, row_provenance["run_fingerprint"])
    write_manifest(
        args.output,
        {
            **row_provenance,
            "argv": vars(args)
            | {
                "output": str(args.output),
                "data_root": str(args.data_root),
                "extended_dir": str(args.extended_dir),
            },
            "started_unix": time.time(),
        },
    )

    generator = GeneratorClient(args.generator_endpoint, args.generator_model, gen_config)
    if config.scorer == "judge":
        scorer = JudgeScorer(args.judge_endpoint, args.judge_model, fmt, args.judge_concurrency)
    elif config.scorer == "probe":
        scorer = ProbeScorer(generator, fmt)
    else:
        scorer = RandomScorer(args.seed)
    lock = asyncio.Lock()

    with open(args.output, "a") as out:

        async def one(record, budget, semaphore):
            async with semaphore:
                start = time.time()
                try:
                    result = await run_filter(
                        record,
                        budget,
                        config,
                        generator,
                        scorer,
                        fmt,
                        generator_messages(record, gen_config.think_trigger),
                        item_rng(args.seed, record.unique_id, budget),
                    )
                    row = item_row(record, budget, result, fmt, row_provenance, args.store_texts)
                    row["latency_s"] = round(time.time() - start, 1)
                except Exception as error:  # recorded as an error row and retried on resume
                    row = error_row(record, budget, error, row_provenance)
            async with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
            return row

        for budget in budgets:
            todo = [r for r in records if (r.unique_id, budget) not in done]
            concurrent = max(1, args.max_inflight // budget)
            logger.info(
                "[%s b%d] %d to do, %d concurrent items",
                config.method,
                budget,
                len(todo),
                concurrent,
            )
            start = time.time()
            semaphore = asyncio.Semaphore(concurrent)
            rows = await asyncio.gather(*(one(r, budget, semaphore) for r in todo))
            errors = sum(1 for r in rows if r.get("error"))
            logger.info("  done in %.0fs (%d errors)", time.time() - start, errors)

    if isinstance(scorer, JudgeScorer):
        logger.info("judge calls %d (%d failed)", scorer.calls, scorer.failures)
        await scorer.close()
    await generator.close()
    summarize(args.output, row_provenance["run_fingerprint"])


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    main()
