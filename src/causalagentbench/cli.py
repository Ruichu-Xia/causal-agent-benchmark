"""Build a frozen benchmark, run a baseline, or evaluate tool-calling models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from causalagentbench.agent.eval import REPORTING_VERSION, evaluate_task
from causalagentbench.agent.loop import MAX_TOOL_CALLS, MAX_TURNS
from causalagentbench.benchmark import make_task
from causalagentbench.generation.causenet import CauseNetLabeler
from causalagentbench.scoring.score import SCORING_VERSION
from causalagentbench.task.dataset import load_dataset, save_dataset


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _summarize(rows: list[dict[str, Any]], model: str | None) -> dict[str, Any]:
    """Pool candidate errors before normalization; incomplete runs have no headline metric."""
    scored = [row for row in rows if row["type"] == "episode"]
    failed = len(rows) - len(scored)
    n_candidates = sum(row["n_candidates"] for row in scored)
    effect_mae = zero_mae = normalized_error = None
    if not failed and n_candidates:
        # Weight by candidate count: averaging task ratios would define a different metric.
        effect_mae = sum(row["effect_mae"] * (row["n_candidates"] / n_candidates) for row in scored)
        zero_mae = sum(row["zero_effect_mae"] * (row["n_candidates"] / n_candidates) for row in scored)
        normalized_error = effect_mae / zero_mae if zero_mae > 0 else None
        if normalized_error is not None and not math.isfinite(normalized_error):
            raise ValueError("normalized effect error exceeds the finite numeric range")
    return {
        "type": "summary",
        "scoring_version": SCORING_VERSION,
        "reporting_version": REPORTING_VERSION,
        "primary_metric": "normalized_effect_error",
        "model": model,
        "normalized_effect_error": normalized_error,
        "effect_mae": effect_mae,
        "zero_effect_mae": zero_mae,
        "answer_validity_rate": sum(row["parse_ok"] for row in scored) / len(rows) if rows else None,
        "scored": len(scored),
        "errors": failed,
        "parse_failures": sum(not row["parse_ok"] for row in scored),
        **{
            metric: sum(row[metric] for row in scored) / len(scored) if scored else None
            for metric in ("reward", "adjustment_score", "effect_score", "ranking_tau")
        },
        "estimated_cost_usd": (
            sum(row["estimated_cost_usd"] for row in rows)
            if all(row["estimated_cost_usd"] is not None for row in rows)
            else None
        ),
    }


def _evaluate(args: argparse.Namespace) -> int:
    tasks = load_dataset(args.dataset)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    models: list[str | None] = list(args.model) if args.command == "eval" else [None]
    baseline = args.agent if args.command == "baseline" else None
    if baseline in (None, "pc"):
        # Fail before starting: a missing discovery dependency must not silently lower a model's score.
        from causallearn.search.ConstraintBased.PC import pc  # noqa: F401
    if baseline is None:
        from causalagentbench.agent.clients import LiteLLMClient

        for model in models:
            assert model is not None
            LiteLLMClient(model=model, seed=None).validate_tool_capability()
    stem = Path(args.dataset).with_suffix("")
    digest = hashlib.sha256(stem.with_suffix(".json").read_bytes() + stem.with_suffix(".npz").read_bytes()).hexdigest()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with out.open("x", encoding="utf-8") as handle:

        def write(record: dict[str, Any]) -> None:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()

        write(
            {
                "type": "run",
                "scoring_version": SCORING_VERSION,
                "reporting_version": REPORTING_VERSION,
                "primary_metric": "normalized_effect_error",
                "dataset": str(stem),
                "dataset_sha256": digest,
                "tasks": len(tasks),
                "models": models if baseline is None else [baseline],
                "workers": args.workers,
                "max_tool_calls": args.max_tool_calls,
                "max_turns": args.max_turns,
                "temperature": 0.0,
                "sampling_seed": None,
            }
        )
        for model in models:
            rows = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(
                        evaluate_task,
                        index,
                        task,
                        model=model,
                        baseline=baseline,
                        max_tool_calls=args.max_tool_calls,
                        max_turns=args.max_turns,
                    )
                    for index, task in enumerate(tasks)
                ]
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    write(row)
            summary = _summarize(rows, model or baseline)
            errors += summary["errors"]
            write(summary)
            print(json.dumps(summary, allow_nan=False))
    return int(errors > 0)


def main() -> int:
    """Run the command-line interface; failures exit nonzero with an actionable message."""
    parser = argparse.ArgumentParser(description="CausalAgentBench: generate, evaluate, train.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="generate five-candidate tasks locally, without an LLM")
    build.add_argument("--out", required=True, help="dataset path stem (.json and .npz)")
    build.add_argument("--causenet", required=True, help="CauseNet precision JSONL or JSONL.bz2 dump")
    build.add_argument("-n", type=_positive, default=20, help="number of tasks (default: 20)")
    build.add_argument("--seed", type=int, default=0, help="first seed; tasks use consecutive seeds")
    for command in ("baseline", "eval"):
        sub = commands.add_parser(
            command, help="run a scripted baseline" if command == "baseline" else "evaluate models"
        )
        sub.add_argument("--dataset", required=True)
        sub.add_argument("--out", required=True, help="new JSONL results file; never overwrites")
        sub.add_argument("--limit", type=_positive, default=None)
        sub.add_argument("--workers", type=_positive, default=1, help="concurrent episodes (default: 1)")
        sub.add_argument("--max-tool-calls", type=_positive, default=MAX_TOOL_CALLS)
        sub.add_argument("--max-turns", type=_positive, default=MAX_TURNS)
        if command == "baseline":
            sub.add_argument("--agent", choices=("zero", "pc", "correlation", "oracle"), default="pc")
        else:
            sub.add_argument(
                "--model", action="append", required=True, help="LiteLLM model ID; repeat to compare models"
            )
    args = parser.parse_args()
    try:
        if args.command == "build":
            if args.seed < 0:
                parser.error("--seed must be nonnegative")
            labeler = CauseNetLabeler.from_path(args.causenet)
            tasks = [make_task(seed, labeler=labeler) for seed in range(args.seed, args.seed + args.n)]
            save_dataset(args.out, tasks)
            print(f"Saved {len(tasks)} tasks (seeds {args.seed}..{args.seed + args.n - 1}) to {args.out}.json + .npz")
            return 0
        return _evaluate(args)
    except ImportError as exc:
        parser.error(f"{exc}. Install evaluation dependencies: uv sync --extra agents --extra discovery")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
