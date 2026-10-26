"""Build frozen task sets, run baselines or models, rescore saved answers, and report results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from causalagentbench.agent.eval import episode_record, evaluate_task
from causalagentbench.agent.loop import MAX_TOOL_CALLS, MAX_TURNS, PROTOCOL_VERSION
from causalagentbench.benchmark import make_task
from causalagentbench.generation.causenet import CauseNetLabeler
from causalagentbench.scoring.baselines import BASELINES
from causalagentbench.scoring.report import compare, names_ablation, summarize
from causalagentbench.scoring.score import SCORING_VERSION
from causalagentbench.task.dataset import load_dataset, save_dataset
from causalagentbench.task.instance import Instance, InstanceKey, with_placeholder_names

_RESUME_KEYS = (
    "scoring_version",
    "protocol",
    "dataset_sha256",
    "tasks",
    "models",
    "placeholder_names",
    "max_tool_calls",
    "max_turns",
)
"""Header fields that must match for ``--resume`` to append to an existing results file."""


def _positive(value: str) -> int:
    """Argparse type for a positive integer."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _dataset_sha256(stem: Path) -> str:
    """SHA-256 over the dataset's JSON and NPZ files."""
    return hashlib.sha256(stem.with_suffix(".json").read_bytes() + stem.with_suffix(".npz").read_bytes()).hexdigest()


def _version(package: str) -> str | None:
    """Installed version of ``package``, or None."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    """Output of a git command in this checkout, or None outside a repository."""
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _provenance() -> dict[str, Any]:
    """Code and environment versions recorded in every results header."""
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in ("causalagentbench", "numpy", "networkx", "litellm", "causal-learn")},
    }


def _placeholder_tasks(tasks: list[tuple[Instance, InstanceKey]]) -> list[tuple[Instance, InstanceKey]]:
    """The names ablation: the same tasks with columns renamed ``X1..Xk``, reproducibly per task."""
    return [with_placeholder_names(task, seed=np.random.SeedSequence([index, 7])) for index, task in enumerate(tasks)]


_worker_labeler: CauseNetLabeler | None = None
"""Each build process loads CauseNet once."""


def _load_worker_labeler(causenet: str) -> None:
    """Process-pool initializer: load the CauseNet labeler."""
    global _worker_labeler
    _worker_labeler = CauseNetLabeler.from_path(causenet)


def _build_one(split: str, index: int, salt: str | None) -> tuple[Instance, InstanceKey]:
    """Build one task in a worker process."""
    assert _worker_labeler is not None
    return make_task(split, index, labeler=_worker_labeler, salt=salt)


def _build(args: argparse.Namespace) -> int:
    """Build a frozen split of ``args.n`` tasks; task ``i`` depends only on the split, ``i`` and the salt."""
    salt = os.environ[args.salt_env] if args.salt_env else None
    tasks: dict[int, tuple[Instance, InstanceKey]] = {}
    with ProcessPoolExecutor(args.workers, initializer=_load_worker_labeler, initargs=(args.causenet,)) as pool:
        futures = {pool.submit(_build_one, args.split, i, salt): i for i in range(args.n)}
        for future in as_completed(futures):
            tasks[futures[future]] = future.result()
            print(f"built {len(tasks)}/{args.n}", file=sys.stderr, flush=True)
    save_dataset(args.out, [tasks[i] for i in range(args.n)], split=args.split)
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every record of a JSONL file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _evaluate(args: argparse.Namespace) -> int:
    """Run a baseline or models over a dataset, appending episodes, errors and transcripts."""
    stem = Path(args.dataset).with_suffix("")
    tasks = load_dataset(stem)[: args.limit]
    if args.placeholder_names:
        tasks = _placeholder_tasks(tasks)
    baseline = args.agent if args.command == "baseline" else None
    models: list[str] = [baseline] if baseline else list(args.model)
    if baseline in (None, "pc"):
        from causallearn.search.ConstraintBased.PC import pc  # noqa: F401  # fail before starting
    if baseline is None:
        from causalagentbench.agent.clients import LiteLLMClient

        for model in models:
            LiteLLMClient(model=model).validate_tool_capability()
    header = {
        "type": "run",
        "scoring_version": SCORING_VERSION,
        "protocol": PROTOCOL_VERSION,
        "dataset": str(stem),
        "dataset_sha256": _dataset_sha256(stem),
        "tasks": len(tasks),
        "models": models,
        "placeholder_names": args.placeholder_names,
        "max_tool_calls": args.max_tool_calls,
        "max_turns": args.max_turns,
        "request_timeout_s": args.timeout,
        "sampling": "provider defaults",
        "started_utc": datetime.now(UTC).isoformat(),
        **_provenance(),
    }
    out = Path(args.out)
    done: set[tuple[str, int]] = set()
    if out.exists():
        if not args.resume:
            raise FileExistsError(f"{out} exists; pass --resume to continue it")
        existing = _read_jsonl(out)
        mismatched = [k for k in _RESUME_KEYS if existing[0].get(k) != header[k]]
        if mismatched:
            raise ValueError(f"cannot resume: header differs in {mismatched}")
        done = {(r["model"], r["index"]) for r in existing if r["type"] == "episode"}
    out.parent.mkdir(parents=True, exist_ok=True)
    transcripts = out.with_suffix(".transcripts.jsonl")
    with out.open("a", encoding="utf-8") as results:
        if not done and out.stat().st_size == 0:
            results.write(json.dumps(header) + "\n")
        for model in models:
            pending = [i for i in range(len(tasks)) if (model, i) not in done]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(
                        evaluate_task,
                        i,
                        tasks[i],
                        model=None if baseline else model,
                        baseline=baseline,
                        max_tool_calls=args.max_tool_calls,
                        max_turns=args.max_turns,
                        request_timeout=args.timeout,
                    )
                    for i in pending
                ]
                for future in as_completed(futures):
                    record, transcript = future.result()
                    results.write(json.dumps(record, allow_nan=False) + "\n")
                    results.flush()
                    if transcript is not None:
                        with transcripts.open("a", encoding="utf-8") as log:
                            log.write(json.dumps(transcript) + "\n")
    return _report_file(out)


def _episodes_by_model(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Scored episodes in a results file, grouped by model; the last record per task wins."""
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for record in _read_jsonl(path):
        if record["type"] == "episode":
            latest[(record["model"], record["index"])] = record
    grouped: dict[str, list[dict[str, Any]]] = {}
    for (model, _), record in sorted(latest.items()):
        grouped.setdefault(model, []).append(record)
    return grouped


def _report_file(path: Path) -> int:
    """Print each model's summary; return 1 if any task of any model has no scored episode."""
    header = _read_jsonl(path)[0]
    grouped = _episodes_by_model(path)
    incomplete = 0
    for model in header["models"]:
        episodes = grouped.get(model, [])
        if episodes:
            print(json.dumps({"model": model, **summarize(episodes)}, indent=1))
        incomplete += header["tasks"] - len(episodes)
    if incomplete:
        print(f"{incomplete} task(s) have no scored episode; rerun with --resume", file=sys.stderr)
    return int(incomplete > 0)


def _report(args: argparse.Namespace) -> int:
    """Summarize a results file, optionally against a second run or a placeholder-names run on the same tasks."""
    path = Path(args.results)
    status = _report_file(path)
    other = args.compare or args.ablation
    if other is None:
        return status
    first_header, second_header = _read_jsonl(path)[0], _read_jsonl(Path(other))[0]
    if first_header["dataset_sha256"] != second_header["dataset_sha256"]:
        raise ValueError("the two results files were run on different datasets")
    first, second = _episodes_by_model(path), _episodes_by_model(Path(other))
    if args.ablation:
        if first_header["placeholder_names"] or not second_header["placeholder_names"]:
            raise ValueError("--ablation needs a real-names run first and a placeholder-names run second")
        for model in sorted(first.keys() & second.keys()):
            print(json.dumps({"model": model, "names_ablation": names_ablation(first[model], second[model])}, indent=1))
    else:
        for a in first:
            for b in second:
                print(json.dumps({"first": a, "second": b, **compare(first[a], second[b])}, indent=1))
    return status


def _rescore(args: argparse.Namespace) -> int:
    """Re-score every saved answer with the current scorer into a new results file."""
    records = _read_jsonl(Path(args.results))
    header = records[0]
    stem = Path(header["dataset"])
    if _dataset_sha256(stem) != header["dataset_sha256"]:
        raise ValueError("dataset files changed since the run")
    tasks = load_dataset(stem)
    if header["placeholder_names"]:
        tasks = _placeholder_tasks(tasks)
    out = Path(args.out)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({**header, "scoring_version": SCORING_VERSION, "rescored_from": args.results}) + "\n")
        for record in records[1:]:
            if record["type"] == "episode":
                extra = {k: v for k, v in record.items() if k in _EPISODE_EXTRAS}
                _, key = tasks[record["index"]]
                record = episode_record(record["index"], record["model"], record["answer"], key, **extra)
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    return _report_file(out)


_EPISODE_EXTRAS = (
    "seconds",
    "turns",
    "tool_calls",
    "prompt_tokens",
    "completion_tokens",
    "estimated_cost_usd",
    "served_models",
    "cached_tokens",
    "reasoning_tokens",
    "turn_usage",
)
"""Run measurements copied unchanged when an episode is rescored."""


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(prog="causalagentbench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="generate a frozen split locally, without an LLM")
    build.add_argument("--causenet", required=True, help="CauseNet-Precision JSONL or JSONL.bz2 dump")
    build.add_argument("--split", required=True, choices=("dev", "test", "private"))
    build.add_argument("-n", type=_positive, required=True, help="number of tasks")
    build.add_argument("--out", required=True, help="dataset path stem (.json and .npz)")
    build.add_argument("--salt-env", help="environment variable holding the private split's secret salt")
    build.add_argument("--workers", type=_positive, default=1, help="parallel build processes")

    for name in ("baseline", "eval"):
        sub = commands.add_parser(name, help=f"score a {'reference baseline' if name == 'baseline' else 'model'}")
        sub.add_argument("--dataset", required=True)
        sub.add_argument("--out", required=True, help="results JSONL; transcripts go to <out>.transcripts.jsonl")
        sub.add_argument("--resume", action="store_true", help="append to an existing results file")
        sub.add_argument("--limit", type=_positive, default=None)
        sub.add_argument("--workers", type=_positive, default=1, help="concurrent episodes")
        sub.add_argument("--placeholder-names", action="store_true", help="rename columns X1..Xk (ablation)")
        sub.add_argument("--max-tool-calls", type=_positive, default=MAX_TOOL_CALLS)
        sub.add_argument("--max-turns", type=_positive, default=MAX_TURNS)
        sub.add_argument("--timeout", type=float, default=120.0, help="seconds per model request")
        if name == "baseline":
            sub.add_argument("--agent", choices=BASELINES, required=True)
        else:
            sub.add_argument("--model", action="append", required=True, help="LiteLLM model id; repeatable")

    report = commands.add_parser("report", help="summarize a results file with task-clustered 95% intervals")
    report.add_argument("results")
    group = report.add_mutually_exclusive_group()
    group.add_argument("--compare", help="second results file on the same tasks: paired difference")
    group.add_argument("--ablation", help="placeholder-names results on the same tasks: names effect")

    rescore = commands.add_parser("rescore", help="re-score saved answers with the current scorer")
    rescore.add_argument("results")
    rescore.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "build":
        return _build(args)
    if args.command == "report":
        return _report(args)
    if args.command == "rescore":
        return _rescore(args)
    return _evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
