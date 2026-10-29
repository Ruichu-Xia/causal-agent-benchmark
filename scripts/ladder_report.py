r"""Summarize base models on the environment-v2 dev splits, to check that difficulty is calibrated.

    python scripts/ladder_report.py --model qwen3-0.6b --model qwen3-1.7b --model qwen3-4b-2507 --model qwen3-8b

Replies come from ``modal_eval.py`` (``outputs/transfer/<split>_dev.jsonl``) and are graded by
``transfer_eval.judge``. Prints, per model: accuracy, no-answer rate, truncation rate and mean reply length by
split; accuracy by family, format and variant on each split; and accuracy by number of variables, pooling
``iid`` and ``ood_size``. Each cell shows its item count, so its binomial noise can be read off.
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from typing import Any

from transfer_eval import OUT, _load, items, judge

SPLITS = ("iid", "ood_wording", "ood_vocab", "ood_size")


def graded(split: str, model: str) -> list[dict[str, Any]]:
    """One row per answered item of a split: the task's attributes and the reply's grade."""
    items_by_id = {t["id"]: t for t in items(f"{split}_dev")}
    tasks = {t["id"]: t for t in _tasks(split)}
    rows = []
    for record in _load(OUT / f"{split}_dev.jsonl"):
        if record["model"] != model:
            continue
        task = tasks[record["id"]]
        correct, _, missing = judge(items_by_id[record["id"]], record["reply"])
        rows.append(
            {
                "id": record["id"],
                "split": split,
                "cell": f"{task['family']}/{task['kind']}/{task['meta']['variant']}",
                "size": len(task["meta"]["names"]),
                "family": task["family"],
                "correct": correct,
                "missing": missing,
                "truncated": record["finish_reason"] == "length",
                "tokens": record["completion_tokens"],
            }
        )
    if sorted(r["id"] for r in rows) != sorted(tasks):
        raise ValueError(f"{model} does not have exactly one reply to each of the {len(tasks)} {split} items")
    return rows


def _tasks(split: str) -> list[dict[str, Any]]:
    """The raw task records of a dev split, with their meta."""
    from causalagentbench.reasoning.dataset import load

    return load(OUT.parent / "reasoning" / "v2" / f"{split}_dev.jsonl")


def _mean(rows: list[dict[str, Any]], key: str) -> str:
    """Mean of one field over rows, with the row count."""
    return f"{statistics.mean(r[key] for r in rows):.2f} (n={len(rows)})" if rows else "-"


def report(model: str) -> None:
    """Print one model's tables."""
    rows = [r for split in SPLITS for r in graded(split, model)]
    print(f"\n## {model}\n\nsplit | accuracy | no answer | truncated | mean tokens")
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        print(
            f"{split} | {_mean(part, 'correct')} | {_mean(part, 'missing')} | {_mean(part, 'truncated')} | "
            f"{statistics.mean(r['tokens'] for r in part):.0f}"
        )
    print("\ncell | " + " | ".join(SPLITS))
    for cell in sorted({r["cell"] for r in rows}):
        values = [_mean([r for r in rows if r["cell"] == cell and r["split"] == s], "correct") for s in SPLITS]
        print(f"{cell} | " + " | ".join(values))
    print("\nfamily | accuracy by number of variables (iid and ood_size)")
    by_size: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["split"] in ("iid", "ood_size"):
            by_size[r["family"]][r["size"]].append(r)
    for family, sizes in sorted(by_size.items()):
        print(f"{family} | " + " | ".join(f"{n}: {_mean(sizes[n], 'correct')}" for n in sorted(sizes)))


def main() -> None:
    """Parse arguments and print every model's report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", required=True)
    for model in parser.parse_args().model:
        report(model)


if __name__ == "__main__":
    main()
