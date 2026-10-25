"""Aggregate scored episodes into the headline metric with task-clustered uncertainty.

The headline is a ratio of sums over candidates, ``R = sum_t a_t / sum_t b_t``, where for task ``t``
``a_t`` is the sum of capped errors and ``b_t`` the sum of |true effects|. Tasks are the independent units,
so the standard error uses the delta method with tasks as clusters:

    SE(R) = sqrt( sum_t (a_t - R b_t)^2 / (T (T - 1)) ) / mean_t(b_t)

Two runs on the same tasks share ``b_t``, so their difference ``R1 - R2 = sum_t (a1_t - a2_t) / sum_t b_t``
gets the same formula with ``a_t`` replaced by the per-task difference: a paired comparison.

The names ablation runs the same tasks with real and placeholder names. Its statistic is a difference in
differences: how much placeholders raise the error on name-critical candidates, minus how much they raise
it on data-identified candidates (expected near zero). Its interval comes from a paired task bootstrap.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

Z_95 = 1.959964


def _ratio_with_se(numerators: Sequence[float], denominators: Sequence[float]) -> dict[str, float]:
    """Ratio of sums, its task-clustered delta-method SE, and a normal 95% interval."""
    n = len(numerators)
    total = sum(denominators)
    if total == 0:
        return {"value": math.nan, "se": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    ratio = sum(numerators) / total
    if n < 2:
        return {"value": ratio, "se": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    residual = sum((a - ratio * b) ** 2 for a, b in zip(numerators, denominators, strict=True))
    se = math.sqrt(residual / (n * (n - 1))) / (total / n)
    return {"value": ratio, "se": se, "ci_low": ratio - Z_95 * se, "ci_high": ratio + Z_95 * se}


def _task_sums(
    episode: dict[str, Any], keep: Callable[[dict[str, Any]], bool], field: str = "capped_error"
) -> tuple[float, float]:
    """Sum of ``field`` and of |truth| over an episode's candidates selected by ``keep``."""
    candidates = [c for c in episode["candidates"] if keep(c)]
    return sum(c[field] for c in candidates), sum(abs(c["truth"]) for c in candidates)


def _slice(episodes: Sequence[dict[str, Any]], keep: Callable[[dict[str, Any]], bool], field: str) -> dict[str, float]:
    """Pooled metric over the candidates selected by ``keep``, clustered by task."""
    sums = [_task_sums(e, keep, field) for e in episodes]
    return _ratio_with_se([a for a, _ in sums], [b for _, b in sums])


def summarize(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Headline and diagnostic metrics for one model's scored episodes."""
    if not episodes:
        raise ValueError("no scored episodes")
    candidates = [c for e in episodes for c in e["candidates"]]
    costs = [e["estimated_cost_usd"] for e in episodes]
    return {
        "tasks": len(episodes),
        "capped_error": _slice(episodes, lambda c: True, "capped_error"),
        "uncapped_error": _slice(episodes, lambda c: True, "error")["value"],
        "capped_candidates": sum(c["error"] > c["capped_error"] for c in candidates),
        "name_critical_error": _slice(episodes, lambda c: c["name_critical"], "capped_error"),
        "data_identified_error": _slice(episodes, lambda c: not c["name_critical"], "capped_error"),
        "adjustment_valid_rate": sum(c["adjustment_valid"] for c in candidates) / len(candidates),
        "answer_valid_rate": sum(e["parse_ok"] for e in episodes) / len(episodes),
        "mean_reward": sum(e["reward"] for e in episodes) / len(episodes),
        "mean_tool_calls": sum(e["tool_calls"] for e in episodes) / len(episodes),
        "mean_tokens": sum(e["prompt_tokens"] + e["completion_tokens"] for e in episodes) / len(episodes),
        "total_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
    }


def compare(first: Sequence[dict[str, Any]], second: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Paired difference in headline error (first minus second) over the tasks both runs scored."""
    a = {e["index"]: e for e in first}
    b = {e["index"]: e for e in second}
    shared = sorted(a.keys() & b.keys())
    if len(shared) < 2:
        raise ValueError("need at least two shared tasks to compare")
    differences, denominators = [], []
    for index in shared:
        a_err, total = _task_sums(a[index], lambda c: True)
        b_err, _ = _task_sums(b[index], lambda c: True)
        differences.append(a_err - b_err)
        denominators.append(total)
    return {"tasks": len(shared), "difference": _ratio_with_se(differences, denominators)}


def _is_critical(candidate: dict[str, Any]) -> bool:
    """Selector for name-critical candidates."""
    return bool(candidate["name_critical"])


def _is_identified(candidate: dict[str, Any]) -> bool:
    """Selector for data-identified candidates."""
    return not candidate["name_critical"]


def names_ablation(
    named: Sequence[dict[str, Any]], placeholder: Sequence[dict[str, Any]], *, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Effect of hiding the names, per slice, and its difference in differences, with bootstrap intervals."""
    a = {e["index"]: e for e in named}
    b = {e["index"]: e for e in placeholder}
    shared = sorted(a.keys() & b.keys())
    if len(shared) < 2:
        raise ValueError("need at least two shared tasks")

    def sums(episodes: dict[int, dict[str, Any]], keep: Callable[[dict[str, Any]], bool]) -> np.ndarray:
        """Per-task (error, |truth|) sums for one condition and slice."""
        return np.array([_task_sums(episodes[i], keep) for i in shared])

    parts = {
        (cond, name): sums(episodes, keep)
        for cond, episodes in (("named", a), ("placeholder", b))
        for name, keep in (("critical", _is_critical), ("identified", _is_identified))
    }

    def statistics(rows: np.ndarray) -> tuple[float, float, float]:
        """(placeholder - named) error on each slice, and their difference, for resampled task rows."""

        def ratio(cond: str, name: str) -> float:
            """Pooled error of one condition and slice over the resampled tasks."""
            values = parts[(cond, name)][rows]
            total = values[:, 1].sum()
            return float(values[:, 0].sum() / total) if total > 0 else math.nan

        critical = ratio("placeholder", "critical") - ratio("named", "critical")
        identified = ratio("placeholder", "identified") - ratio("named", "identified")
        return critical, identified, critical - identified

    point = statistics(np.arange(len(shared)))
    rng = np.random.default_rng(seed)
    draws = np.array([statistics(rng.integers(len(shared), size=len(shared))) for _ in range(n_boot)])
    low, high = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    names = ("name_critical_increase", "data_identified_increase", "difference_in_differences")
    return {
        "tasks": len(shared),
        **{
            name: {"value": point[i], "ci_low": float(low[i]), "ci_high": float(high[i])}
            for i, name in enumerate(names)
        },
    }
