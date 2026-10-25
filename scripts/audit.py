"""Generator audit: gate acceptance, orientation leakage, and finite-sample discovery.

Usage::

    uv run python scripts/audit.py --attempts 3000 --out outputs/audit.json
    uv run python scripts/audit.py --attempts 0 --dataset artifacts/test --out outputs/audit_test.json

Leakage is measured on every edge the CPDAG leaves undirected. For each such edge ``u - v`` (``u < v``),
``p`` is the share of equivalent graphs that orient it ``u -> v``: the best any analyst can do from the
data alone is to predict with ``p``. A logistic model using ``p`` and graph structure is compared, on
held-out tasks, with one that also sees statistics of the data (R^2 of each endpoint given all others and
correlation with the outcome). If the data leak orientation, the second model predicts better.

On a named dataset the same comparison also covers selection by naming: tasks are accepted only when
CauseNet can name the true graph, which could favour some orientations. Calibration of ``p`` against the
realized orientations shows whether it did.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from causalagentbench.agent.discovery import discover_graph
from causalagentbench.scm.identifiability import equivalence_class, unoriented_edges
from causalagentbench.scoring.gates import attempt
from causalagentbench.task.dataset import load_dataset
from causalagentbench.types import SCM

AUDIT_NAMESPACE = 9_000
GATES = ("structure", "names_matter", "no_shortcut", "decoy_tempting", "effect_scale", "orientable")


def edge_rows(scm: SCM, task: int) -> list[dict[str, Any]]:
    """One row per CPDAG-undirected edge: true orientation, its share in the class, and data features."""
    scaffold = scm.scaffold
    members = [set(m) for m in equivalence_class(scaffold)]
    precision = np.linalg.inv(scm.sigma)
    r2 = 1.0 - 1.0 / np.diag(precision)  # R^2 of each node given all others (unit variances)
    y = scaffold.index(scaffold.outcome)
    skeleton = nx.Graph(scaffold.edges)
    rows = []
    for parent, child in unoriented_edges(scaffold):
        u, v = sorted((parent, child))
        iu, iv = scaffold.index(u), scaffold.index(v)
        rows.append(
            {
                "task": task,
                "label": int((u, v) == (parent, child)),
                "p": sum((u, v) in m for m in members) / len(members),
                "degree_diff": skeleton.degree(u) - skeleton.degree(v),
                "r2_diff": float(r2[iu] - r2[iv]),
                "y_corr_diff": float(abs(scm.sigma[iu, y]) - abs(scm.sigma[iv, y])),
            }
        )
    return rows


def _fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Ridge-regularized logistic regression by Newton's method; returns coefficients incl. intercept."""
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1])
    for _ in range(50):
        prob = 1.0 / (1.0 + np.exp(-design @ beta))
        gradient = design.T @ (y - prob) - ridge * beta
        hessian = design.T @ (design * (prob * (1 - prob))[:, None]) + ridge * np.eye(len(beta))
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def _predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Logistic predictions, clipped away from 0 and 1."""
    prob = 1.0 / (1.0 + np.exp(-(np.column_stack([np.ones(len(x)), x]) @ beta)))
    return np.clip(prob, 1e-6, 1 - 1e-6)


def leak_test(rows: list[dict[str, Any]], seed: int = 0, n_boot: int = 1000) -> dict[str, Any]:
    """Held-out accuracy and log-loss of structure-only vs structure-plus-data orientation models."""
    tasks = np.array(sorted({r["task"] for r in rows}))
    rng = np.random.default_rng(seed)
    test_tasks = set(rng.choice(tasks, size=len(tasks) * 3 // 10, replace=False).tolist())
    label = np.array([r["label"] for r in rows], dtype=float)
    logit_p = np.array(
        [math.log(min(max(r["p"], 1e-3), 1 - 1e-3) / (1 - min(max(r["p"], 1e-3), 1 - 1e-3))) for r in rows]
    )
    structure = np.column_stack([logit_p, [r["degree_diff"] for r in rows]])
    data = np.column_stack([structure, [r["r2_diff"] for r in rows], [r["y_corr_diff"] for r in rows]])
    test = np.array([r["task"] in test_tasks for r in rows])
    task_of = np.array([r["task"] for r in rows])
    results: dict[str, Any] = {"edges": len(rows), "tasks": len(tasks), "held_out_edges": int(test.sum())}
    losses = {}
    for name, x in (("structure_only", structure), ("structure_plus_data", data)):
        prob = _predict(_fit_logistic(x[~test], label[~test]), x[test])
        losses[name] = -(label[test] * np.log(prob) + (1 - label[test]) * np.log(1 - prob))
        results[name] = {
            "accuracy": float(np.mean((prob > 0.5) == label[test])),
            "log_loss": float(np.mean(losses[name])),
        }
    # Paired task bootstrap of the log-loss improvement from adding data features (positive = leak).
    gain = losses["structure_only"] - losses["structure_plus_data"]
    held = task_of[test]
    by_task = {t: gain[held == t] for t in np.unique(held)}
    keys = list(by_task)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys))
        draws.append(np.concatenate([by_task[keys[i]] for i in pick]).mean())
    results["log_loss_gain_from_data"] = {
        "value": float(gain.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }
    balanced = [r for r in rows if r["p"] == 0.5]
    if balanced:
        results["lower_r2_is_child_on_balanced_edges"] = {
            "edges": len(balanced),
            "accuracy": float(np.mean([(r["r2_diff"] > 0) == bool(r["label"]) for r in balanced])),
        }
    results["calibration"] = calibration(rows)
    return results


def calibration(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Realized share of ``u -> v`` orientations against the class share ``p``, in bins of ``p``."""
    bins = [(0.0, 0.3), (0.3, 0.45), (0.45, 0.55), (0.55, 0.7), (0.7, 1.0001)]
    table = []
    for low, high in bins:
        chosen = [r for r in rows if low <= r["p"] < high]
        if chosen:
            n = len(chosen)
            observed = sum(r["label"] for r in chosen) / n
            expected = sum(r["p"] for r in chosen) / n
            se = math.sqrt(sum(r["p"] * (1 - r["p"]) for r in chosen)) / n
            table.append(
                {
                    "p_low": low,
                    "p_high": min(high, 1.0),
                    "edges": n,
                    "expected": expected,
                    "observed": observed,
                    "se": se,
                }
            )
    return table


def acceptance(attempts: int) -> tuple[dict[str, Any], list[SCM]]:
    """Per-gate pass rates over generation attempts, and the accepted SCMs."""
    passed = dict.fromkeys(GATES, 0)
    accepted = []
    for i in range(attempts):
        scm, report = attempt(np.random.SeedSequence([AUDIT_NAMESPACE, i]))
        for g in GATES:
            passed[g] += getattr(report, g)
        if report.passed:
            accepted.append(scm)
    return {
        "attempts": attempts,
        "pass_rate": {g: passed[g] / attempts for g in GATES},
        "accepted_rate": len(accepted) / attempts,
    }, accepted


def discovery(dataset: str) -> dict[str, float]:
    """How well PC (the agent's ``discover`` tool, default alpha) recovers each task's true CPDAG."""
    true_pos = false_pos = false_neg = 0
    compelled = compelled_right = undirected = undirected_kept = 0
    for instance, key in load_dataset(dataset):
        scaffold = key.scm.scaffold
        graph = discover_graph(instance, 0.01)
        columns = instance.columns
        found_directed, found_undirected = set(), set()
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                head, tail = int(graph[i][j]), int(graph[j][i])
                a, b = key.label_to_node[columns[i]], key.label_to_node[columns[j]]
                if head == -1 and tail == -1:
                    found_undirected.add(frozenset((a, b)))
                elif head == -1 and tail == 1:
                    found_directed.add((a, b))
                elif head == 1 and tail == -1:
                    found_directed.add((b, a))
        found_skeleton = found_undirected | {frozenset(e) for e in found_directed}
        true_skeleton = {frozenset(e) for e in scaffold.edges}
        true_pos += len(found_skeleton & true_skeleton)
        false_pos += len(found_skeleton - true_skeleton)
        false_neg += len(true_skeleton - found_skeleton)
        open_edges = {frozenset(e) for e in unoriented_edges(scaffold)}
        for edge in scaffold.edges:
            if frozenset(edge) not in found_skeleton:
                continue
            if frozenset(edge) in open_edges:
                undirected += 1
                undirected_kept += frozenset(edge) in found_undirected
            else:
                compelled += 1
                compelled_right += edge in found_directed
    return {
        "skeleton_precision": true_pos / (true_pos + false_pos),
        "skeleton_recall": true_pos / (true_pos + false_neg),
        "compelled_edges_oriented_correctly": compelled_right / compelled,
        "open_edges_left_undirected": undirected_kept / undirected,
    }


def main() -> None:
    """Run the audit and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attempts", type=int, default=3000, help="generation attempts for the generator audit")
    parser.add_argument("--dataset", help="named dataset stem for the post-naming leak audit and PC recovery")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report: dict[str, Any] = {}
    if args.attempts:
        report["acceptance"], accepted = acceptance(args.attempts)
        rows = [row for task, scm in enumerate(accepted) for row in edge_rows(scm, task)]
        report["generator_leak"] = leak_test(rows)
    if args.dataset:
        tasks = load_dataset(args.dataset)
        rows = [row for index, (_, key) in enumerate(tasks) for row in edge_rows(key.scm, index)]
        report["named_dataset_leak"] = leak_test(rows)
        report["discovery"] = discovery(args.dataset)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
