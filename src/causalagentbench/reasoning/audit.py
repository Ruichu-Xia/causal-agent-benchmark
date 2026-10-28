"""Audit generated tasks: re-derive every answer from the stored problem, and score shortcuts and constants.

The audit recomputes everything from ``meta`` with the verifier functions, independently of the
sampling code, so a rendering or bookkeeping bug in a family shows up as a label mismatch.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from statistics import mean
from typing import Any

import networkx as nx

from causalagentbench.reasoning.families import _number, _relations
from causalagentbench.reasoning.grading import grade
from causalagentbench.reasoning.shortcuts import confounder_rule, undirected_separated
from causalagentbench.reasoning.worlds import World, causal_effect, d_separated, observed_difference, valid_adjustment
from causalagentbench.scm.identifiability import equivalence_class
from causalagentbench.types import Role, Scaffold

CHOICES = {(True, True): "both", (True, False): "1", (False, True): "2", (False, False): "neither"}


def _graph(meta: dict[str, Any]) -> nx.DiGraph:
    """The task's DAG on node indices."""
    graph = nx.DiGraph([tuple(e) for e in meta["edges"]])
    graph.add_nodes_from(range(len(meta["names"])))
    return graph


def _independence(meta: dict[str, Any]) -> tuple[str, str]:
    """(answer, undirected-separation shortcut answer)."""
    graph, x, y, z = _graph(meta), meta["x"], meta["y"], set(meta["z"])
    flip = meta["variant"] == "associated"
    truth = d_separated(graph, x, y, z) != flip
    shortcut = undirected_separated(graph, x, y, z) != flip
    return ("yes" if truth else "no"), ("yes" if shortcut else "no")


def _adjustment(meta: dict[str, Any]) -> tuple[str, str]:
    """(answer, confounder-rule shortcut answer)."""
    graph, x, y = _graph(meta), meta["x"], meta["y"]
    sets = [set(z) for z in meta["sets"]]
    truth = [valid_adjustment(graph, x, y, z) for z in sets]
    shortcut = [confounder_rule(graph, x, y, z) for z in sets]
    if meta["variant"] == "single":
        return ("yes" if truth[0] else "no"), ("yes" if shortcut[0] else "no")
    return CHOICES[(truth[0], truth[1])], CHOICES[(shortcut[0], shortcut[1])]


def _effect(meta: dict[str, Any]) -> tuple[str, str | None]:
    """(answer, observed-difference shortcut answer, or None for variants without one)."""
    world = World(tuple(meta["names"]), tuple(tuple(e) for e in meta["edges"]), tuple(tuple(c) for c in meta["cpts"]))
    ate, observed = causal_effect(world, meta["x"], meta["y"]), observed_difference(world, meta["x"], meta["y"])
    variant = meta["variant"]
    if variant == "causal":
        return _number(ate), _number(observed)
    if variant == "observed":
        return _number(observed), None
    if variant == "sign":
        return ("yes" if ate > 0 else "no"), ("yes" if observed > 0 else "no")
    return ("yes" if observed > ate else "no"), None


def _discovery(meta: dict[str, Any]) -> tuple[str, None]:
    """(answer, None): whether the statement holds in every (or some) DAG of the equivalence class."""
    nodes = tuple(str(i) for i in range(len(meta["names"])))
    edges = tuple((str(p), str(c)) for p, c in meta["edges"])
    scaffold = Scaffold(nodes, dict.fromkeys(nodes, Role.CONFOUNDER), edges, (), nodes[0])
    key = (meta["relation"], str(meta["a"]), str(meta["b"]))
    holds = [_relations(member, nodes)[key] for member in equivalence_class(scaffold)]
    label = all(holds) if meta["variant"] == "must" else any(holds)
    return ("yes" if label else "no"), None


DERIVE: dict[str, Callable[[dict[str, Any]], tuple[str, str | None]]] = {
    "independence": _independence,
    "adjustment": _adjustment,
    "effect": _effect,
    "discovery": _discovery,
}


def _score(kind: str, answer: str, given: str) -> float:
    """The grader's score for replying ``given``."""
    return grade(kind, answer, f"Answer: {given}")


def audit(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Label mismatches, and per (family, kind): best constant answer and shortcut accuracy.

    For discovery, the best constant is taken within each (relation, mode) cell, the shortcut that a
    policy could learn from the question type alone.
    """
    mismatches = []
    groups: dict[tuple[str, str], list[tuple[dict[str, Any], str | None]]] = defaultdict(list)
    for task in tasks:
        truth, shortcut = DERIVE[task["family"]](task["meta"])
        if truth != task["answer"]:
            mismatches.append(task["id"])
        groups[(task["family"], task["kind"])].append((task, shortcut))
    report: dict[str, Any] = {"tasks": len(tasks), "label_mismatches": mismatches, "groups": {}}
    for (family, kind), members in sorted(groups.items()):
        answers = Counter(t["answer"] for t, _ in members)
        candidates = [*answers, "0"]  # every answer seen, and zero
        constant = max(mean(_score(kind, t["answer"], c) for t, _ in members) for c in candidates)
        shortcut_scores = [_score(kind, t["answer"], s) for t, s in members if s is not None]
        entry: dict[str, Any] = {
            "n": len(members),
            "answers": dict(answers.most_common(6)),
            "best_constant": round(constant, 3),
            "shortcut": round(mean(shortcut_scores), 3) if shortcut_scores else None,
            "shortcut_n": len(shortcut_scores),
        }
        if family == "discovery":
            cells: dict[tuple[str, str], list[str]] = defaultdict(list)
            for t, _ in members:
                cells[(t["meta"]["relation"], t["meta"]["variant"])].append(t["answer"])
            right = sum(max(Counter(v).values()) for v in cells.values())
            entry["best_constant_per_cell"] = round(right / len(members), 3)
        report["groups"][f"{family}/{kind}"] = entry
    return report
