"""How well single-source rules and constant answers score on built tasks; correct analysis scores 1 by construction.

- ``data_only``: a perfect analysis of the population that ignores names, guessing uniformly among
  equivalence-class members where they disagree (its expected credit).
- ``names_only``: the answer in ``Gn``, the graph of CauseNet relations among the names.
- family shortcuts: the marginal correlation (independence), the confounder rule (adjustment), and the
  unadjusted or adjust-for-everything regression (effect).
- ``constant:<answer>``: the same answer every time.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import networkx as nx

from causalagentbench.dataenv.dataset import world_from_dict
from causalagentbench.dataenv.families import TOLERANCE, _ancestry, _member_effect, _population_regression
from causalagentbench.dataenv.world import DataWorld
from causalagentbench.reasoning.shortcuts import confounder_rule
from causalagentbench.reasoning.worlds import d_separated, valid_adjustment

CONSTANTS = {
    "yesno": ("yes", "no"),
    "letter": ("a", "b", "c"),
    "choice": ("1", "2", "both", "neither"),
    "number": ("0.00",),
}


def _answer_independence(independent: bool, meta: dict[str, Any]) -> str:
    """The yes/no answer for a dependence verdict, under the question's variant."""
    return "yes" if independent == (meta["variant"] == "independent") else "no"


def _independence(world: DataWorld, meta: dict[str, Any]) -> dict[str, float]:
    """Rule credits on an independence task."""
    x, y = meta["x"], meta["y"]
    truth = _answer_independence(meta["independent"], meta)
    names = world.name_graph()
    names_independent = not (nx.has_path(names, x, y) or nx.has_path(names, y, x))
    return {
        "data_only": 1.0,
        "names_only": float(_answer_independence(names_independent, meta) == truth),
        "marginal": float(_answer_independence(d_separated(world.graph(), x, y, set()), meta) == truth),
    }


def _structure(world: DataWorld, meta: dict[str, Any]) -> dict[str, float]:
    """Rule credits on a structure task."""
    x, y, truth = meta["x"], meta["y"], meta["truth"]
    members = world.member_graphs()
    return {
        "data_only": statistics.mean(_ancestry(member, x, y) == truth for member in members),
        "names_only": float(_ancestry(world.name_graph(), x, y) == truth),
    }


def _adjustment_answer(valid: Sequence[bool], variant: str) -> str:
    """The canonical answer for per-set validity verdicts."""
    if variant == "single":
        return "yes" if valid[0] else "no"
    return {(True, True): "both", (True, False): "1", (False, True): "2", (False, False): "neither"}[
        (valid[0], valid[1])
    ]


def _adjustment(world: DataWorld, meta: dict[str, Any]) -> dict[str, float]:
    """Rule credits on an adjustment task."""
    x, y, variant = meta["x"], meta["y"], meta["variant"]
    sets = [set(z) for z in meta["sets"]]
    graph = world.graph()
    truth = _adjustment_answer([valid_adjustment(graph, x, y, z) for z in sets], variant)
    members = world.member_graphs()
    by_member = [_adjustment_answer([valid_adjustment(m, x, y, z) for z in sets], variant) for m in members]
    return {
        "data_only": statistics.mean(answer == truth for answer in by_member),
        "confounder_rule": float(_adjustment_answer([confounder_rule(graph, x, y, z) for z in sets], variant) == truth),
    }


def _effect(world: DataWorld, meta: dict[str, Any]) -> dict[str, float]:
    """Rule credits on an effect task."""
    x, y, truth = meta["x"], meta["y"], meta["effect"]
    adjust_all = _population_regression(world, x, y, [v for v in range(world.n) if v not in (x, y)])
    members = world.member_graphs()
    return {
        "data_only": statistics.mean(abs(_member_effect(world, m, x, y) - truth) <= TOLERANCE for m in members),
        "unadjusted": float(abs(world.sigma[x, y] - truth) <= TOLERANCE),
        "adjust_all": float(abs(adjust_all - truth) <= TOLERANCE),
    }


RULES = {"independence": _independence, "structure": _structure, "adjustment": _adjustment, "effect": _effect}


def credits(record: dict[str, Any]) -> dict[str, float]:
    """Every rule's and constant answer's credit on one task record."""
    world = world_from_dict(record["world"])
    scores = RULES[record["family"]](world, record["meta"])
    for constant in CONSTANTS[record["kind"]]:
        if record["kind"] == "number":
            scores[f"constant:{constant}"] = float(abs(float(record["answer"]) - float(constant)) <= TOLERANCE)
        else:
            scores[f"constant:{constant}"] = float(record["answer"] == constant)
    return scores


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean credit of every rule, per family and answer format."""
    by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        group = f"{record['family']}/{record['kind']}"
        for rule, credit in credits(record).items():
            by_group[group][rule].append(credit)
    return {
        group: {rule: statistics.mean(values) for rule, values in sorted(rules.items())}
        | {"n": float(len(next(iter(rules.values()))))}
        for group, rules in sorted(by_group.items())
    }
