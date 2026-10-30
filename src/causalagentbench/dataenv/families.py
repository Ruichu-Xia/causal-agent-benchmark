"""Question families over named data worlds: each draws a world and a query in a target cell, with its answer.

- ``independence``: are two columns dependent given a set? (d-separation in the true DAG ``D``)
- ``structure``: does one column cause the other, the reverse, or neither? (ancestry in ``D``)
- ``adjustment``: does a regression adjusting for a set recover the causal effect? (the adjustment criterion)
- ``effect``: the total causal effect in standard-deviation units. (``D``'s weights)

Two reference rules show what one source of evidence gives alone (``docs/environment_v3.md``):
*data-only* answers in every member of ``D``'s equivalence class, and *names-only* answers in ``Gn``, the
graph of CauseNet relations among the column names. Each family first draws the cell it wants (the label
and which rules are right), then draws worlds until one has a query in that cell whose sampled rows also
support the label.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from causalagentbench.dataenv.wording import SLOTS, instruction
from causalagentbench.dataenv.world import ROWS, DataWorld, draw_world
from causalagentbench.generation.causenet import CauseNetLabeler
from causalagentbench.reasoning.shortcuts import confounder_rule
from causalagentbench.reasoning.worlds import d_separated, valid_adjustment

MAX_WORLDS = 50_000
"""Worlds drawn before giving up on a target cell. The rarest effect cells on 7- to 8-node graphs, where most
draws fail the data check, need more than 5,000."""
SIZES = ((4, 6), (7, 8))
"""(training, ood_size) inclusive ranges of the number of variables."""
MAX_GIVEN = 3
"""Largest conditioning or adjustment set asked about."""
TOLERANCE = 0.10
"""Graded tolerance of a numeric effect answer, in standard deviations."""
FAR = 0.20
"""A value more than this from the truth counts as clearly wrong for a rule or an equivalence-class member."""
NEAR = 0.02
"""Every equivalence-class member within this of the truth makes an effect identified from data."""
ORACLE = 0.03
"""The regression on ``D``'s parents, on the task's own rows, must land this close to the true effect."""
P_NONZERO = 0.9
"""Share of effect questions whose answer is not zero, so answering zero is not a shortcut."""
ALPHA_DEPENDENT, ALPHA_INDEPENDENT = 0.01, 0.05
"""A dependent query needs Fisher-z p below the first on its rows, an independent one p above the second."""


@dataclass(frozen=True)
class Spec:
    """How a split draws problems.

    Attributes:
        large: Draw graph sizes from the ood_size range.
        held_out_wording: Use held-out templates.
    """

    large: bool = False
    held_out_wording: bool = False


@dataclass(frozen=True)
class Task:
    """One question about one dataset, with its exact answer.

    Attributes:
        family: The question family.
        kind: The answer format (``wording.FORMATS``).
        prompt: The full user message.
        answer: The canonical answer.
        world: The world, whose rows the tool loads.
        meta: The query and its cell, for audits.
    """

    family: str
    kind: str
    prompt: str
    answer: str
    world: DataWorld
    meta: dict[str, Any] = field(default_factory=dict)


class Phrases:
    """Draws templates from the training or held-out pool of each slot."""

    def __init__(self, rng: np.random.Generator, held_out: bool) -> None:
        """Bind the surface random stream and the template pool."""
        self.rng = rng
        self.held_out = held_out

    def __call__(self, slot: str, **fill: Any) -> str:
        """A random template of ``slot``, filled in."""
        pool = SLOTS[slot][1 if self.held_out else 0]
        return pool[int(self.rng.integers(len(pool)))].format(**fill)


def _join(words: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _prompt(world: DataWorld, phrase: Phrases, question: str, kind: str) -> str:
    """Context (data, assumptions, names, tools), the question, and the answer instruction."""
    order = [int(i) for i in phrase.rng.permutation(world.n)]
    columns = ", ".join(world.names[i] for i in order)
    context = " ".join(
        [
            phrase("intro", rows=f"{ROWS:,}", n=world.n, columns=columns),
            phrase("assumptions"),
            phrase("names"),
        ]
    )
    return f"{context}\n\n{question}\n\n{phrase('tools')}\n\n{instruction(kind)}"


def _size(rng: np.random.Generator, spec: Spec) -> int:
    """A graph size from this split's range."""
    lo, hi = SIZES[spec.large]
    return int(rng.integers(lo, hi + 1))


def _search(
    rng: np.random.Generator, spec: Spec, labeler: CauseNetLabeler, find: Callable[[DataWorld], Any | None]
) -> tuple[DataWorld, Any]:
    """Draw worlds until ``find`` returns a query for one."""
    for _ in range(MAX_WORLDS):
        world = draw_world(rng, _size(rng, spec), labeler)
        if world is None:
            continue
        query = find(world)
        if query is not None:
            return world, query
    raise RuntimeError("no world had a query in the target cell")


def _pick(rng: np.random.Generator, options: Sequence[Any]) -> Any | None:
    """A random element, or None if there is none."""
    return options[int(rng.integers(len(options)))] if options else None


def _subsets(others: Sequence[int]) -> list[tuple[int, ...]]:
    """Every subset of ``others`` with at most ``MAX_GIVEN`` elements."""
    return [z for k in range(min(MAX_GIVEN, len(others)) + 1) for z in itertools.combinations(others, k)]


def _related(graph: nx.DiGraph, a: int, b: int) -> bool:
    """Whether ``graph`` has a directed path between ``a`` and ``b`` in either direction."""
    return bool(nx.has_path(graph, a, b) or nx.has_path(graph, b, a))


def fisher_z_p(rows: np.ndarray, x: int, y: int, z: Sequence[int]) -> float:
    """Two-sided p-value of the Fisher-z test of zero partial correlation of ``x`` and ``y`` given ``z``."""
    columns = [x, y, *z]
    precision = np.linalg.inv(np.corrcoef(rows[:, columns], rowvar=False))
    r = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
    statistic = np.sqrt(len(rows) - len(z) - 3) * abs(np.arctanh(r))
    return math.erfc(statistic / math.sqrt(2))


def independence(rng: np.random.Generator, phrase: Phrases, spec: Spec, labeler: CauseNetLabeler) -> Task:
    """Are two columns dependent given a set? Label, names-only rule and marginal rule are drawn independently."""
    label, names_right, marginal_right = (bool(rng.random() < 0.5) for _ in range(3))

    def find(world: DataWorld) -> tuple[int, int, tuple[int, ...]] | None:
        """A query (x, y, Z) in the target cell whose rows support the label."""
        graph, names = world.graph(), world.name_graph()
        cell = []
        for x, y in itertools.combinations(range(world.n), 2):
            names_independent = not _related(names, x, y)
            for z in _subsets([v for v in range(world.n) if v not in (x, y)]):
                independent = d_separated(graph, x, y, set(z))
                if (
                    independent == label
                    and (names_independent == independent) == names_right
                    and (d_separated(graph, x, y, set()) == independent) == marginal_right
                ):
                    cell.append((x, y, z))
        query = _pick(rng, cell)
        if query is None:
            return None
        p = fisher_z_p(world.data(), *query)
        return query if (p > ALPHA_INDEPENDENT if label else p < ALPHA_DEPENDENT) else None

    world, (x, y, z) = _search(rng, spec, labeler, find)
    if rng.random() < 0.5:
        x, y = y, x
    ask_independent = bool(rng.random() < 0.5)
    names = world.names
    where = phrase("where_given", z=_join([names[v] for v in z])) if z else phrase("where_none")
    question = phrase("ask_independent" if ask_independent else "ask_associated", x=names[x], y=names[y], where=where)
    answer = "yes" if label == ask_independent else "no"
    meta = {
        "x": x,
        "y": y,
        "z": list(z),
        "independent": label,
        "names_right": names_right,
        "marginal_right": marginal_right,
        "variant": "independent" if ask_independent else "associated",
    }
    return Task("independence", "yesno", _prompt(world, phrase, question, "yesno"), answer, world, meta)


def _ancestry(graph: nx.DiGraph, a: int, b: int) -> str:
    """'ab' if ``a`` causes ``b``, 'ba' if the reverse, 'none' if neither, 'both' if paths run both ways."""
    forward, backward = nx.has_path(graph, a, b), nx.has_path(graph, b, a)
    return {(True, False): "ab", (False, True): "ba", (False, False): "none", (True, True): "both"}[(forward, backward)]


STRUCTURE_CELLS = (("name_critical", 0.5), ("data_critical", 0.25), ("both", 0.25))
"""Which source settles a structure question, and how often each cell is drawn."""


def structure(rng: np.random.Generator, phrase: Phrases, spec: Spec, labeler: CauseNetLabeler) -> Task:
    """Which of 'x causes y', 'y causes x' and 'neither' holds? The cell says which source settles it.

    - ``name_critical``: one causes the other, the equivalence class disagrees, and the names are right;
    - ``data_critical``: neither causes the other in every class member, but the names suggest a link;
    - ``both``: the class agrees and the names are right.
    """
    cells = [cell for cell, _ in STRUCTURE_CELLS]
    target = cells[int(rng.choice(len(cells), p=[weight for _, weight in STRUCTURE_CELLS]))]

    def find(world: DataWorld) -> tuple[int, int] | None:
        """A pair in the target cell."""
        graph, names, members = world.graph(), world.name_graph(), world.member_graphs()
        cell = []
        for a, b in itertools.combinations(range(world.n), 2):
            truth = _ancestry(graph, a, b)
            determined = len({_ancestry(member, a, b) for member in members}) == 1
            names_say = _ancestry(names, a, b)
            if target == "name_critical":
                matched = truth != "none" and not determined and names_say == truth
            elif target == "data_critical":
                matched = truth == "none" and determined and names_say != "none"
            else:
                matched = determined and names_say == truth
            if matched:
                cell.append((a, b))
        return _pick(rng, cell)

    world, (x, y) = _search(rng, spec, labeler, find)
    if rng.random() < 0.5:
        x, y = y, x
    names = world.names
    statements = {
        "ab": phrase("causes", x=names[x], y=names[y]),
        "ba": phrase("causes", x=names[y], y=names[x]),
        "none": phrase("neither", x=names[x], y=names[y]),
    }
    order = [list(statements)[int(i)] for i in rng.permutation(3)]
    # Column names keep their exact case, so an option may start in lower case.
    options = "\n".join(f"{letter}. {statements[key]}" for letter, key in zip("ABC", order, strict=True))
    truth = _ancestry(world.graph(), x, y)
    answer = "ABC"[order.index(truth)].lower()
    question = phrase("ask_structure", options=options)
    meta = {"x": x, "y": y, "truth": truth, "cell": target, "order": order}
    return Task("structure", "letter", _prompt(world, phrase, question, "letter"), answer, world, meta)


def _adjustment_cell(rng: np.random.Generator, determined: bool) -> tuple[bool, bool, bool]:
    """A random (valid, data-determined, confounder rule right) cell for one adjustment set.

    The cell (invalid, undetermined, confounder rule wrong) does not occur: none in about 150,000 sets of
    random 4- to 7-node graphs. So a valid set's confounder-rule agreement is a fair coin, and an invalid
    set's is fixed by ``determined``, which keeps "confounder rule right" at 1/2 given the label.
    """
    if rng.random() < 0.5:
        return True, determined, bool(rng.random() < 0.5)
    return False, determined, not determined


def adjustment(rng: np.random.Generator, phrase: Phrases, spec: Spec, labeler: CauseNetLabeler) -> Task:
    """Does regressing on X and Z recover X's total effect on Y? Each set's cell comes from ``_adjustment_cell``.

    Whether the data determine the sets' validity is drawn once per question: two sets of one pair with
    different answers to it are rare.
    """
    variant = str(rng.choice(["single", "compare"]))
    determined = bool(rng.random() < 0.5)
    targets = [_adjustment_cell(rng, determined) for _ in range(1 + (variant == "compare"))]

    def find(world: DataWorld) -> tuple[int, int, list[tuple[int, ...]]] | None:
        """A pair (x, y), x not caused by y, with distinct sets in every target cell."""
        graph, members = world.graph(), world.member_graphs()
        pairs = [(x, y) for x, y in itertools.permutations(range(world.n), 2) if not nx.has_path(graph, y, x)]
        for x, y in [pairs[int(i)] for i in rng.permutation(len(pairs))]:
            by_cell: dict[tuple[bool, ...], list[tuple[int, ...]]] = {}
            for z in _subsets([v for v in range(world.n) if v not in (x, y)]):
                valid = valid_adjustment(graph, x, y, set(z))
                determined = len({valid_adjustment(member, x, y, set(z)) for member in members}) == 1
                confounder_right = confounder_rule(graph, x, y, set(z)) == valid
                by_cell.setdefault((valid, determined, confounder_right), []).append(z)
            if all(len(by_cell.get(t, [])) >= targets.count(t) for t in targets):
                chosen: list[tuple[int, ...]] = []
                for t in targets:
                    pool = [z for z in by_cell[t] if z not in chosen]
                    chosen.append(pool[int(rng.integers(len(pool)))])
                return x, y, chosen
        return None

    world, (x, y, sets) = _search(rng, spec, labeler, find)
    names = world.names
    fill = {"x": names[x], "y": names[y]}

    def procedure(z: tuple[int, ...]) -> str:
        """The regression adjusting for ``z``, in a random template."""
        if not z:
            return phrase("proc_empty", **fill)
        return phrase("proc_set", z=_join([names[v] for v in z]), **fill)

    valid = [t[0] for t in targets]
    if variant == "single":
        question = phrase("ask_single", proc=procedure(sets[0]), **fill)
        kind, answer = "yesno", "yes" if valid[0] else "no"
    else:
        question = phrase("ask_compare", proc1=procedure(sets[0]), proc2=procedure(sets[1]), **fill)
        kind = "choice"
        answer = {(True, True): "both", (True, False): "1", (False, True): "2", (False, False): "neither"}[
            (valid[0], valid[1])
        ]
    meta = {
        "x": x,
        "y": y,
        "sets": [list(z) for z in sets],
        "cells": [dict(zip(("valid", "determined", "confounder_right"), t, strict=True)) for t in targets],
        "variant": variant,
    }
    return Task("adjustment", kind, _prompt(world, phrase, question, kind), answer, world, meta)


def _member_effect(world: DataWorld, member: nx.DiGraph, x: int, y: int) -> float:
    """The effect of ``x`` on ``y`` that ``member`` implies from Sigma: regress ``y`` on ``x`` and its parents."""
    rows = [x, *sorted(member.predecessors(x))]
    return float(np.linalg.solve(world.sigma[np.ix_(rows, rows)], world.sigma[rows, y])[0])


def _population_regression(world: DataWorld, x: int, y: int, z: Sequence[int]) -> float:
    """The population coefficient of ``x`` in the regression of ``y`` on ``x`` and ``z``."""
    rows = [x, *z]
    return float(np.linalg.solve(world.sigma[np.ix_(rows, rows)], world.sigma[rows, y])[0])


def _shortcut_matches(value: float, truth: float, right: bool) -> bool:
    """Whether a shortcut's value is within the tolerance of the truth (``right``) or more than ``FAR`` off."""
    miss = abs(value - truth)
    return miss <= TOLERANCE if right else miss > FAR


def _regression(rows: np.ndarray, x: int, y: int, z: Sequence[int]) -> float:
    """The OLS coefficient of ``x`` in the regression of ``y`` on ``x`` and ``z`` (with an intercept)."""
    design = np.column_stack([rows[:, [x, *z]], np.ones(len(rows))])
    return float(np.linalg.lstsq(design, rows[:, y], rcond=None)[0][0])


def _number(value: float) -> str:
    """Two-decimal canonical form; never "-0.00"."""
    return f"{round(value, 2) + 0.0:.2f}"


def effect(rng: np.random.Generator, phrase: Phrases, spec: Spec, labeler: CauseNetLabeler) -> Task:
    """Total effect in SD units.

    Drawn independently: nonzero, name-critical, and whether each shortcut is right (within the tolerance) or
    clearly wrong (more than ``FAR`` off): the unadjusted correlation, and the regression on every other column.
    """
    nonzero = bool(rng.random() < P_NONZERO)
    name_critical, unadjusted_right, adjust_all_right = (bool(rng.random() < 0.5) for _ in range(3))

    def find(world: DataWorld) -> tuple[int, int] | None:
        """An ordered pair in the target cell whose rows support the label."""
        graph, members = world.graph(), world.member_graphs()
        cell = []
        for x, y in itertools.permutations(range(world.n), 2):
            truth = world.total_effect(x, y)
            if nonzero != (abs(truth) >= TOLERANCE) or (not nonzero and nx.has_path(graph, x, y)):
                continue
            spread = max(abs(_member_effect(world, member, x, y) - truth) for member in members)
            if not (spread > FAR if name_critical else spread <= NEAR):
                continue
            if not _shortcut_matches(world.sigma[x, y], truth, unadjusted_right):
                continue
            others = [v for v in range(world.n) if v not in (x, y)]
            if not _shortcut_matches(_population_regression(world, x, y, others), truth, adjust_all_right):
                continue
            cell.append((x, y))
        query = _pick(rng, cell)
        if query is None:
            return None
        x, y = query
        estimate = _regression(world.data(), x, y, sorted(graph.predecessors(x)))
        return query if abs(estimate - world.total_effect(x, y)) <= ORACLE else None

    world, (x, y) = _search(rng, spec, labeler, find)
    names = world.names
    question = phrase("ask_effect", x=names[x], y=names[y])
    truth = world.total_effect(x, y)
    meta = {
        "x": x,
        "y": y,
        "effect": truth,
        "nonzero": nonzero,
        "name_critical": name_critical,
        "unadjusted_right": unadjusted_right,
        "adjust_all_right": adjust_all_right,
        "tolerance": TOLERANCE,
    }
    return Task("effect", "number", _prompt(world, phrase, question, "number"), _number(truth), world, meta)


FAMILIES: dict[str, Callable[[np.random.Generator, Phrases, Spec, CauseNetLabeler], Task]] = {
    "independence": independence,
    "structure": structure,
    "adjustment": adjustment,
    "effect": effect,
}
