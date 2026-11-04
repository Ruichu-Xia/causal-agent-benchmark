"""Transfer tasks on published Gaussian Bayesian networks (bnRep), in environment v3's four question families.

Each network in ``data/bnrep`` (exported by ``scripts/bnrep_export.R``) has a published structure and
coefficients. Standardizing it gives a correlation matrix, which ``world._scm`` turns back into the same
graph's unit-variance weights, so answers use the same definitions as the generated tasks: d-separation,
ancestry, the adjustment criterion and the total effect in standard deviations. Only the graph, the
coefficients and the names come from outside the generator. Every item draws its own 2,000 rows.

Items come in pairs: the same query, rows and wording, once with the authors' variable names and once with
placeholders ``V1``..``Vn`` in a random order. Each item's target label is drawn once, before its search
(independent with probability 1/2, causally related with probability 2/3, each adjustment set valid with
probability 1/2, a nonzero effect with probability ``P_NONZERO``), and each item passes the same data checks
as the generated tasks. The names sentence of the prompt says what the
names are; the shortcut cells are recorded, not controlled.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from causalagentbench.families import (
    ALPHA_DEPENDENT,
    ALPHA_INDEPENDENT,
    NEAR,
    ORACLE,
    P_NONZERO,
    TOLERANCE,
    Phrases,
    Task,
    _ancestry,
    _member_effect,
    _population_regression,
    _regression,
    _subsets,
    adjustment_task,
    effect_task,
    fisher_z_p,
    independence_task,
    structure_task,
)
from causalagentbench.graphs import confounder_rule, d_separated, equivalence_class, valid_adjustment
from causalagentbench.scm import marginal_covariance
from causalagentbench.world import ROWS, DataWorld, _scaffold, _scm, moments, simulate

NETWORKS = ("algal2", "expenditure", "lexical", "liquefaction")
SEED = 300
"""Build seed; training, dev and test use 0, 100 and 200."""
NAMES_SENTENCE = {
    "real": "The column names are the variable names used by the published study this causal model comes from.",
    "placeholder": "The column names are placeholders and say nothing about what the variables are.",
}


class FixedNames(Phrases):
    """Training templates, except a fixed names sentence that consumes no randomness."""

    def __init__(self, rng: np.random.Generator, sentence: str) -> None:
        """Bind the surface random stream and the names sentence."""
        super().__init__(rng, held_out=False)
        self.sentence = sentence

    def __call__(self, slot: str, **fill: Any) -> str:
        """The fixed sentence for the names slot, else a random training template."""
        return self.sentence if slot == "names" else super().__call__(slot, **fill)


def network_world(path: Path) -> DataWorld:
    """The standardized world of one exported network, with the authors' names and no rows yet (seed 0).

    Nodes, and so the data's columns, come in a random order: the export lists them in its own order, and
    a column order that follows the causal order would give away directions.
    """
    entries = json.loads(path.read_text(encoding="utf-8"))["nodes"]
    order = np.random.default_rng([SEED, NETWORKS.index(path.stem), 98]).permutation(len(entries))
    entries = [entries[int(i)] for i in order]
    names = tuple(entry["node"] for entry in entries)
    index = {name: i for i, name in enumerate(names)}
    weights = np.zeros((len(names), len(names)))
    for entry in entries:
        for parent, coefficient in entry["parents"].items():
            weights[index[parent], index[entry["node"]]] = coefficient
    covariance = marginal_covariance(weights, np.array([entry["sd"] ** 2 for entry in entries]))
    scale = np.sqrt(np.diag(covariance))
    sigma = covariance / np.outer(scale, scale)
    edges = tuple(sorted((int(i), int(j)) for i, j in zip(*np.nonzero(weights), strict=True)))
    scm = _scm(edges, sigma)
    members = tuple(
        tuple(sorted((int(p), int(c)) for p, c in member)) for member in equivalence_class(_scaffold(edges, len(names)))
    )
    return DataWorld(
        names=names,
        edges=edges,
        members=members,
        name_edges=(),
        weights=np.array(scm.W),
        noise_var=np.array(scm.noise_var),
        sigma=sigma,
        data_seed=0,
        data_moments=(),
    )


def with_rows(world: DataWorld, rng: np.random.Generator) -> DataWorld:
    """The world with a fresh data seed and the moments of its rows."""
    seed = int(rng.integers(2**63))
    rows = simulate(world.edges, world.weights, world.noise_var, ROWS, seed)
    return replace(world, data_seed=seed, data_moments=moments(rows))


def _independence(rng: np.random.Generator, world: DataWorld, label: bool) -> tuple[Any, ...] | None:
    """A query (x, y, Z) with the target label whose rows support it, with its key and record."""
    graph = world.graph()
    pool = [
        (x, y, z)
        for x, y in itertools.combinations(range(world.n), 2)
        for z in _subsets([v for v in range(world.n) if v not in (x, y)])
        if d_separated(graph, x, y, set(z)) == label
    ]
    if not pool:
        return None
    query = pool[int(rng.integers(len(pool)))]
    p = fisher_z_p(world.data(), *query)
    if not (p > ALPHA_INDEPENDENT if label else p < ALPHA_DEPENDENT):
        return None
    x, y, _ = query
    return query, (query, label, {"marginal_right": d_separated(graph, x, y, set()) == label})


def _structure(rng: np.random.Generator, world: DataWorld, related: bool) -> tuple[Any, ...] | None:
    """A pair that is causally related or not, as targeted, and whether the data alone settle its ancestry."""
    graph, members = world.graph(), world.member_graphs()
    pool = [
        (a, b) for a, b in itertools.combinations(range(world.n), 2) if (_ancestry(graph, a, b) != "none") == related
    ]
    if not pool:
        return None
    a, b = pool[int(rng.integers(len(pool)))]
    determined = len({_ancestry(member, a, b) for member in members}) == 1
    return (a, b), ((a, b), {"determined": determined})


def _adjustment(rng: np.random.Generator, world: DataWorld, wanted: tuple[bool, ...]) -> tuple[Any, ...] | None:
    """A pair and one or two distinct sets with the targeted validities, with their cells."""
    graph, members = world.graph(), world.member_graphs()
    pairs = [(x, y) for x, y in itertools.permutations(range(world.n), 2) if not nx.has_path(graph, y, x)]
    x, y = pairs[int(rng.integers(len(pairs)))]
    by_validity: dict[bool, list[tuple[int, ...]]] = {True: [], False: []}
    for z in _subsets([v for v in range(world.n) if v not in (x, y)]):
        by_validity[valid_adjustment(graph, x, y, set(z))].append(z)
    sets: list[tuple[int, ...]] = []
    for valid in wanted:
        pool = [z for z in by_validity[valid] if z not in sets]
        if not pool:
            return None
        sets.append(pool[int(rng.integers(len(pool)))])
    targets = []
    for z, valid in zip(sets, wanted, strict=True):
        determined = len({valid_adjustment(member, x, y, set(z)) for member in members}) == 1
        targets.append((valid, determined, confounder_rule(graph, x, y, set(z)) == valid))
    variant = "single" if len(wanted) == 1 else "compare"
    return ((x, y), tuple(sets)), ((x, y), sets, targets, variant)


def _effect(rng: np.random.Generator, world: DataWorld, nonzero: bool) -> tuple[Any, ...] | None:
    """An ordered pair with a nonzero or zero effect, as targeted, whose rows pass the oracle check."""
    graph, members = world.graph(), world.member_graphs()
    pool = []
    for x, y in itertools.permutations(range(world.n), 2):
        truth = world.total_effect(x, y)
        if nonzero == (abs(truth) >= TOLERANCE) and (nonzero or not nx.has_path(graph, x, y)):
            pool.append((x, y))
    if not pool:
        return None
    x, y = pool[int(rng.integers(len(pool)))]
    truth = world.total_effect(x, y)
    if abs(_regression(world.data(), x, y, sorted(graph.predecessors(x))) - truth) > ORACLE:
        return None
    others = [v for v in range(world.n) if v not in (x, y)]
    spread = max(abs(_member_effect(world, member, x, y) - truth) for member in members)
    cell = {
        "nonzero": nonzero,
        "determined": bool(spread <= NEAR),
        "unadjusted_right": bool(abs(world.sigma[x, y] - truth) <= TOLERANCE),
        "adjust_all_right": bool(abs(_population_regression(world, x, y, others) - truth) <= TOLERANCE),
    }
    return (x, y), ((x, y), cell)


def _target(family: str, rng: np.random.Generator) -> Any:
    """The label an item aims for, drawn once before its search, as in the training families."""
    if family == "independence":
        return bool(rng.random() < 0.5)
    if family == "structure":
        return bool(rng.random() < 2 / 3)  # 'neither' a third of the time; the swap balances the two directions
    if family == "adjustment":
        return tuple(bool(rng.random() < 0.5) for _ in range(1 + (str(rng.choice(["single", "compare"])) == "compare")))
    return bool(rng.random() < P_NONZERO)


FINDERS: dict[str, Callable[[np.random.Generator, DataWorld, Any], tuple[Any, ...] | None]] = {
    "independence": _independence,
    "structure": _structure,
    "adjustment": _adjustment,
    "effect": _effect,
}
MAX_TRIES = 10_000
"""Row draws before giving up on an item."""


def _render(
    family: str, found: tuple[Any, ...], surface: np.random.Generator, phrase: Phrases, world: DataWorld
) -> Task:
    """The task for a found query, rendered with the training families' question code."""
    if family == "independence":
        query, label, cell = found
        return independence_task(surface, phrase, world, query, label, cell)
    if family == "structure":
        pair, cell = found
        return structure_task(surface, phrase, world, pair, cell)
    if family == "adjustment":
        pair, sets, targets, variant = found
        return adjustment_task(phrase, world, pair, sets, targets, variant)
    pair, cell = found
    return effect_task(phrase, world, pair, cell)


def placeholders(world: DataWorld, rng: np.random.Generator) -> tuple[str, ...]:
    """Names ``V1``..``Vn`` assigned to the nodes in a random order."""
    order = rng.permutation(world.n)
    return tuple(f"V{int(order[i]) + 1}" for i in range(world.n))


def items(
    network: str, world: DataWorld, family: str, k: int, placeholder_names: tuple[str, ...], used: set[Any]
) -> list[Task]:
    """Item ``k`` of a network and family, a query not in ``used``: the real-names task, then the placeholder one."""
    seed = [SEED, NETWORKS.index(network), list(FINDERS).index(family), k]
    problem = np.random.default_rng(seed)
    target = _target(family, problem)
    for _ in range(MAX_TRIES):
        sampled = with_rows(world, problem)
        result = FINDERS[family](problem, sampled, target)
        if result is not None and result[0] not in used:
            key, found = result
            break
    else:
        raise RuntimeError(f"no new {family} query with target {target} on {network}")
    used.add(key)
    tasks = []
    for naming, names in (("real", sampled.names), ("placeholder", placeholder_names)):
        surface = np.random.default_rng([*seed, 1])
        named = replace(sampled, names=names)
        tasks.append(_render(family, found, surface, FixedNames(surface, NAMES_SENTENCE[naming]), named))
    return tasks
