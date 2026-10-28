"""Small random causal worlds: a DAG over binary variables, its conditional probability tables, and exact queries.

Node ``i`` has index ``i`` and every edge points from a lower to a higher index, so index order is a
topological order; names, not indices, are shown to the model. A world has at most a handful of nodes, so
every probability is computed exactly by enumerating all 2^n joint states.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class World:
    """A causal DAG over binary variables with its mechanism.

    Attributes:
        names: Display name of each node.
        edges: Directed edges ``(parent, child)`` with ``parent < child``.
        cpts: ``cpts[i][k]`` is P(node i = 1) when its parents (ascending) take the bits of ``k``: bit ``j``
            of ``k`` is the value of the ``j``-th parent.
    """

    names: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    cpts: tuple[tuple[float, ...], ...]

    def parents(self, node: int) -> tuple[int, ...]:
        """Parents of ``node`` in ascending order."""
        return tuple(sorted(p for p, c in self.edges if c == node))

    def graph(self) -> nx.DiGraph:
        """The DAG as a networkx graph on node indices."""
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(self.names)))
        graph.add_edges_from(self.edges)
        return graph


def random_dag(rng: np.random.Generator, n: int, edge_prob: float) -> tuple[tuple[int, int], ...]:
    """A connected random DAG on ``n`` nodes: each forward pair ``i < j`` is an edge with ``edge_prob``."""
    while True:
        edges = tuple((i, j) for i, j in itertools.combinations(range(n), 2) if rng.random() < edge_prob)
        graph = nx.Graph(edges)
        graph.add_nodes_from(range(n))
        if nx.is_connected(graph):
            return edges


def random_world(rng: np.random.Generator, names: tuple[str, ...], edge_prob: float) -> World:
    """A random DAG over ``names`` with probabilities drawn from [0.05, 0.95] and rounded to two decimals."""
    edges = random_dag(rng, len(names), edge_prob)
    cpts = []
    for node in range(len(names)):
        k = sum(1 for _, c in edges if c == node)
        cpts.append(tuple(float(round(p, 2)) for p in rng.uniform(0.05, 0.95, size=2**k)))
    return World(names, edges, tuple(cpts))


def joint(world: World, do: dict[int, int] | None = None) -> np.ndarray:
    """Probability of every joint state (bit ``i`` of the state index is node ``i``), optionally under do()."""
    do = do or {}
    n = len(world.names)
    probs = np.ones(2**n)
    for state in range(2**n):
        for node in range(n):
            value = (state >> node) & 1
            if node in do:
                probs[state] *= float(value == do[node])
                continue
            pattern = sum(((state >> parent) & 1) << j for j, parent in enumerate(world.parents(node)))
            p_one = world.cpts[node][pattern]
            probs[state] *= p_one if value else 1 - p_one
    return probs


def probability(
    world: World, event: dict[int, int], given: dict[int, int] | None = None, do: dict[int, int] | None = None
) -> float:
    """P(event | given) in the world, or in the world after the intervention ``do``."""
    given = given or {}
    probs = joint(world, do)
    states = np.arange(len(probs))

    def holds(assignment: dict[int, int]) -> np.ndarray:
        """Mask of the states that agree with ``assignment``."""
        mask = np.ones(len(probs), dtype=bool)
        for node, value in assignment.items():
            mask &= ((states >> node) & 1) == value
        return mask

    condition = holds(given)
    return float(probs[condition & holds(event)].sum() / probs[condition].sum())


def causal_effect(world: World, x: int, y: int) -> float:
    """P(y = 1 | do(x = 1)) - P(y = 1 | do(x = 0))."""
    return probability(world, {y: 1}, do={x: 1}) - probability(world, {y: 1}, do={x: 0})


def observed_difference(world: World, x: int, y: int) -> float:
    """P(y = 1 | x = 1) - P(y = 1 | x = 0)."""
    return probability(world, {y: 1}, {x: 1}) - probability(world, {y: 1}, {x: 0})


def d_separated(graph: nx.DiGraph, x: int, y: int, z: set[int]) -> bool:
    """Whether ``z`` d-separates ``x`` and ``y``."""
    return bool(nx.is_d_separator(graph, {x}, {y}, z))


def valid_adjustment(graph: nx.DiGraph, x: int, y: int, z: set[int]) -> bool:
    """The adjustment criterion (Shpitser et al., 2010; Perkovic et al., 2018) for the effect of ``x`` on ``y``.

    ``z`` is valid if and only if it contains no descendant of a node on a causal path from ``x`` to ``y``
    (nor ``x`` itself) and it d-separates ``x`` and ``y`` once the first edge of every causal path is removed.
    Unlike the back-door criterion, this is complete: every valid set passes.
    """
    if x in z or y in z:
        raise ValueError("the adjustment set must not contain x or y")
    on_causal_path = {w for w in nx.descendants(graph, x) if w == y or y in nx.descendants(graph, w)}
    forbidden = {x} | on_causal_path
    for w in on_causal_path:
        forbidden |= nx.descendants(graph, w)
    if forbidden & z:
        return False
    proper = graph.copy()
    proper.remove_edges_from([(x, w) for w in graph.successors(x) if w in on_causal_path])
    return d_separated(proper, x, y, z)
