"""Named linear-Gaussian causal worlds with sampled data, the answer key for every data-analysis task.

A world is drawn in four steps (``docs/environment_v3.md``):

1. a random connected DAG ``G0`` and a unit-variance correlation matrix Sigma on it;
2. the true DAG ``D``, drawn uniformly from ``G0``'s Markov equivalence class, with weights set by regression
   on Sigma, so the data carry no orientation information beyond the CPDAG;
3. 2,000 rows from ``D`` (``simulate``), kept only if PC on them recovers ``D``'s CPDAG exactly;
4. CauseNet concept names for ``D``'s nodes (``CauseNetLabeler``): every edge of ``D`` is a CauseNet relation
   in its direction, and no relation between the chosen names points against ``D``'s causal order.

Nodes are indices ``0..n-1``. ``name_edges`` is ``Gn``, every CauseNet relation among the chosen names, which
may include links the data refute.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from causalagentbench.generation.causenet import CauseNetLabeler, GroundingError
from causalagentbench.reasoning.worlds import random_dag
from causalagentbench.scm.generate import draw_correlation, realize
from causalagentbench.scm.identifiability import MAX_UNDIRECTED_EDGES, equivalence_class, unoriented_edges
from causalagentbench.types import SCM, FloatArray, Role, Scaffold

ROWS = 2000
"""Rows of data per task."""
PC_ALPHA = 0.01
"""Significance level of the PC run that checks a world's data recover its CPDAG."""
EDGE_PROB = (0.3, 0.6)
"""Range of the forward-pair edge probability of ``G0``."""
MOMENT_RTOL, MOMENT_ATOL = 1e-12, 1e-9
"""Tolerance of the regeneration check on the data's moments."""

Edges = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DataWorld:
    """A named causal world and the seed of its data.

    Attributes:
        names: Column name of each node.
        edges: The true DAG ``D``.
        members: Every DAG Markov equivalent to ``D`` (including ``D``).
        name_edges: ``Gn``: every CauseNet relation among the chosen names, as node pairs.
        weights: ``weights[i, j]`` is the coefficient of ``i`` in ``j``'s equation.
        noise_var: Noise variance of each node.
        sigma: The population correlation matrix (unit variances).
        data_seed: Seed of the 2,000 sampled rows.
        data_moments: Each column's sum, then each column's sum of squares, to check a regeneration.
    """

    names: tuple[str, ...]
    edges: Edges
    members: tuple[Edges, ...]
    name_edges: Edges
    weights: FloatArray
    noise_var: FloatArray
    sigma: FloatArray
    data_seed: int
    data_moments: tuple[float, ...]

    @property
    def n(self) -> int:
        """Number of variables."""
        return len(self.names)

    def graph(self, edges: Edges | None = None) -> nx.DiGraph:
        """``D`` (or the given edges) as a networkx graph on all nodes."""
        graph = nx.DiGraph(self.edges if edges is None else edges)
        graph.add_nodes_from(range(self.n))
        return graph

    def member_graphs(self) -> list[nx.DiGraph]:
        """Every equivalence-class member as a graph."""
        return [self.graph(member) for member in self.members]

    def name_graph(self) -> nx.DiGraph:
        """``Gn`` as a graph (it may contain cycles)."""
        return self.graph(self.name_edges)

    def total_effect(self, x: int, y: int) -> float:
        """The total causal effect of ``x`` on ``y`` in standard-deviation units."""
        tau = np.linalg.inv(np.eye(self.n) - self.weights) - np.eye(self.n)
        return float(tau[x, y])

    def data(self) -> FloatArray:
        """The world's rows, regenerated from its seed and checked against the stored moments.

        The check allows rounding-level differences: numpy's normal sampler calls the platform's ``log`` in a
        rare branch, which can differ in the last bit between machines. A wrong seed or simulator changes the
        moments at order one.
        """
        rows = simulate(self.edges, self.weights, self.noise_var, ROWS, self.data_seed)
        if not np.allclose(moments(rows), self.data_moments, rtol=MOMENT_RTOL, atol=MOMENT_ATOL):
            raise RuntimeError("regenerated data differ from the data the world was built and checked with")
        return rows


def _scaffold(edges: Edges, n: int) -> Scaffold:
    """A ``Scaffold`` on nodes ``"0".."n-1"``; roles are unused here, and the outcome is a sink."""
    nodes = tuple(str(i) for i in range(n))
    graph = nx.DiGraph(edges)
    graph.add_nodes_from(range(n))
    sink = str(next(v for v in reversed(list(nx.topological_sort(graph))) if graph.out_degree(v) == 0))
    return Scaffold(nodes, dict.fromkeys(nodes, Role.CONFOUNDER), tuple((str(p), str(c)) for p, c in edges), (), sink)


def _scm(edges: Edges, sigma: FloatArray) -> SCM:
    """The SCM on ``edges`` that reproduces ``sigma`` exactly."""
    return realize(_scaffold(edges, len(sigma)), sigma)


def simulate(edges: Edges, weights: FloatArray, noise_var: FloatArray, n_rows: int, seed: int) -> FloatArray:
    """``n_rows`` draws from the linear-Gaussian model, bit-identical on every platform.

    Each column is its noise plus its parents' weighted columns, added in a fixed order with elementwise
    operations only: a linear solve would go through LAPACK, whose last bits differ across machines, and the
    data must be exactly the rows the build checked.
    """
    n = len(noise_var)
    noise = np.random.default_rng(seed).standard_normal((n_rows, n)) * np.sqrt(noise_var)
    graph = nx.DiGraph(edges)
    graph.add_nodes_from(range(n))
    rows = np.zeros((n_rows, n))
    for j in nx.lexicographical_topological_sort(graph):
        column = noise[:, j].copy()
        for parent in sorted(graph.predecessors(j)):
            column = column + weights[parent, j] * rows[:, parent]
        rows[:, j] = column
    return rows


def moments(rows: FloatArray) -> tuple[float, ...]:
    """Each column's sum, then each column's sum of squares."""
    return tuple(float(v) for v in np.concatenate([rows.sum(axis=0), (rows**2).sum(axis=0)]))


def cpdag(edges: Edges, n: int) -> frozenset[tuple[int, int] | frozenset[int]]:
    """The CPDAG of a DAG: directed edges as pairs, undirected ones as frozensets."""
    undirected = {(int(p), int(c)) for p, c in unoriented_edges(_scaffold(edges, n))}
    return frozenset(frozenset(edge) if edge in undirected else edge for edge in edges)


def pc_cpdag(rows: FloatArray, alpha: float = PC_ALPHA) -> frozenset[tuple[int, int] | frozenset[int]] | None:
    """The CPDAG PC (stable) estimates from ``rows``, or None if PC returns a bidirected or conflicting edge."""
    from causallearn.search.ConstraintBased.PC import pc

    graph = pc(rows, alpha=alpha, stable=True, show_progress=False).G.graph
    found: set[tuple[int, int] | frozenset[int]] = set()
    n = rows.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = int(graph[i, j]), int(graph[j, i])
            if (a, b) == (0, 0):
                continue
            if (a, b) == (-1, 1):
                found.add((i, j))
            elif (a, b) == (1, -1):
                found.add((j, i))
            elif (a, b) == (-1, -1):
                found.add(frozenset((i, j)))
            else:
                return None
    return frozenset(found)


def draw_world(rng: np.random.Generator, size: int, labeler: CauseNetLabeler) -> DataWorld | None:
    """A named world on ``size`` nodes whose data recover its CPDAG, or None if this draw is rejected."""
    g0 = random_dag(rng, size, float(rng.uniform(*EDGE_PROB)))
    sigma = draw_correlation(_scaffold(g0, size), rng)
    if len(unoriented_edges(_scaffold(g0, size))) > MAX_UNDIRECTED_EDGES:
        return None
    members = tuple(
        tuple(sorted((int(p), int(c)) for p, c in member)) for member in equivalence_class(_scaffold(g0, size))
    )
    edges = members[int(rng.integers(len(members)))]
    scm = _scm(edges, sigma)
    weights, noise_var = np.array(scm.W), np.array(scm.noise_var)
    data_seed = int(rng.integers(2**63))
    rows = simulate(edges, weights, noise_var, ROWS, data_seed)
    if pc_cpdag(rows) != cpdag(edges, size):
        return None
    try:
        mapping = labeler(scm, rng).node_to_label
    except GroundingError:
        return None
    names = tuple(mapping[str(i)] for i in range(size))
    name_edges = tuple(
        (a, b) for a in range(size) for b in range(size) if a != b and labeler.graph.has_edge(names[a], names[b])
    )
    return DataWorld(
        names=names,
        edges=edges,
        members=members,
        name_edges=name_edges,
        weights=weights,
        noise_var=noise_var,
        sigma=np.array(sigma),
        data_seed=data_seed,
        data_moments=moments(rows),
    )
