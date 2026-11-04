"""Graph facts that label the tasks: random DAGs, d-separation, valid adjustment and Markov equivalence classes.

The equivalence class is found from the CPDAG: orient the v-structures (colliders with non-adjacent parents),
then apply Meek's rules 1-3 to exhaustion. Rules 1-3 are complete for deriving a CPDAG purely from a DAG's
v-structures (Meek, 1995; Chickering, JMLR 2002). An edge that stays undirected is unrecoverable by any amount
of observational data, not merely hard to recover from the rows that were sampled.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np

from causalagentbench.scm import Scaffold

MAX_UNDIRECTED_EDGES = 16
"""Brute force enumerates 2^k orientations of the undirected edges."""


def random_dag(rng: np.random.Generator, n: int, edge_prob: float) -> tuple[tuple[int, int], ...]:
    """A connected random DAG on ``n`` nodes: each forward pair ``i < j`` is an edge with ``edge_prob``."""
    while True:
        edges = tuple((i, j) for i, j in itertools.combinations(range(n), 2) if rng.random() < edge_prob)
        graph = nx.Graph(edges)
        graph.add_nodes_from(range(n))
        if nx.is_connected(graph):
            return edges


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


def confounder_rule(graph: nx.DiGraph, x: int, y: int, z: set[int]) -> bool:
    """The textbook shortcut: adjust for every common cause of ``x`` and ``y`` and for no descendant of ``x``.

    It is wrong when a back door is blocked by a non-confounder, under M-bias, and for harmless descendants
    of ``x`` that the complete adjustment criterion allows. Task generation balances labels against it.
    """
    common = nx.ancestors(graph, x) & nx.ancestors(graph, y)
    return common <= z and not z & nx.descendants(graph, x)


def unoriented_edges(scaffold: Scaffold) -> frozenset[tuple[str, str]]:
    """Return the edges of ``scaffold`` that stay undirected in its CPDAG.

    Each returned tuple is the ``(parent, child)`` pair exactly as it appears in ``scaffold.edges``: its
    direction, not its existence, is unrecoverable from data (the skeleton is always recoverable).
    """
    graph = scaffold.digraph()
    skeleton = {frozenset((u, v)) for u, v in graph.edges}
    directed: set[frozenset[str]] = set()

    for node in graph.nodes:
        parents = list(graph.predecessors(node))
        for i, x in enumerate(parents):
            for y in parents[i + 1 :]:
                if frozenset((x, y)) not in skeleton:  # non-adjacent parents -> v-structure at node
                    directed.add(frozenset((x, node)))
                    directed.add(frozenset((y, node)))

    def has_directed_edge(a: str, b: str) -> bool:
        """True if ``{a, b}`` is already oriented, in its true direction ``a -> b``."""
        return frozenset((a, b)) in directed and graph.has_edge(a, b)

    def parent_forces(p: str, c: str) -> bool:
        """Meek rule 1: some ``x -> p`` directed, with ``x`` and ``c`` non-adjacent."""
        return any(has_directed_edge(x, p) and frozenset((x, c)) not in skeleton for x in graph.predecessors(p))

    def path_forces(p: str, c: str) -> bool:
        """Meek rule 2: a directed path ``p -> x -> c`` (avoids a cycle ``p -> x -> c -> p``)."""
        return any(has_directed_edge(p, x) and has_directed_edge(x, c) for x in graph.successors(p))

    def diamond_forces(p: str, c: str) -> bool:
        """Meek rule 3: ``p`` has two non-adjacent undirected neighbors that both point into ``c``."""
        candidates = [x for x in graph.nodes if x not in (p, c) and frozenset((p, x)) in skeleton - directed]
        return any(
            has_directed_edge(x, c) and has_directed_edge(y, c) and frozenset((x, y)) not in skeleton
            for i, x in enumerate(candidates)
            for y in candidates[i + 1 :]
        )

    undirected = skeleton - directed
    changed = True
    while changed:
        changed = False
        for edge in list(undirected):
            a, b = tuple(edge)
            p, c = (a, b) if graph.has_edge(a, b) else (b, a)  # true direction; only it can ever be forced
            if parent_forces(p, c) or path_forces(p, c) or diamond_forces(p, c):
                directed.add(edge)
                undirected.discard(edge)
                changed = True

    return frozenset((p, c) for p, c in scaffold.edges if frozenset((p, c)) in undirected)


def v_structures(graph: nx.DiGraph) -> frozenset[tuple[str, str, str]]:
    """Return the unshielded colliders ``(a, c, b)`` of ``graph``, with ``a < b``."""
    found = set()
    for c in graph:
        for a, b in itertools.combinations(sorted(graph.predecessors(c)), 2):
            if not graph.has_edge(a, b) and not graph.has_edge(b, a):
                found.add((a, c, b))
    return frozenset(found)


def equivalence_class(scaffold: Scaffold) -> list[tuple[tuple[str, str], ...]]:
    """Return every DAG Markov equivalent to ``scaffold``, each as a sorted tuple of edges.

    Markov equivalent DAGs share the skeleton and the unshielded colliders (Verma & Pearl, 1990), so they
    differ only on the CPDAG's undirected edges. Every orientation of those edges is tried and kept if it
    is acyclic and creates no new unshielded collider. The true scaffold is always a member.
    """
    undirected = sorted(unoriented_edges(scaffold))
    if len(undirected) > MAX_UNDIRECTED_EDGES:
        raise ValueError(f"{len(undirected)} undirected edges exceed the brute-force limit")
    fixed = [edge for edge in scaffold.edges if edge not in set(undirected)]
    target = v_structures(scaffold.digraph())
    members = []
    for flips in itertools.product((False, True), repeat=len(undirected)):
        graph = nx.DiGraph(fixed + [(c, p) if flip else (p, c) for (p, c), flip in zip(undirected, flips, strict=True)])
        graph.add_nodes_from(scaffold.node_names)
        if nx.is_directed_acyclic_graph(graph) and v_structures(graph) == target:
            members.append(tuple(sorted(graph.edges)))
    return members
