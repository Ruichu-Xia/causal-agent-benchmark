"""Exact identifiability discovery: which edges are unorientable from observational data alone.

Builds the CPDAG of a DAG - finds v-structures (colliders with non-adjacent parents), orients them, then
applies Meek's orientation rules 1-3 to exhaustion. Rules 1-3 are complete for deriving a CPDAG purely from
a DAG's v-structures (Meek, 1995; Chickering, "Learning Equivalence Classes of Bayesian-Network
Structures," JMLR 2002); rule 4 is only needed when orienting under extra background knowledge beyond what
v-structures imply, which doesn't apply here - every seed orientation below is a genuine v-structure in the
true graph, nothing external is injected.

An edge surviving this process - staying undirected in the CPDAG - is genuinely unrecoverable by any amount
of observational data, not merely "hard with the data we happened to sample."
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np

from causalagentbench.types import SCM, FloatArray, Scaffold


def unoriented_edges(scaffold: Scaffold) -> frozenset[tuple[str, str]]:
    """Return the edges of ``scaffold`` that stay undirected in its CPDAG.

    Each returned tuple is the ``(parent, child)`` pair exactly as it appears in ``scaffold.edges`` - it
    marks that edge's *direction* as unrecoverable from data, not the edge's existence (the skeleton is
    always fully recoverable).
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


MAX_UNDIRECTED_EDGES = 16
"""Brute force enumerates 2^k orientations; motif scaffolds stay far below this (measured maximum 13)."""


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


def possible_effects(
    sigma: FloatArray, node_names: tuple[str, ...], members: list[tuple[tuple[str, str], ...]], x: str, y: str
) -> list[float]:
    """Total effect of ``x`` on ``y`` implied by ``sigma`` under each member DAG, in ``members`` order.

    Regressing ``y`` on ``x`` and its parents in a DAG gives that DAG's total effect (parents block every
    back-door path). This is global IDA (Maathuis et al., 2009): the multiset of values, with one entry per
    member, is all that data alone can say about the effect. It depends only on ``sigma`` and the CPDAG.
    """
    index = {name: i for i, name in enumerate(node_names)}
    values = []
    for edges in members:
        parents = sorted(p for p, c in edges if c == x)
        rows = [index[x], *(index[p] for p in parents)]
        coefficients = np.linalg.solve(sigma[np.ix_(rows, rows)], sigma[rows, index[y]])
        values.append(float(coefficients[0]))
    return values


def name_critical_candidates(scm: SCM) -> tuple[str, ...]:
    """Candidates whose total effect differs across the equivalence class: only names can settle them."""
    members = equivalence_class(scm.scaffold)
    critical = []
    for candidate in scm.scaffold.drivers:
        values = possible_effects(scm.sigma, scm.scaffold.node_names, members, candidate, scm.scaffold.outcome)
        if max(values) - min(values) > 1e-9:
            critical.append(candidate)
    return tuple(critical)
