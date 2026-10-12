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

from causalagentbench.types import Role, Scaffold


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


def unidentifiable_drivers(scaffold: Scaffold) -> tuple[str, ...]:
    """Return ambiguous candidates in the generated gadget family, in canonical order.

    This role-based shortcut is specific to these gadgets, not a general effect-identification algorithm.

    Only counts ambiguity from an unoriented **confounder** edge, not an unoriented **instrument** edge -
    edge-orientation ambiguity and effect/adjustment identifiability are not the same thing, and for an
    instrumented driver (``scm/gadgets.py``) they come apart. A confounder always has its own direct edge
    into the outcome, so whether it's oriented confounder->driver or driver->confounder changes whether
    it's a legitimate common cause that must be adjusted for - the driver's correct adjustment set (and
    hence its effect estimate) genuinely depends on that orientation. An instrument has no path to the
    outcome except through its own driver, so reversing instrument<->driver can never introduce or remove
    confounding: the driver's total effect and its (always-empty) adjustment set are identical either way,
    regardless of which way that edge points - the driver is genuinely identified, just not fully oriented.

    Returns a tuple, not a set: this feeds a naming-priority order (``generation/labeler.py``), which must
    stay deterministic - set/frozenset iteration order is not guaranteed stable.
    """
    ambiguous_children = {
        child for parent, child in unoriented_edges(scaffold) if scaffold.roles[parent] is Role.CONFOUNDER
    }
    return tuple(driver for driver in scaffold.drivers if driver in ambiguous_children)
