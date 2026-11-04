import itertools

import networkx as nx
import pytest

from causalagentbench.graphs import (
    MAX_UNDIRECTED_EDGES,
    confounder_rule,
    d_separated,
    equivalence_class,
    unoriented_edges,
    v_structures,
    valid_adjustment,
)
from causalagentbench.scm import Scaffold


def _graph(edges: list[tuple[int, int]], n: int) -> nx.DiGraph:
    """A DAG on nodes 0..n-1."""
    graph = nx.DiGraph(edges)
    graph.add_nodes_from(range(n))
    return graph


def _scaffold(edges, outcome: str = "Y") -> Scaffold:
    """A scaffold over the nodes of ``edges``."""
    nodes = tuple(dict.fromkeys(node for edge in edges for node in edge))
    return Scaffold(node_names=nodes, edges=tuple(edges), outcome=outcome)


def _reference_class(scaffold: Scaffold) -> set[tuple[tuple[str, str], ...]]:
    """Exhaustive reference: every orientation of the whole skeleton with the same v-structures."""
    skeleton = list(scaffold.edges)
    target = v_structures(scaffold.digraph())
    found = set()
    for flips in itertools.product((False, True), repeat=len(skeleton)):
        graph = nx.DiGraph([(c, p) if flip else (p, c) for (p, c), flip in zip(skeleton, flips, strict=True)])
        graph.add_nodes_from(scaffold.node_names)
        if nx.is_directed_acyclic_graph(graph) and v_structures(graph) == target:
            found.add(tuple(sorted(graph.edges)))
    return found


# Confounder C, driver A, mediator M all feed Y; B makes every edge into Y a v-structure.
_TRIANGLE_EDGES = (("C", "A"), ("C", "Y"), ("A", "M"), ("A", "Y"), ("M", "Y"), ("B", "Y"))
_EXAMPLE_EDGES = (*_TRIANGLE_EDGES, ("C", "Dz"), ("Y", "O"))
_MULTI_CONFOUNDER_EDGES = (
    ("C1", "A"), ("C1", "Y"), ("C1", "C2"), ("C2", "A"), ("C2", "Y"),
    ("A", "M"), ("A", "Y"), ("M", "Y"), ("B", "Y"), ("Y", "O"),
)  # fmt: skip


@pytest.mark.parametrize(
    ("edges", "n", "z", "valid"),
    [
        ([(0, 1), (0, 2), (1, 2)], 3, set(), False),  # confounder Z=0 left open
        ([(0, 1), (0, 2), (1, 2)], 3, {0}, True),
        ([(1, 0), (0, 2)], 3, {0}, False),  # X=1 -> M=0 -> Y=2: adjusting for a mediator
        ([(1, 2), (1, 0), (2, 0)], 3, {0}, False),  # collider X -> C <- Y, a descendant of Y
        ([(1, 2), (1, 0)], 3, {0}, True),  # X -> W off the causal path: valid, though not back-door
        ([(3, 1), (3, 0), (4, 0), (4, 2), (1, 2)], 5, {0}, False),  # M-bias: conditioning on the collider
        ([(3, 1), (3, 0), (4, 0), (4, 2), (1, 2)], 5, {0, 3}, True),
    ],
)
def test_adjustment_criterion_known_cases(edges, n, z, valid):
    """Classic cases of the adjustment criterion for the effect of node 1 on node 2."""
    assert valid_adjustment(_graph(edges, n), 1, 2, z) is valid


def test_confounder_rule_agrees_on_a_confounder_and_errs_on_a_harmless_descendant():
    """Z -> X, Z -> Y, X -> Y: both say adjust for Z. X -> W, X -> Y: W is allowed, but the rule forbids it."""
    confounded = _graph([(0, 1), (0, 2), (1, 2)], 3)
    assert confounder_rule(confounded, 1, 2, {0}) and valid_adjustment(confounded, 1, 2, {0})
    assert not confounder_rule(confounded, 1, 2, set()) and not valid_adjustment(confounded, 1, 2, set())
    descendant = _graph([(1, 2), (1, 0)], 3)
    assert not confounder_rule(descendant, 1, 2, {0}) and valid_adjustment(descendant, 1, 2, {0})


def test_conditioning_on_a_collider_connects_its_parents():
    """X -> C <- Y: independent marginally, dependent given C."""
    graph = _graph([(0, 2), (1, 2)], 3)
    assert d_separated(graph, 0, 1, set())
    assert not d_separated(graph, 0, 1, {2})


def test_example_has_exactly_the_known_unoriented_edges():
    """C-A, A-M and C-Dz are unresolvable from data; the v-structures at Y orient Y's edges, then Y -> O."""
    assert unoriented_edges(_scaffold(_EXAMPLE_EDGES)) == frozenset({("C", "A"), ("A", "M"), ("C", "Dz")})


def test_a_collider_is_oriented_and_a_chain_is_not():
    assert unoriented_edges(_scaffold((("X", "Y"), ("Z", "Y")))) == frozenset()
    assert unoriented_edges(_scaffold((("X", "Y"), ("Y", "Z")), outcome="Z")) == frozenset({("X", "Y"), ("Y", "Z")})


def test_v_structures_are_only_unshielded_colliders():
    assert v_structures(nx.DiGraph([("X", "Y"), ("Z", "Y")])) == {("X", "Y", "Z")}
    assert v_structures(nx.DiGraph([("X", "Y"), ("Z", "Y"), ("X", "Z")])) == frozenset()
    assert v_structures(nx.DiGraph([("X", "Y"), ("Y", "Z")])) == frozenset()


@pytest.mark.parametrize(
    ("edges", "size"),
    [
        ((("X", "Y"), ("Y", "Z")), 3),  # chain
        ((("Y", "X"), ("Y", "Z")), 3),  # fork
        ((("X", "Y"), ("Z", "Y")), 1),  # v-structure
        ((("H", "L1"), ("H", "L2"), ("H", "L3")), 4),  # undirected star: the root is H or one leaf
        (_TRIANGLE_EDGES, 3),  # the chain C - A - M
        (_EXAMPLE_EDGES, 4),  # the tree Dz - C - A - M
    ],
)
def test_equivalence_class_sizes_of_known_graphs(edges, size):
    assert len(equivalence_class(_scaffold(edges, outcome=edges[0][1]))) == size


@pytest.mark.parametrize("edges", [_TRIANGLE_EDGES, _EXAMPLE_EDGES, _MULTI_CONFOUNDER_EDGES])
def test_equivalence_class_matches_an_exhaustive_search(edges):
    scaffold = _scaffold(edges)
    members = equivalence_class(scaffold)
    assert len(set(members)) == len(members)
    assert set(members) == _reference_class(scaffold)


def test_equivalence_class_refuses_a_brute_force_beyond_the_limit():
    leaves = [f"L{i}" for i in range(MAX_UNDIRECTED_EDGES + 1)]
    star = _scaffold(tuple(("H", leaf) for leaf in leaves), outcome="L0")
    with pytest.raises(ValueError, match="brute-force limit"):
        equivalence_class(star)
