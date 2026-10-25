import itertools

import networkx as nx
import numpy as np
import pytest

from causalagentbench.scm.effects import marginal_covariance, total_effects
from causalagentbench.scm.gadgets import procedural_scaffold
from causalagentbench.scm.identifiability import (
    MAX_UNDIRECTED_EDGES,
    equivalence_class,
    name_critical_candidates,
    possible_effects,
    unoriented_edges,
    v_structures,
)
from causalagentbench.types import SCM, Role, Scaffold


def _scaffold(edges, outcome: str = "Y", drivers=()) -> Scaffold:
    """A scaffold over the nodes of ``edges``; roles are irrelevant to identifiability."""
    nodes = tuple(dict.fromkeys(node for edge in edges for node in edge))
    roles = {node: Role.OUTCOME if node == outcome else Role.CONFOUNDER for node in nodes}
    return Scaffold(node_names=nodes, roles=roles, edges=tuple(edges), drivers=tuple(drivers), outcome=outcome)


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


def _skeleton(edges) -> set[frozenset[str]]:
    """Undirected version of an edge list."""
    return {frozenset(edge) for edge in edges}


# Confounder C, driver A, mediator M all feed Y; the clean driver B makes every edge into Y a v-structure.
_TRIANGLE_EDGES = (("C", "A"), ("C", "Y"), ("A", "M"), ("A", "Y"), ("M", "Y"), ("B", "Y"))
_TRIANGLE_WEIGHTS = {
    ("C", "A"): 1.0, ("C", "Y"): 1.0, ("A", "M"): 1.0, ("A", "Y"): 0.5, ("M", "Y"): 0.5, ("B", "Y"): 0.7,
}  # fmt: skip
_EXAMPLE_EDGES = (*_TRIANGLE_EDGES, ("C", "Dz"), ("Y", "O"))
_MULTI_CONFOUNDER_EDGES = (
    ("C1", "A"), ("C1", "Y"), ("C1", "C2"), ("C2", "A"), ("C2", "Y"),
    ("A", "M"), ("A", "Y"), ("M", "Y"), ("B", "Y"), ("Y", "O"),
)  # fmt: skip


def _triangle_scm() -> SCM:
    """The triangle with known weights and unit noise (not standardized, so effects are easy by hand)."""
    scaffold = _scaffold(_TRIANGLE_EDGES, drivers=("A", "B"))
    weights = np.zeros((5, 5))
    for (parent, child), w in _TRIANGLE_WEIGHTS.items():
        weights[scaffold.index(parent), scaffold.index(child)] = w
    noise = np.ones(5)
    return SCM(scaffold, weights, noise, marginal_covariance(weights, noise), total_effects(weights))


def test_example_has_exactly_the_known_unoriented_edges():
    """C-A, A-M and C-Dz are unresolvable from data; the v-structures at Y orient Y's edges, then Y -> O."""
    assert unoriented_edges(_scaffold(_EXAMPLE_EDGES)) == frozenset({("C", "A"), ("A", "M"), ("C", "Dz")})


def test_a_collider_with_non_adjacent_parents_is_fully_oriented():
    assert unoriented_edges(_scaffold((("X", "Y"), ("Z", "Y")))) == frozenset()


def test_a_chain_is_fully_unoriented():
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
    scaffold = _scaffold(edges, outcome=edges[0][1])
    assert len(equivalence_class(scaffold)) == size


@pytest.mark.parametrize("edges", [_TRIANGLE_EDGES, _EXAMPLE_EDGES, _MULTI_CONFOUNDER_EDGES])
def test_equivalence_class_matches_an_exhaustive_search(edges):
    scaffold = _scaffold(edges)
    members = equivalence_class(scaffold)
    assert len(set(members)) == len(members)
    assert set(members) == _reference_class(scaffold)


@pytest.mark.parametrize("seed", range(8))
def test_every_member_of_a_motif_class_is_markov_equivalent(seed):
    motif = procedural_scaffold(seed)
    members = equivalence_class(motif)
    target = v_structures(motif.digraph())
    assert tuple(sorted(motif.edges)) in members
    assert len(set(members)) == len(members)
    for edges in members:
        graph = nx.DiGraph(edges)
        graph.add_nodes_from(motif.node_names)
        assert nx.is_directed_acyclic_graph(graph)
        assert _skeleton(edges) == _skeleton(motif.edges)
        assert v_structures(graph) == target


@pytest.mark.parametrize("seed", range(4))
def test_any_member_generates_the_same_class(seed):
    """Equivalence is symmetric: the class computed from any member is the class of the motif."""
    motif = procedural_scaffold(seed)
    members = equivalence_class(motif)
    rng = np.random.default_rng(seed)
    for k in rng.choice(len(members), size=min(3, len(members)), replace=False):
        member = _scaffold(members[int(k)])
        assert set(equivalence_class(member)) == set(members)


def test_equivalence_class_refuses_a_brute_force_beyond_the_limit():
    chain = [(f"N{i}", f"N{i + 1}") for i in range(MAX_UNDIRECTED_EDGES + 1)]
    with pytest.raises(ValueError, match="brute-force"):
        equivalence_class(_scaffold(chain, outcome=chain[-1][1]))


def _effects_by_parents(scm: SCM, x: str) -> dict[tuple[str, ...], float]:
    """Possible effects of ``x`` on the outcome, keyed by ``x``'s parent set in each member."""
    members = equivalence_class(scm.scaffold)
    values = possible_effects(scm.sigma, scm.scaffold.node_names, members, x, scm.scaffold.outcome)
    return {tuple(sorted(p for p, c in edges if c == x)): v for edges, v in zip(members, values, strict=True)}


def test_possible_effects_of_the_triangle_match_hand_computation():
    """Hand-computed effects with C->A = C->Y = A->M = 1, A->Y = M->Y = 0.5 and unit noise.

    parents {C}: back-door blocked, the total effect 0.5 + 1 * 0.5 = 1.0;
    parents {}: cov(A, Y) / var(A) = (1 + 1.0 * 2) / 2 = 1.5;
    parents {M}: E[C | A, M] = A / 2, so Y's coefficient on A is 0.5 + 0.5 = 1.0.
    """
    scm = _triangle_scm()
    assert _effects_by_parents(scm, "A") == pytest.approx({("C",): 1.0, (): 1.5, ("M",): 1.0})
    assert _effects_by_parents(scm, "M") == pytest.approx({("A",): 0.5, (): 3.5 / 3})
    assert _effects_by_parents(scm, "C") == pytest.approx({(): 2.0, ("A",): 1.0})
    assert _effects_by_parents(scm, "B") == pytest.approx({(): 0.7})


def test_the_true_member_gives_the_true_total_effect():
    scm = _triangle_scm()
    members = equivalence_class(scm.scaffold)
    true_member = members.index(tuple(sorted(scm.scaffold.edges)))
    for x in ("A", "B", "C", "M"):
        values = possible_effects(scm.sigma, scm.scaffold.node_names, members, x, "Y")
        assert values[true_member] == pytest.approx(scm.total_effect_on_outcome(x))


def test_name_critical_candidates_are_those_whose_effect_varies_across_the_class():
    assert name_critical_candidates(_triangle_scm()) == ("A",)
