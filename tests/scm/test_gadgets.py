import networkx as nx
import numpy as np
import pytest

from causalagentbench.scm.gadgets import DECOY_COUNT_RANGE, OUTCOME, OUTCOME_CHILD, procedural_scaffold
from causalagentbench.scm.generate import draw_correlation
from causalagentbench.scm.identifiability import equivalence_class, unoriented_edges
from causalagentbench.scoring.gates import evaluate_gates
from causalagentbench.types import Role

SEEDS = range(30)


def _with_role(scaffold, role: Role) -> list[str]:
    """Nodes composed in ``role``."""
    return [node for node, r in scaffold.roles.items() if r is role]


def test_composition_is_deterministic():
    a, b = procedural_scaffold(42), procedural_scaffold(42)
    assert (a.node_names, dict(a.roles), a.edges, a.drivers) == (b.node_names, dict(b.roles), b.edges, b.drivers)


def test_different_seeds_give_different_scaffolds():
    assert len({procedural_scaffold(seed).edges for seed in range(10)}) > 1


@pytest.mark.parametrize("seed", SEEDS)
def test_every_scaffold_has_the_outcome_child_and_no_instrument(seed):
    scaffold = procedural_scaffold(seed)
    graph = scaffold.digraph()
    assert scaffold.outcome == OUTCOME
    assert scaffold.roles[OUTCOME_CHILD] is Role.OUTCOME_CHILD
    assert list(graph.predecessors(OUTCOME_CHILD)) == [OUTCOME] and graph.out_degree(OUTCOME_CHILD) == 0
    assert graph.out_degree(OUTCOME) == 1
    assert OUTCOME_CHILD not in scaffold.drivers
    assert all(isinstance(role, Role) for role in scaffold.roles.values())


@pytest.mark.parametrize("n_candidates", [5, 7])
def test_candidate_count_is_fixed_and_decoys_are_carved_out_of_it(n_candidates):
    decoy_counts = set()
    for seed in SEEDS:
        scaffold = procedural_scaffold(seed, n_candidates=n_candidates)
        assert len(scaffold.drivers) == len(set(scaffold.drivers)) == n_candidates
        decoys = _with_role(scaffold, Role.DECOY)
        assert set(decoys) <= set(scaffold.drivers)
        decoy_counts.add(len(decoys))
        for driver in scaffold.drivers:
            assert scaffold.roles[driver] in (Role.DRIVER_CLEAN, Role.DRIVER_CONFOUNDED, Role.DECOY)
    assert decoy_counts == set(range(DECOY_COUNT_RANGE[0], DECOY_COUNT_RANGE[1] + 1))


def test_default_task_has_five_candidates():
    assert len(procedural_scaffold(0).drivers) == 5


def test_too_few_candidates_to_hold_the_reserved_slots_is_rejected():
    with pytest.raises(ValueError, match="n_candidates"):
        procedural_scaffold(0, n_candidates=DECOY_COUNT_RANGE[1] + 1)


@pytest.mark.parametrize("seed", SEEDS)
def test_decoys_hang_off_a_confounder_with_no_path_to_the_outcome(seed):
    scaffold = procedural_scaffold(seed)
    graph = scaffold.digraph()
    for decoy in _with_role(scaffold, Role.DECOY):
        (parent,) = graph.predecessors(decoy)
        assert scaffold.roles[parent] is Role.CONFOUNDER
        assert graph.out_degree(decoy) == 0
        assert not nx.has_path(graph, decoy, OUTCOME)


@pytest.mark.parametrize("seed", SEEDS)
def test_colliders_are_always_oriented_by_the_data(seed):
    scaffold = procedural_scaffold(seed)
    unoriented = unoriented_edges(scaffold)
    for collider in _with_role(scaffold, Role.COLLIDER):
        assert all((parent, collider) not in unoriented for parent in scaffold.parents(collider))


@pytest.mark.parametrize("seed", SEEDS)
def test_reserved_slots_leave_one_confounded_driver_ambiguous_and_resolve_another(seed):
    """One confounded driver keeps an undirected confounder edge; another is oriented by upstream causes.

    The second driver's confounder has two exogenous parents, whose v-structure orients its edges.
    """
    scaffold = procedural_scaffold(seed)
    unoriented = unoriented_edges(scaffold)
    confounded = [d for d in scaffold.drivers if scaffold.roles[d] is Role.DRIVER_CONFOUNDED]
    ambiguous = [d for d in confounded if any((p, d) in unoriented for p in scaffold.parents(d))]
    resolved = [d for d in confounded if all((p, d) not in unoriented for p in scaffold.parents(d))]
    assert ambiguous and resolved
    for driver in resolved:
        for confounder in scaffold.parents(driver):
            upstream = scaffold.parents(confounder)
            assert [scaffold.roles[u] for u in upstream].count(Role.UPSTREAM_CAUSE) >= 2


@pytest.mark.parametrize("seed", range(12))
def test_reserved_slots_satisfy_the_structure_gate_for_any_correlation(seed):
    """At least one name-critical candidate and one identified candidate that needs adjustment."""
    motif = procedural_scaffold(seed)
    sigma = draw_correlation(motif, np.random.default_rng(seed))
    report = evaluate_gates(motif, sigma, equivalence_class(motif))
    assert report.structure
    assert report.name_critical
