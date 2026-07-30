import networkx as nx
import numpy as np
import pytest

from causalagentbench.scm.gadgets import (
    DECOY_COUNT_RANGE,
    DRIVER_COUNT_RANGE,
    RESOLVE_PROBABILITY,
    _add_upstream_cause,
    _clean_driver,
    _confounded_driver,
    _maybe_resolve,
    _resolvable_targets,
    procedural_scaffold,
)
from causalagentbench.scm.identifiability import unidentifiable_drivers, unoriented_edges
from causalagentbench.types import Role, Scaffold


def test_composition_is_deterministic():
    a = procedural_scaffold(42)
    b = procedural_scaffold(42)
    assert a.node_names == b.node_names
    assert a.edges == b.edges
    assert a.correct_adjustment == b.correct_adjustment


def test_different_seeds_give_different_scaffolds():
    assert procedural_scaffold(1).node_names != procedural_scaffold(2).node_names


def test_every_seed_is_a_valid_dag():
    for seed in range(20):
        assert nx.is_directed_acyclic_graph(procedural_scaffold(seed).digraph())


def test_driver_and_decoy_counts_stay_in_range():
    for seed in range(20):
        scaffold = procedural_scaffold(seed)
        n_decoys = len(scaffold.nodes_with_role(Role.DECOY))
        assert DRIVER_COUNT_RANGE[0] <= len(scaffold.drivers) <= DRIVER_COUNT_RANGE[1] + DECOY_COUNT_RANGE[1]
        assert DECOY_COUNT_RANGE[0] <= n_decoys <= DECOY_COUNT_RANGE[1]


def test_n_drivers_can_be_fixed_explicitly():
    for n_drivers in (2, 5, 12):
        for seed in range(5):
            scaffold = procedural_scaffold(seed, n_drivers=n_drivers)
            n_decoys = len(scaffold.nodes_with_role(Role.DECOY))
            assert len(scaffold.drivers) - n_decoys == n_drivers


def test_resolve_probability_zero_leaves_only_the_guaranteed_slot_resolved():
    """resolve_probability=0.0: free slots never resolve.

    Exactly one resolved-confounded driver remains - the one forced slot §11.3 guarantees, not the free
    ones landing there by chance.
    """
    for seed in range(20):
        scaffold = procedural_scaffold(seed, n_drivers=6, resolve_probability=0.0)
        ambiguous = set(unidentifiable_drivers(scaffold))
        non_decoy = scaffold.non_decoy_drivers()
        resolved_confounded = [d for d in non_decoy if d not in ambiguous and scaffold.correct_adjustment.get(d)]
        assert len(resolved_confounded) == 1


def test_resolve_probability_one_leaves_only_the_guaranteed_slot_ambiguous():
    """resolve_probability=1.0: free slots always resolve.

    Exactly one ambiguous driver remains - the one forced slot §11.3 guarantees, not the free ones landing
    there by chance.
    """
    for seed in range(20):
        scaffold = procedural_scaffold(seed, n_drivers=6, resolve_probability=1.0)
        ambiguous = set(unidentifiable_drivers(scaffold)) & set(scaffold.non_decoy_drivers())
        assert len(ambiguous) == 1


def test_resolve_probability_out_of_range_is_rejected():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="resolve_probability"):
            procedural_scaffold(0, resolve_probability=bad)


def test_n_drivers_below_two_is_rejected():
    """§11.3 reserves one slot for the forced-ambiguous driver and one for the forced-resolved driver.

    Below 2, there's nowhere to put both.
    """
    for n_drivers in (0, 1):
        with pytest.raises(ValueError, match="n_drivers"):
            procedural_scaffold(0, n_drivers=n_drivers)


def test_confounded_driver_gadget_has_the_expected_shape():
    """The confounded_driver gadget: one confounder, one confounded driver, one mediator.

    Drawn freely now, no slot forces it anymore (§11) - the already-verified-ambiguous pattern the naming
    experiment leans on whenever it's the one that lands unresolved in a composed instance.
    """
    gadget = _confounded_driver(0)
    assert set(gadget.roles) == {"C0", "A0", "M0"}
    for edge in [("C0", "A0"), ("C0", "Y"), ("A0", "M0"), ("A0", "Y"), ("M0", "Y")]:
        assert edge in gadget.edges
    assert gadget.adjustment == ("C0",)


def _as_scaffold(gadget) -> Scaffold:
    """Wrap a single gadget (plus its implied outcome) into a minimal, valid Scaffold for direct testing."""
    return Scaffold(
        node_names=("Y", *gadget.roles),
        roles={"Y": Role.OUTCOME, **gadget.roles},
        edges=tuple(gadget.edges),
        drivers=(gadget.driver,),
        outcome="Y",
        correct_adjustment={gadget.driver: gadget.adjustment},
    )


def test_add_upstream_cause_resolves_the_confounder_edge():
    """Directly verify the §11.1 mechanism.

    Adding upstream causes to a confounder resolves both that edge and the driver's own edge into it,
    without changing the driver's adjustment set.
    """
    gadget = _confounded_driver(0)
    assert ("C0", "A0") in unoriented_edges(_as_scaffold(gadget))  # unresolved before

    resolved = _add_upstream_cause(gadget, "C0", ("U0_1", "U0_2"))
    assert unoriented_edges(_as_scaffold(resolved)) == frozenset()  # fully resolved after
    assert resolved.adjustment == ("C0",)  # still genuinely confounded, just no longer ambiguous
    assert resolved.driver == gadget.driver


def test_resolvable_targets_matches_each_archetypes_confounder_or_instrument():
    assert _resolvable_targets("clean", 0) == ()
    assert _resolvable_targets("confounded", 0) == ("C0",)
    assert _resolvable_targets("multi_confounder", 0) == ("C0", "C0_2")
    assert _resolvable_targets("instrumented", 0) == ("Z0",)


def test_maybe_resolve_never_touches_a_clean_driver():
    """A clean driver has nothing to resolve.

    ``_maybe_resolve`` must always be a no-op for it, regardless of the coin flip (there's no resolvable
    target to apply it to).
    """
    gadget = _clean_driver(0)
    for trial_seed in range(10):
        rng = np.random.default_rng(trial_seed)
        assert _maybe_resolve(gadget, "clean", 0, rng, RESOLVE_PROBABILITY) == gadget


def test_maybe_resolve_produces_both_outcomes_for_a_confounded_driver():
    outcomes = set()
    for trial_seed in range(30):
        rng = np.random.default_rng(trial_seed)
        gadget = _maybe_resolve(_confounded_driver(0), "confounded", 0, rng, RESOLVE_PROBABILITY)
        outcomes.add(("C0", "A0") in unoriented_edges(_as_scaffold(gadget)))
    assert outcomes == {True, False}


def test_correct_adjustment_shape_per_archetype():
    for seed in range(30):
        scaffold = procedural_scaffold(seed)
        for driver, adjustment in scaffold.correct_adjustment.items():
            role = scaffold.roles[driver]
            if role is Role.DRIVER_CLEAN:
                assert adjustment == ()
            elif role is Role.DECOY:
                assert len(adjustment) == 1
            elif role is Role.DRIVER_CONFOUNDED:
                assert len(adjustment) in (0, 1, 2)  # instrumented (0), single (1), multi-confounder (2)


def test_decoys_attach_to_a_real_existing_confounder():
    for seed in range(20):
        scaffold = procedural_scaffold(seed)
        for decoy in scaffold.nodes_with_role(Role.DECOY):
            (confounder,) = scaffold.correct_adjustment[decoy]
            assert scaffold.roles[confounder] is Role.CONFOUNDER
            assert (confounder, decoy) in scaffold.edges


def test_instruments_have_no_direct_edge_to_the_outcome():
    for seed in range(20):
        scaffold = procedural_scaffold(seed)
        for instrument in scaffold.nodes_with_role(Role.INSTRUMENT):
            assert (instrument, scaffold.outcome) not in scaffold.edges
