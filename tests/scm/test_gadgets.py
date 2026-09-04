import networkx as nx
import numpy as np
import pytest

from causalagentbench.scm.gadgets import (
    CANDIDATE_COUNT_RANGE,
    DECOY_COUNT_RANGE,
    RESOLVE_PROBABILITY,
    _add_upstream_cause,
    _clean_driver,
    _collider_driver,
    _confounded_driver,
    _maybe_resolve,
    _mbias_driver,
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


def test_candidate_and_decoy_counts_stay_in_range():
    for seed in range(20):
        scaffold = procedural_scaffold(seed)
        n_decoys = len(scaffold.nodes_with_role(Role.DECOY))
        assert CANDIDATE_COUNT_RANGE[0] <= len(scaffold.drivers) <= CANDIDATE_COUNT_RANGE[1]
        assert DECOY_COUNT_RANGE[0] <= n_decoys <= DECOY_COUNT_RANGE[1]


def test_every_instance_contains_at_least_one_decoy():
    """The correlation trap has to be present in every task, not most of them.

    Under the old ``(0, 2)`` range, 39-42% of accepted instances contained no decoy at all.
    """
    for seed in range(30):
        assert scaffold_decoys(procedural_scaffold(seed)) >= 1


def scaffold_decoys(scaffold) -> int:
    return len(scaffold.nodes_with_role(Role.DECOY))


def test_fixing_n_candidates_fixes_task_size_while_the_split_still_varies():
    """The total is what must stay constant; how it splits into drivers vs decoys must not be predictable."""
    for n_candidates in (4, 7, 12):
        splits = set()
        for seed in range(12):
            scaffold = procedural_scaffold(seed, n_candidates=n_candidates)
            assert len(scaffold.drivers) == n_candidates  # task size exactly as requested, decoys included
            splits.add(scaffold_decoys(scaffold))
        assert len(splits) > 1, f"decoy count never varied at n_candidates={n_candidates}"


def test_resolve_probability_zero_leaves_only_the_guaranteed_slot_resolved():
    """resolve_probability=0.0: free slots never resolve.

    Exactly one resolved-confounded driver remains - the one forced slot §11.3 guarantees, not the free
    ones landing there by chance.
    """
    for seed in range(20):
        scaffold = procedural_scaffold(seed, n_candidates=6, resolve_probability=0.0)
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
        scaffold = procedural_scaffold(seed, n_candidates=6, resolve_probability=1.0)
        ambiguous = set(unidentifiable_drivers(scaffold)) & set(scaffold.non_decoy_drivers())
        assert len(ambiguous) == 1


def test_resolve_probability_out_of_range_is_rejected():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="resolve_probability"):
            procedural_scaffold(0, resolve_probability=bad)


def test_too_few_candidates_to_hold_the_reserved_slots_is_rejected():
    """§11.3 reserves one slot for the forced-ambiguous driver and one for the forced-resolved driver.

    Decoys are carved out of the same budget, so the floor is the largest decoy draw plus those two.
    """
    for n_candidates in (0, 1, DECOY_COUNT_RANGE[1] + 1):
        with pytest.raises(ValueError, match="n_candidates"):
            procedural_scaffold(0, n_candidates=n_candidates)


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


def test_collider_driver_gadget_has_the_expected_shape():
    """collider_driver: a clean A, a confounder-shaped U, and a collider Co they both feed."""
    gadget = _collider_driver(0)
    assert set(gadget.roles) == {"A0", "U0", "Co0"}
    assert gadget.roles["A0"] is Role.DRIVER_CLEAN
    assert gadget.roles["U0"] is Role.CONFOUNDER
    assert gadget.roles["Co0"] is Role.COLLIDER
    for edge in [("A0", "Y"), ("U0", "Y"), ("A0", "Co0"), ("U0", "Co0")]:
        assert edge in gadget.edges
    assert gadget.adjustment == ()


def test_collider_driver_is_a_genuine_unshielded_v_structure():
    """A0 and U0 are non-adjacent, so Co0 is a real collider - both incoming edges must fully orient."""
    scaffold = _as_scaffold(_collider_driver(0))
    assert unoriented_edges(scaffold) == frozenset()


def test_mbias_driver_gadget_has_the_expected_shape():
    """mbias_driver: V feeds A and the collider but never Y; U feeds the collider and Y but never A."""
    gadget = _mbias_driver(0)
    assert set(gadget.roles) == {"A0", "V0", "U0", "Mb0"}
    assert gadget.roles["A0"] is Role.DRIVER_CLEAN
    assert gadget.roles["V0"] is Role.UPSTREAM_CAUSE
    assert gadget.roles["U0"] is Role.CONFOUNDER
    assert gadget.roles["Mb0"] is Role.COLLIDER
    for edge in [("V0", "A0"), ("A0", "Y"), ("V0", "Mb0"), ("U0", "Mb0"), ("U0", "Y")]:
        assert edge in gadget.edges
    assert ("V0", "Y") not in gadget.edges  # V must never reach Y except through the blocked collider
    assert ("U0", "A0") not in gadget.edges  # U must never touch A directly
    assert gadget.adjustment == ()


def test_mbias_driver_is_a_genuine_unshielded_v_structure():
    """V0 and U0 are non-adjacent, so Mb0 is a real collider - both its incoming edges must orient.

    Unlike ``_collider_driver``, this checks only Mb0's own two edges, not every edge in the gadget: V0->A0
    is a plain, single edge with no v-structure of its own (the same benign phenomenon
    ``test_identifiability.py::test_a_chain_is_fully_unoriented`` documents for a chain link) - and it is
    benign here too, since V0 is ``Role.UPSTREAM_CAUSE``, not a confounder, so its orientation never affects
    A's identifiability.
    """
    scaffold = _as_scaffold(_mbias_driver(0))
    assert {("V0", "Mb0"), ("U0", "Mb0")} & unoriented_edges(scaffold) == set()


def test_resolvable_targets_is_empty_for_collider_and_mbias():
    """Neither archetype resolves an edge into its own driver.

    The collider is the trap, not an ambiguous edge on the driver itself.
    """
    assert _resolvable_targets("collider", 0) == ()
    assert _resolvable_targets("mbias", 0) == ()


def test_collider_and_mbias_drivers_are_never_marked_unidentifiable():
    """Confirms neither collider archetype's driver is ever flagged ambiguous.

    Neither driver's own parent (if any) carries `Role.CONFOUNDER` - `A` in `collider_driver` has none at
    all, and `mbias_driver`'s `V` is `Role.UPSTREAM_CAUSE`, not a confounder - so `unidentifiable_drivers`
    must never flag them, across every seed that draws either archetype.
    """
    n_checked = 0
    for seed in range(200):
        scaffold = procedural_scaffold(seed, n_candidates=7)
        ambiguous = set(unidentifiable_drivers(scaffold))
        for driver in scaffold.drivers:
            children = [c for p, c in scaffold.edges if p == driver]
            if any(scaffold.roles[c] is Role.COLLIDER for c in children):
                n_checked += 1
                assert driver not in ambiguous
    assert n_checked > 0  # sanity: the loop actually exercised at least one collider/mbias driver


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
