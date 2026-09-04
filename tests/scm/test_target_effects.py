import itertools

import numpy as np
import pytest

from causalagentbench.scm.gadgets import procedural_scaffold
from causalagentbench.scm.generate import generate
from causalagentbench.scm.target_effects import (
    CONFOUNDING_MARGIN,
    MIN_MEDIATED_SHARE,
    REFERENCE_MIN_EFFECT,
    REFERENCE_MIN_GAP,
    scaled_threshold,
    target_effects,
    with_constructed_confounding,
    with_constructed_effects,
)
from causalagentbench.scoring.baselines import population_coefficient
from causalagentbench.types import Role, Scaffold

DRIVERS = ("A", "B", "C", "D", "E")


def _non_decoy_drivers(scaffold):
    decoys = scaffold.nodes_with_role(Role.DECOY)
    return tuple(d for d in scaffold.drivers if d not in decoys)


def _fully_constructed(seed):
    """Build a scaffold through the whole construction pipeline; None if any step is infeasible."""
    scaffold = procedural_scaffold(seed)
    drivers = _non_decoy_drivers(scaffold)
    reference = generate(seed, scaffold)
    targets = target_effects(np.random.default_rng(seed), drivers)
    scm = with_constructed_effects(reference, targets, np.random.default_rng(seed))
    if scm is None:
        return None, None
    return with_constructed_confounding(scm, np.random.default_rng(seed)), targets


def test_magnitudes_clear_the_scaled_floor_and_gap():
    n = len(DRIVERS)
    floor = scaled_threshold(REFERENCE_MIN_EFFECT, n)
    gap = scaled_threshold(REFERENCE_MIN_GAP, n)
    for seed in range(50):
        targets = target_effects(np.random.default_rng(seed), DRIVERS)
        magnitudes = sorted(abs(v) for v in targets.values())
        assert magnitudes[0] >= floor
        assert all(b - a >= gap for a, b in itertools.pairwise(magnitudes))


def test_driver_assignment_is_not_positionally_predictable():
    """The largest target must not always land on the same driver - the anti-memorization requirement."""
    largest_drivers = set()
    for seed in range(30):
        targets = target_effects(np.random.default_rng(seed), DRIVERS)
        largest_drivers.add(max(targets, key=lambda d: abs(targets[d])))
    assert len(largest_drivers) > 1


def test_with_constructed_effects_hits_targets_exactly():
    for seed in range(20):
        scaffold = procedural_scaffold(seed)
        drivers = _non_decoy_drivers(scaffold)
        reference = generate(seed, scaffold)
        targets = target_effects(np.random.default_rng(seed), drivers)
        scm = with_constructed_effects(reference, targets, np.random.default_rng(seed))
        if scm is None:
            continue
        for driver, target in targets.items():
            assert scm.total_effect_on_outcome(driver) == pytest.approx(target)


def test_with_constructed_effects_guarantees_mediation_realized():
    """The mediated path carries at least MIN_MEDIATED_SHARE of that driver's *own* total effect."""
    n_checked = 0
    for seed in range(30):
        scaffold = procedural_scaffold(seed)
        drivers = _non_decoy_drivers(scaffold)
        reference = generate(seed, scaffold)
        targets = target_effects(np.random.default_rng(seed), drivers)
        scm = with_constructed_effects(reference, targets, np.random.default_rng(seed))
        if scm is None:
            continue
        for driver in drivers:
            mediator = scaffold.mediator_of(driver)
            if mediator is None:
                continue
            n_checked += 1
            outcome_idx = scaffold.index(scaffold.outcome)
            edge = (
                scm.W[scaffold.index(driver), scaffold.index(mediator)] * scm.W[scaffold.index(mediator), outcome_idx]
            )
            assert abs(edge) >= MIN_MEDIATED_SHARE * abs(targets[driver])
    assert n_checked > 0  # sanity: the loop actually exercised at least one mediated driver


def test_mediated_path_sometimes_opposes_its_own_drivers_effect():
    """Both mediation directions must occur - consistent *and* inconsistent (suppression).

    Pinning the mediated path to the target's sign is cheaper on the outcome's variance budget, but it
    deletes suppression (a real structure) and makes `|direct| < |total|` universal - which turns "adding Z
    to the adjustment set shrinks my estimate" into a perfect test for "Z is a mediator", letting an agent
    dodge the over-adjustment trap by numeric probing instead of reasoning about what is downstream of the
    treatment (measured 100% vs 53.7%).
    """
    agrees = opposes = 0
    for seed in range(30):
        scaffold = procedural_scaffold(seed)
        drivers = _non_decoy_drivers(scaffold)
        reference = generate(seed, scaffold)
        targets = target_effects(np.random.default_rng(seed), drivers)
        scm = with_constructed_effects(reference, targets, np.random.default_rng(seed))
        if scm is None:
            continue
        outcome_idx = scaffold.index(scaffold.outcome)
        for driver in drivers:
            mediator = scaffold.mediator_of(driver)
            if mediator is None:
                continue
            mediated = float(
                scm.W[scaffold.index(driver), scaffold.index(mediator)] * scm.W[scaffold.index(mediator), outcome_idx]
            )
            if mediated * targets[driver] > 0:
                agrees += 1
            else:
                opposes += 1
    assert agrees > 0, "no consistent mediation drawn"
    assert opposes > 0, "no inconsistent mediation (suppression) drawn - the over-adjustment trap is probeable"


def test_with_constructed_effects_touches_only_the_outcomes_mediator_collider_and_mbias_driver_rows():
    """Confirms construction touches exactly the rows it claims to and nothing else.

    Every node except the outcome, its drivers' mediators, any collider, and an M-bias driver's own row
    (``A``, via its sole parent ``V`` - the fixup ``_mbias_driver``'s trap strength depends on) keeps its
    reference row untouched.
    """
    scaffold = procedural_scaffold(0)
    outcome_idx = scaffold.index(scaffold.outcome)
    drivers = _non_decoy_drivers(scaffold)
    reference = generate(0, scaffold)
    targets = target_effects(np.random.default_rng(0), drivers)
    scm = with_constructed_effects(reference, targets, np.random.default_rng(0))
    assert scm is not None

    mediator_indices = {scaffold.index(m) for d in drivers if (m := scaffold.mediator_of(d)) is not None}
    collider_indices = {scaffold.index(c) for c in scaffold.nodes_with_role(Role.COLLIDER)}
    mbias_driver_indices = {
        scaffold.index(d)
        for c in scaffold.nodes_with_role(Role.COLLIDER)
        for p in scaffold.parents(c)
        for d in scaffold.drivers
        if scaffold.parents(d) == (p,)
    }
    touched = mediator_indices | collider_indices | mbias_driver_indices | {outcome_idx}
    untouched = [i for i in range(len(scaffold.node_names)) if i not in touched]
    assert np.array_equal(reference.W[np.ix_(untouched, untouched)], scm.W[np.ix_(untouched, untouched)])


def test_with_constructed_effects_returns_none_when_infeasible():
    scaffold = procedural_scaffold(0)
    drivers = _non_decoy_drivers(scaffold)
    reference = generate(0, scaffold)
    impossible = dict.fromkeys(drivers, 100.0)
    assert with_constructed_effects(reference, impossible, np.random.default_rng(0)) is None


def test_with_constructed_confounding_guarantees_the_margin():
    n_checked = 0
    for seed in range(30):
        scm, _ = _fully_constructed(seed)
        if scm is None:
            continue
        scaffold = scm.scaffold
        outcome_idx = scaffold.index(scaffold.outcome)
        decoys = scaffold.nodes_with_role(Role.DECOY)
        if not decoys:
            continue
        drivers = _non_decoy_drivers(scaffold)
        weakest = min(abs(float(scm.sigma[scaffold.index(d), outcome_idx])) for d in drivers)
        margin = scaled_threshold(CONFOUNDING_MARGIN, len(drivers))
        for decoy in decoys:
            n_checked += 1
            decoy_corr = abs(float(scm.sigma[scaffold.index(decoy), outcome_idx]))
            assert decoy_corr > weakest + margin
    assert n_checked > 0


def test_collider_edges_are_constructed_away_from_zero():
    """Confirms the near-zero-weight failure mode measured against raw UUMC is actually fixed.

    A collider's two incoming weights must not land near zero (raw UUMC sometimes did, making the
    adjust-for-it trap toothless: measured bias collapsing under 0.01 whenever one weight was near-zero).
    """
    scaffold = Scaffold(
        node_names=("A", "U", "Co", "Y"),
        roles={"A": Role.DRIVER_CLEAN, "U": Role.CONFOUNDER, "Co": Role.COLLIDER, "Y": Role.OUTCOME},
        edges=(("A", "Y"), ("U", "Y"), ("A", "Co"), ("U", "Co")),
        drivers=("A",),
        outcome="Y",
        correct_adjustment={"A": ()},
    )
    for seed in range(50):
        reference = generate(seed, scaffold)
        targets = target_effects(np.random.default_rng(seed), ("A",))
        scm = with_constructed_effects(reference, targets, np.random.default_rng(seed))
        assert scm is not None
        w_a_co = scm.W[scaffold.index("A"), scaffold.index("Co")]
        w_u_co = scm.W[scaffold.index("U"), scaffold.index("Co")]
        assert abs(w_a_co) >= 0.3
        assert abs(w_u_co) >= 0.3
        assert scm.sigma[scaffold.index("Co"), scaffold.index("Co")] == pytest.approx(1.0)


def test_adjusting_for_a_collider_biases_a_clean_drivers_estimate():
    """The point of the trap: the empty adjustment set is exact, but adjusting for the collider is not."""
    scaffold = Scaffold(
        node_names=("A", "U", "Co", "Y"),
        roles={"A": Role.DRIVER_CLEAN, "U": Role.CONFOUNDER, "Co": Role.COLLIDER, "Y": Role.OUTCOME},
        edges=(("A", "Y"), ("U", "Y"), ("A", "Co"), ("U", "Co")),
        drivers=("A",),
        outcome="Y",
        correct_adjustment={"A": ()},
    )
    n_checked = 0
    for seed in range(50):
        reference = generate(seed, scaffold)
        targets = target_effects(np.random.default_rng(seed), ("A",))
        scm = with_constructed_effects(reference, targets, np.random.default_rng(seed))
        assert scm is not None
        n_checked += 1
        true_effect = scm.total_effect_on_outcome("A")
        assert population_coefficient(scm, "A") == pytest.approx(true_effect)  # empty set: exact
        biased = population_coefficient(scm, "A", ["Co"])
        assert abs(biased - true_effect) > 0.02  # adjusting for the collider: not exact, by a real margin
        # measured floor over 500 seeds: min=0.0348 - 0.02 clears it with margin, not just on average
    assert n_checked > 0


def test_mbias_driver_needs_no_adjustment_and_none_of_its_upstream_causes_reach_the_outcome():
    """Confirms the sharper claim `_mbias_driver` makes.

    A's effect is exact with the empty set, and V - despite causing A - has no path to Y that does not run
    through the blocked collider.
    """
    scaffold = Scaffold(
        node_names=("A", "V", "U", "Mb", "Y"),
        roles={
            "A": Role.DRIVER_CLEAN,
            "V": Role.UPSTREAM_CAUSE,
            "U": Role.CONFOUNDER,
            "Mb": Role.COLLIDER,
            "Y": Role.OUTCOME,
        },
        edges=(("V", "A"), ("A", "Y"), ("V", "Mb"), ("U", "Mb"), ("U", "Y")),
        drivers=("A",),
        outcome="Y",
        correct_adjustment={"A": ()},
    )
    n_checked = 0
    for seed in range(50):
        reference = generate(seed, scaffold)
        targets = target_effects(np.random.default_rng(seed), ("A",))
        scm = with_constructed_effects(reference, targets, np.random.default_rng(seed))
        assert scm is not None
        n_checked += 1
        true_effect = scm.total_effect_on_outcome("A")
        assert population_coefficient(scm, "A") == pytest.approx(true_effect)
        biased = population_coefficient(scm, "A", ["Mb"])
        # a three-hop effect (V->A, V/U->Mb, U->Y), smaller than the two-hop plain-collider case by
        # construction - measured floor over 500 seeds: min=0.0148; 0.01 clears it with margin.
        assert abs(biased - true_effect) > 0.01  # conditioning on Mb manufactures bias from nothing
    assert n_checked > 0


def test_with_constructed_confounding_is_a_noop_without_decoys():
    """Composition always emits at least one decoy now (`DECOY_COUNT_RANGE`), so build the case directly.

    The no-decoy branch is still reachable by any caller assembling a Scaffold itself, and it must stay a
    no-op rather than an error.
    """
    seed = 14
    composed = procedural_scaffold(seed, n_candidates=5)
    decoys = set(composed.nodes_with_role(Role.DECOY))
    scaffold = Scaffold(
        node_names=tuple(n for n in composed.node_names if n not in decoys),
        roles={n: r for n, r in composed.roles.items() if n not in decoys},
        edges=tuple((a, b) for a, b in composed.edges if a not in decoys and b not in decoys),
        drivers=tuple(d for d in composed.drivers if d not in decoys),
        outcome=composed.outcome,
        correct_adjustment={d: a for d, a in composed.correct_adjustment.items() if d not in decoys},
    )
    assert not scaffold.nodes_with_role(Role.DECOY)
    reference = generate(seed, scaffold)
    targets = target_effects(np.random.default_rng(seed), scaffold.drivers)
    scm = with_constructed_effects(reference, targets, np.random.default_rng(seed))
    assert scm is not None
    assert with_constructed_confounding(scm, np.random.default_rng(seed)) is scm
