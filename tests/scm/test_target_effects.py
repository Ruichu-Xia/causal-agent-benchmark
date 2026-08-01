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
from causalagentbench.types import Role

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
    treatment (measured 100% vs 53.7%, `variance_budget_plan.md` §10).
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


def test_with_constructed_effects_touches_only_the_outcomes_and_mediator_rows():
    """Every node except the outcome and its drivers' mediators keeps its reference row untouched."""
    scaffold = procedural_scaffold(0)
    outcome_idx = scaffold.index(scaffold.outcome)
    drivers = _non_decoy_drivers(scaffold)
    reference = generate(0, scaffold)
    targets = target_effects(np.random.default_rng(0), drivers)
    scm = with_constructed_effects(reference, targets, np.random.default_rng(0))
    assert scm is not None

    mediator_indices = {scaffold.index(m) for d in drivers if (m := scaffold.mediator_of(d)) is not None}
    untouched = [i for i in range(len(scaffold.node_names)) if i not in mediator_indices and i != outcome_idx]
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


def test_with_constructed_confounding_is_a_noop_without_decoys():
    seed = 14  # confirmed to draw zero decoys at n_drivers=3, and construction succeeds
    scaffold = procedural_scaffold(seed, n_drivers=3)
    assert not scaffold.nodes_with_role(Role.DECOY)
    reference = generate(seed, scaffold)
    targets = target_effects(np.random.default_rng(seed), scaffold.drivers)
    scm = with_constructed_effects(reference, targets, np.random.default_rng(seed))
    assert scm is not None
    assert with_constructed_confounding(scm, np.random.default_rng(seed)) is scm
