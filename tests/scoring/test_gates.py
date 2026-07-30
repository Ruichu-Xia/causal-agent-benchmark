import numpy as np
import pytest

from causalagentbench import sample
from causalagentbench.scm.identifiability import unidentifiable_drivers, unoriented_edges
from causalagentbench.scoring.baselines import (
    correlation_effects,
    full_adjustment_effects,
    oracle_effects,
    ranking,
)
from causalagentbench.scoring.gates import (
    ORACLE_TAU_THRESHOLD,
    _has_balanced_identifiability,
    generate_gated_procedural,
)
from causalagentbench.scoring.metrics import kendall_tau
from causalagentbench.types import Role, Scaffold

SEEDS = list(range(20))
#: Curated, not just range(5): each of these is verified to have at least one decoy, which the "naive
#: correlation gets fooled" assertion below needs - procedural composition can legitimately draw zero
#: decoys (`DECOY_COUNT_RANGE`), and plain range(5) doesn't guarantee that under the current RNG sequence.
PROCEDURAL_EMPIRICAL_SEEDS = [1, 4, 6, 10, 14]
LARGE_N = 20_000


@pytest.mark.parametrize("seed", SEEDS)
def test_gated_procedural_instance_passes_every_gate(seed):
    _, report = generate_gated_procedural(seed)
    assert report.passed
    assert report.structure_valid
    assert report.confounding_realized
    assert report.mediation_realized
    assert report.oracle_recovers


@pytest.mark.parametrize("seed", PROCEDURAL_EMPIRICAL_SEEDS)
def test_procedural_oracle_recovers_truth_but_naive_shortcuts_fail(seed):
    scm, _ = generate_gated_procedural(seed)
    data = sample(scm, LARGE_N, seed=seed)
    truth = scm.true_ranking()

    # The oracle (correct adjustment sets) recovers the true ranking - matching the gate's own bar
    # (tau >= ORACLE_TAU_THRESHOLD), not exact equality: the gate only guarantees a *minimum* effect gap,
    # not a wide one, so two adjacent drivers can legitimately sit close enough that finite-sample noise
    # occasionally swaps their estimated order even at a large sample size, without tau dropping below it.
    oracle = oracle_effects(data, scm)
    drivers = scm.scaffold.drivers
    true_effects = [abs(scm.total_effect_on_outcome(d)) for d in drivers]
    oracle_estimates = [abs(oracle[d]) for d in drivers]
    assert kendall_tau(true_effects, oracle_estimates) >= ORACLE_TAU_THRESHOLD
    # ...while the "no adjustment" shortcut (raw correlation) mis-ranks (the decoy fools it).
    assert ranking(correlation_effects(data, scm)) != truth
    # ...and the "adjust for everything" shortcut biases a mediated driver (mediator trap). No slot is
    # special-cased anymore (§11), but the gate's guaranteed resolved-but-confounded driver is always
    # `confounded`/`multi_confounder` archetype (never `instrumented`, which has no adjustment set to be
    # non-empty) - both of those archetypes have a mediator, so at least one always exists.
    mediated_drivers = [d for d in scm.scaffold.drivers if scm.scaffold.mediator_of(d) is not None]
    assert mediated_drivers
    mediated_driver = mediated_drivers[0]
    full_estimate = full_adjustment_effects(data, scm)[mediated_driver]
    assert abs(full_estimate - scm.total_effect_on_outcome(mediated_driver)) > 0.02


def test_gated_procedural_generation_is_deterministic():
    a, _ = generate_gated_procedural(7)
    b, _ = generate_gated_procedural(7)
    assert np.array_equal(a.W, b.W)


def test_gated_procedural_respects_n_drivers():
    for n_drivers in (3, 5, 8):
        scm, report = generate_gated_procedural(0, n_drivers=n_drivers)
        decoys = scm.scaffold.nodes_with_role(Role.DECOY)
        assert len([d for d in scm.scaffold.drivers if d not in decoys]) == n_drivers
        assert report.passed


def test_gated_procedural_respects_resolve_probability():
    """resolve_probability survives the full gate/retry loop, not just raw composition."""
    for seed in range(5):
        scm, report = generate_gated_procedural(seed, n_drivers=6, resolve_probability=1.0)
        ambiguous = set(unidentifiable_drivers(scm.scaffold)) & set(scm.scaffold.non_decoy_drivers())
        assert len(ambiguous) == 1
        assert report.passed


def test_has_balanced_identifiability_rejects_a_lopsided_composition():
    """A single confounded_driver, with nothing else - ambiguous but nothing resolved.

    The exact shape §11.3's guarantee rules out of `procedural_scaffold`'s real output - tested directly
    here since composition itself can no longer produce it (procedural_scaffold_plan.md §11.0/§11.3).
    """
    scaffold = Scaffold(
        node_names=("Y", "C0", "A0", "M0"),
        roles={"Y": Role.OUTCOME, "C0": Role.CONFOUNDER, "A0": Role.DRIVER_CONFOUNDED, "M0": Role.MEDIATOR},
        edges=(("C0", "A0"), ("C0", "Y"), ("A0", "M0"), ("A0", "Y"), ("M0", "Y")),
        drivers=("A0",),
        outcome="Y",
        correct_adjustment={"A0": ("C0",)},
    )
    assert ("C0", "A0") in unoriented_edges(scaffold)  # confirms the setup: genuinely ambiguous
    assert _has_balanced_identifiability(scaffold) is False


def test_every_generated_instance_has_both_an_ambiguous_and_a_resolved_driver():
    """What §11.3's gate actually guarantees, measured on real accepted instances.

    Not just trusted from the gate's own logic - the same standard `procedural_scaffold_plan.md` §11.5 sets.
    """
    for seed in range(30):
        scm, _ = generate_gated_procedural(seed)
        assert _has_balanced_identifiability(scm.scaffold) is True
