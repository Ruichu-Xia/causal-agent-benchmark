import numpy as np
import pytest

from causalagentbench import generate_gated, sample
from causalagentbench.scoring.baselines import (
    correlation_effects,
    full_adjustment_effects,
    oracle_effects,
    ranking,
)
from causalagentbench.types import Role

SEEDS = list(range(20))
EMPIRICAL_SEEDS = list(range(5))
LARGE_N = 20_000


@pytest.mark.parametrize("seed", SEEDS)
def test_gated_instance_passes_every_gate(seed):
    _, report = generate_gated(seed)
    assert report.passed
    assert report.structure_valid
    assert report.confounding_realized
    assert report.mediation_realized
    assert report.oracle_recovers


@pytest.mark.parametrize("seed", EMPIRICAL_SEEDS)
def test_oracle_recovers_truth_but_naive_shortcuts_fail(seed):
    scm, _ = generate_gated(seed)
    data = sample(scm, LARGE_N, seed=seed)
    truth = scm.true_ranking()

    # The oracle (correct adjustment sets) recovers the true ranking...
    assert ranking(oracle_effects(data, scm)) == truth
    # ...while the "no adjustment" shortcut (raw correlation) mis-ranks (confounding).
    assert ranking(correlation_effects(data, scm)) != truth
    # ...and the "adjust for everything" shortcut biases the confounded driver (mediator trap).
    confounded = scm.scaffold.nodes_with_role(Role.DRIVER_CONFOUNDED)[0]
    full_estimate = full_adjustment_effects(data, scm)[confounded]
    assert abs(full_estimate - scm.total_effect_on_outcome(confounded)) > 0.02


def test_gated_generation_is_deterministic():
    a, _ = generate_gated(7)
    b, _ = generate_gated(7)
    assert np.array_equal(a.W, b.W)
