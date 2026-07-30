import numpy as np

from causalagentbench import generate, sample
from causalagentbench.scm.effects import marginal_covariance
from causalagentbench.types import Role, Scaffold

# A small, hand-built example graph (confounder C, confounded driver A + mediator M, clean driver B,
# decoy Dz) - the same shape used across the test suite (tests/scm/test_scaffold.py). `generate` is
# topology-agnostic, so any valid Scaffold would do; this one is reused for the decoy fixture below.
_SCAFFOLD = Scaffold(
    node_names=("C", "A", "B", "M", "Dz", "Y"),
    roles={
        "C": Role.CONFOUNDER,
        "A": Role.DRIVER_CONFOUNDED,
        "B": Role.DRIVER_CLEAN,
        "M": Role.MEDIATOR,
        "Dz": Role.DECOY,
        "Y": Role.OUTCOME,
    },
    edges=(("C", "A"), ("C", "Y"), ("C", "Dz"), ("A", "M"), ("A", "Y"), ("M", "Y"), ("B", "Y")),
    drivers=("A", "B", "Dz"),
    outcome="Y",
    correct_adjustment={"A": ("C",), "B": (), "Dz": ("C",)},
)


def test_generation_is_deterministic():
    a = generate(42, _SCAFFOLD)
    b = generate(42, _SCAFFOLD)
    assert np.array_equal(a.W, b.W)
    assert np.array_equal(a.noise_var, b.noise_var)


def test_different_seeds_give_different_models():
    assert not np.array_equal(generate(1, _SCAFFOLD).W, generate(2, _SCAFFOLD).W)


def test_every_node_has_unit_variance():
    scm = generate(0, _SCAFFOLD)
    assert np.allclose(np.diag(scm.sigma), 1.0)


def test_incremental_sigma_matches_closed_form():
    scm = generate(3, _SCAFFOLD)
    assert np.allclose(scm.sigma, marginal_covariance(scm.W, scm.noise_var))


def test_decoy_effect_is_a_structural_zero():
    assert abs(generate(5, _SCAFFOLD).total_effect_on_outcome("Dz")) < 1e-12


def test_noise_variance_is_a_valid_fraction():
    scm = generate(7, _SCAFFOLD)
    assert np.all(scm.noise_var > 0.0)
    assert np.all(scm.noise_var <= 1.0 + 1e-9)


def test_sampled_data_is_approximately_unit_variance():
    data = sample(generate(0, _SCAFFOLD), 20_000, seed=0)
    assert np.allclose(data.var(axis=0), 1.0, atol=0.1)
