import numpy as np

from causalagentbench import generate, sample
from causalagentbench.effects import marginal_covariance
from causalagentbench.scaffold import DECOY


def test_generation_is_deterministic():
    a = generate(42)
    b = generate(42)
    assert np.array_equal(a.W, b.W)
    assert np.array_equal(a.noise_var, b.noise_var)


def test_different_seeds_give_different_models():
    assert not np.array_equal(generate(1).W, generate(2).W)


def test_every_node_has_unit_variance():
    scm = generate(0)
    assert np.allclose(np.diag(scm.sigma), 1.0)


def test_incremental_sigma_matches_closed_form():
    scm = generate(3)
    assert np.allclose(scm.sigma, marginal_covariance(scm.W, scm.noise_var))


def test_decoy_effect_is_a_structural_zero():
    assert abs(generate(5).total_effect_on_outcome(DECOY)) < 1e-12


def test_noise_variance_is_a_valid_fraction():
    scm = generate(7)
    assert np.all(scm.noise_var > 0.0)
    assert np.all(scm.noise_var <= 1.0 + 1e-9)


def test_sampled_data_is_approximately_unit_variance():
    data = sample(generate(0), 20_000, seed=0)
    assert np.allclose(data.var(axis=0), 1.0, atol=0.1)
