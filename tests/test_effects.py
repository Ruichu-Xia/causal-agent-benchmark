import numpy as np

from causalagentbench.effects import marginal_covariance, total_effects


def test_total_effects_include_mediated_paths():
    # chain 0 -> 1 -> 2
    weights = np.zeros((3, 3))
    weights[0, 1] = 0.5
    weights[1, 2] = 0.4
    tau = total_effects(weights)
    assert np.isclose(tau[0, 1], 0.5)
    assert np.isclose(tau[1, 2], 0.4)
    assert np.isclose(tau[0, 2], 0.5 * 0.4)  # the only 0->2 path is mediated through 1
    assert np.isclose(tau[2, 0], 0.0)


def test_total_effects_have_zero_diagonal():
    weights = np.zeros((2, 2))
    weights[0, 1] = 0.7
    assert np.allclose(np.diag(total_effects(weights)), 0.0)


def test_marginal_covariance_single_edge():
    # X0 = N0, X1 = w*X0 + N1, unit noise
    w = 0.6
    weights = np.zeros((2, 2))
    weights[0, 1] = w
    sigma = marginal_covariance(weights, np.array([1.0, 1.0]))
    assert np.isclose(sigma[0, 0], 1.0)
    assert np.isclose(sigma[1, 1], w * w + 1.0)
    assert np.isclose(sigma[0, 1], w)  # Cov(X0, X1) = w * Var(X0)
