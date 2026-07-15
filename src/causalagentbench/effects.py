"""Exact effect and covariance computations for a linear-Gaussian SCM.

Everything here is closed-form and topology-agnostic: it takes the weighted adjacency and noise
variances and returns exact quantities. This exactness is what makes the SCM a trustworthy answer key.
"""

from __future__ import annotations

import numpy as np

from causalagentbench.types import FloatArray


def total_effects(weights: FloatArray) -> FloatArray:
    """Return the total-effect matrix ``tau = (I - W)^-1 - I``.

    ``tau[i, j]`` sums the path-weight products over all directed paths from i to j: the standardized
    total causal effect of i on j.
    """
    identity = np.eye(weights.shape[0])
    return np.linalg.inv(identity - weights) - identity


def marginal_covariance(weights: FloatArray, noise_var: FloatArray) -> FloatArray:
    """Return the marginal covariance ``Sigma = (I - W.T)^-1 @ diag(noise_var) @ (I - W.T)^-T``.

    From ``X = W.T @ X + N`` we get ``X = (I - W.T)^-1 @ N``. Unlike ``total_effects``, this includes
    non-causal (backdoor) association, which is the confounding the benchmark tests for.
    """
    inv = np.linalg.inv(np.eye(weights.shape[0]) - weights.T)
    return inv @ np.diag(noise_var) @ inv.T
