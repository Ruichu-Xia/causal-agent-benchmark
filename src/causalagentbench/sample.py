"""Draw an observational data table from an SCM.

A pure, generator-agnostic simulator that returns the raw draws. Because the SCM has population unit
variance (UUMC), the draws are already ~unit-variance, and standardization is an affine-invariant
transform that changes neither identification nor the ranking. Normalization or rescaling for the agent
is a separate presentation concern handled by later stages.
"""

from __future__ import annotations

import numpy as np

from causalagentbench.types import SCM, FloatArray, SeedLike


def sample(scm: SCM, n_samples: int, seed: SeedLike) -> FloatArray:
    """Return an ``(n_samples, n_nodes)`` matrix drawn from ``scm``.

    Simulates ``X = (I - W.T)^-1 @ N`` with independent Gaussian noise; columns follow canonical order.
    """
    rng = np.random.default_rng(seed)
    n_nodes = scm.n_nodes
    noise_std = np.sqrt(scm.noise_var)
    noise = rng.standard_normal((n_nodes, n_samples)) * noise_std[:, np.newaxis]
    return np.linalg.solve(np.eye(n_nodes) - scm.W.T, noise).T
