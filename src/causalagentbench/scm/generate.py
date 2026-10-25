"""Numeric generation: a correlation matrix drawn on the motif graph, then realized on the true graph.

Two steps keep the data from saying anything about edge directions beyond the CPDAG:

1. ``draw_correlation`` draws a unit-variance linear-Gaussian model on the motif graph ``G0`` and returns
   its correlation matrix ``Sigma``.
2. ``realize`` takes the true DAG ``D``, drawn from ``G0``'s Markov equivalence class independently of
   ``Sigma``, and sets each node's weights to its population regression on its parents in ``D``:

       beta_j = Sigma[pa(j), pa(j)]^-1 Sigma[pa(j), j],   noise_j = 1 - Sigma[j, pa(j)] beta_j

   Markov equivalent DAGs encode the same conditional independences, so ``D`` reproduces ``Sigma``
   exactly. The observational distribution is therefore identical whichever member was drawn, and no
   statistic of the data (variance, R^2, coefficient sizes) can predict an orientation the CPDAG leaves
   open ("covariance transfer").

Every variable has unit variance, so marginal variances carry no ordering (varsortability is 1/2).
Per-parent weights are bounded away from zero and each non-root's explained variance is bounded, which
keeps effects estimable at the benchmark's sample size; random signs keep effect signs unpredictable.
"""

from __future__ import annotations

import numpy as np

from causalagentbench.scm.effects import marginal_covariance, total_effects
from causalagentbench.types import SCM, FloatArray, Scaffold

EDGE_MAGNITUDE = (0.5, 1.5)
"""Relative per-parent weight magnitude before rescaling; bounded away from zero so no edge is negligible."""

EXPLAINED_VARIANCE = (0.5, 0.9)
"""R^2 of each non-root node given its parents in the motif graph."""


def draw_correlation(scaffold: Scaffold, rng: np.random.Generator) -> FloatArray:
    """Return the correlation matrix of a random unit-variance linear-Gaussian model on ``scaffold``.

    Per node, parent weights are drawn with random signs and magnitudes in ``EDGE_MAGNITUDE``, then
    rescaled so the parents explain a fraction drawn from ``EXPLAINED_VARIANCE`` of its unit variance.
    """
    index = {name: i for i, name in enumerate(scaffold.node_names)}
    sigma = np.eye(len(index))
    for name in scaffold.topological_order():
        parents = [index[p] for p in scaffold.parents(name)]
        if not parents:
            continue
        raw = rng.uniform(*EDGE_MAGNITUDE, size=len(parents)) * rng.choice((-1.0, 1.0), size=len(parents))
        r2 = rng.uniform(*EXPLAINED_VARIANCE)
        weights = raw * np.sqrt(r2 / float(raw @ sigma[np.ix_(parents, parents)] @ raw))
        # Covariances with every earlier node follow from the parents' covariances; later nodes fill
        # their own rows when they are reached in topological order.
        column = sigma[:, parents] @ weights
        j = index[name]
        sigma[j, :] = column
        sigma[:, j] = column
        sigma[j, j] = 1.0
    return sigma


def realize(scaffold: Scaffold, sigma: FloatArray) -> SCM:
    """Return the SCM on ``scaffold`` whose population distribution is exactly ``N(0, sigma)``.

    Raises ``ValueError`` if ``scaffold`` is not Markov to ``sigma`` (for example, a DAG outside the
    equivalence class ``sigma`` was drawn on).
    """
    n = len(scaffold.node_names)
    index = {name: i for i, name in enumerate(scaffold.node_names)}
    weights = np.zeros((n, n))
    noise_var = np.ones(n)
    for name in scaffold.node_names:
        parents = [index[p] for p in scaffold.parents(name)]
        if not parents:
            continue
        j = index[name]
        beta = np.linalg.solve(sigma[np.ix_(parents, parents)], sigma[parents, j])
        weights[parents, j] = beta
        noise_var[j] = 1.0 - float(sigma[j, parents] @ beta)
    if np.max(np.abs(marginal_covariance(weights, noise_var) - sigma)) > 1e-8:
        raise ValueError("scaffold is not Markov to sigma")
    return SCM(scaffold=scaffold, W=weights, noise_var=noise_var, sigma=sigma, tau=total_effects(weights))
