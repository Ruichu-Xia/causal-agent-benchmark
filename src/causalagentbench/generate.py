"""Structural generation via the UUMC parameterization.

Given a ``Scaffold`` (fixed topology + roles), this draws the numeric payload of an ``SCM``: edge
weights and noise variances. It uses UUMC (Herman, Wahl, Ninad & Runge, "Unitless Unrestricted
Markov-Consistent SCM Generation", CLeaR 2025, PMLR v275; arXiv:2503.17037), which makes every node
unit-variance so the causal order is not inferable from marginal variance.

This is a faithful, independent reimplementation of the UUMC coefficient step (the authors' ``UUMCdata``
package is GPL-3.0; this project stays permissively licensed and cites the paper). Per node with ``d``
parents::

    a ~ N(0, 1) per parent;  norm = sum(a^2);  r = U(0, 1)^(1/d)
    S     = a.T @ Sigma_pa @ a                 (raw signal, using the parent covariance)
    scale = sqrt(r^2 * S + (1 - r^2) * norm)
    coef  = (r / scale) * a                    (rescaled edge weights)
    noise_var = (1 - r^2) * norm / scale^2

which gives ``Var(X_j) = coef.T @ Sigma_pa @ coef + noise_var = 1`` exactly. Roots (d = 0) are
unit-variance noise. The parent covariance ``Sigma_pa`` is accumulated exactly in topological order.
Only Gaussian noise and the UUMC scheme are supported; alternatives would slot in here as siblings.
"""

from __future__ import annotations

import numpy as np

from causalagentbench.effects import total_effects
from causalagentbench.scaffold import CANONICAL
from causalagentbench.types import SCM, Scaffold, SeedLike


def generate(seed: SeedLike, scaffold: Scaffold = CANONICAL) -> SCM:
    """Return an ``SCM`` realizing ``scaffold`` with UUMC-sampled parameters, seeded by ``seed``."""
    rng = np.random.default_rng(seed)
    n_nodes = len(scaffold.node_names)
    index = {name: i for i, name in enumerate(scaffold.node_names)}

    weights = np.zeros((n_nodes, n_nodes))
    noise_var = np.zeros(n_nodes)
    sigma = np.zeros((n_nodes, n_nodes))
    processed: list[int] = []

    for name in scaffold.topological_order():
        node = index[name]
        parent_idx = [index[p] for p in scaffold.parents(name)]
        degree = len(parent_idx)

        if degree == 0:
            noise_var[node] = 1.0
            sigma[node, node] = 1.0
        else:
            raw = rng.standard_normal(degree)
            norm = float(raw @ raw)
            r = float(rng.random() ** (1.0 / degree))
            parent_cov = sigma[np.ix_(parent_idx, parent_idx)]
            raw_signal = float(raw @ parent_cov @ raw)
            scale = np.sqrt(r * r * raw_signal + (1.0 - r * r) * norm)

            coef = (r / scale) * raw
            weights[parent_idx, node] = coef
            noise_var[node] = (1.0 - r * r) * norm / (scale * scale)

            sigma[node, node] = 1.0
            for other in processed:
                covariance = float(coef @ sigma[parent_idx, other])
                sigma[node, other] = covariance
                sigma[other, node] = covariance

        processed.append(node)

    return SCM(
        scaffold=scaffold,
        W=weights,
        noise_var=noise_var,
        sigma=sigma,
        tau=total_effects(weights),
    )
