"""Shared numeric primitives: ranking metrics and adjusted regression.

Later stages grow this into the scorer library; Stage 0 needs only rank agreement plus one regression
primitive shared by the gate-validation baselines (indexed by scaffold node name) and the agent's
``regress`` tool (indexed by display label) - the computation is identical, only the name scheme differs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from causalagentbench.types import FloatArray


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall rank correlation (tau-a): +1 identical, -1 reversed, 0 none; ties are ignored."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            product = (x[i] - x[j]) * (y[i] - y[j])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    denominator = concordant + discordant
    return (concordant - discordant) / denominator if denominator else 1.0


def kendall_tau_zscore(tau: float, n: int) -> float | None:
    """Standardize ``tau`` against its null distribution (a uniformly random ranking of ``n`` items).

    Raw tau isn't comparable across ranking tasks of different sizes: both its achievable granularity and
    its variance under chance depend on ``n`` (at ``n=3``, tau can only take {-1, -1/3, +1/3, +1}; at
    ``n=8``, a much finer grid) - a given tau is unremarkable at small ``n`` and unusual at large ``n``.
    Uses the standard asymptotic variance of Kendall's tau-a under the null of no association,
    ``Var(tau) = 2(2n+5) / (9n(n-1))`` (Kendall 1938) - ``None`` when ``n < 2``, where tau itself is a
    vacuous constant (``kendall_tau`` returns 1.0), not a real correlation to standardize.
    """
    if n < 2:
        return None
    variance = 2 * (2 * n + 5) / (9 * n * (n - 1))
    return tau / math.sqrt(variance)


def regression_coefficient(
    data: FloatArray, columns: Sequence[str], outcome: str, treatment: str, adjust: Sequence[str] = ()
) -> float:
    """OLS coefficient on ``treatment`` when regressing ``outcome`` on ``treatment`` plus ``adjust``.

    ``columns`` indexes ``data`` by name - scaffold node names for the gate-validation baselines,
    display labels for the agent's tool - the numeric computation is name-scheme-agnostic.
    """
    index = {name: i for i, name in enumerate(columns)}
    design_idx = [index[treatment], *(index[z] for z in adjust)]
    design = np.column_stack([np.ones(data.shape[0]), data[:, design_idx]])
    target = data[:, index[outcome]]
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(coefficients[1])  # column 0 is the intercept
