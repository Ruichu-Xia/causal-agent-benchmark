"""Shared numeric primitives: ranking metrics and adjusted regression.

Later stages grow this into the scorer library; Stage 0 needs only rank agreement plus one regression
primitive shared by the gate-validation baselines (indexed by scaffold node name) and the agent's
``regress`` tool (indexed by display label) - the computation is identical, only the name scheme differs.
"""

from __future__ import annotations

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
