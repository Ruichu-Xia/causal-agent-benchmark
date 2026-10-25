"""Adjusted regression shared by the agent's ``regress`` tool and the baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from causalagentbench.types import FloatArray


def regression_coefficient(
    data: FloatArray, columns: Sequence[str], outcome: str, treatment: str, adjust: Sequence[str] = ()
) -> float:
    """OLS coefficient on ``treatment`` when regressing ``outcome`` on ``treatment`` plus ``adjust``."""
    index = {name: i for i, name in enumerate(columns)}
    design_idx = [index[treatment], *(index[z] for z in adjust)]
    design = np.column_stack([np.ones(data.shape[0]), data[:, design_idx]])
    coefficients, *_ = np.linalg.lstsq(design, data[:, index[outcome]], rcond=None)
    return float(coefficients[1])  # column 0 is the intercept
