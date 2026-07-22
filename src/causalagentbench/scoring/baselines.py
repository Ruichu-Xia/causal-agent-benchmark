"""Reference solvers used to define and check instance soundness.

Each returns a per-driver estimated effect (``driver -> effect``); ``ranking`` derives an order.

* ``oracle_effects`` - regress the outcome on each driver plus its correct adjustment set (gold).
* ``correlation_effects`` - raw marginal correlation (the "no adjustment" shortcut; must fail).
* ``full_adjustment_effects`` - regress on every covariate at once (the "adjust for everything"
  shortcut; must fail because conditioning on the mediator biases the total effect).

Interpreting a coefficient as a standardized effect relies on ~unit-variance data (the UUMC property).
"""

from __future__ import annotations

import numpy as np

from causalagentbench.scoring.metrics import regression_coefficient
from causalagentbench.types import SCM, FloatArray


def _ols_coefficients(design: FloatArray, target: FloatArray) -> FloatArray:
    """Ordinary least squares: coefficient vector for ``design @ beta ~ target``."""
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def oracle_effects(data: FloatArray, scm: SCM) -> dict[str, float]:
    """Estimate each driver's effect using its correct backdoor adjustment set."""
    adjust = scm.scaffold.correct_adjustment
    return {
        d: regression_coefficient(data, scm.scaffold.node_names, scm.scaffold.outcome, d, adjust.get(d, ()))
        for d in scm.scaffold.drivers
    }


def correlation_effects(data: FloatArray, scm: SCM) -> dict[str, float]:
    """Estimate each driver's effect as its raw marginal correlation with the outcome."""
    index = scm.scaffold.index
    outcome = data[:, index(scm.scaffold.outcome)]
    return {d: float(np.corrcoef(data[:, index(d)], outcome)[0, 1]) for d in scm.scaffold.drivers}


def full_adjustment_effects(data: FloatArray, scm: SCM) -> dict[str, float]:
    """Estimate each driver's effect by regressing the outcome on every covariate at once."""
    index = scm.scaffold.index
    covariates = tuple(n for n in scm.scaffold.node_names if n != scm.scaffold.outcome)
    design = np.column_stack([np.ones(data.shape[0]), data[:, [index(n) for n in covariates]]])
    target = data[:, index(scm.scaffold.outcome)]
    coefficients = _ols_coefficients(design, target)
    position = {name: i + 1 for i, name in enumerate(covariates)}  # +1 for the intercept
    return {d: float(coefficients[position[d]]) for d in scm.scaffold.drivers}


def ranking(effects: dict[str, float]) -> tuple[str, ...]:
    """Order driver names by descending absolute effect."""
    return tuple(sorted(effects, key=lambda name: abs(effects[name]), reverse=True))
