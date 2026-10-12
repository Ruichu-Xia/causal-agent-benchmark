"""Reference solvers used to define and check instance soundness.

Each returns a per-driver estimated effect (``driver -> effect``); ``ranking`` derives an order.

* ``oracle_effects`` - regress the outcome on each driver plus its correct adjustment set (gold).
* ``correlation_effects`` - raw marginal correlation (the "no adjustment" shortcut; must fail).
* ``full_adjustment_effects`` - regress on every covariate at once (the "adjust for everything"
  shortcut; must fail because conditioning on the mediator biases the total effect).

Interpreting a coefficient as a standardized effect relies on ~unit-variance data (the UUMC property).
"""

from __future__ import annotations

from collections.abc import Sequence

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


def effect_scale(scm: SCM) -> float:
    """The instance's own effect magnitude, the denominator that makes estimation error comparable.

    Median |true effect| over the candidate drivers. Effects shrink as ``1/sqrt(N)`` against the outcome's
    fixed variance budget, so an absolute error floor would mean something different at every driver count.
    Falls back to 1.0 on a degenerate all-decoy instance (which the gates reject) rather than dividing by
    zero.
    """
    magnitudes = [abs(scm.total_effect_on_outcome(d)) for d in scm.scaffold.drivers]
    scale = float(np.median(magnitudes)) if magnitudes else 0.0
    return scale if scale > 0.0 else 1.0


def normalized_effect_error(scm: SCM, estimates: dict[str, float]) -> float:
    """Mean over drivers of ``|estimate - true| / effect_scale``, each clipped at 1.

    The single definition of "how wrong are these estimates", shared by ``scoring/gates.py``'s causal-load
    gate and ``scoring/score.py``'s ``effect_accuracy`` so the property certified and the property scored
    cannot drift. They previously differed: the gate divided after averaging while the scorer clipped per
    driver first, which agreed only while nothing clipped - measured to disagree (0.554 vs 0.471) as soon as
    one driver did.

    Clipped per driver so a single absurd estimate cannot swamp an instance. Loss 1 means every error
    reaches the task's effect scale; predicting zero can have lower loss, especially on zero-effect candidates.
    """
    scale = effect_scale(scm)
    errors = [min(abs(estimates[d] - scm.total_effect_on_outcome(d)) / scale, 1.0) for d in scm.scaffold.drivers]
    return float(np.mean(errors)) if errors else 0.0


def population_coefficient(scm: SCM, treatment: str, adjust: Sequence[str] = ()) -> float:
    """OLS coefficient on ``treatment`` regressing the outcome on ``treatment`` + ``adjust``, at n -> inf.

    Read off the SCM's own marginal covariance instead of a sampled table: for regressor set ``S``, the
    coefficient vector is ``Sigma[S, S]^-1 @ Sigma[S, outcome]``, and ``treatment`` is placed first so its
    coefficient is entry 0. No intercept term - every variable is mean-zero by construction.

    Exact, so a gate built on this is a property of the *SCM* rather than of one draw, and it costs no
    sampling - which is what lets the shortcut gate run before ``scoring/gates.py``'s expensive sampling
    step rather than after it.
    """
    index = scm.scaffold.index
    regressors = [index(treatment), *(index(z) for z in adjust)]
    coefficients, *_ = np.linalg.lstsq(
        scm.sigma[np.ix_(regressors, regressors)], scm.sigma[regressors, index(scm.scaffold.outcome)], rcond=None
    )
    return float(coefficients[0])


def shortcut_effects(scm: SCM) -> dict[str, dict[str, float]]:
    """Each blind policy's per-driver effect estimate, at the population level.

    "Blind" here means the policy reads no column names and needs no causal graph - it is reachable by
    regression alone, with no orientation and no adjustment-set reasoning. If any of these ranks the drivers
    well, the instance does not require causal reasoning to rank, which is exactly what
    ``scoring/gates.py::evaluate_gates`` rejects on.

    Deliberately *not* including an "adjust for all non-descendants of the treatment" policy: it is a valid
    backdoor set for every driver in any causally sufficient DAG, but applying it requires knowing which
    columns are descendants, and deriving that is orientation - genuine causal work, and the skill the
    benchmark means to reward. Gating against it would reject instances for being solvable by the intended
    method.
    """
    scaffold = scm.scaffold
    covariates = tuple(n for n in scaffold.node_names if n != scaffold.outcome)
    return {
        # The measured 0.63-tau shortcut: one marginal regression per candidate, no graph at all.
        "correlation": {d: population_coefficient(scm, d) for d in scaffold.drivers},
        "full_adjustment": {
            d: population_coefficient(scm, d, [c for c in covariates if c != d]) for d in scaffold.drivers
        },
        "candidate_adjustment": {
            d: population_coefficient(scm, d, [c for c in scaffold.drivers if c != d]) for d in scaffold.drivers
        },
    }
