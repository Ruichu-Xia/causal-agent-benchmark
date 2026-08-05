"""The agent's causal-analysis toolset: preview, partial correlation, adjusted regression.

Three tools, each a safe, parameterized numeric function over the instance's own data - never arbitrary
code execution, so there is no sandboxing surface to build: inputs are validated column names, outputs
are floats or CSV, and every failure mode returns a clear error string instead of raising, so the agent
can read the mistake and retry (exactly like a real tool-calling API).

* ``preview`` - a first look at the table (sanity-check scale/format).
* ``correlate`` - partial correlation between two columns given a conditioning set; the conditional-
  independence primitive for inferring structure (``given=[]`` is the marginal correlation).
* ``regress`` - the OLS coefficient of a treatment on the fixed outcome, adjusting for a given set; lets
  the agent test any adjustment set it hypothesizes, the same backdoor-adjusted regression the scorer
  itself validates against.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from causalagentbench.task.instance import Instance
from causalagentbench.types import FloatArray

_MAX_PREVIEW_ROWS = 20


def _row_count_error(value: object) -> str | None:
    """Return an error string if ``value`` is not an ``int``, else ``None``. Type checked, never coerced.

    The tool schema handed to the model declares ``n`` as an integer, so honoring that is the model's job -
    a quoted ``"10"`` is a schema violation and is reported as one, not quietly accepted. What this fixes is
    only the *message*: the type mismatch used to surface from deep inside ``min()`` as
    ``'<' not supported between instances of 'int' and 'str'``, which names nothing the model can act on and
    was observed live being read as the tool itself being broken. ``bool`` is rejected despite subclassing
    ``int``: ``preview(n=True)`` is a mistake, not a request for one row.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return f"Error: n must be an integer number of rows, got {value!r}."
    return None


def _as_columns(value: object, name: str) -> tuple[list[str], None] | tuple[None, str]:
    """Validate a column-set argument, returning ``(columns, None)`` or ``(None, error)``. Never coerced.

    Same contract as ``_row_count_error``: the schema declares an array of strings, so ``given="X3"`` is the
    model's error to fix - but it is told exactly that, rather than failing somewhere unhelpful downstream.
    Returns the list rather than just a verdict so the caller gets a properly typed value out of the check
    it already had to perform, instead of re-deriving it behind a cast.
    """
    if value is None:
        return [], None
    if isinstance(value, str) or not isinstance(value, Sequence) or not all(isinstance(x, str) for x in value):
        return None, f"Error: {name} must be a list of column names, got {value!r}."
    return list(value), None


@dataclass(frozen=True)
class CausalTools:
    """The tool implementations, bound to one instance's public (agent-visible) data."""

    instance: Instance

    def preview(self, n: object = 5) -> str:
        """Return the first ``n`` rows (max 20) as CSV, for a quick sanity check of scale/format."""
        error = _row_count_error(n)
        if error:
            return error
        assert isinstance(n, int)
        rows = max(1, min(n, _MAX_PREVIEW_ROWS))
        buffer = io.StringIO()
        buffer.write(",".join(self.instance.columns) + "\n")
        np.savetxt(buffer, self.instance.data[:rows], delimiter=",", fmt="%.6g")
        return buffer.getvalue()

    def correlate(self, x: str, y: str, given: object = None) -> str:
        """Return partial correlation with sample size and an approximate 95% Fisher interval."""
        given, columns_error = _as_columns(given, "given")
        if given is None:
            return columns_error or ""
        error = self._validate(x=x, y=y, given=given)
        if error:
            return error
        if x == y:
            return "Error: x and y must be different columns."
        if len(set(given)) != len(given):
            return "Error: given must not contain duplicate columns."
        if x in given or y in given:
            return "Error: given must not contain x or y (conditioning on a variable on itself)."
        try:
            value = _partial_corr(self.instance.data, self.instance.columns, x, y, tuple(given))
        except np.linalg.LinAlgError:
            return "Error: correlation matrix is singular for these columns; choose a different conditioning set."
        cond = f" given {given}" if given else ""
        uncertainty = _correlation_uncertainty(value, len(self.instance.data), len(given))
        return f"corr({x}, {y}{cond}) = {value:+.4f} ({uncertainty})"

    def regress(self, treatment: str, adjust: object = None) -> str:
        """Return an OLS coefficient with sample size, standard error, and approximate 95% interval."""
        adjust, columns_error = _as_columns(adjust, "adjust")
        if adjust is None:
            return columns_error or ""
        outcome = self.instance.outcome
        error = self._validate(treatment=treatment, given=adjust)
        if error:
            return error
        if treatment == outcome:
            return "Error: treatment cannot be the outcome column."
        if treatment in adjust:
            return "Error: treatment cannot also appear in adjust."
        if outcome in adjust:
            return "Error: adjust cannot contain the outcome column."
        if len(set(adjust)) != len(adjust):
            return "Error: adjust must not contain duplicate columns."
        coef, standard_error = _ols_coefficient_and_standard_error(
            self.instance.data, self.instance.columns, outcome, treatment, tuple(adjust)
        )
        adjust_desc = f"{{{', '.join(adjust)}}}" if adjust else "{}"
        uncertainty = _regression_uncertainty(coef, standard_error, len(self.instance.data), len(adjust) + 2)
        return f"regress({outcome} ~ {treatment} | adjust={adjust_desc}) = {coef:+.4f} ({uncertainty})"

    def _validate(self, given: list[str], **named: str) -> str | None:
        """Return an error string if any referenced column is unknown, else None."""
        columns = set(self.instance.columns)
        for label, value in named.items():
            if value not in columns:
                return f"Error: unknown column {value!r} for {label!r}. Known columns: {list(self.instance.columns)}"
        for value in given:
            if value not in columns:
                return f"Error: unknown column {value!r} in given/adjust. Known columns: {list(self.instance.columns)}"
        return None


def _column_index(columns: tuple[str, ...], name: str) -> int:
    """Return the position of ``name`` in ``columns``."""
    return columns.index(name)


def _partial_corr(data: FloatArray, columns: tuple[str, ...], x: str, y: str, given: tuple[str, ...]) -> float:
    """Partial correlation of ``x``, ``y`` given ``given``, via the precision-matrix identity.

    ``-precision[i,j] / sqrt(precision[i,i] * precision[j,j])`` on the correlation submatrix of
    ``{x, y, *given}`` - numerically stable and, for ``given=()``, exactly the marginal correlation.
    """
    idx = [_column_index(columns, name) for name in (x, y, *given)]
    corr = np.corrcoef(data[:, idx], rowvar=False)
    corr = np.atleast_2d(corr)
    precision = np.linalg.inv(corr)
    return float(-precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1]))


def _correlation_uncertainty(value: float, n: int, n_conditioned: int) -> str:
    """Return sample-size context and an approximate Fisher-z 95% interval for a partial correlation."""
    fisher_df = n - n_conditioned - 3
    if fisher_df <= 0:
        return f"n={n}, insufficient degrees of freedom for an interval"
    # Clipping avoids atanh(+/-1) when a finite sample has a numerically perfect correlation.
    fisher_z = math.atanh(float(np.clip(value, -0.999999, 0.999999)))
    margin = 1.96 / math.sqrt(fisher_df)
    lower, upper = math.tanh(fisher_z - margin), math.tanh(fisher_z + margin)
    return f"n={n}, approx_95%CI=[{lower:+.4f}, {upper:+.4f}]"


def _ols_coefficient_and_standard_error(
    data: FloatArray,
    columns: tuple[str, ...],
    outcome: str,
    treatment: str,
    adjust: tuple[str, ...],
) -> tuple[float, float | None]:
    """Fit OLS and return the treatment coefficient plus standard error when the design is full rank."""
    index = {name: i for i, name in enumerate(columns)}
    design_idx = [index[treatment], *(index[z] for z in adjust)]
    design = np.column_stack([np.ones(data.shape[0]), data[:, design_idx]])
    target = data[:, index[outcome]]
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    coefficient = float(coefficients[1])
    degrees_of_freedom = data.shape[0] - rank
    if rank != design.shape[1] or degrees_of_freedom <= 0:
        return coefficient, None
    residuals = target - design @ coefficients
    residual_variance = float(residuals @ residuals) / degrees_of_freedom
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    return coefficient, float(math.sqrt(max(0.0, covariance[1, 1])))


def _regression_uncertainty(coefficient: float, standard_error: float | None, n: int, n_parameters: int) -> str:
    """Render OLS uncertainty using a normal-approximation 95% interval without extra dependencies."""
    if standard_error is None:
        return f"n={n}, parameters={n_parameters}, standard error unavailable (rank-deficient design)"
    margin = 1.96 * standard_error
    return f"n={n}, se={standard_error:.4f}, approx_95%CI=[{coefficient - margin:+.4f}, {coefficient + margin:+.4f}]"
