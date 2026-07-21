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
from dataclasses import dataclass

import numpy as np

from causalagentbench.task.instance import Instance
from causalagentbench.types import FloatArray

_MAX_PREVIEW_ROWS = 20


@dataclass(frozen=True)
class CausalTools:
    """The tool implementations, bound to one instance's public (agent-visible) data."""

    instance: Instance

    def preview(self, n: int = 5) -> str:
        """Return the first ``n`` rows (max 20) as CSV, for a quick sanity check of scale/format."""
        rows = max(1, min(n, _MAX_PREVIEW_ROWS))
        buffer = io.StringIO()
        buffer.write(",".join(self.instance.columns) + "\n")
        np.savetxt(buffer, self.instance.data[:rows], delimiter=",", fmt="%.6g")
        return buffer.getvalue()

    def correlate(self, x: str, y: str, given: list[str] | None = None) -> str:
        """Return the partial correlation of columns ``x`` and ``y`` given the columns in ``given``."""
        given = given or []
        error = self._validate(x=x, y=y, given=given)
        if error:
            return error
        if x == y:
            return "Error: x and y must be different columns."
        value = _partial_corr(self.instance.data, self.instance.columns, x, y, tuple(given))
        cond = f" given {given}" if given else ""
        return f"corr({x}, {y}{cond}) = {value:+.4f}"

    def regress(self, treatment: str, adjust: list[str] | None = None) -> str:
        """Return the OLS coefficient of ``treatment`` on the outcome, adjusting for ``adjust``."""
        adjust = adjust or []
        outcome = self.instance.outcome
        error = self._validate(treatment=treatment, given=adjust)
        if error:
            return error
        if treatment == outcome:
            return "Error: treatment cannot be the outcome column."
        if treatment in adjust:
            return "Error: treatment cannot also appear in adjust."
        coef = _adjusted_coefficient(self.instance.data, self.instance.columns, outcome, treatment, tuple(adjust))
        adjust_desc = f"{{{', '.join(adjust)}}}" if adjust else "{}"
        return f"regress({outcome} ~ {treatment} | adjust={adjust_desc}) = {coef:+.4f}"

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


def _adjusted_coefficient(
    data: FloatArray, columns: tuple[str, ...], outcome: str, treatment: str, adjust: tuple[str, ...]
) -> float:
    """OLS coefficient on ``treatment`` when regressing ``outcome`` on ``treatment`` plus ``adjust``."""
    predictor_idx = [_column_index(columns, name) for name in (treatment, *adjust)]
    design = np.column_stack([np.ones(data.shape[0]), data[:, predictor_idx]])
    target = data[:, _column_index(columns, outcome)]
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(coefficients[1])  # column 0 is the intercept
