"""The agent's causal-analysis toolset: partial correlation, adjusted regression, PC discovery.

Each is a safe, parameterized numeric function over the instance's own data - never arbitrary code
execution, so there is no sandboxing surface to build: inputs are validated column names, outputs are
short text, and every failure mode returns a clear error string instead of raising, so the agent
can read the mistake and retry (exactly like a real tool-calling API).

* ``correlate`` - partial correlation between two columns given a conditioning set; the conditional-
  independence primitive for inferring structure (``given=[]`` is the marginal correlation).
* ``regress`` - the OLS coefficient of a treatment on the fixed outcome, adjusting for a given set; lets
  the agent test any adjustment set it hypothesizes, the same backdoor-adjusted regression the scorer
  itself validates against.
* ``discover`` - the PC algorithm over every column, returning the CPDAG it recovers. Present because a
  practitioner facing this problem runs a discovery algorithm; without it the task silently becomes
  "reimplement constraint-based discovery from individual CI tests inside a turn budget", which measures
  stamina rather than causal reasoning and transfers nowhere.

``discover`` needs the optional ``discovery`` extra (causal-learn) and is imported lazily, so the rest of
the toolset works without it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from causalagentbench.agent.discovery import DEFAULT_ALPHA, discover_graph
from causalagentbench.task.instance import Instance
from causalagentbench.types import FloatArray

_DEFAULT_ALPHA = DEFAULT_ALPHA
"""Significance level for ``discover``'s conditional-independence tests.

Documented and exposed rather than hidden: it is the one knob that materially changes what PC returns, and
a discovery baseline whose hyperparameters are unstated is not reproducible. Tighter than the usual 0.05
because these tables are large (10,000 rows), where 0.05 admits noticeably more spurious edges."""


def _as_columns(value: object, name: str) -> tuple[list[str], None] | tuple[None, str]:
    """Validate a column-set argument, returning ``(columns, None)`` or ``(None, error)``. Never coerced.

    The schema declares an array of strings, so ``given="X3"`` is the model's error to fix - but it is told
    exactly that, rather than failing somewhere unhelpful downstream.
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

    def correlate(self, x: str, y: str, given: object = None) -> str:
        """Return the partial correlation of two columns given a set, with an approximate 95% interval."""
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
        return f"corr({x}, {y} | {_column_set(given)}) = {value:+.3f}, " + _correlation_interval(
            value, len(self.instance.data), len(given)
        )

    def regress(self, treatment: str, adjust: object = None) -> str:
        """Return the OLS coefficient of ``treatment`` on the outcome under an adjustment set, with its SE."""
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
        se = "unavailable (rank-deficient design)" if standard_error is None else f"{standard_error:.3f}"
        return f"regress({treatment} | {_column_set(adjust)}) = {coef:+.3f} (se {se})"

    def discover(self, alpha: object = _DEFAULT_ALPHA) -> str:
        """Run PC over every column and return the CPDAG it recovers, as an edge list.

        The tool a data scientist would actually reach for, rather than making the agent hand-simulate
        constraint-based discovery out of individual ``correlate`` calls - which measures turn budget more
        than causal skill. It does not hand over the answer: PC returns a *CPDAG*, so edges it cannot orient
        come back undirected, and turning that into an adjustment set (and deciding what is not identified
        at all) is left to the agent. Nor is it exact at finite n - measured on a generated instance at
        n=3000 it missed one true edge and mis-oriented another - so its output is evidence to check, not
        ground truth to copy.
        """
        if isinstance(alpha, bool) or not isinstance(alpha, int | float) or not 0.0 < float(alpha) < 1.0:
            return "Error: alpha must be a number strictly between 0 and 1 (e.g. 0.01)."
        try:
            graph = discover_graph(self.instance, float(alpha))
        except ImportError:  # pragma: no cover - depends on whether the `discovery` extra is installed
            return "Error: the causal discovery tool is not available in this environment."

        columns = self.instance.columns
        oriented: list[str] = []
        undecided: list[str] = []
        for i, j in itertools.combinations(range(len(columns)), 2):
            # causal-learn's encoding, verified against known graphs: (-1, 1) is i->j, (1, -1) is j->i, and
            # (-1, -1) is an edge whose direction the observational data cannot pin down.
            head, tail = int(graph[i][j]), int(graph[j][i])
            if head == -1 and tail == -1:
                undecided.append(f"  {columns[i]} -- {columns[j]}")
            elif head == -1 and tail == 1:
                oriented.append(f"  {columns[i]} -> {columns[j]}")
            elif head == 1 and tail == -1:
                oriented.append(f"  {columns[j]} -> {columns[i]}")
        if not oriented and not undecided:
            return f"PC (alpha={float(alpha):g}) found no edges between any columns."
        return "\n".join(
            [
                f'PC (alpha={float(alpha):g}). "A -> B": oriented by the data. "A -- B": direction not '
                "determined by the data.",
                f"Oriented ({len(oriented)}):",
                *(oriented or ["  (none)"]),
                f"Undirected ({len(undecided)}):",
                *(undecided or ["  (none)"]),
            ]
        )

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


def _column_set(columns: Sequence[str]) -> str:
    """Render a conditioning or adjustment set as ``{a, b}``; ``{}`` when empty."""
    return "{" + ", ".join(columns) + "}"


def _correlation_interval(value: float, n: int, n_conditioned: int) -> str:
    """Render an approximate Fisher-z 95% interval for a partial correlation."""
    fisher_df = n - n_conditioned - 3
    if fisher_df <= 0:
        return "95% CI unavailable (too few rows)"
    # Clipping avoids atanh(+/-1) when a finite sample has a numerically perfect correlation.
    fisher_z = math.atanh(float(np.clip(value, -0.999999, 0.999999)))
    margin = 1.96 / math.sqrt(fisher_df)
    return f"95% CI [{math.tanh(fisher_z - margin):+.3f}, {math.tanh(fisher_z + margin):+.3f}]"


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
