"""Linear-Gaussian structural causal models: the graph ``Scaffold``, the ``SCM``, and how a world's numbers are drawn.

Two steps keep the data from saying anything about edge directions beyond the CPDAG:

1. ``draw_correlation`` draws a unit-variance linear-Gaussian model on the drawn graph ``G0`` and returns
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
keeps effects estimable at the environment's sample size; random signs keep effect signs unpredictable.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

EDGE_MAGNITUDE = (0.5, 1.5)
"""Relative per-parent weight magnitude before rescaling; bounded away from zero so no edge is negligible."""

EXPLAINED_VARIANCE = (0.5, 0.9)
"""R^2 of each non-root node given its parents in the drawn graph."""


@dataclass(frozen=True)
class Scaffold:
    """A causal graph over named nodes.

    Attributes:
        node_names: Canonical node order indexing every SCM matrix row/column.
        edges: Directed edges as ``(parent, child)`` pairs.
        outcome: A sink node; CauseNet naming starts its search there.
    """

    node_names: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    outcome: str

    def __post_init__(self) -> None:
        """Validate structural invariants so malformed scaffolds fail fast."""
        names = set(self.node_names)
        if len(names) != len(self.node_names):
            raise ValueError("node_names contains duplicates")
        for parent, child in self.edges:
            if parent not in names or child not in names:
                raise ValueError(f"edge {(parent, child)} references an unknown node")
        if self.outcome not in names:
            raise ValueError("outcome must be a node")
        if not nx.is_directed_acyclic_graph(self.digraph()):
            raise ValueError("scaffold must be acyclic")

    def parents(self, name: str) -> tuple[str, ...]:
        """Return the parents of a node, in canonical order."""
        return tuple(p for p, c in self.edges if c == name)

    def digraph(self) -> nx.DiGraph:
        """Return the scaffold as a networkx directed graph."""
        graph = nx.DiGraph()
        graph.add_nodes_from(self.node_names)
        graph.add_edges_from(self.edges)
        return graph

    def topological_order(self) -> tuple[str, ...]:
        """Return a deterministic topological order (each node after its parents)."""
        return tuple(nx.lexicographical_topological_sort(self.digraph()))


@dataclass(frozen=True)
class SCM:
    """A linear-Gaussian SCM with unit-variance variables.

    ``W[i, j]`` is the coefficient of parent ``i`` in the equation for child ``j`` (``X = W.T @ X + N``).

    Attributes:
        scaffold: The graph this SCM realizes.
        W: Weighted adjacency (n x n).
        noise_var: Per-node exogenous noise variance (length n).
        sigma: Marginal covariance (n x n); unit diagonal by construction.
    """

    scaffold: Scaffold
    W: FloatArray
    noise_var: FloatArray
    sigma: FloatArray

    def __post_init__(self) -> None:
        """Freeze the numeric payload against accidental mutation."""
        for array in (self.W, self.noise_var, self.sigma):
            array.flags.writeable = False


def marginal_covariance(weights: FloatArray, noise_var: FloatArray) -> FloatArray:
    """Return the marginal covariance ``Sigma = (I - W.T)^-1 @ diag(noise_var) @ (I - W.T)^-T``.

    From ``X = W.T @ X + N`` we get ``X = (I - W.T)^-1 @ N``.
    """
    inv = np.linalg.inv(np.eye(weights.shape[0]) - weights.T)
    return inv @ np.diag(noise_var) @ inv.T


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
    return SCM(scaffold=scaffold, W=weights, noise_var=noise_var, sigma=sigma)
