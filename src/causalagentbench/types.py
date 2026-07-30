"""Domain model: causal roles, the graph ``Scaffold``, and the hidden ``SCM`` answer key.

These abstractions are topology-agnostic. The generator, effects, sampler, baselines, and gates all
operate through them, so new graph structures are added simply by defining more ``Scaffold`` instances.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import networkx as nx
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: Anything ``numpy.random.default_rng`` accepts.
SeedLike = int | np.random.SeedSequence | np.random.Generator


class Role(Enum):
    """The causal role a node plays in an instance."""

    CONFOUNDER = "confounder"
    DRIVER_CONFOUNDED = "driver_confounded"
    DRIVER_CLEAN = "driver_clean"
    MEDIATOR = "mediator"
    DECOY = "decoy"
    INSTRUMENT = "instrument"
    UPSTREAM_CAUSE = "upstream_cause"
    OUTCOME = "outcome"


@dataclass(frozen=True)
class Scaffold:
    """A fixed causal-graph template: topology, roles, and the identification ground truth.

    Adding a benchmark structure means constructing another ``Scaffold``; all consumers target this type.

    Attributes:
        node_names: Canonical node order indexing every SCM matrix row/column.
        roles: Node name to its causal role (must cover exactly ``node_names``).
        edges: Directed edges as ``(parent, child)`` pairs.
        drivers: Candidate drivers to be ranked.
        outcome: The outcome node name.
        correct_adjustment: Minimal valid backdoor adjustment set per driver.
    """

    node_names: tuple[str, ...]
    roles: Mapping[str, Role]
    edges: tuple[tuple[str, str], ...]
    drivers: tuple[str, ...]
    outcome: str
    correct_adjustment: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        """Validate structural invariants so malformed scaffolds fail fast."""
        names = set(self.node_names)
        if len(names) != len(self.node_names):
            raise ValueError("node_names contains duplicates")
        if set(self.roles) != names:
            raise ValueError("roles must cover exactly the nodes")
        for parent, child in self.edges:
            if parent not in names or child not in names:
                raise ValueError(f"edge {(parent, child)} references an unknown node")
        if not set(self.drivers) <= names:
            raise ValueError("drivers must be nodes")
        if self.outcome not in names:
            raise ValueError("outcome must be a node")
        if not set(self.correct_adjustment) <= set(self.drivers):
            raise ValueError("correct_adjustment keys must be drivers")
        if not nx.is_directed_acyclic_graph(self.digraph()):
            raise ValueError("scaffold must be acyclic")

    def index(self, name: str) -> int:
        """Return the matrix index of a node."""
        return self.node_names.index(name)

    def parents(self, name: str) -> tuple[str, ...]:
        """Return the parents of a node, in canonical order."""
        return tuple(p for p, c in self.edges if c == name)

    def nodes_with_role(self, role: Role) -> tuple[str, ...]:
        """Return the nodes carrying a given role, in canonical order."""
        return tuple(n for n in self.node_names if self.roles[n] == role)

    def non_decoy_drivers(self) -> tuple[str, ...]:
        """Candidate drivers that aren't decoys, in canonical order.

        The drivers whose effect size is actually being separated/ranked - a decoy's effect is a
        structural zero by design, so it's a distractor, not a ranked candidate (`scm/target_effects.py`).
        """
        decoys = self.nodes_with_role(Role.DECOY)
        return tuple(d for d in self.drivers if d not in decoys)

    def mediator_of(self, driver: str) -> str | None:
        """Return the driver's own mediator, if it has one.

        A child carrying ``Role.MEDIATOR`` - not every driver archetype has one (instrumented and clean
        drivers don't).
        """
        mediators = [c for p, c in self.edges if p == driver and self.roles[c] is Role.MEDIATOR]
        return mediators[0] if mediators else None

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
    """A linear-Gaussian SCM: the hidden answer key for one instance.

    Variables are standardized to unit variance, so weights and effects are standardized. ``W[i, j]`` is
    the coefficient of parent ``i`` in the equation for child ``j`` (``X = W.T @ X + N``).

    Attributes:
        scaffold: The graph template this SCM realizes.
        W: Weighted adjacency (n x n).
        noise_var: Per-node exogenous noise variance (length n).
        sigma: Marginal covariance (n x n); unit diagonal by construction.
        tau: Total-effect matrix ``(I - W)^-1 - I``; ``tau[i, j]`` is the total effect of i on j.
    """

    scaffold: Scaffold
    W: FloatArray
    noise_var: FloatArray
    sigma: FloatArray
    tau: FloatArray

    def __post_init__(self) -> None:
        """Freeze the numeric payload against accidental mutation."""
        for array in (self.W, self.noise_var, self.sigma, self.tau):
            array.flags.writeable = False

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self.scaffold.node_names)

    def total_effect_on_outcome(self, name: str) -> float:
        """Return the standardized total causal effect of ``name`` on the outcome."""
        outcome_idx = self.scaffold.index(self.scaffold.outcome)
        return float(self.tau[self.scaffold.index(name), outcome_idx])

    def true_ranking(self) -> tuple[str, ...]:
        """Return drivers ordered by descending absolute total effect on the outcome."""
        return tuple(
            sorted(
                self.scaffold.drivers,
                key=lambda d: abs(self.total_effect_on_outcome(d)),
                reverse=True,
            )
        )
