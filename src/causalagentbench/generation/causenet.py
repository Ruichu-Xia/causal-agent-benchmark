"""Deterministic names backed by actual CauseNet relations, without an LLM."""

from __future__ import annotations

import hashlib
from pathlib import Path

import networkx as nx
import numpy as np

from causalagentbench.generation.grounding import load_causenet
from causalagentbench.task.labeling import Labeling, LabelProvenance
from causalagentbench.types import SCM, Scaffold

_GENERIC = frozenset(
    {
        "problem",
        "problems",
        "change",
        "changes",
        "issue",
        "issues",
        "damage",
        "harm",
        "death",
        "deaths",
        "failure",
        "effects",
        "loss",
        "losses",
        "disease",
        "diseases",
        "condition",
        "conditions",
        "activity",
        "activities",
        "benefit",
        "benefits",
        "cause",
        "effect",
        "factor",
        "factors",
        "process",
        "development",
        "behavior",
        "behaviour",
    }
)


class GroundingError(ValueError):
    """No edge-preserving naming was found within the bounded search."""


class CauseNetLabeler:
    """Map SCM nodes to distinct CauseNet concepts while preserving every simulated edge.

    CauseNet is a source of claimed relations, not numerical effects or an exhaustive causal model.
    We select a subgraph: extra relations in CauseNet need not be included in the simulated SCM.
    The public task contains names only. Relation choices and the corpus hash stay in the evaluator key.
    """

    def __init__(self, graph: nx.DiGraph, *, source: str = "causenet", max_steps: int = 20_000) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.graph = graph
        self.source = source
        self.max_steps = max_steps
        self.concepts = frozenset(
            node
            for node in graph
            if isinstance(node, str) and 2 < len(node) <= 60 and "\n" not in node and node not in _GENERIC
        )

    @classmethod
    def from_path(cls, path: str | Path) -> CauseNetLabeler:
        """Load the public CauseNet JSONL/bz2 dump once and record its SHA-256."""
        path = Path(path)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        return cls(load_causenet(str(path)), source=f"causenet:sha256:{digest}")

    def __call__(self, scm: SCM, rng: np.random.Generator) -> Labeling:
        """Return exact source concept names; never invent a name or emit a relationship description."""
        mapping = self._match(scm.scaffold, rng)
        if mapping is None:
            raise GroundingError(f"no CauseNet embedding found within {self.max_steps} search steps")
        assert all(self.graph.has_edge(mapping[a], mapping[b]) for a, b in scm.scaffold.edges)
        return Labeling(
            node_to_label=mapping,
            context=(
                "Synthetic continuous measurements named after concepts in CauseNet. "
                "The data and effect sizes are simulated, not measured real-world effects."
            ),
            column_schema={},
            provenance=LabelProvenance(
                alignment=1.0,
                n_aligned=len(mapping),
                n_neutral=0,
                n_misaligned=0,
                node_kind=dict.fromkeys(mapping, "aligned"),
                domain=mapping[scm.scaffold.outcome],
                producer=self.source,
            ),
        )

    def _match(self, scaffold: Scaffold, rng: np.random.Generator) -> dict[str, str] | None:
        template = scaffold.digraph()
        parents = {n: set(template.predecessors(n)) for n in template}
        children = {n: set(template.successors(n)) for n in template}
        assigned: dict[str, str] = {}
        used: set[str] = set()
        remaining = self.max_steps

        def candidates(node: str) -> list[str] | None:
            domains = [set(self.graph.successors(assigned[p])) for p in parents[node] if p in assigned]
            domains += [set(self.graph.predecessors(assigned[c])) for c in children[node] if c in assigned]
            if not domains and assigned:
                return None
            possible = set.intersection(*domains) if domains else set(self.concepts)
            return sorted(
                concept
                for concept in (possible & self.concepts) - used
                if self.graph.in_degree(concept) >= len(parents[node])
                and self.graph.out_degree(concept) >= len(children[node])
            )

        def visit() -> dict[str, str] | None:
            nonlocal remaining
            if len(assigned) == len(template):
                return dict(assigned)
            options = []
            for node in template:
                if node in assigned or (not assigned and node != scaffold.outcome):
                    continue
                values = candidates(node)
                if values is not None:
                    options.append((node, values))
            if not options:
                return None
            node, values = min(options, key=lambda item: (len(item[1]), -template.degree(item[0]), item[0]))
            for index in rng.permutation(len(values)):
                if remaining <= 0:
                    return None
                remaining -= 1
                concept = values[int(index)]
                assigned[node] = concept
                used.add(concept)
                result = visit()
                if result is not None:
                    return result
                used.remove(concept)
                del assigned[node]
            return None

        return visit()
