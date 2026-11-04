"""Name a world's variables with CauseNet concepts so that the names never contradict the hidden graph.

CauseNet (causenet.org, CC BY 4.0) is a graph of cause -> effect relations mined from the web. The dump is
fetched locally and never committed.
"""

from __future__ import annotations

import bz2
import functools
import hashlib
import itertools
import json
import logging
import re

import networkx as nx
import numpy as np

from causalagentbench.scm import Scaffold

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def load_causenet(dump_path: str) -> nx.DiGraph:
    """Load a CauseNet ``*.jsonl(.bz2)`` dump into a directed cause -> effect graph weighted by support.

    The edge weight is the record's ``support``: CauseNet's count of distinct linguistic patterns that state
    the relation, which favors robustly attested relations over boilerplate repeated across many pages.
    """
    logger.info("loading CauseNet dump %s (first use; parsing can take a minute)...", dump_path)
    graph = nx.DiGraph()
    opener = bz2.open if dump_path.endswith(".bz2") else open
    with opener(dump_path, "rt", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            record = json.loads(line)
            relation = record.get("causal_relation")
            if not relation:
                continue
            cause, effect = relation["cause"]["concept"], relation["effect"]["concept"]
            # Fall back to the source count only for a dump that predates the `support` field.
            support = max(int(record.get("support") or len(record.get("sources", []))), 1)
            if graph.has_edge(cause, effect):
                graph[cause][effect]["weight"] += support
            else:
                graph.add_edge(cause, effect, weight=support)
            if lineno % 500_000 == 0:
                logger.info("  ...parsed %d lines", lineno)
    logger.info("loaded CauseNet: %d concepts, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def held_out_concept(concept: str) -> bool:
    """Whether a concept is one of the fifth reserved for ``ood_vocab`` (a stable hash, independent of order)."""
    return int.from_bytes(hashlib.sha256(concept.encode("utf-8")).digest()[:8], "big") % 5 == 0


def _stem(word: str) -> str:
    """Strip a plural ending and a final "e", so singular and plural forms share a stem.

    ``diseases``/``disease``, ``gases``/``gas`` and ``headaches``/``headache`` all collide. The rule merges a
    few unrelated words (``rate``/``rat``), which only makes the near-duplicate rule stricter.
    """
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        word = word[:-1]
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def concept_words(concept: str) -> frozenset[str]:
    """Return a concept's words, split on ``_`` and ``-`` and stemmed."""
    return frozenset(_stem(word) for word in re.split(r"[_-]+", concept) if word)


def is_near_duplicate(first: str, second: str) -> bool:
    """True if one concept's words are a subset of the other's (``diabetes`` / ``diabetes_mellitus``)."""
    a, b = concept_words(first), concept_words(second)
    return a <= b or b <= a


def one_way_relations(graph: nx.DiGraph) -> nx.DiGraph:
    """Return the relations CauseNet states in one direction only; pairs claimed both ways are dropped."""
    return nx.DiGraph((u, v) for u, v in graph.edges if u != v and not graph.has_edge(v, u))


class GroundingError(ValueError):
    """No edge-preserving naming was found within the bounded search."""


class CauseNetLabeler:
    """Map a scaffold's nodes to CauseNet concepts so that the names never contradict the hidden graph.

    Four rules, all deterministic lookups:

    1. Every hidden edge A -> B is a CauseNet relation A -> B.
    2. Only relations CauseNet states in one direction may realize a hidden edge.
    3. No CauseNet relation between two chosen concepts points against the hidden causal order: if
       CauseNet says a -> b, the node named b must not be an ancestor of the node named a. Relations between
       concepts with no hidden path either way are allowed; the data shows those variables as unrelated.
    4. No two chosen concepts are near-duplicates (``is_near_duplicate``).
    """

    def __init__(self, graph: nx.DiGraph, *, max_steps: int = 3_000) -> None:
        """Index ``graph`` for the bounded search."""
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.graph = graph
        self.one_way = one_way_relations(graph)
        self.max_steps = max_steps
        # Names must work as column identifiers; this is a format check, not a vocabulary filter.
        self.concepts = frozenset(
            node for node in self.one_way if isinstance(node, str) and 2 < len(node) <= 60 and node.isprintable()
        )

    def __call__(self, scaffold: Scaffold, rng: np.random.Generator) -> dict[str, str]:
        """Return node -> concept name; never invent a name."""
        mapping = self._match(scaffold, rng)
        if mapping is None:
            raise GroundingError(f"no CauseNet embedding found within {self.max_steps} search steps")
        violations = naming_violations(self.graph, scaffold, mapping)
        if violations:
            raise RuntimeError(f"matcher produced a naming that breaks its own rules: {violations}")
        return mapping

    def _match(self, scaffold: Scaffold, rng: np.random.Generator) -> dict[str, str] | None:
        """Bounded backtracking search for a mapping that satisfies all four rules, or None."""
        template = scaffold.digraph()
        parents = {n: set(template.predecessors(n)) for n in template}
        children = {n: set(template.successors(n)) for n in template}
        ancestors = {n: nx.ancestors(template, n) for n in template}
        assigned: dict[str, str] = {}
        remaining = self.max_steps

        def consistent(node: str, concept: str) -> bool:
            """Rules 3 and 4 between ``concept`` (for ``node``) and every concept already chosen."""
            for other, name in assigned.items():
                if is_near_duplicate(concept, name):
                    return False
                if self.graph.has_edge(concept, name) and other in ancestors[node]:
                    return False
                if self.graph.has_edge(name, concept) and node in ancestors[other]:
                    return False
            return True

        def candidates(node: str) -> list[str] | None:
            """Concepts that keep every hidden edge to an assigned neighbor (rules 1-2) and pass rules 3-4."""
            domains = [set(self.one_way.successors(assigned[p])) for p in parents[node] if p in assigned]
            domains += [set(self.one_way.predecessors(assigned[c])) for c in children[node] if c in assigned]
            if not domains and assigned:
                return None
            possible = set.intersection(*domains) if domains else set(self.concepts)
            return sorted(
                concept
                for concept in possible & self.concepts
                if self.one_way.in_degree(concept) >= len(parents[node])
                and self.one_way.out_degree(concept) >= len(children[node])
                and consistent(node, concept)
            )

        def visit() -> dict[str, str] | None:
            """Assign the most constrained unassigned node next, starting from the outcome."""
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
                assigned[node] = values[int(index)]
                result = visit()
                if result is not None:
                    return result
                del assigned[node]
            return None

        return visit()


def naming_violations(graph: nx.DiGraph, scaffold: Scaffold, mapping: dict[str, str]) -> list[str]:
    """Independently re-check the four naming rules; return one message per violation."""
    template = scaffold.digraph()
    one_way = {(u, v) for u, v in graph.edges if u != v and not graph.has_edge(v, u)}
    problems = []
    for a, b in scaffold.edges:
        if (mapping[a], mapping[b]) not in one_way:
            problems.append(f"hidden edge {a}->{b} is not a one-way CauseNet relation")
    for first, second in itertools.combinations(scaffold.node_names, 2):
        x, y = mapping[first], mapping[second]
        if is_near_duplicate(x, y):
            problems.append(f"near-duplicate concepts {x!r} and {y!r}")
        if graph.has_edge(x, y) and nx.has_path(template, second, first):
            problems.append(f"CauseNet {x}->{y} points against the hidden path {second}->...->{first}")
        if graph.has_edge(y, x) and nx.has_path(template, first, second):
            problems.append(f"CauseNet {y}->{x} points against the hidden path {first}->...->{second}")
    return problems
