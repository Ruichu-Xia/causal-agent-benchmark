"""CauseNet grounding: sample real cause-effect concepts to steer the LLM's candidate names.

Optional and offline: reads a user-downloaded CauseNet dump (CC BY-NC 4.0, never committed). Cheap 1-2
hop queries around an effect hub yield role-candidate menus; the LLM turns them into a coherent domain.
"""

from __future__ import annotations

import bz2
import functools
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np

from causalagentbench.types import Scaffold

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CauseNetMenus:
    """Role-candidate menus sampled from CauseNet - grounding context for one scenario."""

    outcome: str
    drivers: tuple[tuple[str, float], ...]  # (concept, normalized co-occurrence strength in [0, 1])
    confounders: tuple[str, ...]
    mediators: tuple[str, ...]
    decoys: tuple[str, ...]


@functools.lru_cache(maxsize=1)
def load_causenet(dump_path: str) -> nx.DiGraph:
    """Load a CauseNet ``*.jsonl(.bz2)`` dump into a directed cause->effect graph weighted by support."""
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
            support = max(len(record.get("sources", [])), 1)
            if graph.has_edge(cause, effect):
                graph[cause][effect]["weight"] += support
            else:
                graph.add_edge(cause, effect, weight=support)
            if lineno % 500_000 == 0:
                logger.info("  ...parsed %d lines", lineno)
    logger.info("loaded CauseNet: %d concepts, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def _take(items: Iterable[str], n: int) -> tuple[str, ...]:
    """Return the first ``n`` distinct items, preserving order."""
    kept: list[str] = []
    for item in items:
        if item not in kept:
            kept.append(item)
            if len(kept) == n:
                break
    return tuple(kept)


def sample_menus(
    dump_path: str,
    scaffold: Scaffold,
    rng: np.random.Generator,
    *,
    min_outcome_causes: int = 8,
    pool_size: int = 8,
) -> CauseNetMenus:
    """Sample role-candidate menus from CauseNet via cheap 1-2 hop queries around an outcome hub.

    * outcome: a concept with at least ``min_outcome_causes`` causes (a real effect hub);
    * drivers: its causes, strength = support normalized by the strongest in the pool;
    * confounder: a cause of the outcome that also causes a pool driver (a common cause);
    * mediator: an effect of a pool driver that in turn causes the outcome (on the path);
    * decoy: an effect of a confounder candidate with no edge to the outcome (correlate, not cause).
    """
    graph = load_causenet(dump_path)
    # Both defaults are floors, not caps: the scaffold's own driver count is the real requirement, and it is
    # known here, so scale to it rather than picking a hub and discovering afterwards that it was too small.
    # Left as-is, `pool_size` silently made some scaffolds impossible rather than merely unlucky - an
    # 8-driver scaffold plus any decoy needs 9-10 causes, so every such instance failed here 100% of the
    # time regardless of which hub was drawn. Behavior is unchanged for scaffolds within the old defaults.
    needed = len(scaffold.drivers)
    hubs = [n for n in graph if graph.in_degree(n) >= max(min_outcome_causes, needed)]
    if not hubs:
        raise ValueError(f"no outcome hub with >= {max(min_outcome_causes, needed)} causes for this scaffold")
    outcome = hubs[int(rng.integers(len(hubs)))]
    logger.debug("grounded on outcome hub %r (in-degree %d)", outcome, graph.in_degree(outcome))

    causes = sorted(graph.predecessors(outcome), key=lambda c: graph[c][outcome]["weight"], reverse=True)
    drivers_pool = causes[: max(pool_size, needed)]
    if len(drivers_pool) < needed:
        raise ValueError(f"outcome {outcome!r} has fewer causes than the scaffold needs")
    max_weight = max(graph[c][outcome]["weight"] for c in drivers_pool)
    drivers = tuple((c, graph[c][outcome]["weight"] / max_weight) for c in drivers_pool)

    confounders = _take(
        (x for x in causes if any(graph.has_edge(x, d) for d in drivers_pool if d != x)),
        pool_size,
    )
    mediators = _take(
        (
            m
            for d in drivers_pool
            for m in graph.successors(d)
            if m != outcome and m != d and graph.has_edge(m, outcome)
        ),
        pool_size,
    )
    decoys = _take(
        (dd for x in confounders for dd in graph.successors(x) if dd != outcome and not graph.has_edge(dd, outcome)),
        pool_size,
    )
    return CauseNetMenus(outcome=outcome, drivers=drivers, confounders=confounders, mediators=mediators, decoys=decoys)


def format_menus(menus: CauseNetMenus) -> str:
    """Render sampled menus as a grounding line for the generator prompt (offline; not agent-facing)."""
    drivers = ", ".join(f"{concept} ({strength:.2f})" for concept, strength in menus.drivers)
    return (
        f"outcome concept: {menus.outcome}; candidate driver concepts: {drivers}; "
        f"common-cause concepts: {', '.join(menus.confounders)}; "
        f"pathway concepts: {', '.join(menus.mediators)}; "
        f"associated-only concepts: {', '.join(menus.decoys)}"
    )
