"""CauseNet grounding: a real-world concept pool to steer the LLM's naming of a hidden SCM.

Optional and offline: reads a user-downloaded CauseNet dump (CC BY 4.0, never committed; MD5
``8bf12257e71713a63403bc8fe8bf71bf`` for the precision dump this project targets). For one build it picks
an outcome hub and gathers the concepts around it, which the naming generator then draws a coherent domain
from (``generation/labeler.py``).

What CauseNet is: one flat list of ``(cause, effect)`` concept pairs extracted from web/Wikipedia
sentences. It carries no ontology and no role typing, so it cannot say which concepts are "confounders"
and which are "drivers" - measured on this dump, structural queries for those two roles return the *same*
concepts (Jaccard 0.87), because a real concept has no single causal role. Role accuracy therefore comes
from the naming model's own world knowledge; CauseNet's job is to supply a real, coherent, domain-specific
vocabulary rather than to assign it (see ``concept_pool``).
"""

from __future__ import annotations

import bz2
import functools
import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

#: The structural queries ``menus_for_hub`` runs, in a stable order. These are *gathering* strategies -
#: several hops around the hub - not a claim that the resulting concepts are role-distinct; they are
#: flattened into one pool by ``concept_pool``.
MENU_CATEGORIES: tuple[str, ...] = (
    "driver",
    "confounder",
    "mediator",
    "instrument",
    "decoy",
    "upstream",
    "collider",
)


@dataclass(frozen=True)
class CauseNetMenus:
    """Concepts gathered around one outcome hub, keyed by the query that found them."""

    outcome: str
    menus: Mapping[str, tuple[str, ...]]


@functools.lru_cache(maxsize=1)
def load_causenet(dump_path: str) -> nx.DiGraph:
    r"""Load a CauseNet ``*.jsonl(.bz2)`` dump into a directed cause->effect graph weighted by support.

    One JSON object per line::

        {"causal_relation": {"cause": {"concept": "lack_of_oxygen"},
                             "effect": {"concept": "coma"}},
         "sources": [{"type": "wikipedia_sentence", "payload": {..., "sentence": "...",
                      "path_pattern": "[[cause]]/N\t+nsubj\tcause/NN\t+nmod:of\t[[effect]]/N"}}, ...],
         "support": 3}

    The edge weight is the record's own ``support``, which is CauseNet's count of **distinct linguistic
    path patterns** expressing the relation (verified against this dump: it equals the number of distinct
    ``path_pattern`` values in 2909 of 3001 sampled records, and differs from ``len(sources)`` in 98% of
    them). ``len(sources)`` counts raw matched sentences instead, which rewards boilerplate repeated across
    many web pages rather than robustly-attested causation - measured, it ranks
    ``fraudulent_misrepresentation`` above ``disease`` among causes of ``death``, and ``use``/``animals``
    above ``fire``/``storm`` among causes of ``damage``, with only 1-3 of the top 8 agreeing between the
    two. Since this weight orders the concept pool shown to the naming generator, the difference is a
    direct quality difference in the vocabulary a built instance is named from.
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


def _take(items: Iterable[str], n: int) -> tuple[str, ...]:
    """Return the first ``n`` distinct items, preserving order."""
    kept: list[str] = []
    for item in items:
        if item not in kept:
            kept.append(item)
            if len(kept) == n:
                break
    return tuple(kept)


def menus_for_hub(graph: nx.DiGraph, hub: str, *, base_size: int, cap: int) -> dict[str, tuple[str, ...]]:
    """Gather concepts from ``hub``'s 1-2 hop neighborhood via several structural queries.

    ``base_size`` caps how many of the hub's strongest causes seed the queries; ``cap`` bounds each result.
    The queries mirror the SCM roles - not because CauseNet's concepts *are* role-typed (they are not; see
    ``concept_pool``), but because running several different structural queries is what yields a varied
    vocabulary instead of just the hub's direct causes:

    * ``driver``     - direct causes of the outcome.
    * ``confounder`` - a cause of the outcome that *also* causes another cause.
    * ``mediator``   - an effect of a cause that *itself* causes the outcome.
    * ``instrument`` - a cause of a cause that does *not* itself cause the outcome.
    * ``decoy``      - an effect of a confounder-ish concept with *no* edge to the outcome.
    * ``upstream``   - a cause of a confounder-ish concept (a distal cause).
    * ``collider``   - a common effect of *two* of the outcome's causes, with no edge to the outcome.
    """
    causes = sorted(graph.predecessors(hub), key=lambda c: graph[c][hub]["weight"], reverse=True)[:base_size]
    cause_set = set(causes)
    shared_effects: dict[str, int] = {}
    for d in causes:
        for e in graph.successors(d):
            if e != hub and not graph.has_edge(e, hub):
                shared_effects[e] = shared_effects.get(e, 0) + 1

    confounders = _take((x for x in causes if any(graph.has_edge(x, d) for d in causes if d != x)), cap)
    return {
        "driver": tuple(causes[:cap]),
        "confounder": confounders,
        "mediator": _take(
            (m for d in causes for m in graph.successors(d) if m != hub and m != d and graph.has_edge(m, hub)), cap
        ),
        "instrument": _take(
            (
                z
                for d in causes
                for z in graph.predecessors(d)
                if z != hub and z not in cause_set and not graph.has_edge(z, hub)
            ),
            cap,
        ),
        "decoy": _take(
            (dd for x in confounders for dd in graph.successors(x) if dd != hub and not graph.has_edge(dd, hub)), cap
        ),
        "upstream": _take(
            (u for x in confounders for u in graph.predecessors(x) if u != hub and u not in cause_set), cap
        ),
        "collider": _take((e for e, shared in shared_effects.items() if shared >= 2), cap),
    }


#: Concepts too generic to steer a naming ("problems", "changes") or malformed by CauseNet's extraction
#: ("rapid", "metal_to_expand"). Filtered out of the pool: they crowd out usable vocabulary and invite
#: vacuous column names.
_GENERIC_CONCEPTS = frozenset(
    {
        "condition", "conditions", "problem", "problems", "symptoms", "change", "changes", "issues",
        "damage", "harm", "death", "deaths", "failure", "effects", "loss", "losses", "disorder",
        "illness", "illnesses", "disease", "diseases", "health_problems", "stresses", "reaction",
        "exposure", "activity", "rapid", "increased", "excessive",
    }
)  # fmt: skip
_MALFORMED = re.compile(r"_(to|of|or|the|a)_")


def concept_pool(menus: CauseNetMenus, limit: int = 24) -> tuple[str, ...]:
    """Flatten the gathered menus into ONE deduplicated, quality-filtered pool of real concepts.

    The honest shape of what CauseNet can contribute. Its structural queries are *not* disjoint - measured
    on the real dump, ``instrument`` and ``upstream`` are frequently identical (Jaccard 1.00) and
    ``driver`` and ``confounder`` overlap at 0.87, because a real concept has no single causal role
    (``obesity`` is a cause of one thing and an effect of another). Presenting them to a generator as seven
    labelled role menus asserted a distinction the data does not contain, listed the same concepts up to
    seven times (303 entries for 160 distinct concepts), and consumed 45% of the prompt to do it.

    So the queries are used for what they genuinely are - a way to *gather* related vocabulary across
    several hops - and then flattened. Role accuracy comes from the generator's own world knowledge, the
    only place it actually exists.
    """
    pool: list[str] = []
    for category in MENU_CATEGORIES:
        for concept in menus.menus.get(category, ()):
            if (
                concept not in pool
                and 3 < len(concept) <= 28
                and concept.count("_") <= 2
                and concept not in _GENERIC_CONCEPTS
                and not _MALFORMED.search(concept)
            ):
                pool.append(concept)
    return tuple(pool[:limit])


def sample_menus(
    dump_path: str,
    rng: np.random.Generator,
    *,
    min_outcome_causes: int = 8,
    base_size: int = 15,
    cap: int = 64,
    min_pool: int = 24,
    max_tries: int = 400,
) -> CauseNetMenus:
    """Pick a random outcome hub whose neighborhood yields at least ``min_pool`` usable concepts.

    Hubs with at least ``min_outcome_causes`` causes are visited in a seeded random order and the first
    with a rich enough pool is returned. Gating on the *pool* rather than on per-role counts is the point:
    the pool is what the generator actually sees, and per-role coverage would reject hubs over a role
    distinction the data does not carry. Measured over 600 random hubs, the median pool holds 128 distinct
    concepts and 97.7% clear a floor of 24, so this rarely costs more than a try or two.
    """
    graph = load_causenet(dump_path)
    hubs = [n for n in graph if graph.in_degree(n) >= min_outcome_causes]
    if not hubs:
        raise ValueError(f"no outcome hub with >= {min_outcome_causes} causes in this dump")
    for index in rng.permutation(len(hubs))[:max_tries]:
        hub = hubs[int(index)]
        candidate = CauseNetMenus(hub, menus_for_hub(graph, hub, base_size=base_size, cap=cap))
        if len(concept_pool(candidate, limit=min_pool)) >= min_pool:
            logger.debug("grounded on hub %r (in-degree %d)", hub, graph.in_degree(hub))
            return candidate
    raise ValueError(
        f"no CauseNet hub within {min(len(hubs), max_tries)} tries yielded {min_pool} usable concepts; "
        f"use a larger dump or lower min_pool"
    )


def format_menus(menus: CauseNetMenus, limit: int = 24) -> str:
    """Render the grounding as one line: the outcome concept plus a deduplicated related-concept pool."""
    return f'the outcome concept is "{menus.outcome}"; related real concepts: {", ".join(concept_pool(menus, limit))}'
