"""Load the CauseNet directed relation graph from its public JSONL/bz2 format."""

from __future__ import annotations

import bz2
import functools
import json
import logging

import networkx as nx

logger = logging.getLogger(__name__)


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
