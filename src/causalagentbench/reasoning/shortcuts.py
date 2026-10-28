"""Plausible wrong rules a model could learn instead of the real criteria; sampling balances labels against them.

Each rule is right on many random cases, so a policy rewarded on randomly drawn cases could earn reward
by following it. The generator draws the label and the rule's agreement with it independently, which
holds the rule at chance (see ``docs/environment_v2.md``).
"""

from __future__ import annotations

import networkx as nx


def undirected_separated(graph: nx.DiGraph, x: int, y: int, z: set[int]) -> bool:
    """Independence read off the skeleton: every path between ``x`` and ``y`` passes through ``z``.

    It ignores edge directions, so it is wrong exactly where colliders matter.
    """
    skeleton = graph.to_undirected()
    skeleton.remove_nodes_from(z)
    return not nx.has_path(skeleton, x, y)


def confounder_rule(graph: nx.DiGraph, x: int, y: int, z: set[int]) -> bool:
    """The textbook rule: adjust for every common cause of ``x`` and ``y`` and for no descendant of ``x``.

    It is wrong when a back door is blocked by a non-confounder, under M-bias, and for harmless
    descendants of ``x`` that the complete adjustment criterion allows.
    """
    common = nx.ancestors(graph, x) & nx.ancestors(graph, y)
    return common <= z and not z & nx.descendants(graph, x)
