"""Score an agent's answer against the true SCM.

The task is: rank the candidate drivers by causal effect, and give a valid adjustment set for each. Both
metrics are computed on the hidden SCM via ``key.label_to_node``, so the score is invariant to how the
columns were named - the whole point of the naming axis.

* ``ranking_tau`` - Kendall rank agreement between the agent's order and the true ``|effect|`` order (the
  headline: a reasoner stays flat across naming alignment, a reciter degrades).
* ``adjustment_accuracy`` - fraction of drivers whose proposed set is a valid backdoor adjustment set,
  judged by the backdoor criterion against the true graph (so any valid set counts, not just the minimal).

``alignment`` carries the naming-alignment level for stratified analysis. A parse failure or a malformed
ranking scores zero rather than raising. New metrics are added by extending ``AnswerScore`` and
``score_answer``; nothing else needs to change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import networkx as nx

from causalagentbench.scoring.metrics import kendall_tau
from causalagentbench.task.answer import Answer, AnswerParseError, parse_answer
from causalagentbench.task.instance import InstanceKey
from causalagentbench.types import Scaffold


@dataclass(frozen=True)
class AnswerScore:
    """The score of one answer. Metrics are on [0, 1] except ``ranking_tau`` which is on [-1, 1]."""

    parse_ok: bool
    ranking_tau: float  # Kendall tau vs the true order; 1.0 perfect, 0.0 malformed/unparsed
    adjustment_accuracy: float  # fraction of drivers with a valid backdoor set
    adjustment_valid: Mapping[str, bool]  # per-driver (node name) validity
    alignment: float | None  # naming-alignment level, for stratification


def _valid_backdoor(scaffold: Scaffold, driver: str, adjust: set[str]) -> bool:
    """True if ``adjust`` is a valid backdoor adjustment set for ``driver -> outcome`` in the true graph."""
    graph = scaffold.digraph()
    outcome = scaffold.outcome
    if driver in adjust or outcome in adjust:
        return False
    if adjust & (nx.descendants(graph, driver) | {driver}):  # no descendant of the treatment (e.g. a mediator)
        return False
    backdoor = graph.copy()
    backdoor.remove_edges_from(list(graph.out_edges(driver)))
    return bool(nx.is_d_separator(backdoor, {driver}, {outcome}, adjust))


def _ranking_tau(answer: Answer, key: InstanceKey) -> float:
    """Kendall agreement of the agent's ranking with the true order; 0.0 unless it is a clean permutation."""
    drivers = key.scm.scaffold.drivers
    agent_nodes = [key.label_to_node.get(label) for label in answer.ranking]
    if len(agent_nodes) != len(drivers) or set(agent_nodes) != set(drivers):
        return 0.0  # not a well-formed ranking of exactly the candidate drivers
    true_position = {node: i for i, node in enumerate(key.scm.true_ranking())}
    agent_position = {node: i for i, node in enumerate(agent_nodes)}
    return kendall_tau([true_position[d] for d in drivers], [agent_position[d] for d in drivers])


def _adjustment_valid(answer: Answer, key: InstanceKey, driver: str) -> bool:
    """True if the agent's adjustment set for ``driver`` is a valid backdoor set (False if absent/unknown)."""
    label = key.node_to_label[driver]
    verdict = next((v for v in answer.verdicts if v.driver == label), None)
    if verdict is None:
        return False
    mapped = [key.label_to_node.get(z) for z in verdict.adjustment_set]
    if any(node is None for node in mapped):
        return False  # references a column that does not exist
    return _valid_backdoor(key.scm.scaffold, driver, {node for node in mapped if node is not None})


def score_answer(answer: Answer, key: InstanceKey) -> AnswerScore:
    """Score a parsed ``answer`` against the true SCM in ``key``."""
    valid = {driver: _adjustment_valid(answer, key, driver) for driver in key.scm.scaffold.drivers}
    return AnswerScore(
        parse_ok=True,
        ranking_tau=_ranking_tau(answer, key),
        adjustment_accuracy=sum(valid.values()) / len(valid),
        adjustment_valid=valid,
        alignment=key.provenance.alignment,
    )


def score_output(text: str, key: InstanceKey) -> AnswerScore:
    """Parse ``text`` and score it; an unparseable answer scores zero rather than raising."""
    try:
        answer = parse_answer(text)
    except AnswerParseError:
        drivers = key.scm.scaffold.drivers
        return AnswerScore(False, 0.0, 0.0, dict.fromkeys(drivers, False), key.provenance.alignment)
    return score_answer(answer, key)
