"""Batch the agent loop over many instances and aggregate scores by naming alignment.

The headline metric this benchmark measures is score *as a function of* naming alignment: a reasoner
should stay flat across the grid, a reciter should degrade as names go neutral or misaligned.
``run_evaluation`` produces one ``AnswerScore`` per instance (each already carries its own
``alignment``); ``aggregate_by_alignment`` groups and summarizes them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from causalagentbench.agent.loop import MAX_TURNS, ToolCallingClient, run_agent
from causalagentbench.scoring.score import AnswerScore, score_output
from causalagentbench.task.instance import Instance, InstanceKey

logger = logging.getLogger(__name__)


def run_evaluation(
    instances: Iterable[tuple[Instance, InstanceKey]], client: ToolCallingClient, *, max_turns: int = MAX_TURNS
) -> list[AnswerScore]:
    """Run the agent on every instance and score each reply; one ``AnswerScore`` per instance.

    A single instance whose agent run raises (a client/provider error the tool loop itself can't turn into
    text) is logged and scored as a parse failure rather than aborting the whole batch - one bad instance
    should never cost every other instance's already-computed score.
    """
    scores = []
    for instance, key in instances:
        try:
            reply = run_agent(instance, client, max_turns=max_turns)
        except Exception:
            logger.exception("agent run failed; scoring this instance as a parse failure")
            reply = ""
        scores.append(score_output(reply, key))
    return scores


@dataclass(frozen=True)
class AlignmentSummary:
    """Mean scores for one naming-alignment level (``None`` = name-blind)."""

    alignment: float | None
    n: int
    mean_ranking_tau: float
    mean_adjustment_accuracy: float
    parse_rate: float


def aggregate_by_alignment(scores: list[AnswerScore]) -> list[AlignmentSummary]:
    """Group scores by ``alignment`` and summarize each group; sorted with name-blind (``None``) last."""
    groups: dict[float | None, list[AnswerScore]] = {}
    for score in scores:
        groups.setdefault(score.alignment, []).append(score)

    summaries = [
        AlignmentSummary(
            alignment=alignment,
            n=len(group),
            mean_ranking_tau=sum(s.ranking_tau for s in group) / len(group),
            mean_adjustment_accuracy=sum(s.adjustment_accuracy for s in group) / len(group),
            parse_rate=sum(s.parse_ok for s in group) / len(group),
        )
        for alignment, group in groups.items()
    ]
    return sorted(summaries, key=lambda s: (s.alignment is None, s.alignment))
