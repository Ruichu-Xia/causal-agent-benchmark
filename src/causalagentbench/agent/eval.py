"""Batch the agent loop over many instances and aggregate scores by naming alignment.

The headline metric this benchmark measures is score *as a function of* naming alignment: a reasoner
should stay flat across the grid, a reciter should degrade as names go neutral or misaligned.
``run_evaluation`` produces one ``AnswerScore`` per instance (each already carries its own
``alignment``); ``aggregate_by_alignment`` groups and summarizes them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from causalagentbench.agent.loop import MAX_TURNS, ToolCallingClient, TurnTrace, run_agent
from causalagentbench.scoring.score import AnswerScore, score_output
from causalagentbench.task.instance import Instance, InstanceKey

logger = logging.getLogger(__name__)


def run_evaluation(
    instances: Iterable[tuple[Instance, InstanceKey]],
    client: ToolCallingClient,
    *,
    max_turns: int = MAX_TURNS,
    on_instance: Callable[[int, AnswerScore], None] | None = None,
    on_turn: Callable[[int, TurnTrace], None] | None = None,
) -> list[AnswerScore]:
    """Run the agent on every instance and score each reply; one ``AnswerScore`` per instance.

    A single instance whose agent run raises (a client/provider error the tool loop itself can't turn into
    text) is logged and scored as a parse failure rather than aborting the whole batch - one bad instance
    should never cost every other instance's already-computed score.

    ``on_instance``, if given, is called with (0-based index, that instance's score) right after each one
    finishes - optional because a batch over many instances against a slow model (an extended-thinking
    model needing a dozen-plus turns each) can otherwise run for a long time with zero visibility until the
    entire batch returns; the default (``None``) keeps this function silent, as before.

    ``on_turn``, if given, is called with (0-based instance index, ``TurnTrace``) for every turn of every
    instance - the same instance-index tagging as ``on_instance``, one level more granular. Passed straight
    through to each ``run_agent`` call.
    """
    scores = []
    for i, (instance, key) in enumerate(instances):
        instance_on_turn = (lambda trace, i=i: on_turn(i, trace)) if on_turn is not None else None
        try:
            reply = run_agent(instance, client, max_turns=max_turns, on_turn=instance_on_turn)
        except Exception:
            logger.exception("agent run failed; scoring this instance as a parse failure")
            reply = ""
        score = score_output(reply, key)
        scores.append(score)
        if on_instance is not None:
            on_instance(i, score)
    return scores


@dataclass(frozen=True)
class AlignmentSummary:
    """Mean scores for one naming-alignment level (``None`` = name-blind)."""

    alignment: float | None
    n: int
    mean_ranking_tau: float
    mean_adjustment_accuracy: float
    mean_abstention_accuracy: float
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
            mean_abstention_accuracy=sum(s.abstention_accuracy for s in group) / len(group),
            parse_rate=sum(s.parse_ok for s in group) / len(group),
        )
        for alignment, group in groups.items()
    ]
    return sorted(summaries, key=lambda s: (s.alignment is None, s.alignment))
