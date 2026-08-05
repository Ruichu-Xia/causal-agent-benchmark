"""Batch the agent loop over many instances and aggregate scores by naming alignment.

The headline metric this benchmark measures is score *as a function of* naming alignment: a reasoner
should stay flat across the grid, a reciter should degrade as names go neutral or misaligned.
``run_evaluation`` produces one ``AnswerScore`` per instance (each already carries its own
``alignment``); ``aggregate_by_alignment`` groups and summarizes them.

``aggregate_by_driver_cell`` reports the same run one level down, at *driver* grain, crossing naming
alignment with whether a driver's effect is identifiable from data at all - an instance-level mean
otherwise conflates failing at something the data supports with failing at something it provably does not.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from causalagentbench.agent.loop import MAX_TURNS, ToolCallingClient, TurnTrace, TurnTranscript, run_agent
from causalagentbench.scoring.score import AnswerScore, RankingPairs, score_output
from causalagentbench.task.instance import Instance, InstanceKey

logger = logging.getLogger(__name__)


def run_evaluation(
    instances: Iterable[tuple[Instance, InstanceKey]],
    client: ToolCallingClient,
    *,
    max_turns: int = MAX_TURNS,
    on_instance: Callable[[int, AnswerScore], None] | None = None,
    on_turn: Callable[[int, TurnTrace], None] | None = None,
    on_transcript: Callable[[int, TurnTranscript], None] | None = None,
    on_instance_error: Callable[[int, Exception], None] | None = None,
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

    ``on_transcript`` similarly receives an instance index and full ``TurnTranscript`` for each model call.
    It is opt-in because transcripts include prompts, tool results, and unredacted model output.

    ``on_instance_error`` distinguishes a provider/client failure from an unparseable model answer. The
    returned list retains a zero score for compatibility with existing callers; evaluation runners should use
    this callback to record and exclude transport failures from benchmark aggregates.
    """
    scores = []
    for i, (instance, key) in enumerate(instances):
        instance_on_turn = (lambda trace, i=i: on_turn(i, trace)) if on_turn is not None else None
        instance_on_transcript = (
            (lambda transcript, i=i: on_transcript(i, transcript)) if on_transcript is not None else None
        )
        try:
            reply = run_agent(
                instance,
                client,
                max_turns=max_turns,
                on_turn=instance_on_turn,
                on_transcript=instance_on_transcript,
            )
        except Exception as exc:
            logger.exception("agent run failed; scoring this instance as a parse failure")
            if on_instance_error is not None:
                on_instance_error(i, exc)
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


@dataclass(frozen=True)
class DriverCellSummary:
    """Pooled per-driver scores for one identifiability x naming-alignment cell.

    ``n`` counts *driver observations*, not instances - the unit change from ``AlignmentSummary``, and the
    point of the breakdown: one instance contributes as many rows here as it has candidate drivers.
    Decoys are included, because ``adjustment_accuracy``/``abstention_accuracy`` already grade every
    candidate driver; excluding them would make the two tables incomparable.

    Averaging is pooled (micro), so per-cell means will *not* exactly reconcile with ``AlignmentSummary``'s
    mean-of-instance-means at the same alignment - instances differ in driver count, and here every driver
    weighs the same. ``n_graded`` and ``n_ranking_pairs`` are the denominators behind the two means, kept
    visible because both can shrink silently: a cell's ``mean_adjustment_accuracy`` rises if the agent
    simply abstained on its hard drivers, and ``ranking_tau`` says little over a handful of pairs.
    ``parse_rate`` is deliberately absent - parse failure is per-instance and ``AlignmentSummary`` reports
    it; note that a parse failure contributes no ranking pairs at all, so it is invisible in ``ranking_tau``
    here while still pulling the instance-level mean down.
    """

    identifiable: bool
    alignment: float | None
    n: int  # driver observations in this cell, NOT instances
    n_graded: int  # of those, the ones graded on adjustment (the rest were abstained on)
    mean_adjustment_accuracy: float
    mean_abstention_accuracy: float
    n_ranking_pairs: int  # non-tied comparisons involving a driver in this cell
    ranking_tau: float  # pooled (concordant - discordant) / (concordant + discordant) over those pairs


def _driver_cell_summary(
    cell: tuple[bool, float | None], group: list[tuple[bool | None, bool, RankingPairs]]
) -> DriverCellSummary:
    """Summarize one cell's ``(adjustment_valid, abstention_correct, ranking_pairs)`` observations."""
    identifiable, alignment = cell
    graded = [valid for valid, _, _ in group if valid is not None]
    concordant = sum(pairs.concordant for _, _, pairs in group)
    discordant = sum(pairs.discordant for _, _, pairs in group)
    decided = concordant + discordant
    return DriverCellSummary(
        identifiable=identifiable,
        alignment=alignment,
        n=len(group),
        n_graded=len(graded),
        # Nothing to grade is vacuously accurate, the convention score_answer and kendall_tau both already
        # use for an empty comparison - not a zero, which would read as a failure the agent never had.
        mean_adjustment_accuracy=sum(graded) / len(graded) if graded else 1.0,
        mean_abstention_accuracy=sum(correct for _, correct, _ in group) / len(group),
        n_ranking_pairs=decided,
        ranking_tau=(concordant - discordant) / decided if decided else 1.0,
    )


def aggregate_by_driver_cell(scores: list[AnswerScore]) -> list[DriverCellSummary]:
    """Group every driver observation by (identifiability, alignment); identifiable first, name-blind last."""
    groups: dict[tuple[bool, float | None], list[tuple[bool | None, bool, RankingPairs]]] = {}
    for score in scores:
        for driver, identifiable in score.identifiable.items():
            groups.setdefault((identifiable, score.alignment), []).append(
                (score.adjustment_valid[driver], score.abstention_correct[driver], score.ranking_pairs[driver])
            )

    summaries = [_driver_cell_summary(cell, group) for cell, group in groups.items()]
    # Identifiability leads so the table reads as two blocks, each ordered like aggregate_by_alignment's.
    # The `alignment is None` component is still needed for the same reason it is there: it splits None
    # from float *before* any `<` is applied, so a None alignment is only ever compared for equality.
    return sorted(summaries, key=lambda s: (not s.identifiable, s.alignment is None, s.alignment))
