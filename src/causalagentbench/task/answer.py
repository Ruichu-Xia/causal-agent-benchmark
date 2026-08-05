"""The agent's answer contract.

A minimal, validated deliverable: per candidate driver an adjustment set and an estimate of its total
causal effect. The agent emits this as JSON inside an ``<answer>...</answer>`` block; ``parse_answer``
extracts and validates its JSON shape. ``validate_answer_contract`` then checks the public instance
contract (every displayed candidate exactly once, an effect estimate wherever the agent did not abstain)
without consulting any hidden causal ground truth.

The ranking the task asks for is derived from those estimates rather than stated separately - see
``Answer``.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, ValidationError

_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class AnswerParseError(ValueError):
    """Raised when no valid ``<answer>`` block can be parsed into an ``Answer``."""


class DriverVerdict(BaseModel):
    """The agent's adjustment set and effect estimate for one candidate driver, or an abstention.

    ``abstain=True`` means the agent judges the driver's effect not identifiable from the data - neither
    ``adjustment_set`` nor ``effect`` is then graded on content, only whether abstaining was the right call
    (``scoring/score.py``).

    ``effect`` is the estimated total causal effect, and it is what makes the deliverable gradeable beyond a
    shortcut. Measured on the generator: blind policies (marginal correlation, adjust-for-everything) rank
    the drivers at tau 0.68-0.78 but estimate them ~25% of a typical effect away, while correct adjustment
    recovers the truth exactly. An order alone cannot separate those; a number can.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    driver: str
    adjustment_set: tuple[str, ...]
    effect: float | None = None
    abstain: bool = False


class Answer(BaseModel):
    """The agent's full deliverable for one instance.

    No separate ranking field: the ranking is *derived* from the per-driver ``effect`` estimates
    (``scoring/score.py::_ranking_tau``). A separately stated order could be produced by ranking on raw
    correlation - the measured tau=+0.627 shortcut - independently of the estimates, so scoring it as its own
    deliverable left the ranking metric reachable without causal reasoning. Deriving it means ranking credit
    can only be earned through the estimates, which no blind policy gets right.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdicts: tuple[DriverVerdict, ...]


def parse_answer(text: str) -> Answer:
    """Extract and validate the JSON in the last ``<answer>...</answer>`` block of ``text``.

    The last block wins, so trailing final answers override earlier scratch work. Raises
    ``AnswerParseError`` on a missing block or invalid JSON/schema.
    """
    matches = _ANSWER_BLOCK.findall(text)
    if not matches:
        raise AnswerParseError("no <answer>...</answer> block found")
    try:
        return Answer.model_validate_json(matches[-1].strip())
    except ValidationError as exc:
        raise AnswerParseError(f"invalid answer payload: {exc}") from exc


def validate_answer_contract(
    answer: Answer,
    candidate_drivers: tuple[str, ...],
    *,
    columns: tuple[str, ...] | None = None,
    outcome: str | None = None,
) -> Answer:
    """Validate an answer against public candidate labels, never hidden graph roles or effects."""
    expected = set(candidate_drivers)
    verdict_labels = [verdict.driver for verdict in answer.verdicts]
    if len(set(verdict_labels)) != len(verdict_labels):
        raise AnswerParseError("verdicts must not repeat a candidate driver")
    if len(verdict_labels) != len(candidate_drivers) or set(verdict_labels) != expected:
        raise AnswerParseError("verdicts must contain every candidate driver exactly once")
    for verdict in answer.verdicts:
        # The ranking is derived from these, so a committed driver without one has no position to hold.
        if not verdict.abstain and verdict.effect is None:
            raise AnswerParseError("a driver that is not abstained on must carry an effect estimate")
    if columns is not None:
        allowed_columns = set(columns)
        for verdict in answer.verdicts:
            adjustment_set = set(verdict.adjustment_set)
            if len(adjustment_set) != len(verdict.adjustment_set):
                raise AnswerParseError("adjustment_set must not repeat a column")
            if not adjustment_set <= allowed_columns:
                raise AnswerParseError("adjustment_set contains a column not present in the dataset")
            if verdict.driver in adjustment_set:
                raise AnswerParseError("adjustment_set cannot contain its driver")
            if outcome is not None and outcome in adjustment_set:
                raise AnswerParseError("adjustment_set cannot contain the outcome")
    return answer


def parse_answer_for_instance(
    text: str,
    candidate_drivers: tuple[str, ...],
    *,
    columns: tuple[str, ...] | None = None,
    outcome: str | None = None,
) -> Answer:
    """Parse an answer and enforce its public instance-level response contract."""
    return validate_answer_contract(parse_answer(text), candidate_drivers, columns=columns, outcome=outcome)
