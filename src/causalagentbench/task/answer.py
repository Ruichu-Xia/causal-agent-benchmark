"""The agent's answer contract.

A minimal, validated deliverable: per candidate driver an adjustment set, plus an overall ranking of the
drivers by causal effect. The agent emits this as JSON inside an ``<answer>...</answer>`` block;
``parse_answer`` extracts and validates its JSON shape. ``validate_answer_contract`` then checks the
public instance contract (every displayed candidate exactly once, and a ranking over exactly those labels)
without consulting any hidden causal ground truth.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, ValidationError

_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class AnswerParseError(ValueError):
    """Raised when no valid ``<answer>`` block can be parsed into an ``Answer``."""


class DriverVerdict(BaseModel):
    """The agent's proposed adjustment set for one candidate driver, or an abstention.

    ``abstain=True`` means the agent judges the driver's role genuinely undetermined by the data (and,
    where naming is shown, not resolved by the name either) - ``adjustment_set`` is then not graded on
    content, only whether abstaining was the right call (``scoring/score.py``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    driver: str
    adjustment_set: tuple[str, ...]
    abstain: bool = False


class Answer(BaseModel):
    """The agent's full deliverable for one instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdicts: tuple[DriverVerdict, ...]
    ranking: tuple[str, ...]


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
    if len(set(answer.ranking)) != len(answer.ranking):
        raise AnswerParseError("ranking must not repeat a candidate driver")
    if len(answer.ranking) != len(candidate_drivers) or set(answer.ranking) != expected:
        raise AnswerParseError("ranking must contain every candidate driver exactly once")
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
