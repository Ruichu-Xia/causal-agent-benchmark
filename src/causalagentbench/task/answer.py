"""Validated answers: one adjustment set and finite signed effect per candidate."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, ValidationError

_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class AnswerParseError(ValueError):
    """Raised when no valid ``<answer>`` block can be parsed into an ``Answer``."""


class DriverVerdict(BaseModel):
    """An adjustment set and signed total-effect estimate for one candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    driver: str
    adjustment_set: tuple[str, ...] = ()
    effect: float | None = None
    identifiable_from_data: bool | None = None
    """Optional legacy diagnostic, not requested or rewarded in the default task."""


class Answer(BaseModel):
    """Signed total-effect estimates and the adjustment sets used to obtain them."""

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
        # Every candidate contributes to the estimation objective.
        if verdict.effect is None:
            raise AnswerParseError("every driver must carry an effect estimate")
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
