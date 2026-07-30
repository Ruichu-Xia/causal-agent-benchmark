"""The agent's answer contract.

A minimal, validated deliverable: per candidate driver an adjustment set, plus an overall ranking of the
drivers by causal effect. The agent emits this as JSON inside an ``<answer>...</answer>`` block;
``parse_answer`` extracts and validates it. Validation is shape-only - instance-consistency and
correctness are the (later) scorer's job.
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
