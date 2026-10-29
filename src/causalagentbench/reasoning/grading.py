"""Grade a reply against a task's canonical answer: the last "Answer:" line decides, anything else scores 0."""

from __future__ import annotations

import re

# "Answer:" (ASCII or full-width colon, optionally in markdown emphasis) and its value on the same line.
# Whitespace must not cross a line break: "Final answer:\n\nAnswer: Yes" would otherwise read as "Answer: Yes".
_ANSWER = re.compile("answer[*_ \t]*[:\uff1a][*_ \t]*(\\S.*)", re.IGNORECASE)
_MINUS = "\u2212"
_NUMBER = re.compile(f"[-{_MINUS}]?\\d*\\.?\\d+")
NUMBER_TOLERANCE = 0.011
"""Answers are asked to two decimals; this accepts the truth's rounding either way."""


def final_answer(reply: str) -> str | None:
    """Text after the last "Answer:" in the reply, stripped of markdown emphasis and math markup.

    The instruction's ``<answer>`` placeholder tags are stripped too: models often echo them around the answer.
    """
    matches = _ANSWER.findall(reply)
    if not matches:
        return None
    return re.sub(r"</?answer>|[*_`$\\]|boxed|text", "", matches[-1], flags=re.IGNORECASE).strip(" .{}()").lower()


def parse(kind: str, text: str) -> str | None:
    """The canonical form of an answer of format ``kind``, or None if it is not one."""
    if kind == "yesno":
        word = re.match(r"(yes|no)\b", text)
        return word.group(1) if word else None
    if kind == "number":
        number = _NUMBER.search(text)
        return f"{float(number.group().replace(_MINUS, '-')):.4f}" if number else None
    if kind == "letters":
        if re.match(r"none\b", text):
            return "none"
        letters = re.findall(r"\b([a-d])\b", text)
        return ",".join(sorted(set(letters))).upper() if letters else None
    if kind == "choice":
        word = re.match(r"(1|2|both|neither)\b", text)
        return word.group(1) if word else None
    raise ValueError(f"unknown answer format {kind!r}")


def grade(kind: str, answer: str, reply: str) -> float:
    """1.0 if the reply's final answer matches ``answer`` (numbers within ``NUMBER_TOLERANCE``), else 0.0."""
    text = final_answer(reply)
    given = parse(kind, text) if text is not None else None
    if given is None:
        return 0.0
    if kind == "number":
        return float(abs(float(given) - float(answer)) <= NUMBER_TOLERANCE)
    return float(given == answer)
