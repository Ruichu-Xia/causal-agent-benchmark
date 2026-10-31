"""QRData (Liu et al., 2024) as environment-v3 tasks: questions about real data files, answered with the python tool.

Source: ``benchmark/QRData.json`` and ``benchmark/data.zip`` of github.com/xxxiaol/QRData (CC BY-NC 4.0, kept
local). The causal subset is every question whose keywords include "Causality" (269); the rest is the
statistics subset (142).

``grade`` applies the rule of the benchmark's ``eval.py`` to the text after the reply's last "Answer:" line,
as normalized by ``final_answer`` (lowercased; markdown emphasis, math markup and surrounding " .{}()"
removed): a multiple-choice prediction is right if it starts with the gold choice; a numerical one if its
first number (a trailing % divides by 100) lies strictly between 0.97 gold and 1.03 gold (the smaller and
larger of the two, as in ``eval.py``, so negative values work). ``strict_choice`` is a stricter reading for
analysis: the gold choice must not continue into a longer word, so "Both" is not "B" and "not" is not "no".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from causalagentbench.dataenv.wording import SLOTS
from causalagentbench.reasoning.grading import final_answer

ERROR_SCALE = 0.03
"""Relative tolerance of a numerical answer, as in QRData's ``eval.py``."""


def _join(words: list[str]) -> str:
    """'a', 'a or b', 'a, b or c'."""
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]


def prompt(item: dict[str, Any]) -> str:
    """The data description, where the files are, the question, the tools, and the answer instruction."""
    files = item["data_files"]
    if len(files) == 1:
        where = f"The file {files[0]} is loaded in the python tool as the pandas DataFrame `df`."
    else:
        where = (
            f"The files {', '.join(files)} are loaded in the python tool as pandas DataFrames in the dict `dfs`, "
            f"keyed by file name; `df` is {files[0]}."
        )
    where += " The files themselves are in the working directory."
    if item["meta_data"]["question_type"] == "multiple_choice":
        choices = item["meta_data"]["multiple_choices"]
        spec = _join([choice.upper() if len(choice) == 1 else choice for choice in choices])
    else:
        spec = "a number"
    tools = SLOTS["tools"][0][0]
    instruction = f'When you are done, end your reply with a final line "Answer: <answer>", where <answer> is {spec}.'
    return f"{item['data_description']}\n\n{where}\n\n{item['question']}\n\n{tools}\n\n{instruction}"


def records(benchmark: Path) -> list[dict[str, Any]]:
    """Every QRData question as a task record, in file order, with the subset it belongs to."""
    questions = json.loads((benchmark / "QRData.json").read_text(encoding="utf-8"))
    out = []
    for i, item in enumerate(questions):
        keywords = item["meta_data"]["keywords"]
        subset = "qrdata_causal" if "Causality" in keywords else "qrdata_statistics"
        numerical = item["meta_data"]["question_type"] == "numerical"
        out.append(
            {
                "id": f"qrdata-{i}",
                "split": subset,
                "family": subset,
                "kind": "qrdata_number" if numerical else "qrdata_choice",
                "prompt": prompt(item),
                "answer": item["answer"],
                "tolerance": None,
                "meta": {"keywords": keywords, "reference": item["meta_data"]["reference"]},
                "files": item["data_files"],
            }
        )
    return out


def _first_number(text: str) -> str | None:
    """The first number in the text, with a trailing %, as in QRData's ``extract_first_number``."""
    match = re.search(r"-?\d+\.?\d*%?", text)
    return match.group() if match else None


def _value(number: str) -> float:
    """A number string as a float; a trailing % divides by 100."""
    return float(number[:-1]) / 100 if number.endswith("%") else float(number)


def strict_choice(answer: str, reply: str) -> float:
    """1.0 if the final answer is the gold choice as a whole token ("b", "b. x causes y"), else 0.0."""
    text = final_answer(reply)
    return float(text is not None and re.match(rf"{re.escape(answer.lower())}(?![a-z0-9])", text) is not None)


def grade(kind: str, answer: str, reply: str) -> float:
    """QRData's ``calc_acc`` rule for one reply's final answer: 1.0 if right, else 0.0."""
    text = final_answer(reply)
    if text is None:
        return 0.0
    gold = answer.lower()
    if kind == "qrdata_choice":
        return float(text[: len(gold)] == gold)
    number = _first_number(text)
    if number is None:
        return 0.0
    bounds = sorted((_value(gold) * (1 - ERROR_SCALE), _value(gold) * (1 + ERROR_SCALE)))
    return float(bounds[0] < _value(number) < bounds[1])
