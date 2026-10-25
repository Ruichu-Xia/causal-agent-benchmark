"""Evaluate one task with a baseline or a tool-calling model, producing a scored episode record."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from causalagentbench.agent.loop import TurnTrace, TurnTranscript, run_agent
from causalagentbench.scoring.baselines import baseline_answer
from causalagentbench.scoring.score import SCORING_VERSION, AnswerScore, score_output
from causalagentbench.task.instance import Instance, InstanceKey


def episode_record(index: int, model: str, answer: str, key: InstanceKey, **extra: Any) -> dict[str, Any]:
    """Score ``answer`` and return the JSON-safe episode record, with one entry per candidate."""
    score: AnswerScore = score_output(answer, key)
    candidates = [
        {
            "node": node,
            "label": key.node_to_label[node],
            "estimate": score.estimate[node],
            "truth": score.truth[node],
            "error": abs(score.estimate[node] - score.truth[node]),
            "capped_error": score.capped_error(node),
            "name_critical": score.name_critical[node],
            "adjustment_valid": score.adjustment_valid[node],
        }
        for node in key.scm.scaffold.drivers
    ]
    return {
        "type": "episode",
        "index": index,
        "model": model,
        "scoring_version": SCORING_VERSION,
        "parse_ok": score.parse_ok,
        "task_error": score.task_error,
        "reward": score.reward,
        "candidates": candidates,
        "answer": answer,
        **extra,
    }


def evaluate_task(
    index: int,
    task: tuple[Instance, InstanceKey],
    *,
    model: str | None,
    baseline: str | None,
    max_tool_calls: int,
    max_turns: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run one task; return its episode (or error) record and, for models, its transcript record.

    Provider failures are recorded as ``"error"`` rows, never scored, so a resumed run retries them.
    """
    instance, key = task
    start = time.monotonic()
    turns: list[TurnTrace] = []
    transcripts: list[TurnTranscript] = []
    name = model or baseline
    assert name is not None
    try:
        if baseline is not None:
            answer = "<answer>" + baseline_answer(baseline, instance, key).model_dump_json() + "</answer>"
        else:
            from causalagentbench.agent.clients import LiteLLMClient

            assert model is not None
            answer = run_agent(
                instance,
                LiteLLMClient(model=model),
                max_tool_calls=max_tool_calls,
                max_turns=max_turns,
                on_turn=turns.append,
                on_transcript=transcripts.append,
            )
    except Exception as exc:
        record = {"type": "error", "index": index, "model": name, "error": f"{type(exc).__name__}: {exc}"}
        return record, _transcript(index, name, transcripts, None)
    costs = [t.cost_usd for t in turns]
    record = episode_record(
        index,
        name,
        answer,
        key,
        seconds=time.monotonic() - start,
        turns=len(turns),
        tool_calls=sum(len(t.tool_calls) for t in turns),
        prompt_tokens=sum(t.prompt_tokens or 0 for t in turns),
        completion_tokens=sum(t.completion_tokens or 0 for t in turns),
        estimated_cost_usd=sum(costs) if all(c is not None for c in costs) else None,
        served_models=sorted({t.model for t in turns if t.model}),
    )
    return record, _transcript(index, name, transcripts, answer if baseline is None else None)


def _transcript(index: int, model: str, transcripts: list[TurnTranscript], final: str | None) -> dict[str, Any] | None:
    """The full conversation: the last request's message history plus the final reply."""
    if not transcripts:
        return None
    last = transcripts[-1]
    messages = list(last.messages)
    if last.content is not None or last.tool_calls:
        messages.append(
            {
                "role": "assistant",
                "content": last.content,
                "tool_calls": [dataclasses.asdict(call) for call in last.tool_calls],
            }
        )
    return {
        "type": "transcript",
        "index": index,
        "model": model,
        "messages": messages,
        "final_answer": final,
        "error": last.error,
    }
