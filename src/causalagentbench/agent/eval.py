"""Evaluate a single task with a scripted baseline or a tool-calling model."""

from __future__ import annotations

import time
from typing import Any

from causalagentbench.agent.loop import TurnTrace, run_agent
from causalagentbench.scoring.metrics import regression_coefficient
from causalagentbench.scoring.score import SCORING_VERSION, score_output
from causalagentbench.task.answer import Answer, DriverVerdict
from causalagentbench.task.instance import Instance, InstanceKey

REPORTING_VERSION = "normalized-error-v1"
"""Aggregate absolute error relative to zero predictions, including invalid answers."""


def _baseline(instance: Instance, key: InstanceKey, name: str) -> Answer:
    if name not in ("zero", "pc", "correlation", "oracle"):
        raise ValueError(f"unknown baseline: {name}")
    if name == "pc":
        from causalagentbench.scoring.scripted_agent import AmbiguityPolicy, scripted_answer

        return scripted_answer(instance, policy=AmbiguityPolicy.EXCLUDE)
    verdicts = []
    for label in instance.candidate_drivers:
        nodes = key.scm.scaffold.correct_adjustment[key.label_to_node[label]] if name == "oracle" else ()
        adjust = tuple(key.node_to_label[node] for node in nodes)
        estimate = (
            0.0
            if name == "zero"
            else regression_coefficient(instance.data, instance.columns, instance.outcome, label, adjust)
        )
        verdicts.append(DriverVerdict(driver=label, adjustment_set=adjust, effect=estimate))
    return Answer(verdicts=tuple(verdicts))


def evaluate_task(
    index: int,
    task: tuple[Instance, InstanceKey],
    *,
    model: str | None,
    baseline: str | None,
    max_tool_calls: int,
    max_turns: int,
) -> dict[str, Any]:
    """Evaluate one task, recording provider errors separately from invalid model answers."""
    instance, key = task
    start = time.monotonic()
    turns: list[TurnTrace] = []
    try:
        if baseline is not None:
            answer = "<answer>" + _baseline(instance, key, baseline).model_dump_json() + "</answer>"
        else:
            from causalagentbench.agent.clients import LiteLLMClient

            assert model is not None
            answer = run_agent(
                instance,
                LiteLLMClient(model=model, seed=None),
                max_tool_calls=max_tool_calls,
                max_turns=max_turns,
                on_turn=turns.append,
            )
        score = score_output(answer, key)
        n_candidates = len(key.scm.scaffold.drivers)
        zero_effect_mae = sum(
            abs(key.scm.total_effect_on_outcome(driver)) / n_candidates for driver in key.scm.scaffold.drivers
        )
        result: dict[str, Any] = {
            "type": "episode",
            "index": index,
            "model": model or baseline,
            "parse_ok": score.parse_ok,
            "reward": score.overall,
            "adjustment_score": score.identification,
            "effect_score": score.estimation,
            # Invalid answers are evaluated as all-zero estimates, never dropped from the metric.
            "effect_mae": score.effect_mae if score.parse_ok else zero_effect_mae,
            "zero_effect_mae": zero_effect_mae,
            "n_candidates": n_candidates,
            "ranking_tau": score.ranking_tau,
            "answer": answer,
        }
    except Exception as exc:
        result = {
            "type": "error",
            "index": index,
            "model": model or baseline,
            "error": f"{type(exc).__name__}: {exc}",
        }
    result.update(
        {
            "scoring_version": SCORING_VERSION,
            "reporting_version": REPORTING_VERSION,
            "seconds": time.monotonic() - start,
            "turns": len(turns),
            "tool_calls_requested": sum(len(t.tool_calls) for t in turns),
            "prompt_tokens": sum(t.prompt_tokens or 0 for t in turns),
            "completion_tokens": sum(t.completion_tokens or 0 for t in turns),
            "estimated_cost_usd": (
                sum(t.cost_usd for t in turns if t.cost_usd is not None)
                if all(t.cost_usd is not None for t in turns)
                else None
            ),
        }
    )
    return result
