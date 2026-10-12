"""A small episodic RL interface using the benchmark's tools and exact scorer."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from causalagentbench.agent.loop import _TOOL_SCHEMAS, MAX_TOOL_CALLS, ToolCall, _dispatch, _execute
from causalagentbench.agent.tools import CausalTools
from causalagentbench.benchmark import make_task
from causalagentbench.generation.causenet import CauseNetLabeler
from causalagentbench.scoring.score import SCORING_VERSION, score_output
from causalagentbench.task.instance import Instance, InstanceKey


class CausalEnv:
    """One investigation followed by one scored submission, without a model or RL dependency.

    Tool actions return zero reward. Submission ends the episode with the benchmark's [0, 1] reward.
    Every tool attempt, including malformed calls, spends one query. After the last query a submission
    remains available; another tool attempt truncates the episode with zero reward. ``reset`` starts a
    fresh deterministic task or accepts a frozen evaluator-owned ``(Instance, InstanceKey)`` pair.
    """

    def __init__(self, *, labeler: CauseNetLabeler | None = None, max_tool_calls: int = MAX_TOOL_CALLS) -> None:
        if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or max_tool_calls < 0:
            raise ValueError("max_tool_calls must be a nonnegative integer")
        self.max_tool_calls = max_tool_calls
        self._labeler = labeler
        self._instance: Instance | None = None
        self._key: InstanceKey | None = None
        self._tools: CausalTools | None = None
        self._calls = 0
        self._done = True

    def reset(
        self, *, seed: int = 0, task: tuple[Instance, InstanceKey] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return a public task prompt, tool schemas and budget; never return the key or raw data."""
        self._done = True
        if task is None:
            if self._labeler is None:
                raise ValueError("provide a frozen task to reset(), or a CauseNetLabeler to CausalEnv()")
            task = make_task(seed, labeler=self._labeler)
        self._instance, self._key = task
        self._tools = CausalTools(self._instance)
        self._calls = 0
        self._done = False
        observation = self._observation(self._instance.prompt)
        observation["tools"] = deepcopy(list(_TOOL_SCHEMAS))
        return observation, {}

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute ``tool`` or ``submit`` and return observation, reward, terminated, truncated, info.

        Tool: ``{"type": "tool", "name": "regress", "arguments": {...}}``.
        Submit: ``{"type": "submit", "answer": {"verdicts": [...]}}``; answer may also be model text
        containing an ``<answer>`` block. Invalid submissions terminate with zero reward.
        """
        if self._done or self._key is None or self._tools is None:
            raise RuntimeError("call reset before stepping a new episode")
        if isinstance(action, dict) and action.get("type") == "submit":
            answer = action.get("answer")
            if isinstance(answer, dict):
                try:
                    text = "<answer>" + json.dumps(answer, allow_nan=False) + "</answer>"
                except (TypeError, ValueError):
                    text = ""
            else:
                text = answer if isinstance(answer, str) else ""
            score = score_output(text, self._key)
            self._done = True
            info = {
                "scoring_version": SCORING_VERSION,
                "parse_ok": score.parse_ok,
                "adjustment_score": score.identification,
                "effect_score": score.estimation,
                "effect_mae": score.effect_mae,
                "ranking_tau": score.ranking_tau,
                "tool_calls": self._calls,
            }
            return self._observation("Episode complete."), score.overall, True, False, info

        if self._calls >= self.max_tool_calls:
            self._done = True
            return self._observation("Query budget exhausted."), 0.0, False, True, {"tool_calls": self._calls}

        self._calls += 1
        if not isinstance(action, dict) or action.get("type") != "tool":
            result = "Error: action must have type 'tool' or 'submit'."
        elif not isinstance(action.get("name"), str) or not isinstance(action.get("arguments", {}), dict):
            result = "Error: tool name must be a string and arguments must be an object."
        else:
            call = ToolCall(str(self._calls), action["name"], action.get("arguments", {}))
            result = _execute(_dispatch(self._tools), call)
        return self._observation(result), 0.0, False, False, {}

    def _observation(self, content: str) -> dict[str, Any]:
        return {
            "content": content,
            "tool_calls_remaining": self.max_tool_calls - self._calls,
            "must_submit": self._calls >= self.max_tool_calls,
        }
