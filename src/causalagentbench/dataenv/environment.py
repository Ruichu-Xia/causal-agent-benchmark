"""The data-analysis RL environment: a ``python`` tool over the task's data, then the final answer is graded.

The class follows TRL's ``GRPOTrainer(environment_factory=...)`` contract, like ``causalagentbench.environment``:
``reset(**row)`` returns the task prompt and every other public method is a tool, so ``python`` is the only
public method besides ``reset``. The episode ends when the model
replies without a tool call; ``reward_func`` grades that reply (1 if correct, else 0). Evaluation drives the
same class, so training and evaluation share one prompt, tool and grader.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from causalagentbench.dataenv.dataset import world_from_dict
from causalagentbench.dataenv.sandbox import PythonSession
from causalagentbench.environment import final_reply
from causalagentbench.reasoning.grading import NUMBER_TOLERANCE, final_answer, grade

MAX_TOOL_CALLS = 10
"""Tool calls per episode."""


class DataEnv:
    """One episode: a question about a dataset, a persistent Python session over it, and an exact reward."""

    def __init__(self, tasks: Sequence[dict[str, Any]], max_tool_calls: int = MAX_TOOL_CALLS) -> None:
        """Bind task records (``dataset.load``)."""
        self._tasks = tasks
        self._max_tool_calls = max_tool_calls
        self._task: dict[str, Any] | None = None
        self._session: PythonSession | None = None
        self._calls = 0
        self.done = True
        self.reward = 0.0
        self.info: dict[str, Any] = {}

    def reset(self, index: int | None = None, task: dict[str, Any] | None = None, **row: Any) -> str:
        """Start an episode on task ``index`` of the bound tasks (or the given record); return its prompt."""
        self._close()
        if task is None:
            if index is None:
                raise ValueError("reset needs a task index or a task record")
            task = self._tasks[int(index)]
        world = world_from_dict(task["world"])
        self._task = task
        self._session = PythonSession(world.data(), world.names)
        self._calls = 0
        self.done = False
        self.reward = 0.0
        self.info = {}
        return str(task["prompt"])

    def python(self, code: str) -> str:
        """Run Python code in a persistent session where the pandas DataFrame `df` holds the data.

        Variables persist between calls. Printed output is returned, and so is the value of a final expression.

        Args:
            code: Python source code to run.
        """
        if self.done or self._session is None:
            raise ValueError("The episode is over.")
        if self._calls >= self._max_tool_calls:
            raise ValueError("Tool budget exhausted. Give your final answer now.")
        self._calls += 1
        output = self._session.run(code)
        return f"{output}\n[{self._max_tool_calls - self._calls} of {self._max_tool_calls} tool calls left]"

    def _score(self, reply: str) -> float:
        """Grade the final reply, end the episode and return the reward."""
        if self.done or self._task is None:
            raise ValueError("The episode is over.")
        task = self._task
        tolerance = task.get("tolerance") or NUMBER_TOLERANCE
        self.reward = grade(task["kind"], task["answer"], reply, tolerance)
        self.done = True
        self.info = {
            "id": task["id"],
            "family": task["family"],
            "correct": self.reward == 1.0,
            "answered": final_answer(reply) is not None,
            "tool_calls": self._calls,
        }
        self._close()
        return self.reward

    def _close(self) -> None:
        """Stop the episode's Python worker, if any."""
        if self._session is not None:
            self._session.close()
            self._session = None


def reward_func(
    completions: Sequence[str | list[dict[str, Any]]], environments: Sequence[DataEnv], **kwargs: Any
) -> list[float]:
    """TRL reward function: grade each rollout's final reply against its own task."""
    return [env._score(final_reply(completion)) for completion, env in zip(completions, environments, strict=True)]
