"""The RL environment: an investigation with the three tools, then the final reply's answer block is scored.

The class follows the TRL ``GRPOTrainer(environment_factory=...)`` contract: ``reset(**row)`` returns the
task prompt (TRL appends it to the last user message) and every public method is a named, documented tool.
The episode ends when the model replies without a tool call; ``reward_func`` scores that final reply exactly
as evaluation does, so training and evaluation share one prompt and one answer format. Use it as::

    tasks = load_dataset("artifacts/train")  # built with `build --split train --exclude <evaluation sets>`
    trainer = GRPOTrainer(..., environment_factory=lambda: CausalEnv(tasks), reward_funcs=reward_func)

with a dataset whose rows carry a ``seed`` column indexing ``tasks``. Training tasks come from the
``train`` seed namespace, and their hidden edges never use a CauseNet relation between two concepts that
appear together in an evaluation task, so a policy cannot be rewarded for memorizing evaluation answers.

The reward is ``AnswerScore.reward``: ``1 - min(task error, 2)`` for a valid answer and ``-1`` otherwise,
including when the final reply has no answer block.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from causalagentbench.agent.loop import MAX_TOOL_CALLS
from causalagentbench.agent.tools import CausalTools
from causalagentbench.benchmark import make_task
from causalagentbench.generation.causenet import CauseNetLabeler
from causalagentbench.scoring.score import INVALID_REWARD, score_output
from causalagentbench.task.dataset import concept_pairs, load_dataset
from causalagentbench.task.instance import Instance, InstanceKey


def training_labeler(causenet: str, evaluation_datasets: Sequence[str]) -> CauseNetLabeler:
    """A labeler for training tasks that avoids every concept pair used together in evaluation tasks."""
    tasks = [task for path in evaluation_datasets for task in load_dataset(path)]
    return CauseNetLabeler.from_path(causenet, exclude=concept_pairs(tasks))


class CausalEnv:
    """One episode of signed total-effect estimation with a tool budget and an exact reward."""

    def __init__(
        self,
        tasks: Sequence[tuple[Instance, InstanceKey]] | None = None,
        labeler: CauseNetLabeler | None = None,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ) -> None:
        """Bind pre-built training tasks, or a labeler that builds training task ``seed`` on ``reset``."""
        self._tasks = tasks
        self._labeler = labeler
        self._max_tool_calls = max_tool_calls
        self._tools: CausalTools | None = None
        self._key: InstanceKey | None = None
        self._calls = 0
        self.done = True
        self.reward = INVALID_REWARD
        self.info: dict[str, Any] = {}

    def reset(self, seed: int | None = None, task: tuple[Instance, InstanceKey] | None = None, **row: Any) -> str:
        """Start an episode on training task ``seed`` (or a frozen ``task``) and return its prompt.

        Task ``seed`` of a pre-built training set is the task ``make_task("train", seed)`` builds, because
        building is deterministic; both sources therefore give the same episode.
        """
        if task is None:
            if seed is None or (self._tasks is None and self._labeler is None):
                raise ValueError("reset needs a frozen task, or a seed with pre-built tasks or a labeler")
            if self._tasks is not None:
                task = self._tasks[int(seed)]
            else:
                assert self._labeler is not None
                task = make_task("train", int(seed), labeler=self._labeler)
        instance, self._key = task
        self._tools = CausalTools(instance)
        self._calls = 0
        self.done = False
        self.reward = INVALID_REWARD
        self.info = {}
        return instance.prompt

    def _spend(self) -> CausalTools:
        """Charge one tool call, or raise once the episode is over or the budget is spent."""
        if self.done or self._tools is None:
            raise ValueError("The episode is over.")
        if self._calls >= self._max_tool_calls:
            raise ValueError("Tool budget exhausted. Give your final answer now.")
        self._calls += 1
        return self._tools

    def _with_budget(self, result: str) -> str:
        """Append the remaining call budget, as the evaluation harness reports it after each turn."""
        return (
            f"{result}\n\n[budget: {self._max_tool_calls - self._calls} of {self._max_tool_calls} tool calls remaining]"
        )

    def correlate(self, x: str, y: str, given: list[str] | None = None) -> str:
        """Partial correlation between two columns given a conditioning set, with a 95% interval.

        Args:
            x: First column.
            y: Second column.
            given: Columns to condition on; empty for the marginal correlation.
        """
        return self._with_budget(self._spend().correlate(x, y, given))

    def regress(self, treatment: str, adjust: list[str] | None = None) -> str:
        """OLS coefficient of a column on the outcome, adjusting for a set of columns, with its standard error.

        Args:
            treatment: The candidate column.
            adjust: Columns to adjust for; may be empty.
        """
        return self._with_budget(self._spend().regress(treatment, adjust))

    def discover(self, alpha: float = 0.01) -> str:
        """Run the PC algorithm over all columns and list oriented and undirected edges.

        Args:
            alpha: Significance level of the conditional-independence tests.
        """
        return self._with_budget(self._spend().discover(alpha))

    def _score(self, reply: str) -> float:
        """Score the final reply as evaluation does, end the episode and return the reward."""
        if self.done or self._key is None:
            raise ValueError("The episode is over.")
        score = score_output(reply, self._key)
        self.done = True
        self.reward = score.reward
        self.info = {"parse_ok": score.parse_ok, "task_error": score.task_error, "tool_calls": self._calls}
        return self.reward


def final_reply(completion: str | list[dict[str, Any]]) -> str:
    """Return the text of a completion's last assistant message ("" if it ended on a tool call)."""
    if isinstance(completion, str):
        return completion
    for message in reversed(completion):
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, list):  # multimodal parts: keep the text ones
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return str(content)
    return ""


def reward_func(
    completions: Sequence[str | list[dict[str, Any]]], environments: Sequence[CausalEnv], **kwargs: Any
) -> list[float]:
    """TRL reward function: score each rollout's final reply against its own episode's hidden key."""
    return [env._score(final_reply(completion)) for completion, env in zip(completions, environments, strict=True)]
