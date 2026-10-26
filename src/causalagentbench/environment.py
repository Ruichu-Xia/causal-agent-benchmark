"""The RL environment: one episode is an investigation with the three tools, then one scored submission.

The class follows the TRL ``GRPOTrainer(environment_factory=...)`` contract: ``reset(**row)`` returns the
task prompt, and every public method is a named, documented tool. Use it as::

    labeler = training_labeler("causenet-precision.jsonl.bz2", ["data/dev", "data/test"])
    trainer = GRPOTrainer(..., environment_factory=lambda: CausalEnv(labeler), reward_funcs=reward_func)

with a dataset whose rows carry a ``seed`` column. Training tasks come from the ``train`` seed namespace,
and their hidden edges never use a CauseNet relation between two concepts that appear together in an
evaluation task, so a policy cannot be rewarded for memorizing evaluation answers.

The reward is ``AnswerScore.reward``: ``1 - min(task error, 2)`` for a valid submission and ``-1``
otherwise, including when the episode ends without a submission.
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

    def __init__(self, labeler: CauseNetLabeler | None = None, max_tool_calls: int = MAX_TOOL_CALLS) -> None:
        """Bind a labeler for generated training tasks (or pass frozen tasks to ``reset``)."""
        self._labeler = labeler
        self._max_tool_calls = max_tool_calls
        self._tools: CausalTools | None = None
        self._key: InstanceKey | None = None
        self._calls = 0
        self.done = True
        self.reward = INVALID_REWARD
        self.info: dict[str, Any] = {}

    def reset(self, seed: int | None = None, task: tuple[Instance, InstanceKey] | None = None, **row: Any) -> str:
        """Start an episode on training task ``seed`` (or a frozen ``task``) and return its prompt."""
        if task is None:
            if self._labeler is None or seed is None:
                raise ValueError("reset needs a frozen task, or a seed and a CauseNetLabeler")
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
            raise ValueError("Tool budget exhausted. Submit your answer now.")
        self._calls += 1
        return self._tools

    def correlate(self, x: str, y: str, given: list[str] | None = None) -> str:
        """Partial correlation between two columns given a conditioning set, with a 95% interval.

        Args:
            x: First column.
            y: Second column.
            given: Columns to condition on; empty for the marginal correlation.
        """
        return self._spend().correlate(x, y, given)

    def regress(self, treatment: str, adjust: list[str] | None = None) -> str:
        """OLS coefficient of a column on the outcome, adjusting for a set of columns, with a 95% interval.

        Args:
            treatment: The candidate column.
            adjust: Columns to adjust for; may be empty.
        """
        return self._spend().regress(treatment, adjust)

    def discover(self, alpha: float = 0.01) -> str:
        """Run the PC algorithm over all columns and list oriented and undirected edges.

        Args:
            alpha: Significance level of the conditional-independence tests.
        """
        return self._spend().discover(alpha)

    def submit(self, answer: str) -> str:
        """Submit the final answer and end the episode.

        Args:
            answer: The JSON answer, optionally inside an <answer>...</answer> block.
        """
        if self.done or self._key is None:
            raise ValueError("The episode is over.")
        text = answer if "<answer>" in answer else f"<answer>{answer}</answer>"
        score = score_output(text, self._key)
        self.done = True
        self.reward = score.reward
        self.info = {"parse_ok": score.parse_ok, "task_error": score.task_error, "tool_calls": self._calls}
        return "Answer recorded." if score.parse_ok else "Invalid answer; the episode is over."


def reward_func(environments: Sequence[CausalEnv], **kwargs: Any) -> list[float]:
    """TRL reward function: each episode's reward, ``-1`` if it never submitted a valid answer."""
    return [env.reward for env in environments]
