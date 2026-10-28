"""Text causal reasoning over random causal worlds, as a Verifiers taskset.

Each task describes a small causal system in words and asks one question with an exact answer:
d-separation (``independence``), identification by adjustment (``adjustment``), causal and observed
effects from stated probabilities (``effect``), or what the independences imply about the structure
(``discovery``). The model replies in plain text, with no tools, and ends with "Answer: <answer>".
The reward is 1.0 for a correct final answer and 0.0 otherwise.

Splits (``causalagentbench.reasoning.dataset``): ``train``; ``iid`` (same distribution, other seeds);
``ood_wording`` (held-out phrasings); ``ood_size`` (larger graphs); ``ood_vocab`` (held-out concept
names). The generator, verifiers and audits are in ``causalagentbench.reasoning``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.harnesses.null import NullHarness

from causalagentbench.reasoning.dataset import build, load_concepts
from causalagentbench.reasoning.families import FAMILIES
from causalagentbench.reasoning.grading import final_answer, grade


class CausalReasoningConfig(vf.TasksetConfig):
    """Which tasks to generate."""

    split: Literal["train", "iid", "ood_wording", "ood_size", "ood_vocab"] = "train"
    seed: int = 0
    """Problem seed; within one seed, iid, ood_wording and ood_vocab pose the same problems."""
    num_tasks: int = 1000
    families: list[str] = Field(default_factory=lambda: list(FAMILIES))


class CausalReasoningData(vf.TaskData):
    """One generated task: the prompt, plus what the grader needs."""

    family: str
    kind: str
    """Answer format: yesno, number or choice."""
    answer: str
    split: str


class CausalReasoningTask(vf.Task[CausalReasoningData, vf.State, vf.TaskConfig]):
    """A single-turn question scored by its exact answer (Verifiers awaits every reward and metric)."""

    @vf.reward
    async def correct(self, task: CausalReasoningData, trace: vf.Trace) -> float:
        """1.0 if the reply's last "Answer:" line matches the exact answer, else 0.0."""
        return grade(task.kind, task.answer, trace.last_reply)

    @vf.metric
    async def answered(self, trace: vf.Trace) -> float:
        """1.0 if the reply has an "Answer:" line at all."""
        return float(final_answer(trace.last_reply) is not None)

    @vf.metric
    async def truncated(self, trace: vf.Trace) -> float:
        """1.0 if the reply hit a token or turn limit."""
        return float(trace.is_truncated)


class CausalReasoningHarness(NullHarness):
    """Plain chat with no tools: the task tests reasoning, not computation."""


class CausalReasoningTaskset(vf.Taskset[CausalReasoningTask, CausalReasoningConfig]):
    """Tasks of one split, generated deterministically from the config."""

    def load(self) -> Iterator[CausalReasoningTask]:
        """Generate the configured tasks."""
        config = self.config
        for t in build(config.num_tasks, config.seed, load_concepts(), config.split, tuple(config.families)):
            data = CausalReasoningData(
                id=t["id"],
                prompt=t["prompt"],
                family=t["family"],
                kind=t["kind"],
                answer=t["answer"],
                split=t["split"],
            )
            yield CausalReasoningTask(data, config.task)
