"""CausalAgentBench: a verifiable causal-reasoning benchmark and RL environment.

A hidden linear-Gaussian SCM is the answer key, on a procedurally-composed graph topology
(``scm/gadgets.py``) that varies driver count, archetype mix, and identifiability per instance; a
construct-then-gate loop (``generate_gated_procedural``) guarantees every instance requires genuine causal
reasoning. The task layer presents an ``Instance`` (data + context + prompt) and parses/scores an agent's
``Answer`` against the hidden ``InstanceKey``. Naming is a pure presentation layer (``Labeling``): the
abstract, always-available ``abstract_labels`` needs nothing beyond this package;
``causalagentbench.generation`` additionally offers LLM/CauseNet-grounded naming at a controlled alignment
level, for anyone who wants that axis. ``causalagentbench.agent`` is a minimal tool-using agent (a fixed
causal toolset + a model-agnostic tool-calling loop) and a batch evaluator that aggregates scores by naming
alignment.
"""

from causalagentbench.config import GeneratorConfig
from causalagentbench.scm.generate import generate
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.gates import GateReport, evaluate_gates, generate_gated_procedural
from causalagentbench.scoring.score import AnswerScore, score_answer, score_output
from causalagentbench.task.answer import (
    Answer,
    AnswerParseError,
    DriverVerdict,
    parse_answer,
    parse_answer_for_instance,
    validate_answer_contract,
)
from causalagentbench.task.dataset import load_dataset
from causalagentbench.task.instance import Instance, InstanceKey, build_instance, build_prompt
from causalagentbench.task.labeling import Labeler, Labeling, LabelProvenance, abstract_labels
from causalagentbench.types import SCM, Role, Scaffold

__all__ = [
    "SCM",
    "Answer",
    "AnswerParseError",
    "AnswerScore",
    "DriverVerdict",
    "GateReport",
    "GeneratorConfig",
    "Instance",
    "InstanceKey",
    "LabelProvenance",
    "Labeler",
    "Labeling",
    "Role",
    "Scaffold",
    "abstract_labels",
    "build_instance",
    "build_prompt",
    "evaluate_gates",
    "generate",
    "generate_gated_procedural",
    "load_dataset",
    "parse_answer",
    "parse_answer_for_instance",
    "sample",
    "score_answer",
    "score_output",
    "validate_answer_contract",
]
