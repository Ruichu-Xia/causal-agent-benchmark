"""CausalAgentBench: a verifiable causal-reasoning benchmark and RL environment.

Stage 0 exposes the generator foundation: a canonical linear-Gaussian SCM whose exact effects are the
hidden answer key, plus the reject loop whose gates guarantee each instance requires causal reasoning.
Stage 1 adds the agent-facing task interface: instance presentation and the answer contract.
"""

from causalagentbench.config import GeneratorConfig
from causalagentbench.scm.generate import generate
from causalagentbench.scm.sample import sample
from causalagentbench.scm.scaffold import CANONICAL
from causalagentbench.scoring.gates import GateReport, evaluate_gates, generate_gated
from causalagentbench.task.answer import Answer, AnswerParseError, DriverVerdict, parse_answer
from causalagentbench.task.instance import Instance, InstanceKey, build_instance, build_prompt
from causalagentbench.task.labeling import Labeler, abstract_labels
from causalagentbench.types import SCM, Role, Scaffold

__all__ = [
    "CANONICAL",
    "SCM",
    "Answer",
    "AnswerParseError",
    "DriverVerdict",
    "GateReport",
    "GeneratorConfig",
    "Instance",
    "InstanceKey",
    "Labeler",
    "Role",
    "Scaffold",
    "abstract_labels",
    "build_instance",
    "build_prompt",
    "evaluate_gates",
    "generate",
    "generate_gated",
    "parse_answer",
    "sample",
]
