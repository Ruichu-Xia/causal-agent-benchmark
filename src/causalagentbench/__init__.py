"""CausalAgentBench: a verifiable causal-reasoning benchmark and RL environment.

Stage 0 exposes the generator foundation: a canonical linear-Gaussian SCM whose exact effects are the
hidden answer key, plus the reject loop whose gates guarantee each instance requires causal reasoning.
Stage 1 adds the agent-facing task interface: instance presentation and the answer contract.
"""

from causalagentbench.answer import Answer, AnswerParseError, DriverVerdict, parse_answer
from causalagentbench.config import GeneratorConfig
from causalagentbench.gates import GateReport, evaluate_gates, generate_gated
from causalagentbench.generate import generate
from causalagentbench.instance import Instance, InstanceKey, build_instance, build_prompt
from causalagentbench.labeling import Labeler, abstract_labels
from causalagentbench.sample import sample
from causalagentbench.scaffold import CANONICAL
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
