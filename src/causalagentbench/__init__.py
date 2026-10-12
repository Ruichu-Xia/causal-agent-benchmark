"""CausalAgentBench: CauseNet-grounded causal analysis with exact rewards."""

from causalagentbench.benchmark import make_task
from causalagentbench.config import GeneratorConfig
from causalagentbench.environment import CausalEnv
from causalagentbench.generation.causenet import CauseNetLabeler
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
from causalagentbench.task.dataset import load_dataset, save_dataset
from causalagentbench.task.instance import Instance, InstanceKey, build_instance, build_prompt
from causalagentbench.task.labeling import Labeler, Labeling, LabelProvenance, abstract_labels
from causalagentbench.types import SCM, Role, Scaffold

__all__ = [
    "SCM",
    "Answer",
    "AnswerParseError",
    "AnswerScore",
    "CausalEnv",
    "CauseNetLabeler",
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
    "make_task",
    "parse_answer",
    "parse_answer_for_instance",
    "sample",
    "save_dataset",
    "score_answer",
    "score_output",
    "validate_answer_contract",
]
