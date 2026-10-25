"""CausalAgentBench: CauseNet-grounded causal analysis with exact rewards."""

from causalagentbench.benchmark import make_task
from causalagentbench.environment import CausalEnv
from causalagentbench.generation.causenet import CauseNetLabeler
from causalagentbench.scm.generate import draw_correlation, realize
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.gates import GateReport, evaluate_gates, generate_scm
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
from causalagentbench.task.instance import Instance, InstanceKey, build_instance, build_prompt, with_placeholder_names
from causalagentbench.task.labeling import Labeler, Labeling, placeholder_labels
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
    "Instance",
    "InstanceKey",
    "Labeler",
    "Labeling",
    "Role",
    "Scaffold",
    "build_instance",
    "build_prompt",
    "draw_correlation",
    "evaluate_gates",
    "generate_scm",
    "load_dataset",
    "make_task",
    "parse_answer",
    "parse_answer_for_instance",
    "placeholder_labels",
    "realize",
    "sample",
    "save_dataset",
    "score_answer",
    "score_output",
    "validate_answer_contract",
    "with_placeholder_names",
]
