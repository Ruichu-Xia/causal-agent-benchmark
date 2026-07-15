"""CausalAgentBench: a verifiable causal-reasoning benchmark and RL environment.

Stage 0 exposes the generator foundation: a canonical linear-Gaussian SCM whose exact effects are the
hidden answer key, plus the reject loop whose gates guarantee each instance requires causal reasoning.
"""

from causalagentbench.config import GeneratorConfig
from causalagentbench.gates import GateReport, evaluate_gates, generate_gated
from causalagentbench.generate import generate
from causalagentbench.sample import sample
from causalagentbench.scaffold import CANONICAL
from causalagentbench.types import SCM, Role, Scaffold

__all__ = [
    "CANONICAL",
    "SCM",
    "GateReport",
    "GeneratorConfig",
    "Role",
    "Scaffold",
    "evaluate_gates",
    "generate",
    "generate_gated",
    "sample",
]
