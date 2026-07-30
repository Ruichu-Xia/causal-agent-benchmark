"""Instance generation (optional, LLM-backed).

Everything here is used only to *produce* CauseNet/LLM-grounded naming; consuming the benchmark with the
always-available ``abstract_labels`` needs none of it. The LLM client is an optional dependency
(``pip install 'causalagentbench[generation]'``); imports are lazy, so this package stays importable
without the extra.

The naming alignment is a fixed discrete grid ``{-1,0,+1}``, *constructed* per instance - applied
uniformly to every non-outcome node - from per-node aligned/neutral/misaligned name triples (one generator
call + optional correctness critic per SCM). Example::

    from causalagentbench.generation import alpha_sweep, GRID
    skins = alpha_sweep(seed=0, levels=GRID, gen_model="openai/gpt-5",
                        causenet_dump="causenet.jsonl.bz2", critic_model="openai/gpt-5")
"""

from causalagentbench.generation.config import BuildConfig
from causalagentbench.generation.generate import alpha_sweep, alpha_sweep_with, generate_instance, generate_instances
from causalagentbench.generation.labeler import GRID, LLMLabeler, construct_labeling, llm_labeler, propose_triples

__all__ = [
    "GRID",
    "BuildConfig",
    "LLMLabeler",
    "alpha_sweep",
    "alpha_sweep_with",
    "construct_labeling",
    "generate_instance",
    "generate_instances",
    "llm_labeler",
    "propose_triples",
]
