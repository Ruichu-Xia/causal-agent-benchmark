"""Generate instances from scratch: sample a gated SCM, sample data, apply a labeler.

The train-data path. Stores nothing - SCM, data, and names are produced on the fly. ``generate_instance``
makes one; ``generate_instances`` streams many; ``alpha_sweep`` builds all grid levels of one SCM (paired
skins) from a single naming proposal. Pass an ``LLMLabeler`` (see ``llm_labeler``) for named instances, or
``abstract_labels`` for name-blind ones.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from causalagentbench.config import GeneratorConfig
from causalagentbench.generation.clients import LiteLLMProposer
from causalagentbench.generation.labeler import construct_labeling, grounding_fn, propose_triples
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.gates import generate_gated
from causalagentbench.task.instance import Instance, InstanceKey, build_instance
from causalagentbench.task.labeling import Labeler, Labeling
from causalagentbench.types import SCM


def _replay(labeling: Labeling) -> Labeler:
    """A labeler that returns a pre-constructed ``Labeling`` (for paired skins over one triple set)."""

    def labeler(_scm: SCM, _rng: np.random.Generator) -> Labeling:
        return labeling

    return labeler


def generate_instance(
    seed: int,
    labeler: Labeler,
    *,
    n_samples: int = 2000,
    config: GeneratorConfig | None = None,
) -> tuple[Instance, InstanceKey]:
    """Generate a gated SCM at ``seed``, sample ``n_samples`` rows, and label it with ``labeler``."""
    scm, _ = generate_gated(seed, config)
    data = sample(scm, n_samples, seed=seed)
    return build_instance(scm, data, seed, labeler=labeler)


def generate_instances(
    n: int,
    labeler: Labeler,
    *,
    seed: int = 0,
    n_samples: int = 2000,
    config: GeneratorConfig | None = None,
) -> Iterator[tuple[Instance, InstanceKey]]:
    """Stream ``n`` instances from independent seeds derived from ``seed`` (reproducible train data)."""
    for raw in np.random.SeedSequence(seed).generate_state(n):
        yield generate_instance(int(raw), labeler, n_samples=n_samples, config=config)


def alpha_sweep(
    seed: int,
    levels: tuple[float, ...],
    *,
    gen_model: str,
    causenet_dump: str,
    critic_model: str | None = None,
    max_revisions: int = 2,
    n_samples: int = 2000,
) -> list[tuple[float, Instance, InstanceKey]]:
    """Build one fixed SCM at each grid ``level`` from a *single* naming proposal - paired skins.

    All skins share the SCM, data, and column order (same ``seed``) and the same domain/name pool; only
    the alignment level (which nodes are aligned / neutral / misaligned) differs. One LLM proposal call
    (+ critic/revise) is amortized across every level.
    """
    scm, _ = generate_gated(seed)
    data = sample(scm, n_samples, seed=seed)
    triples = propose_triples(
        scm,
        np.random.default_rng(seed),
        proposer=LiteLLMProposer(gen_model),
        grounding=grounding_fn(causenet_dump),
        critic=LiteLLMProposer(critic_model) if critic_model else None,
        max_revisions=max_revisions,
    )
    producer = f"llm:{gen_model}" + (f"+critic:{critic_model}" if critic_model else "")

    skins: list[tuple[float, Instance, InstanceKey]] = []
    for level in levels:
        labeling = construct_labeling(triples, scm, level, producer)
        instance, key = build_instance(scm, data, seed, labeler=_replay(labeling))
        skins.append((level, instance, key))
    return skins
