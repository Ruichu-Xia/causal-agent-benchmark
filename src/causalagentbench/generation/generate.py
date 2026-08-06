"""Generate instances from scratch: sample a gated SCM, sample data, apply a labeler.

The train-data path. Stores nothing - SCM, data, and names are produced on the fly. ``generate_instance``
makes one; ``generate_instances`` streams many; ``alpha_sweep`` builds all grid levels of one SCM (paired
skins) from a single naming proposal - ``alpha_sweep_with`` is its dependency-injected core (plain
``proposer``/``grounding``, the same seam ``propose_triples`` uses), ``alpha_sweep`` the plain-string
convenience wrapper over it. Pass an ``LLMLabeler`` (see ``llm_labeler``) for named instances, or
``abstract_labels`` for name-blind ones.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from causalagentbench.config import GeneratorConfig
from causalagentbench.generation.clients import LabelProposer, LiteLLMProposer
from causalagentbench.generation.labeler import GroundingFn, construct_labeling, grounding_fn, propose_triples
from causalagentbench.scm.gadgets import RESOLVE_PROBABILITY
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.gates import generate_gated_procedural
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
    n_samples: int | None = None,
    n_candidates: int | None = None,
    resolve_probability: float = RESOLVE_PROBABILITY,
    config: GeneratorConfig | None = None,
) -> tuple[Instance, InstanceKey]:
    """Generate a gated procedural SCM at ``seed`` and label it with ``labeler``.

    ``n_samples=None`` (the default) uses the exact table the oracle gate already validated
    (``report.data``) - the only way to guarantee the gate's oracle-recovery guarantee actually holds on
    what's returned. Pass an explicit ``n_samples`` to draw a fresh,
    independent sample at that size instead; that sample is not guaranteed to clear the same oracle bar.
    """
    scm, report = generate_gated_procedural(
        seed, config, n_candidates=n_candidates, resolve_probability=resolve_probability
    )
    if n_samples is None:
        assert report.data is not None  # guaranteed: generate_gated_procedural only returns a passing report
        data = report.data
    else:
        data = sample(scm, n_samples, seed=seed)
    return build_instance(scm, data, seed, labeler=labeler)


def generate_instances(
    n: int,
    labeler: Labeler,
    *,
    seed: int = 0,
    n_samples: int | None = None,
    n_candidates: int | None = None,
    resolve_probability: float = RESOLVE_PROBABILITY,
    config: GeneratorConfig | None = None,
) -> Iterator[tuple[Instance, InstanceKey]]:
    """Stream ``n`` instances from independent seeds derived from ``seed`` (reproducible train data)."""
    for raw in np.random.SeedSequence(seed).generate_state(n):
        yield generate_instance(
            int(raw),
            labeler,
            n_samples=n_samples,
            n_candidates=n_candidates,
            resolve_probability=resolve_probability,
            config=config,
        )


def alpha_sweep_with(
    seed: int,
    levels: tuple[float, ...],
    *,
    proposer: LabelProposer,
    grounding: GroundingFn,
    producer: str,
    critic: LabelProposer | None = None,
    max_revisions: int = 2,
    n_samples: int | None = None,
    n_candidates: int | None = None,
    resolve_probability: float = RESOLVE_PROBABILITY,
) -> list[tuple[float, Instance, InstanceKey]]:
    """Build one fixed SCM at each grid ``level`` from a *single* naming proposal - paired skins.

    All skins share the SCM, data, and column order (same ``seed``) and the same domain/name pool; only
    the alignment level (which nodes are aligned / neutral / misaligned) differs. One proposal call
    (+ critic/revise) is amortized across every level. Dependency-injected core: takes a ``proposer``/
    ``grounding`` directly, the same seam ``propose_triples`` itself uses, so it's testable with a stub -
    no network, no monkeypatching. ``alpha_sweep`` is the plain-string convenience wrapper over this for
    real (LiteLLM-backed) use.

    ``n_samples=None`` (the default) uses the exact table the oracle gate already validated
    (``report.data``); pass an explicit ``n_samples`` to draw a fresh, independent sample instead (not
    guaranteed to clear the same oracle bar - see ``generate_instance``).
    """
    scm, report = generate_gated_procedural(seed, n_candidates=n_candidates, resolve_probability=resolve_probability)
    if n_samples is None:
        assert report.data is not None  # guaranteed: generate_gated_procedural only returns a passing report
        data = report.data
    else:
        data = sample(scm, n_samples, seed=seed)
    triples = propose_triples(
        scm,
        np.random.default_rng(seed),
        proposer=proposer,
        grounding=grounding,
        critic=critic,
        max_revisions=max_revisions,
    )

    skins: list[tuple[float, Instance, InstanceKey]] = []
    for level in levels:
        labeling = construct_labeling(triples, scm, level, producer)
        instance, key = build_instance(scm, data, seed, labeler=_replay(labeling))
        skins.append((level, instance, key))
    return skins


def alpha_sweep(
    seed: int,
    levels: tuple[float, ...],
    *,
    gen_model: str,
    causenet_dump: str,
    critic_model: str | None = None,
    max_revisions: int = 2,
    n_samples: int | None = None,
    n_candidates: int | None = None,
    resolve_probability: float = RESOLVE_PROBABILITY,
) -> list[tuple[float, Instance, InstanceKey]]:
    """Convenience wrapper over ``alpha_sweep_with``.

    Builds its proposer/grounding/producer from plain model-name strings and a CauseNet dump path - the
    usual way to call this for real.
    """
    return alpha_sweep_with(
        seed,
        levels,
        proposer=LiteLLMProposer(gen_model),
        grounding=grounding_fn(causenet_dump),
        producer=f"llm:{gen_model}" + (f"+critic:{critic_model}" if critic_model else ""),
        critic=LiteLLMProposer(critic_model) if critic_model else None,
        max_revisions=max_revisions,
        n_samples=n_samples,
        n_candidates=n_candidates,
        resolve_probability=resolve_probability,
    )
