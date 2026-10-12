"""The default benchmark: synthetic data with names grounded in CauseNet relations."""

from __future__ import annotations

import numpy as np

from causalagentbench.generation.causenet import CauseNetLabeler, GroundingError
from causalagentbench.scoring.gates import generate_gated_procedural
from causalagentbench.task.instance import Instance, InstanceKey, build_instance


def make_task(seed: int, *, labeler: CauseNetLabeler) -> tuple[Instance, InstanceKey]:
    """Build a five-candidate, 2,000-row task using only actual CauseNet concept/edge matches.

    Bounded retries use fresh SCMs when the corpus cannot support a topology. The caller's seed determines
    every draw. Names provide domain evidence; no graph, adjustment sets or direction hints are disclosed.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for child in np.random.SeedSequence(seed).spawn(32):
        scm_seed, label_seed = (int(s) for s in child.generate_state(2))
        scm, report = generate_gated_procedural(scm_seed, n_candidates=5)
        assert report.data is not None
        try:
            return build_instance(scm, report.data, label_seed, labeler=labeler)
        except GroundingError:
            continue
    raise GroundingError(f"could not ground task seed {seed} after 32 SCM attempts; use a richer CauseNet dump")
