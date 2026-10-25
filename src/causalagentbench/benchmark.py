"""Build benchmark tasks: a gated SCM, sampled observations, and CauseNet names."""

from __future__ import annotations

import hashlib

import numpy as np

from causalagentbench.generation.causenet import CauseNetLabeler, GroundingError
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.gates import generate_scm
from causalagentbench.task.instance import Instance, InstanceKey, build_instance

N_ROWS = 10_000
"""Observations per task. The oracle's sampling error shrinks as 1/sqrt(n); tools return summaries only."""

SPLITS = {"train": 0, "dev": 1, "test": 2, "private": 3}
"""Seed namespaces. ``private`` additionally requires a secret salt held by the benchmark authors."""

SCM_ATTEMPTS = 32
"""Fresh SCMs to try when CauseNet cannot name one."""

NAMING_ATTEMPTS = 4
"""Naming searches per SCM before it is abandoned.

Names must match the true graph, so naming is the one acceptance step that reads it. Retrying names on the
same graph, rather than redrawing it, keeps that dependence small: abandoning an SCM is rare (measured: 0 of
8 dev tasks, about 1% of SCMs on average), and realized orientations on the frozen splits match their
equivalence-class shares (``scripts/audit.py``)."""


def split_seed(split: str, index: int, salt: str | None = None) -> np.random.SeedSequence:
    """Return the seed of task ``index`` in ``split``; splits never share a seed."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a nonnegative integer")
    if (split == "private") != (salt is not None):
        raise ValueError("a salt is required for, and only for, the private split")
    entropy = [SPLITS[split], index]
    if salt is not None:
        entropy.append(int.from_bytes(hashlib.sha256(salt.encode()).digest()[:16], "big"))
    return np.random.SeedSequence(entropy)


def make_task(
    split: str, index: int, *, labeler: CauseNetLabeler, salt: str | None = None, n_rows: int = N_ROWS
) -> tuple[Instance, InstanceKey]:
    """Build task ``index`` of ``split`` deterministically; names are exact CauseNet concepts."""
    for child in split_seed(split, index, salt).spawn(SCM_ATTEMPTS):
        scm_seed, data_seed, label_seed = child.spawn(3)
        scm, _ = generate_scm(scm_seed)
        # Stored at float32 precision, so a frozen dataset reloads exactly the table the task was built with.
        data = sample(scm, n_rows, data_seed).astype(np.float32).astype(np.float64)
        for naming_seed in label_seed.spawn(NAMING_ATTEMPTS):
            try:
                return build_instance(scm, data, naming_seed, labeler=labeler)
            except GroundingError:
                continue
    raise GroundingError(f"could not name task {index} of {split} after {SCM_ATTEMPTS} SCMs")
