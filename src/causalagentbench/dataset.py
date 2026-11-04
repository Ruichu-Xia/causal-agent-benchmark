"""Build, save and load data-analysis task splits (``docs/environment_v3.md``).

Item ``k`` of a role (``train``, ``dev``, ``test``) draws its problem from seed ``(role seed, k)`` and its
wording from ``(role seed, k, 1)``, so ``iid`` and ``ood_wording`` pose the same problems item by item. Item
``k`` asks family ``k mod 4``. ``ood_size`` draws larger graphs, and ``ood_vocab`` names columns with the fifth
of CauseNet concepts held out of every other split (``causenet.held_out_concept``).
"""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from causalagentbench.causenet import CauseNetLabeler, held_out_concept, load_causenet
from causalagentbench.families import FAMILIES, Phrases, Spec, Task
from causalagentbench.world import DataWorld

ROLE_SEEDS = {"train": 0, "dev": 100, "test": 200}
SPLITS = {
    "train": (Spec(), False),
    "iid": (Spec(), False),
    "ood_wording": (Spec(held_out_wording=True), False),
    "ood_size": (Spec(large=True), False),
    "ood_vocab": (Spec(), True),
}
"""Split name -> (problem spec, whether names come from held-out concepts)."""

_LABELERS: dict[bool, CauseNetLabeler] = {}
"""Per-process labelers, built once by ``_init``."""


def labeler(causenet: str, held_out: bool) -> CauseNetLabeler:
    """A labeler over the training concepts of CauseNet, or over the held-out ones."""
    graph = load_causenet(causenet)
    keep = [concept for concept in graph if isinstance(concept, str) and held_out_concept(concept) == held_out]
    return CauseNetLabeler(graph.subgraph(keep).copy())


def _init(causenet: str) -> None:
    """Build both labelers in this process, once; forked workers inherit them."""
    if not _LABELERS:
        for held_out in (False, True):
            _LABELERS[held_out] = labeler(causenet, held_out)


def make(split: str, role: str, k: int) -> Task:
    """Item ``k`` of ``split`` in ``role``; needs ``_init`` to have run in this process."""
    spec, held_out_names = SPLITS[split]
    family = list(FAMILIES)[k % len(FAMILIES)]
    problem = np.random.default_rng([ROLE_SEEDS[role], k])
    surface = np.random.default_rng([ROLE_SEEDS[role], k, 1])
    return FAMILIES[family](problem, Phrases(surface, spec.held_out_wording), spec, _LABELERS[held_out_names])


def world_to_dict(world: DataWorld) -> dict[str, Any]:
    """A JSON-ready form of a world."""
    return {
        "names": list(world.names),
        "edges": [list(edge) for edge in world.edges],
        "members": [[list(edge) for edge in member] for member in world.members],
        "name_edges": [list(edge) for edge in world.name_edges],
        "weights": world.weights.tolist(),
        "noise_var": world.noise_var.tolist(),
        "sigma": world.sigma.tolist(),
        "data_seed": world.data_seed,
        "data_moments": list(world.data_moments),
    }


def world_from_dict(record: dict[str, Any]) -> DataWorld:
    """The world a ``world_to_dict`` record describes."""

    def edges(pairs: list[list[int]]) -> tuple[tuple[int, int], ...]:
        """Pairs as a tuple of int tuples."""
        return tuple((int(p), int(c)) for p, c in pairs)

    return DataWorld(
        names=tuple(record["names"]),
        edges=edges(record["edges"]),
        members=tuple(edges(member) for member in record["members"]),
        name_edges=edges(record["name_edges"]),
        weights=np.array(record["weights"], dtype=float),
        noise_var=np.array(record["noise_var"], dtype=float),
        sigma=np.array(record["sigma"], dtype=float),
        data_seed=int(record["data_seed"]),
        data_moments=tuple(float(v) for v in record["data_moments"]),
    )


def to_record(task: Task, item_id: str, split: str) -> dict[str, Any]:
    """A JSON-ready task record."""
    return {
        "id": item_id,
        "split": split,
        "family": task.family,
        "kind": task.kind,
        "prompt": task.prompt,
        "answer": task.answer,
        "tolerance": task.meta.get("tolerance"),
        "meta": task.meta,
        "world": world_to_dict(task.world),
    }


def _build_one(job: tuple[str, str, int]) -> dict[str, Any]:
    """One task record (runs in a worker process)."""
    split, role, k = job
    return to_record(make(split, role, k), f"{split}_{role}-{k}", split)


def build(causenet: str, split: str, role: str, n: int, workers: int) -> list[dict[str, Any]]:
    """``n`` task records of ``split`` in ``role``, built in parallel, in item order."""
    jobs = [(split, role, k) for k in range(n)]
    _init(causenet)
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        return pool.map(_build_one, jobs, chunksize=4)


def drop_seen(records: Sequence[dict[str, Any]], seen: set[str]) -> list[dict[str, Any]]:
    """Records whose prompt is not in ``seen`` (prompts of earlier roles)."""
    return [record for record in records if record["prompt"] not in seen]


def save(records: Sequence[dict[str, Any]], path: str | Path) -> None:
    """Write records as JSON lines."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def load(path: str | Path) -> list[dict[str, Any]]:
    """Read records written by ``save``."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
