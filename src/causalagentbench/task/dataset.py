"""Freeze and load evaluator-owned tasks, including exact prompts and hidden keys."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from causalagentbench.task.instance import Instance, InstanceKey, build_prompt, outcome_and_drivers
from causalagentbench.task.labeling import LabelProvenance
from causalagentbench.types import SCM, FloatArray, Role, Scaffold


def save_dataset(path: str | Path, tasks: Sequence[tuple[Instance, InstanceKey]]) -> None:
    """Freeze tasks, exact prompts and keys as a JSON/NPZ pair. Refuse empty sets and overwrites.

    These are evaluator artifacts: the JSON contains the hidden SCM. Agents receive only ``Instance``
    prompts and tool outputs, never the files or ``InstanceKey``.
    """
    if not tasks:
        raise ValueError("cannot save an empty dataset")
    stem = Path(path).with_suffix("")
    if stem.with_suffix(".json").exists() or stem.with_suffix(".npz").exists():
        raise FileExistsError(f"dataset already exists: {stem}")
    records = []
    tables: dict[str, Any] = {}
    for index, (instance, key) in enumerate(tasks):
        rid = f"task_{index:05d}"
        records.append(
            {
                "id": rid,
                "scaffold": serialize_scaffold(key.scm.scaffold),
                "W": key.scm.W.tolist(),
                "noise_var": key.scm.noise_var.tolist(),
                "sigma": key.scm.sigma.tolist(),
                "tau": key.scm.tau.tolist(),
                "node_to_label": dict(key.node_to_label),
                "columns": list(instance.columns),
                "context": instance.context,
                "column_schema": dict(instance.column_schema),
                "prompt": instance.prompt,
                "provenance": asdict(key.provenance),
            }
        )
        tables[rid] = instance.data
    payload = json.dumps({"version": "causalagentbench-v1", "records": records}, indent=2, allow_nan=False)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with stem.with_suffix(".npz").open("xb") as handle:
        np.savez_compressed(handle, **tables)
    with stem.with_suffix(".json").open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")


def serialize_scaffold(scaffold: Scaffold) -> dict[str, Any]:
    """JSON-safe representation of ``scaffold``, for round-tripping a per-instance composed topology."""
    return {
        "node_names": list(scaffold.node_names),
        "roles": {node: role.value for node, role in scaffold.roles.items()},
        "edges": [list(edge) for edge in scaffold.edges],
        "drivers": list(scaffold.drivers),
        "outcome": scaffold.outcome,
        "correct_adjustment": {driver: list(adj) for driver, adj in scaffold.correct_adjustment.items()},
    }


def deserialize_scaffold(data: dict[str, Any]) -> Scaffold:
    """Reconstruct a ``Scaffold`` from ``serialize_scaffold``'s output."""
    return Scaffold(
        node_names=tuple(data["node_names"]),
        roles={node: Role(role) for node, role in data["roles"].items()},
        edges=tuple(tuple(edge) for edge in data["edges"]),
        drivers=tuple(data["drivers"]),
        outcome=data["outcome"],
        correct_adjustment={driver: tuple(adj) for driver, adj in data["correct_adjustment"].items()},
    )


def load_dataset(path: str | Path) -> list[tuple[Instance, InstanceKey]]:
    """Load every frozen task from a dataset JSON/NPZ pair.

    ``path`` is the output stem (or either the ``.json``/``.npz`` file directly - the suffix is
    normalized), matching ``causalagentbench build --out``. Reconstructs each record's SCM, labeling, and
    prompt exactly as originally built.
    """
    stem = Path(path).with_suffix("")
    blob = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if not blob["records"]:
        raise ValueError("dataset is empty")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as tables:
        return [_load_record(record, tables[record["id"]]) for record in blob["records"]]


def _load_record(record: dict[str, Any], data: FloatArray) -> tuple[Instance, InstanceKey]:
    """Reconstruct one record's ``(Instance, InstanceKey)`` pair."""
    scaffold = deserialize_scaffold(record["scaffold"])
    scm = SCM(
        scaffold=scaffold,
        W=np.array(record["W"]),
        noise_var=np.array(record["noise_var"]),
        sigma=np.array(record["sigma"]),
        tau=np.array(record["tau"]),
    )

    node_to_label: dict[str, str] = record["node_to_label"]
    label_to_node = {label: node for node, label in node_to_label.items()}
    columns = tuple(record["columns"])
    outcome, candidate_drivers = outcome_and_drivers(scaffold, node_to_label, columns)
    context = record["context"]
    column_schema = record["column_schema"]
    prompt = record.get("prompt")
    if prompt is None:  # legacy builds predate frozen prompts
        prompt = build_prompt(columns, outcome, candidate_drivers, context, column_schema)

    instance = Instance(
        columns=columns,
        data=np.ascontiguousarray(data),
        outcome=outcome,
        candidate_drivers=candidate_drivers,
        prompt=prompt,
        context=context,
        column_schema=column_schema,
    )
    key = InstanceKey(
        scm=scm,
        label_to_node=label_to_node,
        node_to_label=node_to_label,
        provenance=LabelProvenance(**record["provenance"]),
    )
    return instance, key
