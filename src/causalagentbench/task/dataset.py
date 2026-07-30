"""Load a pre-built dataset back into ``(Instance, InstanceKey)`` pairs.

Reads ``scripts/build_testset.py``'s ``<out>.json`` + ``<out>.npz``. This is the read side of a fixed
test set: build once with the ``generation`` extra, then evaluate many models against the identical
frozen instances with no LLM naming calls (no extra needed here at all - loading is plain numpy/json).

Each record embeds its own full ``Scaffold`` (``serialize_scaffold``/``deserialize_scaffold``), not a
name looked up in a small fixed set - every procedurally-composed instance has a genuinely different
topology (``scm/gadgets.py``), unlike the single shared ``CANONICAL`` graph this format used to assume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from causalagentbench.task.instance import Instance, InstanceKey, build_prompt, outcome_and_drivers
from causalagentbench.task.labeling import LabelProvenance
from causalagentbench.types import SCM, FloatArray, Role, Scaffold


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
    """Load every record from a ``build_testset.py`` output.

    ``path`` is the output stem (or either the ``.json``/``.npz`` file directly - the suffix is
    normalized), matching ``build_testset.py --out``. Reconstructs each record's SCM, labeling, and
    prompt exactly as originally built.
    """
    stem = Path(path).with_suffix("")
    blob = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    tables = np.load(stem.with_suffix(".npz"))
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
