import json

import numpy as np
import pytest

from causalagentbench import build_instance, generate_gated, sample
from causalagentbench.task.dataset import load_dataset
from causalagentbench.task.labeling import Labeling, LabelProvenance


def _fake_labeler(scm, rng):
    """A stub Labeler with no LLM call - mirrors the shape a real LLMLabeler would produce."""
    node_to_label = {n: f"col_{n.lower()}" for n in scm.scaffold.node_names}
    return Labeling(
        node_to_label=node_to_label,
        context="a test domain",
        column_schema={label: f"description of {label}" for label in node_to_label.values()},
        provenance=LabelProvenance(
            alignment=0.5, n_aligned=2, n_neutral=3, n_misaligned=1, domain="test", producer="fake"
        ),
    )


def _write_dataset(out, records_and_data):
    """Serialize (id, Instance, InstanceKey) triples exactly as scripts/build_testset.py does."""
    records = []
    tables = {}
    for rid, instance, key in records_and_data:
        prov = key.provenance
        records.append(
            {
                "id": rid,
                "seed": 0,
                "scaffold": "canonical",
                "W": key.scm.W.tolist(),
                "noise_var": key.scm.noise_var.tolist(),
                "sigma": key.scm.sigma.tolist(),
                "tau": key.scm.tau.tolist(),
                "node_to_label": dict(key.node_to_label),
                "columns": list(instance.columns),
                "context": instance.context,
                "column_schema": dict(instance.column_schema),
                "provenance": {
                    "alignment": prov.alignment,
                    "n_aligned": prov.n_aligned,
                    "n_neutral": prov.n_neutral,
                    "n_misaligned": prov.n_misaligned,
                    "domain": prov.domain,
                    "producer": prov.producer,
                },
            }
        )
        tables[rid] = instance.data
    out.with_suffix(".json").write_text(
        json.dumps({"version": "test", "scaffold": "canonical", "n_samples": 200, "records": records})
    )
    np.savez_compressed(out.with_suffix(".npz"), **tables)
    return out


def test_load_dataset_round_trips_a_built_instance(tmp_path):
    scm, _ = generate_gated(0)
    data = sample(scm, 200, seed=0)
    instance, key = build_instance(scm, data, seed=0, labeler=_fake_labeler)

    out = _write_dataset(tmp_path / "testset", [("s0000", instance, key)])
    [(loaded_instance, loaded_key)] = load_dataset(out)

    assert loaded_instance.columns == instance.columns
    assert loaded_instance.outcome == instance.outcome
    assert loaded_instance.candidate_drivers == instance.candidate_drivers
    assert loaded_instance.context == instance.context
    assert dict(loaded_instance.column_schema) == dict(instance.column_schema)
    assert loaded_instance.prompt == instance.prompt  # rebuilt via build_prompt, must match byte-for-byte
    np.testing.assert_array_equal(loaded_instance.data, instance.data)
    assert dict(loaded_key.node_to_label) == dict(key.node_to_label)
    assert loaded_key.provenance == key.provenance
    np.testing.assert_array_equal(loaded_key.scm.W, key.scm.W)
    np.testing.assert_array_equal(loaded_key.scm.tau, key.scm.tau)


def test_load_dataset_accepts_stem_or_either_file_path(tmp_path):
    scm, _ = generate_gated(1)
    data = sample(scm, 200, seed=1)
    instance, key = build_instance(scm, data, seed=1, labeler=_fake_labeler)
    out = _write_dataset(tmp_path / "testset", [("s0001", instance, key)])

    assert len(load_dataset(out)) == 1
    assert len(load_dataset(out.with_suffix(".json"))) == 1
    assert len(load_dataset(str(out) + ".npz")) == 1


def test_load_dataset_rejects_unknown_scaffold(tmp_path):
    out = tmp_path / "bad"
    record = {
        "id": "r0",
        "scaffold": "not-a-real-scaffold",
        "W": [[0]],
        "noise_var": [1],
        "sigma": [[1]],
        "tau": [[0]],
        "node_to_label": {},
        "columns": [],
        "context": "",
        "column_schema": {},
        "provenance": {
            "alignment": None,
            "n_aligned": 0,
            "n_neutral": 0,
            "n_misaligned": 0,
            "domain": None,
            "producer": "x",
        },
    }
    blob = {"version": "x", "scaffold": "?", "n_samples": 1, "records": [record]}
    out.with_suffix(".json").write_text(json.dumps(blob))
    np.savez_compressed(out.with_suffix(".npz"), r0=np.zeros((1, 1)))

    with pytest.raises(ValueError, match="unknown scaffold"):
        load_dataset(out)
