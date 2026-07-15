import dataclasses

import numpy as np

from causalagentbench import generate_gated, sample
from causalagentbench.instance import Instance, build_instance


def _instance(seed: int = 0):
    scm, _ = generate_gated(seed)
    data = sample(scm, 128, seed=seed)
    return scm, data, build_instance(scm, data, seed=seed)


def test_instance_is_public_only():
    # No field should expose the SCM, graph, or roles.
    field_types = {f.name: f.type for f in dataclasses.fields(Instance)}
    assert "scm" not in field_types
    assert not any("SCM" in str(t) or "Role" in str(t) for t in field_types.values())


def test_shapes_and_partition():
    scm, _data, (instance, _key) = _instance()
    k = scm.n_nodes
    assert instance.data.shape == (128, k)
    assert len(instance.columns) == k
    assert len(instance.candidate_drivers) == 3
    assert instance.outcome not in instance.candidate_drivers
    assert set(instance.other_columns) == set(instance.columns) - {instance.outcome, *instance.candidate_drivers}


def test_key_is_a_bijection_over_columns():
    _, _, (instance, key) = _instance()
    assert set(key.label_to_node) == set(instance.columns)
    assert {key.label_to_node[c] for c in instance.columns} == set(key.scm.scaffold.node_names)
    for node, label in key.node_to_label.items():
        assert key.label_to_node[label] == node


def test_display_permutation_preserves_column_data():
    # Each displayed column's data must equal the SCM-order column for the node it maps to.
    scm, data, (instance, key) = _instance()
    for j, label in enumerate(instance.columns):
        node_idx = scm.scaffold.index(key.label_to_node[label])
        assert np.array_equal(instance.data[:, j], data[:, node_idx])


def test_build_instance_is_deterministic():
    scm, _ = generate_gated(1)
    data = sample(scm, 64, seed=1)
    a, _ = build_instance(scm, data, seed=1)
    b, _ = build_instance(scm, data, seed=1)
    assert a.columns == b.columns
    assert np.array_equal(a.data, b.data)


def test_to_csv_header_matches_columns():
    _, _, (instance, _) = _instance()
    header = instance.to_csv().splitlines()[0]
    assert header == ",".join(instance.columns)


def test_prompt_names_outcome_and_drivers_but_not_roles():
    _, _, (instance, _) = _instance()
    assert instance.outcome in instance.prompt
    for driver in instance.candidate_drivers:
        assert driver in instance.prompt
    # The prompt must not leak causal-role vocabulary.
    for word in ("confounder", "mediator", "decoy", "adjust for these"):
        assert word not in instance.prompt.lower()
