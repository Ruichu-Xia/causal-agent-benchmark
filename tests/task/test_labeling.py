import numpy as np

from causalagentbench import generate_gated
from causalagentbench.task.labeling import Labeling, abstract_labels


def _scm(seed: int = 0):
    scm, _ = generate_gated(seed)
    return scm


def test_abstract_labels_returns_labeling_with_xlabels():
    scm = _scm()
    labeling = abstract_labels(scm, np.random.default_rng(0))
    assert isinstance(labeling, Labeling)
    names = scm.scaffold.node_names
    assert set(labeling.node_to_label) == set(names)
    assert sorted(labeling.node_to_label.values()) == sorted(f"X{i + 1}" for i in range(len(names)))
    assert labeling.context == ""
    assert dict(labeling.column_schema) == {}
    assert labeling.provenance.alignment is None
    assert labeling.provenance.producer == "abstract"


def test_abstract_labels_is_deterministic_in_rng():
    scm = _scm()
    a = abstract_labels(scm, np.random.default_rng(7))
    b = abstract_labels(scm, np.random.default_rng(7))
    assert a.node_to_label == b.node_to_label


def test_abstract_labels_leak_no_role_information():
    labeling = abstract_labels(_scm(), np.random.default_rng(3))
    assert all(v.startswith("X") and v[1:].isdigit() for v in labeling.node_to_label.values())
