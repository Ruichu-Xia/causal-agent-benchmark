import numpy as np

from causalagentbench.labeling import abstract_labels
from causalagentbench.scaffold import CANONICAL


def test_abstract_labels_assigns_one_distinct_xlabel_per_node():
    labels = abstract_labels(CANONICAL, np.random.default_rng(0))
    assert set(labels) == set(CANONICAL.node_names)
    assert sorted(labels.values()) == sorted(f"X{i + 1}" for i in range(len(CANONICAL.node_names)))


def test_abstract_labels_is_deterministic_in_rng():
    a = abstract_labels(CANONICAL, np.random.default_rng(7))
    b = abstract_labels(CANONICAL, np.random.default_rng(7))
    assert a == b


def test_abstract_labels_leak_no_role_information():
    # Labels are generic X-tokens; none contains a role/name hint.
    labels = abstract_labels(CANONICAL, np.random.default_rng(3))
    assert all(v.startswith("X") and v[1:].isdigit() for v in labels.values())
