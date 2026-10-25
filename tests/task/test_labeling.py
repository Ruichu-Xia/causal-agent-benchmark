import numpy as np

from causalagentbench.scoring.gates import generate_scm
from causalagentbench.task.labeling import Labeling, placeholder_labels


def _scm(seed: int = 0):
    """An accepted SCM."""
    scm, _ = generate_scm(np.random.SeedSequence(seed))
    return scm


def test_placeholder_labels_are_a_permutation_of_x_names():
    scm = _scm()
    labeling = placeholder_labels(scm, np.random.default_rng(0))
    names = scm.scaffold.node_names
    assert isinstance(labeling, Labeling)
    assert labeling.producer == "placeholder"
    assert set(labeling.node_to_label) == set(names)
    assert sorted(labeling.node_to_label.values()) == sorted(f"X{i + 1}" for i in range(len(names)))


def test_placeholder_labels_are_deterministic_in_the_rng():
    scm = _scm()
    first = placeholder_labels(scm, np.random.default_rng(7))
    assert first == placeholder_labels(scm, np.random.default_rng(7))
    assert first.node_to_label != placeholder_labels(scm, np.random.default_rng(8)).node_to_label


def test_the_outcome_label_varies_with_the_rng():
    """A fixed label for the outcome (or any role) would let the name carry the role."""
    scm = _scm()
    outcome_labels = {placeholder_labels(scm, np.random.default_rng(s)).node_to_label["Y"] for s in range(20)}
    assert len(outcome_labels) > 1
