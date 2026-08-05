import numpy as np

from causalagentbench.scoring.gates import generate_gated_procedural
from causalagentbench.task.labeling import Labeling, abstract_labels


def _scm(seed: int = 0):
    scm, _ = generate_gated_procedural(seed)
    return scm


def test_abstract_labels_returns_labeling_with_xlabels():
    scm = _scm()
    labeling = abstract_labels(scm, np.random.default_rng(0))
    assert isinstance(labeling, Labeling)
    names = scm.scaffold.node_names
    assert set(labeling.node_to_label) == set(names)
    assert sorted(labeling.node_to_label.values()) == sorted(f"X{i + 1}" for i in range(len(names)))
    assert labeling.provenance.alignment is None
    assert labeling.provenance.producer == "abstract"


def test_abstract_labels_carry_a_schema_that_is_uniform_and_therefore_informationless():
    """Present so the prompt has the same shape as a named condition, identical so it says nothing.

    Without a context/schema, `build_prompt` omits both blocks, and a name-blind instance would differ
    from a named one in layout as well as naming - making the naming axis uninterpretable, since two
    things would vary at once. Every column sharing one description is what keeps the block contentless.
    """
    labeling = abstract_labels(_scm(), np.random.default_rng(0))
    assert labeling.context  # non-empty, so the `Context:` block is emitted
    assert set(labeling.column_schema) == set(labeling.node_to_label.values())  # every column described
    assert len(set(labeling.column_schema.values())) == 1  # ...with the very same sentence


def test_abstract_labels_is_deterministic_in_rng():
    scm = _scm()
    a = abstract_labels(scm, np.random.default_rng(7))
    b = abstract_labels(scm, np.random.default_rng(7))
    assert a.node_to_label == b.node_to_label


def test_abstract_labels_leak_no_role_information():
    labeling = abstract_labels(_scm(), np.random.default_rng(3))
    assert all(v.startswith("X") and v[1:].isdigit() for v in labeling.node_to_label.values())
