"""Tests for ``generation/generate.py`` - the train-data path.

Covers generate_instance/generate_instances/alpha_sweep. Named ``test_instance_generation`` rather than
``test_generate`` to avoid colliding with tests/scm/test_generate.py: this tree has no __init__.py files,
so pytest's default import mode requires every test file to have a globally unique basename.
"""

import numpy as np

from causalagentbench.generation.generate import alpha_sweep_with, generate_instance, generate_instances
from causalagentbench.generation.labeler import GRID
from causalagentbench.scm.identifiability import unidentifiable_drivers
from causalagentbench.scoring.gates import generate_gated_procedural
from causalagentbench.task.labeling import abstract_labels
from causalagentbench.types import Role


def _stub_triples_json(scm) -> str:
    """A valid propose_triples response for ``scm``.

    One coherent, distinct name per node/kind, keyed by the same v0..vk ids ``_generator_prompt`` uses.
    """
    entries = []
    for i in range(len(scm.scaffold.node_names)):
        entries.append(
            f'"v{i}": {{'
            f'"aligned": {{"name": "n{i}_aligned", "description": "a standardized measure"}}, '
            f'"neutral": {{"name": "n{i}_neutral", "description": "a record identifier"}}, '
            f'"misaligned": {{"name": "n{i}_misaligned", "description": "a standardized measure"}}}}'
        )
    return '{"domain": "study", "context": "one row per unit", "nodes": {' + ",".join(entries) + "}}"


class _Stub:
    """A ``LabelProposer`` returning canned responses in order - same shape as test_labeler.py's own."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[str] = []

    def propose(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


def _grounding(_scm, _rng) -> str:
    return "grounding"


# ---- generate_instance / generate_instances ------------------------------------------------------------


def test_generate_instance_produces_a_labeled_instance():
    instance, key = generate_instance(0, abstract_labels, n_samples=64)
    assert instance.data.shape[0] == 64
    assert len(instance.candidate_drivers) == len(key.scm.scaffold.drivers)
    assert all(label.startswith("X") for label in instance.columns)  # abstract_labels applied


def test_generate_instance_is_deterministic():
    a, _ = generate_instance(0, abstract_labels, n_samples=32)
    b, _ = generate_instance(0, abstract_labels, n_samples=32)
    assert a.columns == b.columns
    assert np.array_equal(a.data, b.data)


def test_generate_instance_respects_n_drivers():
    for n_drivers in (3, 5):
        _instance, key = generate_instance(0, abstract_labels, n_samples=32, n_drivers=n_drivers)
        decoys = key.scm.scaffold.nodes_with_role(Role.DECOY)
        assert len([d for d in key.scm.scaffold.drivers if d not in decoys]) == n_drivers


def test_generate_instance_respects_resolve_probability():
    """Mirrors scoring/gates.py's own resolve_probability test - the parameter survives this call chain too."""
    _instance, key = generate_instance(0, abstract_labels, n_samples=32, n_drivers=6, resolve_probability=1.0)
    ambiguous = set(unidentifiable_drivers(key.scm.scaffold)) & set(key.scm.scaffold.non_decoy_drivers())
    assert len(ambiguous) == 1


def test_generate_instances_streams_n_independent_instances():
    instances = list(generate_instances(3, abstract_labels, seed=0, n_samples=32))
    assert len(instances) == 3
    node_name_sets = {key.scm.scaffold.node_names for _, key in instances}
    assert len(node_name_sets) > 1  # independent seeds draw different topologies, not identical clones


def test_generate_instances_is_deterministic_in_seed():
    a = list(generate_instances(3, abstract_labels, seed=7, n_samples=32))
    b = list(generate_instances(3, abstract_labels, seed=7, n_samples=32))
    assert [inst.columns for inst, _ in a] == [inst.columns for inst, _ in b]


# ---- alpha_sweep_with (the DI core; alpha_sweep is a thin string-based wrapper over it, untested
# directly here for the same reason llm_labeler isn't - see LLMLabeler in test_labeler.py) ---------------


def test_alpha_sweep_with_builds_paired_skins_sharing_the_same_scm_and_data():
    scm, _ = generate_gated_procedural(0)
    proposer = _Stub([_stub_triples_json(scm)])

    skins = alpha_sweep_with(0, GRID, proposer=proposer, grounding=_grounding, producer="p")

    assert [level for level, _inst, _key in skins] == list(GRID)
    first_data = skins[0][1].data
    first_order = [skins[0][2].label_to_node[c] for c in skins[0][1].columns]
    for _level, instance, key in skins:
        assert np.array_equal(instance.data, first_data)  # same SCM + data across every level
        node_order = [key.label_to_node[c] for c in instance.columns]
        assert node_order == first_order  # same display order - only the names differ across levels
        assert key.scm.scaffold.node_names == scm.scaffold.node_names


def test_alpha_sweep_with_levels_map_to_the_right_alignment_and_producer():
    scm, _ = generate_gated_procedural(0)
    proposer = _Stub([_stub_triples_json(scm)])

    skins = alpha_sweep_with(0, GRID, proposer=proposer, grounding=_grounding, producer="llm:stub-model")

    for level, _instance, key in skins:
        assert key.provenance.alignment == level
        assert key.provenance.producer == "llm:stub-model"
        assert key.provenance.domain == "study"


def test_alpha_sweep_with_amortizes_one_proposal_call_across_every_level():
    scm, _ = generate_gated_procedural(0)
    proposer = _Stub([_stub_triples_json(scm)])

    alpha_sweep_with(0, GRID, proposer=proposer, grounding=_grounding, producer="p")

    assert len(proposer.calls) == 1  # one call, reused to construct every level (no critic configured)


def test_alpha_sweep_with_respects_n_drivers_and_resolve_probability():
    scm, _ = generate_gated_procedural(0, n_drivers=6, resolve_probability=1.0)
    proposer = _Stub([_stub_triples_json(scm)])

    skins = alpha_sweep_with(
        0, GRID, proposer=proposer, grounding=_grounding, producer="p", n_drivers=6, resolve_probability=1.0
    )

    key = skins[0][2]
    decoys = key.scm.scaffold.nodes_with_role(Role.DECOY)
    assert len([d for d in key.scm.scaffold.drivers if d not in decoys]) == 6
    ambiguous = set(unidentifiable_drivers(key.scm.scaffold)) & set(key.scm.scaffold.non_decoy_drivers())
    assert len(ambiguous) == 1
