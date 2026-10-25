import dataclasses
import re

import networkx as nx
import numpy as np
import pytest

from causalagentbench.generation.causenet import CauseNetLabeler
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.gates import generate_scm
from causalagentbench.task.instance import Instance, build_instance, build_prompt, with_placeholder_names
from causalagentbench.task.labeling import placeholder_labels
from causalagentbench.types import SCM, Role


def _scm(seed: int = 0) -> SCM:
    """An accepted SCM."""
    scm, _ = generate_scm(np.random.SeedSequence(seed))
    return scm


def _instance(seed: int = 0, n: int = 128):
    """An accepted SCM, its canonical-order data, and its placeholder-named task."""
    scm = _scm(seed)
    data = sample(scm, n, seed=seed)
    return scm, data, build_instance(scm, data, seed=seed)


def _concept_labeler(scm: SCM) -> CauseNetLabeler:
    """A CauseNet labeler over a synthetic graph whose relations are exactly the SCM's own edges."""
    return CauseNetLabeler(nx.relabel_nodes(scm.scaffold.digraph(), lambda n: "concept" + n.replace("_", "x")))


def _rename(prompt: str, mapping: dict[str, str]) -> str:
    """Replace every whole-word column name in ``prompt`` by its image under ``mapping``."""
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(mapping, key=len, reverse=True))) + r")\b")
    return pattern.sub(lambda match: mapping[match.group(0)], prompt)


def test_instance_is_public_only():
    field_types = {f.name: f.type for f in dataclasses.fields(Instance)}
    assert set(field_types) == {"columns", "data", "outcome", "candidate_drivers", "prompt"}
    assert not any("SCM" in str(t) or "Role" in str(t) for t in field_types.values())


def test_shapes_and_partition():
    scm, _, (instance, _) = _instance()
    assert instance.data.shape == (128, scm.n_nodes)
    assert len(set(instance.columns)) == scm.n_nodes
    assert len(instance.candidate_drivers) == len(scm.scaffold.drivers)
    assert instance.outcome not in instance.candidate_drivers
    assert set(instance.other_columns) == set(instance.columns) - {instance.outcome, *instance.candidate_drivers}
    assert not instance.data.flags.writeable


def test_key_is_a_bijection_over_columns():
    scm, _, (instance, key) = _instance()
    assert set(key.label_to_node) == set(instance.columns)
    assert {key.label_to_node[c] for c in instance.columns} == set(scm.scaffold.node_names)
    assert all(key.label_to_node[label] == node for node, label in key.node_to_label.items())
    assert key.label_to_node[instance.outcome] == scm.scaffold.outcome
    assert {key.label_to_node[c] for c in instance.candidate_drivers} == set(scm.scaffold.drivers)
    assert key.producer == "placeholder"


def test_display_permutation_preserves_column_data():
    scm, data, (instance, key) = _instance()
    for j, label in enumerate(instance.columns):
        assert np.array_equal(instance.data[:, j], data[:, scm.scaffold.index(key.label_to_node[label])])


def test_build_instance_is_deterministic():
    scm, data, (first, _) = _instance(1)
    second, _ = build_instance(scm, data, seed=1)
    assert first.columns == second.columns
    assert np.array_equal(first.data, second.data)


def test_to_csv_header_matches_columns():
    _, _, (instance, _) = _instance()
    assert instance.to_csv().splitlines()[0] == ",".join(instance.columns)


def test_prompt_states_the_question_without_revealing_structure():
    scm, _, (instance, key) = _instance()
    prompt = instance.prompt
    assert prompt == build_prompt(instance.columns, instance.outcome, instance.candidate_drivers)
    assert f"Outcome: {instance.outcome}" in prompt
    assert f"Candidate drivers: {', '.join(instance.candidate_drivers)}" in prompt
    for word in ("confounder", "mediator", "decoy", "collider", "instrument", "upstream", "adjust for these"):
        assert word not in prompt.lower()
    for role in set(Role) - {Role.OUTCOME}:  # "outcome" is the question's own term
        assert role.value not in prompt
    assert "->" not in prompt and "→" not in prompt
    for parent, child in scm.scaffold.edges:
        assert f"{key.node_to_label[parent]} causes {key.node_to_label[child]}" not in prompt


def test_prompt_discloses_that_signs_and_sizes_are_synthetic():
    _, _, (instance, _) = _instance()
    assert "Coefficient signs\nand sizes are synthetic" in instance.prompt
    assert "simulated, not measured real-world effects" in instance.prompt


def test_real_and_placeholder_prompts_differ_only_in_column_names():
    scm, data, _ = _instance(2)
    named, named_key = build_instance(scm, data, seed=5, labeler=_concept_labeler(scm))
    blind, blind_key = build_instance(scm, data, seed=5, labeler=placeholder_labels)
    assert named_key.producer == "causenet" and blind_key.producer == "placeholder"
    # The label and shuffle RNGs are split from one seed, so both share the display order.
    assert [named_key.label_to_node[c] for c in named.columns] == [blind_key.label_to_node[c] for c in blind.columns]
    assert np.array_equal(named.data, blind.data)
    assert named.prompt != blind.prompt
    mapping = {named_key.node_to_label[node]: label for node, label in blind_key.node_to_label.items()}
    assert _rename(named.prompt, mapping) == blind.prompt


def test_with_placeholder_names_keeps_data_order_and_answer_key():
    scm, data, _ = _instance(2)
    task = build_instance(scm, data, seed=5, labeler=_concept_labeler(scm))
    renamed, renamed_key = with_placeholder_names(task, seed=0)
    instance, key = task
    assert renamed.data is instance.data
    assert renamed_key.scm is key.scm
    assert renamed_key.producer == "placeholder"
    assert sorted(renamed.columns) == sorted(f"X{i + 1}" for i in range(scm.n_nodes))
    assert [renamed_key.label_to_node[c] for c in renamed.columns] == [key.label_to_node[c] for c in instance.columns]
    assert renamed_key.label_to_node[renamed.outcome] == key.label_to_node[instance.outcome]
    assert [renamed_key.label_to_node[c] for c in renamed.candidate_drivers] == [
        key.label_to_node[c] for c in instance.candidate_drivers
    ]
    mapping = {key.node_to_label[node]: label for node, label in renamed_key.node_to_label.items()}
    assert _rename(instance.prompt, mapping) == renamed.prompt


def test_with_placeholder_names_is_deterministic_per_seed():
    _, _, task = _instance()
    assert with_placeholder_names(task, seed=4)[0].columns == with_placeholder_names(task, seed=4)[0].columns
    assert with_placeholder_names(task, seed=4)[0].columns != with_placeholder_names(task, seed=5)[0].columns


@pytest.mark.parametrize("seed", range(3))
def test_candidates_appear_in_display_order(seed):
    """Candidate order follows the shuffled columns, so a candidate's position carries no role."""
    _, _, (instance, _) = _instance(seed)
    positions = [instance.columns.index(c) for c in instance.candidate_drivers]
    assert positions == sorted(positions)
