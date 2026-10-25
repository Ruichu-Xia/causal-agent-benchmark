import hashlib
import json

import networkx as nx
import numpy as np
import pytest

from causalagentbench.generation.causenet import (
    CauseNetLabeler,
    GroundingError,
    is_near_duplicate,
    naming_violations,
    one_way_relations,
)
from causalagentbench.scm.generate import draw_correlation, realize
from causalagentbench.scm.sample import sample
from causalagentbench.task.instance import build_instance
from causalagentbench.types import Role, Scaffold


@pytest.fixture
def scm():
    """A confounder-driver-mediator SCM small enough to name by hand."""
    scaffold = Scaffold(
        node_names=("C", "A", "M", "Y"),
        roles={"C": Role.CONFOUNDER, "A": Role.DRIVER_CONFOUNDED, "M": Role.MEDIATOR, "Y": Role.OUTCOME},
        edges=(("C", "A"), ("C", "Y"), ("A", "M"), ("A", "Y"), ("M", "Y")),
        drivers=("A",),
        outcome="Y",
    )
    return _realize(scaffold)


def _realize(scaffold):
    """An SCM on ``scaffold`` with a seeded random correlation matrix."""
    return realize(scaffold, draw_correlation(scaffold, np.random.default_rng(0)))


def test_every_edge_is_backed_by_a_direct_source_relation(scm):
    # Synthetic fixture for the matching algorithm, not a claimed real CauseNet excerpt.
    graph = nx.relabel_nodes(scm.scaffold.digraph(), lambda n: f"concept_{n}")
    labeler = CauseNetLabeler(graph)
    instance, key = build_instance(scm, sample(scm, 30, 0), 0, labeler=labeler)
    assert len(set(instance.columns)) == scm.n_nodes
    assert set(instance.columns) <= set(graph)
    assert all(graph.has_edge(key.node_to_label[a], key.node_to_label[b]) for a, b in scm.scaffold.edges)
    assert "precedes" not in instance.prompt
    assert "->" not in instance.prompt
    assert "confounder" not in instance.prompt
    assert "mediator" not in instance.prompt
    assert all(n not in instance.columns for n in scm.scaffold.node_names)


def test_grounding_never_falls_back_to_invented_names(scm):
    # Every possible concept is present, but one required relation is missing.
    graph = nx.relabel_nodes(scm.scaffold.digraph(), lambda n: f"concept_{n}")
    graph.remove_edge("concept_C", "concept_A")
    with pytest.raises(GroundingError):
        CauseNetLabeler(graph)(scm, np.random.default_rng(0))


def test_relation_matching_is_deterministic(scm):
    graph = nx.relabel_nodes(scm.scaffold.digraph(), lambda n: f"concept_{n}")
    labeler = CauseNetLabeler(graph)
    first = labeler(scm, np.random.default_rng(4))
    second = labeler(scm, np.random.default_rng(4))
    assert first == second


def _named(scm, names):
    """Return the scaffold relabeled with ``names`` (node -> concept) as a CauseNet-like graph."""
    return nx.relabel_nodes(scm.scaffold.digraph(), names)


NAMES = {"C": "smoking", "A": "tar_buildup", "M": "lung_damage", "Y": "cough"}


def test_two_way_relations_never_realize_a_hidden_edge(scm):
    graph = _named(scm, NAMES)
    graph.add_edge("tar_buildup", "smoking")  # CauseNet claims C-A both ways
    with pytest.raises(GroundingError):
        CauseNetLabeler(graph)(scm, np.random.default_rng(0))
    graph.add_edges_from([("air_pollution", "tar_buildup"), ("air_pollution", "cough")])
    labeling = CauseNetLabeler(graph)(scm, np.random.default_rng(0))
    assert labeling.node_to_label["C"] == "air_pollution"


def test_relations_against_a_hidden_path_are_rejected(scm):
    graph = _named(scm, NAMES)
    graph.add_edge("lung_damage", "smoking")  # hidden path C -> A -> M, CauseNet says M -> C
    with pytest.raises(GroundingError):
        CauseNetLabeler(graph)(scm, np.random.default_rng(0))
    graph.add_edges_from([("tar_buildup", "bronchitis"), ("bronchitis", "cough")])
    labeling = CauseNetLabeler(graph)(scm, np.random.default_rng(0))
    assert labeling.node_to_label["M"] == "bronchitis"


def test_relations_between_unconnected_variables_are_allowed():
    scaffold = Scaffold(
        node_names=("A", "U", "Co", "Y"),
        roles={"A": Role.DRIVER_CLEAN, "U": Role.CONFOUNDER, "Co": Role.COLLIDER, "Y": Role.OUTCOME},
        edges=(("A", "Y"), ("U", "Y"), ("A", "Co"), ("U", "Co")),
        drivers=("A",),
        outcome="Y",
    )
    scm = _realize(scaffold)
    graph = _named(scm, {"A": "aging", "U": "obesity", "Co": "hospital_stay", "Y": "diabetes"})
    graph.add_edge("aging", "obesity")  # no hidden path between A and U
    labeling = CauseNetLabeler(graph)(scm, np.random.default_rng(0))
    assert labeling.node_to_label["U"] == "obesity"


def test_near_duplicate_concepts_are_never_chosen_together(scm):
    graph = _named(scm, {**NAMES, "A": "smokings"})
    with pytest.raises(GroundingError):
        CauseNetLabeler(graph)(scm, np.random.default_rng(0))


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("infection", "infections", True),
        ("injury", "injuries", True),
        ("diabetes", "diabetes_mellitus", True),
        ("heart_disease", "heart_attack", False),
        ("stress", "distress", False),
        ("diseases", "disease", True),
        ("chronic_kidney_disease", "diseases", True),
        ("causes", "cause", True),
        ("headaches", "headache", True),
        ("gases", "gas", True),
        ("viruses", "virus", True),
        ("stresses", "stress", True),
    ],
)
def test_is_near_duplicate(first, second, expected):
    assert is_near_duplicate(first, second) is expected


def test_naming_violations_flags_each_rule(scm):
    graph = _named(scm, NAMES)
    assert naming_violations(graph, scm.scaffold, NAMES) == []
    graph.add_edge("cough", "tar_buildup")  # reverse of hidden edge A -> Y and against its order
    graph.add_edge("lung_damage", "smoking")  # against the hidden path C -> A -> M
    problems = naming_violations(graph, scm.scaffold, {**NAMES})
    assert any("not a one-way" in p for p in problems)
    assert any("lung_damage->smoking" in p for p in problems)
    duplicate = naming_violations(_named(scm, {**NAMES, "M": "smoking_s"}), scm.scaffold, {**NAMES, "M": "smoking_s"})
    assert any("near-duplicate" in p for p in duplicate)


def test_one_way_relations_drop_two_way_claims_and_self_loops():
    graph = nx.DiGraph([("a", "b"), ("b", "a"), ("b", "c"), ("c", "c")])
    assert set(one_way_relations(graph).edges) == {("b", "c")}


def test_excluded_pairs_never_realize_a_hidden_edge(scm):
    graph = _named(scm, NAMES)
    exclude = frozenset({frozenset(("smoking", "tar_buildup"))})
    with pytest.raises(GroundingError):
        CauseNetLabeler(graph, exclude=exclude)(scm, np.random.default_rng(0))
    graph.add_edges_from([("air_pollution", "tar_buildup"), ("air_pollution", "cough")])
    labeling = CauseNetLabeler(graph, exclude=exclude)(scm, np.random.default_rng(0))
    assert labeling.node_to_label["C"] == "air_pollution"


def test_excluded_relations_still_forbid_names_against_the_hidden_order(scm):
    graph = _named(scm, NAMES)
    graph.add_edge("cough", "smoking")  # against the hidden path C -> Y
    exclude = frozenset({frozenset(("cough", "smoking"))})
    with pytest.raises(GroundingError):
        CauseNetLabeler(graph, exclude=exclude)(scm, np.random.default_rng(0))


def test_from_path_records_the_dump_hash(tmp_path, scm):
    path = tmp_path / "relations.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"causal_relation": {"cause": {"concept": a}, "effect": {"concept": b}}, "support": 1})
            for a, b in _named(scm, NAMES).edges
        )
    )
    labeler = CauseNetLabeler.from_path(path)
    assert labeler.source == f"causenet:sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    assert labeler(scm, np.random.default_rng(0)).node_to_label == NAMES
