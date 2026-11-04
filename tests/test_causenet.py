import bz2
import json

import networkx as nx
import numpy as np
import pytest

from causalagentbench.causenet import (
    CauseNetLabeler,
    GroundingError,
    held_out_concept,
    is_near_duplicate,
    load_causenet,
    naming_violations,
    one_way_relations,
)
from causalagentbench.scm import Scaffold

SCAFFOLD = Scaffold(
    node_names=("C", "A", "M", "Y"),
    edges=(("C", "A"), ("C", "Y"), ("A", "M"), ("A", "Y"), ("M", "Y")),
    outcome="Y",
)
"""A confounder-driver-mediator graph small enough to name by hand."""

NAMES = {"C": "smoking", "A": "tar_buildup", "M": "lung_damage", "Y": "cough"}


def _named(names: dict[str, str]) -> nx.DiGraph:
    """The scaffold relabeled with ``names`` (node -> concept), as a CauseNet-like graph."""
    return nx.relabel_nodes(SCAFFOLD.digraph(), names)


def _name(graph: nx.DiGraph, seed: int = 0) -> dict[str, str]:
    """Name the scaffold from ``graph``."""
    return CauseNetLabeler(graph)(SCAFFOLD, np.random.default_rng(seed))


def test_every_hidden_edge_is_a_causenet_relation():
    graph = nx.relabel_nodes(SCAFFOLD.digraph(), lambda n: f"concept_{n}")
    mapping = _name(graph)
    assert all(graph.has_edge(mapping[a], mapping[b]) for a, b in SCAFFOLD.edges)
    assert _name(graph, 4) == _name(graph, 4)


def test_naming_never_falls_back_to_invented_names():
    graph = _named(NAMES)
    graph.remove_edge("smoking", "tar_buildup")
    with pytest.raises(GroundingError):
        _name(graph)


def test_two_way_relations_never_realize_a_hidden_edge():
    graph = _named(NAMES)
    graph.add_edge("tar_buildup", "smoking")  # CauseNet claims C-A both ways
    with pytest.raises(GroundingError):
        _name(graph)
    graph.add_edges_from([("air_pollution", "tar_buildup"), ("air_pollution", "cough")])
    assert _name(graph)["C"] == "air_pollution"


def test_relations_against_a_hidden_path_are_rejected():
    graph = _named(NAMES)
    graph.add_edge("lung_damage", "smoking")  # hidden path C -> A -> M, CauseNet says M -> C
    with pytest.raises(GroundingError):
        _name(graph)
    graph.add_edges_from([("tar_buildup", "bronchitis"), ("bronchitis", "cough")])
    assert _name(graph)["M"] == "bronchitis"


def test_near_duplicate_concepts_are_never_chosen_together():
    with pytest.raises(GroundingError):
        _name(_named({**NAMES, "A": "smokings"}))


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("infection", "infections", True),
        ("injury", "injuries", True),
        ("diabetes", "diabetes_mellitus", True),
        ("heart_disease", "heart_attack", False),
        ("stress", "distress", False),
        ("headaches", "headache", True),
        ("gases", "gas", True),
    ],
)
def test_is_near_duplicate(first, second, expected):
    assert is_near_duplicate(first, second) is expected


def test_naming_violations_flags_each_rule():
    graph = _named(NAMES)
    assert naming_violations(graph, SCAFFOLD, NAMES) == []
    graph.add_edge("cough", "tar_buildup")  # reverse of hidden edge A -> Y
    graph.add_edge("lung_damage", "smoking")  # against the hidden path C -> A -> M
    problems = naming_violations(graph, SCAFFOLD, NAMES)
    assert any("not a one-way" in p for p in problems)
    assert any("lung_damage->smoking" in p for p in problems)


def test_one_way_relations_drop_two_way_claims_and_self_loops():
    graph = nx.DiGraph([("a", "b"), ("b", "a"), ("b", "c"), ("c", "c")])
    assert set(one_way_relations(graph).edges) == {("b", "c")}


@pytest.mark.parametrize("compressed", [False, True])
def test_load_causenet_preserves_direction_and_sums_support(tmp_path, compressed):
    load_causenet.cache_clear()
    path = tmp_path / ("relations.jsonl.bz2" if compressed else "relations.jsonl")
    records = [
        {"causal_relation": {"cause": {"concept": "smoking"}, "effect": {"concept": "lung_cancer"}}, "support": 4},
        {"causal_relation": {"cause": {"concept": "smoking"}, "effect": {"concept": "lung_cancer"}}, "support": 2},
    ]
    content = "\n".join(json.dumps(r) for r in records)
    path.write_bytes(bz2.compress(content.encode()) if compressed else content.encode())
    graph = load_causenet(str(path))
    assert graph["smoking"]["lung_cancer"]["weight"] == 6
    assert not graph.has_edge("lung_cancer", "smoking")


def test_held_out_concepts_are_about_a_fifth_and_stable():
    concepts = [f"concept_{i}" for i in range(5000)]
    share = sum(map(held_out_concept, concepts)) / len(concepts)
    assert 0.18 < share < 0.22
    assert [held_out_concept(c) for c in concepts[:50]] == [held_out_concept(c) for c in concepts[:50]]
