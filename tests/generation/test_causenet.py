import networkx as nx
import numpy as np
import pytest

from causalagentbench.generation.causenet import CauseNetLabeler, GroundingError
from causalagentbench.scm.generate import generate
from causalagentbench.scm.sample import sample
from causalagentbench.task.instance import build_instance
from causalagentbench.types import Role, Scaffold


@pytest.fixture
def scm():
    scaffold = Scaffold(
        node_names=("C", "A", "M", "Y"),
        roles={"C": Role.CONFOUNDER, "A": Role.DRIVER_CONFOUNDED, "M": Role.MEDIATOR, "Y": Role.OUTCOME},
        edges=(("C", "A"), ("C", "Y"), ("A", "M"), ("A", "Y"), ("M", "Y")),
        drivers=("A",),
        outcome="Y",
        correct_adjustment={"A": ("C",)},
    )
    return generate(0, scaffold)


def test_every_edge_is_backed_by_a_direct_source_relation(scm):
    # Synthetic fixture for the matching algorithm, not a claimed real CauseNet excerpt.
    graph = nx.relabel_nodes(scm.scaffold.digraph(), lambda n: f"concept_{n}")
    labeler = CauseNetLabeler(graph)
    instance, key = build_instance(scm, sample(scm, 30, 0), 0, labeler=labeler)
    assert len(set(instance.columns)) == scm.n_nodes
    assert set(instance.columns) <= set(graph)
    assert all(graph.has_edge(key.node_to_label[a], key.node_to_label[b]) for a, b in scm.scaffold.edges)
    assert instance.column_schema == {}
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
