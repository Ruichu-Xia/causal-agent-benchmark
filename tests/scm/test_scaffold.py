import networkx as nx
import pytest

from causalagentbench.scm.scaffold import CANONICAL
from causalagentbench.types import Role, Scaffold


def test_canonical_is_a_dag():
    assert nx.is_directed_acyclic_graph(CANONICAL.digraph())


def test_topological_order_places_parents_first():
    order = CANONICAL.topological_order()
    position = {name: i for i, name in enumerate(order)}
    for parent, child in CANONICAL.edges:
        assert position[parent] < position[child]


def test_role_lookup_is_unique_for_canonical_roles():
    assert len(CANONICAL.nodes_with_role(Role.DECOY)) == 1
    assert len(CANONICAL.nodes_with_role(Role.OUTCOME)) == 1
    assert len(CANONICAL.nodes_with_role(Role.MEDIATOR)) == 1


def test_scaffold_rejects_a_cycle():
    with pytest.raises(ValueError, match="acyclic"):
        Scaffold(
            node_names=("X", "Y"),
            roles={"X": Role.DRIVER_CLEAN, "Y": Role.OUTCOME},
            edges=(("X", "Y"), ("Y", "X")),
            drivers=("X",),
            outcome="Y",
            correct_adjustment={"X": ()},
        )


def test_scaffold_rejects_incomplete_roles():
    with pytest.raises(ValueError, match="roles"):
        Scaffold(
            node_names=("X", "Y"),
            roles={"X": Role.DRIVER_CLEAN},
            edges=(("X", "Y"),),
            drivers=("X",),
            outcome="Y",
            correct_adjustment={"X": ()},
        )
