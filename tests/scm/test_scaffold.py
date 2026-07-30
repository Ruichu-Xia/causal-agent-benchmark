import networkx as nx
import pytest

from causalagentbench.types import Role, Scaffold

# A small, hand-built example graph (confounder C, confounded driver A + mediator M, clean driver B, decoy
# Dz) - not a production scaffold (those are procedurally composed, scm/gadgets.py), just a convenient,
# readable fixture for testing Scaffold's own generic behavior.
_EXAMPLE = Scaffold(
    node_names=("C", "A", "B", "M", "Dz", "Y"),
    roles={
        "C": Role.CONFOUNDER,
        "A": Role.DRIVER_CONFOUNDED,
        "B": Role.DRIVER_CLEAN,
        "M": Role.MEDIATOR,
        "Dz": Role.DECOY,
        "Y": Role.OUTCOME,
    },
    edges=(("C", "A"), ("C", "Y"), ("C", "Dz"), ("A", "M"), ("A", "Y"), ("M", "Y"), ("B", "Y")),
    drivers=("A", "B", "Dz"),
    outcome="Y",
    correct_adjustment={"A": ("C",), "B": (), "Dz": ("C",)},
)


def test_example_is_a_dag():
    assert nx.is_directed_acyclic_graph(_EXAMPLE.digraph())


def test_topological_order_places_parents_first():
    order = _EXAMPLE.topological_order()
    position = {name: i for i, name in enumerate(order)}
    for parent, child in _EXAMPLE.edges:
        assert position[parent] < position[child]


def test_role_lookup_is_unique_for_each_singleton_role():
    assert len(_EXAMPLE.nodes_with_role(Role.DECOY)) == 1
    assert len(_EXAMPLE.nodes_with_role(Role.OUTCOME)) == 1
    assert len(_EXAMPLE.nodes_with_role(Role.MEDIATOR)) == 1


def test_non_decoy_drivers_excludes_only_decoys():
    assert _EXAMPLE.non_decoy_drivers() == ("A", "B")


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
