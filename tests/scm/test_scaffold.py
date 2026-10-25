import networkx as nx
import pytest

from causalagentbench.types import Role, Scaffold

# A small hand-built graph (confounder C, confounded driver A with mediator M, clean driver B, decoy Dz,
# outcome Y and its child O) for Scaffold's own generic behavior.
_EXAMPLE = Scaffold(
    node_names=("C", "A", "B", "M", "Dz", "Y", "O"),
    roles={
        "C": Role.CONFOUNDER,
        "A": Role.DRIVER_CONFOUNDED,
        "B": Role.DRIVER_CLEAN,
        "M": Role.MEDIATOR,
        "Dz": Role.DECOY,
        "Y": Role.OUTCOME,
        "O": Role.OUTCOME_CHILD,
    },
    edges=(("C", "A"), ("C", "Y"), ("C", "Dz"), ("A", "M"), ("A", "Y"), ("M", "Y"), ("B", "Y"), ("Y", "O")),
    drivers=("A", "B", "Dz"),
    outcome="Y",
)


def _scaffold(**overrides) -> Scaffold:
    """``_EXAMPLE`` with some constructor arguments replaced."""
    fields = {
        "node_names": _EXAMPLE.node_names,
        "roles": dict(_EXAMPLE.roles),
        "edges": _EXAMPLE.edges,
        "drivers": _EXAMPLE.drivers,
        "outcome": _EXAMPLE.outcome,
    }
    return Scaffold(**{**fields, **overrides})


def test_example_is_a_dag():
    assert nx.is_directed_acyclic_graph(_EXAMPLE.digraph())
    assert set(_EXAMPLE.digraph().nodes) == set(_EXAMPLE.node_names)


def test_topological_order_places_parents_first():
    order = _EXAMPLE.topological_order()
    position = {name: i for i, name in enumerate(order)}
    assert sorted(order) == sorted(_EXAMPLE.node_names)
    for parent, child in _EXAMPLE.edges:
        assert position[parent] < position[child]


def test_parents_and_index_follow_the_declared_order():
    assert _EXAMPLE.parents("Y") == ("C", "A", "M", "B")
    assert _EXAMPLE.parents("C") == ()
    assert [_EXAMPLE.index(n) for n in _EXAMPLE.node_names] == list(range(len(_EXAMPLE.node_names)))


def test_roles_include_the_outcome_child_and_no_instrument():
    assert Role.OUTCOME_CHILD.value == "outcome_child"
    assert "INSTRUMENT" not in Role.__members__


def test_scaffold_rejects_a_cycle():
    with pytest.raises(ValueError, match="acyclic"):
        _scaffold(edges=(*_EXAMPLE.edges, ("O", "C")))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"roles": {"C": Role.CONFOUNDER}}, "roles"),
        ({"node_names": (*_EXAMPLE.node_names, "C")}, "duplicates"),
        ({"edges": (*_EXAMPLE.edges, ("C", "Q"))}, "unknown node"),
        ({"drivers": ("A", "Q")}, "drivers"),
        ({"outcome": "Q"}, "outcome"),
    ],
)
def test_scaffold_rejects_malformed_fields(overrides, message):
    with pytest.raises(ValueError, match=message):
        _scaffold(**overrides)
