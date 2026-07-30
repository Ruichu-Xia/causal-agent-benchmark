from causalagentbench.scm.gadgets import procedural_scaffold
from causalagentbench.scm.identifiability import unoriented_edges
from causalagentbench.types import Role, Scaffold

# The same small example graph as tests/scm/test_scaffold.py (confounder C, confounded driver A +
# mediator M, clean driver B, decoy Dz) - has exactly the mix of resolvable/unresolvable edges this
# module's tests need.
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


def test_example_has_exactly_the_known_unoriented_edges():
    """backlog.md §4: C-A, A-M, and C-Dz are unresolvable from data alone; nothing else in the example is."""
    assert unoriented_edges(_EXAMPLE) == frozenset({("C", "A"), ("A", "M"), ("C", "Dz")})


def test_a_collider_with_non_adjacent_parents_is_fully_oriented():
    scaffold = Scaffold(
        node_names=("X", "Y", "Z"),
        roles={"X": Role.DRIVER_CLEAN, "Y": Role.OUTCOME, "Z": Role.DRIVER_CLEAN},
        edges=(("X", "Y"), ("Z", "Y")),
        drivers=("X", "Z"),
        outcome="Y",
        correct_adjustment={"X": (), "Z": ()},
    )
    assert unoriented_edges(scaffold) == frozenset()


def test_a_chain_is_fully_unoriented():
    """No v-structure in a chain (Y has one parent, Z has one parent) - direction is unrecoverable."""
    scaffold = Scaffold(
        node_names=("X", "Y", "Z"),
        roles={"X": Role.DRIVER_CLEAN, "Y": Role.MEDIATOR, "Z": Role.OUTCOME},
        edges=(("X", "Y"), ("Y", "Z")),
        drivers=("X",),
        outcome="Z",
        correct_adjustment={"X": ()},
    )
    assert unoriented_edges(scaffold) == frozenset({("X", "Y"), ("Y", "Z")})


def test_confounded_driver_ambiguity_varies_across_composed_instances():
    """A confounded_driver's confounder edge is no longer fixed one way.

    §11's resolving modifier is drawn independently per instance - both still-ambiguous and resolved
    outcomes must occur across composed instances, discovered per instance via `unoriented_edges`, never
    declared by archetype alone.
    """
    outcomes = set()
    for seed in range(200):
        scaffold = procedural_scaffold(seed)
        unoriented = unoriented_edges(scaffold)
        for driver, adjustment in scaffold.correct_adjustment.items():
            if len(adjustment) != 1:
                continue  # multi_confounder (2 adjustment vars) or instrumented/clean (0) - not this archetype
            (confounder,) = adjustment
            outcomes.add((confounder, driver) in unoriented)
    assert outcomes == {True, False}
