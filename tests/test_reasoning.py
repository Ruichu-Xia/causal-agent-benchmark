import networkx as nx
import numpy as np
import pytest

from causalagentbench.reasoning.dataset import build, nonsense_names
from causalagentbench.reasoning.families import FAMILIES, _relations
from causalagentbench.reasoning.grading import grade
from causalagentbench.reasoning.worlds import World, causal_effect, d_separated, observed_difference, valid_adjustment
from causalagentbench.scm.identifiability import equivalence_class
from causalagentbench.types import Role, Scaffold


def _graph(edges: list[tuple[int, int]], n: int) -> nx.DiGraph:
    """A DAG on nodes 0..n-1."""
    graph = nx.DiGraph(edges)
    graph.add_nodes_from(range(n))
    return graph


def test_effects_of_a_confounded_world_match_hand_computation():
    """Z -> X, Z -> Y, X -> Y: do() averages over P(Z); conditioning weights by P(Z | X)."""
    # node 0 = Z, 1 = X, 2 = Y; Y's parents are (Z, X): bit 0 is Z, bit 1 is X
    world = World(("z", "x", "y"), ((0, 1), (0, 2), (1, 2)), ((0.3,), (0.2, 0.8), (0.1, 0.5, 0.4, 0.9)))
    ate = 0.7 * (0.4 - 0.1) + 0.3 * (0.9 - 0.5)
    assert causal_effect(world, 1, 2) == pytest.approx(ate)
    p_z_given_x1 = 0.3 * 0.8 / (0.3 * 0.8 + 0.7 * 0.2)
    p_z_given_x0 = 0.3 * 0.2 / (0.3 * 0.2 + 0.7 * 0.8)
    p_y_x1 = p_z_given_x1 * 0.9 + (1 - p_z_given_x1) * 0.4
    p_y_x0 = p_z_given_x0 * 0.5 + (1 - p_z_given_x0) * 0.1
    assert observed_difference(world, 1, 2) == pytest.approx(p_y_x1 - p_y_x0)


@pytest.mark.parametrize(
    ("edges", "n", "z", "valid"),
    [
        ([(0, 1), (0, 2), (1, 2)], 3, set(), False),  # confounder Z=0 left open
        ([(0, 1), (0, 2), (1, 2)], 3, {0}, True),
        ([(1, 0), (0, 2)], 3, {0}, False),  # X=1 -> M=0 -> Y=2: adjusting for a mediator
        ([(1, 2), (1, 0), (2, 0)], 3, {0}, False),  # collider X -> C <- Y... and descendant of Y
        ([(1, 2), (1, 0)], 3, {0}, True),  # X -> W off the causal path: valid, though not back-door
        ([(3, 1), (3, 0), (4, 0), (4, 2), (1, 2)], 5, {0}, False),  # M-bias: conditioning on the collider
        ([(3, 1), (3, 0), (4, 0), (4, 2), (1, 2)], 5, {0, 3}, True),
    ],
)
def test_adjustment_criterion_known_cases(edges, n, z, valid):
    """Classic cases of the adjustment criterion for the effect of node 1 on node 2."""
    assert valid_adjustment(_graph(edges, n), 1, 2, z) is valid


def test_conditioning_on_a_collider_connects_its_parents():
    """X -> C <- Y: independent marginally, dependent given C."""
    graph = _graph([(0, 2), (1, 2)], 3)
    assert d_separated(graph, 0, 1, set())
    assert not d_separated(graph, 0, 1, {2})


def _members(edges: list[tuple[str, str]], nodes: tuple[str, ...]) -> list[dict]:
    """Structural facts of every DAG equivalent to ``edges``."""
    scaffold = Scaffold(nodes, dict.fromkeys(nodes, Role.CONFOUNDER), tuple(edges), (), nodes[0])
    return [_relations(member, nodes) for member in equivalence_class(scaffold)]


def test_chain_direction_is_unknown_but_a_collider_is_fixed():
    """A - B - C chain: 3 equivalent DAGs; A -> C <- B collider: 1."""
    chain = _members([("a", "b"), ("b", "c")], ("a", "b", "c"))
    assert len(chain) == 3
    assert not all(m[("cause", "a", "c")] for m in chain)
    assert any(m[("cause", "a", "c")] for m in chain)
    collider = _members([("a", "c"), ("b", "c")], ("a", "b", "c"))
    assert len(collider) == 1
    assert collider[0][("direct", "a", "c")] and collider[0][("common_effect", "a", "b")]


@pytest.mark.parametrize(
    ("kind", "answer", "reply", "score"),
    [
        ("yesno", "yes", "so...\nAnswer: Yes", 1.0),
        ("yesno", "yes", "**Answer: No**", 0.0),
        ("yesno", "no", "Answer: yes\nWait. Answer: no.", 1.0),
        ("number", "-0.12", "Answer: $-0.12$", 1.0),
        ("number", "0.12", "Answer: 0.13", 1.0),
        ("number", "0.12", "Answer: 0.14", 0.0),
        ("letters", "A,C", "Answer: C, A", 1.0),
        ("letters", "none", "Answer: None", 1.0),
        ("letters", "A,C", "Answer: A", 0.0),
        ("choice", "both", "Answer: Both", 1.0),
        ("choice", "1", "the answer is 1", 0.0),
    ],
)
def test_grade(kind, answer, reply, score):
    """The last "Answer:" line decides; numbers within the tolerance count."""
    assert grade(kind, answer, reply) == score


def test_build_is_deterministic_and_balances_yes_no():
    """The same seed gives the same tasks; yes/no families are near half yes."""
    first = build(200, 3, ["rain", "flood", "crop failure", "famine", "drought", "migration"])
    second = build(200, 3, ["rain", "flood", "crop failure", "famine", "drought", "migration"])
    assert first == second
    yes = [t["answer"] == "yes" for t in first if t["family"] in ("independence", "discovery")]
    assert 0.35 < np.mean(yes) < 0.65
    assert {t["family"] for t in first} == set(FAMILIES)


def test_nonsense_names_are_distinct():
    """Names within one task never repeat."""
    names = nonsense_names(np.random.default_rng(0), 6)
    assert len(set(names)) == 6


def _adjusted(world: World, x: int, y: int, z: tuple[int, ...]) -> float:
    """The adjustment formula sum_z P(z) (P(y=1 | x=1, z) - P(y=1 | x=0, z)), by enumeration."""
    from causalagentbench.reasoning.worlds import probability

    total = 0.0
    for values in np.ndindex(*(2,) * len(z)):
        stratum = dict(zip(z, (int(v) for v in values), strict=True))
        weight = probability(world, stratum) if z else 1.0
        total += weight * (
            probability(world, {y: 1}, {**stratum, x: 1}) - probability(world, {y: 1}, {**stratum, x: 0})
        )
    return total


def test_verifiers_agree_with_brute_force_probability():
    """Valid sets reproduce the causal effect exactly and invalid ones do not; d-separation matches independence."""
    from itertools import combinations

    from causalagentbench.reasoning.worlds import probability, random_world

    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(3, 6))
        world = random_world(rng, tuple(str(i) for i in range(n)), 0.5)
        graph = world.graph()
        x, y = sorted(int(v) for v in rng.choice(n, size=2, replace=False))
        others = [v for v in range(n) if v not in (x, y)]
        for k in range(len(others) + 1):
            for z in combinations(others, k):
                bias = _adjusted(world, x, y, z) - causal_effect(world, x, y)
                assert (abs(bias) < 1e-9) is valid_adjustment(graph, x, y, set(z))
                stratum_gaps = [
                    probability(world, {y: 1}, {**dict(zip(z, v, strict=True)), x: 1})
                    - probability(world, {y: 1}, {**dict(zip(z, v, strict=True)), x: 0})
                    for v in np.ndindex(*(2,) * len(z))
                ]
                independent = max(abs(g) for g in stratum_gaps) < 1e-9
                assert independent is d_separated(graph, x, y, set(z))
