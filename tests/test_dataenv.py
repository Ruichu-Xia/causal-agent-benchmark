import itertools

import networkx as nx
import numpy as np
import pytest

from causalagentbench.dataenv import sandbox
from causalagentbench.dataenv.dataset import to_record, world_from_dict, world_to_dict
from causalagentbench.dataenv.environment import DataEnv
from causalagentbench.dataenv.families import (
    FAMILIES,
    TOLERANCE,
    Phrases,
    Spec,
    _ancestry,
    _member_effect,
    _number,
    fisher_z_p,
)
from causalagentbench.dataenv.world import cpdag, pc_cpdag
from causalagentbench.generation.causenet import CauseNetLabeler
from causalagentbench.reasoning.grading import grade
from causalagentbench.reasoning.shortcuts import confounder_rule
from causalagentbench.reasoning.worlds import d_separated, valid_adjustment


@pytest.fixture(scope="module")
def labeler() -> CauseNetLabeler:
    """A labeler over a random sparse one-way relation graph on 300 made-up concepts."""
    rng = np.random.default_rng(0)
    words = [f"concept{i}" for i in range(300)]
    graph = nx.DiGraph((words[i], words[j]) for i, j in itertools.combinations(range(300), 2) if rng.random() < 0.3)
    return CauseNetLabeler(graph)


@pytest.fixture(scope="module")
def tasks(labeler: CauseNetLabeler) -> dict[str, list]:
    """Four tasks of every family, small graphs."""
    return {
        family: [
            make(np.random.default_rng([7, k]), Phrases(np.random.default_rng(k), False), Spec(), labeler)
            for k in range(4)
        ]
        for family, make in FAMILIES.items()
    }


def test_independence_answer_and_cell_follow_the_world(tasks):
    """The answer is d-separation in the true DAG, and the names-only and marginal rules are right as recorded."""
    for task in tasks["independence"]:
        world, meta = task.world, task.meta
        x, y, z = meta["x"], meta["y"], set(meta["z"])
        independent = d_separated(world.graph(), x, y, z)
        assert independent == meta["independent"]
        assert task.answer == ("yes" if independent == (meta["variant"] == "independent") else "no")
        names = world.name_graph()
        names_independent = not (nx.has_path(names, x, y) or nx.has_path(names, y, x))
        assert (names_independent == independent) == meta["names_right"]
        assert (d_separated(world.graph(), x, y, set()) == independent) == meta["marginal_right"]
        p = fisher_z_p(world.data(), x, y, sorted(z))
        assert p > 0.05 if independent else p < 0.01


def test_structure_answer_and_cell_follow_the_world(tasks):
    """The keyed letter is the true ancestry, and the cell says which source settles it."""
    for task in tasks["structure"]:
        world, meta = task.world, task.meta
        truth = _ancestry(world.graph(), meta["x"], meta["y"])
        assert truth == meta["truth"] and task.answer == "abc"[meta["order"].index(truth)]
        determined = len({_ancestry(m, meta["x"], meta["y"]) for m in world.member_graphs()}) == 1
        names_right = _ancestry(world.name_graph(), meta["x"], meta["y"]) == truth
        expected = {"name_critical": (False, True), "data_critical": (True, False), "both": (True, True)}[meta["cell"]]
        assert (determined, names_right) == expected


def test_adjustment_answer_and_cells_follow_the_world(tasks):
    """Each set's validity, data-determinedness and confounder-rule agreement are as recorded."""
    for task in tasks["adjustment"]:
        world, meta = task.world, task.meta
        x, y = meta["x"], meta["y"]
        valid = []
        for z, cell in zip(meta["sets"], meta["cells"], strict=True):
            ok = valid_adjustment(world.graph(), x, y, set(z))
            valid.append(ok)
            assert ok == cell["valid"]
            assert (len({valid_adjustment(m, x, y, set(z)) for m in world.member_graphs()}) == 1) == cell["determined"]
            assert (confounder_rule(world.graph(), x, y, set(z)) == ok) == cell["confounder_right"]
        if meta["variant"] == "single":
            assert task.answer == ("yes" if valid[0] else "no")
        else:
            keyed = {(True, True): "both", (True, False): "1", (False, True): "2", (False, False): "neither"}
            assert task.answer == keyed[(valid[0], valid[1])]


def test_effect_answer_and_cell_follow_the_world(tasks):
    """The answer is the true total effect; name-criticality and the unadjusted rule are as recorded."""
    for task in tasks["effect"]:
        world, meta = task.world, task.meta
        x, y = meta["x"], meta["y"]
        truth = world.total_effect(x, y)
        assert task.answer == _number(truth)
        spread = max(abs(_member_effect(world, m, x, y) - truth) for m in world.member_graphs())
        assert spread > 0.2 if meta["name_critical"] else spread <= 0.02
        miss = abs(world.sigma[x, y] - truth)
        assert miss <= TOLERANCE if meta["unadjusted_right"] else miss > 0.2
        others = [v for v in range(world.n) if v not in (x, y)]
        all_miss = abs(_member_effect(world, nx.DiGraph([(v, x) for v in others]), x, y) - truth)
        assert all_miss <= TOLERANCE if meta["adjust_all_right"] else all_miss > 0.2


def test_true_effect_matches_a_regression_on_the_parents(tasks):
    """In the population, regressing on the treatment's true parents gives the total effect."""
    for task in tasks["effect"]:
        world, x, y = task.world, task.meta["x"], task.meta["y"]
        assert _member_effect(world, world.graph(), x, y) == pytest.approx(world.total_effect(x, y), abs=1e-9)


def test_world_data_regenerate_recover_the_cpdag_and_round_trip(tasks):
    """Rows regenerate to the hashed rows, PC on them gives the true CPDAG, and the record round-trips."""
    world = tasks["structure"][0].world
    rows = world.data()
    assert rows.shape == (2000, world.n)
    assert pc_cpdag(rows) == cpdag(world.edges, world.n)
    again = world_from_dict(world_to_dict(world))
    assert again.edges == world.edges and again.members == world.members and np.array_equal(again.sigma, world.sigma)
    assert np.array_equal(again.data(), rows)


def test_names_never_point_against_the_causal_order(tasks):
    """Every true edge is a names relation in its direction, and no relation runs against a causal path."""
    for task in tasks["structure"] + tasks["adjustment"]:
        world = task.world
        names, graph = set(world.name_edges), world.graph()
        assert set(world.edges) <= names
        assert not any(nx.has_path(graph, b, a) for a, b in names)


def test_grading_accepts_a_letter_and_a_number_within_tolerance():
    """Letters parse alone (not as the start of "Both"); numbers use the task's tolerance."""
    assert grade("letter", "b", "work\nAnswer: B") == 1.0
    assert grade("letter", "b", "Answer: Both") == 0.0
    assert grade("number", "0.30", "Answer: 0.39", 0.10) == 1.0
    assert grade("number", "0.30", "Answer: 0.41", 0.10) == 0.0


def test_python_session_keeps_state_reports_errors_and_restarts_after_a_timeout(monkeypatch):
    """Variables persist, a final expression prints, errors return a traceback, and a hang restarts the session."""
    monkeypatch.setattr(sandbox, "CALL_SECONDS", 3.0)
    rows = np.random.default_rng(0).standard_normal((50, 2))
    session = sandbox.PythonSession(rows, ["a", "b"])
    try:
        assert session.run("x = 2\nx + 1").strip() == "3"
        assert session.run("df.shape").strip() == "(50, 2)"
        assert "ZeroDivisionError" in session.run("1 / 0")
        assert "did not finish" in session.run("while True:\n    pass")
        assert "NameError" in session.run("x")
    finally:
        session.close()


def test_data_env_exposes_one_tool_enforces_its_budget_and_grades(tasks):
    """``python`` is the only tool; the budget is enforced; the final reply is graded against the task."""
    public = [name for name in dir(DataEnv) if not name.startswith("_") and callable(getattr(DataEnv, name))]
    assert sorted(public) == ["python", "reset"]
    task = tasks["structure"][0]
    record = to_record(task, "toy-0", "iid")
    env = DataEnv([record], max_tool_calls=2)
    assert env.reset(index=0) == task.prompt
    assert "2000" in env.python("len(df)")
    env.python("1")
    with pytest.raises(ValueError, match="budget"):
        env.python("1")
    assert env._score(f"Answer: {task.answer.upper()}") == 1.0
    assert env.info["correct"] and env.info["tool_calls"] == 2
    env.reset(task=record)
    assert env._score("no answer here") == 0.0 and not env.info["answered"]
