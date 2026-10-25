import networkx as nx
import numpy as np
import pytest

from causalagentbench.benchmark import SPLITS, make_task, split_seed
from causalagentbench.generation.causenet import CauseNetLabeler, GroundingError, naming_violations
from causalagentbench.scm.identifiability import equivalence_class


def _state(seed: np.random.SeedSequence) -> tuple[int, ...]:
    """The state a seed hands to generators, which is what must differ between tasks."""
    return tuple(int(word) for word in seed.generate_state(4))


def test_split_seed_is_deterministic_and_disjoint_across_splits_and_indices():
    assert _state(split_seed("dev", 3)) == _state(split_seed("dev", 3))
    seeds = {_state(split_seed(split, i)) for split in ("train", "dev", "test") for i in range(5)}
    assert len(seeds) == 15


def test_private_split_requires_a_salt_and_only_it_accepts_one():
    assert _state(split_seed("private", 0, "a")) != _state(split_seed("private", 0, "b"))
    with pytest.raises(ValueError, match="salt"):
        split_seed("private", 0)
    with pytest.raises(ValueError, match="salt"):
        split_seed("dev", 0, "a")
    assert set(SPLITS) == {"train", "dev", "test", "private"}


@pytest.mark.parametrize(("split", "index"), [("validation", 0), ("dev", -1), ("dev", True), ("dev", 1.0)])
def test_split_seed_rejects_unknown_splits_and_bad_indices(split, index):
    with pytest.raises(ValueError):
        split_seed(split, index)


@pytest.fixture
def labeler(causenet_graph):
    """A CauseNet labeler over the synthetic concept graph."""
    return CauseNetLabeler(causenet_graph, source="causenet:sha256:synthetic")


def test_make_task_is_deterministic(labeler):
    first, first_key = make_task("dev", 0, labeler=labeler, n_rows=300)
    second, second_key = make_task("dev", 0, labeler=labeler, n_rows=300)
    assert first.columns == second.columns and first.prompt == second.prompt
    assert np.array_equal(first.data, second.data)
    assert np.array_equal(first_key.scm.W, second_key.scm.W)


def test_make_task_names_every_node_by_the_rules(labeler, causenet_graph):
    instance, key = make_task("dev", 1, labeler=labeler, n_rows=300)
    assert key.producer == "causenet:sha256:synthetic"
    assert set(instance.columns) <= set(causenet_graph)
    assert naming_violations(causenet_graph, key.scm.scaffold, dict(key.node_to_label)) == []
    assert instance.data.shape == (300, key.scm.n_nodes)
    assert len(instance.candidate_drivers) == 5
    assert tuple(sorted(key.scm.scaffold.edges)) in equivalence_class(key.scm.scaffold)


def test_make_task_tables_are_exact_at_float32_precision(labeler):
    """Datasets store float32, so the task is built on the table a frozen dataset reloads."""
    instance, _ = make_task("dev", 0, labeler=labeler, n_rows=300)
    assert np.array_equal(instance.data, instance.data.astype(np.float32).astype(np.float64))


def test_splits_produce_different_tasks(labeler):
    dev, _ = make_task("dev", 0, labeler=labeler, n_rows=300)
    test, _ = make_task("test", 0, labeler=labeler, n_rows=300)
    assert not np.array_equal(dev.data, test.data)


def test_make_task_fails_loudly_when_nothing_can_be_named():
    labeler = CauseNetLabeler(nx.DiGraph([("rain", "flood")]))
    with pytest.raises(GroundingError, match="could not name task"):
        make_task("dev", 0, labeler=labeler, n_rows=10)
