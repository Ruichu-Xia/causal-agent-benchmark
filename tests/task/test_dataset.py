import dataclasses
import itertools
import json

import networkx as nx
import numpy as np
import pytest

from causalagentbench.generation.causenet import CauseNetLabeler
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.gates import generate_scm
from causalagentbench.task.dataset import CANARY, DATASET_VERSION, concept_pairs, load_dataset, save_dataset
from causalagentbench.task.instance import build_instance


def _tasks(seeds=(0, 1)):
    """Accepted tasks named by a synthetic CauseNet graph built from each SCM's own edges.

    The tables are rounded to float32 precision, as ``make_task`` does, since datasets store float32.
    """
    tasks = []
    for seed in seeds:
        scm, _ = generate_scm(np.random.SeedSequence(seed))
        graph = nx.relabel_nodes(scm.scaffold.digraph(), lambda n: "concept" + n.replace("_", "x"))
        labeler = CauseNetLabeler(graph, source="causenet:sha256:test")
        data = sample(scm, 200, seed).astype(np.float32).astype(np.float64)
        tasks.append(build_instance(scm, data, seed, labeler=labeler))
    return tasks


def test_round_trip_reproduces_every_public_and_hidden_field(tmp_path):
    tasks = _tasks()
    save_dataset(tmp_path / "dev", tasks, split="dev")
    loaded = load_dataset(tmp_path / "dev")
    assert len(loaded) == len(tasks)
    for (instance, key), (got, got_key) in zip(tasks, loaded, strict=True):
        assert got.columns == instance.columns
        assert got.outcome == instance.outcome
        assert got.candidate_drivers == instance.candidate_drivers
        assert got.prompt == instance.prompt
        np.testing.assert_array_equal(got.data, instance.data)
        scaffold, got_scaffold = key.scm.scaffold, got_key.scm.scaffold
        assert got_scaffold.node_names == scaffold.node_names
        assert got_scaffold.edges == scaffold.edges
        assert dict(got_scaffold.roles) == dict(scaffold.roles)
        assert got_scaffold.drivers == scaffold.drivers
        assert got_scaffold.outcome == scaffold.outcome
        np.testing.assert_array_equal(got_key.scm.sigma, key.scm.sigma)
        np.testing.assert_allclose(got_key.scm.W, key.scm.W, atol=1e-12)
        np.testing.assert_allclose(got_key.scm.tau, key.scm.tau, atol=1e-12)
        assert dict(got_key.node_to_label) == dict(key.node_to_label)
        assert dict(got_key.label_to_node) == dict(key.label_to_node)
        assert got_key.producer == key.producer == "causenet:sha256:test"


def test_tables_are_stored_as_float32_and_reloaded_as_float64(tmp_path):
    save_dataset(tmp_path / "dev", _tasks((0,)), split="dev")
    with np.load(tmp_path / "dev.npz") as tables:
        assert [tables[name].dtype for name in tables.files] == [np.float32]
    assert load_dataset(tmp_path / "dev")[0][0].data.dtype == np.float64


def test_the_saved_prompt_is_frozen_verbatim(tmp_path):
    instance, key = _tasks((0,))[0]
    frozen = dataclasses.replace(instance, prompt="x")
    save_dataset(tmp_path / "frozen", [(frozen, key)], split="test")
    assert load_dataset(tmp_path / "frozen")[0][0].prompt == "x"


def test_json_carries_version_canary_and_split(tmp_path):
    save_dataset(tmp_path / "dev", _tasks((0,)), split="dev")
    text = (tmp_path / "dev.json").read_text(encoding="utf-8")
    blob = json.loads(text)
    assert CANARY in text
    assert blob["canary"] == CANARY
    assert blob["version"] == DATASET_VERSION
    assert blob["split"] == "dev"


def test_load_accepts_the_stem_or_either_file(tmp_path):
    save_dataset(tmp_path / "dev", _tasks((0,)), split="dev")
    for path in (tmp_path / "dev", tmp_path / "dev.json", tmp_path / "dev.npz"):
        assert len(load_dataset(path)) == 1


@pytest.mark.parametrize("existing", [".json", ".npz"])
def test_save_refuses_to_overwrite_either_file(tmp_path, existing):
    (tmp_path / f"dev{existing}").write_text("")
    with pytest.raises(FileExistsError):
        save_dataset(tmp_path / "dev", _tasks((0,)), split="dev")


def test_save_refuses_an_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_dataset(tmp_path / "empty", [], split="dev")


def _edit_json(stem, edit) -> None:
    """Apply ``edit`` to a saved dataset's JSON blob in place."""
    path = stem.with_suffix(".json")
    blob = json.loads(path.read_text(encoding="utf-8"))
    edit(blob)
    path.write_text(json.dumps(blob), encoding="utf-8")


def test_load_rejects_another_dataset_version(tmp_path):
    stem = tmp_path / "dev"
    save_dataset(stem, _tasks((0,)), split="dev")
    _edit_json(stem, lambda blob: blob.update(version="causalagentbench-v1"))
    with pytest.raises(ValueError, match="unsupported dataset version"):
        load_dataset(stem)


def test_load_rechecks_the_graph_against_sigma(tmp_path):
    """A hidden graph that no longer matches the frozen correlation matrix fails loudly on load."""
    stem = tmp_path / "dev"
    save_dataset(stem, _tasks((0,)), split="dev")
    _edit_json(stem, lambda blob: blob["records"][0]["graph"]["edges"].pop())
    with pytest.raises(ValueError, match="not Markov"):
        load_dataset(stem)


def test_concept_pairs_lists_every_pair_of_columns_used_together(tmp_path):
    tasks = _tasks()
    pairs = concept_pairs(tasks)
    expected = {frozenset(p) for instance, _ in tasks for p in itertools.combinations(instance.columns, 2)}
    assert pairs == expected
    a, b = tasks[0][0].columns[:2]
    assert frozenset((a, b)) in pairs and frozenset((b, a)) in pairs
