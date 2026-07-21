import bz2
import json

import numpy as np
import pytest

from causalagentbench.generation.grounding import format_menus, load_causenet, sample_menus
from causalagentbench.scm.scaffold import CANONICAL

# Synthetic CauseNet slice: heart_disease is the effect hub, with an embedded confounder
# (age -> obesity + age -> heart_disease), mediator (obesity -> arterial_plaque -> heart_disease),
# and decoy (age -> gray_hair, no edge to heart_disease).
_EDGES = [
    ("smoking", "heart_disease", 5),
    ("obesity", "heart_disease", 4),
    ("stress", "heart_disease", 3),
    ("alcohol", "heart_disease", 3),
    ("poor_diet", "heart_disease", 2),
    ("high_salt", "heart_disease", 2),
    ("age", "heart_disease", 4),
    ("family_history", "heart_disease", 1),
    ("arterial_plaque", "heart_disease", 2),
    ("obesity", "arterial_plaque", 3),
    ("age", "obesity", 2),
    ("age", "gray_hair", 2),
]


@pytest.fixture
def dump(tmp_path) -> str:
    path = tmp_path / "causenet.jsonl.bz2"
    with bz2.open(path, "wt", encoding="utf-8") as handle:
        for cause, effect, support in _EDGES:
            record = {
                "causal_relation": {"cause": {"concept": cause}, "effect": {"concept": effect}},
                "sources": [{} for _ in range(support)],
            }
            handle.write(json.dumps(record) + "\n")
    return str(path)


def test_load_causenet_builds_weighted_directed_graph(dump):
    graph = load_causenet(dump)
    assert graph["smoking"]["heart_disease"]["weight"] == 5
    assert graph.in_degree("heart_disease") == 9


def test_sample_menus_recovers_roles_by_graph_position(dump):
    menus = sample_menus(dump, CANONICAL, np.random.default_rng(0))
    assert menus.outcome == "heart_disease"
    strengths = [s for _, s in menus.drivers]
    assert max(strengths) == 1.0
    assert all(0.0 <= s <= 1.0 for s in strengths)
    assert "age" in menus.confounders
    assert "arterial_plaque" in menus.mediators
    assert "gray_hair" in menus.decoys


def test_sample_menus_raises_without_a_hub(dump):
    with pytest.raises(ValueError, match="outcome hub"):
        sample_menus(dump, CANONICAL, np.random.default_rng(0), min_outcome_causes=99)


def test_format_menus_is_a_nonempty_line(dump):
    text = format_menus(sample_menus(dump, CANONICAL, np.random.default_rng(0)))
    assert "heart_disease" in text
