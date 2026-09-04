import bz2
import json

import numpy as np
import pytest

from causalagentbench.generation.grounding import (
    CauseNetMenus,
    concept_pool,
    format_menus,
    load_causenet,
    menus_for_hub,
    sample_menus,
)

# Synthetic CauseNet slice with every role represented around the heart_disease hub:
#   drivers      - direct causes of heart_disease
#   confounder   - age -> obesity (a common cause of two drivers), stress -> poor_diet likewise
#   mediator     - obesity -> arterial_plaque -> heart_disease (on the path)
#   instrument   - cigarette_tax -> smoking, with no edge to heart_disease (shifts a driver only)
#   decoy        - age -> gray_hair (an effect of a confounder that does not cause the outcome)
#   upstream     - socioeconomic_status -> stress (a distal cause of a confounder)
#   collider     - smoking -> cardiac_biomarker <- obesity (a common effect of two causes, not a cause)
_EDGES = [
    ("smoking", "heart_disease", 5),
    ("obesity", "heart_disease", 4),
    ("age", "heart_disease", 4),
    ("stress", "heart_disease", 3),
    ("alcohol", "heart_disease", 3),
    ("poor_diet", "heart_disease", 2),
    ("high_salt", "heart_disease", 2),
    ("arterial_plaque", "heart_disease", 2),
    ("family_history", "heart_disease", 1),
    ("obesity", "arterial_plaque", 3),
    ("age", "obesity", 2),
    ("stress", "poor_diet", 2),
    ("age", "gray_hair", 2),
    ("stress", "irritability", 1),
    ("cigarette_tax", "smoking", 2),
    ("nicotine_dependence", "smoking", 1),
    ("socioeconomic_status", "stress", 2),
    ("urban_living", "stress", 1),
    ("smoking", "cardiac_biomarker", 1),
    ("obesity", "cardiac_biomarker", 1),
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


def test_menus_for_hub_recovers_every_role_by_graph_position(dump):
    menus = menus_for_hub(load_causenet(dump), "heart_disease", base_size=15, cap=64)
    assert "smoking" in menus["driver"]
    assert "age" in menus["confounder"]
    assert "arterial_plaque" in menus["mediator"]
    assert "cigarette_tax" in menus["instrument"]
    assert "gray_hair" in menus["decoy"]
    assert "socioeconomic_status" in menus["upstream"]
    assert "cardiac_biomarker" in menus["collider"]
    # A confounder concept is a cause of the outcome too, so menus overlap - but each menu is internally
    # distinct, which is what the labeler relies on when drawing without replacement.
    for concepts in menus.values():
        assert len(set(concepts)) == len(concepts)


def test_concept_pool_is_deduplicated_and_filtered(dump):
    """The pool flattens overlapping menus and drops generic/malformed CauseNet artifacts."""
    menus = CauseNetMenus("heart_disease", menus_for_hub(load_causenet(dump), "heart_disease", base_size=15, cap=64))
    pool = concept_pool(menus, limit=100)
    assert len(set(pool)) == len(pool)  # deduplicated across menus
    assert "smoking" in pool
    # menus_for_hub returns 'age' under both driver and confounder; the pool lists it once.
    assert sum(c == "age" for c in pool) <= 1


def test_sample_menus_picks_a_hub_with_a_rich_enough_pool(dump):
    menus = sample_menus(dump, np.random.default_rng(0), min_pool=5)
    assert menus.outcome == "heart_disease"
    assert len(concept_pool(menus, limit=5)) >= 5


def test_sample_menus_raises_without_a_hub(dump):
    with pytest.raises(ValueError, match="no outcome hub"):
        sample_menus(dump, np.random.default_rng(0), min_outcome_causes=99)


def test_sample_menus_raises_when_no_hub_has_enough_concepts(dump):
    with pytest.raises(ValueError, match="usable concepts"):
        sample_menus(dump, np.random.default_rng(0), min_pool=500)


def test_format_menus_is_one_line_with_the_outcome_and_pool(dump):
    text = format_menus(sample_menus(dump, np.random.default_rng(0), min_pool=5), limit=8)
    assert "heart_disease" in text and "related real concepts" in text
    assert "\n" not in text
