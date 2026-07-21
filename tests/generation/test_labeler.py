import json

import numpy as np
import pytest

from causalagentbench import build_instance, generate_gated
from causalagentbench.generation.labeler import (
    GRID,
    LLMLabeler,
    NamingTriples,
    _informative_order,
    _pre_check,
    construct_labeling,
    propose_triples,
)
from causalagentbench.scm.sample import sample

# CANONICAL node order: C, A, B, M, Dz, Y -> ids v0..v5.
_GOOD = {
    "C": ("age_years", "record_code", "arterial_stiffness"),
    "A": ("ldl_cholesterol", "lab_batch", "appointment_delay"),
    "B": ("systolic_pressure", "form_version", "barometric_pressure"),
    "M": ("cardiorespiratory_fitness", "panel_id", "family_history"),
    "Dz": ("clinic_room", "badge_reading", "coronary_calcium"),
    "Y": ("cardiovascular_risk", "outcome_alt_b", "outcome_alt_c"),
}


def _scm(seed: int = 0):
    scm, _ = generate_gated(seed)
    return scm


def _triples(scm, table=_GOOD) -> NamingTriples:
    choices = {
        node: {
            k: (nm, f"{nm} standardized measure")
            for k, nm in zip(("aligned", "neutral", "misaligned"), names, strict=True)
        }
        for node, names in table.items()
    }
    return NamingTriples(domain="study", context="one row per unit", choices=choices)


def _triples_json(scm, table=_GOOD) -> str:
    ids = {n: f"v{i}" for i, n in enumerate(scm.scaffold.node_names)}
    nodes = {
        ids[node]: {
            k: {"name": nm, "description": f"{nm} standardized measure"}
            for k, nm in zip(("aligned", "neutral", "misaligned"), names, strict=True)
        }
        for node, names in table.items()
    }
    return json.dumps({"domain": "study", "context": "one row per unit", "nodes": nodes})


class _Stub:
    def __init__(self, responses):
        self._responses = list(responses)

    def propose(self, prompt):
        return self._responses.pop(0)


def _grounding(scm, rng):
    return "grounding"


# ---- level construction -----------------------------------------------------------------------------


def test_construct_levels_and_composition():
    scm = _scm()
    trip = _triples(scm)
    outcome = scm.scaffold.outcome

    aligned = construct_labeling(trip, scm, 1.0, "p")
    assert aligned.node_to_label[outcome] == "cardiovascular_risk"
    assert all(nm == trip.choices[n]["aligned"][0] for n, nm in aligned.node_to_label.items())
    assert aligned.provenance.alignment == 1.0
    assert aligned.provenance.n_aligned == len(scm.scaffold.node_names)

    neutral = construct_labeling(trip, scm, 0.0, "p")
    assert neutral.node_to_label[outcome] == "cardiovascular_risk"  # outcome always its real name
    assert all(
        neutral.node_to_label[n] == trip.choices[n]["neutral"][0] for n in scm.scaffold.node_names if n != outcome
    )
    assert neutral.provenance.n_neutral == 5 and neutral.provenance.n_aligned == 1

    misaligned = construct_labeling(trip, scm, -1.0, "p")
    assert all(
        misaligned.node_to_label[n] == trip.choices[n]["misaligned"][0] for n in scm.scaffold.node_names if n != outcome
    )
    assert misaligned.provenance.n_misaligned == 5


def test_partial_level_flips_load_bearing_nodes_first():
    scm = _scm()
    trip = _triples(scm)
    order = _informative_order(scm)  # decoy, strongest driver, confounder, mediator, clean driver
    half = construct_labeling(trip, scm, 0.5, "p")
    n_inf = round(0.5 * len(order))
    informative = set(order[:n_inf])
    for node in scm.scaffold.node_names:
        if node == scm.scaffold.outcome or node in informative:
            assert half.node_to_label[node] == trip.choices[node]["aligned"][0]
        else:
            assert half.node_to_label[node] == trip.choices[node]["neutral"][0]


# ---- pre-checks -------------------------------------------------------------------------------------


def test_pre_check_passes_good_and_flags_bad():
    scm = _scm()
    assert _pre_check(_triples(scm), scm) == []
    placeholder = {**_GOOD, "Dz": ("feature_1", "badge_reading", "coronary_calcium")}
    assert any("placeholder" in i for i in _pre_check(_triples(scm, placeholder), scm))
    dup = {**_GOOD, "B": ("age_years", "form_version", "barometric_pressure")}  # collides with C aligned
    assert any("distinct" in i for i in _pre_check(_triples(scm, dup), scm))


# ---- propose -> critique -> revise loop -------------------------------------------------------------


def test_propose_revises_until_prechecks_pass():
    scm = _scm()
    bad = _triples_json(scm, {**_GOOD, "Dz": ("feature_1", "badge_reading", "coronary_calcium")})
    good = _triples_json(scm)
    triples = propose_triples(
        scm, np.random.default_rng(0), proposer=_Stub([bad, good]), grounding=_grounding, max_revisions=2
    )
    assert triples.choices["Dz"]["aligned"][0] == "clinic_room"


def test_critic_triggers_revision():
    scm = _scm()
    proposer = _Stub([_triples_json(scm), _triples_json(scm)])
    critic = _Stub([json.dumps({"ok": False, "issues": ["v4: weak"]}), json.dumps({"ok": True})])
    triples = propose_triples(
        scm, np.random.default_rng(0), proposer=proposer, grounding=_grounding, critic=critic, max_revisions=2
    )
    assert triples.domain == "study"


def test_propose_raises_when_never_valid():
    scm = _scm()
    bad = _triples_json(scm, {**_GOOD, "Dz": ("feature_1", "badge_reading", "coronary_calcium")})
    with pytest.raises(ValueError, match="could not produce"):
        propose_triples(
            scm, np.random.default_rng(0), proposer=_Stub([bad, bad]), grounding=_grounding, max_revisions=1
        )


# ---- labeler + build_instance -----------------------------------------------------------------------


def test_llm_labeler_and_build_instance():
    scm = _scm()
    data = sample(scm, 32, seed=0)
    labeler = LLMLabeler(level=1.0, proposer=_Stub([_triples_json(scm)]), grounding=_grounding)
    instance, key = build_instance(scm, data, seed=0, labeler=labeler)
    assert instance.outcome == "cardiovascular_risk"
    assert key.provenance.alignment == 1.0
    assert set(key.label_to_node) == set(instance.columns)


def test_level_out_of_range_rejected():
    with pytest.raises(ValueError, match="level"):
        LLMLabeler(level=1.5, proposer=_Stub([]), grounding=_grounding)


def test_paired_skins_same_scm_differ_only_in_names():
    scm = _scm()
    data = sample(scm, 32, seed=0)
    trip = _triples(scm)
    pos = construct_labeling(trip, scm, 1.0, "p")
    neg = construct_labeling(trip, scm, -1.0, "p")
    pos_inst, pos_key = build_instance(scm, data, seed=0, labeler=lambda _s, _r: pos)
    neg_inst, neg_key = build_instance(scm, data, seed=0, labeler=lambda _s, _r: neg)

    pos_order = [pos_key.label_to_node[c] for c in pos_inst.columns]
    neg_order = [neg_key.label_to_node[c] for c in neg_inst.columns]
    assert pos_order == neg_order  # same node order (RNG decouple)
    assert np.array_equal(pos_inst.data, neg_inst.data)
    assert pos_inst.columns != neg_inst.columns
    assert pos_key.scm.true_ranking() == neg_key.scm.true_ranking()  # gold identical


def test_grid_is_the_five_levels():
    assert GRID == (-1.0, -0.5, 0.0, 0.5, 1.0)
