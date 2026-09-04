import json
import re

import numpy as np
import pytest

from causalagentbench import build_instance, generate
from causalagentbench.generation.labeler import (
    GRID,
    LLMLabeler,
    NamingTriples,
    _critic_prompt,
    _graph_facts,
    _pre_check,
    construct_labeling,
    discrete_name_tokens,
    propose_triples,
    role_vocabulary_hits,
)
from causalagentbench.scm.sample import sample
from causalagentbench.types import Role, Scaffold

# A small, hand-built example graph (confounder C, confounded driver A + mediator M, clean driver B,
# decoy Dz) - the same shape used across the test suite (tests/scm/test_scaffold.py). Node order:
# C, A, B, M, Dz, Y -> ids v0..v5.
_SCAFFOLD = Scaffold(
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

_GOOD = {
    "C": ("age_years", "arterial_stiffness"),
    "A": ("ldl_cholesterol", "appointment_delay"),
    "B": ("systolic_pressure", "barometric_pressure"),
    "M": ("cardiorespiratory_fitness", "family_history"),
    "Dz": ("clinic_room", "coronary_calcium"),
    "Y": ("cardiovascular_risk", "outcome_alt_c"),
}


def _scm(seed: int = 0):
    return generate(seed, _SCAFFOLD)


def _triples(scm, table=_GOOD) -> NamingTriples:
    choices = {
        node: {k: (nm, f"{nm} standardized measure") for k, nm in zip(("aligned", "misaligned"), names, strict=True)}
        for node, names in table.items()
    }
    return NamingTriples(domain="study", context="one row per unit", choices=choices)


def _triples_json(scm, table=_GOOD) -> str:
    ids = {n: f"v{i}" for i, n in enumerate(scm.scaffold.node_names)}
    nodes = {
        ids[node]: {
            k: {"name": nm, "description": f"{nm} standardized measure"}
            for k, nm in zip(("aligned", "misaligned"), names, strict=True)
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

    with pytest.raises(ValueError, match="level 0"):
        construct_labeling(trip, scm, 0.0, "p")  # neutral naming is no longer generated

    misaligned = construct_labeling(trip, scm, -1.0, "p")
    assert all(
        misaligned.node_to_label[n] == trip.choices[n]["misaligned"][0] for n in scm.scaffold.node_names if n != outcome
    )
    assert misaligned.provenance.n_misaligned == 5


def test_uniform_level_ignores_magnitude_only_sign_matters():
    """An intermediate level collapses to its sign's kind - only sign is read now.

    ``GRID`` only ever passes {-1, +1}, but ``construct_labeling`` itself has no partial-flip machinery
    left to get wrong at any other nonzero value.
    """
    scm = _scm()
    trip = _triples(scm)
    outcome = scm.scaffold.outcome
    quarter = construct_labeling(trip, scm, 0.25, "p")
    full = construct_labeling(trip, scm, 1.0, "p")
    assert quarter.node_to_label == full.node_to_label
    assert quarter.node_to_label[outcome] == "cardiovascular_risk"


def test_alignment_level_alone_determines_whether_a_driver_is_aligned():
    """Naming is uniform per level, so a driver's "aligned" bit is exactly ``level > 0``.

    Load-bearing well beyond this module: it is why the driver-grain report
    (``agent/eval.py::aggregate_by_driver_cell``) crosses identifiability with ``alignment`` instead of
    storing a ``Quadrant`` per driver - the naming half of a quadrant would be a second copy of ``alignment``.
    If partial-flip naming ever returns, this fails loudly instead of that report silently mislabeling cells.
    """
    scm = _scm()
    trip = _triples(scm)
    outcome = scm.scaffold.outcome
    for level in GRID:
        node_kind = construct_labeling(trip, scm, level, "p").provenance.node_kind
        for node in scm.scaffold.node_names:
            if node != outcome:  # the outcome is always aligned, and is never a candidate driver
                assert (node_kind[node] == "aligned") == (level > 0)


# ---- pre-checks -------------------------------------------------------------------------------------


def test_pre_check_passes_good_and_flags_bad():
    scm = _scm()
    assert _pre_check(_triples(scm), scm) == []
    placeholder = {**_GOOD, "Dz": ("feature_1", "coronary_calcium")}
    assert any("placeholder" in i for i in _pre_check(_triples(scm, placeholder), scm))
    dup = {**_GOOD, "B": ("age_years", "barometric_pressure")}  # collides with C aligned
    assert any("distinct" in i for i in _pre_check(_triples(scm, dup), scm))


# ---- propose -> critique -> revise loop -------------------------------------------------------------


def test_propose_revises_until_prechecks_pass():
    scm = _scm()
    bad = _triples_json(scm, {**_GOOD, "Dz": ("feature_1", "coronary_calcium")})
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
    bad = _triples_json(scm, {**_GOOD, "Dz": ("feature_1", "coronary_calcium")})
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


def test_role_vocabulary_hits_catches_structure_and_spares_domain_names():
    """The firewall must reject role/relationship words and leave ordinary domain measurements alone."""
    for leaking in (
        "deprivation_common_cause_index",  # every one of these appeared in the stored pilot
        "stress_to_guilt_mediator",
        "baseline_temperament_confound",
        "runoff_pathway_flux",
        "core_mood_driver",
    ):
        assert role_vocabulary_hits(leaking), f"{leaking!r} should be rejected"
    for clean in (
        "groundwater_nitrate_mg_l",
        "systolic_blood_pressure",
        "annual_rainfall_mm",
        "household_income_index",
    ):
        assert role_vocabulary_hits(clean) == [], f"{clean!r} should pass"
    # Underscores are normalized, so snake_case cannot smuggle a banned phrase past the scan.
    assert role_vocabulary_hits("a_common_cause_of_things")
    # Descriptions are scanned for narration, not just names.
    assert role_vocabulary_hits("Standardized index that influences the outcome.")
    assert role_vocabulary_hits("Standardized reading with no bearing on pollution.")


def test_discrete_name_tokens_flags_identifiers_and_spares_continuous_measures():
    """Whole-token matching, so continuous names containing similar substrings survive."""
    for discrete in ("form_revision_number", "record_id_sequence", "survey_language_version", "blood_type"):
        assert discrete_name_tokens(discrete), f"{discrete!r} should be flagged"
    for continuous in (
        "sample_spacing_resolution",
        "calibration_drift_rate",
        "body_mass_index_standardized",
        "questionnaire_item_count",  # counts are deliberately allowed - see _DISCRETE_NAME_TOKENS
        "tsh_assay_precision",
        "typing_speed_words_per_minute",  # contains "typ" but not the token "type"
    ):
        assert discrete_name_tokens(continuous) == [], f"{continuous!r} should pass"


def test_pre_check_rejects_a_nominal_identifier_name():
    scm = _scm()
    nominal = {**_GOOD, "B": ("device_serial_code", "barometric_pressure")}
    issues = _pre_check(_triples(scm, nominal), scm)
    assert any("nominal/identifier" in i for i in issues)
    # The exact failure that cost a whole SCM in a real build.
    assert discrete_name_tokens("survey_language_version") == ["version"]


def test_pre_check_rejects_role_vocabulary_in_names_and_descriptions():
    scm = _scm()
    leaked_name = {**_GOOD, "C": ("shared_confounder_index", "arterial_stiffness")}
    issues = _pre_check(_triples(scm, leaked_name), scm)
    assert any("structural role" in i for i in issues)

    # Same card, clean names, but a description that narrates the relationship.
    triples = _triples(scm)
    choices = {
        node: {
            kind: (name, "Standardized score that causes the outcome." if node == "A" else desc)
            for kind, (name, desc) in kinds.items()
        }
        for node, kinds in triples.choices.items()
    }
    leaked_desc = NamingTriples(domain=triples.domain, context=triples.context, choices=choices)
    assert any("structural role or relationship" in i for i in _pre_check(leaked_desc, scm))


def test_propose_revises_away_a_leaking_name_then_accepts():
    """A leaking proposal is re-prompted, not discarded - the firewall costs a retry, not the SCM."""
    scm = _scm()
    leaking = _triples_json(scm, {**_GOOD, "M": ("causal_pathway_step", "family_history")})
    proposer = _Stub([leaking, _triples_json(scm)])
    triples = propose_triples(scm, np.random.default_rng(0), proposer=proposer, grounding=_grounding)
    assert role_vocabulary_hits(triples.choices["M"]["aligned"][0]) == []


def test_graph_facts_bands_effects_instead_of_disclosing_magnitudes():
    """The generator sees a coarse band, never the number it would otherwise encode into a name."""
    scm = _scm()
    facts = _graph_facts(scm)
    assert "effect" in facts
    # No signed two-decimal magnitudes (the previous "true effect on outcome = +0.37" disclosure).
    assert not re.search(r"[+-]\d\.\d\d", facts)
    for driver in scm.scaffold.drivers:
        assert scm.scaffold.roles[driver].value in facts or "decoy" in facts


def test_critic_prompt_shows_descriptions_and_context():
    """The critic must review the whole card; names-only was the audited gap."""
    scm = _scm()
    prompt = _critic_prompt(scm, _triples(scm))
    assert "one row per unit" in prompt  # the shared context
    assert "standardized measure" in prompt  # a description, not just a name
    assert "age_years" in prompt


def test_grid_is_the_two_endpoints():
    assert GRID == (-1.0, 1.0)
