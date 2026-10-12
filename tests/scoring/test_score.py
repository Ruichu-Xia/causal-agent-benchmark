"""Tests for the scorer.

The property most of these defend is that **no metric's denominator is the agent's to choose**. Every
candidate driver is graded on identification and estimation, always, in every naming condition. The previous
design let the agent abstain, which excluded drivers from three metrics at once and let a blanket-abstainer
be scored on a smaller, easier population than a model that answered everything.
"""

import numpy as np
import pytest

from causalagentbench import build_instance, generate, sample
from causalagentbench.scoring.baselines import effect_scale, population_coefficient
from causalagentbench.scoring.score import (
    AnswerScore,
    _adjustment_bias,
    _effect_error,
    _ranking_tau,
    score_answer,
    score_output,
)
from causalagentbench.task.answer import Answer, DriverVerdict
from causalagentbench.task.instance import InstanceKey
from causalagentbench.types import Role, Scaffold

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


def _instance() -> tuple[InstanceKey, dict[str, str]]:
    scm = generate(0, _SCAFFOLD)
    data = sample(scm, 64, seed=0)
    _, key = build_instance(scm, data, seed=0)
    return key, dict(key.node_to_label)  # node -> label


def _verdict(
    label: dict[str, str],
    node: str,
    adjust: tuple[str, ...] = (),
    effect: float = 0.0,
    identifiable: bool | None = None,
) -> DriverVerdict:
    return DriverVerdict(
        driver=label[node],
        adjustment_set=tuple(label[z] for z in adjust),
        effect=effect,
        identifiable_from_data=identifiable,
    )


def _perfect(key: InstanceKey, label: dict[str, str]) -> Answer:
    """Gold adjustment sets and exact true effects. Also the true order, so ranking is perfect too."""
    return Answer(
        verdicts=tuple(
            _verdict(label, node, adjust, key.scm.total_effect_on_outcome(node))
            for node, adjust in (("A", ("C",)), ("B", ()), ("Dz", ("C",)))
        )
    )


# ---- nothing is ever ungraded --------------------------------------------------------------------------


def test_every_driver_is_graded_on_both_axes():
    """The core invariant: fixed denominators, set by the instance and not by the answer."""
    key, label = _instance()
    score = score_answer(_perfect(key, label), key)
    assert set(score.adjustment_bias) == set(key.scm.scaffold.drivers)
    assert set(score.effect_error) == set(key.scm.scaffold.drivers)
    assert all(isinstance(v, float) for v in score.adjustment_bias.values())
    assert all(isinstance(v, float) for v in score.effect_error.values())


def test_a_perfect_answer_scores_one():
    key, label = _instance()
    score = score_answer(_perfect(key, label), key)
    assert score.identification == pytest.approx(1.0, abs=1e-9)
    assert score.estimation == pytest.approx(1.0, abs=1e-9)
    assert score.overall == pytest.approx(1.0, abs=1e-9)
    assert score.ranking_tau == pytest.approx(1.0)
    assert score.effect_mae == pytest.approx(0.0)


def test_adjustment_quality_does_not_change_reward_for_fixed_estimates():
    key, label = _instance()
    exact = _perfect(key, label)
    no_controls = Answer(verdicts=tuple(v.model_copy(update={"adjustment_set": ()}) for v in exact.verdicts))
    correct = score_answer(exact, key)
    wrong = score_answer(no_controls, key)
    assert wrong.identification < correct.identification
    assert wrong.overall == correct.overall == 1.0


def test_reward_matches_effect_error_and_mae_remains_unclipped():
    key, label = _instance()
    scale = effect_scale(key.scm)
    exact = _perfect(key, label)
    # Independent reference: errors of 0, half a scale, and two scales.
    offsets = (0.0, 0.5 * scale, 2.0 * scale)
    shifted = Answer(
        verdicts=tuple(
            v.model_copy(update={"effect": v.effect + delta}) for v, delta in zip(exact.verdicts, offsets, strict=True)
        )
    )
    score = score_answer(shifted, key)
    assert score.overall == score.estimation == pytest.approx(0.5)
    assert score.effect_mae == pytest.approx(2.5 * scale / 3)


def test_an_empty_answer_cannot_beat_a_real_one():
    """Answering nothing used to be free on the graded metrics; now it takes the maximum penalty."""
    key, label = _instance()
    blank = Answer(verdicts=tuple(_verdict(label, d) for d in key.scm.scaffold.drivers))
    assert score_answer(blank, key).overall < score_answer(_perfect(key, label), key).overall


def test_a_parse_failure_scores_zero_rather_than_raising():
    key, _label = _instance()
    score = score_output("no answer block here", key)
    assert not score.parse_ok
    assert score.overall == 0.0
    assert score.effect_mae is None
    assert set(score.identifiable) == set(key.scm.scaffold.drivers)  # instance property, still filled
    assert all(v is None for v in score.identifiability_judgment.values())


# ---- identification is continuous ----------------------------------------------------------------------


def test_adjustment_bias_orders_qualitatively_different_mistakes():
    """A validity boolean scored all of these an identical 0; the bias measure ranks them."""
    key, label = _instance()
    scale = effect_scale(key.scm)

    def bias(adjust: tuple[str, ...]) -> float:
        answer = Answer(verdicts=(_verdict(label, "A", adjust, 0.0),))
        return _adjustment_bias(answer, key, "A", scale)

    assert bias(("C",)) == pytest.approx(0.0, abs=1e-9)  # the gold set
    assert bias(()) > 0.0  # missing the confounder leaves bias


def test_a_harmless_extra_covariate_costs_nothing():
    """Conditioning on a variable that blocks no path is not an error, and must not be scored as one."""
    key, label = _instance()
    scale = effect_scale(key.scm)
    gold = Answer(verdicts=(_verdict(label, "A", ("C",), 0.0),))
    plus_decoy = Answer(verdicts=(_verdict(label, "A", ("C", "Dz"), 0.0),))
    assert _adjustment_bias(plus_decoy, key, "A", scale) == pytest.approx(
        _adjustment_bias(gold, key, "A", scale), abs=1e-9
    )


def test_conditioning_on_the_driver_or_outcome_is_maximally_penalized():
    key, label = _instance()
    scale = effect_scale(key.scm)
    on_outcome = Answer(verdicts=(_verdict(label, "A", ("Y",), 0.0),))
    assert _adjustment_bias(on_outcome, key, "A", scale) == 1.0


def test_adjustment_bias_matches_the_estimand_the_set_implies():
    """The metric is defined off `population_coefficient`, not re-derived - pin that they agree."""
    key, label = _instance()
    scale = effect_scale(key.scm)
    answer = Answer(verdicts=(_verdict(label, "A", (), 0.0),))
    induced = population_coefficient(key.scm, "A", [])
    expected = min(abs(induced - key.scm.total_effect_on_outcome("A")) / scale, 1.0)
    assert _adjustment_bias(answer, key, "A", scale) == pytest.approx(expected)


# ---- estimation -----------------------------------------------------------------------------------------


def test_effect_error_is_signed_so_a_flipped_sign_is_wrong():
    key, label = _instance()
    scale = effect_scale(key.scm)
    truth = key.scm.total_effect_on_outcome("A")
    flipped = Answer(verdicts=(_verdict(label, "A", ("C",), -truth),))
    exact = Answer(verdicts=(_verdict(label, "A", ("C",), truth),))
    assert _effect_error(exact, key, "A", scale) == pytest.approx(0.0, abs=1e-9)
    assert _effect_error(flipped, key, "A", scale) > 0.0


def test_effect_error_is_clipped_at_one():
    key, label = _instance()
    scale = effect_scale(key.scm)
    wild = Answer(verdicts=(_verdict(label, "A", ("C",), 1e6),))
    assert _effect_error(wild, key, "A", scale) == 1.0


# ---- ranking --------------------------------------------------------------------------------------------


def test_ranking_compares_every_candidate_in_every_condition():
    """Ranking must cover the same population in every naming condition.

    The old exclusion rule ranked 5 candidates in one condition and 3.38 in another - measured, and the only
    contrast in that run whose CI excluded zero - which made every cross-condition comparison meaningless.
    """
    key, label = _instance()
    answer = _perfect(key, label)
    # Kendall tau over all three drivers; a driver dropped from the comparison would change this.
    assert _ranking_tau(answer, key) == pytest.approx(1.0)


def test_ranking_is_invariant_to_rescaling_the_estimates():
    """The one thing ranking measures that estimation cannot.

    A model with the right causal structure and the wrong units is a different diagnosis from one that got
    the structure wrong, and only tau separates them.
    """
    key, label = _instance()
    shrunk = Answer(
        verdicts=tuple(
            _verdict(label, node, adjust, key.scm.total_effect_on_outcome(node) * 0.1)
            for node, adjust in (("A", ("C",)), ("B", ()), ("Dz", ("C",)))
        )
    )
    assert _ranking_tau(shrunk, key) == pytest.approx(1.0)
    assert score_answer(shrunk, key).estimation < 1.0  # but estimation notices
    assert score_answer(shrunk, key).overall < 1.0  # perfect ranking cannot rescue reward


def test_a_missing_verdict_forfeits_the_ranking():
    key, label = _instance()
    partial = Answer(verdicts=(_verdict(label, "A", ("C",), 1.0),))
    assert _ranking_tau(partial, key) == 0.0


def test_equal_estimates_do_not_gain_ranking_credit_from_hidden_order():
    key, label = _instance()
    answer = Answer(verdicts=tuple(_verdict(label, node, effect=0.0) for node in key.scm.scaffold.drivers))
    assert _ranking_tau(answer, key) == 0.0


# ---- the identifiability judgement ----------------------------------------------------------------------


def test_the_identifiability_judgement_is_stored_raw_including_absence():
    """The raw judgement is kept, absence included.

    Scoring pools precision/recall across instances, so the scorer stores the answer rather than a
    per-instance summary, and `None` has to survive - omitting the field must not be cheaper than
    answering it wrong.
    """
    key, label = _instance()
    answer = Answer(
        verdicts=(
            _verdict(label, "A", ("C",), 0.0, identifiable=False),
            _verdict(label, "B", (), 0.0, identifiable=True),
            _verdict(label, "Dz", ("C",), 0.0),  # not answered
        )
    )
    judgments = score_answer(answer, key).identifiability_judgment
    assert judgments["A"] is False
    assert judgments["B"] is True
    assert judgments["Dz"] is None


def test_identifiability_is_not_part_of_the_headline():
    """It is near-mechanical - gold is the population CPDAG - so it must not move `overall`."""
    key, label = _instance()
    right = Answer(verdicts=tuple(_verdict(label, d, (), 0.0, identifiable=True) for d in ("A", "B", "Dz")))
    wrong = Answer(verdicts=tuple(_verdict(label, d, (), 0.0, identifiable=False) for d in ("A", "B", "Dz")))
    assert score_answer(right, key).overall == score_answer(wrong, key).overall


# ---- stratification axes and baselines ------------------------------------------------------------------


def test_identifiable_is_a_property_of_the_graph_not_the_naming():
    """The placebo row depends on this: the same SCM must classify drivers identically in every skin."""
    scm = generate(0, _SCAFFOLD)
    data = sample(scm, 64, seed=0)
    _a, key_a = build_instance(scm, data, seed=0)
    _b, key_b = build_instance(scm, data, seed=7)  # a different labeling of the same SCM
    by_node_a = score_output("x", key_a).identifiable
    by_node_b = score_output("x", key_b).identifiable
    assert by_node_a == by_node_b


def test_answer_score_is_invariant_to_the_display_names():
    """Scoring goes through `label_to_node`, which is what lets naming be a condition rather than a key."""
    scm = generate(0, _SCAFFOLD)
    data = sample(scm, 64, seed=0)
    _a, key_a = build_instance(scm, data, seed=0)
    _b, key_b = build_instance(scm, data, seed=11)
    a = score_answer(_perfect(key_a, dict(key_a.node_to_label)), key_a)
    b = score_answer(_perfect(key_b, dict(key_b.node_to_label)), key_b)
    assert a.overall == pytest.approx(b.overall)


def test_answer_score_has_no_abstention_surface_left():
    """Guards the deletion: a re-added abstain field would silently reintroduce a movable denominator."""
    fields = set(AnswerScore.__dataclass_fields__)
    assert not {f for f in fields if "abstain" in f or "coverage" in f or "committed" in f}
    assert not hasattr(DriverVerdict, "abstain")
    assert "abstain" not in DriverVerdict.model_fields


def test_scale_is_finite_and_positive():
    key, _label = _instance()
    scale = effect_scale(key.scm)
    assert np.isfinite(scale) and scale > 0
