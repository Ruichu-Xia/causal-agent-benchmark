"""Tests for the capped pooled-error scorer and the RL reward derived from it."""

import numpy as np
import pytest

from causalagentbench.scm.generate import draw_correlation, realize
from causalagentbench.scm.identifiability import name_critical_candidates
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.score import (
    ERROR_CAP,
    INVALID_REWARD,
    REWARD_CLIP,
    score_answer,
    score_output,
)
from causalagentbench.task.answer import Answer, DriverVerdict
from causalagentbench.task.instance import InstanceKey, build_instance
from causalagentbench.types import Role, Scaffold

_SCAFFOLD = Scaffold(
    node_names=("C", "A", "B", "M", "Dz", "Y", "O"),
    roles={
        "C": Role.CONFOUNDER,
        "A": Role.DRIVER_CONFOUNDED,
        "B": Role.DRIVER_CLEAN,
        "M": Role.MEDIATOR,
        "Dz": Role.DECOY,
        "Y": Role.OUTCOME,
        "O": Role.OUTCOME_CHILD,
    },
    edges=(("C", "A"), ("C", "Y"), ("C", "Dz"), ("A", "M"), ("A", "Y"), ("M", "Y"), ("B", "Y"), ("Y", "O")),
    drivers=("A", "B", "Dz"),
    outcome="Y",
)


def _key(seed: int = 0) -> InstanceKey:
    """The hidden key of a task on ``_SCAFFOLD`` (scoring never reads the data)."""
    scm = realize(_SCAFFOLD, draw_correlation(_SCAFFOLD, np.random.default_rng(seed)))
    _, key = build_instance(scm, sample(scm, 64, seed), seed)
    return key


def _truth(key: InstanceKey) -> dict[str, float]:
    """True total effect of every candidate, by hidden node."""
    return {d: key.scm.total_effect_on_outcome(d) for d in key.scm.scaffold.drivers}


def _answer(key: InstanceKey, effects: dict[str, float], adjust: dict[str, tuple[str, ...]] | None = None) -> Answer:
    """An answer with node-keyed effects and adjustment sets, written in display labels."""
    adjust = adjust or {}
    return Answer(
        verdicts=tuple(
            DriverVerdict(
                driver=key.node_to_label[node],
                adjustment_set=tuple(key.node_to_label[z] for z in adjust.get(node, ())),
                effect=effect,
            )
            for node, effect in effects.items()
        )
    )


def _text(answer: Answer) -> str:
    """Model output carrying ``answer`` in an answer block."""
    return f"reasoning...\n<answer>{answer.model_dump_json()}</answer>"


def test_all_true_effects_are_below_one():
    """Precondition for the reference points below: zero predictions earn error exactly 1."""
    assert all(abs(t) < 1.0 for t in _truth(_key()).values())


def test_zero_answer_scores_exactly_one():
    key = _key()
    score = score_answer(_answer(key, dict.fromkeys(_truth(key), 0.0)), key)
    assert score.parse_ok
    assert score.task_error == 1.0
    assert score.reward == 0.0


def test_perfect_answer_scores_zero_error_and_full_reward():
    key = _key()
    score = score_output(_text(_answer(key, _truth(key))), key)
    assert score.parse_ok
    assert score.task_error == 0.0
    assert score.reward == 1.0


@pytest.mark.parametrize("text", ["no answer block", '<answer>{"verdicts": []}</answer>', "<answer>{bad json</answer>"])
def test_invalid_output_scores_zero_estimates_and_the_floor_reward(text):
    key = _key()
    score = score_output(text, key)
    assert not score.parse_ok
    assert score.estimate == dict.fromkeys(key.scm.scaffold.drivers, 0.0)
    assert score.reward == INVALID_REWARD == -1.0
    assert score.task_error == 1.0
    assert not any(score.adjustment_valid.values())


def test_contract_violations_are_invalid_even_with_exact_effects():
    key = _key()
    truth = _truth(key)
    missing = _answer(key, {d: t for d, t in truth.items() if d != "B"})
    on_outcome = _answer(key, truth, adjust={"A": ("Y",)})
    for answer in (missing, on_outcome):
        score = score_output(_text(answer), key)
        assert not score.parse_ok
        assert score.reward == INVALID_REWARD


def test_score_answer_with_none_is_invalid():
    score = score_answer(None, _key())
    assert not score.parse_ok
    assert score.reward == INVALID_REWARD


def test_each_candidate_error_is_capped():
    key = _key()
    truth = _truth(key)
    score = score_answer(_answer(key, {**truth, "A": truth["A"] + 5.0}), key)
    assert score.capped_error("A") == ERROR_CAP == 1.0
    assert score.capped_error("B") == 0.0
    assert score.task_error == pytest.approx(1.0 / sum(abs(t) for t in truth.values()))


def test_a_flipped_sign_is_an_error():
    key = _key()
    truth = _truth(key)
    score = score_answer(_answer(key, {**truth, "B": -truth["B"]}), key)
    assert score.capped_error("B") == pytest.approx(min(2 * abs(truth["B"]), 1.0))


def test_reward_is_one_minus_task_error():
    key = _key()
    truth = _truth(key)
    score = score_answer(_answer(key, {**truth, "B": truth["B"] + 0.1}), key)
    assert score.reward == pytest.approx(1.0 - 0.1 / sum(abs(t) for t in truth.values()))


def test_reward_is_clipped_so_it_never_falls_below_the_invalid_floor():
    key = _key()
    truth = _truth(key)
    score = score_answer(_answer(key, {d: t + 10.0 for d, t in truth.items()}), key)
    assert score.task_error > REWARD_CLIP  # precondition: three capped errors over a small total effect
    assert score.reward == 1.0 - REWARD_CLIP == INVALID_REWARD


def test_adjustment_validity_is_judged_against_the_true_graph():
    key = _key()
    truth = _truth(key)
    assert truth["A"] != 0.0 and truth["B"] != 0.0
    parents = {d: key.scm.scaffold.parents(d) for d in truth}
    assert all(score_answer(_answer(key, truth, adjust=parents), key).adjustment_valid.values())
    # Adjusting for a descendant of the outcome biases a candidate with a nonzero effect.
    on_child = score_answer(_answer(key, truth, adjust={"A": ("O",), "B": ("O",), "Dz": parents["Dz"]}), key)
    assert on_child.adjustment_valid == {"A": False, "B": False, "Dz": True}
    # Validity is a diagnostic: it never moves the error.
    assert on_child.task_error == 0.0


def test_name_critical_flags_follow_the_equivalence_class():
    key = _key()
    score = score_answer(_answer(key, _truth(key)), key)
    critical = set(name_critical_candidates(key.scm))
    assert score.name_critical == {d: d in critical for d in key.scm.scaffold.drivers}
    assert score.name_critical["A"]  # C - A - M is undirected, so A's effect depends on the orientation
    assert not score.name_critical["B"]  # B -> Y is a v-structure edge


def test_scores_are_keyed_by_hidden_node_and_invariant_to_display_names():
    scm = _key(0).scm
    data = sample(scm, 64, 0)
    _, key_a = build_instance(scm, data, 1)
    _, key_b = build_instance(scm, data, 2)
    assert key_a.node_to_label != key_b.node_to_label
    effects = {"A": 0.3, "B": -0.2, "Dz": 0.0}
    a, b = score_answer(_answer(key_a, effects), key_a), score_answer(_answer(key_b, effects), key_b)
    assert set(a.estimate) == set(scm.scaffold.drivers)
    assert a == b
