"""Tests for batching the agent over instances and aggregating the result.

The aggregation is the experiment: ``driver_cells`` crosses naming against identifiability, and
``naming_table`` reduces that to per-metric contrasts. Most of these defend properties that a naive
aggregation gets wrong - pooling F1 from counts rather than averaging per-instance F1s, and resampling
whole SCMs rather than skins or drivers.
"""

import json

import pytest

from causalagentbench import build_instance, sample
from causalagentbench.agent.eval import (
    driver_cells,
    driver_metric,
    identifiability_f1,
    naming_result,
    naming_table,
    paired_bootstrap_ci,
    run_evaluation,
    scm_group_key,
)
from causalagentbench.agent.loop import ModelTurn
from causalagentbench.scoring.gates import generate_gated_procedural
from causalagentbench.scoring.score import AnswerScore
from causalagentbench.task.instance import Instance, InstanceKey


def _instance(seed: int) -> tuple[Instance, InstanceKey]:
    scm, _ = generate_gated_procedural(seed)
    data = sample(scm, 200, seed=seed)
    return build_instance(scm, data, seed=seed)


class _FixedReplyClient:
    """Answers immediately every call, ignoring the message history (no tool use)."""

    def __init__(self, reply: str):
        self._reply = reply

    def complete(self, messages, tools, *, force_final: bool) -> ModelTurn:
        return ModelTurn(content=self._reply)


class _RaisesOnFirstCallClient:
    """Simulates a client/provider error on the first instance only."""

    def __init__(self, reply: str):
        self._reply = reply
        self._calls = 0

    def complete(self, messages, tools, *, force_final: bool) -> ModelTurn:
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("simulated provider failure")
        return ModelTurn(content=self._reply)


def _perfect_answer_text(key: InstanceKey) -> str:
    """A well-formed answer matching gold exactly, including the identifiability judgement."""
    label = dict(key.node_to_label)
    from causalagentbench.scm.identifiability import unidentifiable_drivers

    unidentifiable = set(unidentifiable_drivers(key.scm.scaffold))
    verdicts = [
        {
            "driver": label[node],
            "adjustment_set": [label[z] for z in key.scm.scaffold.correct_adjustment.get(node, ())],
            "effect": key.scm.total_effect_on_outcome(node),
            "identifiable_from_data": node not in unidentifiable,
        }
        for node in key.scm.true_ranking()
    ]
    return f"<answer>{json.dumps({'verdicts': verdicts})}</answer>"


def _score(alignment: float | None, drivers: dict[str, tuple[bool, float, float, bool | None]]) -> AnswerScore:
    """Build a score directly from ``{driver: (identifiable, bias, error, judgment)}``.

    Constructed rather than run through a model so identifiability and alignment can be varied
    independently - the real pipeline ties them together, which is exactly what these tests must not rely on.
    """
    identification = 1.0 - sum(v[1] for v in drivers.values()) / len(drivers)
    estimation = 1.0 - sum(v[2] for v in drivers.values()) / len(drivers)
    return AnswerScore(
        parse_ok=True,
        overall=(identification + estimation) / 2.0,
        identification=identification,
        estimation=estimation,
        ranking_tau=0.0,
        adjustment_bias={d: v[1] for d, v in drivers.items()},
        effect_error={d: v[2] for d, v in drivers.items()},
        identifiability_judgment={d: v[3] for d, v in drivers.items()},
        identifiable={d: v[0] for d, v in drivers.items()},
        alignment=alignment,
    )


# ---- batching -------------------------------------------------------------------------------------------


def test_run_evaluation_isolates_a_crashing_instance_instead_of_losing_the_batch():
    second = _instance(1)
    instances = [_instance(0), second]
    errors: list[tuple[int, Exception]] = []
    scores = run_evaluation(
        instances,
        _RaisesOnFirstCallClient(_perfect_answer_text(second[1])),
        on_instance_error=lambda index, error: errors.append((index, error)),
    )
    assert len(scores) == 2
    assert not scores[0].parse_ok
    assert scores[1].parse_ok
    assert len(errors) == 1
    assert errors[0][0] == 0 and isinstance(errors[0][1], RuntimeError)


def test_run_evaluation_returns_one_score_per_instance():
    instances = [_instance(0), _instance(1)]
    scores = run_evaluation(instances, _FixedReplyClient("not an answer"))
    assert len(scores) == 2
    assert all(not s.parse_ok for s in scores)


def test_a_perfect_answer_scores_one_end_to_end():
    instance, key = _instance(0)
    scores = run_evaluation([(instance, key)], _FixedReplyClient(_perfect_answer_text(key)))
    assert scores[0].parse_ok
    assert scores[0].overall > 0.99


# ---- the experiment table -------------------------------------------------------------------------------


def test_driver_cells_cross_identifiability_with_naming():
    aligned = _score(1.0, {"a": (True, 0.0, 0.0, True), "b": (False, 1.0, 1.0, False)})
    misaligned = _score(-1.0, {"a": (True, 0.0, 0.0, True), "b": (False, 0.5, 0.5, False)})
    cells = {(c.identifiable, c.alignment): c for c in driver_cells([aligned, misaligned])}
    assert len(cells) == 4
    assert cells[(True, 1.0)].estimation == 1.0
    assert cells[(False, 1.0)].estimation == 0.0
    assert cells[(False, -1.0)].estimation == 0.5
    assert all(c.n == 1 for c in cells.values())


def test_driver_cells_pool_across_instances_rather_than_averaging_instance_means():
    one = _score(1.0, {"a": (False, 0.0, 0.0, None)})
    many = _score(1.0, {"a": (False, 1.0, 1.0, None), "b": (False, 1.0, 1.0, None), "c": (False, 1.0, 1.0, None)})
    (cell,) = driver_cells([one, many])
    assert cell.n == 4
    assert cell.estimation == 1.0 - 3 / 4  # pooled, not the 0.5 a mean-of-means would give


def test_driver_cells_sort_identifiable_first_and_name_blind_last():
    scores = [
        _score(None, {"a": (False, 0.0, 0.0, None)}),
        _score(1.0, {"a": (True, 0.0, 0.0, None)}),
        _score(-1.0, {"a": (True, 0.0, 0.0, None)}),
    ]
    order = [(c.identifiable, c.alignment) for c in driver_cells(scores)]
    assert order == [(True, -1.0), (True, 1.0), (False, None)]


# ---- identifiability F1 ---------------------------------------------------------------------------------


def test_identifiability_f1_pools_counts_rather_than_averaging_per_instance_f1():
    """Averaging per-instance F1s is not F1. One instance with a single positive would otherwise dominate."""
    a = _score(1.0, {"x": (False, 0.0, 0.0, False)})  # 1 true positive
    b = _score(1.0, {f"y{i}": (False, 0.0, 0.0, True) for i in range(9)})  # 9 false negatives
    pooled = identifiability_f1([a, b])
    assert pooled.n_positive == 10
    assert pooled.recall == 0.1  # 1 of 10, not the 0.5 a mean of (1.0, 0.0) would give


def test_not_answering_the_identifiability_field_counts_as_a_miss():
    """Omitting the field must never be cheaper than answering it wrong."""
    silent = _score(1.0, {"a": (False, 0.0, 0.0, None)})
    wrong = _score(1.0, {"a": (False, 0.0, 0.0, True)})
    assert identifiability_f1([silent]).recall == identifiability_f1([wrong]).recall == 0.0


def test_identifiability_f1_penalizes_calling_everything_unidentifiable():
    """The reason for F1 over accuracy: a blanket answer must not score well."""
    blanket = _score(1.0, {"a": (True, 0.0, 0.0, False), "b": (False, 0.0, 0.0, False)})
    assert identifiability_f1([blanket]).recall == 1.0
    assert identifiability_f1([blanket]).precision == 0.5  # precision is what catches it


# ---- the two result numbers -----------------------------------------------------------------------------


def test_driver_metric_is_none_when_the_class_is_absent():
    """So a pairing drops the SCM rather than averaging in a value that was never measured."""
    all_identifiable = _score(1.0, {"a": (True, 0.0, 0.0, None)})
    assert driver_metric(all_identifiable, "estimation", identifiable=False) is None
    assert driver_metric(all_identifiable, "estimation", identifiable=True) == 1.0


def test_instance_level_metrics_have_no_meaning_on_a_driver_subset():
    """Instance-level metrics cannot be restricted to a driver subset.

    `ranking_tau` orders all candidates and `overall` mixes both axes, so neither has a meaning on a subset
    without silently changing the population it describes.
    """
    score = _score(1.0, {"a": (True, 0.0, 0.0, None), "b": (False, 0.0, 0.0, None)})
    assert driver_metric(score, "ranking_tau", identifiable=False) is None
    assert driver_metric(score, "overall", identifiable=False) is None
    assert driver_metric(score, "ranking_tau") is not None  # but fine over the whole instance


def test_correctness_holds_semantic_presence_fixed_while_gain_does_not():
    """The reason `correctness` is first-class.

    Here BOTH named conditions beat name-blind by the same amount: the names carry no usable information,
    they just help generally. `gain` reports a large positive effect anyway; `correctness` correctly
    reports zero. Measured live, exactly this gap decided whether the benchmark's central claim held.
    """
    _instance_obj, key = _instance(0)
    paired = [
        (key, _score(1.0, {"a": (False, 0.0, 0.2, None)})),  # aligned
        (key, _score(-1.0, {"a": (False, 0.0, 0.2, None)})),  # misaligned - identical
        (key, _score(None, {"a": (False, 0.0, 0.6, None)})),  # blind - worse
    ]
    row = naming_result(paired, "estimation", identifiable=False)
    assert row.gain.delta == pytest.approx(0.4)  # "names help" - true but not the claim
    assert row.correctness.delta == pytest.approx(0.0)  # "correct names help" - false, and caught


def test_naming_contrasts_are_measured_on_the_unidentifiable_drivers():
    """Aligned names help and misleading names hurt - a model using names without checking them."""
    _instance_obj, key = _instance(0)
    paired = [
        (key, _score(1.0, {"a": (False, 0.0, 0.0, None)})),
        (key, _score(None, {"a": (False, 0.0, 0.5, None)})),
        (key, _score(-1.0, {"a": (False, 0.0, 1.0, None)})),
    ]
    row = naming_result(paired, "estimation", identifiable=False)
    assert row.gain.delta == pytest.approx(0.5)
    assert row.harm.delta == pytest.approx(0.5)
    assert row.correctness.delta == pytest.approx(1.0)


def test_a_model_that_ignores_names_shows_nothing_on_any_contrast():
    _instance_obj, key = _instance(0)
    paired = [(key, _score(level, {"a": (False, 0.0, 0.4, None)})) for level in (1.0, -1.0, None)]
    row = naming_result(paired, "estimation", identifiable=False)
    assert row.correctness.delta == row.gain.delta == row.harm.delta == 0.0


def test_contrasts_pair_within_scm_and_skip_incomplete_ones():
    """Skins of different SCMs are not comparable, so an SCM missing a condition contributes nothing."""
    _ia, key_a = _instance(0)
    _ib, key_b = _instance(1)
    paired = [
        (key_a, _score(1.0, {"a": (False, 0.0, 0.0, None)})),
        (key_a, _score(None, {"a": (False, 0.0, 1.0, None)})),
        (key_b, _score(1.0, {"a": (False, 0.0, 0.0, None)})),  # key_b has no blind skin
    ]
    row = naming_result(paired, "estimation", identifiable=False)
    assert row.gain.delta == 1.0
    assert row.gain.n_scms == 1
    assert row.correctness.n_scms == 0  # no SCM has both alpha skins


def test_contrasts_are_nan_when_nothing_pairs_and_nan_is_never_significant():
    _i, key = _instance(0)
    row = naming_result([(key, _score(1.0, {"a": (False, 0.0, 0.0, None)}))], "estimation", identifiable=False)
    assert row.correctness.delta != row.correctness.delta  # NaN, not a confident zero
    assert not row.correctness.excludes_zero


def test_naming_table_covers_every_metric_and_both_subsets():
    """A model must be measured on every metric.

    `overall` averages two axes that moved in opposite directions under the same manipulation, so reading
    the naming result off it alone hid the effect entirely.
    """
    _i, key = _instance(0)
    paired = [
        (key, _score(level, {"a": (False, 0.0, 0.4, None), "b": (True, 0.0, 0.2, None)})) for level in (1.0, -1.0, None)
    ]
    rows = naming_table(paired)
    assert {r.metric for r in rows if r.subset == "all"} == {"overall", "identification", "estimation", "ranking_tau"}
    assert {r.subset for r in rows} == {"all", "identifiable", "unidentifiable"}


# ---- the resampling unit --------------------------------------------------------------------------------


def test_skins_of_one_scm_share_a_group_key_and_different_scms_do_not():
    scm, _ = generate_gated_procedural(0)
    data = sample(scm, 200, seed=0)
    _a, key_a = build_instance(scm, data, seed=0)
    _b, key_b = build_instance(scm, data, seed=1)  # different labeling, same SCM
    _c, key_c = _instance(1)
    assert scm_group_key(key_a) == scm_group_key(key_b)
    assert scm_group_key(key_a) != scm_group_key(key_c)


def test_bootstrap_resamples_scms_so_duplicating_skins_does_not_narrow_the_interval():
    groups = [[0.1], [0.9], [0.2], [0.8], [0.3]]
    inflated = [values * 7 for values in groups]
    assert paired_bootstrap_ci(groups) == paired_bootstrap_ci(inflated)
    low, high = paired_bootstrap_ci(groups)
    flat = [[value] for values in inflated for value in values]
    flat_low, flat_high = paired_bootstrap_ci(flat)
    assert (high - low) > (flat_high - flat_low)


def test_bootstrap_on_no_groups_is_nan_rather_than_a_confident_zero():
    low, high = paired_bootstrap_ci([])
    assert low != low and high != high
