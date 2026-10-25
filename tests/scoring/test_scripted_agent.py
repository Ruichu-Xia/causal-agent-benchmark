"""Tests for the no-LLM PC baseline: it must go through the model's answer contract and ignore names."""

import numpy as np
import pytest

from causalagentbench.scm.sample import sample
from causalagentbench.scoring.gates import generate_scm
from causalagentbench.scoring.metrics import regression_coefficient
from causalagentbench.scoring.scripted_agent import scripted_answer
from causalagentbench.task.answer import validate_answer_contract
from causalagentbench.task.instance import build_instance, with_placeholder_names

pytest.importorskip("causallearn", reason="the scripted baseline needs the discovery extra")


def _task(seed: int = 0, n: int = 2000):
    """An accepted task with placeholder names."""
    scm, _ = generate_scm(np.random.SeedSequence(seed))
    return build_instance(scm, sample(scm, n, seed), seed)


def test_the_script_satisfies_the_answer_contract():
    instance, _ = _task()
    answer = scripted_answer(instance)
    validate_answer_contract(answer, instance.candidate_drivers, columns=instance.columns, outcome=instance.outcome)
    for verdict in answer.verdicts:
        assert verdict.driver not in verdict.adjustment_set
        assert instance.outcome not in verdict.adjustment_set


def test_each_estimate_is_the_regression_on_its_reported_adjustment_set():
    instance, _ = _task()
    for verdict in scripted_answer(instance).verdicts:
        expected = regression_coefficient(
            instance.data, instance.columns, instance.outcome, verdict.driver, verdict.adjustment_set
        )
        assert verdict.effect == pytest.approx(expected)


def test_the_script_is_blind_to_the_column_names():
    """Renaming every column, with data and column order fixed, must not change a single estimate."""
    task = _task()
    renamed, renamed_key = with_placeholder_names(task, seed=3)
    instance, key = task
    rename = {key.node_to_label[node]: label for node, label in renamed_key.node_to_label.items()}
    after = {v.driver: v for v in scripted_answer(renamed).verdicts}
    for before in scripted_answer(instance).verdicts:
        verdict = after[rename[before.driver]]
        assert verdict.effect == pytest.approx(before.effect, abs=1e-12)
        assert set(verdict.adjustment_set) == {rename[z] for z in before.adjustment_set}
