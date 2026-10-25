import numpy as np
import pytest

from causalagentbench.benchmark import N_ROWS
from causalagentbench.scm.generate import realize
from causalagentbench.scm.identifiability import equivalence_class, possible_effects
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.baselines import BASELINES, baseline_answer
from causalagentbench.scoring.gates import SHORTCUT_ERROR_THRESHOLD, generate_scm
from causalagentbench.scoring.score import score_answer, score_output
from causalagentbench.task.answer import validate_answer_contract
from causalagentbench.task.instance import InstanceKey, build_instance
from causalagentbench.types import SCM, Scaffold


def _population_data(sigma: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Rows with zero sample mean and sample covariance (divisor n) exactly ``sigma``.

    OLS with an intercept on such a table returns population coefficients, so baselines run on it are
    at the population limit.
    """
    z = np.random.default_rng(seed).standard_normal((n, len(sigma)))
    z -= z.mean(axis=0)
    white = z @ np.linalg.inv(np.linalg.cholesky(z.T @ z / n)).T
    return white @ np.linalg.cholesky(sigma).T


def _scm(seed: int) -> SCM:
    """An accepted SCM."""
    scm, _ = generate_scm(np.random.SeedSequence(seed))
    return scm


def _population_task(seed: int):
    """A task whose table carries the population moments of an accepted SCM."""
    scm = _scm(seed)
    return build_instance(scm, _population_data(scm.sigma, 500, seed), seed)


@pytest.mark.parametrize("name", BASELINES)
def test_every_baseline_returns_a_contract_valid_answer(name):
    scm = _scm(0)
    instance, key = build_instance(scm, sample(scm, 2000, 0), 0)
    answer = baseline_answer(name, instance, key)
    validate_answer_contract(answer, instance.candidate_drivers, columns=instance.columns, outcome=instance.outcome)
    assert score_output(f"<answer>{answer.model_dump_json()}</answer>", key).parse_ok


def test_unknown_baseline_is_rejected():
    instance, key = _population_task(0)
    with pytest.raises(ValueError, match="unknown baseline"):
        baseline_answer("magic", instance, key)


def test_zero_baseline_scores_exactly_one():
    instance, key = _population_task(0)
    assert score_answer(baseline_answer("zero", instance, key), key).task_error == 1.0


@pytest.mark.parametrize("seed", range(3))
def test_oracle_is_exact_at_the_population_limit(seed):
    instance, key = _population_task(seed)
    score = score_answer(baseline_answer("oracle", instance, key), key)
    assert score.task_error < 1e-9
    assert all(score.adjustment_valid.values())


@pytest.mark.parametrize("seed", range(3))
def test_oracle_error_at_benchmark_size_is_sampling_noise(seed):
    scm = _scm(seed)
    instance, key = build_instance(scm, sample(scm, N_ROWS, seed), seed)
    score = score_answer(baseline_answer("oracle", instance, key), key)
    assert all(score.adjustment_valid.values())
    assert max(abs(score.estimate[d] - score.truth[d]) for d in score.truth) < 0.1  # several standard errors


@pytest.mark.parametrize("seed", range(3))
def test_data_only_best_reports_the_median_possible_effect(seed):
    instance, key = _population_task(seed)
    scaffold = key.scm.scaffold
    members = equivalence_class(scaffold)
    answer = baseline_answer("data_only_best", instance, key)
    for verdict in answer.verdicts:
        node = key.label_to_node[verdict.driver]
        values = possible_effects(key.scm.sigma, scaffold.node_names, members, node, scaffold.outcome)
        assert verdict.effect == pytest.approx(float(np.median(values)), abs=1e-9)
        # The reported set is the parent set of the equivalent graph whose effect is nearest that median.
        parent_sets = {tuple(sorted(key.node_to_label[p] for p, c in m if c == node)) for m in members}
        assert tuple(sorted(verdict.adjustment_set)) in parent_sets


def test_data_only_best_does_not_depend_on_which_member_is_true():
    """It reads the class, not the true graph: swapping in another member leaves its answer unchanged."""
    instance, key = _population_task(1)
    scaffold = key.scm.scaffold
    other = next(m for m in equivalence_class(scaffold) if m != tuple(sorted(scaffold.edges)))
    member = Scaffold(scaffold.node_names, scaffold.roles, other, scaffold.drivers, scaffold.outcome)
    swapped = InstanceKey(
        scm=realize(member, key.scm.sigma),
        label_to_node=key.label_to_node,
        node_to_label=key.node_to_label,
        producer=key.producer,
    )
    assert baseline_answer("data_only_best", instance, swapped) == baseline_answer("data_only_best", instance, key)


@pytest.mark.parametrize("seed", range(3))
def test_fixed_adjustment_recipes_fail_on_accepted_tasks(seed):
    """The no_shortcut gate, seen through the baselines that implement two of its recipes."""
    instance, key = _population_task(seed)
    for name in ("unadjusted", "adjust_all"):
        assert score_answer(baseline_answer(name, instance, key), key).task_error >= SHORTCUT_ERROR_THRESHOLD
