import inspect

import numpy as np
import pytest

from causalagentbench.scm.generate import draw_correlation
from causalagentbench.scm.identifiability import equivalence_class, name_critical_candidates
from causalagentbench.scoring import gates
from causalagentbench.scoring.gates import (
    MIN_TOTAL_EFFECT,
    GateReport,
    attempt,
    capped_error,
    evaluate_gates,
    generate_scm,
)
from causalagentbench.types import Role, Scaffold

GATES = ("structure", "names_matter", "no_shortcut", "decoy_tempting", "effect_scale", "orientable")


def _report(**gate_values) -> GateReport:
    """A report with every gate passing unless overridden."""
    fields = dict.fromkeys(GATES, True) | gate_values
    return GateReport(**fields, class_size=1, name_critical=(), min_shortcut_error=0.5, min_orientation_margin=0.5)


def test_passed_requires_every_gate():
    assert _report().passed
    for gate in GATES:
        assert not _report(**{gate: False}).passed


def test_capped_error_pools_capped_absolute_errors():
    truth = {"a": 0.2, "b": -0.5}
    assert capped_error({"a": 0.5, "b": 3.0}, truth) == pytest.approx((0.3 + 1.0) / 0.7)
    assert capped_error(dict.fromkeys(truth, 0.0), truth) == pytest.approx(1.0)
    assert capped_error(truth, truth) == 0.0


def test_evaluate_gates_takes_no_true_graph():
    """The gates see the motif, sigma and the class; the drawn member never reaches them."""
    assert list(inspect.signature(evaluate_gates).parameters) == ["motif", "sigma", "members"]


def _spied_attempt(monkeypatch, seed: int):
    """Run ``attempt`` and also return the exact inputs it handed to ``evaluate_gates``."""
    calls = []

    def spy(motif, sigma, members):
        """Record the gate inputs, then evaluate them for real."""
        calls.append((motif, sigma, members))
        return evaluate_gates(motif, sigma, members)

    monkeypatch.setattr(gates, "evaluate_gates", spy)
    scm, report = attempt(np.random.SeedSequence(seed))
    [(motif, sigma, members)] = calls
    return scm, report, motif, sigma, members


@pytest.mark.parametrize("seed", [0, 3, 5])
def test_attempt_gates_on_the_motif_and_class_not_the_drawn_member(monkeypatch, seed):
    scm, report, motif, sigma, members = _spied_attempt(monkeypatch, seed)
    assert sorted(motif.edges) != sorted(scm.scaffold.edges)  # the drawn member differs from the motif
    assert tuple(sorted(scm.scaffold.edges)) in members
    assert members == equivalence_class(motif)
    assert sigma is scm.sigma
    assert report == evaluate_gates(motif, sigma, members)


@pytest.mark.parametrize("seed", [0, 3])
def test_gate_report_is_the_same_whichever_member_represents_the_class(monkeypatch, seed):
    """Every gate is a function of the class: passing any member in place of the motif changes nothing."""
    _, report, motif, sigma, members = _spied_attempt(monkeypatch, seed)
    for edges in members[:10]:
        member = Scaffold(motif.node_names, motif.roles, edges, motif.drivers, motif.outcome)
        assert evaluate_gates(member, sigma, members) == report


def test_structure_gate_rejects_a_motif_without_name_critical_candidates():
    motif = Scaffold(
        node_names=("B0", "B1", "Y", "O"),
        roles={"B0": Role.DRIVER_CLEAN, "B1": Role.DRIVER_CLEAN, "Y": Role.OUTCOME, "O": Role.OUTCOME_CHILD},
        edges=(("B0", "Y"), ("B1", "Y"), ("Y", "O")),
        drivers=("B0", "B1"),
        outcome="Y",
    )
    members = equivalence_class(motif)
    report = evaluate_gates(motif, draw_correlation(motif, np.random.default_rng(0)), members)
    assert report.class_size == 1
    assert report.name_critical == ()
    assert not report.structure
    assert not report.passed


def test_generate_scm_is_deterministic_for_a_fixed_seed_sequence():
    a, report_a = generate_scm(np.random.SeedSequence(5))
    b, report_b = generate_scm(np.random.SeedSequence(5))
    assert a.scaffold.edges == b.scaffold.edges
    assert np.array_equal(a.W, b.W)
    assert np.array_equal(a.sigma, b.sigma)
    assert report_a == report_b


@pytest.mark.parametrize("seed", range(6))
def test_generate_scm_returns_an_accepted_scm(seed):
    scm, report = generate_scm(np.random.SeedSequence(seed))
    assert report.passed
    assert all(getattr(report, gate) for gate in GATES)
    # The scorer's view (the true graph's class) agrees with the gate's view (the motif's class).
    assert set(name_critical_candidates(scm)) == set(report.name_critical)
    assert report.class_size == len(equivalence_class(scm.scaffold))
    truth = [scm.total_effect_on_outcome(d) for d in scm.scaffold.drivers]
    assert all(abs(t) < 1.0 for t in truth)
    assert sum(abs(t) for t in truth) >= MIN_TOTAL_EFFECT
    assert scm.scaffold.roles["O"] is Role.OUTCOME_CHILD
    assert len(scm.scaffold.drivers) == 5
    assert report.min_shortcut_error >= gates.SHORTCUT_ERROR_THRESHOLD
    assert report.min_orientation_margin >= gates.ORIENTATION_MARGIN


def test_generate_scm_gives_up_after_max_attempts(monkeypatch):
    seed = np.random.SeedSequence(0)
    _, first = attempt(np.random.SeedSequence(0).spawn(1)[0])
    assert not first.passed  # precondition: this seed's first attempt is rejected
    monkeypatch.setattr(gates, "MAX_ATTEMPTS", 1)
    with pytest.raises(RuntimeError, match="no accepted SCM"):
        generate_scm(seed)
