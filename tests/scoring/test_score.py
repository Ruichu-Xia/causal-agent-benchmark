from causalagentbench import build_instance, generate, sample
from causalagentbench.scoring.gates import generate_gated_procedural
from causalagentbench.scoring.metrics import kendall_tau_zscore
from causalagentbench.scoring.score import (
    Quadrant,
    RankingPairs,
    _gold_abstain,
    quadrant,
    score_answer,
    score_output,
)
from causalagentbench.task.answer import Answer, DriverVerdict
from causalagentbench.task.instance import InstanceKey
from causalagentbench.task.labeling import LabelProvenance
from causalagentbench.types import Role, Scaffold

# A small, hand-built example graph (confounder C, confounded driver A + mediator M, clean driver B,
# decoy Dz) - the same shape used across the test suite as a readable fixture (tests/scm/test_scaffold.py).
# Scoring logic only needs a valid SCM on a known graph, never gate-passing, so this skips the gate loop
# entirely rather than depending on it.
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


def _verdict(label: dict[str, str], node: str, adjust: tuple[str, ...] = (), effect: float = 0.0) -> DriverVerdict:
    return DriverVerdict(driver=label[node], adjustment_set=tuple(label[z] for z in adjust), effect=effect)


def _abstain(label: dict[str, str], node: str) -> DriverVerdict:
    return DriverVerdict(driver=label[node], adjustment_set=(), abstain=True)


def _ordered(label: dict[str, str], nodes: tuple[str, ...]) -> Answer:
    """Verdicts whose estimates place ``nodes`` in exactly that order, strongest first.

    The ranking is derived from the estimates now (``task/answer.py::Answer``), so a test that wants a
    particular order states it as decreasing magnitudes rather than as a separate ranking field.
    """
    return Answer(verdicts=tuple(_verdict(label, node, effect=1.0 - i / 10) for i, node in enumerate(nodes)))


def _perfect(key: InstanceKey, label: dict[str, str]) -> Answer:
    # Each driver's true effect and its gold adjustment (A<-C so {C}; B exogenous so {}; Dz via C). The true
    # effects also derive the true order, so this is simultaneously a perfect ranking and perfect estimates.
    return Answer(
        verdicts=(
            _verdict(label, "A", ("C",), key.scm.total_effect_on_outcome("A")),
            _verdict(label, "B", (), key.scm.total_effect_on_outcome("B")),
            _verdict(label, "Dz", ("C",), key.scm.total_effect_on_outcome("Dz")),
        )
    )


def test_perfect_answer_scores_one():
    key, label = _instance()
    score = score_answer(_perfect(key, label), key)
    assert score.ranking_tau == 1.0
    assert score.adjustment_accuracy == 1.0
    assert all(score.adjustment_valid.values())
    # Exact true effects were reported, so there is no estimation error anywhere.
    assert score.effect_accuracy == 1.0
    assert set(score.effect_error.values()) == {0.0}
    # n_compared is the candidate count actually behind ranking_tau (all 3: A, B, Dz - nothing abstained).
    assert score.n_compared == 3
    assert score.ranking_tau_z == kendall_tau_zscore(1.0, 3)


def test_reversed_ranking_scores_minus_one():
    key, label = _instance()
    reversed_order = tuple(reversed(key.scm.true_ranking()))
    assert score_answer(_ordered(label, reversed_order), key).ranking_tau == -1.0


def test_estimating_no_effect_anywhere_scores_zero_effect_accuracy():
    """The metric's ceiling has a meaning: 1.0 error is "no better than asserting no effect"."""
    key, label = _instance()
    score = score_answer(Answer(verdicts=tuple(_verdict(label, n) for n in ("A", "B", "Dz"))), key)
    # Dz's true effect really is 0, so a zero estimate is exactly right there and costs nothing.
    assert score.effect_error["Dz"] == 0.0
    # A and B carry real effects; estimating zero is the maximum penalty the clip allows for the median
    # driver, and A is the fixture's strongest so its raw error exceeds the scale outright.
    assert score.effect_error["A"] == 1.0
    assert 0.0 < score.effect_accuracy < 1.0


def test_a_sign_error_is_scored_as_an_error_not_a_success():
    """Ranking reads magnitude, but estimation must not: +0.3 where the truth is -0.3 is a wrong answer."""
    key, label = _instance()
    flipped = tuple(_verdict(label, n, effect=-key.scm.total_effect_on_outcome(n)) for n in ("A", "B", "Dz"))
    score = score_answer(Answer(verdicts=flipped), key)
    assert score.ranking_tau == 1.0  # magnitudes are unchanged, so the derived order is still perfect
    assert score.effect_error["A"] > 0.0
    assert score.effect_accuracy < 1.0


def test_backdoor_validity_accepts_supersets_rejects_bad_sets():
    key, label = _instance()

    # A's valid backdoor set is {C}; {C, B} is also valid (superset); {} and {M} (a mediator) are not.
    def acc(node, adjust):
        answer = Answer(verdicts=(_verdict(label, node, adjust),))
        return score_answer(answer, key).adjustment_valid[node]

    assert acc("A", ("C",)) is True
    assert acc("A", ("C", "B")) is True
    assert acc("A", ()) is False
    assert acc("A", ("M",)) is False
    assert acc("B", ()) is True  # B is exogenous


def test_unknown_adjustment_column_is_invalid():
    key, label = _instance()
    answer = Answer(verdicts=(DriverVerdict(driver=label["A"], adjustment_set=("nope",), effect=0.1),))
    assert score_answer(answer, key).adjustment_valid["A"] is False


def test_malformed_ranking_scores_zero():
    key, label = _instance()
    partial = Answer(verdicts=(_verdict(label, "A"),))  # omits B and Dz
    score = score_answer(partial, key)
    assert score.ranking_tau == 0.0
    # Nothing was actually compared (the ranking itself is invalid) - n_compared falls back to every
    # candidate, giving honest task-size context for the 0.0 penalty rather than a meaningless 0.
    assert score.n_compared == 3
    assert score.ranking_tau_z == 0.0


def test_parse_failure_state():
    key, _ = _instance()
    score = score_output("no answer block", key)
    assert score.parse_ok is False
    assert score.ranking_tau == 0.0
    assert score.adjustment_accuracy == 0.0
    assert not any(score.adjustment_valid.values())
    assert score.n_compared == 3
    assert score.ranking_tau_z == 0.0


def test_incomplete_but_json_shaped_answer_is_a_parse_failure():
    key, _ = _instance()
    score = score_output('<answer>{"verdicts": []}</answer>', key)

    assert score.parse_ok is False


def test_alignment_is_carried_through():
    key, label = _instance()
    # abstract labeler -> alignment is None; the field is passed through for stratification.
    assert score_answer(_perfect(key, label), key).alignment is None


def test_gold_abstain_flips_with_the_drivers_own_naming():
    """The capability under test: fall back on the name only where the data ran out.

    The same unidentifiable driver (A) is abstain-correct when naming doesn't resolve it, and abstain-*wrong*
    when its own displayed name is the aligned one - because then the name settles the orientation the data
    left open. That the agent can tell those two worlds apart is what `build_prompt` makes true: naming is
    uniform, so it checks the names against the structure discovery does pin down.
    """
    scm = generate(0, _SCAFFOLD)
    neutral_kind = dict.fromkeys(scm.scaffold.node_names, "neutral")
    aligned_a_kind = {**neutral_kind, "A": "aligned"}

    def key_with(node_kind):
        provenance = LabelProvenance(
            alignment=1.0, n_aligned=1, n_neutral=5, n_misaligned=0, node_kind=node_kind, domain="d", producer="p"
        )
        return InstanceKey(scm=scm, label_to_node={}, node_to_label={}, provenance=provenance)

    assert _gold_abstain(key_with(neutral_kind), "A") is True  # unidentifiable, naming doesn't resolve it
    assert _gold_abstain(key_with(aligned_a_kind), "A") is False  # same driver, now its own name IS aligned


def test_quadrant_covers_all_four_cells():
    """`quadrant` survives as a *reporting* axis, no longer as the source of any gold answer.

    Directly verify all four cells classify as §6's table says, using the fixture's known-unidentifiable A
    and known-identifiable B. `_gold_abstain` no longer reads any of them - see the naming test above.
    """
    scm = generate(0, _SCAFFOLD)
    neutral_kind = dict.fromkeys(scm.scaffold.node_names, "neutral")
    aligned_kind = {**neutral_kind, "A": "aligned", "B": "aligned"}

    def key_with(node_kind):
        provenance = LabelProvenance(
            alignment=1.0, n_aligned=2, n_neutral=4, n_misaligned=0, node_kind=node_kind, domain="d", producer="p"
        )
        return InstanceKey(scm=scm, label_to_node={}, node_to_label={}, provenance=provenance)

    assert quadrant(key_with(aligned_kind), "A") == Quadrant.UNIDENTIFIABLE_ALIGNED  # trust naming
    assert quadrant(key_with(neutral_kind), "A") == Quadrant.UNIDENTIFIABLE_NOT_ALIGNED  # abstain correct
    assert quadrant(key_with(aligned_kind), "B") == Quadrant.IDENTIFIABLE_ALIGNED  # trivial - both agree
    assert quadrant(key_with(neutral_kind), "B") == Quadrant.IDENTIFIABLE_NOT_ALIGNED  # trust data
    assert _gold_abstain(key_with(neutral_kind), "B") is False  # never ambiguous, regardless of naming


# ---- per-driver breakdown -----------------------------------------------------------------------------
# The three per-driver maps exist so results can be split below instance grain (agent/eval.py's
# aggregate_by_driver_cell). `identifiable` is the topology axis; on this fixture A and Dz are the
# unidentifiable ones (only C is a CONFOUNDER, and C-A / C-Dz are the unoriented edges), B is exogenous.


def _pooled_tau(score) -> float:
    """Recompute tau from the per-driver pair counts, the way an aggregator pools a cell."""
    concordant = sum(pairs.concordant for pairs in score.ranking_pairs.values())
    discordant = sum(pairs.discordant for pairs in score.ranking_pairs.values())
    decided = concordant + discordant
    return (concordant - discordant) / decided if decided else 1.0


def test_pooling_per_driver_pairs_reproduces_ranking_tau_exactly():
    """The per-driver ranking counts are a decomposition of `ranking_tau`, not a second metric.

    Every pair has two endpoints, so pooling gives (2C-2D)/(2C+2D) = kendall_tau's own (C-D)/(C+D). Asserted
    over several answer shapes because the two values are computed independently - `ranking_tau` from
    `kendall_tau`, the counts from `_ranking_tau`'s own loop - so this is what catches them drifting apart.
    Exact equality, not approx: both sides are the same small-integer ratio, so any gap is a real bug.
    """
    key, label = _instance()
    effect = key.scm.total_effect_on_outcome
    answers = [
        _perfect(key, label),  # fully concordant
        _ordered(label, tuple(reversed(key.scm.true_ranking()))),  # fully discordant
        # One driver correctly abstained - it leaves the comparison, so its pairs must vanish from both sides.
        Answer(
            verdicts=(
                _verdict(label, "A", ("C",), effect("A")),
                _verdict(label, "B", (), effect("B")),
                _abstain(label, "Dz"),
            )
        ),
    ]
    for answer in answers:
        score = score_answer(answer, key)
        assert _pooled_tau(score) == score.ranking_tau


def test_drivers_outside_the_comparison_carry_no_ranking_pairs():
    key, label = _instance()
    effect = key.scm.total_effect_on_outcome
    verdicts = (
        _verdict(label, "A", ("C",), effect("A")),
        _verdict(label, "B", (), effect("B")),
        _abstain(label, "Dz"),
    )
    score = score_answer(Answer(verdicts=verdicts), key)

    # Dz abstained correctly, so it is dropped from the comparison - contributing to neither side of a
    # pooled ratio, rather than being scored as if it had been ranked wrongly.
    assert score.ranking_pairs["Dz"] == RankingPairs(0, 0)
    assert score.ranking_pairs["A"].concordant + score.ranking_pairs["A"].discordant == 1  # only against B


def test_identifiable_flags_the_topologically_ambiguous_drivers():
    key, _ = _instance()
    # An instance property read off the graph, so it is answer-independent - see the parse-failure test.
    assert score_answer(_perfect(key, dict(key.node_to_label)), key).identifiable == {
        "A": False,
        "B": True,
        "Dz": False,
    }


def test_parse_failure_still_records_identifiability_and_a_consistent_key_space():
    key, _ = _instance()
    score = score_output("no answer block here", key)

    assert score.parse_ok is False
    assert score.identifiable == {"A": False, "B": True, "Dz": False}  # graph fact, not an answer fact
    assert score.abstention_correct == dict.fromkeys(("A", "B", "Dz"), False)
    assert score.ranking_pairs == dict.fromkeys(("A", "B", "Dz"), RankingPairs(0, 0))
    # The key-space contract aggregate_by_driver_cell indexes on: AnswerScore is built at two separate
    # sites, and a driver present in one map but missing from another would only surface at aggregation.
    assert score.adjustment_valid.keys() == score.abstention_correct.keys()
    assert score.abstention_correct.keys() == score.identifiable.keys() == score.ranking_pairs.keys()


# ---- abstention ---------------------------------------------------------------------------------------
# The fixture scaffold + abstract_labels (no naming manipulation at all, so nothing is ever "aligned"): A and Dz are
# gold-abstain-correct (their edges are genuinely unidentifiable, `unidentifiable_drivers`), B is not
# (exogenous, no incoming edge to be ambiguous about).


def test_correctly_abstaining_on_every_unidentifiable_driver_scores_full_credit():
    key, label = _instance()
    verdicts = (
        _abstain(label, "A"),
        _verdict(label, "B", (), key.scm.total_effect_on_outcome("B")),
        _abstain(label, "Dz"),
    )
    score = score_answer(Answer(verdicts=verdicts), key)

    assert score.abstention_accuracy == 1.0
    # A and Dz excluded from the pairwise comparison entirely - only B is left, vacuously perfect.
    assert score.ranking_tau == 1.0
    # n_compared reflects that real exclusion (1, not 3) - too few items for a meaningful z-score.
    assert score.n_compared == 1
    assert score.ranking_tau_z is None
    # A and Dz excluded from the accuracy denominator too - only B is graded, and it's valid.
    assert score.adjustment_accuracy == 1.0
    assert score.adjustment_valid["A"] is None
    assert score.adjustment_valid["Dz"] is None
    assert score.adjustment_valid["B"] is True


def test_committing_instead_of_abstaining_is_graded_normally_but_scores_no_abstention_credit():
    key, label = _instance()
    # The perfect answer commits valid backdoor sets for A and Dz too - both gold-abstain-correct, but
    # committing isn't penalized on adjustment_accuracy/ranking_tau, only on abstention_accuracy.
    score = score_answer(_perfect(key, label), key)
    assert score.ranking_tau == 1.0
    assert score.adjustment_accuracy == 1.0
    assert score.abstention_accuracy == 1 / 3  # only B (not gold-abstain) was called correctly


def test_wrongly_abstaining_on_an_identifiable_driver_is_graded_not_excluded():
    key, label = _instance()
    # true_ranking is (A, B, Dz). Only Dz correctly abstains (excluded); A commits (also gold-abstain-
    # correct here, so this commit is itself an abstention miss - see the "committing instead" test above);
    # B - exogenous, never ambiguous - wrongly abstains. If a bug excluded *any* abstained driver (not just
    # correct ones), B would drop out too, leaving only A - too few items for kendall_tau to catch the
    # swap below.
    verdicts = (
        _verdict(label, "A", ("C",), key.scm.total_effect_on_outcome("A")),
        _abstain(label, "B"),
        _abstain(label, "Dz"),
    )
    score = score_answer(Answer(verdicts=verdicts), key)

    assert score.abstention_accuracy == 1 / 3  # only Dz's abstention call is correct
    # All three metrics now use the same exclusion rule - only a *correct* abstention is exempt. B is
    # identifiable, so declining to answer is a failed adjustment and a failed estimate, not an ungraded one.
    assert score.adjustment_valid["B"] is False
    assert score.adjustment_accuracy == 1 / 2  # A valid, B failed; only Dz (correctly abstained) is exempt
    assert score.effect_error["B"] is not None  # graded on estimation too, at the "asserted no effect" cost
    assert score.effect_error["Dz"] is None  # ...whereas the correct abstention genuinely is exempt
    # A and B are both still compared. Only Dz leaves, so the comparison is over two drivers, not one -
    # which is what a bug that exempted *every* abstention would produce.
    assert score.n_compared == 2
    assert score.ranking_pairs["B"] == RankingPairs(1, 0)


def test_missing_verdict_scores_no_abstention_credit():
    key, label = _instance()
    # A and Dz correctly abstained; B has no verdict at all (not "correctly didn't abstain").
    answer = Answer(verdicts=(_abstain(label, "A"), _abstain(label, "Dz")))
    score = score_answer(answer, key)
    assert score.abstention_accuracy == 2 / 3  # A, Dz correct; B (missing) is not
    assert score.abstention_correct == {"A": True, "B": False, "Dz": True}
    # Exactly why abstention_correct has to be stored rather than derived from adjustment_valid: B's
    # missing verdict makes adjustment_valid False (not None), so "abstained?" read off that map says
    # "no" - which matches gold for identifiable B and would wrongly derive True.
    assert score.adjustment_valid["B"] is False


def test_abstaining_on_everything_cannot_buy_adjustment_accuracy():
    """Blanket abstention must not score well: the denominator is not the agent's to choose.

    This pinned the opposite before, and a live run showed why it mattered - a model abstained on 22 of 48
    identifiable drivers (gold commits on every one) and scored a *perfect* 1.000 in a cell where it
    abstained on all 12, beating its own 0.583 in the cell it answered in full. Only a correct abstention
    (A and Dz here) is ungraded; B is identifiable, so abstaining on it is a failure to supply a set.
    """
    key, label = _instance()
    verdicts = tuple(_abstain(label, node) for node in ("A", "B", "Dz"))
    score = score_answer(Answer(verdicts=verdicts), key)
    assert score.adjustment_valid["A"] is None  # correctly abstained - genuinely unanswerable, so ungraded
    assert score.adjustment_valid["Dz"] is None
    assert score.adjustment_valid["B"] is False  # wrongly abstained - graded, and failed
    assert score.adjustment_accuracy == 0.0


def test_ranking_tau_does_not_penalize_order_between_exactly_tied_decoys():
    """Two decoys always share exactly the same true effect (0) - there is no correct order between them.

    An agent's guess either way must score identically, not get graded as a coin flip. seed=5/n_candidates=5
    has two decoys (Dz0, Dz1) alongside three real drivers with distinct, non-zero effects (A0, A1, A2) -
    true order is (A1, A0, A2, Dz0, Dz1) with the last pair a genuine tie.
    """
    scm, _ = generate_gated_procedural(5, n_candidates=5)
    data = sample(scm, 64, seed=5)
    _, key = build_instance(scm, data, seed=5)
    label = dict(key.node_to_label)
    dz0_first = _ordered(label, ("A1", "A0", "A2", "Dz0", "Dz1"))
    dz1_first = _ordered(label, ("A1", "A0", "A2", "Dz1", "Dz0"))

    assert score_answer(dz0_first, key).ranking_tau == score_answer(dz1_first, key).ranking_tau == 1.0
