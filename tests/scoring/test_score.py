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


def _verdict(label: dict[str, str], node: str, adjust: tuple[str, ...] = ()) -> DriverVerdict:
    return DriverVerdict(driver=label[node], adjustment_set=tuple(label[z] for z in adjust))


def _abstain(label: dict[str, str], node: str) -> DriverVerdict:
    return DriverVerdict(driver=label[node], adjustment_set=(), abstain=True)


def _perfect(key: InstanceKey, label: dict[str, str]) -> Answer:
    # True order in labels, and each driver's gold adjustment (A<-C so {C}; B exogenous so {}; Dz via C).
    ranking = tuple(label[node] for node in key.scm.true_ranking())
    verdicts = (_verdict(label, "A", ("C",)), _verdict(label, "B"), _verdict(label, "Dz", ("C",)))
    return Answer(verdicts=verdicts, ranking=ranking)


def test_perfect_answer_scores_one():
    key, label = _instance()
    score = score_answer(_perfect(key, label), key)
    assert score.ranking_tau == 1.0
    assert score.adjustment_accuracy == 1.0
    assert all(score.adjustment_valid.values())
    # n_compared is the candidate count actually behind ranking_tau (all 3: A, B, Dz - nothing abstained).
    assert score.n_compared == 3
    assert score.ranking_tau_z == kendall_tau_zscore(1.0, 3)


def test_reversed_ranking_scores_minus_one():
    key, label = _instance()
    answer = _perfect(key, label)
    reversed_ranking = tuple(reversed(answer.ranking))
    assert score_answer(Answer(verdicts=answer.verdicts, ranking=reversed_ranking), key).ranking_tau == -1.0


def test_backdoor_validity_accepts_supersets_rejects_bad_sets():
    key, label = _instance()

    # A's valid backdoor set is {C}; {C, B} is also valid (superset); {} and {M} (a mediator) are not.
    def acc(node, adjust):
        answer = Answer(verdicts=(_verdict(label, node, adjust),), ranking=())
        return score_answer(answer, key).adjustment_valid[node]

    assert acc("A", ("C",)) is True
    assert acc("A", ("C", "B")) is True
    assert acc("A", ()) is False
    assert acc("A", ("M",)) is False
    assert acc("B", ()) is True  # B is exogenous


def test_unknown_adjustment_column_is_invalid():
    key, label = _instance()
    answer = Answer(verdicts=(DriverVerdict(driver=label["A"], adjustment_set=("nope",)),), ranking=())
    assert score_answer(answer, key).adjustment_valid["A"] is False


def test_malformed_ranking_scores_zero():
    key, label = _instance()
    partial = Answer(verdicts=(), ranking=(label["A"],))  # omits B and Dz
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
    score = score_output('<answer>{"verdicts": [], "ranking": []}</answer>', key)

    assert score.parse_ok is False


def test_alignment_is_carried_through():
    key, label = _instance()
    # abstract labeler -> alignment is None; the field is passed through for stratification.
    assert score_answer(_perfect(key, label), key).alignment is None


def test_gold_abstain_flips_with_the_drivers_own_naming():
    """The core §6/§7 distinction.

    The same unidentifiable driver (A) is abstain-correct when naming doesn't resolve it, but not when
    its own displayed name happens to be the aligned one.
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
    """`_gold_abstain` is just `quadrant`'s "abstain is correct" cell (§11.4).

    Directly verify all four cells classify as §6's table says, using the fixture's known-unidentifiable A
    and known-identifiable B.
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
    ranking = tuple(label[node] for node in key.scm.true_ranking())
    reversed_ranking = tuple(reversed(ranking))
    answers = [
        _perfect(key, label),  # fully concordant
        Answer(verdicts=_perfect(key, label).verdicts, ranking=reversed_ranking),  # fully discordant
        # One driver correctly abstained - it leaves the comparison, so its pairs must vanish from both sides.
        Answer(verdicts=(_verdict(label, "A", ("C",)), _verdict(label, "B"), _abstain(label, "Dz")), ranking=ranking),
    ]
    for answer in answers:
        score = score_answer(answer, key)
        assert _pooled_tau(score) == score.ranking_tau


def test_drivers_outside_the_comparison_carry_no_ranking_pairs():
    key, label = _instance()
    ranking = tuple(label[node] for node in key.scm.true_ranking())
    verdicts = (_verdict(label, "A", ("C",)), _verdict(label, "B"), _abstain(label, "Dz"))
    score = score_answer(Answer(verdicts=verdicts, ranking=ranking), key)

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
    ranking = tuple(label[node] for node in key.scm.true_ranking())  # abstained drivers still get a slot
    verdicts = (_abstain(label, "A"), _verdict(label, "B"), _abstain(label, "Dz"))
    score = score_answer(Answer(verdicts=verdicts, ranking=ranking), key)

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
    verdicts = (_verdict(label, "A", ("C",)), _abstain(label, "B"), _abstain(label, "Dz"))
    swapped_ranking = tuple(label[node] for node in ("B", "A", "Dz"))  # A and B reversed vs. true order
    score = score_answer(Answer(verdicts=verdicts, ranking=swapped_ranking), key)

    assert score.abstention_accuracy == 1 / 3  # only Dz's abstention call is correct
    # Both metrics now use the same exclusion rule - only a *correct* abstention is exempt. B is
    # identifiable, so declining to answer is a failed adjustment, not an ungraded one.
    assert score.adjustment_valid["B"] is False
    assert score.adjustment_accuracy == 1 / 2  # A valid, B failed; only Dz (correctly abstained) is exempt
    # A and B are both still compared (B's wrong abstention doesn't exempt it) - reversed relative to the
    # true order, so this must come out fully discordant, not the vacuous 1.0 a masked bug would give.
    assert score.ranking_tau == -1.0


def test_missing_verdict_scores_no_abstention_credit():
    key, label = _instance()
    # A and Dz correctly abstained; B has no verdict at all (not "correctly didn't abstain").
    answer = Answer(verdicts=(_abstain(label, "A"), _abstain(label, "Dz")), ranking=())
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
    score = score_answer(Answer(verdicts=verdicts, ranking=()), key)
    assert score.adjustment_valid["A"] is None  # correctly abstained - genuinely unanswerable, so ungraded
    assert score.adjustment_valid["Dz"] is None
    assert score.adjustment_valid["B"] is False  # wrongly abstained - graded, and failed
    assert score.adjustment_accuracy == 0.0


def test_ranking_tau_does_not_penalize_order_between_exactly_tied_decoys():
    """Two decoys always share exactly the same true effect (0) - there is no correct order between them.

    An agent's guess either way must score identically, not get graded as a coin flip. seed=5/n_drivers=3
    has two decoys (Dz0, Dz1) alongside three real drivers with distinct, non-zero effects (A0, A1, A2) -
    true order is (A1, A0, A2, Dz0, Dz1) with the last pair a genuine tie.
    """
    scm, _ = generate_gated_procedural(5, n_drivers=3)
    data = sample(scm, 64, seed=5)
    _, key = build_instance(scm, data, seed=5)
    label = dict(key.node_to_label)
    verdicts = tuple(_verdict(label, node) for node in ("A0", "A1", "A2", "Dz0", "Dz1"))

    dz0_first = Answer(verdicts=verdicts, ranking=tuple(label[n] for n in ("A1", "A0", "A2", "Dz0", "Dz1")))
    dz1_first = Answer(verdicts=verdicts, ranking=tuple(label[n] for n in ("A1", "A0", "A2", "Dz1", "Dz0")))

    assert score_answer(dz0_first, key).ranking_tau == score_answer(dz1_first, key).ranking_tau == 1.0
