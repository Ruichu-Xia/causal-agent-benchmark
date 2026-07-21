from causalagentbench import build_instance, generate_gated, sample
from causalagentbench.scoring.score import score_answer, score_output
from causalagentbench.task.answer import Answer, DriverVerdict
from causalagentbench.task.instance import InstanceKey


def _instance() -> tuple[InstanceKey, dict[str, str]]:
    scm, _ = generate_gated(0)
    data = sample(scm, 64, seed=0)
    _, key = build_instance(scm, data, seed=0)
    return key, dict(key.node_to_label)  # node -> label


def _verdict(label: dict[str, str], node: str, adjust: tuple[str, ...] = ()) -> DriverVerdict:
    return DriverVerdict(driver=label[node], adjustment_set=tuple(label[z] for z in adjust))


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
    assert score_answer(partial, key).ranking_tau == 0.0


def test_parse_failure_state():
    key, _ = _instance()
    score = score_output("no answer block", key)
    assert score.parse_ok is False
    assert score.ranking_tau == 0.0
    assert score.adjustment_accuracy == 0.0
    assert not any(score.adjustment_valid.values())


def test_alignment_is_carried_through():
    key, label = _instance()
    # abstract labeler -> alignment is None; the field is passed through for stratification.
    assert score_answer(_perfect(key, label), key).alignment is None
