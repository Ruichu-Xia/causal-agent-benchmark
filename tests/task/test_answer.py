import pytest

from causalagentbench.task.answer import Answer, AnswerParseError, parse_answer, parse_answer_for_instance

_VALID = """
reasoning here...
<answer>
{"verdicts": [{"driver": "X1", "adjustment_set": ["X2"]},
              {"driver": "X3", "adjustment_set": []}],
 "ranking": ["X1"]}
</answer>
"""


def test_parses_valid_answer_block():
    answer = parse_answer(_VALID)
    assert isinstance(answer, Answer)
    assert answer.ranking == ("X1",)
    assert answer.verdicts[0].driver == "X1"
    assert answer.verdicts[0].adjustment_set == ("X2",)
    assert answer.verdicts[1].adjustment_set == ()


def test_last_block_wins():
    text = '<answer>{"verdicts": [], "ranking": ["A"]}</answer><answer>{"verdicts": [], "ranking": ["B"]}</answer>'
    assert parse_answer(text).ranking == ("B",)


def test_missing_block_raises():
    with pytest.raises(AnswerParseError, match="no <answer>"):
        parse_answer("I decline to answer.")


def test_malformed_json_raises():
    with pytest.raises(AnswerParseError, match="invalid answer"):
        parse_answer("<answer>{not json}</answer>")


def test_extra_field_raises():
    with pytest.raises(AnswerParseError, match="invalid answer"):
        parse_answer('<answer>{"verdicts": [], "ranking": [], "bogus": 1}</answer>')


def test_missing_field_raises():
    with pytest.raises(AnswerParseError, match="invalid answer"):
        parse_answer('<answer>{"verdicts": []}</answer>')


def test_unknown_verdict_field_raises():
    # e.g. a stale "identifiable" field from prompt drift - the contract has no such field.
    verdict = '{"driver": "X1", "identifiable": true, "adjustment_set": []}'
    with pytest.raises(AnswerParseError, match="invalid answer"):
        parse_answer(f'<answer>{{"verdicts": [{verdict}], "ranking": []}}</answer>')


def test_instance_contract_requires_each_candidate_once_in_verdicts_and_ranking():
    valid = """<answer>{"verdicts": [
        {"driver": "X1", "adjustment_set": []},
        {"driver": "X3", "adjustment_set": []}
    ], "ranking": ["X3", "X1"]}</answer>"""
    answer = parse_answer_for_instance(valid, ("X1", "X3"))
    assert answer.ranking == ("X3", "X1")

    missing = '<answer>{"verdicts": [], "ranking": []}</answer>'
    with pytest.raises(AnswerParseError, match="every candidate"):
        parse_answer_for_instance(missing, ("X1", "X3"))

    repeated = """<answer>{"verdicts": [
        {"driver": "X1", "adjustment_set": []},
        {"driver": "X1", "adjustment_set": []}
    ], "ranking": ["X1", "X1"]}</answer>"""
    with pytest.raises(AnswerParseError, match="repeat"):
        parse_answer_for_instance(repeated, ("X1", "X3"))


def test_instance_contract_rejects_invalid_adjustment_columns():
    unknown_column = """<answer>{"verdicts": [
        {"driver": "X1", "adjustment_set": ["missing"]},
        {"driver": "X3", "adjustment_set": []}
    ], "ranking": ["X1", "X3"]}</answer>"""
    with pytest.raises(AnswerParseError, match="not present"):
        parse_answer_for_instance(unknown_column, ("X1", "X3"), columns=("X1", "X2", "X3", "Y"), outcome="Y")

    outcome_in_adjustment = """<answer>{"verdicts": [
        {"driver": "X1", "adjustment_set": ["Y"]},
        {"driver": "X3", "adjustment_set": []}
    ], "ranking": ["X1", "X3"]}</answer>"""
    with pytest.raises(AnswerParseError, match="outcome"):
        parse_answer_for_instance(
            outcome_in_adjustment,
            ("X1", "X3"),
            columns=("X1", "X2", "X3", "Y"),
            outcome="Y",
        )
