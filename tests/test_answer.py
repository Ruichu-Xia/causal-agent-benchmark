import pytest

from causalagentbench.answer import Answer, AnswerParseError, parse_answer

_VALID = """
reasoning here...
<answer>
{"verdicts": [{"driver": "X1", "identifiable": true, "adjustment_set": ["X2"]},
              {"driver": "X3", "identifiable": false, "adjustment_set": []}],
 "ranking": ["X1"]}
</answer>
"""


def test_parses_valid_answer_block():
    answer = parse_answer(_VALID)
    assert isinstance(answer, Answer)
    assert answer.ranking == ("X1",)
    assert answer.verdicts[0].driver == "X1"
    assert answer.verdicts[0].adjustment_set == ("X2",)
    assert answer.verdicts[1].identifiable is False


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
