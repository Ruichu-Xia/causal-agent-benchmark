import pytest

from causalagentbench.task.answer import Answer, AnswerParseError, parse_answer, parse_answer_for_instance


@pytest.mark.parametrize("value", ['"NaN"', "NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_effects_are_rejected(value):
    text = '<answer>{"verdicts":[{"driver":"X1","effect":' + value + "}]}</answer>"
    with pytest.raises(AnswerParseError):
        parse_answer_for_instance(text, ("X1",))


_VALID = """
reasoning here...
<answer>
{"verdicts": [{"driver": "X1", "adjustment_set": ["X2"], "effect": 0.4},
              {"driver": "X3", "adjustment_set": [], "effect": -0.1}]}
</answer>
"""


def test_parses_valid_answer_block():
    answer = parse_answer(_VALID)
    assert isinstance(answer, Answer)
    assert answer.verdicts[0].driver == "X1"
    assert answer.verdicts[0].adjustment_set == ("X2",)
    assert answer.verdicts[0].effect == 0.4
    assert answer.verdicts[1].adjustment_set == ()
    assert answer.verdicts[1].effect == -0.1


def test_last_block_wins():
    text = (
        '<answer>{"verdicts": [{"driver": "A", "adjustment_set": [], "effect": 1.0}]}</answer>'
        '<answer>{"verdicts": [{"driver": "B", "adjustment_set": [], "effect": 2.0}]}</answer>'
    )
    assert parse_answer(text).verdicts[0].driver == "B"


def test_missing_block_raises():
    with pytest.raises(AnswerParseError, match="no <answer>"):
        parse_answer("I decline to answer.")


def test_malformed_json_raises():
    with pytest.raises(AnswerParseError, match="invalid answer"):
        parse_answer("<answer>{not json}</answer>")


def test_extra_field_raises():
    with pytest.raises(AnswerParseError, match="invalid answer"):
        parse_answer('<answer>{"verdicts": [], "bogus": 1}</answer>')


def test_missing_field_raises():
    with pytest.raises(AnswerParseError, match="invalid answer"):
        parse_answer("<answer>{}</answer>")


def test_unknown_verdict_field_raises():
    # e.g. a stale "identifiable" field from prompt drift - the contract has no such field.
    verdict = '{"driver": "X1", "identifiable": true, "adjustment_set": []}'
    with pytest.raises(AnswerParseError, match="invalid answer"):
        parse_answer(f'<answer>{{"verdicts": [{verdict}]}}</answer>')


def test_every_driver_must_carry_an_effect_estimate():
    """The ranking is derived from the estimates, so a driver without one has no position to hold."""
    missing_estimate = """<answer>{"verdicts": [
        {"driver": "X1", "adjustment_set": []},
        {"driver": "X3", "adjustment_set": [], "effect": 0.2}
    ]}</answer>"""
    with pytest.raises(AnswerParseError, match="effect estimate"):
        parse_answer_for_instance(missing_estimate, ("X1", "X3"))


def test_there_is_no_way_to_decline_a_driver():
    """Guards the deletion.

    An abstain option let the agent shrink its own denominator on three metrics at once, and made the
    correct answer depend on the column names - the axis this benchmark exists to vary independently.
    """
    abstained = """<answer>{"verdicts": [
        {"driver": "X1", "adjustment_set": [], "abstain": true},
        {"driver": "X3", "adjustment_set": [], "effect": 0.2}
    ]}</answer>"""
    with pytest.raises(AnswerParseError, match="extra_forbidden"):
        parse_answer_for_instance(abstained, ("X1", "X3"))


def test_the_identifiability_judgement_is_optional_to_parse_but_recorded():
    """A missing judgement parses but is penalized at scoring time.

    Keeping the two layers separate is what makes "malformed answer" and "declined one field"
    distinguishable in the results.
    """
    without = """<answer>{"verdicts": [
        {"driver": "X1", "adjustment_set": [], "effect": 0.1},
        {"driver": "X3", "adjustment_set": [], "effect": 0.2}
    ]}</answer>"""
    answer = parse_answer_for_instance(without, ("X1", "X3"))
    assert answer.verdicts[0].identifiable_from_data is None


def test_instance_contract_requires_each_candidate_exactly_once_in_verdicts():
    valid = """<answer>{"verdicts": [
        {"driver": "X1", "adjustment_set": [], "effect": 0.1},
        {"driver": "X3", "adjustment_set": [], "effect": 0.3}
    ]}</answer>"""
    answer = parse_answer_for_instance(valid, ("X1", "X3"))
    assert [v.driver for v in answer.verdicts] == ["X1", "X3"]

    missing = '<answer>{"verdicts": []}</answer>'
    with pytest.raises(AnswerParseError, match="every candidate"):
        parse_answer_for_instance(missing, ("X1", "X3"))

    repeated = """<answer>{"verdicts": [
        {"driver": "X1", "adjustment_set": [], "effect": 0.1},
        {"driver": "X1", "adjustment_set": [], "effect": 0.2}
    ]}</answer>"""
    with pytest.raises(AnswerParseError, match="repeat"):
        parse_answer_for_instance(repeated, ("X1", "X3"))


def test_instance_contract_rejects_invalid_adjustment_columns():
    unknown_column = """<answer>{"verdicts": [
        {"driver": "X1", "adjustment_set": ["missing"], "effect": 0.1},
        {"driver": "X3", "adjustment_set": [], "effect": 0.3}
    ]}</answer>"""
    with pytest.raises(AnswerParseError, match="not present"):
        parse_answer_for_instance(unknown_column, ("X1", "X3"), columns=("X1", "X2", "X3", "Y"), outcome="Y")

    outcome_in_adjustment = """<answer>{"verdicts": [
        {"driver": "X1", "adjustment_set": ["Y"], "effect": 0.1},
        {"driver": "X3", "adjustment_set": [], "effect": 0.3}
    ]}</answer>"""
    with pytest.raises(AnswerParseError, match="outcome"):
        parse_answer_for_instance(
            outcome_in_adjustment,
            ("X1", "X3"),
            columns=("X1", "X2", "X3", "Y"),
            outcome="Y",
        )
