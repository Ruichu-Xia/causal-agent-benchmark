import pytest

from causalagentbench.grading import final_reply, grade


@pytest.mark.parametrize(
    ("kind", "answer", "reply", "score"),
    [
        ("yesno", "yes", "so...\nAnswer: Yes", 1.0),
        ("yesno", "yes", "**Answer: No**", 0.0),
        ("yesno", "no", "Answer: yes\nWait. Answer: no.", 1.0),
        ("yesno", "no", "### Final Answer:\n\nAnswer: No", 1.0),  # a heading above the answer line
        ("yesno", "yes", "**Answer**: Yes", 1.0),
        ("yesno", "yes", "Answer:\nYes", 0.0),  # the value must be on the answer line
        ("yesno", "yes", "Answer: <answer> Yes.", 1.0),  # the instruction's placeholder, echoed
        ("number", "0.53", "Answer: <answer>0.53</answer>", 1.0),
        ("number", "0.14", "$$\\boxed{0.14}$$", 0.0),  # no answer line
        ("number", "-0.12", "Answer: $-0.12$", 1.0),
        ("number", "0.12", "Answer: 0.13", 1.0),
        ("number", "0.12", "Answer: 0.14", 0.0),
        ("letter", "b", "Answer: (B)", 1.0),
        ("choice", "both", "Answer: Both", 1.0),
        ("choice", "1", "the answer is 1", 0.0),
    ],
)
def test_grade(kind, answer, reply, score):
    """The last "Answer:" line decides; numbers within the tolerance count."""
    assert grade(kind, answer, reply) == score


def test_final_reply_is_the_last_assistant_text():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "python"}}]},
        {"role": "tool", "content": "0.13"},
        {"role": "assistant", "content": [{"type": "text", "text": "Answer: Yes"}]},
    ]
    assert final_reply(messages) == "Answer: Yes"
    assert final_reply("Answer: No") == "Answer: No"
