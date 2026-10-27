import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import transfer_bnrep
import transfer_eval

from causalagentbench.scm.generate import realize
from causalagentbench.types import Role, Scaffold


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("reasoning\nAnswer: Yes", True),
        ("**Answer: No**", False),
        ("So the answer is yes.", True),
        ("<answer>no</answer>", False),
        ("reasoning\nNo.", False),
        ("Answer: Yes\nOn reflection, Answer: No", False),
        ("no idea", None),
        ("", None),
    ],
)
def test_extract_takes_the_last_stated_answer(reply, expected):
    """The pre-registered answer rule."""
    assert transfer_eval.extract(reply) is expected


def test_bnrep_standardization_keeps_the_published_coefficients():
    """``realize`` on the correlation recovers each raw coefficient rescaled by sd(parent) / sd(child)."""
    network = {
        "nodes": [
            {"node": "a", "parents": {}, "sd": 2.0},
            {"node": "b", "parents": {"a": 1.5}, "sd": 0.5},
            {"node": "c", "parents": {"a": -0.3, "b": 2.0}, "sd": 1.0},
        ]
    }
    nodes, edges, sigma = transfer_bnrep.correlation(network)
    roles = dict.fromkeys(nodes, Role.CONFOUNDER) | {"c": Role.OUTCOME}
    scm = realize(Scaffold(nodes, roles, edges, ("a", "b"), "c"), sigma)
    var_a = 4.0
    var_b = 1.5**2 * var_a + 0.25
    var_c = (-0.3) ** 2 * var_a + 2.0**2 * var_b + 2 * (-0.3) * 2.0 * 1.5 * var_a + 1.0
    sd = np.sqrt([var_a, var_b, var_c])
    assert scm.W[0, 1] == pytest.approx(1.5 * sd[0] / sd[1])
    assert scm.W[1, 2] == pytest.approx(2.0 * sd[1] / sd[2])
    assert scm.total_effect_on_outcome("a") == pytest.approx((-0.3 + 1.5 * 2.0) * sd[0] / sd[2])
