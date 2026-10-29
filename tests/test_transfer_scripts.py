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


@pytest.mark.skipif(not transfer_eval.DSEP.exists(), reason="d-separation files are not redistributed")
def test_dsep_labels_match_networkx():
    """Every d-separation label agrees with d-separation in the premise's graph, re-derived from the text."""
    import re

    import networkx as nx

    for item in transfer_eval.items("dsep"):
        premise, question = item["prompt"].split("\n\n")[0].removeprefix("Premise: ").split(" Are ")
        graph = nx.DiGraph(re.findall(r"(\S+) causes (\S+?)\.?(?= |$)", premise))
        x, y, given = re.fullmatch(r"(\S+) and (\S+) d-separated(?: given \{(.*)\})?\?", question).groups()
        z = {name.strip() for name in (given or "").split(",") if name.strip()}
        graph.add_nodes_from([x, y, *z])
        assert nx.is_d_separator(graph, {x}, {y}, z) is item["label"], item["id"]


def test_holm_steps_down_and_stops_at_the_first_acceptance():
    """Holm compares the k-th smallest of m p-values with alpha / (m - k) and stops at the first failure."""
    import confirm_v2

    p = {"a": 0.001, "b": 0.012, "c": 0.013, "d": 0.2}
    # thresholds 0.025/4 = 0.00625, 0.025/3 = 0.00833, 0.025/2, 0.025: a passes, b fails, so c and d fail too
    assert confirm_v2.holm(p, 0.025) == {"a": True, "b": False, "c": False, "d": False}
    assert confirm_v2.holm({"a": 0.006, "b": 0.012}, 0.025) == {"a": True, "b": True}


def test_bootstrap_p_value_counts_draws_at_or_below_zero():
    """The one-sided p-value is (1 + #draws <= 0) / (1 + #draws)."""
    assert transfer_eval.p_value([-0.1, 0.0, 0.2, 0.3]) == 3 / 5
    assert transfer_eval.p_value([0.1] * 99) == 1 / 100
