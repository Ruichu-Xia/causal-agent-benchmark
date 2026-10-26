import re

import numpy as np

from causalagentbench import build_instance, draw_correlation, realize, sample
from causalagentbench.agent.tools import CausalTools
from causalagentbench.scoring.metrics import regression_coefficient
from causalagentbench.types import Role, Scaffold

# A small, hand-built example graph (confounder C, confounded driver A + mediator M, clean driver B,
# decoy Dz) - the same shape used across the test suite (tests/scm/test_scaffold.py). A <- C -> Dz is the
# fork the partial-correlation tests below rely on.
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
)


def _scm(seed: int = 0):
    """An SCM on ``_SCAFFOLD`` with a seeded random correlation matrix."""
    return realize(_SCAFFOLD, draw_correlation(_SCAFFOLD, np.random.default_rng(seed)))


def _tools(seed: int = 0, n: int = 2000) -> tuple[CausalTools, dict[str, str]]:
    """Tools over a placeholder-named instance, plus the node -> label map."""
    scm = _scm(seed)
    data = sample(scm, n, seed=seed)
    instance, key = build_instance(scm, data, seed=seed)
    return CausalTools(instance), dict(key.node_to_label)  # node -> label


def _value(result: str) -> float:
    """The number a tool reports after ``=``."""
    match = re.search(r"= ([+-]?\d+(?:\.\d+)?)", result)
    assert match is not None
    return float(match.group(1))


def test_correlate_marginal_matches_numpy():
    tools, label = _tools()
    idx = {c: i for i, c in enumerate(tools.instance.columns)}
    a, c = tools.instance.data[:, idx[label["A"]]], tools.instance.data[:, idx[label["C"]]]
    expected = float(np.corrcoef(a, c)[0, 1])
    assert abs(_value(tools.correlate(label["A"], label["C"])) - expected) < 5e-4  # 3 decimals


def test_correlation_reports_its_conditioning_set_and_interval():
    tools, label = _tools(n=2000)

    result = tools.correlate(label["A"], label["Dz"], given=[label["C"]])

    assert result.startswith(f"corr({label['A']}, {label['Dz']} | {{{label['C']}}}) = ")
    assert re.search(r", 95% CI \[[+-]\d\.\d{3}, [+-]\d\.\d{3}\]$", result), result


def test_correlate_partial_screens_off_confounder_and_decoy():
    # A <- C -> Dz is a fork through C; conditioning on C blocks the only path, so A _||_ Dz | C.
    tools, label = _tools()
    assert abs(_value(tools.correlate(label["A"], label["Dz"], given=[label["C"]]))) < 0.1


def test_correlate_rejects_unknown_column_or_self_pair():
    tools, label = _tools()
    assert "Error" in tools.correlate("nope", label["A"])
    assert "Error" in tools.correlate(label["A"], label["A"])


def test_correlate_rejects_given_that_would_make_the_precision_matrix_singular():
    # given containing x/y (or a repeat) duplicates a column in the correlation submatrix, which is
    # singular and would otherwise crash np.linalg.inv - this must degrade to an error string instead.
    tools, label = _tools()
    assert "Error" in tools.correlate(label["A"], label["Dz"], given=[label["A"]])
    assert "Error" in tools.correlate(label["A"], label["Dz"], given=[label["Dz"]])
    assert "Error" in tools.correlate(label["A"], label["Dz"], given=[label["C"], label["C"]])


def test_correlate_handles_an_inherently_singular_matrix_as_a_tool_error():
    tools, label = _tools()
    duplicate = tools.instance.data.copy()
    duplicate[:, tools.instance.columns.index(label["B"])] = duplicate[:, tools.instance.columns.index(label["A"])]
    singular_tools = CausalTools(
        tools.instance.__class__(
            data=duplicate,
            columns=tools.instance.columns,
            outcome=tools.instance.outcome,
            candidate_drivers=tools.instance.candidate_drivers,
            prompt=tools.instance.prompt,
        )
    )

    assert "singular" in singular_tools.correlate(label["A"], label["B"])


def test_regress_matches_the_reference_regression_on_the_hidden_order_data():
    # instance.data is data with columns permuted and renamed for display; the OLS coefficient does not
    # depend on column order, so regress() on the true parents must match regression_coefficient() on the
    # canonical-order draw, up to the tool's 3-decimal formatting.
    scm = _scm(0)
    data = sample(scm, 2000, seed=0)
    instance, key = build_instance(scm, data, seed=0)
    tools = CausalTools(instance)
    label = dict(key.node_to_label)

    for node in ("A", "B", "Dz"):
        parents = list(scm.scaffold.parents(node))
        expected = regression_coefficient(data, scm.scaffold.node_names, "Y", node, parents)
        coef = _value(tools.regress(label[node], adjust=[label[z] for z in parents]))
        assert abs(coef - expected) < 5e-4


def test_regress_reports_its_adjustment_set_and_standard_error():
    tools, label = _tools(n=2000)

    result = tools.regress(label["A"], adjust=[label["C"]])

    pattern = rf"regress\({label['A']} \| \{{{label['C']}\}}\) = [+-]\d\.\d{{3}} \(se \d\.\d{{3}}\)"
    assert re.fullmatch(pattern, result), result


def test_regress_rejects_unknown_column_treatment_as_outcome_and_self_adjust():
    tools, label = _tools()
    assert "Error" in tools.regress("nope")
    assert "Error" in tools.regress(tools.instance.outcome)
    assert "Error" in tools.regress(label["A"], adjust=[label["A"]])
    assert "Error" in tools.regress(label["A"], adjust=[tools.instance.outcome])
    assert "Error" in tools.regress(label["A"], adjust=[label["C"], label["C"]])


def test_schema_violating_argument_types_are_reported_not_coerced():
    """The declared type is the model's to honor; a violation is named plainly, never quietly accepted.

    Observed live: a model sent a quoted number where an integer was declared and got an error from deep
    inside ``min()`` that named nothing it could act on. The fix is the message, not tolerance: the bad
    value still fails, but says why, so a model can correct itself on the next turn.
    """
    tools, label = _tools()
    for bad_cols in (label["C"], 5, [1, 2]):  # a bare string is not a list of columns
        assert "Error: given must be a list" in tools.correlate(label["A"], label["B"], given=bad_cols)
        assert "Error: adjust must be a list" in tools.regress(label["A"], adjust=bad_cols)


def test_omitted_column_sets_still_default_to_empty():
    """Absent is not malformed: `given`/`adjust` are optional, and omitting them is the marginal case."""
    tools, label = _tools()
    assert "Error" not in tools.correlate(label["A"], label["B"])
    assert "Error" not in tools.regress(label["A"])


# ---- causal discovery ---------------------------------------------------------------------------------


def _edges(result: str) -> list[tuple[str, str, str]]:
    """Parse `discover`'s edge lines into (left, operator, right) triples."""
    return [
        (m.group(1), m.group(2), m.group(3))
        for m in (re.match(r"\s+(\S+) (->|--) (\S+)$", line) for line in result.splitlines())
        if m is not None
    ]


def test_discover_returns_a_cpdag_over_display_columns_only():
    tools, _ = _tools()
    result = tools.discover()
    edges = _edges(result)
    assert edges, result
    columns = set(tools.instance.columns)
    # Hidden node names must never reach the agent - the tool reads the public table, like every other one.
    assert all({left, right} <= columns for left, _, right in edges)
    assert "Oriented" in result and "Undirected" in result


def test_discover_recovers_true_edges_without_inventing_any():
    """Measured behaviour, so the agent can be told what to trust: precise, but incomplete.

    Across six generated instances at n=2000 PC never proposed an edge that is not in the true graph
    (precision 1.00) while recovering 68-93% of them. The thresholds here sit below the observed range
    rather than on it, so ordinary sampling variation does not turn this into a flake.
    """
    tools, node_to_label = _tools()
    label_to_node = {label: node for node, label in node_to_label.items()}
    found = {frozenset((label_to_node[left], label_to_node[right])) for left, _, right in _edges(tools.discover())}
    true_skeleton = {frozenset(edge) for edge in _SCAFFOLD.edges}

    assert found <= true_skeleton  # no invented edge
    assert len(found & true_skeleton) / len(true_skeleton) >= 0.6


def test_discover_rejects_an_alpha_outside_the_unit_interval():
    tools, _ = _tools()
    assert tools.discover(alpha=1.5).startswith("Error:")
    assert tools.discover(alpha=0.0).startswith("Error:")
    assert tools.discover(alpha="0.01").startswith("Error:")  # a string is reported, never coerced
