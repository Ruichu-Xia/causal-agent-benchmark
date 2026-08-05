import re

import numpy as np

from causalagentbench import build_instance, generate, sample
from causalagentbench.agent.tools import CausalTools
from causalagentbench.scoring.baselines import oracle_effects
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
    correct_adjustment={"A": ("C",), "B": (), "Dz": ("C",)},
)


def _tools(seed: int = 0, n: int = 2000) -> tuple[CausalTools, dict[str, str]]:
    scm = generate(seed, _SCAFFOLD)
    data = sample(scm, n, seed=seed)
    instance, key = build_instance(scm, data, seed=seed)
    return CausalTools(instance), dict(key.node_to_label)  # node -> label


def _value(result: str) -> float:
    match = re.search(r"= ([+-]?\d+(?:\.\d+)?)", result)
    assert match is not None
    return float(match.group(1))


def test_preview_returns_header_and_bounded_rows():
    tools, _ = _tools()
    lines = tools.preview(n=3).strip().splitlines()
    assert lines[0] == ",".join(tools.instance.columns)
    assert len(lines) == 4  # header + 3 rows
    assert len(tools.preview(n=1000).strip().splitlines()) == 21  # clamped to the max (20 rows + header)


def test_correlate_marginal_matches_numpy():
    tools, label = _tools()
    idx = {c: i for i, c in enumerate(tools.instance.columns)}
    a, c = tools.instance.data[:, idx[label["A"]]], tools.instance.data[:, idx[label["C"]]]
    expected = float(np.corrcoef(a, c)[0, 1])
    assert abs(_value(tools.correlate(label["A"], label["C"])) - expected) < 1e-4


def test_correlation_reports_sample_size_and_uncertainty():
    tools, label = _tools(n=2000)

    result = tools.correlate(label["A"], label["C"])

    assert "n=2000" in result
    assert "approx_95%CI" in result


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


def test_regress_matches_the_oracle_solver_on_the_same_data():
    # instance.data is data with columns permuted+renamed for display; the OLS coefficient does not
    # depend on column order, so regress() under the gold adjustment set must match oracle_effects()
    # (an already-tested reference solver) computed on the identical draw, near-exactly.
    scm = generate(0, _SCAFFOLD)
    data = sample(scm, 2000, seed=0)
    instance, key = build_instance(scm, data, seed=0)
    tools = CausalTools(instance)
    label = dict(key.node_to_label)
    oracle = oracle_effects(data, scm)

    for node in ("A", "B", "Dz"):
        adjust = [label[z] for z in scm.scaffold.correct_adjustment.get(node, ())]
        coef = _value(tools.regress(label[node], adjust=adjust))
        assert abs(coef - oracle[node]) < 5e-5  # tool formats to 4 decimals


def test_regress_reports_sample_size_and_uncertainty():
    tools, label = _tools(n=2000)

    result = tools.regress(label["A"], adjust=[label["C"]])

    assert "n=2000" in result
    assert "se=" in result
    assert "approx_95%CI" in result


def test_regress_rejects_unknown_column_treatment_as_outcome_and_self_adjust():
    tools, label = _tools()
    assert "Error" in tools.regress("nope")
    assert "Error" in tools.regress(tools.instance.outcome)
    assert "Error" in tools.regress(label["A"], adjust=[label["A"]])
    assert "Error" in tools.regress(label["A"], adjust=[tools.instance.outcome])
    assert "Error" in tools.regress(label["A"], adjust=[label["C"], label["C"]])


def test_schema_violating_argument_types_are_reported_not_coerced():
    """The declared type is the model's to honor; a violation is named plainly, never quietly accepted.

    Observed live: a model sent ``preview(n="10")`` and got
    ``'<' not supported between instances of 'int' and 'str'`` from inside ``min()`` - an error naming
    nothing it could act on, which it read as the tool being broken before abandoning tool arguments
    entirely. The fix is the message, not tolerance: ``"10"`` still fails, but says why, so a model can
    correct itself on the next turn.
    """
    tools, label = _tools()
    for bad in ("10", 10.0, "many", {"rows": 5}, True):  # True: bool subclasses int, still a mistake
        assert "Error: n must be an integer" in tools.preview(n=bad)
    for bad_cols in (label["C"], 5, [1, 2]):  # a bare string is not a list of columns
        assert "Error: given must be a list" in tools.correlate(label["A"], label["B"], given=bad_cols)
        assert "Error: adjust must be a list" in tools.regress(label["A"], adjust=bad_cols)


def test_omitted_column_sets_still_default_to_empty():
    """Absent is not malformed: `given`/`adjust` are optional, and omitting them is the marginal case."""
    tools, label = _tools()
    assert "Error" not in tools.correlate(label["A"], label["B"])
    assert "Error" not in tools.regress(label["A"])
    assert "Error" not in tools.preview()
