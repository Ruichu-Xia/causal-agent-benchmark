import numpy as np

from causalagentbench import build_instance, generate_gated, sample
from causalagentbench.agent.tools import CausalTools
from causalagentbench.scoring.baselines import oracle_effects


def _tools(seed: int = 0, n: int = 2000) -> tuple[CausalTools, dict[str, str]]:
    scm, _ = generate_gated(seed)
    data = sample(scm, n, seed=seed)
    instance, key = build_instance(scm, data, seed=seed)
    return CausalTools(instance), dict(key.node_to_label)  # node -> label


def _value(result: str) -> float:
    return float(result.rsplit("=", 1)[-1])


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


def test_correlate_partial_screens_off_confounder_and_decoy():
    # A <- C -> Dz is a fork through C; conditioning on C blocks the only path, so A _||_ Dz | C.
    tools, label = _tools()
    assert abs(_value(tools.correlate(label["A"], label["Dz"], given=[label["C"]]))) < 0.1


def test_correlate_rejects_unknown_column_or_self_pair():
    tools, label = _tools()
    assert "Error" in tools.correlate("nope", label["A"])
    assert "Error" in tools.correlate(label["A"], label["A"])


def test_regress_matches_the_oracle_solver_on_the_same_data():
    # instance.data is data with columns permuted+renamed for display; the OLS coefficient does not
    # depend on column order, so regress() under the gold adjustment set must match oracle_effects()
    # (an already-tested reference solver) computed on the identical draw, near-exactly.
    scm, _ = generate_gated(0)
    data = sample(scm, 2000, seed=0)
    instance, key = build_instance(scm, data, seed=0)
    tools = CausalTools(instance)
    label = dict(key.node_to_label)
    oracle = oracle_effects(data, scm)

    for node in ("A", "B", "Dz"):
        adjust = [label[z] for z in scm.scaffold.correct_adjustment.get(node, ())]
        coef = _value(tools.regress(label[node], adjust=adjust))
        assert abs(coef - oracle[node]) < 5e-5  # tool formats to 4 decimals


def test_regress_rejects_unknown_column_treatment_as_outcome_and_self_adjust():
    tools, label = _tools()
    assert "Error" in tools.regress("nope")
    assert "Error" in tools.regress(tools.instance.outcome)
    assert "Error" in tools.regress(label["A"], adjust=[label["A"]])
