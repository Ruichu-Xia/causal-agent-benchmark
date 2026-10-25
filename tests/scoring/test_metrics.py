import numpy as np
import pytest

from causalagentbench.scoring.metrics import regression_coefficient


def _table(seed: int = 0, n: int = 500) -> tuple[np.ndarray, tuple[str, ...]]:
    """Columns x, z, y with y = 2x + 3z + 1 exactly, and x correlated with z."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    x = 0.5 * z + rng.standard_normal(n)
    return np.column_stack([x, z, 2 * x + 3 * z + 1]), ("x", "z", "y")


def test_adjusted_coefficient_recovers_the_exact_slope():
    data, columns = _table()
    assert regression_coefficient(data, columns, "y", "x", ["z"]) == pytest.approx(2.0)
    assert regression_coefficient(data, columns, "y", "z", ["x"]) == pytest.approx(3.0)


def test_unadjusted_coefficient_is_the_simple_regression_slope():
    data, columns = _table()
    x, y = data[:, 0], data[:, 2]
    expected = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    assert regression_coefficient(data, columns, "y", "x") == pytest.approx(expected)


def test_coefficient_ignores_column_order_and_location_shifts():
    data, columns = _table()
    reordered = data[:, [2, 0, 1]] + np.array([5.0, -3.0, 7.0])
    assert regression_coefficient(reordered, ("y", "x", "z"), "y", "x", ["z"]) == pytest.approx(
        regression_coefficient(data, columns, "y", "x", ["z"])
    )
