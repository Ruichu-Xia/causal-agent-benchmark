import pytest

from causalagentbench.scoring.metrics import kendall_tau, kendall_tau_zscore


def test_identical_orderings():
    assert kendall_tau([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_reversed_orderings():
    assert kendall_tau([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0


def test_single_swap():
    # 5 concordant, 1 discordant out of 6 pairs
    assert kendall_tau([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 4.0, 3.0]) == pytest.approx(4 / 6)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        kendall_tau([1.0, 2.0], [1.0])


def test_zscore_same_tau_is_more_extreme_at_larger_n():
    # The same tau is less surprising from a small ranking task than a large one - raw tau alone can't
    # distinguish "got 1/3 pairs right by luck on a 3-item task" from "on an 8-item task."
    small = kendall_tau_zscore(1 / 3, 3)
    large = kendall_tau_zscore(1 / 3, 8)
    assert 0 < small < large


def test_zscore_sign_matches_tau_sign():
    assert kendall_tau_zscore(1.0, 5) > 0
    assert kendall_tau_zscore(-1.0, 5) < 0
    assert kendall_tau_zscore(0.0, 5) == 0.0


def test_zscore_undefined_below_two_items():
    assert kendall_tau_zscore(1.0, 1) is None
    assert kendall_tau_zscore(1.0, 0) is None
