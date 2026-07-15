import pytest

from causalagentbench.metrics import kendall_tau


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
