import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import confirm_v3


def _choice(i: int, group: str, answer: str, options: str) -> dict:
    """A QRData-style multiple-choice record in a group with the given choice list."""
    prompt = f'question {i}\n... "Answer: <answer>", where <answer> is {options}.'
    return {"id": f"q{i}", "kind": "qrdata_choice", "answer": answer, "prompt": prompt, "meta": {"keywords": [group]}}


def test_qrdata_weights_sum_to_one_and_cap_answer_habits():
    """Each gold answer's items weigh the same in a group, so a habit that ignores the item gets at most 1/K."""
    items = [_choice(i, "discovery", "C" if i < 6 else "A" if i < 8 else "B", "A, B or C") for i in range(10)]
    items += [_choice(10 + i, "pairwise", "A" if i < 3 else "B", "A or B") for i in range(5)]
    items += [{"id": "n0", "kind": "qrdata_number", "answer": "4.0", "prompt": "", "meta": {"keywords": ["ate"]}}]
    w = confirm_v3.weights("qrdata_causal", items)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["n0"] == pytest.approx(1 / 16)
    for group, letters, size in (("discovery", "ABC", 10), ("pairwise", "AB", 5)):
        members = [item for item in items if item["meta"]["keywords"] == [group]]
        for letter in letters:
            constant = sum(w[item["id"]] for item in members if item["answer"] == letter)
            assert constant == pytest.approx(size / 16 / len(letters))


def test_sign_flip_p_is_small_for_a_clear_gain_and_near_one_half_for_none():
    """A consistent gain gives a p-value near the resolution limit; symmetric noise does not."""
    gain = np.full(200, 0.01)
    assert confirm_v3.sign_flip_p(gain) < 0.001
    noise = np.random.default_rng(1).normal(0, 0.01, size=200)
    noise -= noise.mean()
    assert 0.3 < confirm_v3.sign_flip_p(noise) < 0.7


def test_holm_steps_down_in_order():
    """The smallest p-value is tested at alpha/k, and testing stops at the first failure."""
    p = {"a": 0.004, "b": 0.006, "c": 0.02, "d": 0.3}
    assert confirm_v3.holm(p, 0.025) == {"a": True, "b": True, "c": False, "d": False}


def test_cluster_differences_sum_to_the_weighted_score_difference(monkeypatch):
    """Clusters (shared files) add up the weighted item differences, and the bootstrap brackets their sum."""
    items = [_choice(i, "discovery", "A" if i % 2 else "B", "A or B") for i in range(6)]
    for i, item in enumerate(items):
        item["files"] = [f"f{i // 2}.csv"]
    monkeypatch.setattr(confirm_v3, "items", lambda name: items)
    a = {item["id"]: 1.0 for item in items}
    b = {item["id"]: float(i < 3) for i, item in enumerate(items)}
    differences = confirm_v3.cluster_differences("qrdata_causal", a, b)
    assert len(differences) == 3
    assert differences.sum() == pytest.approx(0.5)
    low, high = confirm_v3.bootstrap_interval(differences)
    assert low <= 0.5 <= high


def test_bootstrap_interval_of_identical_clusters_is_their_sum():
    """With no variation across clusters, every resample gives the observed difference."""
    assert confirm_v3.bootstrap_interval(np.full(10, 0.01)) == pytest.approx((0.1, 0.1))
