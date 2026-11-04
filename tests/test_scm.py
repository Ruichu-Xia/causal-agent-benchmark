import numpy as np
import pytest

from causalagentbench.graphs import equivalence_class
from causalagentbench.scm import EXPLAINED_VARIANCE, Scaffold, draw_correlation, marginal_covariance, realize

_MOTIF = Scaffold(
    node_names=("C", "A", "B", "M", "Dz", "Y", "O"),
    edges=(("C", "A"), ("C", "Y"), ("C", "Dz"), ("A", "M"), ("A", "Y"), ("M", "Y"), ("B", "Y"), ("Y", "O")),
    outcome="Y",
)


def _sigma(seed: int = 0, motif: Scaffold = _MOTIF) -> np.ndarray:
    """A correlation matrix drawn on ``motif``."""
    return draw_correlation(motif, np.random.default_rng(seed))


def _with_edges(motif: Scaffold, edges) -> Scaffold:
    """``motif`` with its edges replaced (for example, by another member of its class)."""
    return Scaffold(motif.node_names, tuple(edges), motif.outcome)


def test_correlation_is_a_valid_unit_variance_matrix():
    sigma = _sigma()
    assert np.allclose(np.diag(sigma), 1.0)
    assert np.allclose(sigma, sigma.T)
    assert np.all(np.linalg.eigvalsh(sigma) > 0)


def test_correlation_is_deterministic_per_generator_seed():
    assert np.array_equal(_sigma(3), _sigma(3))
    assert not np.array_equal(_sigma(3), _sigma(4))


def test_parents_explain_a_bounded_share_of_each_node():
    sigma = _sigma(5)
    index = {name: i for i, name in enumerate(_MOTIF.node_names)}
    for name in _MOTIF.node_names:
        parents = [index[p] for p in _MOTIF.parents(name)]
        if not parents:
            continue
        j = index[name]
        r2 = sigma[j, parents] @ np.linalg.solve(sigma[np.ix_(parents, parents)], sigma[parents, j])
        assert EXPLAINED_VARIANCE[0] - 1e-12 <= r2 <= EXPLAINED_VARIANCE[1] + 1e-12


def test_members_differ_in_weights_but_not_in_distribution():
    """Covariance transfer: no statistic of the data can tell which member is the true graph."""
    sigma = _sigma(1)
    members = equivalence_class(_MOTIF)
    scms = [realize(_with_edges(_MOTIF, edges), sigma) for edges in members]
    assert len({scm.W.tobytes() for scm in scms}) == len(members) > 1
    for scm in scms:
        assert np.max(np.abs(marginal_covariance(scm.W, scm.noise_var) - sigma)) < 1e-10


@pytest.mark.parametrize(
    "edges",
    [
        # Y -> B instead of B -> Y removes the v-structures at Y.
        [e for e in _MOTIF.edges if e != ("B", "Y")] + [("Y", "B")],
        # Dropping B -> Y claims B and Y are independent, but sigma correlates them.
        [e for e in _MOTIF.edges if e != ("B", "Y")],
        # C -> A <- M adds a v-structure the class does not have.
        [e for e in _MOTIF.edges if e != ("A", "M")] + [("M", "A")],
    ],
)
def test_realize_rejects_a_dag_outside_the_class(edges):
    with pytest.raises(ValueError, match="not Markov"):
        realize(_with_edges(_MOTIF, edges), _sigma(2))
