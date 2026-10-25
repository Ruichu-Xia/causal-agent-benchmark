import numpy as np
import pytest

from causalagentbench.scm.effects import marginal_covariance
from causalagentbench.scm.gadgets import procedural_scaffold
from causalagentbench.scm.generate import EXPLAINED_VARIANCE, draw_correlation, realize
from causalagentbench.scm.identifiability import equivalence_class, possible_effects
from causalagentbench.scm.sample import sample
from causalagentbench.types import Role, Scaffold

_MOTIF = Scaffold(
    node_names=("C", "A", "B", "M", "Dz", "Y", "O"),
    roles={
        "C": Role.CONFOUNDER,
        "A": Role.DRIVER_CONFOUNDED,
        "B": Role.DRIVER_CLEAN,
        "M": Role.MEDIATOR,
        "Dz": Role.DECOY,
        "Y": Role.OUTCOME,
        "O": Role.OUTCOME_CHILD,
    },
    edges=(("C", "A"), ("C", "Y"), ("C", "Dz"), ("A", "M"), ("A", "Y"), ("M", "Y"), ("B", "Y"), ("Y", "O")),
    drivers=("A", "B", "Dz"),
    outcome="Y",
)


def _sigma(seed: int = 0, motif: Scaffold = _MOTIF) -> np.ndarray:
    """A correlation matrix drawn on ``motif``."""
    return draw_correlation(motif, np.random.default_rng(seed))


def _with_edges(motif: Scaffold, edges) -> Scaffold:
    """``motif`` with its edges replaced (for example, by another member of its class)."""
    return Scaffold(motif.node_names, motif.roles, tuple(edges), motif.drivers, motif.outcome)


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
    for name in _MOTIF.node_names:
        parents = [_MOTIF.index(p) for p in _MOTIF.parents(name)]
        if not parents:
            continue
        j = _MOTIF.index(name)
        r2 = sigma[j, parents] @ np.linalg.solve(sigma[np.ix_(parents, parents)], sigma[parents, j])
        assert EXPLAINED_VARIANCE[0] - 1e-12 <= r2 <= EXPLAINED_VARIANCE[1] + 1e-12


@pytest.mark.parametrize("seed", range(3))
def test_realize_reproduces_sigma_for_every_member_of_the_class(seed):
    motif = procedural_scaffold(seed)
    sigma = _sigma(seed, motif)
    for edges in equivalence_class(motif):
        scm = realize(_with_edges(motif, edges), sigma)
        assert np.max(np.abs(marginal_covariance(scm.W, scm.noise_var) - sigma)) < 1e-10


def test_members_differ_in_weights_but_not_in_distribution():
    """Covariance transfer: no statistic of the data can tell which member is the true graph."""
    sigma = _sigma(1)
    members = equivalence_class(_MOTIF)
    scms = [realize(_with_edges(_MOTIF, edges), sigma) for edges in members]
    assert len({scm.W.tobytes() for scm in scms}) == len(members)
    for scm in scms:
        assert np.allclose(marginal_covariance(scm.W, scm.noise_var), sigma, atol=1e-10)
        # Possible effects are a function of sigma and the class, so every member yields the same list.
        for driver in _MOTIF.drivers:
            assert possible_effects(
                marginal_covariance(scm.W, scm.noise_var), _MOTIF.node_names, members, driver, "Y"
            ) == pytest.approx(possible_effects(sigma, _MOTIF.node_names, members, driver, "Y"), abs=1e-10)


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


def test_realize_rejects_a_collider_on_a_chain_correlation():
    chain = Scaffold(
        node_names=("X", "Y", "Z"),
        roles={"X": Role.DRIVER_CLEAN, "Y": Role.MEDIATOR, "Z": Role.OUTCOME},
        edges=(("X", "Y"), ("Y", "Z")),
        drivers=("X",),
        outcome="Z",
    )
    with pytest.raises(ValueError, match="not Markov"):
        realize(_with_edges(chain, (("X", "Y"), ("Z", "Y"))), _sigma(0, chain))


def test_realized_scm_has_consistent_weights_effects_and_noise():
    scm = realize(_MOTIF, _sigma(7))
    n = scm.n_nodes
    assert np.allclose(scm.tau, np.linalg.inv(np.eye(n) - scm.W) - np.eye(n))
    edge_mask = np.zeros((n, n), dtype=bool)
    for parent, child in _MOTIF.edges:
        edge_mask[_MOTIF.index(parent), _MOTIF.index(child)] = True
    assert np.all(scm.W[~edge_mask] == 0.0)
    assert np.all(scm.W[edge_mask] != 0.0)
    assert np.all(scm.noise_var > 0.0)
    assert np.all(scm.noise_var <= 1.0)


def test_decoy_effect_is_a_structural_zero_in_the_motif():
    assert abs(realize(_MOTIF, _sigma(5)).total_effect_on_outcome("Dz")) < 1e-12


def test_sampled_data_matches_sigma():
    scm = realize(_MOTIF, _sigma(0))
    data = sample(scm, 20_000, seed=0)
    assert data.shape == (20_000, scm.n_nodes)
    assert np.allclose(np.cov(data, rowvar=False), scm.sigma, atol=0.05)
    assert np.array_equal(data, sample(scm, 20_000, seed=0))
