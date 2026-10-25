"""Task generation with acceptance gates that never read the true graph.

One attempt:

1. compose the motif graph ``G0`` (``scm/gadgets.py``);
2. draw its correlation matrix ``Sigma`` (``scm/generate.py::draw_correlation``);
3. enumerate ``G0``'s Markov equivalence class and draw the true DAG ``D`` uniformly, independently of
   ``Sigma``;
4. realize ``D``'s weights from ``Sigma`` (covariance transfer).

Every gate is a function of ``Sigma`` and the equivalence class only, never of which member ``D`` is.
By Bayes' rule, ``P(D | data, accepted) = P(D | CPDAG)``: rejection sampling cannot make any orientation
more likely given what an agent observes. A property that depends on the true graph is therefore required
to hold for every member of the class.

The gates certify what the benchmark claims:

* ``structure``: at least one candidate is name-critical (its effect differs across the class) and at
  least one is identified from data yet needs adjustment.
* ``names_matter``: for each name-critical candidate, the possible effects differ materially.
* ``no_shortcut``: no fixed recipe (adjust for nothing, everything, or the other candidates) gets within
  25% pooled error of the truth, for any member.
* ``decoy_tempting``: a candidate whose effect may be zero is clearly correlated with the outcome.
* ``effect_scale``: every possible effect is below 1 (so the error cap keeps zero predictions at error 1)
  and every member has a non-trivial total effect.
* ``orientable``: each edge the data orient through a v-structure has one whose explaining-away
  correlation is detectable at the benchmark's sample size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causalagentbench.scm.gadgets import procedural_scaffold
from causalagentbench.scm.generate import draw_correlation, realize
from causalagentbench.scm.identifiability import equivalence_class, possible_effects, v_structures
from causalagentbench.types import SCM, FloatArray, Scaffold

MIN_EFFECT_SPREAD = 0.10
"""Absolute floor on how far apart a name-critical candidate's possible effects must be (about 10 SE)."""

MIN_SPREAD_SHARE = 0.25
"""The same spread as a share of the candidate's largest possible |effect|."""

SHORTCUT_ERROR_THRESHOLD = 0.25
"""Minimum capped pooled error of every fixed adjustment recipe, against every member."""

DECOY_MIN_CORRELATION = 0.10
"""Minimum |corr(candidate, outcome)| for a candidate whose effect may be exactly zero."""

MIN_TOTAL_EFFECT = 0.30
"""Minimum sum of |effect| over the candidates, for every member; keeps per-task error ratios stable."""

ORIENTATION_MARGIN = 0.05
"""Minimum |partial correlation| of a v-structure's parents given the collider (about 5 SE at n = 10,000)."""

MAX_ATTEMPTS = 1000
"""Rejection-sampling safety valve."""


@dataclass(frozen=True)
class GateReport:
    """Gate outcomes for one attempt, plus the class-level facts the gates were computed from."""

    structure: bool
    names_matter: bool
    no_shortcut: bool
    decoy_tempting: bool
    effect_scale: bool
    orientable: bool
    class_size: int
    name_critical: tuple[str, ...]
    min_shortcut_error: float
    min_orientation_margin: float

    @property
    def passed(self) -> bool:
        """True if every gate passed."""
        gates = (self.structure, self.names_matter, self.no_shortcut, self.decoy_tempting, self.effect_scale)
        return all(gates) and self.orientable


def capped_error(estimates: dict[str, float], truth: dict[str, float]) -> float:
    """Pooled error ``sum(min(|e - t|, 1)) / sum(|t|)``, the benchmark's headline metric for one task."""
    total = sum(abs(t) for t in truth.values())
    return sum(min(abs(estimates[d] - truth[d]), 1.0) for d in truth) / total


def _population_coefficient(sigma: FloatArray, index: dict[str, int], x: str, y: str, adjust: list[str]) -> float:
    """OLS coefficient of ``x`` regressing ``y`` on ``x`` and ``adjust``, at the population level."""
    rows = [index[x], *(index[z] for z in adjust)]
    return float(np.linalg.solve(sigma[np.ix_(rows, rows)], sigma[rows, index[y]])[0])


def _partial_correlation(sigma: FloatArray, index: dict[str, int], a: str, b: str, given: list[str]) -> float:
    """Population partial correlation of ``a`` and ``b`` given ``given``."""
    rows = [index[a], index[b], *(index[z] for z in given)]
    precision = np.linalg.inv(sigma[np.ix_(rows, rows)])
    return float(-precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1]))


def _orientation_margin(motif: Scaffold, sigma: FloatArray, index: dict[str, int]) -> float:
    """Smallest, over v-structure edges, of the best explaining-away |partial correlation| supporting it."""
    best: dict[tuple[str, str], float] = {}
    for a, c, b in v_structures(motif.digraph()):
        margin = abs(_partial_correlation(sigma, index, a, b, [c]))
        for edge in ((a, c), (b, c)):
            best[edge] = max(best.get(edge, 0.0), margin)
    return min(best.values(), default=1.0)


def evaluate_gates(motif: Scaffold, sigma: FloatArray, members: list[tuple[tuple[str, str], ...]]) -> GateReport:
    """Evaluate every gate from ``sigma`` and the equivalence class ``members`` of ``motif``."""
    index = {name: i for i, name in enumerate(motif.node_names)}
    candidates, y = motif.drivers, motif.outcome
    possible = {d: possible_effects(sigma, motif.node_names, members, d, y) for d in candidates}
    name_critical = tuple(d for d in candidates if max(possible[d]) - min(possible[d]) > 1e-9)
    parent_sets = [{d: {p for p, c in edges if c == d} for d in candidates} for edges in members]
    structure = bool(name_critical) and any(
        d not in name_critical and all(parents[d] for parents in parent_sets) for d in candidates
    )
    names_matter = all(
        max(possible[d]) - min(possible[d]) >= max(MIN_EFFECT_SPREAD, MIN_SPREAD_SHARE * max(map(abs, possible[d])))
        for d in name_critical
    )
    others = [n for n in motif.node_names if n != y]
    recipes = [
        {d: _population_coefficient(sigma, index, d, y, []) for d in candidates},
        {d: _population_coefficient(sigma, index, d, y, [c for c in others if c != d]) for d in candidates},
        {d: _population_coefficient(sigma, index, d, y, [c for c in candidates if c != d]) for d in candidates},
    ]
    truths = [{d: possible[d][k] for d in candidates} for k in range(len(members))]
    effect_scale = all(abs(v) < 1.0 for values in possible.values() for v in values) and all(
        sum(abs(t) for t in truth.values()) >= MIN_TOTAL_EFFECT for truth in truths
    )
    min_shortcut_error = (
        min(capped_error(recipe, truth) for recipe in recipes for truth in truths) if effect_scale else 0.0
    )
    may_be_zero = [d for d in candidates if min(map(abs, possible[d])) < 1e-9]
    decoy_tempting = all(abs(sigma[index[d], index[y]]) >= DECOY_MIN_CORRELATION for d in may_be_zero)
    margin = _orientation_margin(motif, sigma, index)
    return GateReport(
        structure=structure,
        names_matter=names_matter,
        no_shortcut=min_shortcut_error >= SHORTCUT_ERROR_THRESHOLD,
        decoy_tempting=decoy_tempting,
        effect_scale=effect_scale,
        orientable=margin >= ORIENTATION_MARGIN,
        class_size=len(members),
        name_critical=name_critical,
        min_shortcut_error=min_shortcut_error,
        min_orientation_margin=margin,
    )


def attempt(seed: np.random.SeedSequence, n_candidates: int = 5) -> tuple[SCM, GateReport]:
    """Run one generation attempt; the result is accepted only if ``report.passed``."""
    motif_seed, sigma_seed, member_seed = seed.spawn(3)
    motif = procedural_scaffold(motif_seed, n_candidates)
    sigma = draw_correlation(motif, np.random.default_rng(sigma_seed))
    members = equivalence_class(motif)
    # Drawn with its own generator: the choice of member must not depend on sigma.
    edges = members[int(np.random.default_rng(member_seed).integers(len(members)))]
    true_graph = Scaffold(
        node_names=motif.node_names, roles=motif.roles, edges=edges, drivers=motif.drivers, outcome=motif.outcome
    )
    return realize(true_graph, sigma), evaluate_gates(motif, sigma, members)


def generate_scm(seed: np.random.SeedSequence, n_candidates: int = 5) -> tuple[SCM, GateReport]:
    """Return the first accepted SCM from attempts seeded by children of ``seed``."""
    for child in seed.spawn(MAX_ATTEMPTS):
        scm, report = attempt(child, n_candidates)
        if report.passed:
            return scm, report
    raise RuntimeError(f"no accepted SCM in {MAX_ATTEMPTS} attempts")
