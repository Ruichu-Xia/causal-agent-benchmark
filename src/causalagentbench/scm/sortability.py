"""Sortability diagnostics (Reisach et al.).

Sortability measures how well a per-node scalar (marginal variance, or R^2) agrees with the causal
order; a value near 0.5 means no sort-based shortcut exists. ``varsortability`` gates Stage 0 instances;
``r2_sortability`` is reported for transparency.
"""

from __future__ import annotations

import numpy as np

from causalagentbench.types import SCM, FloatArray


def _path_sortability(adjacency: FloatArray, scores: FloatArray) -> float:
    """Fraction of directed paths along which ``scores`` increases (ties count as 1/2).

    Counts agreement over paths of every length, weighting each by its multiplicity via powers of the
    adjacency matrix (Reisach et al.).
    """
    edges = (adjacency != 0).astype(np.float64)
    increasing = (scores[:, np.newaxis] < scores[np.newaxis, :]).astype(np.float64)
    tied = (scores[:, np.newaxis] == scores[np.newaxis, :]).astype(np.float64)
    total = 0.0
    agreeing = 0.0
    powered = edges.copy()
    for _ in range(edges.shape[0] - 1):
        total += float(powered.sum())
        agreeing += float((powered * increasing).sum()) + 0.5 * float((powered * tied).sum())
        powered = powered @ edges
    return agreeing / total if total > 0.0 else 0.5


def varsortability(data: FloatArray, weights: FloatArray) -> float:
    """Varsortability of ``data`` given the graph in ``weights`` (uses empirical variances)."""
    return _path_sortability(weights, data.var(axis=0))


def r2_sortability(scm: SCM) -> float:
    """R^2-sortability of ``scm``; each node's R^2 equals ``1 - noise_var`` (variances are unit)."""
    return _path_sortability(scm.W, 1.0 - scm.noise_var)
