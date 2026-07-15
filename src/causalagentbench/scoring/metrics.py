"""Ranking metrics.

Later stages grow this into the scorer library; Stage 0 needs only rank agreement.
"""

from __future__ import annotations

from collections.abc import Sequence


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall rank correlation (tau-a): +1 identical, -1 reversed, 0 none; ties are ignored."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            product = (x[i] - x[j]) * (y[i] - y[j])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    denominator = concordant + discordant
    return (concordant - discordant) / denominator if denominator else 1.0
