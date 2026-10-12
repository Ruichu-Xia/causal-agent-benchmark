"""Shared PC configuration for the agent's tool and scripted baseline."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from causalagentbench.task.instance import Instance

DEFAULT_ALPHA = 0.01


def discover_graph(instance: Instance, alpha: float = DEFAULT_ALPHA) -> npt.NDArray[np.int_]:
    """Run PC on public data only; never use hidden graph structure or semantic gold."""
    from causallearn.search.ConstraintBased.PC import pc

    result = pc(instance.data, alpha=alpha, show_progress=False)
    return np.asarray(result.G.graph, dtype=np.int_)
