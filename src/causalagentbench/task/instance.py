"""Instance presentation: turn a hidden SCM + data into what the agent sees, plus the hidden key.

``Instance`` is the public artifact (labeled table + prompt) and deliberately carries no roles, graph,
or SCM. ``InstanceKey`` holds the hidden ground truth (the SCM and the label<->node map) for the future
scorer. ``build_instance`` applies a ``Labeler``, shuffles the display column order, and pairs them.

The prompt states only the *question* - the outcome and the candidate drivers to rank. It never reveals
causal roles, and it does not tell the agent which of the remaining columns to adjust for: discovering
that (confounder vs mechanism, and that a candidate may have zero effect) is the task.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from causalagentbench.task.labeling import Labeler, abstract_labels
from causalagentbench.types import SCM, FloatArray, SeedLike


@dataclass(frozen=True)
class Instance:
    """The public task artifact: a labeled data table and a prompt, with no ground truth."""

    columns: tuple[str, ...]
    data: FloatArray
    outcome: str
    candidate_drivers: tuple[str, ...]
    prompt: str

    def __post_init__(self) -> None:
        """Freeze the data payload against accidental mutation."""
        self.data.flags.writeable = False

    @property
    def other_columns(self) -> tuple[str, ...]:
        """Columns that are neither the outcome nor a candidate driver (roles unknown to the agent)."""
        excluded = {self.outcome, *self.candidate_drivers}
        return tuple(c for c in self.columns if c not in excluded)

    def to_csv(self) -> str:
        """Render the table as CSV with a header row of ``columns``."""
        buffer = io.StringIO()
        buffer.write(",".join(self.columns) + "\n")
        np.savetxt(buffer, self.data, delimiter=",", fmt="%.6g")
        return buffer.getvalue()


@dataclass(frozen=True)
class InstanceKey:
    """The hidden ground truth pairing an ``Instance`` to its SCM (for the scorer, never the agent)."""

    scm: SCM
    label_to_node: Mapping[str, str]
    node_to_label: Mapping[str, str]


def build_prompt(columns: tuple[str, ...], outcome: str, candidate_drivers: tuple[str, ...]) -> str:
    """Return the task prompt: the question (outcome + candidate drivers) and the answer shape.

    Presents the remaining columns only as other data columns; it does not say which to adjust for.
    """
    return f"""\
You are given an observational dataset (CSV) with columns: {", ".join(columns)}.
The outcome variable is "{outcome}". Evaluate these candidate drivers: {", ".join(candidate_drivers)}.
The dataset also contains other columns; it is part of the task to work out what role each plays.

For each candidate driver, decide whether its causal effect on the outcome is identifiable from this
data and, if so, which columns form a valid adjustment set for estimating its total effect (some
candidates may have little or no causal effect). Then rank the identifiable drivers from strongest to
weakest causal effect.

Provide your final answer as JSON inside a single <answer>...</answer> block, with this shape:
<answer>
{{"verdicts": [{{"driver": "<name>", "identifiable": true, "adjustment_set": ["<column>"]}}],
 "ranking": ["<driver>", "..."]}}
</answer>
Only the content of the last <answer> block is evaluated; you may reason freely before it."""


def build_instance(
    scm: SCM,
    data: FloatArray,
    seed: SeedLike,
    *,
    labeler: Labeler = abstract_labels,
) -> tuple[Instance, InstanceKey]:
    """Build the public ``Instance`` and hidden ``InstanceKey`` from an SCM and its sampled data.

    Assigns labels via ``labeler``, shuffles the display column order, and permutes ``data`` to match.
    Deterministic in ``seed``. ``data`` columns must follow the SCM's canonical node order.
    """
    rng = np.random.default_rng(seed)
    scaffold = scm.scaffold
    node_to_label = labeler(scaffold, rng)

    display_nodes = list(scaffold.node_names)
    rng.shuffle(display_nodes)
    columns = tuple(node_to_label[node] for node in display_nodes)
    permutation = [scaffold.index(node) for node in display_nodes]
    display_data = np.ascontiguousarray(data[:, permutation])

    outcome = node_to_label[scaffold.outcome]
    candidate_drivers = tuple(node_to_label[d] for d in scaffold.drivers)
    prompt = build_prompt(columns, outcome, candidate_drivers)

    instance = Instance(
        columns=columns,
        data=display_data,
        outcome=outcome,
        candidate_drivers=candidate_drivers,
        prompt=prompt,
    )
    key = InstanceKey(
        scm=scm,
        label_to_node={label: node for node, label in node_to_label.items()},
        node_to_label=dict(node_to_label),
    )
    return instance, key
