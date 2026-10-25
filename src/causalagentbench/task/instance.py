"""Instance presentation: turn a hidden SCM + data into the agent-facing task, plus the hidden key.

``Instance`` is the public artifact (a labeled table and a prompt) and carries no roles, graph, or SCM.
``InstanceKey`` holds the hidden ground truth for the scorer. ``build_instance`` applies a ``Labeler``,
shuffles the display column order, and pairs them.

The prompt states the question - the outcome and the candidates - and the model assumptions. It never
reveals causal roles or which columns to adjust for. Real and placeholder names produce the same prompt
apart from the names themselves, so the names ablation changes nothing else.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from causalagentbench.task.labeling import Labeler, placeholder_labels
from causalagentbench.types import SCM, FloatArray, Scaffold, SeedLike


@dataclass(frozen=True)
class Instance:
    """The public task artifact: a domain context, a labeled table, a schema, and a prompt."""

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
        csv.writer(buffer, lineterminator="\n").writerow(self.columns)
        np.savetxt(buffer, self.data, delimiter=",", fmt="%.6g")
        return buffer.getvalue()


@dataclass(frozen=True)
class InstanceKey:
    """The hidden ground truth pairing an ``Instance`` to its SCM (for the scorer, never the agent)."""

    scm: SCM
    label_to_node: Mapping[str, str]
    node_to_label: Mapping[str, str]
    producer: str  # where the names came from: "placeholder" or "causenet:sha256:<digest>"


def build_prompt(columns: tuple[str, ...], outcome: str, candidate_drivers: tuple[str, ...]) -> str:
    """Describe the analysis task without revealing graph structure."""
    lines = "\n".join(f"  {col}" for col in columns)
    return f"""Synthetic observational data: {len(columns)} continuous variables. The data and effect sizes are \
simulated, not measured real-world effects.

Columns:
{lines}

Outcome: {outcome}
Candidate drivers: {", ".join(candidate_drivers)}

Assume an acyclic linear causal model with independent Gaussian noise. All common causes are observed.
Assume the population distribution is faithful to the graph. Variables have population mean zero and
standard deviation one; sample moments can differ. Column names may identify real-world concepts; if they
do, use your domain knowledge together with the data to infer which way causation runs. Coefficient signs
and sizes are synthetic: names can indicate causal direction, not how large an effect is or its sign.

Estimate each candidate's signed TOTAL causal effect on the outcome. Report the adjustment set used
for each estimate; it may be empty. The effect is the signed expected change in the outcome
per one-unit intervention increase in the driver, including effects through intermediate variables.
A candidate may have no causal effect. Answer order does not matter.

Use the statistical tools to investigate and estimate the effects. Discovery is imperfect at finite sample
size: check its results when needed.

Return every candidate exactly once, with a finite numerical effect, in this format:
<answer>
{{"verdicts": [{{"driver": "<column>", "adjustment_set": ["<column>"], "effect": 0.23}}]}}
</answer>

The last answer block is scored. The score is the total absolute estimation error, with each candidate's
error capped at 1, divided by the total absolute true effect, over all tasks and candidates. Lower is
better; predicting zero for every candidate scores 1. An invalid answer counts as zero for every candidate."""


def outcome_and_drivers(
    scaffold: Scaffold, node_to_label: Mapping[str, str], columns: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """Return (outcome label, candidate-driver labels in ``columns`` order) from a scaffold + label map.

    Shared by ``build_instance`` (labeling just happened) and ``dataset.load_dataset`` (labeling was
    already serialized) - same derivation, only the display order's source differs.
    """
    label_to_node = {label: node for node, label in node_to_label.items()}
    outcome = node_to_label[scaffold.outcome]
    candidate_drivers = tuple(c for c in columns if label_to_node[c] in scaffold.drivers)
    return outcome, candidate_drivers


def build_instance(
    scm: SCM,
    data: FloatArray,
    seed: SeedLike,
    *,
    labeler: Labeler = placeholder_labels,
) -> tuple[Instance, InstanceKey]:
    """Build the public ``Instance`` and hidden ``InstanceKey`` from an SCM and its sampled data.

    Applies ``labeler``, shuffles the display column order, and permutes ``data`` to match. The label and
    shuffle RNGs are split from ``seed`` so two labelers on the same ``seed`` share the display order and
    differ only in names. ``data`` columns must follow the SCM's canonical node order.
    """
    label_rng, shuffle_rng = np.random.default_rng(seed).spawn(2)
    scaffold = scm.scaffold
    labeling = labeler(scm, label_rng)
    node_to_label = labeling.node_to_label

    display_nodes = list(scaffold.node_names)
    shuffle_rng.shuffle(display_nodes)
    columns = tuple(node_to_label[node] for node in display_nodes)
    permutation = [scaffold.index(node) for node in display_nodes]
    display_data = np.ascontiguousarray(data[:, permutation])

    # Candidates come out in the shuffled display order (not fixed scaffold order), so a driver's
    # position carries no information about its role - no "always abstain on the first candidate" shortcut.
    outcome, candidate_drivers = outcome_and_drivers(scaffold, node_to_label, columns)
    instance = Instance(
        columns=columns,
        data=display_data,
        outcome=outcome,
        candidate_drivers=candidate_drivers,
        prompt=build_prompt(columns, outcome, candidate_drivers),
    )
    key = InstanceKey(
        scm=scm,
        label_to_node={label: node for node, label in node_to_label.items()},
        node_to_label=dict(node_to_label),
        producer=labeling.producer,
    )
    return instance, key


def with_placeholder_names(task: tuple[Instance, InstanceKey], seed: SeedLike) -> tuple[Instance, InstanceKey]:
    """Return the same task, data and column order with every column renamed ``X1..Xk`` at random."""
    instance, key = task
    order = np.random.default_rng(seed).permutation(len(instance.columns))
    rename = {old: f"X{pos + 1}" for old, pos in zip(instance.columns, order, strict=True)}
    columns = tuple(rename[c] for c in instance.columns)
    outcome = rename[instance.outcome]
    candidates = tuple(rename[c] for c in instance.candidate_drivers)
    renamed = Instance(
        columns=columns,
        data=instance.data,
        outcome=outcome,
        candidate_drivers=candidates,
        prompt=build_prompt(columns, outcome, candidates),
    )
    node_to_label = {node: rename[label] for node, label in key.node_to_label.items()}
    return renamed, InstanceKey(
        scm=key.scm,
        label_to_node={label: node for node, label in node_to_label.items()},
        node_to_label=node_to_label,
        producer="placeholder",
    )
