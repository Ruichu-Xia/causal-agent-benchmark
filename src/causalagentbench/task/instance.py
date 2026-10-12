"""Instance presentation: turn a hidden SCM + data into the agent-facing data card, plus the hidden key.

``Instance`` is the public artifact (a domain context, a labeled table, a column schema, and a prompt)
and deliberately carries no roles, graph, or SCM. ``InstanceKey`` holds the hidden ground truth (the SCM,
the label<->node map, and the labeling provenance) for the scorer. ``build_instance`` applies a
``Labeler``, shuffles the display column order, and pairs them.

The prompt states only the *question* - the outcome and the candidate drivers to estimate effects for - plus the domain
context and schema. It never reveals causal roles, and it does not tell the agent which columns to adjust
for: discovering that is the task. Its wording is turn-agnostic, fitting a single-shot call or a
multi-turn tool-use loop.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from causalagentbench.task.labeling import Labeler, LabelProvenance, abstract_labels
from causalagentbench.types import SCM, FloatArray, Scaffold, SeedLike


@dataclass(frozen=True)
class Instance:
    """The public task artifact: a domain context, a labeled table, a schema, and a prompt."""

    columns: tuple[str, ...]
    data: FloatArray
    outcome: str
    candidate_drivers: tuple[str, ...]
    prompt: str
    context: str = ""
    column_schema: Mapping[str, str] = field(default_factory=dict)

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
    provenance: LabelProvenance


def build_prompt(
    columns: tuple[str, ...],
    outcome: str,
    candidate_drivers: tuple[str, ...],
    context: str = "",
    column_schema: Mapping[str, str] | None = None,
) -> str:
    """Describe the analysis task without revealing graph structure."""
    schema = column_schema or {}
    lines = "\n".join(f"  {col}: {schema[col]}" if col in schema else f"  {col}" for col in columns)
    return f"""Context: {context or "A synthetic observational study."}

Columns:
{lines}

Outcome: {outcome}
Candidate drivers: {", ".join(candidate_drivers)}

Assume an acyclic linear causal model with independent Gaussian noise. All common causes are observed.
Assume the population distribution is faithful to the graph. Variables have population mean zero and
standard deviation one; sample moments can differ. Names identify concepts; use your domain knowledge
together with the data to infer their causal relationships.

Estimate each candidate's signed TOTAL causal effect on the outcome. Report the adjustment set used
for each estimate; it may be empty. The effect is the signed expected change in the outcome
per one-unit intervention increase in the driver, including effects through intermediate variables.
Some candidates have no causal effect. Answer order does not matter.

Use the statistical tools to investigate and estimate the effects. Discovery is imperfect at finite sample
size: check its results when needed.

Return every candidate exactly once, with a finite numerical effect, in this format:
<answer>
{{"verdicts": [{{"driver": "<column>", "adjustment_set": ["<column>"], "effect": 0.23}}]}}
</answer>
The last answer block is scored. The benchmark reports total absolute estimation error divided by total
absolute true effect across all tasks and candidates; lower is better, and zero-effect predictions score 1.
Invalid answers count as zero-effect predictions in that metric and reduce the reported answer-validity rate.
RL reward depends only on signed effect accuracy: one minus the average
absolute estimation error, divided by the task's median absolute true effect and capped at one per
candidate. Adjustment quality and ranking agreement are diagnostics only; neither contributes to reward."""


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
    labeler: Labeler = abstract_labels,
) -> tuple[Instance, InstanceKey]:
    """Build the public ``Instance`` and hidden ``InstanceKey`` from an SCM and its sampled data.

    Applies ``labeler``, shuffles the display column order, and permutes ``data`` to match. The label and
    shuffle RNGs are split from ``seed`` so two labelers on the same ``seed`` share the display order and
    differ only in names - the paired skins the naming sweep relies on. ``data`` columns must follow the
    SCM's canonical node order.
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
    prompt = build_prompt(columns, outcome, candidate_drivers, labeling.context, labeling.column_schema)

    instance = Instance(
        columns=columns,
        data=display_data,
        outcome=outcome,
        candidate_drivers=candidate_drivers,
        prompt=prompt,
        context=labeling.context,
        column_schema=dict(labeling.column_schema),
    )
    key = InstanceKey(
        scm=scm,
        label_to_node={label: node for node, label in node_to_label.items()},
        node_to_label=dict(node_to_label),
        provenance=labeling.provenance,
    )
    return instance, key
