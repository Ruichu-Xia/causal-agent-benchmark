"""Instance presentation: turn a hidden SCM + data into the agent-facing data card, plus the hidden key.

``Instance`` is the public artifact (a domain context, a labeled table, a column schema, and a prompt)
and deliberately carries no roles, graph, or SCM. ``InstanceKey`` holds the hidden ground truth (the SCM,
the label<->node map, and the labeling provenance) for the scorer. ``build_instance`` applies a
``Labeler``, shuffles the display column order, and pairs them.

The prompt states only the *question* - the outcome and the candidate drivers to rank - plus the domain
context and schema. It never reveals causal roles, and it does not tell the agent which columns to adjust
for: discovering that is the task. Its wording is turn-agnostic, fitting a single-shot call or a
multi-turn tool-use loop.
"""

from __future__ import annotations

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
        buffer.write(",".join(self.columns) + "\n")
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
    """Return the task prompt: optional context/schema, then the question and the answer shape.

    Any task-level assumptions the agent should work under are carried in ``context`` by the labeler, not
    hard-coded here. Presents the remaining columns only as other data columns; it does not say which to
    adjust for.
    """
    preamble = ""
    if context:
        preamble += f"Context: {context}\n\n"
    if column_schema:
        lines = "\n".join(f"  {col}: {column_schema[col]}" for col in columns if col in column_schema)
        preamble += f"Columns:\n{lines}\n\n"
    return f"""\
{preamble}You are given an observational dataset (CSV) with columns: {", ".join(columns)}.
The row values are available only through the provided data tools; their results will be returned in the
conversation after you call them. Do not infer or invent values that a tool has not returned.
All columns are standardized (mean 0, standard deviation 1), so magnitudes are directly comparable across
variables regardless of what they measure.
The outcome variable is "{outcome}". Evaluate these candidate drivers: {", ".join(candidate_drivers)}.
The dataset also contains other columns; it is part of the task to work out what role each plays.

For each candidate driver, decide which columns form a valid adjustment set for estimating its total
causal effect on the outcome (some candidates may have little or no causal effect; the adjustment set may
be empty), then report "effect": your estimate of that total causal effect, as a number. Because every
column is standardized, this is the expected change in the outcome, in standard deviations, per one
standard deviation increase in the driver. The drivers are ranked by the size of these estimates, so their
relative magnitudes matter as well as each one's accuracy.

Some drivers cannot be resolved from this data alone - two different underlying explanations fit the same
observations equally well, and no amount of analysis distinguishes them.

Where that happens, the column names are your only remaining evidence. They come from a domain source that
may be accurate or may be misleading, and nothing states which. You can find out: the data settles most of
the structure, so you can check whether the names agree with what you were able to establish. Names that
are consistent with the structure you recovered can be relied on for the parts the data left open; names
that contradict it cannot be relied on anywhere.

So for each unresolved driver: commit if the names have proven trustworthy and yours identifies the driver's
role, and otherwise set "abstain": true and omit "effect". A wrong committed answer is worse than an honest
abstention, but abstaining where the evidence did settle the answer is also wrong.

Work through the problem step by step, inspecting the data as needed. When your analysis is complete,
give your final answer as JSON inside a single <answer>...</answer> block, with this shape:
<answer>
{{"verdicts": [{{"driver": "<name>", "adjustment_set": ["<column>"], "effect": 0.0, "abstain": false}}]}}
</answer>
Only the content of the last <answer> block is scored, so it must reflect your final conclusions."""


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
