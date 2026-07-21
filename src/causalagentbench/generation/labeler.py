"""Three-pole naming: propose per-node (aligned/neutral/misaligned) triples, then CONSTRUCT the level.

Per SCM, a generator LLM (grounded by CauseNet, seeing the edges + true effects + roles) proposes, for
each node, three names from one coherent domain: aligned (prior matches the node's true role), neutral
(no causal prior), misaligned (a task-relevant trap - a compelling cause-name on the zero-effect decoy,
swapped confounder/mediator cues). A propose -> critique -> revise loop (deterministic pre-checks + an
optional LLM correctness critic) de-risks the proposal, since all quality now rests on the generator.

The alignment level is then **constructed**, not measured: on the discrete grid ``{-1,-0.5,0,+0.5,+1}``,
``round(|level|*K)`` load-bearing nodes use their aligned (level>0) or misaligned (level<0) name and the
rest go neutral - so the level is exact and reproducible, and paired skins (same SCM, all levels) are
free. Nodes are shown under neutral ids ``v0..vk`` so ids do not leak roles.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from causalagentbench.generation.clients import LabelProposer, LiteLLMProposer
from causalagentbench.generation.grounding import format_menus, sample_menus
from causalagentbench.task.labeling import Labeling, LabelProvenance
from causalagentbench.types import SCM, Role

logger = logging.getLogger(__name__)

#: Grounding: turn an SCM + rng into a domain-context string for the generator prompt.
GroundingFn = Callable[[SCM, np.random.Generator], str]

#: The discrete alignment grid the test set is built on.
GRID: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)

_KINDS = ("aligned", "neutral", "misaligned")
_SNAKE = re.compile(r"[a-z][a-z0-9_]*")
# Placeholder / degenerate name patterns that must be rejected (generators fall back to these).
_PLACEHOLDER = re.compile(r"^(feature|latent_component|component|channel|col|var|node|v|x)_?\d*$|^[a-z]\d+$")
# Tokens implying a non-continuous variable (the data is standardized continuous).
_NONCONTINUOUS = ("binary", "yes/no", "yes-no", "categorical", "count of", "0/1", "indicator")


@dataclass(frozen=True)
class NamingTriples:
    """One coherent domain's three name choices per node: node -> {kind -> (name, description)}."""

    domain: str
    context: str
    choices: Mapping[str, Mapping[str, tuple[str, str]]]


def _informative_order(scm: SCM) -> list[str]:
    """Non-outcome nodes ordered by load-bearing importance (filled into informative slots first)."""
    scaffold = scm.scaffold
    decoys = list(scaffold.nodes_with_role(Role.DECOY))
    drivers = sorted(
        (d for d in scaffold.drivers if d not in decoys),
        key=lambda d: abs(scm.total_effect_on_outcome(d)),
        reverse=True,
    )
    ordered = [
        *decoys,
        *drivers[:1],  # strongest driver
        *scaffold.nodes_with_role(Role.CONFOUNDER),
        *scaffold.nodes_with_role(Role.MEDIATOR),
        *drivers[1:],  # remaining drivers
    ]
    for node in scaffold.node_names:  # any leftover non-outcome nodes
        if node != scaffold.outcome and node not in ordered:
            ordered.append(node)
    seen: list[str] = []
    for node in ordered:
        if node not in seen:
            seen.append(node)
    return seen


def construct_labeling(triples: NamingTriples, scm: SCM, level: float, producer: str) -> Labeling:
    """Deterministically build a ``Labeling`` at ``level`` from a triple set (no LLM, exact alpha)."""
    scaffold = scm.scaffold
    order = _informative_order(scm)
    n_informative = round(abs(level) * len(order))
    informative = set(order[:n_informative])
    informative_kind = "aligned" if level > 0 else "misaligned"

    node_to_label: dict[str, str] = {}
    schema: dict[str, str] = {}
    counts = dict.fromkeys(_KINDS, 0)
    for node in scaffold.node_names:
        if node == scaffold.outcome:
            kind = "aligned"  # the outcome is disclosed to the agent; always its real name
        elif node in informative:
            kind = informative_kind
        else:
            kind = "neutral"
        name, description = triples.choices[node][kind]
        node_to_label[node] = name
        schema[name] = description
        counts[kind] += 1

    provenance = LabelProvenance(
        alignment=level,
        n_aligned=counts["aligned"],
        n_neutral=counts["neutral"],
        n_misaligned=counts["misaligned"],
        domain=triples.domain,
        producer=producer,
    )
    return Labeling(node_to_label=node_to_label, context=triples.context, column_schema=schema, provenance=provenance)


def propose_triples(
    scm: SCM,
    rng: np.random.Generator,
    *,
    proposer: LabelProposer,
    grounding: GroundingFn,
    critic: LabelProposer | None = None,
    max_revisions: int = 2,
) -> NamingTriples:
    """Generate a triple set with a propose -> critique -> revise loop; raise if it never passes."""
    grounding_text = grounding(scm, rng)
    prompt = _generator_prompt(scm, grounding_text)
    issues: list[str] = []
    for attempt in range(max_revisions + 1):
        raw = proposer.propose(prompt)
        try:
            triples = _parse_triples(raw, scm)
            issues = _pre_check(triples, scm)
            if not issues and critic is not None:
                issues = _critique(scm, triples, critic)
        except (ValueError, KeyError, TypeError) as exc:
            triples, issues = None, [f"output was not valid JSON matching the schema: {exc}"]
        if triples is not None and not issues:
            return triples
        logger.debug("triple proposal attempt %d rejected: %s", attempt, issues)
        prompt = _revise_prompt(scm, grounding_text, raw, issues)
    raise ValueError(f"could not produce a valid naming for SCM in {max_revisions + 1} attempts: {issues}")


def _pre_check(triples: NamingTriples, scm: SCM) -> list[str]:
    """Deterministic structural checks; returns a list of issues (empty == ok)."""
    nodes = set(scm.scaffold.node_names)
    issues: list[str] = []
    if set(triples.choices) != nodes:
        issues.append("does not cover exactly the graph's nodes")
        return issues
    names: list[str] = []
    for node, kinds in triples.choices.items():
        if set(kinds) != set(_KINDS):
            issues.append(f"node {node} is missing one of aligned/neutral/misaligned")
            continue
        for _, (name, desc) in kinds.items():
            names.append(name)
            if not _SNAKE.fullmatch(name):
                issues.append(f"name {name!r} is not snake_case")
            if _PLACEHOLDER.match(name):
                issues.append(f"name {name!r} is a placeholder, not a real-world concept")
            if not desc.strip():
                issues.append(f"empty description for {name!r}")
            if any(tok in desc.lower() for tok in _NONCONTINUOUS):
                issues.append(f"description of {name!r} implies a non-continuous variable")
    if len(set(names)) != len(names):
        issues.append("names are not all distinct")
    return issues


def _critique(scm: SCM, triples: NamingTriples, critic: LabelProposer) -> list[str]:
    """Ask the critic to verify the triples' correctness; returns a list of issues (empty == ok)."""
    parsed = _extract_json(critic.propose(_critic_prompt(scm, triples)))
    if parsed.get("ok"):
        return []
    return [str(x) for x in parsed.get("issues", ["critic rejected the naming"])]


def llm_labeler(
    gen_model: str,
    causenet_dump: str,
    *,
    level: float,
    critic_model: str | None = None,
    max_revisions: int = 2,
) -> LLMLabeler:
    """Build an ``LLMLabeler`` at a fixed grid ``level`` from a generator model + CauseNet dump.

    ``critic_model`` (a separate correctness-critic call; ``None`` skips it, running pre-checks only) may
    be the same model as ``gen_model`` or a different one.
    """
    producer = f"llm:{gen_model}" + (f"+critic:{critic_model}" if critic_model else "")
    return LLMLabeler(
        level=level,
        proposer=LiteLLMProposer(gen_model),
        grounding=grounding_fn(causenet_dump),
        critic=LiteLLMProposer(critic_model) if critic_model else None,
        max_revisions=max_revisions,
        producer=producer,
    )


def grounding_fn(causenet_dump: str) -> GroundingFn:
    """A ``GroundingFn`` that formats CauseNet role-candidate menus for the generator prompt."""

    def grounding(scm: SCM, rng: np.random.Generator) -> str:
        return format_menus(sample_menus(causenet_dump, scm.scaffold, rng))

    return grounding


@dataclass(frozen=True)
class LLMLabeler:
    """A ``Labeler`` that proposes triples for the SCM and constructs its labeling at a fixed ``level``.

    For a single instance. To build paired skins across the grid from *one* proposal, call
    ``propose_triples`` once and ``construct_labeling`` per level (see ``alpha_sweep`` / the producer).
    """

    level: float
    proposer: LabelProposer
    grounding: GroundingFn
    critic: LabelProposer | None = None
    max_revisions: int = 2
    producer: str = "llm"

    def __post_init__(self) -> None:
        """Reject a level outside [-1, 1]."""
        if not -1.0 <= self.level <= 1.0:
            raise ValueError(f"level must be in [-1, 1], got {self.level}")

    def __call__(self, scm: SCM, rng: np.random.Generator) -> Labeling:
        """Propose a triple set for ``scm`` and construct its labeling at ``self.level``."""
        triples = propose_triples(
            scm,
            rng,
            proposer=self.proposer,
            grounding=self.grounding,
            critic=self.critic,
            max_revisions=self.max_revisions,
        )
        return construct_labeling(triples, scm, self.level, self.producer)


def _ids(scm: SCM) -> dict[str, str]:
    """Map each node to a neutral display id (``v0..vk``) for the offline prompts."""
    return {node: f"v{i}" for i, node in enumerate(scm.scaffold.node_names)}


def _graph_facts(scm: SCM) -> str:
    """Describe the true graph (edges + each node's role + total effect) under neutral ids."""
    ids = _ids(scm)
    scaffold = scm.scaffold
    edges = "; ".join(f"{ids[p]}->{ids[c]}" for p, c in scaffold.edges)
    lines = []
    for node in scaffold.node_names:
        if node == scaffold.outcome:
            lines.append(f"{ids[node]}: the OUTCOME")
        else:
            effect = scm.total_effect_on_outcome(node)
            role = "zero-effect decoy" if node in scaffold.nodes_with_role(Role.DECOY) else scaffold.roles[node].value
            lines.append(f"{ids[node]}: {role}, true effect on outcome = {effect:+.2f}")
    return f"Directed edges (cause->effect): {edges}.\nNodes:\n  " + "\n  ".join(lines)


def _generator_prompt(scm: SCM, grounding: str) -> str:
    """Prompt the generator for per-node aligned/neutral/misaligned name triples."""
    ids = _ids(scm)
    example = ids[scm.scaffold.node_names[0]]
    return f"""\
Name the variables of a hidden causal graph for a benchmark. Below is the TRUE structure.

{_graph_facts(scm)}

Real domain concepts to draw on (a cause-effect knowledge base): {grounding}

Pick ONE coherent real-world domain. For EVERY node give THREE candidate names, each with a one-line
description of what the variable *measures* (a data dictionary; the data is continuous & standardized -
never binary/count/categorical, and never narrate causal relationships):
- "aligned": a name whose real-world prior MATCHES the node's true role - a strongly-causal name on a
  large-effect driver, an INERT/unrelated name on the zero-effect decoy, a common-cause name on a
  confounder, an on-the-pathway name on a mediator, the real target name on the outcome.
- "neutral": a DATA-COLLECTION ARTIFACT or identifier with no causal link to the outcome - a record /
  batch / form id, an operator or device code, a row index, an enrollment timestamp. NOT a domain
  measurement (those can always be argued to have some causal role); pick something clearly inert.
- "misaligned": a name that MISLEADS about the true role, on the axes the task grades (relevance & role,
  NOT sign) - a strongly-causal name on the ZERO-effect decoy; an inert name on a large-effect driver;
  a mediator-sounding name on the confounder and a confounder-sounding name on the mediator.

All names snake_case and ALL distinct. Return ONLY JSON of the form:
{{"domain": "...", "context": "...", "nodes": {{
  "{example}": {{"aligned": {{"name":"...","description":"..."}}, "neutral": {{...}}, "misaligned": {{...}}}}
}}}}"""


def _critic_prompt(scm: SCM, triples: NamingTriples) -> str:
    """Prompt the critic to verify each node's aligned/neutral/misaligned names are genuinely correct."""
    ids = _ids(scm)
    listed = "\n".join(
        f"  {ids[node]} ({'outcome' if node == scm.scaffold.outcome else scm.scaffold.roles[node].value}): "
        + ", ".join(f"{kind}={name}" for kind, (name, _) in kinds.items())
        for node, kinds in triples.choices.items()
    )
    return f"""\
Check a proposed naming of this TRUE causal graph. Domain: {triples.domain!r}.

{_graph_facts(scm)}

Proposed names per node:
{listed}

Verify, for every node: the "aligned" name's real-world prior genuinely matches the node's role (strong
cause on a large-effect driver, inert on the zero-effect decoy, common-cause on a confounder, on-path on
a mediator); the "neutral" name is a data-collection artifact / identifier (record/batch/form id,
operator or device code, row index, timestamp) with no causal link to the outcome - accept such
administrative names as neutral; the "misaligned" name is a genuine task-relevant trap (a strong
cause-name on the zero-effect decoy, inert on a driver, swapped confounder/mediator cues). Also flag
placeholder / incoherent names.
Return ONLY JSON: {{"ok": true}} or {{"ok": false, "issues": ["<node id>: <what is wrong>", ...]}}"""


def _revise_prompt(scm: SCM, grounding: str, previous: str, issues: list[str]) -> str:
    """Prompt the generator to fix the flagged issues in its previous proposal."""
    return (
        _generator_prompt(scm, grounding)
        + "\n\nYour previous answer had these problems - fix ALL of them and return corrected JSON:\n"
        + "\n".join(f"- {issue}" for issue in issues)
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in ``text`` (tolerating code fences / surrounding prose)."""
    fenced = re.search(r"\{.*\}", text, re.DOTALL)
    if not fenced:
        raise ValueError("no JSON object found in model output")
    parsed: dict[str, Any] = json.loads(fenced.group(0))
    return parsed


def _parse_triples(text: str, scm: SCM) -> NamingTriples:
    """Parse the generator output (keyed by neutral ids) into a ``NamingTriples`` covering every node."""
    id_to_node = {vid: node for node, vid in _ids(scm).items()}
    raw = _extract_json(text)
    choices: dict[str, dict[str, tuple[str, str]]] = {}
    for vid, kinds in raw["nodes"].items():
        node = id_to_node.get(vid)
        if node is None:
            raise ValueError(f"naming references unknown node id {vid!r}")
        choices[node] = {kind: (spec["name"], spec["description"]) for kind, spec in kinds.items()}
    return NamingTriples(domain=raw["domain"], context=raw["context"], choices=choices)
