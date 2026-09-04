"""Two-pole naming: propose per-node (aligned/misaligned) name pairs, then CONSTRUCT the level.

Per SCM, a generator LLM (grounded by a CauseNet concept pool, seeing each node's role and a coarse effect
band) proposes, for each node, two names from one coherent domain: aligned (prior matches the node's true
role) and misaligned (a task-relevant trap - a compelling cause-name on the zero-effect decoy, an
unremarkable quantity on a real driver). A propose -> critique -> revise loop de-risks the proposal:
deterministic pre-checks including an enforced structural-vocabulary ban (``_ROLE_TOKENS``), plus an
optional LLM critic that reviews the complete card.

The alignment level is then **constructed**, not measured: on the discrete grid ``GRID`` (``{-1,+1}``),
every non-outcome node uniformly takes its aligned (level>0) or misaligned (level<0) name, so the level is
exact and reproducible, comparable cleanly across the naming-vs-identifiability quadrant
(``scoring/score.py``), and paired skins (same SCM, both levels) are free. Nodes are shown under opaque
ids ``v0..vk`` so ids do not leak roles.
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
from causalagentbench.scoring.baselines import effect_scale
from causalagentbench.task.labeling import Labeling, LabelProvenance
from causalagentbench.types import SCM, Role

logger = logging.getLogger(__name__)

#: Grounding: turn an SCM + rng into a domain-context string for the generator prompt.
GroundingFn = Callable[[SCM, np.random.Generator], str]

#: The discrete alignment grid the test set is built on: misaligned and aligned, applied uniformly to every
#: non-outcome node (§11.7-adjacent design review: dropped the old ``{-0.5, +0.5}`` partial-flip midpoints,
#: which needed a hand-tuned node-priority order to pick a subset; the endpoints never did).
#:
#: These two endpoints are the capability itself: use a prior that has proven reliable (+1), reject one that
#: has proven misleading (-1). The name-blind labeler (``task/labeling.py``) is the no-prior control and
#: costs no LLM calls, so it stays. Neutral naming was dropped entirely - see ``_KINDS``.
GRID: tuple[float, ...] = (-1.0, 1.0)

#: The name kinds generated and displayed - exactly the two ``GRID`` points, nothing else.
#:
#: A third "neutral" kind (a data-collection artifact: a batch id, a device code) used to be requested
#: alongside these. It was never displayed, because the gold turns on whether a driver's own name is the
#: *aligned* one, so a neutral name produces the same gold as a misaligned one in every cell - a third
#: point on a distractor-strength scale rather than a fourth quadrant. It cost a third of every
#: generation's output tokens and was a live rejection source (administrative names like
#: ``page_footer_version`` are inherently discrete and tripped the continuity check, discarding an
#: otherwise-good SCM). Restoring a 0.0 level means asking for that kind again here and in
#: ``_generator_prompt``.
#:
#: Note this is the *generated name* kind, distinct from ``LabelProvenance.node_kind``, which still uses
#: "neutral" to mean "no naming manipulation at all" for the name-blind control
#: (``task/labeling.py::abstract_labels``) and is read by ``scoring/score.py::quadrant``.
_KINDS: tuple[str, ...] = ("aligned", "misaligned")
# No snake_case gate. It was rejecting names that were correct as *names* and wrong only in letter case -
# scientific unit abbreviations carry capitals, so `groundwater_nitrate_n_mgL_z`, `dissolved_oxygen_mgL_z`
# and `conductivity_in_groundwater_uScm_z` all failed it, discarding an otherwise-good SCM over notation.
# The generator prompt still asks for snake_case, so style stays consistent by default; it is simply no
# longer a reason to throw the instance away.
# Placeholder / degenerate name patterns that must be rejected (generators fall back to these).
_PLACEHOLDER = re.compile(r"^(feature|latent_component|component|channel|col|var|node|v|x)_?\d*$|^[a-z]\d+$")
# Tokens implying a non-continuous variable (the data is standardized continuous). Descriptions only.
_NONCONTINUOUS = ("binary", "yes/no", "yes-no", "categorical", "count of", "0/1", "indicator")

#: Name tokens that denote a *nominal or identifier* variable, which cannot be a standardized continuous
#: column. Matched as whole underscore-separated tokens, so ``body_mass_index`` and ``sample_hemolysis``
#: are untouched while ``form_revision_number`` and ``record_id_sequence`` are rejected.
#:
#: This gate exists because of a measured interaction between two other requirements. Asking the generator
#: for *in-domain procedural* misaligned names (``STAY INSIDE THE DOMAIN``, added to stop it reaching for
#: office furniture) pushes it toward exactly the administrative vocabulary that is naturally discrete -
#: a form revision, a batch id, a language version. A build then lost a whole SCM to
#: ``"description of 'survey_language_version' implies a non-continuous variable"``: the two instructions
#: pulled against each other and the revision budget ran out. The generator prompt now asks for continuous
#: procedural quantities explicitly (a duration, rate, fraction, precision, offset), so this is the
#: backstop rather than the primary mechanism.
#:
#: Deliberately narrower than the description-side list, and narrowed again by measurement. Counts and
#: batch indices are out: a count with many distinct values standardizes to something defensibly
#: continuous, and banning them rejected several otherwise-good names (``questionnaire_item_count``,
#: ``hormone_panel_draw_count``). ``status`` is out for the same reason but at a higher price - it cost a
#: whole SCM in a 5-seed probe over ``serum_vitamin_status_index``, which reads as an ordinary continuous
#: clinical severity index. Every token that remains is unambiguously nominal: a version, an identifier, a
#: class. The rule is that a token earns its place only when no continuous reading of it exists.
_DISCRETE_NAME_TOKENS = frozenset(
    {
        "version", "versions", "revision", "revisions", "id", "ids", "identifier", "identifiers",
        "serial", "serials", "code", "codes", "category", "categories", "class", "classes",
        "flag", "flags", "type", "types", "tier", "tiers", "label", "labels",
    }
)  # fmt: skip


def discrete_name_tokens(name: str) -> list[str]:
    """Return the nominal/identifier tokens in ``name`` (empty == fine for a continuous column)."""
    return sorted(_DISCRETE_NAME_TOKENS.intersection(name.lower().replace("-", "_").split("_")))


#: Structural-role vocabulary that must never reach the agent, enforced deterministically (``_pre_check``).
#:
#: This is the firewall the pipeline previously lacked. The generator prompt has always *asked* for names
#: that merely evoke a role; measured on the stored pilot it wrote the role itself instead - 22% of
#: misaligned candidate names and 47% of misaligned descriptions carried this vocabulary
#: (``deprivation_common_cause_index``, ``stress_to_guilt_mediator``, an "instrument", "no bearing on
#: pollution"). Worse, the leak was 7x asymmetric between the two skins, so counting these tokens
#: identified the naming condition on 9 of 10 pilot instances - enough to score 0.950 against a 0.750
#: no-calibration ceiling without doing any causal reasoning. A prompt instruction cannot fix that: the
#: prompt already said "never narrate causal relationships". Only an enforced reject can.
#:
#: Deliberately over-broad. ``pathway`` is a real biological term and ``driver`` a real business one, but
#: both were observed leaking, and a benchmark whose manipulation is detectable by keyword is worth more
#: than a marginally more natural name. A rejected proposal is re-prompted, not discarded.
_ROLE_TOKENS = (
    "confound", "mediat", "collider", "decoy", "instrument", "moderator", "covariate",
    "common cause", "common_cause", "commoncause", "shared cause", "shared_cause",
    "upstream", "downstream", "pathway", "on-path", "on the causal", "intermediate",
    "driver", "proxy", "inert", "backdoor", "exogenous", "adjust",
    "no bearing", "no effect", "unrelated", "not related", "irrelevant",
)  # fmt: skip
#: Relationship narration - banned in descriptions (and names), for the same reason and by the same gate.
#: A data dictionary states what a column *measures*; the moment it states what that column *does to* another
#: column, it has handed over part of the graph the agent is supposed to recover.
_RELATION_TOKENS = (
    "causes", "caused by", "causal", "leads to", "results in", "gives rise",
    "influenc", "affects", "impacts", "drives", "determines", "depends on",
    "predicts", "associated with", "correlated with", "linked to", "contributes to",
)  # fmt: skip


def role_vocabulary_hits(text: str) -> list[str]:
    """Return the banned role/relationship tokens present in ``text`` (empty == clean).

    Public because it is also the right scan for auditing an already-built dataset, not only for gating a
    fresh proposal - the stored pilot was measured with exactly this list.
    """
    lowered = text.replace("_", " ").lower()
    return [token for token in (*_ROLE_TOKENS, *_RELATION_TOKENS) if token in lowered]


@dataclass(frozen=True)
class NamingTriples:
    """One coherent domain's name choices per node: node -> {kind -> (name, description)}.

    ``kind`` ranges over ``_KINDS`` (aligned / misaligned). The name is historical - it predates neutral
    naming being dropped - and is kept because it is the public type consumers already import.
    """

    domain: str
    context: str
    choices: Mapping[str, Mapping[str, tuple[str, str]]]


def construct_labeling(triples: NamingTriples, scm: SCM, level: float, producer: str) -> Labeling:
    """Deterministically build a ``Labeling`` at ``level`` from a triple set (no LLM, exact alpha).

    ``level``'s sign picks one kind - aligned (>0) or misaligned (<0) - applied uniformly to every
    non-outcome node; only its sign is read, matching ``GRID``'s two points. The outcome is always aligned
    (its real name is disclosed to the agent regardless of level). ``level == 0`` is rejected: no neutral
    name is generated any more (see ``_KINDS``), so there is nothing to display at that level.
    """
    if level == 0:
        raise ValueError("level 0 (neutral naming) is not generated; GRID is (-1, +1) - see _KINDS")
    scaffold = scm.scaffold
    uniform_kind = "aligned" if level > 0 else "misaligned"

    node_to_label: dict[str, str] = {}
    schema: dict[str, str] = {}
    node_kind: dict[str, str] = {}
    counts = dict.fromkeys(_KINDS, 0)
    for node in scaffold.node_names:
        kind = "aligned" if node == scaffold.outcome else uniform_kind
        name, description = triples.choices[node][kind]
        node_to_label[node] = name
        schema[name] = description
        node_kind[node] = kind
        counts[kind] += 1

    provenance = LabelProvenance(
        alignment=level,
        n_aligned=counts["aligned"],
        n_neutral=0,  # no neutral name kind is generated; the field stays for dataset-schema compatibility
        n_misaligned=counts["misaligned"],
        node_kind=node_kind,
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
        if not set(kinds) >= set(_KINDS):
            issues.append(f"node {node} is missing one of {sorted(_KINDS)}")
            continue
        for kind, (name, desc) in kinds.items():
            if kind not in _KINDS:
                continue  # tolerate an extra kind the model volunteered; only _KINDS is displayed
            names.append(name)
            if _PLACEHOLDER.match(name):
                issues.append(f"name {name!r} is a placeholder, not a real-world concept")
            if not desc.strip():
                issues.append(f"empty description for {name!r}")
            if any(tok in desc.lower() for tok in _NONCONTINUOUS):
                issues.append(f"description of {name!r} implies a non-continuous variable")
            if tokens := discrete_name_tokens(name):
                issues.append(
                    f"name {name!r} denotes a nominal/identifier variable ({', '.join(tokens)}); every "
                    f"column is a continuous standardized measurement - use a duration, rate, fraction, "
                    f"concentration, precision or offset instead"
                )
            # The firewall (`_ROLE_TOKENS`): enforced, not requested. Checked on name and description
            # alike, and on both kinds, so neither skin can be identified from its word choice.
            if hits := role_vocabulary_hits(name):
                issues.append(
                    f"name {name!r} states a structural role or relationship ({', '.join(sorted(hits))}); "
                    f"name what the variable measures in the domain, never the part it plays in the graph"
                )
            if hits := role_vocabulary_hits(desc):
                issues.append(
                    f"description of {name!r} states a structural role or relationship "
                    f"({', '.join(sorted(hits))}); describe only what is measured"
                )
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
        return format_menus(sample_menus(causenet_dump, rng))

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
    """Map each node to an opaque display id (``v0..vk``) for the offline prompts."""
    return {node: f"v{i}" for i, node in enumerate(scm.scaffold.node_names)}


def _effect_bucket(effect: float, scale: float) -> str:
    """Coarse magnitude band for ``effect``, relative to the instance's own median |effect|.

    The generator is told a band, never the number. It needs *some* magnitude signal - "aligned" means a
    strongly-causal name belongs on a large-effect driver and an inert one on the zero-effect decoy, which
    is not expressible without it - but the exact value is strictly more than that requires, and handing
    over ``+0.37`` invites a name that encodes the estimate the agent is supposed to compute (audit P0:
    "hide exact effect magnitudes from the label generator"). Relative to ``effect_scale`` rather than
    absolute, because effects shrink as ``1/sqrt(N)`` with driver count, so a fixed cut would mean a
    different thing at every task size.
    """
    if abs(effect) <= 1e-9:
        return "NO effect at all (exactly zero)"
    ratio = abs(effect) / scale
    band = "weak" if ratio < 0.7 else "moderate" if ratio < 1.3 else "strong"
    return f"a {band} effect"


def _graph_facts(scm: SCM) -> str:
    """State the true structure once, per node: role, coarse effect band, and what it attaches to.

    Replaces a global ``v2->v1; v1->v0; ...`` edge dump. The generator names nodes; what it needs for a
    coherent domain is which nodes belong together (this confounder sits behind *that* driver), and the raw
    edge list expressed that only implicitly, in 46 pairs it had to re-derive. Stating each node's own
    attachment is smaller, directly usable, and drops the redundant second encoding of the same graph.
    """
    ids = _ids(scm)
    scaffold = scm.scaffold
    scale = effect_scale(scm)
    decoys = scaffold.nodes_with_role(Role.DECOY)
    children: dict[str, list[str]] = {}
    for parent, child in scaffold.edges:
        children.setdefault(parent, []).append(child)

    lines = []
    for node in scaffold.node_names:
        if node == scaffold.outcome:
            lines.append(f"{ids[node]}: the OUTCOME")
            continue
        role = "zero-effect decoy" if node in decoys else scaffold.roles[node].value
        parts = [f"{ids[node]}: {role}, has {_effect_bucket(scm.total_effect_on_outcome(node), scale)}"]
        if node in scaffold.drivers:
            if adjust := scaffold.correct_adjustment.get(node, ()):
                parts.append(f"shares background with {', '.join(ids[a] for a in adjust)}")
            if (mediator := scaffold.mediator_of(node)) is not None:
                parts.append(f"acts through {ids[mediator]}")
        else:
            # Non-candidates: name what they attach to, so the domain stays coherent around each driver.
            attached = [ids[c] for c in children.get(node, ()) if c != scaffold.outcome]
            if attached:
                parts.append(f"sits behind {', '.join(attached)}")
        lines.append("; ".join(parts))
    return "Nodes (id: role, effect size, what it relates to):\n  " + "\n  ".join(lines)


def _generator_prompt(scm: SCM, grounding: str) -> str:
    """Prompt the generator for per-node aligned/misaligned name pairs."""
    ids = _ids(scm)
    example = ids[scm.scaffold.node_names[0]]
    return f"""\
Name the variables of a hidden causal graph for a benchmark. Below is the TRUE structure.

{_graph_facts(scm)}

Real domain concepts to draw on (a cause-effect knowledge base): {grounding}

Pick ONE coherent real-world domain. For EVERY node give TWO candidate names, each with a one-line
description of what the variable *measures* (a data dictionary; the data is continuous & standardized -
never binary/count/categorical):
- "aligned": a name whose real-world prior MATCHES the node's true role - a well-known cause of the
  outcome on a large-effect driver, a quantity nobody would expect to matter on the zero-effect decoy, a
  background characteristic that plausibly shapes several measurements on a confounder, a step that the
  domain treats as sitting between the driver and the outcome on a mediator, the real target on the outcome.
- "misaligned": a name whose real-world prior points at a DIFFERENT role than the node truly has, on the
  axes the task grades (relevance & role, NOT sign) - a well-known cause of the outcome on the ZERO-effect
  decoy; a PERIPHERAL OR PROCEDURAL MEASUREMENT OF THE SAME DOMAIN on a large-effect driver; and for a
  confounder or mediator, a name the domain would file under the other one of those two.

STAY INSIDE THE DOMAIN - this is as important as the vocabulary ban below. Every name in BOTH sets must be
a quantity a practitioner in the chosen domain would plausibly record in this very dataset. A misaligned
name is one whose ROLE is wrong, never one that is absurd or off-topic. Use the domain's own peripheral
quantities: sampling and instrument metadata, protocol or administrative measures of that field, secondary
or incidental readings. Do NOT reach for unrelated everyday objects. Writing "pen_ink_color_index",
"elevator_wait_seconds" or "shoe_size_metric" into a clinical dataset is a FAILURE - it announces that the
names are junk before the reader has looked at the data, which destroys the whole experiment. Done right,
for a paleoclimate study: "core_recovery_percentage", "iridium_measurement_precision" - real quantities
that field genuinely records, which simply happen not to be causes of the outcome.

EVERY column, in both sets, holds a CONTINUOUS standardized measurement - so every procedural name must be
a continuous quantity too: a duration, rate, interval, fraction, percentage, concentration, precision,
drift, latency or offset. NEVER a version, revision, identifier, serial, code, category, class, type or
flag - those name things that take a handful of discrete values and cannot be what this data holds.
Write "sample_spacing_resolution", "calibration_drift_rate", "data_gap_fraction";
do NOT write "form_revision_number", "record_id_sequence", "survey_language_version".

CRITICAL - the names and descriptions are shown to a model that must recover this structure from data
alone. They must never TELL it the answer. A name or description is REJECTED outright if it contains any
structural or relational vocabulary: confounder, mediator, collider, decoy, instrument, driver, pathway,
upstream, downstream, common cause, proxy, inert, exogenous, adjust, "no bearing", "no effect",
"unrelated", or any verb of influence (causes, affects, influences, drives, determines, predicts,
contributes to, associated with, linked to). Write ONLY what the variable measures, exactly as a real data
dictionary in that domain would: "somatic_tension_index: standardized self-reported muscle tension score".
Do NOT write "on the causal pathway", "acts as a common driver", "has no bearing on X".

A domain expert shown only one of the two column lists, without being told which, must not be able to say
whether it is the aligned or the misaligned one. Both must read as an ordinary data dictionary from the
same study - same register, same specificity, same kind of quantity. The ONLY thing that distinguishes
them is which concepts sit in which structural position, which is invisible without the graph.

All names snake_case and ALL distinct. Return ONLY JSON of the form:
{{"domain": "...", "context": "...", "nodes": {{
  "{example}": {{"aligned": {{"name":"...","description":"..."}}, "misaligned": {{"name":"...","description":"..."}}}}
}}}}"""


def _critic_prompt(scm: SCM, triples: NamingTriples) -> str:
    """Prompt the critic to verify the complete card - names, descriptions, and shared context.

    Sees the *whole* card, not just names. The previous prompt listed names alone, so every description
    went unreviewed - and descriptions were measured to be the worst leak channel in the stored pilot
    (47% of misaligned ones carried structural vocabulary against 27% of aligned, a 7x asymmetry on graded
    candidates). Reviewing what it cannot see was the defect; ``_pre_check`` now also enforces the
    vocabulary ban deterministically, so this is the judgement layer on top of a hard gate, not the gate.
    """
    ids = _ids(scm)
    # Only the displayed kinds (see _KINDS) - the critic must not reject an SCM over a name no instance
    # will ever show.
    listed = "\n".join(
        f"  {ids[node]} ({'outcome' if node == scm.scaffold.outcome else scm.scaffold.roles[node].value}):\n"
        + "\n".join(f"      {kind}: {name} - {desc}" for kind, (name, desc) in kinds.items() if kind in _KINDS)
        for node, kinds in triples.choices.items()
    )
    return f"""\
Check a proposed naming of this TRUE causal graph. Domain: {triples.domain!r}.

{_graph_facts(scm)}

Shared context shown to the agent: {triples.context!r}

Proposed name + description per node:
{listed}

Verify, for every node:
1. LEAKAGE (most important). No name, description, or the context above may state or hint at a structural
   role or a relationship between variables - no "confounder"/"mediator"/"driver"/"pathway"/"instrument"/
   "upstream"/"common cause"/"inert"/"no bearing", and no "causes"/"affects"/"influences"/"predicts"/
   "associated with". Each description must state only what is measured. Flag every instance.
2. DOMAIN DRIFT (the most common failure). Every name in BOTH sets must be a quantity the chosen domain
   would actually record. Flag any name that is an unrelated everyday object or piece of furniture -
   "pen_ink_color_index", "elevator_wait_seconds", "shoe_size_metric" in a clinical study. Misaligned
   names must be peripheral or procedural measurements *of that domain* (sampling metadata, instrument
   precision, protocol measures), not off-topic ones. If one set is visibly more mundane than the other,
   say so - a reader must not be able to tell which set is which without the graph.
3. ROLE FIT. The "aligned" name's real-world prior genuinely matches the node's role, and the "misaligned"
   name genuinely points elsewhere - a familiar cause-name on the zero-effect decoy, an unremarkable
   quantity on a large-effect driver.
4. CONTINUITY. Every column is a continuous standardized measurement. Flag any name or description that
   denotes something discrete - a version, revision, identifier, serial, code, category, class, type or
   flag - since such a variable cannot hold this data.
5. Placeholder or incoherent names.

Judge only what is shown - do not ask for kinds that are not listed.
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
    """Parse the generator output (keyed by opaque ids) into a ``NamingTriples`` covering every node."""
    id_to_node = {vid: node for node, vid in _ids(scm).items()}
    raw = _extract_json(text)
    choices: dict[str, dict[str, tuple[str, str]]] = {}
    for vid, kinds in raw["nodes"].items():
        node = id_to_node.get(vid)
        if node is None:
            raise ValueError(f"naming references unknown node id {vid!r}")
        choices[node] = {kind: (spec["name"], spec["description"]) for kind, spec in kinds.items()}
    return NamingTriples(domain=raw["domain"], context=raw["context"], choices=choices)
