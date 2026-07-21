# CausalAgentBench — Orientation-Ambiguity Scaffold Plan (a non-identifiable row, v1)

> **Status: approved for implementation handoff** (verified across successive design reviews). Builds on
> the shipped SCM layer (`scm/`), task layer (`task/`), scoring layer (`scoring/`), and the semantic
> labeler (`semantic_labeler_plan.md`). Conventions: package `causalagentbench` (src layout, Python 3.12,
> uv, hatchling; ruff + mypy strict + pytest; line-length 120; concise ASCII docstrings). Dataclasses
> `frozen=True`.

---

## Context

Every scaffold today (`CANONICAL`) is graded as identifiable, so the per-driver `identifiable: bool` in
the `Answer` is never `False` and **abstention is never correct**. This is the smallest fair, verifiable
way to activate that half — making "concede when the result can't be trusted" a real scored skill.

Effects are identifiable only *relative to* causal assumptions and to the information an instance actually
supplies, so **identifiability is declared per-scaffold, never benchmark-wide** (§4/§6). This scaffold uses
an **observational-CPDAG** regime: under Markov + faithfulness + causal sufficiency + acyclicity + i.i.d. +
no selection bias + linear-Gaussian, the **population** observational distribution identifies a CPDAG (a
finite sample only supports an *estimate*) that identifies driver `B` but **not** driver `A`, because a
covariate `Z`'s orientation is undetermined. The ambiguity is *detectable from data* (an unoriented CPDAG
edge), so the agent can know it is stuck and must rely on domain knowledge — and abstention, when that
knowledge is absent, is a verdict the agent has the information to reach. (Unobserved confounding is out of
v1: its non-identifiability is undetectable from data, so it could only fairly host "heed a warning.")

The controlled contrast, over one shared data table, is:

* **silent** context → `A`'s effect is not resolved → gold = **abstain**;
* **resolved** context → one truthful sentence supplies `A`'s orientation → gold = **do not abstain**.

The direction does not leak through the observational table or uncontrolled semantics; the resolved
sentence reveals it deliberately. **What this scaffold tests: identification, adjustment reasoning, and
ordinal effect comparison — not numerical effect estimation** (the `Answer` has no effect field; the true
effect `tau` only determines the gold ranking). When resolved, the agent must mark the effect identifiable,
give a valid adjustment set, and place the driver correctly in the ranking derived from true absolute
effects.

This experiment is self-contained and measurable on its own (§12). Using it as the *identified comparison
row* for the broader "agents fail to condition semantic reliance on identifiability" headline additionally
requires **auditing `CANONICAL` under `OBSERVATIONAL_CPDAG`** — see §6.

---

## 1. Scope (minimal)

* One scaffold family with **both true orientations** of the `A–Z` edge, generated as a matched pair
  sharing one data table (§3), so the true direction is not exposed and the observational equivalence is
  exact.
* A **condition-dependent gold rule** (§5): `A` is identifiable only when the context supplies the
  truthful orientation; else the agent must abstain. `B` is always identifiable.
* Supporting pieces, each minimal: per-scaffold `Regime` + task-layer `IdentificationSpec`,
  `TaskCondition`, `EvaluationGroup` on the hidden key (§4); a per-scaffold regime preamble and a fixed
  `identifiable` definition (§6); the one-sentence resolved/silent intervention over **identical neutral
  names** (§7); an exhaustive 4-node CI routine used (separately) for population spec-verification and as a
  data-only sample baseline, plus population-only construction gates (§8); a two-level, contract-strict,
  driver-generic scorer (§9); an evaluation-isolation protocol (§10); and explicit versioned artifact
  migration (§11).

**Out of scope (roadmap, §13):** an unobserved-confounder scaffold; a *misaligned* (false-orientation)
pole on `A` (a wrong orientation is not data-refutable, so trusting it is not a catchable error — the fair
gullibility test lives in the identifiable row via the semantic labeler); crossing the naming-α grid with
orientation modes (α stays neutral until this scaffold is validated); a full PC/FCI library; the runner/RL
harness; additional ambiguity topologies; the CPDAG-given ablation (§13); the public/private artifact split
(§11, a release prerequisite).

---

## 2. The scaffold and its equivalence class (`scm/scaffold.py`)

Four observed nodes; the observed distribution is Markov equivalent across the `A–Z` orientation.

```
mediator truth :  A -> Z -> Y      B -> Y     (Z mediates A's effect; A a real driver, tau(A) != 0)
confounder truth: A <- Z -> Y      B -> Y     (Z a common cause; A has no path to Y, tau(A) = 0)
```

Both DAGs share skeleton `{A-Z, Z-Y, B-Y}`, non-adjacency `A-B`, and the single unshielded collider
`Z -> Y <- B` (B, Z non-adjacent and marginally independent, so conditioning on the collider `Y` makes
them dependent). That collider compels `Z -> Y` and `B -> Y`; Meek's rules do not fire on `A-Z` (nothing
points into `Z`), so **`A - Z` stays unoriented**. Both readings entail `A ⟂ Y | Z`, so no
conditional-independence test can orient the edge — `tau(A)` is not point-identified (mediated effect vs
zero). `B` is compelled and unconfounded → always identifiable.

**Roles** (reuse the enum; correct per twin): mediator truth `A=DRIVER_CLEAN, Z=MEDIATOR`; confounder
truth `A=DECOY, Z=CONFOUNDER` (`A<-Z->Y` *is* the decoy pattern — zero effect, association purely via the
common cause). `B=DRIVER_CLEAN`, `Y=OUTCOME` in both. Candidates `=(A,B)`; `Z` is a presented covariate,
not a candidate; no `A->Y` edge. The scaffold carries **no** identifiability field — that is epistemic,
not structural (§4). (The enum conflating candidate-status with structural role is a known wart →
`backlog.md`; downstream logic reads graph **edges**, not roles.)

---

## 3. Matched-pair generation with shared data (`scm/ambiguity.py`)

Two Markov-equivalent Gaussian DAGs represent the *same* distribution, so `generate_ambiguity_pair(seed)
-> AmbiguityPair`: (a) draw the **mediator-truth** SCM via UUMC `generate`; (b) take its observed
covariance `Sigma`; (c) **construct the confounder-truth twin from `Sigma`** by population regressions —
`Z` root; coefficient `Z->A` = regression of `A` on `Z`; coefficients `Z,B->Y` = regression of `Y` on
`[Z,B]`; noise variances = the corresponding conditional variances — so the twin has **different**
coefficients and noise (reversing the edge with the original weights would *not* preserve the
distribution) but identical `Sigma` and `tau(A)=0`. Assert positive residual variances numerically.

Because the distributions are identical, **one sampled data table is a valid draw from both twins**; the
pair shares the exact same data, and only the true orientation (hence `tau(A)` and `A`'s valid adjustment
sets) differs. `AmbiguityPair` carries both SCMs, the shared data, and a **`pair_id`** deterministic from
`benchmark_version + scaffold_family + generator_config_digest + root_seed` (never Python's runtime hash),
where `generator_config_digest` is **SHA-256 over the canonical JSON of the generator config**.

**Equivalence proof (§12):** equality of the observed covariance is the proof (different `tau(A)`,
different valid-adjustment requirements). Equal covariance already implies equal zero-mean Gaussian
likelihood, so a likelihood check is retained only as a redundant implementation-consistency assertion. The
skeleton/v-structure check is supporting evidence.

---

## 4. Identification descriptor + task condition (task layer)

Identifiability depends on the observed set, query, regime, and information supplied — not the true graph
— so these live on the **hidden key**, in the **task layer** (`task/identification.py`,
`task/conditions.py`), never on `Scaffold` and never imported from `scoring`. Type names are topology-
generic (no `A`/`Z` hard-coding).

```python
class Regime(Enum):
    OBSERVATIONAL_CPDAG = "observational_cpdag"   # agent infers the CPDAG from data (this scaffold)
    LEGACY_UNSPECIFIED  = "legacy_unspecified"    # pre-existing artifacts; no regime was declared

class OrientationMode(Enum):
    SILENT = "silent"
    TRUTHFUL = "truthful"
    NOT_APPLICABLE = "not_applicable"             # non-ambiguity scaffolds (e.g. CANONICAL)

@dataclass(frozen=True)
class UndirectedEdge:                             # canonical, order-free; clean JSON (sorted 2-array)
    a: str
    b: str
    # __post_init__: reject self-edge (a == b); canonicalize order via object.__setattr__ so {A,Z}=={Z,A}.
    # node membership is validated by IdentificationSpec against the scaffold.

@dataclass(frozen=True)
class IdentificationSpec:
    regime: Regime
    ambiguous_edges: frozenset[UndirectedEdge]    # {UndirectedEdge("A","Z")}
    affected_drivers: frozenset[str]              # {"A"} — not point-identified under `regime`

@dataclass(frozen=True)
class TaskCondition:
    orientation_mode: OrientationMode
    orientation_claims: frozenset[tuple[str, str]]  # ONLY the resolving directions added by the controlled
                                                    # intervention (e.g. {("A","Z")}), not every relationship
                                                    # the context sentence might linguistically imply;
                                                    # SILENT/NOT_APPLICABLE -> empty.
```

`InstanceKey` gains `identification: IdentificationSpec`, `condition: TaskCondition`, and
`group: EvaluationGroup | None = None` where `EvaluationGroup` holds the `pair_id` (a per-instance
singleton cannot be defaulted without record identity; the loader assigns singletons — §11).
`LabelProvenance` stays naming-only (naming-α is its single source of truth).

**The `IdentificationSpec` is asserted metadata and must be independently verified** — because gold
consumes `affected_drivers`, a wrong spec would validate its own answer. A construction test derives the
equivalence class from the scaffold via the §8 routine (on the **population** covariance) and requires
`derived ambiguous_edges == spec.ambiguous_edges` and `derived non-identified drivers ==
spec.affected_drivers`. Validators also require: every `UndirectedEdge` references two distinct known
nodes; every `affected_drivers` member is a candidate; and, for a `TRUTHFUL` condition, `orientation_claims`
**equals the required resolving orientation** for the affected edge(s) (exact match, not a subset — a
subset could admit an unrelated truthful edge while still marking the condition `TRUTHFUL`), read from graph
**edges**, not `Role`. A false orientation on `A` is never presented.

---

## 5. Gold rule (`scoring/gold.py`)

`gold_answer(key: InstanceKey) -> Answer` — takes the whole key (so it has `node_to_label`) and emits the
public `Answer` in **display labels**. It **rejects `LEGACY_UNSPECIFIED`** (raises), never silently
treating legacy instances as identified. `tau` is used only internally to derive the ranking; no effect
field is added. Per candidate:

* `B` (not affected): `identifiable=true`; a valid backdoor set (`{}` here); always ranked.
* `A` (affected):
  * `orientation_mode == SILENT` → `identifiable=false` (abstain), **omitted from `ranking`** (only
    non-identifiable candidates are omitted).
  * `orientation_mode == TRUTHFUL` → `identifiable=true`, with the twin's **true** minimal valid set
    (mediator: `{}`, `tau(A)!=0`; confounder: `{Z}`, `tau(A)=0`) and **ranked**, including the zero-effect
    confounder case (ranked last).

Rankings: SILENT → `[B]`; resolved mediator → `sorted([A,B])` by `|tau|`; resolved confounder → `[B, A]`.
Correctness is judged on identifiability, adjustment-set validity, and ranking position — not on a
numerical effect.

---

## 6. Declared regime + the meaning of `identifiable` (`task/instance.py`)

**Definition (fixed).** In the prompt and the gold, *identifiable* means **identifiable under the declared
regime and all information supplied in the instance, including explicit causal context** — not "from the
raw data alone." This keeps the output field's meaning constant across the silent and resolved conditions
(the resolved sentence is *supplied information*, not a regime change). `build_prompt` is updated
accordingly (the current "identifiable from this data" wording is removed).

**Preamble, per this instance's regime** (from the key), never benchmark-wide. For `OBSERVATIONAL_CPDAG`:
*observational effects are identifiable only relative to causal assumptions; under causal sufficiency,
Markov + faithfulness, acyclicity, i.i.d. sampling with no selection bias, and a linear-Gaussian model, the
population observational distribution identifies a CPDAG, and the agent is expected to **infer** that CPDAG
from the data (it is not supplied); domain context in the instance may supply additional identifying
information.*

Analysis keeps three states distinct: observational identification, context-resolved identification, and
non-identification.

> **Prerequisite for the cross-scaffold headline:** before `CANONICAL` serves as the *identified*
> comparison row, formally audit its CPDAG under `OBSERVATIONAL_CPDAG` (what the agent actually has —
> observational data + names), confirming every graded effect is invariant across the equivalence class
> those assumptions leave open, or specify exactly which context statements constitute structural
> assumptions. The self-contained orientation experiment (§12) does not depend on this. Existing artifacts
> stay `LEGACY_UNSPECIFIED` until audited — never silently migrated.

---

## 7. Resolved vs silent context (`generation/labeler.py`)

The controlled intervention is **exactly one factual claim** — the direction of the `A–Z` edge. **All
episodes of a pair use identical, neutral surface names and column schema, independent of `A`'s structural
role** (`DRIVER_CLEAN` vs `DECOY`) — otherwise the hidden orientation leaks through semantics. `A`/`Z`
descriptions are orientation-neutral; the resolved skin adds **exactly one sentence** stating the true
orientation (mediator: "`Z` lies on the causal pathway from `A` to the outcome"; confounder: "`Z` is a
prior common cause of `A` and the outcome"). The correctness critic verifies the silent skin implies no
direction and resolved/silent differ by that single sentence. **Naming-α stays neutral** while validating
this scaffold; do not cross the α grid with orientation yet (§1). For the v1 bootstrap both contexts may
be hand-authored.

**Episode set per pair (no duplicate public tasks).** The silent gold is identical across both twins
(`A` abstains; `B`'s gold matches because the shared `Σ` gives the same `Y~Z+B` fit). So emit **one shared
silent episode + two resolved episodes** per pair (three public episodes), all linked by `pair_id`; the
silent episode anchors both within-pair contrasts. The silent episode's hidden key **deterministically uses
the mediator SCM** by convention — the choice cannot affect gold or scoring, because `A` is non-identified
(abstain) regardless of twin and `B` is invariant across twins.

---

## 8. Construction gates + exhaustive CPDAG routine (`scoring/ambiguity_gates.py`, `scoring/baselines.py`)

**Acceptance uses population (SCM) gates only** — no sample-selection bias. `generate_ambiguity_gated(seed)`
draws pairs until:

* **equivalence:** `‖Sigma_med − Sigma_conf‖_max < COV_MATCH_TOL`; `tau_conf(A) ≈ 0`.
* **effect + ranking:** `|tau_med(A)| ≥ MIN_ABS_EFFECT`; `|tau_med(A) − tau_conf(A)| ≥ MIN_TAU_GAP`; when
  `A`,`B` are both ranked (mediator × resolved), `||tau(A)| − |tau(B)|| ≥ MIN_EFFECT_GAP`; `B` identifiable
  with `|tau(B)| ≥ MIN_ABS_EFFECT`.
* **population strong-faithfulness / beta-min** (so the CPDAG is recoverable at finite n, and still a pure
  SCM gate — no sample noise selected): across the CI decisions defining the equivalence class,
  `min |theoretically-nonzero partial correlation| ≥ MIN_CI_MARGIN` and
  `max |theoretically-zero partial correlation| ≤ NUMERIC_ZERO_TOL`.

`MIN_CI_MARGIN` is **not an arbitrary constant**: calibrate and preregister it against sample size, the CI
test significance threshold, the multiple-testing policy, and the desired skeleton/CPDAG recovery
probability. **Separate calibration from evaluation seeds** — calibrate `MIN_CI_MARGIN`, CI thresholds, and
recovery targets on development seeds and **lock the configuration before generating release/evaluation
pairs**. The gate report **records every attempt** (root seed, attempt seed, rejection reason, accepted
attempt) so the reported **rejected-SCM fraction** and **partial-correlation-margin distribution** are
reproducible and free of cherry-picking; `generate_ambiguity_gated` **fails with a diagnostic report after
`MAX_ATTEMPTS`** rather than looping. Finite-sample conditional-independence strength is otherwise
**diagnostic-only** (reported, never gating).

**Two separate entry points, shared CI mechanics but strict information boundaries:**

* *Population spec-verifier* — may use the scaffold, `Sigma`, and `IdentificationSpec`. Over Gaussian
  partial correlations across **every permissible conditioning subset** of the four nodes it recovers the
  skeleton, identifies the unshielded collider, applies the orientation rule, enumerates the two compatible
  DAGs, and confirms `tau(A)` varies while `tau(B)` does not — deriving the equivalence class that verifies
  the spec (§4).
* *Data-only sample baseline* — `estimate_cpdag(task: ObservationalTask, config: BaselineConfig) ->
  BaselineResult`, where `ObservationalTask(columns, data, outcome, candidate_drivers)` treats `columns` as
  **opaque identifiers** and is constructed from `Instance` **deliberately excluding `context`,
  `column_schema`, and `prompt`** (so a data-only baseline cannot read the resolved sentence or any
  semantics). It must not see `affected_drivers`, `ambiguous_edges`, roles, the hidden SCM, or
  `InstanceKey`. It must be invariant to column order and display-label permutation, run without the key,
  and **identify the ambiguous candidate itself** (rather than being told it is `A`).

**Release criterion (not merely a statistic):** offline over the generated distribution, the **lower 95%
confidence bound on CPDAG recovery by the data-only baseline ≥ a preregistered target** before release, run
on the restricted `ObservationalTask` (equivalently the silent presentation).

---

## 9. Scorer (`scoring/score.py`)

Two levels, so parse failures are representable (the current `answer.parse_answer` raises before any
`Answer` exists):

```python
score_output(text: str, instance: Instance, key: InstanceKey) -> AnswerScore
    # parses; on AnswerParseError -> the parse-failure state below; else delegates.
score_answer(answer: Answer, instance: Instance, key: InstanceKey) -> AnswerScore
    # scores a parseable answer (parse_valid=True).
```

**No silent sanitizing.** Contract problems are collected structurally rather than as many booleans:

```
parse_valid: bool
contract_errors: tuple[ContractError, ...]       # ContractError(kind, detail); kinds cover:
    # UNKNOWN_VERDICT_DRIVER, DUPLICATE_VERDICT, MISSING_CANDIDATE_VERDICT,
    # UNKNOWN_ADJUSTMENT_VARIABLE, DUPLICATE_ADJUSTMENT_VARIABLE,
    # RANKING_UNKNOWN, RANKING_DUPLICATE, RANKING_MISSING, RANKING_EXTRA, VERDICT_RANKING_INCONSISTENT
contract_valid: bool                             # == (not contract_errors)
driver_scores: Mapping[str, DriverScore]         # paper aggregates over affected_drivers (e.g. A); B control
answer_coverage: float                           # fraction of gold-identifiable candidates the agent answered
ranking_score: float | None                      # rank agreement over gold-identifiable candidates
```

```python
class DriverScore:
    identifiable_correct: bool
    overanswered: bool                 # answered a gold-non-identifiable driver (the key failure)
    false_abstention: bool             # abstained on (or failed to answer) a gold-identifiable driver
    adjustment_valid: bool | None      # None when gold non-identifiable OR the agent abstains
    resolved_driver_success: bool | None   # verdict AND adjustment success (NOT ranking, which has its own
                                            # metric); None unless the driver is affected AND mode==TRUTHFUL
```

Applicability semantics (so "not applicable" is never confused with success/failure): `adjustment_valid` /
`resolved_driver_success` as above; `ranking_score = None` when fewer than two gold-identifiable candidates
exist; a **contract-invalid ranking scores 0.0** (not `None`) when ranking was applicable.

**Parse-failure state** (`parse_valid=False`, `contract_valid=False`, `answer_coverage=0.0`,
`ranking_score = 0.0` if ranking would otherwise be applicable else `None`): no driver is `overanswered`
(the agent answered nothing); per driver `identifiable_correct=False`, `adjustment_valid=None`,
`resolved_driver_success=None`; for a **gold-identifiable** driver `false_abstention=True` (no coverage),
for a **gold-non-identifiable** driver `false_abstention=False` (simply unanswered — neither over-answer
nor a produced abstention).

Adjustment sets are scored by **validity against the true graph** (backdoor criterion), not equality — for
mediator-`A` both `{}` and `{B}` are valid; for confounder-`A` both `{Z}` and `{Z,B}`; `{}` is invalid for
confounder-`A`. Reject forbidden/descendant/treatment/outcome/unknown labels before applying the criterion.
Validity is scored **conditional on answering**; never reward a set proposed for a non-identified driver.
Candidate-average accuracy is **not** a primary metric (inflated by the always-identifiable `B`); with two
drivers `ranking` is binary/degenerate.

---

## 10. Evaluation-isolation protocol

The episodes of a pair share one data table, so a persistent agent could leak the orientation from a
resolved episode into the silent one. The runner (separate deliverable) **must** guarantee: independent
agent state per episode; no access to sibling instances; randomized order; no cross-instance memory; and
confidence intervals **clustered by `pair_id`**, not by episode. The release configuration must also
require a **minimum number of independent `pair_id` clusters** (pair-clustered intervals are unreliable with
too few pairs). Documented here as a correctness requirement of the design.

---

## 11. Artifact migration (`testset/loader.py`, `scripts/build_testset.py`, `smoke.py`, exports)

Extending `InstanceKey` requires an **explicit versioned migration**, not reliance on silent dataclass
defaults (which could conceal a scientifically different regime). Bump the artifact schema version; the
loader reads new fields and, for pre-version records, **explicitly stamps**
`regime=LEGACY_UNSPECIFIED`, `orientation_mode=NOT_APPLICABLE`, empty `orientation_claims`, and a singleton
`EvaluationGroup`. `UndirectedEdge` serializes as a sorted two-element array. Touch points: the test-set
JSON schema + `loader.py` (versioned migration), `build_testset.py` (write the new fields and `pair_id`),
`smoke.py`, the canonical fixtures, and `__init__.py` exports (`Regime`, `OrientationMode`, `UndirectedEdge`,
`IdentificationSpec`, `TaskCondition`, `EvaluationGroup`, `ObservationalTask`, `ContractError`,
`gold_answer`, `score_output`, `score_answer`, `AnswerScore`, `DriverScore`, `generate_ambiguity_pair`,
`estimate_cpdag`). Verify the packaged `testset_v1.json` still reconstructs.

> **Release prerequisite (cross-cutting, not built here):** the package currently ships `W`, `tau`, and
> label mappings beside public instances. Matched twins stop *statistical* leakage but not artifact
> inspection; a real evaluation release needs **separate public tasks and private keys**. Tracked as a
> release gate for all scaffolds.

---

## 12. Verification (definition of done)

`uv run ruff check .`, `ruff format --check .`, `mypy src` clean; `pytest` green.

* **Equivalence:** the matched pair has equal `Sigma` (< `COV_MATCH_TOL`), `tau_med(A)!=0`,
  `tau_conf(A)≈0`, and different valid-adjustment requirements; positive residual variances asserted; equal
  Gaussian likelihood holds as a consistency assertion; one sampled table is a valid draw from both;
  deterministic `pair_id` (SHA-256 config digest). The graph test (shared skeleton + `A-B` non-adjacency +
  sole `Z->Y<-B` collider, `A–Z` unoriented, both entail `A ⟂ Y | Z`) is supporting evidence.
* **Distributions, not only seed 0:** keep the deterministic `seed=0` regression test **and** add a bounded
  multi-seed property test covering covariance equality, positive residual variance, `pair_id` determinism,
  and gate termination (accept within `MAX_ATTEMPTS`).
* **Population verifier:** derives `ambiguous_edges`/`affected_drivers` matching the `IdentificationSpec`; a
  deliberately wrong spec fails the construction test.
* **Data-only baseline:** `estimate_cpdag(ObservationalTask, ...)` recovers the skeleton, collider, two
  compatible DAGs, and the `tau(A)`-varies/`tau(B)`-invariant fact; invariant to column order and label
  permutation; constructed without `context`/`column_schema`/`prompt`/`InstanceKey`; identifies the
  ambiguous candidate rather than being told.
* **Gates:** `generate_ambiguity_gated(0)` passes the population gates including strong-faithfulness with a
  calibrated `MIN_CI_MARGIN`; the gate report records every attempt (seeds + rejection reasons); it fails
  with a diagnostic report after `MAX_ATTEMPTS`; rejected-SCM fraction and margin distribution reported; the
  offline CPDAG-recovery release criterion (lower-95%-bound) is defined and its check runs.
* **`identifiable` definition:** the prompt states the regime-and-context definition; no "from this data
  alone" wording remains; the regime preamble matches the key.
* **Gold rule:** across the episodes `gold_answer(key)` yields display-label answers — SILENT: `A`
  `identifiable=false`, `ranking=[B]`; resolved mediator: `A` valid set `{}`, `ranking=sorted([A,B])`;
  resolved confounder: `A` `identifiable=true`, valid set `{Z}`, `ranking=[B,A]`; `gold_answer` raises on
  `LEGACY_UNSPECIFIED`; the silent episode deterministically uses the mediator SCM; `label_to_node`
  round-trips.
* **Scorer:** `score_output` on invalid JSON/block returns the defined parse-failure state (nothing
  `overanswered`, gold-identifiable drivers `false_abstention=True`, zero coverage, `ranking_score`
  0.0-if-applicable/`None`); `score_answer` accepts `{B}` for mediator-`A`, rejects `{}` for confounder-`A`;
  `overanswered` fires on answering a SILENT `A`; `adjustment_valid`/`resolved_driver_success` are `None`
  where inapplicable; contract problems appear in `contract_errors` with `contract_valid == (not
  contract_errors)`; `resolved_driver_success` excludes ranking.
* **Contracts:** `UndirectedEdge` rejects self-edges and unknown nodes; `TRUTHFUL` `orientation_claims` must
  equal the required resolving orientation (an unrelated truthful edge is rejected).
* **Episodes/artifacts:** exactly three public episodes per pair (one shared silent + two resolved) with
  identical CSV, names, and schema, differing only by the one context sentence; packaged `testset_v1.json`
  reconstructs under the bumped schema via explicit migration.

---

## 13. Roadmap after v1

**High-value ablation (publication):** a **CPDAG-given** condition — main = agent infers the CPDAG from
data; oracle = the same CPDAG is explicitly supplied — separating two failure sources: (1) failure to
recover the observational equivalence class, and (2) failure to reason/abstain once the ambiguity is
explicit. It sharpens the over-answering finding ("agents over-answer *even when* the unoriented edge is
supplied").

Also: audit `CANONICAL` under `OBSERVATIONAL_CPDAG` so it (or a replacement) can be the identified
comparison row; an unobserved-confounder "heed the warning" scaffold; at least one additional ambiguity
topology; wiring resolved/silent authoring into the LLM labeler; crossing the two orthogonal axes —
**naming-alignment level** (the *ordinal* α grid of `semantic_labeler_plan.md`; α is ordinal, not
continuous — `backlog.md`) × **orientation mode** — once this simpler experiment is validated; a full
PC/FCI baseline; multiple sample sizes and model families; the runner enforcing §10 with pair-clustered CIs
and a minimum cluster count; the public/private artifact split (§11).

Interpretation, stated precisely: **resolved** performance measures whether the agent correctly
*incorporates an explicit causal assumption*; **silent** measures whether it *detects observational
underdetermination and abstains*; neither alone measures general causal discovery. The scaffold supports:
*holding observational evidence fixed, agents often fail to abstain when an effect is unresolved and fail to
use a valid resolving orientation consistently when it is supplied.* It does **not** support numerical
effect-estimation, broad semantic-reliance, general causal-discovery, or hidden-confounding claims; the
broader semantic-reliance headline waits for a formally valid identified row (naming-α is held neutral here
and `CANONICAL` is unaudited). Cross-ref `semantic_labeler_plan.md`, `backlog.md`.
```
