# CausalAgentBench — Procedural Scaffold Generation Plan (v1)

> **Status: §1-§12 all implemented and tested.** This is the authoritative description of *why* instances
> are generated the way they are. Named values below are the ones in force when each section was written;
> the code is the source of truth for current values, and the ones that have since moved are listed in
> "Superseded values" at the end of this banner. §11's fix is confirmed by
> measurement (§11.5): per-role ambiguity now genuinely splits (49.7%/51.1%, not the old 100%/0%), 0/100
> accepted instances have zero identifiable non-clean drivers (was 30%). §11.6 then made two of the
> retry loop's four rejection sources deterministic (topology and confounding construction), raising
> combined acceptance from 23.5% to 41.6% with no collapse at high driver count -
> `scripts/run_eval.py`/`build_testset.py` both wired to `generate_gated_procedural`. Builds on the shipped SCM layer (`scm/`), scoring layer
> (`scoring/`), task layer (`task/`), and the semantic labeler (`generation/labeler.py`). Superseded in
> turn on the effect-construction thresholds by `variance_budget_plan.md` (R1/R1b), which replaced
> §5a's fixed `MIN_MEDIATED_EFFECT` and confounding margin with driver-count-scaled equivalents.
>
> **Superseded values (2026-08-06).** The reasoning in each section stands; these numbers do not.
> `RESOLVE_PROBABILITY` 0.5 → **0.25** (balances the four quadrants; §11.1/§11.2).
> `ORACLE_TAU_THRESHOLD` 0.99 → **0.90** (0.99 discarded 43% of survivors over near-misses; §5).
> `n_drivers` → **`n_candidates`**, now the *total* task size with decoys carved out of it, fixed at 7.
> `DRIVER_COUNT_RANGE (3,8)` → `CANDIDATE_COUNT_RANGE (4,9)`; `DECOY_COUNT_RANGE (0,2)` → **(1,2)**.
> The naming grid `{-1, 0, +1}` → **`{-1, +1}`** (§11.8; neutral produced the same gold as misaligned).
> Four gates → **five**: a closed-form blind-policy floor (`SHORTCUT_ERROR_THRESHOLD = 0.25`) was added.
> `MIN_MEDIATED_EFFECT` no longer exists — see `variance_budget_plan.md`.
> Conventions: package
> `causalagentbench` (src layout, Python 3.12, uv, hatchling; ruff + mypy strict + pytest; line-length 120;
> concise ASCII docstrings). Dataclasses `frozen=True`.

---

## 0. Context

`CANONICAL` (`scm/scaffold.py`) is one hand-built, fixed topology: exactly three candidate drivers (`A`
confounded, `B` clean, `Dz` decoy), one confounder, one mediator. It was the right first move — it let the
soundness gates and the naming-trap design get verified against one known-good structure before
generalizing. Two costs of staying there, both already on record
(`backlog.md` §4, deck §08):

1. **Structural-position fingerprint.** Every instance shares the same node-degree signature. A model with
   cross-instance memory could learn "structural position X is always the confounder" as trivia instead of
   reasoning fresh per instance.
2. **Capped realism.** Exactly three drivers, always the same one-confounder-one-mediator shape,
   hand-verified once by a human running real regressions (`backlog.md` §4) rather than checked per
   instance. Real causal triage rarely narrows to three candidates, and real confounding rarely comes in
   exactly one flavor.

This plan replaces `CANONICAL` as the *only* scaffold with a **procedural generator**: every instance gets
its own topology — variable driver count, a genuinely varied mix of driver types (clean, single-confounder,
multi-confounder, instrumented), decoys — while preserving every guarantee `CANONICAL` currently provides:

* Exact closed-form ground truth (`τ = (I−W)⁻¹−I`, §02) — untouched, still linear-Gaussian, still Tier A.
* Soundness against the two shortcuts the existing gates defeat (naive correlation, "adjust for
  everything") — the four numeric gates in `scoring/gates.py` are reused, generalized.
* The one non-negotiable property: **at least one driver's incoming edge is genuinely, provably
  unresolvable from data alone**, so §03's naming axis stays load-bearing instead of decorative.
* Well-separated driver effects regardless of driver count (§5a) — discovered mid-implementation to need
  active construction, not something free generation happens to provide: driver effects shrink as `1/√N`
  under UUMC's variance conservation, which silently collapsed gate acceptance to 0% at 7+ drivers until
  fixed.

**Relationship to the earlier orientation-ambiguity plan** (since deleted
2026-08-04 once fully superseded). That plan built one hand-authored 4-node scaffold with
a binary silent/resolved contrast on a single node. This plan generalizes the same underlying idea —
data-unresolvable edges are the fair thing to test abstention on, not unobserved confounders — into
something discovered per driver, per procedurally-generated instance,
at scale. Its core identifiability reasoning was correct and is reused directly here; the scaffold itself
was built, then cut from v1 scope.

**What this deliberately does not touch:** nonlinear SCMs (a separate later change — see `backlog.md`
§11) and unobserved confounders (ruled out — undetectable from data, so naming can
never legitimately resolve it; see chat log 2026-07-23 for the full argument, not reproduced here).

**Correction on record (same day)**: an intermediate review pass cut the multi-confounder and instrument
gadgets, on the reasoning that they weren't guaranteed to preserve ambiguity / didn't add a distinct scoring
signal. Both reasons were wrong — see §1 and §2 for the corrected argument. Realism should not be traded
away for a safety property that doesn't actually require the trade.

---

## 1. Design principle: discover ambiguity, don't declare it — and only one driver has to earn it

An early draft hand-designed gadgets and declared each driver's identifiability status by construction.
That was wrong and unnecessarily restrictive: connecting confounders to each other only removes *one
specific* way an edge could become identifiable, it doesn't prove the edge stays ambiguous — and most real
confounding *is* eventually resolvable from data with enough care, so forcing every confounded driver to
also be ambiguous would claim something false about the world.

**The mechanism** (the fact this whole plan is built on): in a DAG, an edge stays *unoriented* under any
amount of observational data if and only if it survives Meek's four orientation rules applied to the
graph's v-structures (colliders `X→Z←Y` where `X`,`Y` are non-adjacent). If a driver has two *independent*
(non-adjacent) parents, that's an unshielded v-structure and both edges get compelled — ambiguity destroyed
for that driver specifically.

**The point that was missed in an earlier review pass**: this is a fact about *that one driver*, not about
the instance as a whole. The benchmark's only actual requirement is that *at least one* driver in the
instance comes back ambiguous from the exact check (§4) — not that every richly-confounded driver does, and
not that any *specific* driver does. So: guarantee the property cheaply and reliably with one simple,
already-verified building block (§2's `confounded_driver` gadget — single confounder, always included at
least once per instance, §3 step 2), and let every *other* driver be as structurally rich as real causal
triage actually is, with **no restriction on its complexity and no requirement about its outcome**. A
multi-confounder driver that turns out identifiable is a harder, more realistic version of `B` — genuinely
useful. One that turns out to still be ambiguous is a bonus naming-sensitive case. Neither outcome is a
risk, because nothing in the design depends on which way it goes; different drivers don't interact
structurally (they only meet at the shared `Y` sink, which doesn't create v-structures between drivers'
non-adjacent parents).

So: **generate rich structure freely, guarantee the one required property with one cheap, reliable
building block, and discover — via the exact algorithm, never a declared label — what every other driver's
identifiability actually is.** The discovery step (§4) is not a restriction on what gets generated; it's
how the rich, varied structure gets *read*, per instance.

---

## 2. Gadget catalog (`scm/gadgets.py`, new)

Five small, composable edge-fragments. No fragment declares whether it *is* ambiguous — that's discovered
in §4, after composition. `multi_confounder_driver` and `instrumented_driver` are both back in v1 (see §1
for why they were never actually a risk):

| gadget | nodes | edges | purpose |
|---|---|---|---|
| `clean_driver` | `B_i` | `B_i→Y` | naming-invariant control; correct adjustment is always `{}` |
| `confounded_driver` | `C_i, A_i, M_i` | `C_i→A_i, C_i→Y, A_i→M_i, A_i→Y, M_i→Y` | `CANONICAL`'s existing branch, verbatim; the one archetype §3 guarantees at least one instance of, since it's the already-verified-safe building block the required property leans on |
| `multi_confounder_driver` | `C_i, C_i2, A_i, M_i` | as `confounded_driver` plus `C_i2→A_i, C_i2→Y, C_i→C_i2` | richer, realistic confounding (correct adjustment `{C_i, C_i2}`); whether its `C_i`/`C_i2` edges stay ambiguous is discovered, not required either way (§1) |
| `instrumented_driver` | `Z_i, A_i` | `Z_i→A_i` (no `Z_i→Y`, `Z_i` otherwise isolated) | a plausible-looking, scoring-harmless-to-include variable that tests whether the agent substitutes "there's an instrument" for actually finding the real confounder set — the failure CausalDS's own agents showed (`causalDS.pdf` §A.14/Table 26). Caught by the existing `adjustment_accuracy` check exactly like the mediator trap is, no new metric needed — a wrong or empty adjustment set from getting distracted scores as invalid, same mechanism |
| `decoy` | `Dz_i` | `C→Dz_i` where `C` is an *existing* confounder from an already-placed `confounded_driver`/`multi_confounder_driver` | zero true effect, nonzero marginal correlation via the shared confounder — the exact configuration `CANONICAL` already uses and `backlog.md` §4 already verified as safe |

## 3. Composition (`scm/gadgets.py`, `procedural_scaffold(seed, config) -> Scaffold`)

Per seed, deterministically:

1. Draw a driver count `n_drivers ~ Uniform{3, ..., 8}`.
2. **Guarantee at least one `confounded_driver`** — this is the one thing the structural gate (§5) actually
   leans on; everything else is free. Assign it first.
3. For every remaining driver slot, independently draw an archetype from `{clean, confounded,
   multi_confounder, instrumented}` (weights are a `GeneratorConfig` field, not hardcoded — §8; uniform to
   start).
4. Draw a decoy count `n_decoys ~ Uniform{0, 1, 2}`; each decoy attaches to a uniformly-chosen confounder
   from among the placed `confounded_driver`/`multi_confounder_driver` instances (skip decoy placement if
   none exist this instance — "no decoys" is a valid, expected composition).
5. Wire every gadget's terminal edge into one shared outcome `Y` (the anchor — every gadget only touches
   the rest of the graph through `Y`, which is why different drivers never interact structurally,
   regardless of archetype mix).
6. Assemble the `Scaffold`: `node_names`, `roles` (extend `Role` with `INSTRUMENT`), `edges`, `drivers =`
   every `A_i`/`B_i`/`Dz_i` (not confounders, mediators, or instruments), `outcome = Y`.
7. **Build `correct_adjustment` directly while composing, not via a search.** The composition function
   already knows which confounder(s) it wired to which driver at the moment it places each gadget — record
   that mapping as it goes: `{A_i: (C_i,), A_j: (C_j, C_j2), B_i: (), Dz_i: (C,)}`, exactly the shape of
   `CANONICAL`'s own literal dict today, just assembled programmatically instead of typed by hand. No new
   module, no search algorithm; `scoring/score.py::_valid_backdoor` stays exactly as it is, unchanged and
   private, since nothing new needs to call it.

## 4. Exact identifiability discovery (`scm/identifiability.py`, new)

`unoriented_edges(scaffold) -> frozenset[tuple[str, str]]`: build the CPDAG — find all v-structures in
`scaffold.digraph()`, orient those edges, apply Meek's four rules to exhaustion, return every edge from
`scaffold.edges` still undirected in the result. Pure graph theory, no sampling, no weights needed — this
runs on the topology alone, before `scm/generate.py` ever samples UUMC weights. Applies identically to
every driver archetype from §2; this is what determines, per instance, which of the richer drivers (the
`multi_confounder`/`instrumented` ones) landed in which quadrant of §6's table.

**Required regression test before this is trusted for anything**: `unoriented_edges(CANONICAL)` must
return exactly `{(C,A), (A,M), (C,Dz)}` and nothing else — the one fact this whole benchmark already depends on,
previously verified only by hand (one-off 200k-sample regression). This ties the new
automated check to the existing manual finding and is the load-bearing correctness test for this module.
*(Shipped as `test_example_has_exactly_the_known_unoriented_edges`; once `CANONICAL` was deleted per §8a,
the test kept the same graph as a local `Scaffold` literal rather than importing a production constant.)*

## 5. End-to-end generation loop (`scoring/gates.py`, modified) and weight sampling

**`generate()` itself is unmodified** (not the same as "weight sampling is unchanged end to end" — see §5a,
which adds a new post-processing step after it, not a modification to it). `scm/generate.py::generate(seed,
scaffold)` is already fully scaffold-agnostic — it iterates `scaffold.topological_order()` /
`scaffold.parents()` with no `CANONICAL`-specific logic, so it applies unchanged to any
procedurally-composed `Scaffold`, instrument nodes included (zero parents, sampled as an exogenous root
exactly like a confounder is today).

**The loop is flat, not nested**: every attempt in `generate_gated`'s existing `MAX_ATTEMPTS` retry loop
draws a *fresh* topology and fresh weights together. Per attempt, in order:

1. **Structural gate** (topology only, no sampling): draw `candidate = procedural_scaffold(topology_seed,
   config)` (§3); require `unoriented_edges(candidate)` (§4) to contain at least one edge into the
   guaranteed `confounded_driver` (§3 step 2). Fail → discard this attempt, draw a new `topology_seed` next
   attempt. Cheap, runs before any sampling, preserving the existing `gates.py` comment: *"Structural
   gates... are checked first... so most rejections cost no sampling."* Because the gate only depends on the
   one guaranteed simple gadget — not on however many `multi_confounder`/`instrumented` drivers got drawn —
   this should pass close to every time regardless of how rich the rest of the composition is.
2. **Numeric gates** (only reached if step 1 passes): sample weights via `generate(weight_seed, candidate)`,
   then construct driver effects directly on top of that reference SCM (§5a) — this is the step that turned
   out to need real design work, see below — then run the existing four numeric gates, generalized.
   `evaluate_gates()` currently does
   `nodes_with_role(Role.DRIVER_CONFOUNDED)[0]` — hardcoded first-index, silently wrong the instant there's
   more than one. Fix regardless of everything else in this plan: loop over every instance of
   `DRIVER_CONFOUNDED`/`DRIVER_CLEAN`/`DECOY` (and, once §2's richer archetypes are in, every
   `multi_confounder`/`instrumented` instance too), apply the same thresholds (`MIN_ABS_EFFECT`,
   `MIN_EFFECT_GAP`, `CONFOUNDING_MARGIN`, `MIN_MEDIATED_EFFECT`, `ORACLE_TAU_THRESHOLD`) per instance, gate
   passes only if every instance clears its threshold.
3. Accept on both passing; otherwise the next attempt redraws both topology and weights from scratch.

**Measured, not assumed — and the numeric gates turned out badly broken.** The structural gate alone passes
essentially 100% of the time, as expected. But the *combined* (structural + numeric) rate measured only
**2.0%** (6/300), and worse, `structure_valid` collapsed from 23.5% at 3 drivers to **0% at 7+ drivers** — a
systematic bias against exactly the richer instances this plan exists to produce, not just a slow generator.
The root cause, confirmed empirically (`median |effect| × √driver_count` is roughly constant, ~0.34–0.45,
from 3 to 10 drivers): `Y`'s variance is conserved at exactly 1 (UUMC's core guarantee), so as more drivers
compete to explain that same fixed budget, each one's typical effect shrinks as `1/√N`. Loosening gate
thresholds to compensate was considered and rejected - it doesn't fix anything, it just lowers the bar for
what counts as "well-separated," and the same problem resurfaces in `oracle_recovers`'s implicit
zero-discordant-pairs standard for the same underlying reason. The real fix is §5a.

## 5a. Constructing driver effects directly, instead of hoping UUMC's free sampling produces good ones

**Why this is the right fix, not a workaround**: UUMC's unit-variance guarantee is about *marginal
variance*, not about driver effects having to emerge from fully unconstrained random weight draws - that's
just what `generate()` happens to do by default. There is nothing that requires it. A chosen, well-separated
set of driver effects fits comfortably inside `Y`'s variance budget (e.g. 8 drivers spread from 0.10 to
0.45 uses roughly `Σ effect² ≈ 0.7` of the budget of 1, even before the constraint tightens further from
confounders/mediators) - the shrinkage above is an artifact of *how* weights were being chosen, not a hard
limit on what's achievable.

**Construction, in order:**

1. **Generate target effect magnitudes with guaranteed spacing, before any weights are touched.** For `N`
   non-decoy drivers and a chosen range `[EFFECT_MIN, EFFECT_MAX]` with minimum gap `g` (Stage 0 constants,
   same status as `MIN_ABS_EFFECT`/`MIN_EFFECT_GAP` today): draw `N` i.i.d. uniforms over the "compressible"
   leftover range `(EFFECT_MAX − EFFECT_MIN) − (N−1)·g`, sort them, then add back the mandatory gap at each
   step (`target_(i) = EFFECT_MIN + v_(i) + i·g`). This guarantees every consecutive gap is ≥ `g` *by
   construction* - no rejection sampling needed for this piece at all.
2. **Randomly permute which driver gets which target** - shuffle the assignment independent of driver
   index, archetype, or position in the composition. This is not optional: a fixed correspondence (e.g. "the
   guaranteed driver always gets the largest effect") would be exactly the kind of memorizable shortcut
   procedural generation exists to eliminate, just relocated from topology to effect size.
3. **Run `generate()` completely unmodified first**, on the full composed scaffold, to get a valid reference
   SCM. No changes to `scm/generate.py` at all - this corrects an earlier version of this plan, which
   assumed `generate()` itself would need refactoring to defer specific edges; it doesn't, because of the
   next two steps.
4. **Every driver archetype has exactly one direct edge into `Y`** (confirmed in §2's gadget table). Compute
   each driver's *non-direct* contribution to its own total effect straight from the reference SCM:
   `non_direct = total_effect_on_outcome(driver) − W[driver, Y]`. This is generic across every archetype
   with no special-casing: it equals the mediated path's contribution for `confounded_driver`/
   `multi_confounder_driver`, and exactly 0 for `clean_driver`/`instrumented_driver` (they have no other
   path). Then set the new direct weight: `W_new[driver, Y] = target − non_direct`.
5. **Patch only `Y`'s row.** Because `Y` is the terminal sink (no children), changing its incoming weights
   cannot propagate upstream - the parent covariance matrix `Σ_parents` (already computed by UUMC's
   topological accumulation) is completely unaffected. Only `Y`'s coefficient vector and noise variance need
   recomputing: `noise_var(Y)_new = 1 − w_new^T Σ_parents w_new`, using the full updated incoming-coefficient
   vector (confounders/mediators unchanged from the reference draw, drivers now at their target-derived
   values). Recompute `Sigma`'s `Y` row/column from this (`scm/effects.py::marginal_covariance` already does
   this computation) to get the final, valid SCM.
6. **Feasibility check, not an assumption**: if `noise_var(Y)_new < 0`, the chosen targets are infeasible
   against this particular draw's randomly-sampled confounder/mediator structure - a real possibility, not
   provably ruled out by the rough budget estimate above. Treat this as a failed attempt in the *same* flat
   outer loop from §5 (redraw topology and weights fresh next attempt) - no nested retry needed, no special
   casing.

**Consequence worth stating explicitly**: this makes `structure_valid`'s `MIN_ABS_EFFECT`/`MIN_EFFECT_GAP`
checks near-redundant for procedurally-generated instances - separation is now guaranteed at construction
time, not filtered for afterward. They stay as a defensive check (cheap, and still meaningful for `CANONICAL`
via the legacy path), but should trivially pass every time once this lands.

**Verification plan**: two direct checks, not trust in the derivation. (1) After patching, recompute
`total_effect_on_outcome(driver)` for every driver and assert it matches its assigned target to numerical
precision - confirms the solve-for-weight step is exact. (2) Re-run the exact acceptance-rate measurement
from this section (structural + numeric gate pass rate by driver count, 3 through 10) and confirm the
collapse is gone - flat, not declining, acceptance rate across driver counts.

## 6. Naming alignment + abstain (`generation/labeler.py`, `task/answer.py`, modified)

> **Superseded by §11.8.** `_informative_order()` shipped as described below, then was deleted outright -
> the grid it served (the `±0.5` midpoints) was cut, and the endpoints it also handled never needed an
> order in the first place. Kept here for the original reasoning, not as the current implementation.

**Extend the existing global-α mechanism; don't build a new one.** `_informative_order()` (§03/§05) already
generically orders an arbitrary-length node list by "load-bearing importance" (decoys first, strongest
driver, confounder, mediator, remaining drivers) and fills the α-determined top-`k` slots with informative
names. The only change needed: fold every driver whose confounder-edge came back in `unoriented_edges` (§4)
— however many that turns out to be, across whatever archetype mix got composed — into the front of that
priority order, the same way decoys already get priority today. One global α per instance, same six
discrete levels as today, just applied over a variable-length, per-instance node list instead of
`CANONICAL`'s fixed six. No independent per-driver randomization, no new combinatorial naming space — the
2×2 quadrant below gets covered *across the dataset* via the existing α-sweep, the same way today's
per-role numbers in §07 are already produced.

|  | naming aligned (truthfully resolves it) | naming not aligned |
|---|---|---|
| **data-identifiable** | solve (trivial — both agree) | **trust data** — must rank/adjust correctly regardless |
| **data-unidentifiable** | **trust naming** — correct answer requires using the name | **abstain is correct** |

Per explicit instruction (2026-07-23): for v1, "naming not aligned" collapses *both* "no signal" (neutral)
and "actively wrong signal" (misleading) into the same abstain-correct cell. The distinction flagged
earlier — a misleading name on a genuinely unidentifiable edge is not data-refutable, so it's arguably not a
fair test the way a misleading name on an *identifiable* node is — is deferred, not resolved.

**Schema change**: `DriverVerdict` (`task/answer.py`) gains a field — `abstain: bool = False` — alongside
the existing `adjustment_set`. When `abstain=True`, `adjustment_set` and ranking position for that driver
are not graded on content; only whether abstention was the correct call.

## 7. Scoring changes (`scoring/score.py`, modified)

* **Gold rule per driver**: `abstain` is correct iff the driver is unidentifiable (§4) *and* naming doesn't
  correctly resolve it (§6's collapsed rule). Otherwise a committed answer is required, graded exactly as
  today — `is_valid_backdoor` already handles multi-element adjustment sets and instrument-inclusive sets
  correctly with no changes (§3 step 7, §2's `instrumented_driver` row).
* **New metric**: `abstention_accuracy` — fraction of drivers where the agent's abstain/commit choice
  matches the gold rule. Same shape as `adjustment_accuracy`, a new field on `AnswerScore`.
* **Open, proposed-but-not-settled**: how `ranking_tau` treats a correctly-abstained driver. Proposal:
  exclude it from the pairwise Kendall-τ comparison entirely, and let `abstention_accuracy` carry the credit
  for the abstain call itself. Flagged in §10 for confirmation before implementation.

## 8. `GeneratorConfig` additions (`config.py`)

Per the existing comment in `gates.py` ("Thresholds are Stage 0 invariants and will migrate into
`GeneratorConfig` when tunable difficulty arrives") — this is that migration point. New fields: driver-count
range, archetype weights over `{clean, confounded, multi_confounder, instrumented}` for the non-guaranteed
slots, decoy-count range.

## 8a. `generate_gated_procedural` is additive now, not permanently — remove the legacy path once validated

> **Done.** `generate_gated`, `CANONICAL`, and `scm/scaffold.py` are deleted; `generation/generate.py`'s
> `generate_instance`/`generate_instances`/`alpha_sweep` (previously dead code, silently stuck on the
> legacy path) now call `generate_gated_procedural`. Every test that needed a small worked example now
> builds its own local ``Scaffold`` literal instead of importing a shared production constant. 159/159
> tests pass, lint/mypy clean.

`generate_gated(seed, scaffold=CANONICAL)` is being left untouched during implementation deliberately (it's
what `testset_v1` and every existing script currently depend on, and breaking it mid-implementation would
make it impossible to tell a real regression from expected in-progress change). That is a *transitional*
state, not the destination. Once `generate_gated_procedural` is implemented, tested, and validated end to
end (structural gate, numeric gates, naming, scoring, abstention all working against procedurally-composed
instances), the CANONICAL-only path should be **removed, not deprecated-and-kept**: `CANONICAL` as a fixed
scaffold, `generate_gated`'s `scaffold` parameter, and any script that only knows how to call the old path.
Per the project's own convention (no backwards-compatibility shims for code that's certain to be unused) —
explicit instruction, 2026-07-24. Track this as a real follow-up step, not something to quietly keep around
"just in case."

---

## 9. Explicitly deferred / out of scope for v1

* **Independent per-driver naming alignment** — still deferred, but the rationale changed: `_informative_
  order()` (the mechanism this item assumed it would extend) is gone as of §11.8. If this is ever picked
  up, its selection mechanism needs designing fresh, informed by whatever the actual need turns out to be
  - not by resurrecting the deleted priority list.
* Misleading vs. silent naming distinguished on unidentifiable drivers (§6) — collapsed together for now.
* Unobserved confounders — ruled out, not merely deferred (undetectable from data; naming can never
  legitimately resolve it; see chat log 2026-07-23).
* Nonlinear SCMs — Tier C, a separate, later change (§08 of the deck already scopes this out).
* Confounders shared *across different drivers* — a driver's confounder cluster (single or multi) stays
  private to that driver; nothing rules this out mathematically, it's unnecessary composition complexity
  for a first version, not a safety requirement (contrast with §1: the *within-driver* multi-confounder
  case is fully in scope).
* Full dataset versioning/migration plan for `testset_v1` → whatever comes next — this is a schema-breaking
  change (new `abstain` field, variable driver count, new `Role.INSTRUMENT`); needs its own short migration
  note before a build, not specified here.

## 10. Open questions to resolve before/during implementation

1. `ranking_tau` vs. abstained drivers (§7) — confirm the exclude-from-pairwise-comparison proposal, or pick
   an alternative.
2. Archetype weights for the non-guaranteed driver slots (§3 step 3) — uniform across the four is the v1
   default; revisit once real generated instances show whether that produces a reasonable mix (e.g., too
   many/few ambiguous drivers per instance, given `multi_confounder`/`instrumented` outcomes are discovered
   rather than controlled).
3. ~~Structural gate acceptance rate~~ — **measured** (§5): structural alone ~100%, but combined with the
   numeric gates only 2.0%, collapsing to 0% at 7+ drivers. Root-caused to UUMC's variance conservation at
   `Y` and addressed by §5a's direct effect construction. Remaining item: implement §5a and confirm via its
   own verification plan (§5a, last paragraph) that the collapse is actually gone, not just diagnosed.
4. `EFFECT_MIN`, `EFFECT_MAX`, and the minimum gap `g` for target-effect construction (§5a step 1) — need
   concrete Stage 0 defaults; the rough budget check in §5a's intro (8 drivers spread 0.10–0.45 uses ~70% of
   the variance budget) suggests these are comfortably feasible but the exact values aren't chosen yet.

## 11. Correction: identifiability was never actually independent of archetype (v2)

**Status: implemented and verified (§11.5).** Written 2026-07-27, after §1-§10 (and §6/§7's abstain
mechanism) were built and run against a live model. Amends §1-§3; the guarantee mechanism they described
(one forced `confounded_driver`, everything else free) is superseded by §11.3.

### 11.0 What went wrong, measured

A live eval turned up an instance where every single candidate - all three real drivers and both decoys -
required abstention. `abstention_accuracy` and `ranking_tau` on that instance carried almost no signal: with
nothing left un-abstained, there was nothing to rank. Measured across 2000 seeds, this isn't rare: **30% of
composed instances have zero identifiable non-clean drivers**, and in every one of those, literally every
candidate (drivers and decoys) comes back ambiguous.

Root cause, also measured directly (3000 seeds, every driver instance tallied by role):

| role | ambiguous rate |
|---|---|
| `driver_confounded` (covers `confounded_driver`, `multi_confounder_driver`, *and* `instrumented_driver` - they share one `Role`) | 100.0% |
| `driver_clean` | 0.0% |
| `decoy` | 100.0% |

Zero overlap. Despite §1's stated intent ("a multi-confounder driver that turns out identifiable is a
harder, more realistic version of `B`"), that case never actually happens - gadgets are deliberately
structurally isolated from each other (§2: "different drivers don't interact structurally"), so nothing in
the current catalog ever creates the extra structure that would resolve a confounder edge. **Identifiability
is currently a deterministic function of which archetype got drawn, not something discovered per instance
at all** - the "discover, don't declare" principle (§1) was never actually being exercised for the
non-guaranteed drivers; every one of them was fully determined by its archetype the moment it was chosen.

### 11.1 The mechanism that resolves a confounder edge

A confounder's own edge into its driver resolves if the confounder itself becomes part of a v-structure.
Verified directly: give a confounder `C` two new, non-adjacent parents `X1`, `X2` (its own collider) -
`X1->C` and `X2->C` get oriented by ordinary v-structure detection, and Meek rule 1 then propagates that
into forcing `C->A` as well, since `X1` and `A` are non-adjacent. Tested on an isolated `confounded_driver`
gadget: `unoriented_edges` returns `frozenset()` - fully resolved - while `correct_adjustment` for the
driver still requires `{C}` (it's genuinely confounded, just no longer ambiguous about it). This is exactly
the missing quadrant: **data-identifiable, still requires real adjustment** - distinct from `clean_driver`
(identifiable because there's nothing to adjust for at all).

**`multi_confounder_driver`, measured directly rather than derived by hand** (Meek-rule cascades aren't
obvious from inspection): giving *both* `C1` and `C2` their own separate upstream v-structure resolves
everything - `C1->A`, `C2->A`, `C1-C2`, and (same cascade as the single-confounder case) `A->M` too. Giving
*only one* of the two produces a third, distinct outcome: that confounder's edge and the mediator edge
resolve, but the *other* confounder's edge stays unoriented (`C1->C2` being forced doesn't force `C2->A`,
since `C1` is itself adjacent to `A`, so Meek rule 1's non-adjacency condition fails for it). This partial
case is real but not used: `unidentifiable_drivers` only needs one unresolved incoming edge to call a driver
unidentifiable, so a partially- and fully-resolved multi-confounder driver score identically under §7's
binary abstain schema (single adjustment set + single abstain flag per driver, no per-component confidence).
Building independent per-confounder draws would add real complexity for structural variety nothing
downstream can measure or reward - not worth it until the answer schema itself grows to support
component-level confidence, which it doesn't today. **Decision: the upstream-cause modifier is drawn once
per driver, not once per confounder** - for `multi_confounder_driver` specifically, `C1` and `C2` both get
it or neither does, always together.

### 11.2 Design principle: one modifier, applied uniformly, discovered not declared

Identifiability *is* a structural fact - Meek's rules are a pure function of topology, so the outcome can
never be literally independent of structure. What's being fixed is narrower and achievable: the outcome
must not be a **fixed correlate of the declared archetype**, and fixing that must not just relocate the same
problem onto a newly-named "resolved" gadget variant (still "a type of local structure" determining the
outcome, just a different one).

The fix: the upstream-cause structure (§11.1) is not a new archetype, not a new gadget name, and not a
declared "this driver is resolved" label. It is one uniform modifier - add two upstream causes to a
confounder or instrument node, or don't - drawn **randomly and independently per confounder-bearing driver**,
the same way driver count or decoy count already vary randomly per instance. It applies identically whether
the driver is `confounded_driver`, `multi_confounder_driver`, or `instrumented_driver`. Whether the modifier
was applied is never consulted directly; the actual identifiability outcome is always read back off
`unoriented_edges`, exactly as today. Within any one archetype, some instances end up identifiable and some
don't - discovered per instance, not predetermined by which archetype was picked.

Accepted, stated honestly rather than hidden: adding `X1`/`X2` does add two nodes to the graph, a small
addition to a total column count that already varies substantially instance-to-instance for other reasons
(driver count, decoy count, mediators, other confounders). No mechanism can create a v-structure without
adding *some* structure; this is the minimal one that does it.

### 11.3 Guaranteeing the mix: reject on the discovered outcome, not on the draw

> **Superseded by §11.6.** The reject-retry gate below shipped first and worked, but cost a measured 32.4%
> rejection rate. §11.6 replaces it with direct, deterministic construction of the same guarantee -
> described here for the reasoning that motivated it, not as the current implementation.

§3's current guarantee ("driver slot 0 is always exactly `confounded_driver`, forced, no upstream cause")
is replaced, not kept alongside the new mechanism - it was only ever a proxy for "make sure at least one
driver ends up ambiguous," and that's now checked directly and more honestly:

1. Every driver slot (including slot 0 - no more special-cased first slot) draws its archetype freely from
   `{clean, confounded, multi_confounder, instrumented}`, and independently, one coin flip for whether that
   driver's confounder(s)/instrument get the upstream-cause modifier - both draws uniform, both un-declared.
   One flip per *driver*, not per confounder: `multi_confounder_driver`'s `C1` and `C2` both get it or
   neither does (§11.1 - a partial, one-of-two-resolved outcome exists but scores identically to fully
   ambiguous under today's binary abstain schema, so it isn't worth the extra composition complexity).
2. Compose the full topology, then **discover** the mix via the existing `unidentifiable_drivers` (§4,
   already built) - no new discovery machinery needed, just a new check on its output.
3. New topology-gate requirement, checked before any weight is sampled (cheap, same place
   `_has_unresolvable_confounder` already runs): the composed instance is accepted only if it has **at least
   one ambiguous driver** (any archetype) *and* **at least one driver with a non-empty `correct_adjustment`
   that is not ambiguous** (a real, resolvable confounding case - `clean_driver`, having no adjustment set at
   all, doesn't count toward this half). Fails either condition -> reject, redraw the whole topology, exactly
   the same retry loop already in place.

This keeps "discover, don't declare" applied consistently for the first time across the *entire* composition,
not just the previously-guaranteed slot - the guarantee is now a property of the checked outcome, not a
property forced onto one hardcoded slot.

### 11.4 First-class quadrant labels

Add an explicit `Quadrant` classification (four values, matching §6's table: `identifiable_aligned`,
`identifiable_not_aligned`, `unidentifiable_aligned`, `unidentifiable_not_aligned`) as a small function
combining `unidentifiable_drivers` (topology) with a driver's `node_kind` (naming, §6/§7) - the same two
inputs `_gold_abstain` (`scoring/score.py`) already combines ad hoc. `_gold_abstain` becomes a one-line
derivation from this (`quadrant(key, driver) == UNIDENTIFIABLE_NOT_ALIGNED`) instead of its own inline logic
- single source of truth, not two independently-maintained checks of the same underlying fact (the same
principle already applied to `scaled_threshold`/`REFERENCE_MIN_EFFECT` in §5a). Exposed publicly (not
underscore-prefixed) so a future report - a `smoke.py`-style quadrant-coverage breakdown, or a per-instance
column in `run_eval.py`'s output - can show the actual mix directly instead of it staying implicit.

### 11.5 Verification plan

1. ~~Confirm the upstream-cause mechanism resolves what §11.1 claims~~ - **done**, both during planning
   (isolated gadgets) and reproduced in actual composition (`test_add_upstream_cause_resolves_the_
   confounder_edge`, `test_maybe_resolve_produces_both_outcomes_for_a_confounded_driver`).
2. ~~Re-measure the per-role ambiguity table~~ - **done**: `driver_confounded` 49.7%, `decoy` 51.1%,
   `driver_clean` 0.0% (3000 seeds) - genuine splits, not the old 100%/0%.
3. ~~Re-measure the "zero identifiable non-clean drivers" rate~~ - **done**: 0/100 accepted instances
   (was 30%) - measured on `generate_gated_procedural`'s actual output, not assumed from the gate's logic.
4. ~~Full regression~~ - **done**: 185/185 tests pass (up from 177 pre-§11 - new tests added for the
   resolving mechanism, the gate, and `Quadrant`), lint/mypy clean. A handful of pre-existing tests turned
   out to assert things only true by coincidence of the old fixed RNG sequence (e.g. exact-match oracle
   ranking, when the gate only ever guaranteed `tau >= 0.99`; "correlation always fooled by a decoy," when
   decoys were never actually guaranteed present) - fixed to assert what's actually guaranteed, not
   re-seeded around the failure.
5. ~~Live check~~ - **done**: combined acceptance 23.5% overall with no collapse at high driver count
   (13.7% at n=8, same order as pre-§11's 12-15%) - the new gate costs real acceptance rate (down from
   39.3%) but doesn't re-introduce the collapse §5a fixed. A real `build_testset.py` run against live LLM
   naming calls was also started to confirm no error surfaces end-to-end; see the session record for its
   outcome.

### 11.6 Making two of the four rejection sources deterministic

**Status: implemented and verified.** `generate_gated_procedural`'s retry loop has four distinct rejection
sources. Measured directly (2000 draws, post-§11): `topology_reject` (§11.3's gate) 32.4%,
`confounding_construct_infeasible` 24.0%, `mediation_construct_infeasible` 4.7%, `oracle_fail` 16.9%
(irreducible - depends on the sampled data, not construction) - `structure_fail`/`confounding_fail`/
`mediation_fail` all 0% (fully covered by §5a's construction already). Two of the four are themselves
*constructible*, not fundamentally random - fixed here.

**Fix 1 - `topology_reject` -> deterministic construction.** §11.3's gate checked a property after the
fact and retried on failure. Replaced with direct construction in `procedural_scaffold`
(`scm/gadgets.py`): two driver slots are reserved via `rng.choice(n_drivers, size=2, replace=False)` -
one forced resolved-but-confounded (`_force_resolve`, archetype drawn only from `{confounded,
multi_confounder}` since `instrumented`'s empty adjustment set wouldn't count as "resolved"), one forced
ambiguous (left untouched, archetype drawn from `{confounded, multi_confounder, instrumented}`). Every
other slot stays exactly as free as before. `n_drivers` now has a floor of 2 (one slot per guarantee) -
already implicitly required by the old gate, just not previously validated explicitly. Verified 200/200
random seeds always balanced; `topology_reject` measures 0.0% post-fix.

**Fix 2 - `confounding_construct_infeasible` -> comfortable constructed starting value.** The
confounder->outcome edge was the one incoming-to-outcome edge §5a's construction left untouched - raw
UUMC, then only ever shrunk (never boosted) by the shrink ladder. A confounder can share the outcome with
several co-parents (drivers, mediators, other confounders), so its raw UUMC coefficient could land
arbitrarily close to 0, leaving `with_constructed_confounding` no real correlation to ever beat the decoy
margin with. Fixed the same way mediator edges already are (`_construct_confounder_edge`,
`scm/target_effects.py`): draw a comfortable, non-tiny magnitude directly instead of trusting UUMC's raw
draw. Unlike the mediator edge, it is *not* protected from the shrink ladder afterward - it remains the
ladder's only lever for driver-target feasibility, so construction only fixes the starting point, not the
final value.

The range (`_CONFOUNDER_EDGE_RANGE`) was tuned by direct measurement, not guessed: narrower ranges (e.g.
(0.1, 0.25)) made `confounding_construct_infeasible` *worse*, not better - a smaller starting value leaves
less margin after the shrink ladder backs it off for a driver-target solve. `(0.3, 0.6)` (close to
`_MEDIATOR_EDGE_RANGE`) was the measured sweet spot; widening further keeps helping only marginally while
nudging `mediation_construct_infeasible` up.

**Net effect, measured (5000 draws, both fixes applied, `topology_reject` no longer exists as a
category):** `mediation_construct_infeasible` 8.0%, `confounding_construct_infeasible` 27.4%, `oracle_fail`
23.1%, **passed 41.6%** (up from 23.5% pre-§11.6). Important nuance for reading this against the §11.5
numbers above: Fix 1 alone (isolated by disabling Fix 2 in the same measurement) already moves
`confounding_construct_infeasible` up to 32.7% - guaranteeing a confounded, resolved driver on every
instance means confounder-bearing drivers are now the norm rather than one possibility among several, so
more instances compete harder for the outcome's unit-variance budget than before §11.3 existed at all. Fix
2 recovers some of that (32.7% -> 27.4%), but doesn't erase it - the residual is a genuine capacity
constraint (many confounded drivers simultaneously, not a near-zero-value artifact), not something a
starting-value tweak alone can fully absorb. `topology_reject`'s full elimination is what actually drives
the net acceptance gain from 23.5% to 41.6%. 185/185 tests pass, lint/mypy clean; a handful of tests
hardcoding specific seeds needed new seeds re-verified against the new RNG sequence (same pattern as
§11.5's regression note - fixed to assert the same guaranteed property under a valid seed, not re-derived
from coincidence).

### 11.7 `resolve_probability`: the one config knob worth exposing now

**Status: implemented.** Prompted by a design review of the generation pipeline: which axes should be
randomized by default vs. exposed as parameters, for the widest, simplest adoption? Landed on: random by
default everywhere (seed-driven diversity is the whole point, matching CANONICAL's known flaw), with an
explicit override only for axes that answer a real research question. `n_drivers` already qualified. Decoy
count and archetype mix don't yet - no concrete downstream question needs them pinned, so adding knobs for
them now would be speculative, not minimal. The ambiguous/resolved mix does qualify (it's the benchmark's
own core question: does abstention hold up as ambiguity gets more/less prevalent), and the mechanism for it
already existed as an unparameterized module constant (`gadgets.py`'s `RESOLVE_PROBABILITY = 0.5`, the coin
flip for every free, non-guaranteed slot).

The minimal change: thread that existing constant through as an optional parameter -
`procedural_scaffold(seed, n_drivers=None, resolve_probability=0.5)`, propagated to
`generate_gated_procedural` and `generation/generate.py`'s `generate_instance`/`generate_instances`/
`alpha_sweep` (mirroring exactly how `n_drivers` is already threaded through the same call chain). No new
dataclass, no new enum, no new field on `GeneratorConfig` - one float, one natural unit, the same default
that already existed. `0.0`/`1.0` push every free slot to the same extreme (mostly ambiguous / mostly
resolved); the §11.3 guarantee (at least one of each) holds at every value, since the two reserved slots are
always forced regardless of `resolve_probability`. Verified directly: at `0.0`, exactly one resolved-
confounded driver survives across 20 seeds (never more); at `1.0`, exactly one ambiguous driver survives
(never more) - both at the raw-composition level and through the full gate/retry loop. 163/163 tests pass,
lint/mypy clean.

### 11.8 Collapsing the alignment grid to three uniform levels

**Status: implemented.** A second design-review question, same session: the alignment grid was
`{-1,-0.5,0,+0.5,+1}`, with `_informative_order` picking a hand-tuned, partially-ordered subset of nodes
to flip at the two midpoints (`±0.5`) - decoys, then unidentifiable drivers, then the single strongest
driver, then confounders, then mediators, then the rest. That ordering was never well-justified as one
list: "quadrant coverage" (why decoys/unidentifiable drivers must go first - they're the only way to
populate the `UNIDENTIFIABLE_*` cells of `scoring/score.py`'s naming-vs-identifiability quadrant) and
"score sensitivity" (why the strongest driver, if included, should out-prioritize a weaker one) are two
different justifications glued into one queue, and confounders/mediators were in there for coverage
bookkeeping with no clear principle at all.

The fix removes the question rather than re-answering it: drop the two midpoints, keep only
`{-1, 0, +1}`, each applied **uniformly** - every non-outcome node takes the same kind (aligned /
neutral / misaligned) at a given level. This was already exactly what the grid's endpoints did (at
`level=±1.0`, `round(|level|*len(order))` was already every node, regardless of order) - so nothing about
the *generated instances* changes at the levels that survive, only the two midpoints are gone and the
now-unreachable ordering logic (`_informative_order`) is deleted outright, not kept unused. Net effect:
`construct_labeling` no longer needs `unidentifiable_drivers`, `total_effect_on_outcome`, or any per-node
ordering at all - just a sign check.

Trade-off, made deliberately not by oversight: this gives up the dose-response middle of the curve (does
degradation happen gradually or as a cliff) in exchange for a clean, directly-comparable per-quadrant
result - aligned vs. neutral vs. misaligned, same SCM, every node moving together. `GRID` shrank from 5
values to 3; `neutral` and `blind` (`abstract_labels`, still no LLM/CauseNet at all) stay two separate
conditions - one carries a coherent-but-inert domain flavor, the other carries none - so a full sweep is
now 4 views per SCM (aligned/neutral/misaligned + blind), down from 6. 162/162 tests pass, lint/mypy clean.

## 12. Two validity bugs surfaced by external review, both confirmed and fixed

**Status: implemented.** An external methodology review flagged six potential issues; two were verified
directly (not just trusted) as real, and fixed. Both mattered because they penalized *correct* reasoning,
not just added noise - exactly backwards from what a benchmark should do.

### 12.1 Instrument-archetype false-unidentifiability

`unidentifiable_drivers()` (`scm/identifiability.py`) flagged a driver as unidentifiable whenever it had
*any* unoriented incoming edge - conflating edge-orientation ambiguity with effect/adjustment
identifiability, which are not the same thing. Verified directly: for the `instrumented` archetype
(`Z -> A -> Y`, empty adjustment set), across 200 composed seeds, 90 instrument edges were left ambiguous
and *all 90* left their driver flagged unidentifiable - yet the driver's total effect and its (always-
empty) adjustment set are mathematically invariant to `Z-A`'s orientation, because an instrument has no
path to the outcome except through its own driver (unlike a confounder, which always has a direct edge
into the outcome, so its orientation *does* determine whether it must be adjusted for). The effect is
identified either way; the driver was never actually ambiguous.

Fix: `unidentifiable_drivers()` now only counts ambiguity from a `Role.CONFOUNDER` parent, not a
`Role.INSTRUMENT` parent - a one-line role-based filter, not a general CPDAG/generalized-adjustment
engine, since the current gadget catalog only has one archetype (`instrumented`) where this distinction
applies. Ripple effect caught and fixed: `gadgets.py`'s `_AMBIGUITY_CAPABLE_ARCHETYPES` included
`instrumented` for the forced-ambiguous slot (§11.3) - after this fix, forcing an instrumented driver
"ambiguous" no longer produces a genuinely unidentifiable driver, so it stopped fulfilling the §11.3
guarantee (measured: balance dropped to 186/200). `_AMBIGUITY_CAPABLE_ARCHETYPES` is now identical to
`_RESOLVABLE_CONFOUNDED_ARCHETYPES` and defined as an alias of it, not a separately-maintained tuple that
happens to match - the two properties ("can be forced ambiguous," "can be forced resolved") turn out to
be the same underlying question ("does this archetype have a confounder edge") once actually correct.

Verified after both fixes: 0/90 instrument-ambiguous drivers wrongly flagged (200-seed audit); 759/759
confounder-ambiguous drivers still correctly flagged (no regression); §11.3 balance restored to 200/200.

### 12.2 Gate/release sample mismatch

`evaluate_gates` validated the oracle gate (`ORACLE_TAU_THRESHOLD = 0.99`) against `GeneratorConfig`'s
`n_samples` (2000 by default), then discarded that sample - every consumer that needed a data table
(`scripts/build_testset.py`, `generation/generate.py`'s `generate_instance`/`alpha_sweep_with`,
`scripts/run_eval.py`'s no-`--dataset` path) drew an independent one, at a different size (1000, in
`build_testset.py`) *and* a different seed. Measured directly: on the actual released 1000-row tables,
even the **oracle** (correct adjustment sets) recovered the certified `tau >= 0.99` ranking only 55/100 of
the time overall, falling to 0/3 at `n_drivers=8`. Matching the sample size alone wasn't sufficient either
(70/100 at 2000 rows, still 1/3 at `n_drivers=8`) - a fresh draw, even at the right size, isn't guaranteed
to reproduce the specific pass the gate already certified; gate acceptance is tied to one draw, not a
robust property of the SCM across every draw of that size.

Fix: `GateReport` gained a `data: FloatArray | None` field - the exact table the oracle gate was already
validated against, populated for free (`evaluate_gates` already samples it internally to run the oracle
check; it was just being thrown away). Every consumer that needs a data table now defaults to
`report.data` instead of redrawing - `n_samples` parameters became `int | None = None` throughout
(`None` = reuse the certified table; an explicit value = deliberately opt into a fresh, unguaranteed
draw). This closes the gap by construction, not by narrowing the odds: the released table *is* the
validated table, so there are no longer two independently-drawn samples that can disagree.

Verified after the fix: oracle recovers `tau >= 0.99` on `report.data` in 100/100 real
`generate_gated_procedural()` instances - tautologically guaranteed now, not just empirically likely.

176/176 tests pass, lint/mypy clean; smoke report's overall acceptance moved from 38.6% to 35.4% (a real,
modest tightening - fewer instances were coasting through on a false-unidentifiability free pass).
