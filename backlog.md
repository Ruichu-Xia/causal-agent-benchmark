# Design backlog

Notes on parts of the system that work today but may need rethinking. Each item records the current
state, why it might need to change, and options — so a future redesign starts with context, not a blank
page. Not a task tracker; just design memory.

---

## 1. How `alpha` (naming alignment) is set by users

**Current state.** `alpha ∈ [-1, 1]` is drawn `~U(-1, 1)` per instance by default
(`LLMLabeler.alpha = None`), giving a continuous sweep for the benchmark and training. It is *pinnable*
to a fixed float only for ablations (`llm_labeler(..., alpha=)`, `alpha_sweep(...)`). It is deliberately
**not** a config field, and the labeler seam (`Callable[[SCM, rng], Labeling]`) forces `alpha` to live
either on the labeler instance or be drawn from `rng` inside `__call__` — we chose the latter for the
default and a field for the pinned case.

**Why it may need rethinking.**
- The `alpha: float | None` overload (None = draw, float = pin) is a small wart — one field with two
  meanings. A future API might separate "sampling policy" from "labeling given alpha" more explicitly.
- Users may want richer control than uniform draw vs single fixed value: a **schedule/curriculum**
  (alpha as a function of training step), a **restricted range** or **non-uniform distribution**, or a
  **stratified grid** for evaluation — none of which the current single-field design expresses.
- The seam constraint (alpha not a call argument) is what pushed alpha onto the labeler. If the seam
  ever gains an explicit per-instance parameter object, alpha-sampling could move to the generation
  layer (a first-class `AlphaPolicy`), which would be cleaner and more extensible.
- Continuous per-instance alpha means paired skins (same SCM, multiple alpha) only exist via the
  dedicated `alpha_sweep` helper; if paired data becomes central, it may deserve a more prominent place.

**Options to consider later.** (a) An `AlphaPolicy` abstraction (`uniform`, `fixed`, `grid`, `schedule`,
`distribution`) passed to the generation functions; (b) widen the labeler seam to carry a per-instance
context (alpha, target domain, …) instead of baking it on the instance; (c) keep the current design and
only add helpers as concrete needs appear. Prefer the minimal change that a real use case demands —
don't build the policy zoo speculatively.

**Coverage finding (2026-07-17 verification run).** The generator is alpha-blind (generate diverse ->
judge -> select nearest), so the *realized* alpha often differs from the *target* alpha, sometimes
drastically: an instance targeting +0.69 realized -0.50 because the 6 candidates contained no
well-aligned naming for that graph/domain. An independent audit confirmed the judge's realized scores
ARE faithful (aligned namings score high, decoy-trap namings score negative) — so the realized-alpha axis
is trustworthy and is the right analysis coordinate; but per-instance alpha is *measured, not
controlled*, and axis coverage can be uneven/thin in places. Levers if coverage matters: raise
`n_candidates` (more shots per instance), oversample-and-bin on realized alpha to fill the axis, or (last
resort, rejected so far) make the generator alpha-aware. Decide once a real analysis needs a particular
alpha density.

**Update (2026-07-18): largely resolved by re-spec.** The continuous judge-measured α (uncontrolled,
"measured not set") was replaced by a **discrete constructed** grid `{-1,-0.5,0,+0.5,+1}` via the
three-pole mixture (see `semantic_labeler_plan.md`) — α is now exact and controllable by construction,
and the coverage/gap problem is gone. Remaining future item: *finer* grids toward a smooth continuous
curve (smaller mixture steps) if a dense dose-response is wanted; and seeded intra-level subset diversity.

**Update (2026-07-27): grid collapsed further, to `{-1,0,+1}`.** The two midpoints needed
`_informative_order` - a hand-tuned per-node priority list - to pick a partial subset to flip, and that
list conflated two different justifications (quadrant coverage vs. score sensitivity) with no clean
single principle. The endpoints never needed an order at all (every node already flipped there), so
dropping the midpoints removed the ordering question outright rather than re-solving it; `_informative_order`
is deleted, not kept unused. Trade-off made deliberately: gives up the dose-response middle of the curve
for a directly comparable, per-quadrant result (aligned vs. neutral vs. misaligned, same SCM, every node
moving together). See `procedural_scaffold_plan.md` §11.8.

**Related:** `src/causalagentbench/generation/labeler.py` (`LLMLabeler`, `llm_labeler`),
`generation/generate.py` (`alpha_sweep`), `semantic_labeler_plan.md` §2/§3.

---

## 2. Column-description content is only softly constrained

**Current state.** The generator is *instructed* (prompt only) that each column description should state
what the variable **measures** (a data dictionary — "LDL cholesterol, mg/dL"), not narrate the causal
relationships between variables. There is no hard check. We deliberately removed the role/leakage audit
(see `semantic_labeler_plan.md` §3): the gold is label-invariant, so nothing "leaks the answer," and a
substring guard was both false-positive-prone and conceptually wrong (roles are relational).

**Why it may need rethinking.** If a description *narrates structure* ("age, which affects both
cholesterol and heart risk"), that instance's task shifts from "infer the structure" toward "read it off
the prose" — muddying what the naming axis measures. Today only the prompt discourages this; nothing
verifies it.

**Options to consider later.** (a) Leave as a prompt nudge and spot-check during human review of a built
set; (b) if descriptions are found narrating structure, add a *lightweight* judge check on description
content specifically (not a role blocklist); (c) drop per-column descriptions entirely if they add
leakage risk without enough task value. Decide based on what real generated descriptions look like — not
speculatively.

**Related:** `generation/labeler.py` (`_generator_prompt`), `semantic_labeler_plan.md` §3.

**Update (2026-07-18): neutral-name continuity tension.** Live gpt-5 runs skip ~1/3 of SCMs because the
generator's *neutral* (administrative) names are often inherently discrete (counts, ids, page numbers)
while the data is continuous & standardized - the pre-check then rejects the description as
non-continuous, and revisions don't always fix it. Refinement: steer neutral names to
continuous-compatible admin variables (`enrollment_day_offset`, `device_calibration_offset`,
`operator_tenure_days`) rather than counts/ids, to cut the skip rate. Low priority (the skip gate works;
it just wastes some SCMs).

---

## 3. Discrete / mixed-type data (currently all continuous-Gaussian)

**Current state.** The SCM is linear-Gaussian, so every column is a continuous, standardized variable.
As of 2026-07-17 the generator is constrained (prompt) to describe only continuous real-valued variables
(Tier A), because a data-blind generator was writing "binary (0/1)" / "yes-no" / "count" descriptions that
contradicted the continuous data — an agent inspecting the table would see the mismatch. So descriptions
now match the data, but the data is still always continuous.

**Why it may need rethinking.** Real causal analysis is full of **binary treatments, binary/count
outcomes, and categorical covariates**. Testing agents only on continuous data undercovers the space.
Diagnosis of a generated batch confirmed the instances are otherwise sound (correct backdoor adjustment
recovers τ and the ranking; naive correlation and adjust-for-everything both fail as designed).

**Options (tiers), trading realism against the exact closed-form verifier.**
- **Tier B (cheap, keeps exact τ):** in a linear SEM, total effects depend only on `W`, not on the noise
  distribution — so non-Gaussian / discrete **noise** (counts, skew) and **binary exogenous roots** (a
  binary *treatment*) can be added while `τ = (I−W)⁻¹−I` stays exact. Covers the common "binary treatment"
  case. Needs generator/sampler work + care with the standardization/"standardized effect" interpretation.
- **Tier C (large, breaks the closed form):** genuine categorical/binary **nodes** including binary
  *outcomes* need link functions (logistic/threshold); effects become estimand-dependent and require
  Monte-Carlo / do-calculus estimation — i.e. giving up the exact single-artifact verifier that is the
  project's differentiator. This is a roadmap-level change, likely a separate scaffold/verifier family.

**Recommendation captured:** do Tier B before Tier C, and only if evaluation coverage demands it; never
trade away the exact verifier for v1. Decide against real agent-eval needs, not speculatively.

**Related:** `scm/generate.py`, `scm/sample.py`, `scm/effects.py` (`τ`), `types.py` (`SCM`),
`generation/labeler.py` (`_generator_prompt`).

---

## 4. The identifiability floor: a feature, and a constraint on how results may be read

**Current state.** Every instance contains at least one driver whose incoming confounder edge is genuinely
**unoriented from observational data alone** (no v-structure/Meek rule forces it) — guaranteed by
construction, not by luck: `procedural_scaffold` reserves one driver slot as forced-ambiguous and one as
forced-resolved (`procedural_scaffold_plan.md` §11.3/§11.6), and the outcome is read back per instance off
`unoriented_edges`, never declared.

**This is a feature, not a bug** — it is exactly what makes the naming axis load-bearing. If data alone
resolved everything, names would be decorative and the benchmark would collapse to "just run PC". But the
consequence for reporting is permanent and must be stated in any write-up:

**Interpretation floor.** At neutral/name-blind alignment there is an honest information-theoretic ceiling
below 1.0 on the ambiguous drivers' `adjustment_accuracy` — no amount of good reasoning from data alone can
beat it. This is expected, not evidence of poor reasoning; the informative signal is the **slope**
aligned-to-misaligned, not the absolute score at neutral. Omit this and a flat-looking dip at neutral reads
as a reasoning failure when it is the design working. `abstention_accuracy` is the metric that credits the
correct behavior in that regime (abstaining), and `ranking_tau` excludes correctly-abstained drivers, so a
low neutral score is not double-counted against the agent — but the floor on adjustment is real.

**Resolved (2026-07-28): the structural-position memorization risk.** Earlier revisions of this item
recorded a second consequence — `CANONICAL` being one fixed scaffold meant every node's degree/adjacency
signature was a naming-independent fingerprint, learnable as benchmark trivia by an agent with
cross-instance memory (an RL loop, or one context spanning several instances). That no longer applies:
`CANONICAL` and `scm/scaffold.py` are deleted, and every instance now composes its own topology with
variable driver count, archetype mix, and decoy count. The risk was the concrete motivation for the
procedural generator; it is closed, not deferred.

**Related:** `scm/gadgets.py` (`procedural_scaffold`), `scm/identifiability.py` (`unoriented_edges`,
`unidentifiable_drivers`), `scoring/score.py` (`quadrant`, `abstention_accuracy`),
`generation/labeler.py` (the α grid), `procedural_scaffold_plan.md` §11.
