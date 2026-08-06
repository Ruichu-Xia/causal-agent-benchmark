# Research and Environment Backlog

Active design work for making CausalAgentBench a valid research benchmark and a useful RLVR environment.
This is intentionally a prioritized task backlog, not a record of superseded designs. The current naming
grid is constructed as `{-1, 0, +1}` plus a name-blind control; there is no active continuous-`alpha`
design.

## P0: Define a coherent task and gold standard

### 1-2. Information model and outcome metrics — RESOLVED (2026-08-06)

**Chosen: causal tool use *with* external knowledge.** Names are an explicit evidence source, and the
prompt now says so — that they come from a domain source of unknown reliability, that the data settles most
of the structure, and that names consistent with what was recovered can be relied on where it did not. The
old defect was never the four cells; it was that the prompt said "abstain when the data cannot resolve it"
while the gold expected the name to be used, so a model that abstained followed the instructions and was
marked wrong. `_gold_abstain` still reads `quadrant`'s one cell; the instruction now matches it.

**Metrics rebuilt around it.** `mean_driver_score` is the headline: one score per candidate, graded on
*every* candidate, so the denominator is never the agent's to choose — which was the real complaint behind
item 2 (a blanket-abstainer was graded on 4.2 drivers against 7.0 for one that always committed). Ranking
is derived from the effect estimates rather than stated separately, so it cannot be earned by ordering on
raw correlation. Abstention is reported as sensitivity and specificity, because one pooled accuracy has a
0.605 floor from class balance alone. `reference_scores` recomputes fixed-policy oracles per set, including
the 0.750 ceiling for any policy that never decides whether to trust the names.

**Relevant code:** `task/instance.py::build_prompt`, `scoring/score.py`, `agent/eval.py`.

### 3. Establish finite-sample recoverability, not only oracle graph identifiability

**Problem.** The identifiability label is computed from the hidden DAG's exact CPDAG. The gate only checks
that an oracle supplied with the hidden correct adjustment set recovers the ranking on the sampled table.
Neither fact shows that the required structure/effect is recoverable by a specified method from the observed
finite sample.

**Required outcome.** State the assumptions (causal sufficiency, linearity, Gaussianity, faithfulness or a
finite-sample substitute), then calibrate the generated distribution with repeated samples and causal
discovery/effect-estimation baselines. Reject or separately label cases that are structurally identifiable
but not recoverable at the published sample size.

**Relevant code:** `scm/identifiability.py`, `scoring/gates.py`, `scm/sample.py`.

## P0: Validate the intervention and publish a reproducible artifact

### 4. Validate semantic labels independently and control text leakage

**Problem.** The label generator and optional critic receive the hidden graph, roles, and effects. The
description prohibition is a prompt instruction, not a verifier. The all-at-once naming manipulation also
changes domain coherence and language style, not just causal prior.

**Required outcome.** Freeze generated label cards before evaluation; retain provenance and rejection logs;
independently blind-rate alignment, misleadingness, continuity, and relationship leakage. Use a generator
separate from evaluated models. Add lexical/text controls and counterfactual swaps that isolate the intended
semantic variable. Treat descriptions as an explicit design choice, including the option to remove them.

**Relevant code:** `generation/labeler.py`, `task/labeling.py`.

### 5. Release a frozen, complete test set

**Problem.** `outputs/` is ignored and no fixed test set is versioned. The builder silently skips an SCM
when LLM naming fails, so the retained seed population is selected by an unreported process. Existing
experiment summaries are pilots from earlier data-generation versions and their raw data is not committed.

**Required outcome.** Publish a non-empty frozen dataset, manifest, checksums, build configuration, exact
generator/critic model identifiers, CauseNet source snapshot/license provenance, accepted and rejected seed
logs, and raw evaluation records. Fail a release build if its prespecified quota or balance is not met;
never silently publish a partial set.

**Relevant code:** `scripts/build_testset.py`, `configs/testset_v1.yaml`, `task/dataset.py`.

## P1: Make experiments falsifiable and comparable

### 6. Add calibrated baselines and adversarial shortcut tests

**Partly done.** A closed-form blind-policy panel now gates every instance (`SHORTCUT_ERROR_THRESHOLD`),
and `reference_scores` reports fixed-policy oracles per set. Still missing: a causal-discovery baseline
(PC/GES) run end-to-end as an agent, and the R-squared exploitability test.

**Problem.** Current baselines are oracle, raw correlation, and adjust-everything. They establish useful
construction checks but do not calibrate an agent against causal discovery or against simple exploitable
statistics.

**Required outcome.** Add and report:

- causal-discovery/effect baselines appropriate to the declared assumptions (at least PC/constraint-based
  and a score-based alternative, with documented hyperparameters);
- semantic-only, data-only/name-blind, and tool-ablation agent baselines;
- metadata/text-only and simple-statistic baselines, including the observed R-squared/root-status signal;
- oracle ceilings that are clearly labeled as access to hidden gold adjustments, not a discoverability
  baseline.

R-squared sortability is not itself a defect: root/non-root signal is legitimate causal information. Its
task-level exploitability, including the remaining parameterization artifact, is unmeasured and must be
tested rather than declared closed.

### 7. Define the experimental unit and statistical protocol

**Partly done.** Task size is now fixed: `n_candidates=7` counts drivers *and* decoys, with decoys carved
out of that budget rather than added on top, so candidate count no longer varies. Because every instance
has exactly 7, pooling over drivers and averaging per-instance means give the same number. Still missing:
the paired bootstrap over SCMs and reported intervals.

**Problem.** Naming skins of one SCM are paired observations, not independent samples, and the calibration
signal is one bit per instance (naming is uniform across a skin). Current summaries are point estimates
without uncertainty.

**Required outcome.** Pre-specify source-SCM as the primary unit, use paired/block bootstrap or permutation
tests over SCMs, report confidence intervals and all denominators, and power the number of independent SCMs
and independent naming draws for the target interaction. Either fix total candidate count or explicitly
stratify/standardize every affected metric.

**Relevant code:** `scm/gadgets.py`, `agent/eval.py`, `configs/testset_v1.yaml`.

### 8. Expand beyond the small gadget grammar before making broad claims

**Problem.** The current generator composes a small catalog of clean, confounded, multi-confounder,
instrumented, and decoy motifs under linear-Gaussian causal sufficiency. Procedural variation prevents one
fixed-graph lookup, but agents can still learn family-specific strategies.

**Required outcome.** Maintain explicit train/validation/test graph-family splits; hold out motif
compositions and nuisance configurations; measure out-of-family performance. Scope v1 claims to this
well-controlled linear-Gaussian family until stronger evidence exists. Mixed/discrete node families are a
separate future verifier problem, not a current paper prerequisite.

## P1: RLVR environment readiness

### 9. Make resource use part of the environment contract

**Problem.** `max_turns` limits model calls but not tool calls: a model can batch arbitrarily many
correlations/regressions in one turn. That makes training and cross-model comparisons sensitive to API
batching rather than only causal capability.

**Required outcome.** Define a total query budget, per-tool costs, context/token limits, and termination
rules. Record them in every trajectory. Evaluate capability under one or more declared budgets, rather than
implicitly granting unrestricted statistical search.

**Relevant code:** `agent/loop.py`, `agent/tools.py`, `scripts/run_eval.py`.

### 10. Design an RL-safe reward and split protocol

**Problem.** The SCM verifier is fast and exact, making this promising for RLVR, but the current semantic
gold rule would train a policy to exploit an evaluator-only naming convention. A single scalar score can also
hide trading off abstention, ranking, and adjustment failures.

**Required outcome.** Start with the data-only environment after item 1 is resolved. Use deterministic
per-driver verifier components for answer format, valid adjustment, effect/ranking, and calibrated
abstention; make any scalar reward and query penalty explicit. Train on procedural seeds and evaluate on
held-out graph families and frozen label cards. Keep semantic generalization as a separate validated track
until its evidence model is formalized.

## Deferred Scope

### 11. Add non-Gaussian or mixed-type SCM families only with a matching verifier

The current continuous linear-Gaussian family is a deliberate v1 boundary, not an accidental omission.
Binary treatments, discrete covariates, nonlinear links, hidden confounding, and real datasets would improve
external validity, but each changes the estimand and/or removes the present exact closed-form verifier.
Add them as separately validated environment families after the core task, baselines, and release process
are sound.
