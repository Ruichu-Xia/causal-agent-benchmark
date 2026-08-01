# CausalAgentBench — Outcome Variance Budget Plan (R1 / R1b)

> **Status: implemented and measured (§9).** Scoped to the two fixed-magnitude thresholds that compete for
> the outcome's unit-variance budget. A third instance of the same bug pattern (§6) is diagnosed and measured
> here but deliberately *not* fixed by this plan — it is coupled to sample size and needs its own decision,
> and it is now the binding constraint at high driver counts.
> Builds on `procedural_scaffold_plan.md` §5a (direct construction) and §12 (the two validity fixes).

---

## 0. Symptom

Gate acceptance collapses as driver count rises, and hits a cliff — not a slope — around n=9–10:

| n_drivers | 3 | 4 | 5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| accepted | 69.6% | 47.8% | 42.0% | 26.1% | 17.6% | 11.2% | **0.3%** |

At `n_drivers=10` (300 candidates), the dominant rejection is **mediation-construct infeasible: 81.0%**
(confounding 10.3%, oracle-recovery 8.3%). This blocks raising the driver-count floor, which is what makes
`ranking_tau` coarse at small n — a 3-candidate instance minus one correct abstention leaves `n_compared=2`,
where tau is a coin flip (+1 or −1, P=0.5 each) and carries almost no signal.

## 1. Diagnosis: one bug pattern, three instances

Every quantity that consumes the outcome's incoming variance must scale the same way the achievable effects
do — **1/√n** (`scaled_threshold`, the variance-conservation law already documented in §5). Any threshold
held at a *fixed absolute magnitude* therefore grows without bound as a share of the budget, and eventually
consumes all of it. Three thresholds violate this; two are in scope here.

`scm/target_effects.py` currently documents the fixed choice deliberately:

> "Unlike the driver-effect thresholds, these do NOT scale with driver count — construction carves out a
> comfortable, fixed-size mediated effect and confounding margin *before* drivers compete for the outcome's
> remaining budget, so neither one is subject to the `1/sqrt(N)` squeeze in the first place."

The reasoning is where it breaks: being carved out **first** does not make a demand **free**. It still
consumes the same unit budget, and it consumes a share that grows with n while everything else shrinks.

### Measured — mediator edges alone, before drivers or confounders take any share

`w[M→Y] = mediated_target / w[A→M]`, with `mediated_target` fixed at `U(1.2, 2.0) × 0.10` and
`w[A→M] ~ U(0.3, 0.7)`. One such edge per mediated driver, and these are **explicitly protected from the
shrink ladder**:

| n_drivers | mediated drivers | Σ w[M→Y]² (budget = 1.0) | draws already over budget |
|---|---|---|---|
| 3 | 2.5 | 0.30 | 0% |
| 6 | 4.2 | 0.51 | 3% |
| 8 | 5.0 | 0.59 | 5% |
| 10 | 6.2 | **0.75** | **17%** |

## 2. R1 — mediated effect becomes a share of the driver's own effect

**Change:** `mediated_target = ρ · target_effect(driver)`, `ρ ~ U(0.35, 0.60)`, taking the target's sign.
Replaces `MIN_MEDIATED_EFFECT = 0.10` (absolute) with `MIN_MEDIATED_SHARE = 0.25` (fraction).

**Why this is the right quantity, independent of the budget problem.** The mediator exists to punish
"adjust for everything": conditioning on M must visibly move the driver's total-effect estimate. What makes
that trap sharp is the mediated path being a material **share** of that driver's effect (adjust for M →
estimate off by 35–60%), not its absolute size. A fixed 0.10 is simultaneously too large for a weak driver
(the mediated path exceeds the total effect it belongs to) and uninformative for a strong one.

**Why it fixes the budget by construction, not by tuning.** With `k ≤ n` mediated drivers and `β ~ c/√n`:

```
Σ w[M→Y]² = k · (ρβ/e)² ≤ n · ρ²c²/(n·e²) = ρ²c²/e²      — independent of n
```

**Secondary win — sign coherence.** `mediated_target` currently draws an **independent random sign**, so it
can oppose the driver's target; the direct edge then has to over-correct
(`direct = target − mediated`, which can exceed `|target|`). Under R1 the mediated path always shares the
target's sign, so `direct = (1−ρ)·target` is always *smaller* in magnitude than the target.

**Measured, current vs. proposed** (200 draws per n, real `target_effects`):

| n | Σ w[M→Y]² now | Σ w[M→Y]² R1 | Σ direct² now | Σ direct² R1 |
|---|---|---|---|---|
| 3 | 0.33 | 0.138 | 0.198 | 0.042 |
| 6 | 0.51 | 0.216 | 0.311 | 0.066 |
| 8 | 0.65 | 0.292 | 0.405 | 0.090 |
| 10 | 0.76 | **0.356** | 0.507 | **0.116** |
| 12 | 0.89 | 0.478 | 0.660 | 0.153 |

Not perfectly flat in n, because the target ladder itself grows (§6) — but a 2.1× reduction in protected
mediator demand and a 4.4× reduction in direct-edge demand at n=10.

## 3. R1b — confounding margin scales with n

Identical latent wall, same fix. `with_constructed_confounding` solves
`edge = (weakest_driver_corr + slack · CONFOUNDING_MARGIN) / confounder_corr` and rejects when
`edge² > 1`. As n grows, `weakest_driver_corr` and `confounder_corr` both shrink as 1/√n while the added
margin stays at a fixed 0.05 — so the required edge grows and eventually cannot be met. This is the
historical ~25–27% rejection source (§11.6).

**Change:** use `scaled_threshold(CONFOUNDING_MARGIN, n)` in **both** the construction target and the gate
check, so the two stay a single definition (the §5a invariant).

## 4. Exact call sites

**`src/causalagentbench/scm/target_effects.py`**
- Replace `MIN_MEDIATED_EFFECT = 0.10` with `MIN_MEDIATED_SHARE = 0.25`; add
  `_MEDIATED_SHARE_RANGE = (0.35, 0.60)`. Update the module docstring paragraph quoted in §1 — it currently
  documents the behavior being removed.
- `_construct_mediator_edge(...)`: add a `target: float` parameter; compute
  `mediated_target = rng.uniform(*_MEDIATED_SHARE_RANGE) * target` (no independent sign draw — drop the
  `rng.choice((-1.0, 1.0))` on this value only; the `w[A→M]` edge keeps its own random sign).
- `with_constructed_effects(...)`: pass `targets[driver]` at the call site. Decoys are in
  `scaffold.drivers` but have no mediator and no target, so the existing `mediator is None` guard already
  skips them — no extra guard needed, but assert rather than assume.
- `with_constructed_confounding(...)`: derive `n = len(scaffold.non_decoy_drivers())` and use
  `scaled_threshold(CONFOUNDING_MARGIN, n)` in the `target` expression.

**`src/causalagentbench/scoring/gates.py`**
- Import `MIN_MEDIATED_SHARE` in place of `MIN_MEDIATED_EFFECT`.
- Compute `mediated_share[d] = mediated_effect[d] / effect[d]` for non-decoy drivers with a mediator,
  guarding `effect[d] > 0`; `mediation_realized = min_mediated_share >= MIN_MEDIATED_SHARE`.
- `confounding_realized = min_confounding_margin > scaled_threshold(CONFOUNDING_MARGIN, n)`.
- Rename `GateReport.min_mediated_effect` → `min_mediated_share` (the value's meaning changes; a stale name
  here is exactly how the §5a drift started). Update every consumer — `smoke.py` prints gate diagnostics.
- Rewrite the `#:` comment block at ~L57–62, which currently *justifies* the unscaled choice.

**`tests/scoring/test_gates.py`**
- `assert abs(full_estimate - scm.total_effect_on_outcome(mediated_driver)) > 0.02` — an absolute bias
  threshold checking a now-proportional quantity. Make it relative:
  `> 0.5 * MIN_MEDIATED_SHARE * abs(true_effect)` (half the guaranteed share, so the assertion tests the
  guarantee with headroom rather than restating the construction exactly).

## 5. Invariants that must not break

- Both traps still fire: raw correlation still mis-ranks (decoy), full adjustment still biases a mediated
  driver. These are asserted in `test_procedural_oracle_recovers_truth_but_naive_shortcuts_fail`.
- §11.3 balance guarantee stays 200/200 (`_has_balanced_identifiability`) — topology is untouched here.
- §12.2 oracle recovery on `report.data` stays 100/100 — `evaluate_gates` still validates the table it
  returns.
- UUMC unit variance is **retained**. It is what defeats varsortability, and R1 removes the pressure that
  made it look like the problem. Do not free the outcome's variance: it does not help estimability
  (correlations shrink as Var(Y) grows — the same squeeze, relocated) and it costs a defensible property.
- **RNG sequence will shift.** Every construction draw moves, so hardcoded-seed tests may break. Use the
  established fix: search for a new seed satisfying the same structural property, never weaken the
  assertion (`PROCEDURAL_EMPIRICAL_SEEDS`, §12).

## 6. Deferred (measured, not fixed here) — the third instance

`target_effects` stacks the ladder: `magnitude_i ≈ effect_min + i·gap`. Floor and gap each scale correctly
as 1/√n, but stacking `n` of them makes the **top rung grow as √n** and `Σβ² ~ n²`:

| n | min \|β\| | max \|β\| | Σβ² |
|---|---|---|---|
| 3 | 0.141 | 0.303 | 0.164 |
| 6 | 0.090 | 0.339 | 0.322 |
| 8 | 0.075 | 0.363 | 0.458 |
| 10 | 0.066 | 0.387 | 0.618 |
| 12 | 0.059 | 0.409 | **0.803** |

Bounding the ladder (so `effect_max` also scales as 1/√n) forces `gap ~ n^-1.5`, which then collides with
sampling noise (fixed at ~1/√N): the oracle itself stops being able to order adjacent drivers, which is
precisely what the `oracle_recovers` gate rejects. **The ladder fix and the sample-size raise are one
coupled decision (`N ≳ n³`), not two independent ones**, and neither is in scope here.

Expected reach of R1/R1b alone: protected demand at n=10 falls to ≈0.72 of budget (from ≈1.27), so
**n_drivers=10 should become feasible; n=12 will still not be.** Treat that as a prediction to verify, not
a result — it decomposes the budget analytically and ignores parent-correlation effects in the true
bilinear form `incoming @ sigma @ incoming`.

## 7. Verification plan (baselines to beat)

1. **Acceptance by driver count**, 500 candidates each, n = 3…10. Current: 69.6 / 47.8 / 42.0 / 26.1 /
   17.6 / 11.2 / — / 0.3%. Expect mediation-infeasible ≈ 0% at every n, and n=10 usable (target ≳10%).
2. **Rejection breakdown at n=10** — currently mediation 81.0%, confounding 10.3%, oracle 8.3%. Expect
   mediation ≈ 0 and oracle to become the binding constraint (which is the honest one: it means the task is
   genuinely hard, not that construction failed).
3. **`smoke.py` overall acceptance** — currently 35.4%; expect a rise, and report the new per-n table.
4. **Full suite + lint/types**: 186/186 currently, ruff/ruff-format/mypy clean.
5. **Trap sharpness** — confirm the over-adjustment bias is now proportional: report the distribution of
   `|full_adjustment_estimate − true| / |true|` across accepted instances (expect ≈ 0.35–0.60, i.e. the
   share ρ, versus an unbounded ratio today).

## 9. Measured result

500 candidates per driver count, same harness as the §0 baseline.

| n_drivers | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| accepted **before** | 69.6% | 47.8% | 42.0% | 26.1% | 17.6% | 11.2% | — | 0.3% |
| accepted **after** | 71.8% | 61.0% | 51.8% | 37.0% | 27.8% | 16.2% | 10.2% | **4.4%** |
| mediation-infeasible **before** | — | — | — | — | — | — | — | **81.0%** |
| mediation-infeasible **after** | 0.0% | 0.0% | 0.0% | 0.0% | 0.2% | 2.8% | 11.6% | **26.4%** |

- `smoke.py` overall acceptance: **35.4% → 43.6%**.
- Mediation as a failure mode is **eliminated at n ≤ 6** (0.0%) and cut 3× at n=10.
- Confounding rejection improved *conditionally* — of candidates that survive mediation, the rejection rate
  fell from 54% to 36% at n=10 (the raw 10.3% → 26.8% rise is only because 3.7× more candidates now reach
  that step at all).
- **Oracle recovery is now the dominant rejection at n = 5–8 (30–53%)**, which is the intended outcome: it
  is the honest constraint (the ranking task is genuinely hard at that size), not a construction artifact.
- Trap sharpness is now proportional as designed: over-adjustment bias relative to the driver's own effect
  has median **0.47** (p25 0.40, p75 0.55) across 326 mediated drivers — matching `_MEDIATED_SHARE_RANGE`.
- Invariants hold: balance guarantee 200/200; oracle recovery on `report.data` 100/100; realized
  `min_mediated_share` ≥ 0.351 across accepted instances (floor is 0.25). 187/187 tests, ruff/mypy clean.
  The RNG sequence shifted as expected but no hardcoded-seed test needed re-searching this time.

**The §7 target of "n=10 usable (≳10%)" was not met — 4.4%.** R1/R1b did what they were scoped to do (remove
the fixed-magnitude thresholds as a failure mode, ~1.3–2.1× acceptance at every n, 14.7× at n=10), but at
n ≥ 9 the deferred §6 ladder problem takes over: mediation-infeasible climbs back to 11.6% / 26.4% because
the *targets themselves* grow with n, so a fixed share of a growing target is still a growing demand.
**n=9 is borderline (10.2%); n=10 needs §6.**

## 10. Correction: sign coherence was a validity regression, not a win

§2 originally claimed the mediated path taking the target's sign was a "secondary win" because it made the
direct edge `(1−ρ)·target`, always smaller than the target. That was wrong on the merits, and the audit
caught it before any dataset was generated from it.

Forcing the signs to agree makes `|direct| < |total|` hold **universally**, so adjusting for a mediator
always shrinks the estimate. Measured on 120 accepted instances:

| probe | shrinks the estimate (same-sign) | after the fix |
|---|---|---|
| add the **true mediator** to the adjustment set | **395/395 = 100.0%** | 217/403 = 53.8% |
| add some other variable | 212/395 = 53.7% | 200/403 = 49.6% |

A perfectly separating numeric signature for "this variable is a mediator" lets an agent dodge the
over-adjustment trap by probing rather than by reasoning about what is downstream of the treatment — which
is the entire reasoning step the trap exists to force. It also deletes **inconsistent mediation**
(suppression: a direct and an indirect effect that oppose), a real structure; `|direct| < |total|` is a
consequence of the sign choice, not a mathematical necessity.

**Fix:** the mediated share keeps its magnitude range but draws its **sign independently**, so consistent
and inconsistent mediation each occur about half the time. The scaling property R1 exists for — the
mediated path being *proportional to its own driver's effect* rather than a fixed magnitude — is
independent of sign and is fully preserved.

**Cost: essentially nothing.** The budget analysis predicted ~1.7× more demand, but acceptance is unchanged
within noise, because the shrink ladder absorbs it and the binding constraint at mid-n is oracle recovery,
not budget:

| n_drivers | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| same-sign | 71.8% | 61.0% | 51.8% | 37.0% | 27.8% | 16.2% | 10.2% | 4.4% |
| **mixed-sign (final)** | 69.8% | 63.2% | 50.4% | 37.0% | 28.2% | 15.4% | 9.8% | 4.8% |

`smoke.py` overall: 35.4% (baseline) → 43.6% (same-sign) → **40.6% (final)**. Soundness restored for ~3
points of acceptance.

## 8. Sequencing note

This changes generated data. `outputs/databricks_internal` and any eval results measured against it become
stale the moment this lands — the dataset must be rebuilt and the model sweep rerun afterward, not merged
across the change.
