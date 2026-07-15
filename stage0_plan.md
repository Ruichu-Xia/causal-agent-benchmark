# CausalAgentBench — Stage 0 Plan: Graph / SCM Generation

> **Audience: the implementing coding agent (e.g. Codex).** This is the authoritative, self-contained
> specification for **Stage 0** — the graph/SCM generator only. Implement exactly this; do not expand
> scope. If a decision here seems wrong, flag it rather than silently deviating. Later stages are
> sketched in §7 for context but are **out of scope** for Stage 0.
>
> Repo root: `/Users/rxia/workspace/causal-agent-benchmark` (greenfield; not yet a git repo).
> Target Python: **3.12**. Package manager: **uv**. Package name: **`causalagentbench`** (src layout).

---

## 0. Implementation notes (as-built)

Stage 0 is implemented and green (ruff + mypy strict clean; 45 tests pass; smoke acceptance ~9%).
Deltas discovered during implementation, superseding the spec below where they differ:

- **Package name** is `causalagentbench`, not `causalbench` (the latter collides with two existing
  "CausalBench" papers).
- **UUMC exact formula.** Verified against the authors' source and implemented faithfully (not the
  earlier "Mahalanobis" paraphrase, which pins `R2=r2` and is *not* UUMC). Per node with `d` parents:
  `a ~ N(0,1)`, `norm = sum(a^2)`, `r = U(0,1)^(1/d)`, `S = a.T @ Sigma_pa @ a`,
  `scale = sqrt(r^2*S + (1-r^2)*norm)`, `coef = (r/scale)*a`, `noise_var = (1-r^2)*norm/scale^2`
  (gives unit variance exactly). Reimplemented to avoid the GPL-3.0 `UUMCdata` package; cite Herman
  et al. (CLeaR 2025).
- **Four hard gates, not five.** `structure_valid`, `confounding_realized`, `mediation_realized`,
  `oracle_recovers`. Sortability is a **reported diagnostic**, not a gate: per-instance empirical
  varsortability is ensemble-noise on a 6-node graph, and UUMC already guarantees population-level
  neutrality (accepted-mean varsortability ~0.48 confirms it).
- **Confounding gate reframed** as a *construction-validity* check ("confounding realized" for this
  confounded scaffold), comparing the decoy's correlation to the true drivers' **correlations** (not
  effects). It is not a universal "correlation always fails" law; unconfounded regimes arrive as
  sibling scaffolds scored stratified.
- **Data is presented as raw draws** (`sample()` does not standardize); normalization/rescaling is a
  later presentation-layer concern.
- **Layout:** flat `src/causalagentbench/` (no `scm/` subpackage); baselines consolidated into
  `baselines.py`; added `metrics.py`; the `Scaffold` dataclass is the extensibility seam for new
  graph structures. Gate thresholds are constants in `gates.py`, slated to migrate into
  `GeneratorConfig`.
- **Tooling:** uv + ruff + mypy(strict) + pytest + hypothesis; line-length 120. pre-commit and CI
  deferred (add CI at first GitHub push).

---

## 1. Context & goal

**CausalAgentBench** is a verifiable causal-reasoning benchmark **and** RL environment for tool-using
agents (per `causalagentbench_1.html`). Data is generated from a hidden linear-Gaussian SCM, so the SCM
is the exact answer key. An agent must rank the drivers of an outcome `Y` by causal effect, choose
adjustment sets, flag identifiability, and abstain when an effect is not identifiable. Used as a reward,
the same generator is an RL environment.

**Working method:** build the smallest correct thing first, keep it extensible, grow incrementally.
Stage 0 delivers *only* a trustworthy generator whose soundness gates pass — no agent, no scoring axes,
no framework adapters, no CLI.

---

## 2. Research basis (why the design is what it is)

**Engineering conventions** (SWE-bench, Terminal-Bench/Harbor, lm-eval-harness, lighteval, Prime
Intellect `verifiers`, Inspect, METR task-standard):
- Framework-agnostic **core + thin adapters** (don't weld logic to a pre-1.0 framework API).
- **Generator-based instances** from `(seed, config)`; a cached snapshot + the generator can be
  distributed later (HF `GeneratorBasedBuilder` pattern). Gold (the SCM) stays hidden.
- **Immutable-oracle, artifact-graded** harness; **lazy registry/plugin** extensibility; **Gymnasium
  seeding discipline** (seed → deterministic); versioned, metadata-rich artifacts.

**Causal-generation methodology:**
- **Sortability traps** (Reisach et al., *Beware of the Simulated DAG!*, NeurIPS 2021; R²-sortability,
  NeurIPS 2023): naive i.i.d. weights + fixed noise leak the causal order through variance accumulation,
  and **R²-sortability survives standardization**. Mitigation: standardize *during* generation. We use
  **UUMC** (Herman, Wahl, Ninad & Runge, *CLeaR 2025*, PMLR v275); iSCM (Seiler & Reisach 2024) is the
  simpler fallback.
- **Prior causal benchmarks** use *small* graphs, confounder-as-fork (`C→X`, `C→Y`) is universal, and
  canonical role structures (CLadder, CausalFlip, NoisyCausal) beat random DAGs for controlled role /
  identifiability guarantees. *(Caveat: some arXiv IDs found during research looked auto-generated;
  conventions are trusted, specific paper numbers are unverified.)*
- **Tooling:** exact linear-Gaussian effects via **NumPy `(I−W)⁻¹`** (exact, reproducible, no bloat);
  `networkx` for DAG representation. `pgmpy`/`ananke` are for later identifier stages, **not** Stage 0.

---

## 3. Locked design decisions

- Structure family: fixed **role-scaffold** — fix roles + guarantees, randomize the realization.
- v0 topology: **single canonical topology** (6 nodes, §4.1); randomize weights, noise, and
  role→label/column assignment only.
- Sortability: **UUMC**, applied via exact marginal-variance rescaling (parents are correlated → the
  independent-parent shortcut is wrong here; use the Mahalanobis form in §4.3).
- Confounding coverage: **only `A` confounded; `B` clean** (adjustment set `∅`).
- **One bad-control trap in v0: a mediator `M` on `A`'s path to `Y`** (must NOT be adjusted for). This
  makes adjustment-set *selection* load-bearing and defeats the "adjust for everything" shortcut.
- Sample size: **fixed `n = 2000`** (tunable config field).
- Dependencies (Stage 0): **`numpy` + `networkx`** runtime; `pydantic` v2 for config/types.

---

## 4. Stage 0 design — the canonical v0 generator

### 4.1 Roles, nodes, and the canonical DAG

Six nodes, fixed roles: `C` (observed confounder), `A` (confounded driver), `B` (clean driver),
`M` (mediator = observed bad-control), `Dz` (decoy), `Y` (outcome).

Canonical directed edges:
```
C → A      C → Y      C → Dz          (C is a fork: confounds A; induces Dz–Y correlation)
A → M      M → Y                       (M mediates part of A's effect on Y)
A → Y                                  (A's direct effect)
B → Y                                  (clean driver)
```
- `Dz` has **no outgoing edge** ⇒ total effect on `Y` is **exactly 0** (structural, weight-independent).
- `A`'s backdoor path `A ← C → Y` is blocked by `{C}`. Its causal paths `A→Y` and `A→M→Y` must stay
  **open** when targeting the **total** effect ⇒ `M` must **not** be in `A`'s adjustment set.
- `B` is exogenous, unconfounded ⇒ correct adjustment set `∅`.

**Candidate drivers ranked:** `{A, B, Dz}`. **Observed covariates available to adjust:** `{C, M}` (plus
the other drivers). `M` is a cause of `Y` but is a covariate/nuisance, not a ranked driver.

In-degrees: `C=0, B=0, A=1, Dz=1, M=1, Y=4`. Topological order: `{C,B} → {A,Dz} → {M} → {Y}`.

### 4.2 Notation & exact ground truth (linear-Gaussian)

Weighted adjacency `W`, `W[i,j]` = coefficient of parent `i` in the equation for `j`.
`X_j = Σ_i W[i,j] X_i + N_j`, i.e. `X = Wᵀ X + N`, so `X = (I − Wᵀ)⁻¹ N`.

- **Total-effect matrix:** `τ = (I − W)⁻¹ − I`; `τ[i,j]` = sum of path-weight products over all directed
  `i→…→j` paths. For this scaffold:
  - `τ[A,Y] = W[A,Y] + W[A,M]·W[M,Y]`  (direct + mediated),
  - `τ[B,Y] = W[B,Y]`,  `τ[Dz,Y] = 0`,  `τ[M,Y] = W[M,Y]` (M not ranked).
- **Marginal covariance:** `N_j ~ N(0, σ_j²)`, `D = diag(σ²)`: `Σ = (I − Wᵀ)⁻¹ D (I − W)⁻¹`.
- **Standardized units.** The agent sees standardized columns (variance 1). Ranking target =
  **standardized total effect** `τ̃`, computed on `W̃[i,j] = W[i,j]·√(Σ_ii)/√(Σ_jj)`. Standardization
  preserves the structural zero: `τ̃[Dz,Y] = 0`.

**Ranking target:** order `{A, B, Dz}` by `|τ̃[·,Y]|`; by construction + rejection
`|τ̃[A,Y]| > |τ̃[B,Y]| > 0 = τ̃[Dz,Y]`. **Correct adjustment sets:** `A→{C}` (excludes `M`), `B→∅`,
`Dz→` not a driver.

### 4.3 Parameterization — UUMC generalized to correlated parents

Process nodes in topological order, tracking exact parent covariance `Σ_pa` (known from already-set
coefficients). For node `j`, parents `pa(j)`, `d = |pa(j)|`:
1. Draw direction `a ~ N(0, I_d)`, radius `r ~ Unif(0,1)^{1/d}` (`r² = R²_j`, target signal share).
2. **Mahalanobis-normalize** to parent covariance: `c = r · a / √(aᵀ Σ_pa a)` ⇒ signal variance
   `cᵀ Σ_pa c = r²`.
3. Noise variance `σ_j² = 1 − r²` ⇒ node variance `= 1` exactly.
4. Roots (`C`, `B`): unit-variance standard normal, `R² = 0`.

This preserves UUMC's per-node `R² = r²` property while handling correlated parents exactly, and yields
all-unit-variance data (so `var`-sortnregress has no signal by construction).

### 4.4 Reproducibility
`generate(seed, config) -> SCM` deterministic from `seed` (a single `numpy.random.Generator`).
`sample(scm, n, seed) -> X` draws the `n×6` array and standardizes each column.

### 4.5 What v0 tests — and what is deferred
Measures two floor skills together: **(a)** adjust for a confounder (beat naive correlation) and
**(b)** don't over-adjust (exclude the mediator to recover the *total* effect). Two trivial shortcuts
provably fail: **no-adjustment** (raw correlation, gate 2) and **over-adjustment** (adjust for everything
incl. `M`, gate 3).

**Deferred (later stages):** other bad-control types (colliders, M-bias, descendants-of-`Y`), multiple
confounders/mediators, treatment→treatment mediation among ranked drivers, semantic naming +
aligned/misaligned axis, latent-confounder non-identifiability (Knob 5) + abstention, the three-axis
scorer, agent/adapters/CLI/packaging.

### 4.6 Soundness gates (verify-reject) — the generator's contract
Accept an instance only if all hold; these double as the property tests.
1. **Oracle recovers truth.** OLS of `Y` on `{driver} ∪ correct-set` (`A|{C}`, `B|∅`) recovers the true
   standardized ranking of `{A,B,Dz}` (Kendall-τ ≈ 1; `Dz` partial effect ≈ 0).
2. **No-adjustment fails (primary).** Ranking `{A,B,Dz}` by `|corr(·,Y)|` differs from truth; require
   `|corr(Dz,Y)| > |τ̃[B,Y]| + confounding_margin` (decoy outranks the weak driver ⇒ confounding strong
   enough; else discard).
3. **Over-adjustment fails (trap).** OLS of `Y` on `{A,B,C,M,Dz}` yields `A`'s *direct* effect `W̃[A,Y]`,
   biased from total by the mediated part; require `|W̃[A,M]·W̃[M,Y]| ≥ min_mediated_effect` (trap is
   material; also check/report whether it flips the `{A,B,Dz}` ranking).
4. **Sortability hygiene.** All node variances = 1; keep varsortability ∈ [0.4, 0.6]; report
   R²-sortability. (Secondary for a `Y`-designated task; kept for honesty/publishability.)
5. **Structural sanity.** `τ̃[Dz,Y]` exactly 0; `|τ̃[A,Y]|`, `|τ̃[B,Y]|` nonzero and separated ≥
   `min_effect_gap`.

### 4.7 Config schema (pydantic v2 `GeneratorConfig`)
`seed: int`, `n_samples: int = 2000`, `min_effect_gap: float`, `confounding_margin: float`,
`min_mediated_effect: float`, `varsortability_bounds: tuple[float,float] = (0.4, 0.6)`,
`max_rejections: int`. (Choose sensible defaults; expose all as fields.)

### 4.8 Generator output — the hidden answer key (`SCM`)
Frozen object with: `nodes`, `role_map` (Role→node), `W`, `noise_var`, `tau`, `tau_std` (τ̃),
`correct_adjustment` (per driver), `column_order` (randomized observable order), `X` (n×6 standardized
sample). The agent-facing CSV + prompt wrapping is a later stage.

---

## 5. Stage 0 implementation deliverable

Production-grade skeleton + the generator whose gates pass. **No agent, no scoring axes, no adapters.**

**Repo bootstrap:** init git; `uv` project on Python 3.12 (commit `uv.lock`, `.python-version`,
`.gitignore`); `pyproject.toml` (PEP 621) — runtime `numpy`, `networkx`, `pydantic`; dev `pytest`,
`hypothesis`, `ruff`, `mypy`; ruff/mypy/pytest config in `pyproject.toml`; `.pre-commit-config.yaml`;
`.github/workflows/ci.yml` (lint → typecheck → test); `README.md`.

**Package layout (src layout):**
```
src/causalbench/
  __init__.py
  config.py            # GeneratorConfig (pydantic v2)
  types.py             # Role enum; SCM frozen model (see §4.8)
  scm/
    scaffold.py        # canonical 6-node role graph (C,A,B,M,Dz,Y) as networkx DiGraph + edge set
    generate.py        # generate(seed, config) -> SCM  (UUMC + Mahalanobis rescale, topo order)
    effects.py         # tau = (I-W)^-1 - I; marginal covariance Sigma; standardized W~ and tau~
    sample.py          # sample(scm, n, seed) -> X  (X=(I-W^T)^-1 N, per-column standardize)
    sortability.py     # varsortability + R2-sortability
  baselines/           # only what the gates need (plain numpy lstsq; NO statsmodels)
    oracle.py          # per-driver correct-set regression ranking
    correlation.py     # raw |corr| ranking
    full_adjust.py     # regress Y on all covariates (the over-adjustment trap baseline)
  generate_gated.py    # generate + reject loop enforcing the 5 gates
```

**Tests (gates == integration tests):**
- `test_effects.py`: `tau` correct on a hand-checked small `W` (incl. `A→M→Y`); `τ̃[Dz,Y] == 0`.
- `test_generate.py`: determinism (same seed ⇒ identical SCM); all nodes unit variance; effect gap;
  mediated effect ≥ `min_mediated_effect`.
- `test_gates.py` (`hypothesis`, many seeds): oracle Kendall-τ ≈ 1; no-adjustment mis-ranks (gate 2);
  over-adjustment biases `A` (gate 3); varsortability ∈ [0.4, 0.6]; structural sanity.

---

## 6. Verification (definition of done for Stage 0)
- `uv run pytest` green (determinism, effects, all five gates).
- `uv run ruff check` and `uv run mypy src` clean; pre-commit installed.
- A smoke script generates ~200 gated instances and prints: per-gate pass-rates, reject-loop acceptance
  rate, mean/spread of varsortability + R²-sortability, and the fraction where over-adjustment flips the
  ranking — confirming sound, non-trivial instances at a reasonable yield.

---

## 7. Roadmap (later stages — NOT Stage 0; each discussed before coding)
1. **Stage 0 (this doc):** canonical 6-node generator + gates.
2. More bad-control types (colliders, M-bias, descendants-of-`Y`); trap robustness.
3. Instance data model + contracts (hidden SCM vs agent CSV/prompt vs answer schema).
4. Identifiability checker + oracle (networkx/pgmpy; ananke for latents).
5. Three-axis verifier (ranking / identification / calibration).
6. Baselines + verify-reject harness as property tests (extend).
7. Agent protocol `solve(instance) -> Answer` + baselines; LLM agent later.
8. `verifiers` adapter, CLI, result artifacts, packaging (Hub-ready, publish later).
9. Naming axis (aligned/misaligned) → latent-confounder abstention → mediation knobs → time-series.
