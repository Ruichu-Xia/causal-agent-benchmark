# CausalAgentBench — Semantic Labeler Plan (three-pole discrete alignment, constructed)

> **Status: implemented** (`generation/labeler.py`, `generation/grounding.py`, `task/labeling.py`) and the
> authoritative description of the naming layer, with **two amendments**. The α grid was collapsed from
> five values to three (§3.2, §8, §9 carry that correction inline), and then to **two** — `{-1, +1}` —
> because neutral produced the same gold as misaligned in every cell. The generator prompt still asks for a
> neutral variant so a 0.0 level can be restored without a generator change, but no grid level displays it,
> so it is neither validated nor critiqued. The snake_case name check was also removed: it rejected names
> that were correct as names and wrong only in letter case (`dissolved_oxygen_mgL_z`).
> See `procedural_scaffold_plan.md` §11.8 and `backlog.md` for the history.
> **Supersedes** the
> earlier LLM-generate→judge→select design (see "§10 What this replaces"). Package `causalagentbench`
> (src layout, Python 3.12, uv, hatchling; ruff + mypy strict + pytest; line-length 120; concise ASCII
> docstrings).

---

## 1. Why (motivation the design must serve)

A **tool-using causal-investigation** benchmark (peer: CausalDS — LLM agents use tools/code on synthetic
SCMs, with abstention). The agent is a **real-world data scientist with tools**: it investigates the data
(causal discovery like PC/GES, regression, CI tests, backdoor adjustment) to rank the candidate
treatments and choose adjustment sets, **uses domain knowledge when it helps**, and **concedes (abstains)
when the result cannot be trusted**. Solving from data with tools is expected and fine — that is the job.

Variable names + context are that **domain knowledge**, and — as in the real world — their reliability
*varies across datasets*:

- **aligned** — the names genuinely reflect the causal structure → a good analyst *uses* them (to reach
  or cross-check the answer).
- **neutral** — administrative / uninformative → no help, no harm.
- **misaligned** — the names suggest the *wrong* structure → a good analyst notices the conflict with the
  data and does *not* follow them.

The measured skill is **appropriate reliance + calibrated abstention**: give a trustworthy answer when the
data (optionally aided by reliable names) supports one; **override** names that contradict the data; and
**abstain** when neither the data nor trustworthy domain knowledge settles it. Names are neither mandatory
nor useless — there is no "data-alone must fail" requirement. The three-pole naming axis creates these
conditions; the answer schema's per-driver `identifiable` flag is how the agent concedes.

**Contribution (positioning):** a controlled, exactly-verified manipulation of variable-name reliability
over a fixed hidden SCM with real data, measuring the *appropriate reliance + abstention* of tool-using
causal agents — beyond aligned-by-construction (CausalDS, whose "identification nearly solved" partly
reflects reliable names inflating scores), binary name-on/off (Caliper), name-perturbation robustness
(Corr2Cause / Jin et al.), and templated commonsense/anti-commonsense QA (CLadder). We add the exact SCM
+ real data, the neutral pole, and the tie to a procedural task (ranking + adjustment sets + a confounded
decoy) with abstention.

## 2. The three poles (per node)

Alignment is defined on the axes the task actually grades — **relevance/magnitude** (drives the ranking)
and **role** (drives the adjustment set). **Not sign**: the answer is a ranking by |effect|, so
sign-inversion misleads nothing and is excluded.

For each node, one coherent domain supplies **three** name choices:

- **aligned** — prior matches the node's true role: a strong-cause name on a strong-effect driver
  (prominence ∝ |effect|), an **inert** name on the zero-effect decoy, a common-cause-evoking name on the
  confounder, a mechanism/on-the-path name on the mediator, the real target name on the outcome.
- **neutral (non-associated)** — a plausible real variable with **no** causal prior toward the outcome
  (an id, a code, an unrelated measurement, e.g. `record_id`, `site_code`). Neither helps nor misleads.
- **misaligned** — a **task-relevant** trap: a **compelling cause-name on the zero-effect decoy** (the
  spurious-correlate trap), an inert name on a strong driver, and **swapped role-cues** for
  confounder↔mediator (the bad-control trap). No sign-flips.

Why three: for the decoy, *aligned and neutral names are both inert* → a reciter and a reasoner behave
identically. **Only the misaligned name (a compelling cause on a non-cause) creates a trap** that
separates gullible recitation from data-grounded reasoning. Neutral tests *distraction*; misaligned tests
*gullibility* — different failure modes, both needed.

## 3. Mechanism — construct the level, don't measure it

**Per SCM**, a generator LLM (grounded by CauseNet, seeing the edge list + each node's true effect +
roles) returns, per node, the three (name, description) choices above, in one coherent domain, all
distinct. There is **no alignment *judge*** — the level is *constructed* (§3.2), so it is exact and free
of calibration noise. But there **is** a **correctness *critic*** (§3.1): all quality now rests on the
generator, so we verify-don't-trust its proposal and iterate.

### 3.1 Quality control: deterministic pre-checks + LLM critic + revise loop

A **propose → critique → revise** loop de-risks the generator (which we measured failing: placeholder
names, incoherent domains, weak/mislabeled traps). Note this is a *correctness* verifier (discriminative
yes/no per name — reliable), **not** the old continuous α-scorer (which was unreliable and is gone).

1. **Deterministic pre-checks (free, always).** All `3k` names distinct + snake_case; **no placeholder
   patterns** (`feature_\d+`, `latent_component_\d+`, `channel_[a-z]`, `v\d+`, `x\d+`, bare ids);
   descriptions non-empty, continuous-type (no binary/count/yes-no, no causal-graph narration); full node
   coverage.
2. **LLM critic (per SCM).** Given the true graph + roles + the proposed triples, verify per node: is the
   **aligned** name genuinely strong-cause-on-a-strong-node / inert-on-the-decoy? is **neutral** genuinely
   non-associated? is **misaligned** a genuine compelling task-relevant trap (cause-name on the
   zero-effect decoy; swapped confounder/mediator cues)? Returns `ok` or a list of per-node issues.
3. **Revise.** If pre-checks or the critic flag issues, feed them back to the generator for a revised
   proposal; repeat up to `max_revisions`; if still failing, **skip and log** that SCM (do not ship junk).

Cost is per-SCM (1 generate + up to `max_revisions` revise/critique calls), not per-candidate — modest.
The critic is optional (`critic_model=None` runs pre-checks only).

### 3.2 Constructing the discrete levels

**Discrete grid** `α ∈ {−1, 0, +1}`, applied **uniformly** (`GRID`, `construct_labeling`):
- Only α's **sign** is read. Every non-outcome node takes the same kind: **aligned** if `α > 0`,
  **misaligned** if `α < 0`, **neutral** if `α = 0`. The outcome always keeps its real aligned name.
- So `α = +1` → all aligned; `α = 0` → all neutral; `α = −1` → all misaligned.
- Deterministic and reproducible; no per-node ordering or subset selection is involved.

> **Amended 2026-07-27.** This section originally specified a five-value grid `{−1, −0.5, 0, +0.5, +1}`
> built by "fraction through neutral", where the two midpoints flipped only the top `round(|α|·K)` nodes
> of a hand-tuned load-bearing order (`_informative_order`: decoy, strongest driver, confounder, mediator,
> clean driver). That order conflated two unrelated justifications — quadrant coverage and score
> sensitivity — with no single principle, and the endpoints never needed it (every node already flipped
> there). Dropping the midpoints removed the question instead of re-answering it; `_informative_order` was
> deleted, not left unused. Trade-off taken deliberately: the dose-response middle of the curve is given
> up for a directly comparable per-quadrant contrast. See `procedural_scaffold_plan.md` §11.8.

**Name-blind (`X1..Xk`) is a separate control** — "no names at all," distinct from `α = 0`'s "real but
uninformative names." Keep `abstract_labels` as the off-axis anchor.

**Paired skins are free:** the same SCM + one triple set → all grid levels (+ name-blind) by construction
→ identical data, only names differ. This is the low-variance same-SCM contrast the metric wants.

## 4. Data model

```python
@dataclass(frozen=True)
class LabelProvenance:
    alignment: float | None      # the grid level in [-1, 1]; None for name-blind
    n_aligned: int               # constructed composition, for auditing
    n_neutral: int
    n_misaligned: int
    node_kind: Mapping[str, str] # node -> "aligned"/"neutral"/"misaligned"; read by the abstain gold rule
    domain: str | None
    producer: str                # "abstract" | "llm:<gen-model>+critic:<critic-model>"

@dataclass(frozen=True)
class Labeling:                   # unchanged public shape
    node_to_label: Mapping[str, str]
    context: str
    column_schema: Mapping[str, str]
    provenance: LabelProvenance
```

`Instance` carries `context` + `column_schema`; `InstanceKey` carries `provenance`. `abstract_labels`
returns a name-blind `Labeling` (`alignment=None`). The **ranking and adjustment** gold is computed on the
hidden SCM via `label_to_node`, so it is **invariant** to the level — naming is a pure presentation
transform for those two axes.

**One deliberate exception, added later:** the **abstain** gold rule is *not* level-invariant, and cannot
be. A driver whose confounder edge is unresolvable from data alone is answerable only if its own name
truthfully resolves it, so `_gold_abstain` reads `provenance.node_kind[driver]` alongside the topology
(`procedural_scaffold_plan.md` §6/§7). This is the mechanism that makes naming load-bearing rather than
decorative; it is per-driver, not per-instance, which is why `node_kind` exists as a field rather than the
aggregate counts being sufficient.

## 5. The LLM output contract

Generator returns JSON (nodes shown under neutral ids `v0..vk` so ids don't leak roles):
```json
{"domain": "...", "context": "...",
 "nodes": {"v0": {"aligned": {"name": "...", "description": "..."},
                   "neutral": {"name": "...", "description": "..."},
                   "misaligned": {"name": "...", "description": "..."}}, "...": {}}}
```
Parse to per-node triples; enforce structural validity by **skipping malformed** outputs (all nodes
covered, all `3k` names distinct, snake_case). Descriptions state **what the variable measures** (a data
dictionary) and are **continuous, real-valued** (the data is standardized) — never binary/count/yes-no,
never narrating the causal graph.

## 6. Clients & grounding (optional `generation` extra)

- One LLM seam `LabelProposer.propose(prompt) -> str` (`LiteLLMProposer(model)`) serves **both** roles —
  the **generator** and the **critic** are the same call shape with different prompts. Retries transient
  errors (`num_retries`, `tenacity`); `.env`-loaded keys.
- CauseNet grounding: `load_causenet` + `sample_menus` (unchanged) — supplies the real domain concepts.
- No embeddings, no continuous α-judge, no candidate generation/selection. (The critic is a *correctness*
  verifier, §3.1 — discrete yes/no, not a continuous scorer.)

## 7. Two modes

- **Built-on-demand dataset.** For a set of gated SCM seeds, produce the triple set once per SCM and
  **materialize all grid levels + name-blind** (paired skins). `scripts/build_testset.py` writes full
  instances (data in `.npz`, SCM params + naming card in `.json`); `load_dataset()` is pure I/O.
  Incremental write for crash-safety. Consuming needs **no LLM, key, or CauseNet**. Note: nothing is
  *shipped* in-repo — the packaged `testset_v1.json`/`.npz` that this section originally described was a
  hand-authored placeholder predating the real pipeline, and was deleted rather than documented around.
  Every dataset is now built by running the pipeline.
- **Generate from scratch.** `generate_instance(seed, level, labeler=LLMLabeler(...))` /
  `alpha_sweep(seed, levels, ...)` run the pipeline live; store nothing. `level` is a grid value.

## 8. Config

`BuildConfig` (`configs/testset_v1.yaml`):
```yaml
gen_model: openai/gpt-5        # generator
critic_model: openai/gpt-5     # correctness critic (null to skip; same model is fine, or a different one)
max_revisions: 2               # propose -> critique -> revise iterations before skipping an SCM
n_scms: 20                     # base SCM seeds; each yields all grid levels (+ name-blind) as paired skins
```
Secrets + CauseNet dump path stay in `.env`. Grid is fixed `{−1, 0, +1}` (+ name-blind), so a full sweep is
4 views per SCM.

## 9. Verification (definition of done)

`ruff check`, `ruff format --check`, `mypy src`, `pytest` clean/green. Tests use a **stub proposer** (no
network in CI):
- **Level construction (pure):** given a fixed triple set, `α=+1` uses all aligned names, `α=−1` all
  misaligned, `α=0` all neutral; `provenance` composition counts and `node_kind` match; deterministic.
- **`LLMLabeler` with a stub proposer:** parses triples → valid `Labeling` at a requested level;
  descriptions/names pass the continuous/snake_case/distinct checks.
- **Quality loop (stub proposer + stub critic):** deterministic pre-checks reject a planted placeholder
  name (`feature_1`) / duplicate / binary description; a critic that returns issues triggers a revise, and
  a second clean proposal is accepted; if it never passes within `max_revisions`, the SCM is skipped (not
  shipped). `critic_model=None` runs pre-checks only.
- **`build_instance`:** columns are the chosen names; context/schema populated; `provenance.alignment ==
  level`; `label_to_node` bijection. Name-blind → `alignment is None`.
- **Paired skins / reward-invariance:** one SCM across all grid levels shares column order + node-space
  gold (`true_ranking` identical); only names differ. (Ranking/adjustment only — the abstain gold rule is
  level-dependent by design, §4.)
- **`load_dataset`:** rebuilds a built dataset; gold matches the SCM; the set contains all grid levels.
- Grounding (`load_causenet`/`sample_menus`) against the synthetic fixture (unchanged).

## 10. What this replaces (remove)

- The continuous **α-judge**: `AlignmentJudge`/`LiteLLMJudge`, `_judge_prompt`, `_parse_scores`,
  `_select`, and the diverse-candidate generation. (A *correctness critic* is added instead — §3.1 —
  different role: verify the triples, not score α. `judge_model` → `critic_model` in config.)
- Continuous drawn α / `alpha ~ U(-1,1)` → the fixed discrete grid, constructed.

Kept: SCM generator + gates, sampler, `Instance`/`InstanceKey`, the `Labeler` seam, `abstract_labels`,
CauseNet grounding, LiteLLM client + retries, incremental materialize + `load_dataset`, `alpha_sweep`.

## 11. The identifiability × naming matrix (design target) & roadmap

There is **no "data-alone must fail" gate** — a data-solvable instance is valid (the agent uses tools and
overrides misleading names). Instead the set spans a 2×3 matrix, and the *correct* data-scientist
behavior per cell defines the scoring:

| | aligned (credible) | neutral | misaligned |
|---|---|---|---|
| **identifiable from data** | answer | answer (ignore names) | answer (override names) |
| **not identifiable from data** | answer (use domain knowledge) | abstain | abstain (don't trust it) |

- Top row: names are a **robustness** probe — always answer; a parrot is baited by `misaligned`.
- Bottom row: names are **load-bearing** — credible aligned names let you answer (identification via
  domain assumptions); otherwise **abstain**. The agent discriminates by cross-checking names against
  world-plausibility and the data-visible constraints (CPDAG / CI tests); misaligned names typically
  contradict a visible constraint (→ override/abstain), and genuinely-unverifiable misleading names call
  for caution (abstain). A confident wrong answer in the bottom row is the failure to catch.

Scoring rewards **appropriate reliance + calibrated abstention** across this matrix (per-driver
`abstain: bool` is how the agent concedes — the field was named `identifiable` when this section was
written; it was inverted and renamed with the abstain mechanism).

**Built since.** The matrix is live, collapsed to the 2×2 in `procedural_scaffold_plan.md` §6 (neutral and
misaligned share the "not aligned" column, per explicit instruction 2026-07-23 — the distinction between a
merely-uninformative and an actively-wrong name on an unidentifiable driver is deferred, not resolved).
`scoring/score.py`'s `Quadrant`/`quadrant()` is the first-class implementation, and `abstention_accuracy`
is the metric. The bottom row is exercised on every instance: `procedural_scaffold` reserves one
forced-ambiguous driver slot, so non-identifiability comes from **orientation ambiguity** (an unoriented
confounder edge), discovered exactly via `unoriented_edges`.

- **Ruled out, not deferred:** the latent/unobserved-confounder scaffold this section originally proposed
  as the near-term build. Unobserved confounding is undetectable from data, so naming could never
  *legitimately* resolve it, and the bottom row would test credulity rather than reasoning. Orientation
  ambiguity gives the same non-identifiability while staying detectable from the data — the agent can know
  it is stuck. See `procedural_scaffold_plan.md` §0/§9.
- **Roadmap still open:** finer α grids toward a smooth dose-response curve (deliberately given up in
  §11.8 — would have to be re-argued, not just re-enabled); seeded intra-level subset diversity; more
  gadget archetypes (colliders); discrete/mixed-type data (backlog §3).
