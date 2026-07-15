# CausalAgentBench - Stage 1 Plan: Task Interface (presentation plumbing + answer contract)

> **Audience: the implementing coding agent (e.g. Codex).** Authoritative, self-contained spec for
> **Stage 1**. Builds on the merged Stage 0 generator. Implement exactly this; do not add scoring, an
> agent runner, framework adapters, or the semantic labeler (deferred - see below). Flag anything that
> seems wrong rather than silently deviating.
>
> Package: `causalagentbench` (src layout, Python 3.12, uv). Same toolchain as Stage 0
> (ruff + mypy strict + pytest); line-length 120; concise ASCII docstrings.

---

## 1. Goal

Turn a hidden `SCM` (+ a sampled data table) into **what a tool-using agent sees and returns**, as
plumbing only:

* **Presentation** - a labeled data table + a task prompt, behind a `Labeler` **seam**.
* **Answer contract** - a minimal, validated Pydantic schema the agent emits, plus a parser.

## 2. Locked decisions

- **Chosen regime is semantic names, no graph** - but **generating semantic names is deferred**: it
  must be LLM-delegated and is designed/implemented as a separate step. Stage 1 ships only the labeler
  **seam** plus an **abstract placeholder** labeler (`X1..Xk`, randomly assigned, carrying no role
  info) so the interface is buildable and testable now.
- **Answer schema: minimal core** - per driver `identifiable` + `adjustment_set`; overall `ranking`.
  No effect estimates or confidence.
- **Encoding**: Pydantic model as the single source of truth; the agent emits JSON inside an
  `<answer>...</answer>` block; parsing always validates (never assume compliance).
- **Public/hidden split**: `Instance` (public) exposes no roles/graph/SCM; `InstanceKey` (hidden)
  holds the SCM + label<->node map for the future scorer.

## 3. Note on well-posedness

With the abstract placeholder, the presentation is name-blind, so the adjustment task is *not* yet
well-posed (from observational data alone you cannot separate a confounder from a mediator). That is
fine for Stage 1: there is **no scorer** here - this stage only builds the I/O plumbing. The task
becomes well-posed once the semantic labeler (the chosen regime) is added, and the gold answer always
derives from the hidden `SCM` regardless of labels.

## 4. Design

### 4.1 Labeler seam (`labeling.py`)

* **`Labeler = Callable[[Scaffold, np.random.Generator], dict[str, str]]`** - maps each node to a
  display label. This is the seam; the semantic (LLM-delegated) labeler and a later misaligned variant
  are sibling implementations with the same signature.
* **`abstract_labels(scaffold, rng) -> dict[node, label]`** - assigns `X1..Xk` to nodes at random.
  Carries no role information (no structure leak); does not itself make the task well-posed.

### 4.2 Instance and hidden key (`instance.py`)

* **`Instance`** (frozen, public - no roles/graph/SCM):
  - `columns: tuple[str, ...]` display labels in presentation order
  - `data: FloatArray` shape `(n, k)`, columns aligned to `columns`
  - `outcome: str`, `candidate_drivers: tuple[str, ...]`
  - `other_columns` (property): columns minus outcome minus candidate drivers (roles unknown to agent)
  - `prompt: str`; `to_csv() -> str`
* **`InstanceKey`** (frozen, hidden): `scm`, `label_to_node`, `node_to_label`.
* **`build_instance(scm, data, seed, *, labeler=abstract_labels) -> tuple[Instance, InstanceKey]`**:
  assigns labels, shuffles the display column order, permutes `data` to match, builds the prompt,
  returns the public/hidden pair. Deterministic in `seed`.
* **`build_prompt(columns, outcome, candidate_drivers) -> str`**: concise template stating the
  **question only** - the outcome and the candidate drivers to rank (per the proposal's input spec).
  It presents the remaining columns merely as *other columns in the data* and must **not** editorialize
  them as "covariates you may adjust for" (that would nudge the answer). The agent must itself discover,
  among the other columns, which are confounders (adjust) vs mechanisms (do not), and that a candidate
  driver may have ~zero effect. It instructs the agent to (per driver) decide identifiability + choose
  an adjustment set and to rank the identifiable drivers strongest-first, and specifies the exact
  `<answer>` JSON shape. Never reveals causal roles. Free reasoning/tools allowed; only the `<answer>`
  block is graded (later).

### 4.3 Answer contract (`answer.py`)

```python
class DriverVerdict(BaseModel):        # frozen, extra="forbid"
    driver: str
    identifiable: bool
    adjustment_set: tuple[str, ...]    # column labels; may be empty

class Answer(BaseModel):               # frozen, extra="forbid"
    verdicts: tuple[DriverVerdict, ...]
    ranking: tuple[str, ...]           # identifiable drivers, strongest first
```

* **`parse_answer(text) -> Answer`**: extract the JSON in the last `<answer>...</answer>` block and
  `Answer.model_validate_json(...)`; on missing block or invalid JSON/schema raise `AnswerParseError`.
  Validates **shape only** - instance-consistency (labels exist, ranking subset of identifiable
  drivers, bad-control checks) belongs to the Stage 2 scorer.

## 5. Module layout (additions to `src/causalagentbench/`)

```
labeling.py     Labeler seam + abstract_labels placeholder
instance.py     Instance, InstanceKey, build_instance, build_prompt
answer.py       DriverVerdict, Answer, parse_answer, AnswerParseError
```

Export the public names from `__init__.py`. Reuse Stage 0 unchanged.

## 6. Out of scope (later / separate steps)

- **Semantic labeler** (the chosen regime) - LLM-delegated, designed and implemented separately.
- Scoring/verifier; misaligned labeling + the reasoning-vs-recitation metric; effect-estimate/confidence
  answer fields; multiple domains; agent runner / code sandbox / LLM client / tool schema; verifiers or
  Inspect adapters.

## 7. Verification (definition of done)

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src` clean; `uv run pytest` green.
- Tests:
  - `abstract_labels`: assigns `X1..Xk`, distinct, one per node, a deterministic bijection of the seed;
    no label reveals a role.
  - `Instance` is public-only (no `scm`/roles fields); `other_columns` = the two non-outcome,
    non-driver labels; `to_csv()` header equals `columns`, body is `(n, k)`.
  - `build_instance(seed)` deterministic; `candidate_drivers` has 3 labels; `label_to_node` is a
    bijection over columns; the display permutation maps data columns to labels correctly.
  - `parse_answer`: a valid `<answer>` JSON round-trips to `Answer`; the *last* block wins; missing
    block, malformed JSON, extra field, and missing field each raise `AnswerParseError`.
- Example: build one instance from `generate_gated(0)` + `sample`, print the prompt and CSV header.

## 8. Roadmap after Stage 1

Next (separate, careful design): the **semantic LLM-delegated labeler** (aligned scenarios). Then:
Stage 2 three-axis scorer (verify an `Answer` against `InstanceKey`); misaligned labeling + the
aligned-vs-misaligned metric; agent runner + verifiers adapter; more scaffolds + stratified scoring.

## 9. SOTA agent-benchmark alignment (validation findings)

Validated against the current agentic-benchmark literature (SWE-bench, WebArena, OSWorld, tau-bench,
REAL, BrowserGym; the Agentic-RL survey). Framing: agentic benchmarks are **partial-observability,
tool-using decision processes**, not question-answer tests, and "the environment is the product".

Where CausalAgentBench already leads: **deterministic verification** (exact closed-form SCM verifier -
stronger than SWE-bench's tests), **reset/reproducibility** (seeded `generate`, unlimited fresh
instances, zero contamination), and **failure decomposition** (the three-axis score isolates
identification vs execution vs calibration). Keep the exact single-artifact verifier - it is the crown
jewel; do not trade it for interface realism.

Requirements this implies for the **harness stage** (record here so it is not lost; artifact-grading and
a multi-turn loop are *not* in tension - SWE-bench is exactly a multi-turn agent graded on a final
patch):

1. **Multi-turn tool-use loop, not a single `chat()` call.** The default agent runs a ReAct-style loop
   against a code sandbox + Layer-2 causal tools (`backdoor_effect`, `ci_test`, `refute`, `bootstrap_ci`)
   and emits the `<answer>` artifact at the end. The Stage 1 prompt is already turn-agnostic for this.
2. **Trajectory / telemetry logging as a first-class output.** Log tool calls, intermediate states, and
   retries per episode - needed both for "why did it fail" analysis and for RL credit assignment
   (outcome-only reward is too sparse; cf. observation-level supervision).
3. **Reliability metric (`pass^k`).** Agent + sampling are stochastic; report `pass^k` / variance across
   seeds, not just the mean, to keep leaderboards from being noise.
4. **Task-manifold breadth.** One scaffold + one domain is deliberately narrow for now; the motif bank
   (colliders, instruments, latents) and additional domains are needed for the paper's generality claim.

Explicitly **N/A** (do not cargo-cult web/computer-use design): GUI manipulation, visual-spatial
precision, streaming interaction, and hour-long / hundreds-of-call horizons. This is a
reasoning-with-tools task (peer class: CLADDER / Corr2Cause / CausalDS), so episodes should stay
short-to-medium horizon.
