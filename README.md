# CausalAgentBench

A synthetic causal-reasoning benchmark on procedurally-composed, exactly-solvable graphs.

Data is generated from a **hidden linear-Gaussian structural causal model (SCM)**, so the SCM is the
exact answer key. An agent sees a labeled observational table and must rank the candidate drivers of an
outcome by causal effect and choose a valid backdoor adjustment set for each. The SCM itself never
changes what's identifiable or what the gold answer is — what varies is whether the column **names**
agree with, are neutral to, or actively mislead about the true causal structure. That's the axis the
benchmark measures: does the agent reason from the data, or parrot the name it was given?

Every instance composes its own topology from a small gadget catalog (clean, confounded, multi-confounder,
and instrumented drivers, plus confounder-fed decoys), so driver count, archetype mix, and per-driver
identifiability all vary by seed. Exact effects come from `τ = (I − W)⁻¹ − I`; UUMC weight sampling keeps
marginal variance free of causal-order signal; driver effects, mediated shares, and confounding margins are
*constructed* rather than sampled-and-hoped-for; and a reject loop's soundness gates guarantee every
instance requires genuine causal reasoning (naive correlation *and* "adjust-for-everything" both provably
fail). Each instance is guaranteed at least one driver whose confounder edge is unresolvable from data
alone and one that is resolvable — which is what makes the naming axis load-bearing rather than decorative.
See [`backlog.md`](backlog.md) for open design questions and known characteristics of this setup (in
particular §4, on the interpretation floor that identifiability ambiguity imposes on any results write-up).

## Quickstart

```bash
uv sync                      # create the environment
uv run pytest                # run the test suite
uv run causalagentbench-smoke   # generate gated instances and print acceptance/gate stats
```

## Using the benchmark

Building an instance needs nothing beyond this package — no LLM, key, or download:

```python
from causalagentbench import generate_gated_procedural, build_instance, score_output

scm, report = generate_gated_procedural(seed=0)   # a sound instance: composed topology + gated effects
instance, key = build_instance(scm, report.data, seed=0)   # abstract labels (X1..Xk) by default

instance.prompt            # what the agent sees (context + schema + question)
instance.to_csv()          # the data table
key.scm.true_ranking()      # hidden gold ranking, computed on the SCM (label-invariant)

# ... an agent reads instance.prompt/instance.to_csv() and replies with an <answer> block ...
score = score_output(agent_reply_text, key)
score.ranking_tau           # Kendall agreement with the true ranking (correct abstentions excluded)
score.adjustment_accuracy   # fraction of graded drivers with a valid backdoor set
score.abstention_accuracy   # fraction whose abstain/commit choice matches the gold rule
```

`report.data` is the exact table the oracle gate certified — reuse it rather than drawing a fresh sample,
since gate acceptance is tied to that one draw (`procedural_scaffold_plan.md` §12.2).

## Generating naming-alignment data (optional)

Naming is done by an LLM: a generator proposes, per node, aligned / neutral / misaligned name variants in
one coherent CauseNet-grounded domain (an optional critic verifies them), and the alignment level is then
*constructed* from those variants — this is the manipulated axis described above. This path needs the
`generation` extra, an LLM key, and a CauseNet dump — three one-time steps:

```bash
# 1. install the extra
uv sync --extra generation

# 2. set up credentials + data path
cp .env.example .env         # then edit .env: add your API key(s) and the CauseNet path
#    keys are read by provider, e.g. model "anthropic/claude-opus-4-8" uses ANTHROPIC_API_KEY

# 3. download CauseNet grounding (causenet-precision.jsonl.bz2) from https://causenet.org
#    and set CAUSENET_DUMP in .env to its path
```

Naming alignment is a fixed discrete grid `{−1, 0, +1}` — misaligned / neutral / aligned, applied
uniformly to every non-outcome node — plus a name-blind control, so a full sweep is 4 views per SCM. It is
**constructed** per instance from per-node aligned / neutral / misaligned name triples — one generator
call (+ an optional correctness critic) per SCM. `alpha_sweep` builds every level of one SCM from a
single proposal (paired skins, `.env` loaded automatically):

```python
import os
from causalagentbench.generation import alpha_sweep, GRID

skins = alpha_sweep(
    seed=7, levels=GRID,
    gen_model="openai/gpt-5",
    causenet_dump=os.environ["CAUSENET_DUMP"],
    critic_model="openai/gpt-5",              # correctness critic; None to skip
)
for level, instance, key in skins:            # same SCM/data, only names change across levels
    key.provenance.alignment                  # the constructed level in {-1,0,+1}
```

Generate a batch of instances offline — models and size come from a versioned build config
(`configs/testset_v1.yaml`), so a run is reproducible (each SCM → all grid levels + name-blind):

```bash
uv run python scripts/build_testset.py --config configs/testset_v1.yaml --out outputs/my_run
```

There is currently no packaged/shipped fixed test set — every dataset is generated on demand, either
abstractly (free, above) or via this CauseNet+LLM path. `load_dataset` reads a build's output back into
`(Instance, InstanceKey)` pairs (no LLM calls, no extra needed) — build once, then evaluate several
models against the identical frozen instances via `scripts/run_eval.py --dataset outputs/my_run`.

## Evaluating a tool-using agent

`causalagentbench.agent` is a minimal tool-using agent: a fixed, safe causal toolset (`preview`,
`correlate`, `regress` — no arbitrary code execution, so there is no sandboxing surface beyond validating
column names) plus a model-agnostic tool-calling loop. The agent never sees the raw table in its prompt;
it must call a tool to learn anything about the data, then answer:

```python
from causalagentbench import generate_gated_procedural, build_instance
from causalagentbench.agent.clients import LiteLLMClient      # needs the `generation` extra + an LLM key
from causalagentbench.agent.eval import run_evaluation, aggregate_by_alignment

instances = []
for seed in range(5):
    scm, report = generate_gated_procedural(seed)
    instances.append(build_instance(scm, report.data, seed=seed))

scores = run_evaluation(instances, LiteLLMClient(model="anthropic/claude-opus-4-8"))
for summary in aggregate_by_alignment(scores):   # mean ranking_tau / adjustment_accuracy per alignment level
    print(summary)
```

Or from the CLI:

```bash
uv run python scripts/run_eval.py --model anthropic/claude-opus-4-8 -n 5
```

For a comparable multi-model sweep, use one frozen, non-empty dataset and pass the LiteLLM model IDs as a
comma-separated list. Every model is evaluated sequentially on exactly the same SCMs and naming conditions;
the JSONL output is flushed after every model call, instance, and model summary so an interrupted run retains
completed work:

```bash
uv sync --extra generation
uv run python scripts/build_testset.py --config configs/testset_v1.yaml --out outputs/my_testset
uv run python scripts/run_eval.py \
  --dataset outputs/my_testset \
  --models openai/gpt-5,anthropic/claude-opus-4-8 \
  --max-turns 20 \
  --out outputs/litellm_eval_results.jsonl
```

LiteLLM selects credentials from the model prefix (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and so on) in
`.env`; this evaluation path does not use Databricks credentials or the Databricks client. `--model` remains
available for a one-model run. With `--dataset`, `-n` selects approximately that many instances while keeping
the full paired naming-condition set for each selected SCM together.

For an `openrouter/...` model ID, LiteLLM sends the request through OpenRouter and charges the OpenRouter
account identified by `OPENROUTER_API_KEY`; LiteLLM itself is only the client library. The default
`--temperature 0 --sampling-seed 0` and a stable per-model OpenRouter session keep a multi-turn run
reproducible as far as the provider permits. Override the session namespace with
`--openrouter-session-prefix experiment-42` when intentionally separating routing experiments.

For full request/response observability, add `--transcript-out` with a different JSONL path. It writes one
record per model call, including the system and user messages, tool schemas, previous tool results, returned
text/tool calls, token counts, and LiteLLM's cost estimate. Treat this file as sensitive run data and do not
commit it:

```bash
uv run python scripts/run_eval.py \
  --model openrouter/meta-llama/llama-3.3-70b-instruct \
  -n 1 --max-turns 3 \
  --out outputs/llama_metrics.jsonl \
  --transcript-out outputs/llama_transcript.jsonl
```

When a model stops without a syntactically valid `<answer>` block, the agent makes one additional tool-free
formatting repair request. A valid answer, regardless of its score, is never retried; the repair is one extra
model call and is included in the run's cost total.

Every `model_summary` record includes the evaluation configuration, final `estimated_cost_usd`, and the
number of successfully scored and provider-failed instances. A provider failure is written as an
`instance_error` record and excluded from score averages, rather than being reported as a parse failure.

Alongside `summaries` (per naming-alignment level, one row per instance), each `model_summary` carries
`driver_cells`: the same run pooled at *driver* grain, crossing alignment with whether a driver's effect is
identifiable from data at all — so a failure on a driver the data cannot resolve is no longer averaged
together with one it can. Each instance record also keeps raw per-driver rows under `drivers`, so a
completed run can be re-sliced offline without re-running it.

For a fast provider smoke comparison without building names, omit `--dataset`; one generated abstract-label
batch is shared by every model in that invocation:

```bash
uv run python scripts/run_eval.py \
  --models openai/gpt-5,anthropic/claude-opus-4-8 \
  -n 5 \
  --out outputs/litellm_smoke_results.jsonl
```

`DatabricksClient` (same module) is a second `ToolCallingClient`, for comparing against a model served
behind a Databricks AI Gateway — not part of the default path above; needs `uv sync --extra databricks`
plus `DATABRICKS_TOKEN` + `DATABRICKS_HOST` in `.env` (see `.env.example`):

```python
from causalagentbench.agent.clients import DatabricksClient
scores = run_evaluation(instances, DatabricksClient(model="databricks-claude-opus-4-7"))
```

`run_agent`/`run_evaluation` take any `ToolCallingClient` (a tiny `Protocol`), so a stub client with no
network drives the loop's own tests — see `tests/agent/`.

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy src              # strict type check
```

## Layout

```
src/causalagentbench/
  types.py          Role, Scaffold, SCM — the shared domain vocabulary
  config.py         GeneratorConfig
  smoke.py          acceptance-rate / gate-statistics report
  scm/              build & simulate the hidden SCM
    gadgets.py        the gadget catalog + procedural_scaffold (a fresh topology per seed)
    identifiability.py  CPDAG/Meek-rule discovery of which edges data alone cannot orient
    generate.py       UUMC structural generation (unit-variance, neutral sortability)
    target_effects.py direct construction of driver effects, mediated shares, confounding margins
    effects.py        total-effect matrix τ and marginal covariance Σ
    sample.py         draw a data table from an SCM
    sortability.py    varsortability / R²-sortability diagnostics
  task/             the agent-facing interface
    instance.py       Instance (public) / InstanceKey (hidden) + prompt
    labeling.py       Labeler seam + abstract-label placeholder
    answer.py         the Pydantic Answer contract + parser
    dataset.py        load a pre-built dataset back into (Instance, InstanceKey) pairs
  scoring/          verification against the hidden SCM
    gates.py          soundness gates + the construct-then-gate generator
    baselines.py      oracle / correlation / full-adjustment reference solvers
    metrics.py        ranking metrics (Kendall-τ)
    score.py          score an agent's Answer against the hidden SCM; Quadrant classification
  generation/       optional, LLM-backed naming (needs the `generation` extra)
    grounding.py      CauseNet dump loading + role-candidate menu sampling
    labeler.py        propose → critique → revise naming triples; construct the alpha grid
    clients.py        provider-agnostic LLM seam (LiteLLM default)
    config.py         versioned build configuration
  agent/            a minimal tool-using agent
    tools.py          the fixed causal toolset (preview, correlate, regress)
    loop.py           model-agnostic tool-calling loop (ToolCallingClient Protocol)
    clients.py        the real LiteLLM-backed ToolCallingClient (needs the `generation` extra)
    eval.py           batch the loop over instances; aggregate scores by naming alignment
```

Tests mirror this layout under `tests/{scm,task,scoring,generation,agent}/`.

## Design docs

Four docs, each still accurate as of the last commit:

* [`procedural_scaffold_plan.md`](procedural_scaffold_plan.md) — **the authoritative account of how
  instances are generated**: the gadget catalog, identifiability discovery, direct effect construction,
  the gate loop, and the abstain/quadrant scoring rules. Read this first.
* [`variance_budget_plan.md`](variance_budget_plan.md) — why mediated effects and confounding margins scale
  with driver count. §6 is the one **open** item in these docs: the target-effect ladder still grows as
  `√n`, which is what caps usable driver counts at ~9 (n=10 accepts at 4.8%). Fixing it is coupled to a
  sample-size raise (`N ≳ n³`), not an independent change.
* [`semantic_labeler_plan.md`](semantic_labeler_plan.md) — the naming layer: three poles per node, the
  propose→critique→revise loop, and the constructed α grid.
* [`realistic_data_presentation_plan.md`](realistic_data_presentation_plan.md) — why the data is
  standardized-and-disclosed rather than dressed in realistic per-node ranges.

`backlog.md` is the living record of open design questions and known characteristics — read it before
extending the SCM/naming/scoring layers. `experiments/` holds dated write-ups of measured runs; note that
anything measured before the variance-budget change (2026-07-31) was generated by a different construction
and is not comparable to current output.

License: Apache-2.0.
