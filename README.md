# CausalAgentBench

A synthetic causal-reasoning benchmark on one fixed, exactly-solvable graph.

Data is generated from a **hidden linear-Gaussian structural causal model (SCM)**, so the SCM is the
exact answer key. An agent sees a labeled observational table and must rank the candidate drivers of an
outcome by causal effect and choose a valid backdoor adjustment set for each. The SCM itself never
changes what's identifiable or what the gold answer is — what varies is whether the column **names**
agree with, are neutral to, or actively mislead about the true causal structure. That's the axis the
benchmark measures: does the agent reason from the data, or parrot the name it was given?

The current scaffold is a single canonical 6-node graph with an engineered confounder, a mediator
(bad-control trap), and a wired-zero decoy; exact effects via `τ = (I − W)⁻¹ − I`; UUMC weight sampling
so marginal variance carries no causal-order signal; and a reject loop whose soundness gates guarantee
every instance requires genuine causal reasoning (naive correlation *and* "adjust-for-everything" both
provably fail). See [`backlog.md`](backlog.md) for open design questions and known characteristics of
this setup (in particular §4, on what is and isn't identifiable from data alone on this graph).

## Quickstart

```bash
uv sync                      # create the environment
uv run pytest                # run the test suite
uv run causalagentbench-smoke   # generate gated instances and print acceptance/gate stats
```

## Using the benchmark

Building an instance needs nothing beyond this package — no LLM, key, or download:

```python
from causalagentbench import generate_gated, sample, build_instance, score_output

scm, _ = generate_gated(seed=0)          # a sound instance, gated for real confounding/mediation
data = sample(scm, n_samples=1000, seed=0)
instance, key = build_instance(scm, data, seed=0)   # abstract labels (X1..Xk) by default

instance.prompt            # what the agent sees (context + schema + question)
instance.to_csv()          # the data table
key.scm.true_ranking()      # hidden gold ranking, computed on the SCM (label-invariant)

# ... an agent reads instance.prompt/instance.to_csv() and replies with an <answer> block ...
score = score_output(agent_reply_text, key)
score.ranking_tau           # Kendall agreement with the true ranking
score.adjustment_accuracy   # fraction of drivers with a valid backdoor set
```

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

Naming alignment is a fixed discrete grid `{−1, −0.5, 0, +0.5, +1}` (plus a name-blind control),
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
    key.provenance.alignment                  # the constructed level in {-1,-0.5,0,+0.5,+1}
```

Generate a batch of instances offline — models and size come from a versioned build config
(`configs/testset_v1.yaml`), so a run is reproducible (each SCM → all grid levels + name-blind):

```bash
uv run python scripts/build_testset.py --config configs/testset_v1.yaml --out outputs/my_run
```

There is currently no packaged/shipped fixed test set — every dataset is generated on demand, either
abstractly (free, above) or via this CauseNet+LLM path.

## Evaluating a tool-using agent

`causalagentbench.agent` is a minimal tool-using agent: a fixed, safe causal toolset (`preview`,
`correlate`, `regress` — no arbitrary code execution, so there is no sandboxing surface beyond validating
column names) plus a model-agnostic tool-calling loop. The agent never sees the raw table in its prompt;
it must call a tool to learn anything about the data, then answer:

```python
from causalagentbench import generate_gated, sample, build_instance
from causalagentbench.agent.clients import LiteLLMClient      # needs the `generation` extra + an LLM key
from causalagentbench.agent.eval import run_evaluation, aggregate_by_alignment

instances = []
for seed in range(5):
    scm, _ = generate_gated(seed)
    data = sample(scm, n_samples=1000, seed=seed)
    instances.append(build_instance(scm, data, seed=seed))

scores = run_evaluation(instances, LiteLLMClient(model="anthropic/claude-opus-4-8"))
for summary in aggregate_by_alignment(scores):   # mean ranking_tau / adjustment_accuracy per alignment level
    print(summary)
```

Or from the CLI:

```bash
uv run python scripts/run_eval.py --model anthropic/claude-opus-4-8 -n 5
```

`DatabricksClient` (same module) is a second `ToolCallingClient`, for a model served behind a Databricks
AI Gateway (OpenAI-compatible; needs `DATABRICKS_TOKEN` + `DATABRICKS_HOST` in `.env`, see `.env.example`):

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
    scaffold.py       the canonical 6-node role graph
    generate.py       UUMC structural generation (unit-variance, neutral sortability)
    effects.py        total-effect matrix τ and marginal covariance Σ
    sample.py         draw a data table from an SCM
    sortability.py    varsortability / R²-sortability diagnostics
  task/             the agent-facing interface
    instance.py       Instance (public) / InstanceKey (hidden) + prompt
    labeling.py       Labeler seam + abstract-label placeholder
    answer.py         the Pydantic Answer contract + parser
  scoring/          verification against the hidden SCM
    gates.py          soundness gates + rejection-sampling generator
    baselines.py      oracle / correlation / full-adjustment reference solvers
    metrics.py        ranking metrics (Kendall-τ)
    score.py          score an agent's Answer against the hidden SCM
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

`backlog.md` is the living record of open design questions and known characteristics — read it before
extending the SCM/naming/scoring layers. `stage0_plan.md`, `stage1_plan.md`, `semantic_labeler_plan.md`,
and `orientation_ambiguity_plan.md` are historical specs from earlier design stages (the last one
describes an abstention scaffold that was built, then deliberately cut from v1 scope — kept for context,
not as an active plan).

License: Apache-2.0.
