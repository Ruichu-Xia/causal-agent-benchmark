# CausalAgentBench

A small benchmark and RL environment for causal analysis with statistical tools.

An agent receives an observational dataset, an outcome, and five candidate drivers. It chooses an
adjustment set and estimates the signed **total causal effect** of each driver. The headline metric is
normalized absolute estimation error (lower is better). RL uses a separate bounded reward.
A hidden linear-Gaussian structural causal model supplies the exact answer key.

**Names come directly from CauseNet. Every simulated directed edge must match an actual directed
relation in the source dump.** No LLM generates names. The agent sees concept names and data, with no
relationship descriptions, graph edges, roles, or adjustment hints. CauseNet relations are used only
by the offline task builder; they are not a tool available to the agent.

## Try it

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required for these commands.

```bash
uv sync --extra agents --extra discovery

# Free: run a data-only PC → adjustment → regression baseline on the included 10-task dataset.
uv run causalagentbench baseline --dataset data/example --out outputs/pc.jsonl

# Evaluate a tool-calling model using its LiteLLM provider/model identifier.
# Set the provider's API key in your environment or .env, and set MODEL to the desired identifier.
uv run causalagentbench eval --dataset data/example --model "$MODEL" --out outputs/model.jsonl
```

Each result file contains a run configuration and dataset checksum, one record per completed task,
then a summary. Results are flushed as tasks finish. Invalid answers count as zero-effect predictions in
the reported error; provider failures leave the run incomplete with no headline metric and a nonzero exit
code. Output files are never overwritten.

Use `--limit 2` for a quick check and `--workers 4` for concurrent episodes. Repeat `--model` to compare
models on the same tasks. `--max-tool-calls` and `--max-turns` control the investigation budget; defaults
are 100 calls and 25 model replies, plus at most one tool-free formatting repair. The agent is told its
remaining budgets. Token usage, runtime and available cost estimates are recorded.

Three other free references are available:

```bash
uv run causalagentbench baseline --dataset data/example --agent zero --out outputs/zero.jsonl
uv run causalagentbench baseline --dataset data/example --agent correlation --out outputs/correlation.jsonl
uv run causalagentbench baseline --dataset data/example --agent oracle --out outputs/oracle.jsonl
```

`zero` reports zero for every effect without inspecting the data. `correlation` uses unadjusted regression.
`oracle` uses hidden correct adjustment sets, but estimates
coefficients from the sample; it is an estimation reference, not a competing agent. The PC baseline
uses the same discovery configuration as the agent tool and excludes unresolved neighbors.

Measured on all ten bundled tasks with `normalized-error-v1` reporting (no model API calls):

| Reference | Normalized effect error ↓ | Effect MAE ↓ |
| --- | ---: | ---: |
| Always predict zero | 1.000 | 0.1481 |
| Unadjusted regression | 0.867 | 0.1283 |
| PC discovery → regression | 0.722 | 0.1069 |
| Oracle adjustment → regression | 0.123 | 0.0183 |

The oracle uses finite samples, so its effects are not exact and its error is not a strict lower bound.
These are small-sample reference results, not evidence of language-model performance.

## Build more tasks

Download [CauseNet-Precision](https://causenet.org/), then:

```bash
uv run causalagentbench build \
  --causenet /path/to/causenet-precision.jsonl.bz2 \
  --out outputs/train --seed 1000 -n 100
```

Generation is local and deterministic. It needs neither an API key nor the `agents` extra. The dump is
loaded once per build. A bounded matcher assigns distinct source concepts to graph nodes and verifies
all edge directions. A task that cannot be grounded is retried with a new SCM; exhausting the retry
budget fails the build. There is no invented-name fallback or silent partial dataset.

The `.json` / `.npz` pair freezes the exact prompts, data, names, source hash and evaluator keys. Different
seed ranges can be used for training and evaluation. The included dataset is a small example, not a
claim of broad domain coverage. Its source attribution and build command are in [data/README.md](data/README.md).

**Dataset JSON files contain the hidden answer keys.** Keep them in the evaluator process; give an agent
only the public prompt and tool results. The Python object boundary is not a security sandbox for an agent
with unrestricted access to the evaluator's filesystem or process.

## Task, evaluation metric and reward

Available tools are `preview`, `correlate` (including partial correlation), `regress` (OLS with an
adjustment set and uncertainty), and `discover` (PC). Statistical tools use only the observed table.

The final answer is:

```json
{"verdicts": [
  {"driver": "<concept>", "adjustment_set": ["<concept>"], "effect": 0.23}
]}
```

A model wraps this JSON in `<answer>...</answer>`. Every candidate must appear exactly once. Effects
must be finite; adjustment sets cannot contain unknown columns, duplicates, the driver or the outcome.

**The headline metric is `normalized_effect_error`**, pooled over all tasks and candidates. With true
signed effects `t_i` and estimates `e_i`:

```text
normalized_effect_error = sum(|e_i - t_i|) / sum(|t_i|)
effect_mae = mean(|e_i - t_i|)
```

Zero means exact estimates, 1 equals the error of always predicting zero, and values above 1 are worse
than that reference. There is no clipping. Errors and true magnitudes are pooled before division; this
is not an average of per-task ratios. Larger-effect tasks therefore carry more weight. All candidates,
including zero-effect candidates, count. The denominator is positive on generated benchmarks; for an
external all-zero ground-truth dataset, normalized error is undefined and reported as null.

Report `effect_mae` (standardized effect units, lower is better) and `answer_validity_rate` alongside the
headline. An invalid answer replaces the whole task's estimates with zeros for both reported error
metrics. It remains in the denominator and reduces validity; it is never silently excluded. This gives
an invalid answer the same error as an explicit all-zero answer, not a correctness judgment.

Provider/runtime failures are separate from invalid answers. If any task fails, the summary's headline
error and MAE are null and the CLI exits nonzero; partial diagnostic averages remain available. Validity
is the fraction of all attempted tasks with a valid answer. A complete all-invalid run has normalized
error 1 and validity 0. Results record `reporting_version: "normalized-error-v1"`.

**RL reward is separate**, on `[0, 1]`, using each task's scale `s = median(|t_i|)`:

```text
reward = effect_score = 1 - mean(min(|e_i - t_i| / s, 1))
```

All candidates count, including zero-effect candidates. Generated tasks have positive `s`. Exact effects
earn 1; reward is partial credit, not percent correct or improvement over chance. Predicting zero earns
credit on zero and small effects, so compare against the zero baseline. Invalid answers earn zero reward.

The CLI also records this reward as a diagnostic. The RL scorer's MAE is null for invalid submissions;
evaluation applies the explicit zero-prediction policy above when building its reports.

Two diagnostics do **not** contribute to reward:

- `adjustment_score`: one minus normalized, clipped population regression bias from the proposed controls.
  This evaluates their induced bias, not an exact match to a preferred variable list.
- `ranking_tau`: pairwise agreement in absolute effect order on `[-1, 1]`; true ties are excluded and
  submitted ties receive zero ordering credit. Estimates can rank perfectly while having incorrect sizes or signs.

The RL rule remains `scoring_version: "effect-v1"`; older results used an adjustment/effect average and
their rewards are not directly comparable. The bundled prompts describe both the headline metric and
the RL reward. Previously frozen datasets retain their original prompts; rebuild those before new model runs.

The SCM is acyclic, linear and Gaussian, with independent noise and observed common causes. All columns
have population unit variance. Graphs combine clean, confounded, mediated, instrumented, collider and
M-bias patterns with zero-effect decoys. Each task has 2,000 rows. Generation gates ensure an oracle can
recover the ranking on the released table and three specified naive adjustment policies have material bias.

Some orientations are not identifiable from observational data alone; concept names provide domain
knowledge but do not guarantee unique identification. The score measures recovery of the selected
**synthetic** SCM, including effects that require semantic assumptions. It is not a pure test of statistical
identification. CauseNet supplies claimed
relations, not numerical effect sizes or a complete real-world causal graph. Additional relations in
CauseNet may be omitted from the selected subgraph. This benchmark does not measure real-world causal
effects or require instrumental-variable estimation.

## RL interface

No Gymnasium or training framework is required. `CausalEnv` exposes `reset` and `step`:

```python
from causalagentbench import CausalEnv, load_dataset

tasks = load_dataset("data/example")
env = CausalEnv(max_tool_calls=40)
observation, info = env.reset(task=tasks[0])

observation, reward, terminated, truncated, info = env.step({
    "type": "tool", "name": "preview", "arguments": {"n": 3}
})

# The policy chooses further tool actions, then submits its answer:
# env.step({"type": "submit", "answer": {"verdicts": [...]}})
```

Reset returns the public prompt, tool schemas and query budget. Each tool attempt, including an invalid
one, costs one query and gives zero reward. After the final query, a submission is still allowed. Another
tool attempt truncates the episode with zero reward. Submission always ends the episode; invalid answers
score zero. Calls after termination require a new reset. RL submissions have no automatic format repair.

For procedural training, load the source once:

```python
from causalagentbench import CausalEnv, CauseNetLabeler

labeler = CauseNetLabeler.from_path("/path/to/causenet-precision.jsonl.bz2")
env = CausalEnv(labeler=labeler)
observation, info = env.reset(seed=123)
```

Use `make_task(seed, labeler=labeler)` directly when you need the public instance and evaluator key.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run mypy src
```

The tests cover source-edge grounding, generation, exact effects, answer validation, scoring, tools,
episode budgets and termination, serialization, CLI output, and provider-error handling. No live API calls
are needed. Provider-specific tests need the optional SDKs installed by `--all-extras`.

Code is Apache-2.0. CauseNet-derived example data is CC BY 4.0; see [data/README.md](data/README.md).
