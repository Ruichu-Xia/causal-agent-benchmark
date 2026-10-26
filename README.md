# CausalAgentBench

A benchmark and RL environment for one skill: estimating causal effects by combining **statistical
evidence** with **knowledge of what variables mean**.

An agent gets 10,000 rows of observational data whose columns are named after real concepts (from
[CauseNet](https://causenet.org/)), an outcome, and five candidate drivers. Using three statistical tools,
it reports the signed total causal effect of each candidate. A hidden linear-Gaussian causal model gives
the exact answer.

Neither source of knowledge suffices alone:

- **Data alone** cannot orient every edge. Graphs in the same Markov equivalence class produce identical
  data, and for some candidates they imply different effects. Every task has at least one such
  *name-critical* candidate.
- **Names alone** cannot give the numbers. Coefficient signs and sizes are synthetic, and the prompt says so.

A scripted pipeline (PC discovery, then regression) can settle the data-identified candidates, but it has
to guess on the name-critical ones. An agent that reads `smoking` and `yellow_teeth` knows which way the
arrow points.

## Quick start

Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra agents --extra discovery

# Build a split locally (deterministic, no API key). Needs the CauseNet-Precision dump.
uv run causalagentbench build --causenet causenet-precision.jsonl.bz2 --split dev -n 50 --workers 4 --out artifacts/dev

# Free reference baselines: zero, unadjusted, adjust_all, pc, data_only_best, oracle
uv run causalagentbench baseline --dataset artifacts/dev --agent pc --out outputs/dev_pc.jsonl

# A tool-calling model through LiteLLM (set the provider key in .env)
uv run causalagentbench eval --dataset artifacts/dev --model "$MODEL" --out outputs/dev_model.jsonl --workers 4

# The names ablation: same tasks and data, columns renamed X1..Xk
uv run causalagentbench eval --dataset artifacts/dev --model "$MODEL" --placeholder-names --out outputs/dev_model_x.jsonl

uv run causalagentbench report outputs/dev_model.jsonl                                   # headline + 95% CI
uv run causalagentbench report outputs/dev_model.jsonl --ablation outputs/dev_model_x.jsonl  # effect of names
uv run causalagentbench report outputs/dev_model.jsonl --compare outputs/dev_pc.jsonl        # paired difference
```

`eval` appends to its results file only with `--resume`, which checks that the run configuration matches
and skips finished tasks; provider failures are recorded and retried on resume, never scored. Full
conversations go to `<out>.transcripts.jsonl`. `rescore` re-scores saved answers with the current scorer.
Every results file starts with a header recording the protocol, dataset hash, git commit, package versions,
budgets, request timeout and sampling settings (provider defaults). Each episode records tokens per turn,
including cached and reasoning tokens when the provider reports them. Slow or local models need a longer
`--timeout` (seconds per request; default 120).

## The task

The prompt lists the columns, the outcome and the candidates, and states the assumptions: acyclic, linear,
Gaussian noise, all common causes observed, faithful, unit variances. It never reveals roles, edges or
adjustment sets. Tools (protocol `lean-v2`: 30 calls and 7 replies, the last of which must be the answer;
a reply may carry several calls):

| Tool | Returns |
| --- | --- |
| `correlate(x, y, given)` | partial correlation with a 95% interval |
| `regress(treatment, adjust)` | OLS coefficient on the outcome with standard error |
| `discover(alpha)` | PC algorithm: oriented and undirected edges |

The budget is set by cost, not by need. In a pilot under the earlier budget (100 calls, 25 turns), the
number of calls did not predict error, while re-sent history and per-turn reasoning drove the bill. The
PC-then-regress recipe needs two investigation turns and reaches the same score through these tools.

The answer is one JSON block:

```json
{"verdicts": [{"driver": "<column>", "adjustment_set": ["<column>"], "effect": 0.23}]}
```

## Scoring

```text
error = Σ min(|e_i − t_i|, 1) / Σ |t_i|        over every candidate of every task
```

`e_i` is the estimate and `t_i` the true total effect in standardized units. Lower is better; predicting
zero everywhere scores exactly 1 (every |t| < 1). The cap bounds any single blunder so confidence
intervals stay valid. Absolute error rewards committing to the likelier direction rather than averaging.
Reports give a task-clustered 95% interval, the error on the **name-critical** and **data-identified**
slices, answer validity, the share of valid adjustment sets, tool calls, tokens and cost. Model
comparisons are paired by task.

The names ablation reports how much placeholder names raise the error on each slice and the difference
between the two (expected: large on name-critical, near zero on data-identified).

## How tasks are generated

1. **Motif graph.** Five candidates wired into one outcome from small patterns: clean cause; confounded
   cause with a mediator; collider and M-bias traps; zero-effect decoys; plus a child of the outcome.
   Some confounders get two extra parents, which lets the data orient them.
2. **Correlations.** A unit-variance linear-Gaussian model on the motif graph gives the correlation
   matrix Σ (per-parent weights bounded away from zero, random signs).
3. **True graph.** Drawn uniformly from the motif graph's Markov equivalence class, independently of Σ.
   Its weights come from regressing each node on its parents under Σ (*covariance transfer*), so the
   data are identical whichever equivalent graph is true: nothing in the data predicts an orientation
   the CPDAG leaves open.
4. **Gates.** Computed from Σ and the equivalence class only, never the true graph, so rejection cannot
   leak orientation either. They require a name-critical candidate and a data-identified one that needs
   adjustment; material differences between possible effects; no fixed recipe (adjust for nothing,
   everything, or the other candidates) within 25% error for any equivalent graph; tempting decoys;
   effects below 1; and detectable v-structures at 10,000 rows.
5. **Names.** Exact CauseNet concepts under four rules: every hidden edge `A → B` is a CauseNet relation
   `A → B`; pairs CauseNet states both ways are never used; no CauseNet relation between two chosen
   concepts points against the hidden causal order; no near-duplicate concepts (`infection`/`infections`,
   `diabetes`/`diabetes_mellitus`). No word lists, no curation, no language model. Names must match the true
   graph, so naming retries on the same graph rather than redrawing it; the audit finds no measurable effect
   on which orientations are kept.

`scripts/audit.py` measures gate acceptance, orientation leakage (a structure-only model against one that
also sees data statistics, on held-out tasks) and how well PC recovers each task's structure.

**Splits.** Seeds come from separate namespaces: `train`, `dev` (public), `test` (public, frozen), and
`private` (derived from a secret salt; `--salt-env`). Training tasks never use a CauseNet relation between
two concepts that appear together in a dev or test task. Dataset files carry a canary GUID.
**Dataset JSON files contain the hidden answers**; keep them in the evaluator process.

## Reference results

| Reference | All candidates | Name-critical | Data-identified |
| --- | ---: | ---: | ---: |
| Predict zero | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Unadjusted regression | 0.925 ± 0.065 | 0.897 ± 0.118 | 0.949 ± 0.088 |
| Adjust for every other column | 0.725 ± 0.026 | 0.814 ± 0.042 | 0.647 ± 0.032 |
| PC discovery → adjust for discovered parents | 0.515 ± 0.050 | 0.852 ± 0.111 | 0.224 ± 0.038 |
| Best data-only answer (true CPDAG, median over equivalent graphs)* | 0.262 ± 0.024 | 0.494 ± 0.046 | 0.062 ± 0.006 |
| Oracle: adjust for the true parents* | 0.062 ± 0.004 | 0.061 ± 0.007 | 0.063 ± 0.006 |

Headline error on the 200-task test split, with task-clustered 95% intervals (no model API calls;
`outputs/results/`). \*Reads the hidden key; a reference, not an agent. On data-identified candidates the
best data-only answer already matches the oracle; on name-critical candidates the best a name-blind
analyst can expect is about 0.49, while the oracle reaches 0.06. That gap is what reading the names can buy. PC's extra error on
data-identified candidates comes from finite-sample discovery mistakes (at 10,000 rows it recovers 86% of
true edges and orients 91% of data-determined edges correctly).

## Agent results

Lean-v2 protocol (30 tool calls, 7 turns), error with task-clustered 95% intervals; an invalid answer
scores as predicting zero. "Placeholder" is the names ablation (`--placeholder-names`).

| Agent | Split | Real names | Placeholder names | Valid answers (real / placeholder) |
| --- | --- | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 | test (200) | 0.740 [0.70, 0.78] | 0.747 [0.70, 0.79] | 65% / 88% |
| Qwen3-4B-Instruct-2507 + GRPO (120 steps) | test (200) | 0.540 [0.50, 0.58] | 0.661 [0.62, 0.70] | 95% / 99% |
| Qwen3-4B-Instruct-2507, names only (no tools) | test (200) | 1.354 [1.28, 1.43] | | 56% |
| DeepSeek-V4.1-Flash | dev (50) | 0.428 [0.35, 0.51] | 0.527 [0.45, 0.60] | 100% / 100% |
| DeepSeek-V4.1-Flash, names only (no tools) | dev (50) | 1.243 [1.12, 1.37] | | 100% |
| PC discovery → adjust for discovered parents | dev (50) | 0.60 | | |

The names ablation measures domain knowledge: for DeepSeek-V4.1-Flash, hiding names raises error on
name-critical candidates by 0.19 and on data-identified ones by 0.02, a difference in differences of
+0.17 [+0.03, +0.34]. The base 4B model shows no such effect on test (−0.02 [−0.11, +0.08]). Names without
data are worse than predicting zero for both models.

GRPO lowers the 4B model's test error by 0.20 [0.15, 0.25] with real names and by 0.09 [0.04, 0.13] with
placeholders (paired by task); the gain is 0.11 [0.05, 0.18] larger when names are visible. It is not only
validity: on the 123 tasks both models answered validly with real names, error falls by 0.12 [0.06, 0.17].
Training used real names only, so the placeholder arm is also out of the training distribution.

## RL environment

`CausalEnv` follows TRL's `GRPOTrainer(environment_factory=...)` contract: `reset(**row)` returns the
task prompt and the three tools are its public methods, with the remaining budget appended to each result
as in evaluation. The episode ends when the model replies without a tool call; `reward_func` scores that
final reply's `<answer>` block exactly as evaluation does, so training and evaluation share one prompt and
one answer format.

```python
from causalagentbench.environment import CausalEnv, reward_func
from causalagentbench.task.dataset import load_dataset

# causalagentbench build --split train -n 1000 --exclude artifacts/dev artifacts/test --out artifacts/train ...
tasks = load_dataset("artifacts/train")
trainer = GRPOTrainer(..., train_dataset=Dataset.from_dict({"prompt": [...], "seed": list(range(len(tasks)))}),
                      environment_factory=lambda: CausalEnv(tasks), reward_funcs=reward_func)
```

Training tasks avoid every CauseNet relation between two concepts that appear together in an evaluation
task. The reward is `1 − min(task error, 2)` for a valid answer and `−1` otherwise, including an episode
that ends without one; predicting zero earns 0, and failing on purpose never pays.

`scripts/train_grpo.py` trains a LoRA adapter (r 32) on Qwen3-4B-Instruct-2507 with TRL 1.16 GRPO: 8 tasks
× 8 rollouts per step, learning rate 1e-5, vLLM colocated generation. `scripts/modal_grpo.py` runs it on one
Modal H100 (120 steps, about 4.8 hours); `scripts/rl_report.py` plots the curves and checks for reward
hacking. Over the run, mean reward rose from 0.11 to 0.32 (first versus last 10 steps), valid answers from
82% to 98%, no group had zero reward variance, under 0.1% of answers predicted zero everywhere, and 3.6% of
answers reported an effect that no `regress` call had returned, a share flat across training.

## Development

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check src tests scripts && uv run mypy src
```

Code is Apache-2.0. CauseNet-derived data is CC BY 4.0; see [data/README.md](data/README.md).
