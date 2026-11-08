# CausalAgentBench

**An RL environment that trains LLM agents to answer causal questions from data, the way an analyst would.**

[![ci](https://github.com/Ruichu-Xia/causal-agent-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Ruichu-Xia/causal-agent-benchmark/actions/workflows/ci.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/hero-dark.png">
  <img alt="Six generated worlds with 4 to 8 variables named after real CauseNet concepts, drawn as causal graphs whose links raise or lower their effect, each with a scatter of real rows. 3,999 training worlds, 4 to 8 variables, 6,674 concepts, 1 to 36 graphs fitting one world's data." src="docs/figures/hero-light.png">
</picture>

Ask an LLM agent "does X affect Y?" about a dataset and it often takes a shortcut: it reads off a correlation,
or it guesses from what the variable names suggest. This environment rewards the steps a trained analyst
follows instead, and grades every answer exactly against a hidden causal world.

**RL training made a 4B model reliably do the analysis it could already sometimes do, and it could not teach the
step the model never got right.**

- Held-out accuracy rose from **43% to 63%**, and the gain held on new wording, bigger graphs, unseen names and
  causal structures from published studies.
- **Learned:** testing a relationship (63% → 96%) and finding its direction (45% → 64%).
- **Not learned:** choosing which variables to control for. For effect sizes, the model learned a shortcut.

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/episode-dark.png">
  <img alt="One episode: a hidden world decides the answer; the agent sees data.csv and one question, writes Python with up to 10 calls, and gives a final Answer line; a judge compares it with the true answer and returns reward 1 or 0." src="docs/figures/episode-light.png">
</picture>

- **Task:** a table of 2,000 rows from a simulated causal world, and one question about it.
- **Tool:** `python(code)` runs in a sandbox with the table loaded as `df`, plus pandas, scipy, statsmodels and
  three helpers. Up to 10 calls.
- **Judge:** reads only the last `Answer:` line of the final reply and returns 1 or 0. The true graph and answer
  never enter the sandbox.

## Four questions, one analyst workflow

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/workflow-dark.png">
  <img alt="Four steps: test a relationship (Yes or No, chance 51%), find the direction (A, B or C, chance 35%), choose the controls (Yes or No, chance 46%), estimate the effect (a number within 0.10, chance 5%)." src="docs/figures/workflow-light.png">
</picture>

Each task asks about one step, so the results show which steps the model learned. **Accuracy** is the share of
questions answered correctly. **Chance** is the score of giving the best single answer to every question; a
model has learned something only if it beats chance. Numbers count as correct within ±0.10 standard
deviations; a reply without an `Answer:` line scores 0.

## What training changed

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/steps-dark.png">
  <img alt="Accuracy by step, base then trained, against chance: step 1 63% to 96% (chance 51%), learned; step 2 45% to 64% (chance 35%), learned; step 3 42% to 47% (chance 46%), not learned; step 4 25% to 39% (chance 5%), a shortcut; on step 4 tasks where the shortcut is wrong, 5% to 3%." src="docs/figures/steps-light.png">
</picture>

RL strengthens behavior that already earns reward, and each row follows from that:

- **Steps 1 and 2:** the base model sometimes did the right analysis, so training made it the habit.
- **Step 3:** the base model was at chance. There was no correct behavior for RL to strengthen.
- **Step 4:** a lazy regression (Y on X alone, or on every other column) happened to be right on about half the
  tasks, so RL strengthened it. Where it is wrong, the trained model scores 3%.

This matches other findings that RL with right-or-wrong rewards mostly sharpens what a model can already do
([Yue et al., 2025](https://arxiv.org/abs/2504.13837)).

### Before and after, on one task

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/example-dark.png">
  <img alt="Held-out task iid_test-88. The base model rewrote partial_corr so that it dropped the columns it should hold fixed, measured the plain correlation and answered No (wrong). The trained model called the provided helper with the stated variables, got r = 0.134 and p = 1.5e-9, and answered Yes (right)." src="docs/figures/example-light.png">
</picture>

| Across the 500 held-out tasks | Base | Trained |
| --- | ---: | ---: |
| Answered correctly | 42.6% | **63.2%** |
| Rewrote a provided helper instead of calling it | 27.0% | **0.2%** |
| Ended without an `Answer:` line | 8.2% | **2.6%** |

Not every correct answer is this clean: the trained model almost never uses the `pc` helper for direction
questions (2 of 125 tasks).

## Does it generalize?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/transfer-dark.png">
  <img alt="Accuracy of base, control and trained models. Held-out test 43%, 38%, 63%. New wording 43%, 38%, 61%. Bigger graphs 44%, 36%, 61%. Unseen names 46%, 36%, 60%. Published causal studies 44%, 38%, 62%. Real QRData datasets 32%, 38%, 37%, not confirmed." src="docs/figures/transfer-light.png">
</picture>

- **The gain came from the reward.** A control model got identical training except that rewards were shuffled.
  It ended *below* the base model (38% vs 43%).
- **The tests were fixed in advance.** Test sets, comparisons and statistics were
  [pre-registered](docs/preregistration_v3.md) before any test score was computed. A transfer test counts as
  confirmed only if the trained model beats both base and control.
- **Real, messy data did not transfer.** On QRData the trained model beat base by 4 points, but the control
  did as well.

Training: Qwen3-4B-Instruct-2507 with LoRA (rank 32), 60 GRPO steps of 8 tasks × 8 attempts, 4.3 hours on one
H100.

## How a task is built

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/generation-dark.png">
  <img alt="Five steps: draw a random causal graph; find every graph the data cannot rule out; name the variables with CauseNet concepts; simulate 2,000 rows and ask one question; check that a standard method on the rows reaches the answer. Data alone and names alone each fall short; column order and lazy regressions still beat chance." src="docs/figures/generation-light.png">
</picture>

On real data nobody knows the true causal answer, so answers cannot be graded exactly. Simulated worlds give
an exact answer for every question and as many fresh tasks as training needs. The cost is realism: every
relationship is linear with Gaussian noise, there are no hidden common causes, and the graphs are small.

## Limitations

- **Step 4 rewards a shortcut.** A lazy regression is right on about half the effect tasks. Next: make most of
  them ones where both lazy regressions are wrong, or grade the controls too.
- **Step 3 was not learned.** Next: reward it directly, or start from a model that sometimes gets it right.
- **Column order leaks direction.** "The left column causes the right" is right on 48% of direction tasks
  (chance 35%). Gains hold where it is wrong, but the next build should shuffle columns.
- **Realism.** Linear, Gaussian, small graphs, no hidden causes; transfer to real datasets was not confirmed.
- **One training run per model**, and the checkpoint was chosen on development sets.

## Try it

Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

**Run one episode locally, no GPU needed.** The tool calls below are written by hand; model-written code should
run only inside the sandboxed containers.

```python
import json
from causalagentbench import DataEnv, reward_func

tasks = [json.loads(line) for line in open("data/v3_sample.jsonl")]
env = DataEnv(tasks)
print(env.reset(index=0))  # the prompt: a question about fatigue and distracted_drivers
print(env.python("partial_corr(df, 'fatigue', 'distracted_drivers', given=['stresses', 'auto_accidents'])"))
# {'r': -0.2798, 'p_value': 1.0e-37, 'n': 2000}  [9 of 10 tool calls left]
print(reward_func(["Conditioning on their common effect links them.\nAnswer: No"], [env]))  # [1.0]
```

<details>
<summary>Build the tasks, train and evaluate on Modal, reproduce the analysis</summary>

Building tasks needs the CauseNet-Precision dump from [causenet.org](https://causenet.org/).

```bash
uv run python scripts/build_tasks.py --out-dir outputs/dataenv/v3 --train 4000 --dev 200 --test 500 --workers 8
uv run python scripts/build_transfer.py --out-dir outputs/dataenv/transfer --per-family 12 \
    --qrdata <QRData's benchmark/ directory, with data/ unzipped>

uv run --with modal modal deploy scripts/modal_app.py
uv run --with modal modal run scripts/modal_app.py::isolation_main      # the sandbox must fail to read secrets
uv run --with modal modal run scripts/modal_app.py::train_main --dataset outputs/dataenv/v3/train.jsonl \
    --run data-pilot --max-steps 60 --extra "--revision <sha>"
uv run --with modal modal run scripts/modal_app.py::eval_main --tasks outputs/dataenv/v3/iid_test.jsonl \
    --name trained --revision <sha> --adapter data-pilot/adapter

uv run python scripts/confirm.py analyze --control v3-control
uv run python scripts/explore.py --control v3-control
```

</details>

<details>
<summary>Full confirmatory results</summary>

Accuracy per arm (reweighted on QRData causal so that no answer habit scores above chance). Differences are
trained − comparator in percentage points with 95% cluster-bootstrap intervals; an endpoint is confirmed when it
beats both comparators (one-sided sign-flip test, Holm over five endpoints at α = 0.025). bnRep questions are
answered twice and QRData questions four times. One training run per arm.

| Set | Base | Trained | Control | Trained − base | Trained − control | Result |
| --- | ---: | ---: | ---: | --- | --- | --- |
| iid test (manipulation check) | 0.426 | 0.632 | 0.376 | +20.6 [+15.4, +25.8] | | passes |
| Held-out wording | 0.428 | 0.612 | 0.380 | +18.4 [+13.6, +23.2] | +23.2 [+18.2, +28.2] | confirmed |
| Bigger graphs (7–8 variables) | 0.440 | 0.614 | 0.360 | +17.4 [+12.0, +22.8] | +25.4 [+20.4, +30.2] | confirmed |
| Unseen names | 0.462 | 0.600 | 0.364 | +13.8 [+8.8, +18.8] | +23.6 [+18.4, +28.8] | confirmed |
| bnRep, real names | 0.443 | 0.615 | 0.380 | +17.2 [+11.5, +22.9] | +23.4 [+16.4, +30.2] | confirmed |
| QRData causal | 0.323 | 0.367 | 0.382 | +4.4 [+0.5, +8.8] | −1.4 [−6.0, +3.1] | not confirmed |
| QRData statistics (exploratory) | 0.567 | 0.565 | 0.563 | | | |
| bnRep, placeholder names (exploratory) | 0.401 | 0.609 | 0.341 | | | |

Details, scope conditions and deviations: [docs/results_v3.md](docs/results_v3.md). Design:
[docs/environment_v3.md](docs/environment_v3.md). QRData ([Liu et al., 2024](https://github.com/xxxiaol/QRData))
is CC BY-NC 4.0, so it is used locally and never redistributed here.

</details>

<details>
<summary>How the project got here</summary>

| Version | Environment | What happened | Lesson |
| --- | --- | --- | --- |
| v1 | 10,000 rows, three fixed tools, report five signed effects | Error fell 0.20 in-domain; no confirmed transfer ([results](https://github.com/Ruichu-Xia/causal-agent-benchmark/blob/e2020db/docs/transfer_results.md)) | One rigid task and fixed tools train a narrow skill |
| v2 | Text only: a described graph, four question families | +1.9 points, not significant ([results](https://github.com/Ruichu-Xia/causal-agent-benchmark/blob/e2020db/docs/results_v2.md)) | Without data and a tool, there was little to learn |
| **v3** | Data, a general Python tool, four mixed question families | +20.6 in-domain; 4 of 5 transfer tests confirmed | Keep data and tools; mix question types; balance shortcuts |

This repository holds v3 only. The code and write-ups of v1 and v2 are kept in the history, at commit
[`e2020db`](https://github.com/Ruichu-Xia/causal-agent-benchmark/tree/e2020db).

</details>

<details>
<summary>Repository layout</summary>

```text
src/causalagentbench/
  environment.py   DataEnv: the episode (prompt, python tool, final answer) and its reward
  sandbox.py       the python tool: a restricted worker process per episode
  grading.py       reads the final "Answer:" line and grades it
  world.py         a simulated world: random graph, its equivalence class, names and 2,000 rows
  families.py      the four question families, with shortcut balancing and the answerability check
  wording.py       paraphrase templates, some held out of training
  dataset.py       builds and loads task splits
  graphs.py        d-separation, valid adjustment, equivalence classes
  scm.py           linear-Gaussian models: draw correlations, realize weights
  causenet.py      names variables with CauseNet concepts
  audit.py         how often single sources and shortcuts are right
  bnrep.py         transfer tasks on published networks
  qrdata.py        QRData loader and its grading rule
scripts/           build tasks, train (TRL GRPO), evaluate on Modal, pre-registered analysis
docs/              design, pre-registration, results, and the README figures
data/              sample tasks and the bnRep networks
tests/             environment, sandbox, grading, graphs, naming and analysis
```

</details>

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check src tests scripts && uv run mypy src
uv run --with pillow python docs/figures/render.py    # re-render the README figures (needs Google Chrome)
```

Code is Apache-2.0. CauseNet-derived data is CC BY 4.0; see [data/README.md](data/README.md). QRData is
CC BY-NC 4.0 and is not included.
