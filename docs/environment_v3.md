# Causal data-analysis environment v3: design

Written before implementation and reviewed. Decisions (2026-10-30): keep it simple, linear-Gaussian
variables only, 2,000 rows per episode, 60 training steps, and keep cost low.

## Goal

Train and measure one skill: **answering causal questions about a dataset by analyzing it.** The agent gets
a table, a question and a Python tool, and ends with a short answer that can be checked exactly. The
environment should improve that skill in a way that shows up on independent benchmarks that also come with
data, not only on its own format.

## What the earlier work tells us

| Study | Environment | In-domain | Transfer |
| --- | --- | --- | --- |
| Tool-use study (`docs/transfer_results.md`) | v1: 10,000 rows, fixed tools `correlate` / `regress` / `discover`, report 5 signed effects | Error −0.20 after 120 steps | None confirmed. Exploratory: on published networks with placeholder names, error 0.624 (base) → 0.533 (trained) |
| v2 study (`docs/results_v2.md`) | Text only: a described graph, four question families, shortcut-balanced | +1.9 points (not significant) | None |

Lessons, each a design rule below:
1. **Data plus tools gives a training signal; text alone did not.** Keep the agent working on data.
2. **One rigid task trains a narrow skill.** v1 had one prompt, one answer format and fixed tools. v3 mixes
   question families and answer formats, and uses a general tool.
3. **Fixed tools tie the skill to our interface.** Benchmarks with data expect the model to write its own
   analysis code, so the tool is Python.
4. **Shortcut-balanced labels and held-out splits work** (v2's audit). Keep them.
5. **Check for an in-domain gain before running a confirmatory study.** v2 spent its confirmatory budget
   on a training run that barely moved the model in-domain.

## Episode

1. A hidden causal model (DAG plus linear-Gaussian mechanisms) generates 2,000 rows. Columns are named after
   CauseNet concepts, or nonsense words in some items.
2. The prompt gives the column names, the sample size, the modelling assumptions (acyclic, all common causes
   measured, and so on), and one question.
3. The agent calls `python` up to 10 times. Each call runs code in a persistent per-episode session with
   `df` (a pandas DataFrame) already loaded.
4. The final reply ends with `Answer: <answer>`. The reward is 1 if correct, 0 otherwise, the same as v2.

## Question families

Each item asks one question. Labels come from the hidden model, never from the data.

| Family | Format | Example | Shortcut balanced to chance |
| --- | --- | --- | --- |
| Independence | yes/no | "In the population, is `sleep quality` independent of `caffeine` given `stress`?" | Marginal-correlation rule (answer from the unconditional correlation) |
| Structure | choice | "Which is most consistent with the data and the variables' meaning? A: X causes Y, B: Y causes X, C: no direct causal link" (QRData's pairwise format) | "Adjacent in the data means a causal link, in the direction the names suggest" |
| Adjustment | choice | "Analyst 1 adjusts for {A, B}; analyst 2 for {C}. With unlimited data, whose estimate of the effect of X on Y is unbiased: 1, 2, Both or Neither?" | Adjust-for-everything rule; confounder rule (v2) |
| Effect | number (2 decimals, tolerance ±0.05 in standard-deviation units), or sign | "By how many standard deviations does Y change per one-SD intervention on X?" | Unadjusted regression slope; regression adjusting for everything |

As in v2, sampling draws each label and whether the family's shortcut agrees with it independently, so
every shortcut and every constant answer scores 0.5 (or the numeric equivalent for effects).

**Names stay part of the skill, as in v1.** Some structure and effect items have edges the data cannot
orient (same Markov equivalence class), where the concept names settle the direction. These are marked
*name-critical*. The placeholder-name ablation (columns renamed `X1…Xk`) measures how much names contribute.

## Answerable from the data

A label from the hidden model must also be recoverable from the sampled rows, or the reward punishes
correct analysis. Each item passes a gate before it is kept:
- **Data-identified items:** a reference analysis (PC at α = 0.01, then the adjustment criterion and OLS)
  run on the item's own rows must reach the label.
- **Name-critical items:** the data must leave the relevant edge unoriented, and the CauseNet relation
  between the names must orient it.
- **Strength:** every dependence the label relies on must have a partial correlation of at least 0.1, so
  2,000 rows detect it reliably. v1 used 10,000 rows; fewer rows keep tool calls fast.

This filter is about correctness, not curation: an item whose answer the data cannot support is wrong.
Gate pass rates will be reported per family.

## The Python tool

- **Session:** one long-lived Python worker per episode, which keeps variables between calls. Preloaded:
  `df`, numpy, pandas, scipy, statsmodels, and three helpers documented in the prompt:
  - `partial_corr(df, x, y, given)`;
  - `pc(df, alpha)`, which returns the CPDAG;
  - `ols(df, y, X)`.

  The agent can use them or write its own analysis.
- **Limits:** 20 seconds and 2 GB per call. Output is truncated to 2,000 characters. No network.
- **Isolation (a correctness requirement in RL).** The model's code is untrusted, and a policy that can read
  answer keys will learn to.
  - The worker runs as an unprivileged user in an empty working directory, with only the episode's data
    file.
  - Training data, labels, adapters and credentials are owned by another user and unreadable by it.
  - A test tries to read the training file, the volume and the environment variables from inside the tool,
    and must fail each time.
  - Every training run also scans its rollouts for file and OS access.
- **Errors:** they return the traceback text, so the agent can fix its code, as v1's tools do.

## Splits

v2's scheme, with data:

| Split | Graph sizes | Prompt wording | Concept names |
| --- | --- | --- | --- |
| train | 4–6 | training templates | training concepts |
| iid | 4–6 | training templates | training concepts |
| ood_wording | 4–6 | held-out templates | training concepts |
| ood_size | 7–9 | training templates | training concepts |
| ood_vocab | 4–6 | training templates | held-out concepts |

Each split has a `dev` set (seed 100) for development and a `test` set (seed 200) for the confirmatory
study, built as in v2.

## Transfer benchmarks

All of these come with data and are answered with the same Python tool:

| Benchmark | Items | What it tests | Notes |
| --- | ---: | --- | --- |
| QRData causal subset (Liu et al., 2024) | 269: 205 choice, 64 numeric | Pairwise and full-graph discovery on named real variables (neuropathic pain, Arctic sea ice, Sachs flow cytometry), treatment effects (IHDP, Jobs) | CC BY-NC 4.0: used locally, never redistributed. The main far-transfer endpoint. |
| Published Gaussian networks (bnRep), real and placeholder names | as in the tool-use study | Effect estimation on real structures | The one dataset where v1 showed an (exploratory) gain |
| QRData statistics subset | 142 | Non-causal data analysis (probability, testing, regression) | Exploratory: does general data-analysis skill move? |

QRData is too small to split. It stays untouched until the confirmatory study, and development decisions use
in-domain dev splits only.

## Implementation

| Piece | Reuses | New |
| --- | --- | --- |
| Graph and label sampling | v2's families, shortcut balance and splits | Structure family; label from the realized model |
| Linear-Gaussian realization and data | v1's `realize` and faithfulness gates | The answerability gate |
| Python tool | — | Sandboxed worker, helpers, isolation tests |
| Training | `train_grpo.py` (TRL `environment_factory`, multi-turn tool use), `modal_grpo.py` | Episode adapter for the new tool |
| Evaluation | `modal_serve.py` (vLLM server, LoRA), v1's agent loop | QRData and bnRep loaders in the new format |
| Hub package | v2's Verifiers taskset | A multi-turn Verifiers tool environment over the same core |

## Plan and cost

Prices assumed: H100 about $4.2 per hour and L40S about $2.0 per hour on Modal, as billed so far. Step time is
the main uncertainty. v1's agentic run took 145 s per step for 64 rollouts (about 630 generated tokens and 7
tool calls per episode). Python replies are longer, so the estimate is 250–400 s per step until phase B
measures it.

Cost is kept low in three ways:
- **The pilot adapter becomes the trained arm.** Development decisions use dev splits only, and test splits,
  QRData and bnRep stay untouched. So the pilot adapter can be fixed and registered before any test
  evaluation, and no fresh training run is needed.
- **The control is gated, as in v2.** The shuffled-reward run is trained only if the trained adapter beats
  base (and chance, where relevant) on some endpoint.
- **Evaluations are small and run on L40S.** Calibration uses 200 dev episodes per split.

| Phase | What | Gate to continue | GPU cost |
| --- | --- | --- | ---: |
| A. Build | Generator, families, answerability gate, sandbox, isolation tests, audits | Every test passes; shortcut audit at chance; isolation test fails to read secrets | $0 |
| B. Calibrate | Base Qwen3-4B-Instruct-2507 on 200 dev episodes per split, and a 3-step training throughput test | Base dev accuracy 40–75%; at least 90% of episodes end with a valid answer | $5–8 |
| C. Pilot | 60 GRPO steps (8 tasks × 8 rollouts); evaluate base and pilot on dev splits | In-domain dev gain of at least 5 points with an interval above zero | $20–32 |
| D. Confirmatory | Pre-register; evaluate base and the pilot adapter on test splits, QRData and bnRep (the gate) | — | $8–12 |
| D'. Control, only if the gate opens | 60-step shuffled-reward run and its evaluation | — | $22–34 |
| **Total** | | | **$33–52 without the control, $55–86 with it; about $25–40 if the pilot stops it** |
