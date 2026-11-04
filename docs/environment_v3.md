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
| Tool-use study ([v1 results](https://github.com/Ruichu-Xia/causal-agent-benchmark/blob/e2020db/docs/transfer_results.md)) | v1: 10,000 rows, fixed tools `correlate` / `regress` / `discover`, report 5 signed effects | Error −0.20 after 120 steps | None confirmed. Exploratory: on published networks with placeholder names, error 0.624 (base) → 0.533 (trained) |
| v2 study ([v2 results](https://github.com/Ruichu-Xia/causal-agent-benchmark/blob/e2020db/docs/results_v2.md)) | Text only: a described graph, four question families, shortcut-balanced | +1.9 points (not significant) | None |

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

1. **World.** Draw a random connected DAG G0, and a unit-variance linear-Gaussian correlation matrix Σ on it
   (`scm.generate.draw_correlation`). Then draw the true DAG D uniformly from G0's Markov equivalence class,
   with weights set by regression (`realize`, v1's "covariance transfer"). The data then carry no orientation
   information beyond the CPDAG: no variance, R² or coefficient size can hint at a direction the data leave
   open.
2. **Names.** v1's `CauseNetLabeler` names D's nodes with its four rules:
   - every edge of D is a one-way CauseNet relation in that direction;
   - no CauseNet relation between chosen names points against D's causal order;
   - no near-duplicate names.

   Pairs with no causal path between them may still carry CauseNet relations, so names can suggest links
   the data refute. Naming takes about 0.1 s per graph, and held-out concepts name every size tried (4–8
   nodes).
3. **Data.** 2,000 rows sampled from D. The task stores the model and the sampling seed, not the rows, so a
   task file stays small. Sampling adds each node's weighted parents to its noise in a fixed order, using
   elementwise operations only: a first version solved a linear system, whose last bits (LAPACK) differed
   between the build machine and the containers. numpy's normal sampler still calls the platform's `log` in
   a rare branch: 4 of 400 training worlds differed in the containers in a few last bits. So a regeneration
   is checked against the build's column sums and sums of squares to a relative 1e-12, which a wrong seed or
   simulator misses by order one, and a last-bit difference cannot move any of the gate's tests across its
   threshold in practice.
4. **Prompt.** It gives:
   - the column names and the row count;
   - the assumptions: linear, Gaussian noise, acyclic, no hidden common causes, no exact cancellations, unit
     variances;
   - the names statement: "Every direct cause in this system is one the variables' meanings make plausible,
     in that direction; some plausible links are absent, and effect sizes and signs are synthetic";
   - one question.
5. **Tool.** The agent calls `python` up to 10 times. Each call runs in a persistent per-episode session with
   `df` already loaded.
6. **Answer.** The final reply ends with `Answer: <answer>`. The reward is 1 if correct, else 0.

## Question families

Labels come from D. Two reference rules show what each source alone can do:
- **data-only:** the answer in every member of D's equivalence class (a population-level, perfect analysis
  that ignores names). When members disagree it guesses, with credit 1/2 on binary questions.
- **names-only:** the answer in Gn, the graph of CauseNet relations among the chosen names.

Using both, skeleton and compelled orientations from data and the remaining orientations from names, always
gives D, because every edge of D is a CauseNet relation in its direction.

| Family | Format | Question | Drawn independently (cells) |
| --- | --- | --- | --- |
| Independence | yes/no | "Is X independent of Y given Z?" (or "associated") | label; names-only right (Gn has a directed path between X and Y in either direction iff "associated"); marginal rule right (the answer with Z empty) |
| Structure | A/B/C | "Which is true in this system? X causes Y (directly or indirectly) / Y causes X / neither causes the other", the letters shuffled (QRData's pairwise format) | source cell: name-critical 1/2 (D has a path, the class disagrees on it, names right); data-critical 1/4 (no path in D, the class agrees, Gn has a path); both suffice 1/4 |
| Adjustment | yes/no (one set) or 1/2/Both/Neither (two analysts) | "Would regressing Y on X and Z recover X's total effect on Y?" (one set, or two analysts' sets) | per question: data-determined (every class member gives each set the same validity). Per set: valid; confounder rule right, a fair coin for valid sets and fixed by determinedness for invalid ones, since no invalid, undetermined set has the confounder rule wrong (none in about 150,000 sets of random 4- to 7-node graphs) |
| Effect | number, 2 decimals, graded within ±0.10 | "How many standard deviations does Y change when X is set one standard deviation higher?" | name-critical (some class member's effect is more than 0.2 from the truth) or data-identified (all within 0.02); unadjusted correlation right (within 0.10) or wrong (more than 0.2 off); regression on every other column right or wrong, likewise; nonzero (90%) |

Reported, not balanced: each family's single-source accuracies, the confounder rule on structure-like
questions, and constant answers. By construction on Structure, names-only and data-only each score about
0.75, both together score 1, and a constant letter scores about 1/3.

### Audit of the training set (4,000 items, `src/causalagentbench/audit.py`)

Mean credit of each rule; correct analysis of the data and names scores 1 by construction.

| Family, format (items) | Data-only | Names-only | Family shortcuts | Constant answers |
| --- | ---: | ---: | --- | --- |
| Independence, yes/no (1,000) | 1.00 | 0.50 | marginal 0.50 | yes 0.50, no 0.50 |
| Structure, A/B/C (1,000) | 0.76 | 0.74 | — | A 0.35, B 0.35, C 0.30 |
| Adjustment, yes/no (493) | 0.79 | — | confounder rule 0.48 | yes 0.55, no 0.45 |
| Adjustment, two analysts (507) | 0.76 | — | confounder rule 0.30 (chance 0.25) | 1 0.25, 2 0.29, Both 0.21, Neither 0.25 |
| Effect, number (1,000) | 0.73 | — | unadjusted 0.51, adjust for all others 0.51 | 0.00 scores 0.10 |

The first build left adjusting for every other column unbalanced (0.64 on effect questions), so it was added
to the effect family's drawn cells.

## Answerable from the data

A label from D must also be recoverable from the sampled rows, or the reward punishes correct analysis.
Every item passes these checks on its own rows:
- **Structure from data:** PC (α = 0.01, stable) on the rows returns exactly D's CPDAG. Measured on 100 random
  graphs per size with 2,000 rows, this keeps 90% at 4 nodes, 66% at 5, 34% at 6, 16% at 7 and 10% at 8.
  Rejected draws cost milliseconds. The kept graphs have clearer signal; that is the price of exact labels.
- **Independence:** the Fisher-z test of the queried pair given Z agrees with the label, with a margin: p
  below 0.01 for dependent pairs, above 0.05 for independent ones.
- **Effect:** OLS of Y on X and X's parents in D, on the rows, lands within 0.03 of the true effect.

This filter is about correctness, not curation. Pass rates are reported per family and split.

## The Python tool

- **Session:** one long-lived Python worker per episode, which keeps variables between calls. Preloaded:
  `df`, numpy (`np`), pandas (`pd`), scipy, statsmodels, and three documented helpers:
  - `partial_corr(df, x, y, given=())`;
  - `pc(df, alpha=0.01)`, which returns the CPDAG's directed and undirected edges;
  - `ols(df, y, X)`.

  The agent can use them or write its own analysis.
- **Limits:** 20 seconds per call (a longer call restarts the session), and 600 seconds of CPU and 2 GB of memory per worker. Output is truncated to 2,000 characters.
- **Isolation (a correctness requirement in RL).** The model's code is untrusted, and a policy that can read
  answer keys will learn to.
  - All model code runs in Modal containers, for both training and evaluation, never on a laptop.
  - The worker runs as an unprivileged user with a cleared environment, in an empty directory holding only
    the episode's data.
  - The volume and the code directory are made unreadable to that user at container start.
  - A Modal smoke test runs code that tries to read the task file, the volume, `/proc` of the parent and
    the environment from inside the tool, and each attempt must fail.
  - Every training run's rollouts are scanned for file and OS access.
  - Network access is not blocked: nothing secret is reachable, and no answer exists anywhere online.
- **Errors:** they return the traceback text, so the agent can fix its code.
- **Format:** training (TRL `environment_factory`) and evaluation (a multi-turn loop over offline vLLM) render
  tools with the model's own chat template and parse calls with TRL's `parse_response`, so they see one
  format. Evaluation uses the CUDA devel image, because vLLM compiles FlashInfer's top-k/top-p sampling
  kernels with nvcc; training samples at temperature 1.0 and keeps v1's image. The isolation check and a
  data-regeneration check run in both images.

## Splits

v2's scheme, with data:

| Split | Graph sizes | Prompt wording | Concept names |
| --- | --- | --- | --- |
| train | 4–6 | training templates | training concepts |
| iid | 4–6 | training templates | training concepts |
| ood_wording | 4–6 | held-out templates | training concepts |
| ood_size | 7–8 | training templates | training concepts |
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

| Piece | Code (`src/causalagentbench/` unless noted) |
| --- | --- |
| Worlds, equivalence classes and data | `world.py`, `graphs.py`, `scm.py` |
| CauseNet names | `causenet.py` |
| Question families, wording, splits and audit | `families.py`, `wording.py`, `dataset.py`, `audit.py` |
| Python tool | `sandbox.py` |
| Episode and judge | `environment.py`, `grading.py` |
| Transfer sets | `bnrep.py`, `qrdata.py` |
| Training and evaluation | `scripts/train.py`, `scripts/modal_app.py` |
| Confirmatory and exploratory analysis | `scripts/confirm.py`, `scripts/explore.py` |

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

## Phase B: calibration (2026-10-30)

Base Qwen3-4B-Instruct-2507 (revision `cdbee75`), 200 dev episodes per split, one sample, generation-config
sampling, 8,192 completion tokens (`scripts/modal_app.py::eval_main`):

| Split | Accuracy | Ends with an answer | Truncated | Tool calls per episode |
| --- | ---: | ---: | ---: | ---: |
| iid | 0.465 | 0.920 | 0.055 | 2.5 |
| ood_wording | 0.520 | 0.925 | 0.050 | 2.9 |
| ood_size | 0.415 | 0.885 | 0.070 | 2.7 |
| ood_vocab | 0.430 | 0.925 | 0.050 | 2.4 |

- iid by family: independence 0.70, structure 0.44, adjustment 0.40, effect 0.32.
- About half the base model's tool outputs are its own errors: guessing the helpers' result keys instead of
  calling `help()`, overwriting a helper with its result, importing functions that do not exist.
- Training throughput (3 steps, 64 rollouts, H100): about 380 s per step, so 60 steps cost about $27.
- No training rollout's code touched files, the OS or other processes.
- Gate: accuracy is inside 40–75% on every split. Ending with an answer is at least 90% on three splits and
  0.885 on ood_size, where the misses are mostly truncation on larger graphs.

## Phase C: pilot (2026-10-30)

60 GRPO steps on `train.jsonl` (8 tasks × 8 rollouts, LoRA rank 32, learning rate 2e-5, DAPO loss, beta 0,
temperature 1.0, seed 0; code at `dd79e10`; `scripts/modal_app.py::train_main --run data-pilot --max-steps 60
--extra "--revision cdbee75f17c01a7cc42f958dc650907174af0554"`).
Training took 4.3 hours on one H100. Adapter digest `722501e4…` (`data-pilot/adapter`).

Training reward rose from about 0.44 (steps 1–30) to 0.62 (steps 51–60); ending with an answer stayed at
0.95–0.98 and tool calls at about 2.5 per episode.

Dev evaluation, same protocol as phase B (seed 1000, one sample, same tasks). Differences are pilot minus
base, with 95% paired bootstrap intervals over tasks (10,000 resamples, seed 0):

| Split | Base | Pilot | Difference | 95% interval | Ends with an answer |
| --- | ---: | ---: | ---: | --- | --- |
| iid | 0.465 | 0.655 | +0.190 | [+0.120, +0.260] | 0.920 → 0.970 |
| ood_wording | 0.520 | 0.600 | +0.080 | [+0.005, +0.155] | 0.925 → 0.980 |
| ood_size | 0.415 | 0.605 | +0.190 | [+0.105, +0.275] | 0.885 → 0.960 |
| ood_vocab | 0.430 | 0.595 | +0.165 | [+0.080, +0.250] | 0.925 → 0.990 |

**Gate: passes** (iid gain at least 0.05, interval above zero).

- Truncation fell from 5–7% to about 0, but it explains only part of the gain: on episodes both arms answer,
  pooled over splits, the difference is +0.121.
- By family, pooled over splits: independence 0.655 → 0.960, structure 0.485 → 0.640,
  adjustment 0.355 → 0.430, effect 0.335 → 0.425.
- Independence, structure and adjustment gain at least as much where a shortcut is wrong as where it is
  right: the marginal rule (+0.385 wrong, +0.219 right) and the confounder rule (+0.120 wrong, +0.037 right).
  Structure gains more where the names are right (+0.189) than wrong (+0.058): the pilot uses the names.
- **Effect gains come from shortcuts.** Where the unadjusted estimate is wrong, the pilot's answer lands on
  it in 42 of 104 episodes (base 25) and is right in 22 (base 24). Of those 22, 21 equal the
  adjust-for-everything estimate, and only 1 is right where both shortcuts are wrong (base 5). The pilot
  always regresses (96–99% of effect episodes) but does not learn a valid adjustment set.
- No training rollout's code touched files, the OS or other processes (3,840 rollouts, 9,866 tool calls).

## Phase D: confirmatory study (2026-10-31)

Pre-registered in `docs/preregistration_v3.md`; results in `docs/results_v3.md`. The trained adapter beats
base and the shuffled-reward control on the three held-out test splits and on bnRep; QRData is not confirmed.
