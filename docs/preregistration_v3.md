# Environment v3 study: pre-registration

Does GRPO on environment v3 (`docs/environment_v3.md`), where the model answers causal questions by analyzing
data with a Python tool, teach skills that hold beyond its training distribution? This file fixes the arms,
evaluation sets, protocol, statistics and decision rules before any test, QRData or bnRep reply exists. The
code that applies it is `scripts/confirm_v3.py`, committed with this file.

## Revisions before any evaluation

An independent review of the first draft found the problems below. Every fix was made before any test,
QRData or bnRep reply existed.

| Problem | Fix |
| --- | --- |
| The evaluation code was not recorded, and the analysis trusted each reply's stored reward. A redeploy between arms could grade them with different code. | Every reply records a digest of the deployed evaluation code (`src/` and `modal_dataenv.py`) and the installed package versions. The analysis requires one code digest, package set and generation setting across all arms, and re-grades every reply from its text with the registered grader (`environment.grade_reply`; empty if truncated), refusing any mismatch. On the 1,600 dev replies every re-grade matched. |
| The control was not pinned to the pilot's code. | The control is trained from the pilot's commit (`3ec1cd4`), with the pilot's exact arguments plus `--shuffle-rewards`. Its `run.json` must show code digest `82150b10…`, as the pilot's does. |
| Columns follow node order. In training and test data 81–85% of edges point from an earlier to a later column, so "the earlier column causes the later" is right on 67–71% of related structure items and scores 0.42–0.48 on all structure items (a constant answer scores 1/3). The trained arm may have learned this. | Test splits keep the training distribution, so this is disclosed, and structure results are split by whether the column-order rule is right (exploratory). bnRep, which no model had answered, now orders nodes at random per network: 27 of 58 edges point forward, and the rule is right on 15 of 33 related structure items. Its items were rebuilt and re-hashed. |
| QRData's prefix rule rewards formatting: "Cannot be determined" counts as C, "Both" as B, "not" as "no". The trained arm learned two-decimal numbers, but some QRData answers need three. | The official rule stays the endpoint. A strict reading (`qrdata.strict_choice`: the gold choice as a whole token) is a scope condition, and replies right only under the prefix rule are counted. Numerical errors are broken down by rounding. |
| The bnRep scope rule named no test, and its shortcut-wrong cells are tiny (5 effect items with both shortcuts wrong). | bnRep cannot separate skill from shortcuts; a confirmed bnRep gain is described as such (scope rules below). |
| Gains could come from finishing within the budget rather than from analysis; base fails to answer more often. | A scope condition: the gain on items every sample of both arms answered. |
| `eval_main` printed each set's accuracy, and a set could be deleted and collected again before recording. | No score is printed. Each set is recorded and committed right after it is collected. |
| Tool timeouts are wall-clock and depend on the load an arm's own code creates. | Timeout, restart and startup-failure rates per arm and set are reported. |
| Draft claims did not match the data: "balanced" bnRep labels, the best constant on QRData, the manipulation check's comparator. | Corrected below. |

The review also checked, and found sound: the sign-flip test, the bootstrap, Holm, the maximum over
comparators, the QRData weights and their 1/K cap, the item hashes, train/dev/test separation, the bnRep
pairing and retry logic, the sandbox changes and the isolation of answer keys. Its claim that QRData's
`eval.py` cannot score negative answers was wrong: `eval.py` takes the smaller and larger of
gold × (1 ± 0.03), as `qrdata.grade` does.

## What has been seen before registration

- **Environment v3:** base and the pilot adapter on the four dev splits (200 items each, seed 1000), with
  breakdowns by family and shortcut cell (`docs/environment_v3.md`, phases B and C). The pilot adapter became
  the trained arm because it passed the dev gate. No model has answered a v3 test item.
- **QRData:** no model in this project has answered it. Before registration we read its questions, answer
  key and gold-answer distribution, which motivated the cell-balanced score below.
- **bnRep:** v1's tool-use study used the four networks in a different task format (five effects per
  outcome), with other adapters (`docs/transfer_results.md`). The v3 items are new; no model has answered
  them.
- **Known weakness of the trained arm, from dev:** on effect questions its gains come from the unadjusted and
  adjust-for-everything regressions, not from valid adjustment sets.

## Arms

All arms are Qwen3-4B-Instruct-2507 at revision `cdbee75f17c01a7cc42f958dc650907174af0554`.

| Name | What it is |
| --- | --- |
| `v3-base` | The untrained model |
| `v3-trained` | The pilot adapter: LoRA after 60 GRPO steps on `outputs/dataenv/v3/train.jsonl` (run `data-pilot`, code `3ec1cd4`), digest `722501e468587c49e50e314bca62b6a62b1ed9d0d1502d6fafcfbb68bc514547` |
| `v3-control` | The same run with each step's 64 rewards shuffled (run `data-control`); trained only if the gate opens. Trained after the gate opened (2026-10-31) from `3ec1cd4`, `run.json` code `82150b10…`, digest `80c8fa815f66f9f55753e1ca21577a31b44e524622770f5623388e228ac70e2e` |

**Training configuration (both runs).** `scripts/train_dataenv.py` via `scripts/modal_dataenv.py::train_main`
on one H100:
- 60 steps, 8 tasks × 8 rollouts per step, at most 10 tool calls per episode;
- learning rate 2e-5 (constant), LoRA rank 32 on all linear layers;
- at most 8,192 tokens per episode (model and tool output together), temperature 1.0;
- DAPO loss, β = 0, `scale_rewards="batch"`, truncated episodes masked from the loss;
- vLLM colocated with 35% of GPU memory, seed 0.

The pilot's `run.json`: data `9caeb241033d2d7c-train.jsonl`, arguments
`--max-steps 60 --revision cdbee75f17c01a7cc42f958dc650907174af0554`, code `82150b10…`. The control is
trained from a checkout of `3ec1cd4`, deployed for training only:

```bash
git worktree add ../cab-3ec1cd4 3ec1cd4 && cd ../cab-3ec1cd4
uv run --with modal modal deploy scripts/modal_dataenv.py
uv run --with modal modal run scripts/modal_dataenv.py::train_main \
    --dataset /Users/ericxia/workspace/causal-agent-benchmark/outputs/dataenv/v3/train.jsonl \
    --run data-control --max-steps 60 \
    --extra "--revision cdbee75f17c01a7cc42f958dc650907174af0554 --shuffle-rewards"
```

(`outputs/` is not in the checkout, so the training file is given by absolute path; its upload name carries
its hash and must match the pilot's.) Then the registration commit
is deployed again for evaluation; the code digest every reply records shows that it was. The control's
adapter digest is committed to `confirm_v3.ADAPTER_SHA256` right after training, before evaluation.

**Control: shuffled rewards.** Each step's rewards are permuted across all its rollouts (generator seeded by
the run seed and the step). It keeps prompts, tool, sampling, each step's reward distribution and compute. It
rules out that a gain comes from running GRPO with uninformative rewards on these prompts. It does not rule
out gains from learning to finish within the budget or to write the answer line, since the trained run is
rewarded for both; the scope conditions below address that.

**Gate.** `v3-trained` is first compared with `v3-base` alone. The control is trained and evaluated only if
Holm's procedure passes at least one endpoint. Adding a comparator can only raise an endpoint's p-value, so if
none passes, none can be confirmed, and the study stops with no confirmed effect.

## Evaluation protocol

- **Generation:** `scripts/modal_dataenv.py::eval_main`, offline vLLM 0.31.0 on one L40S, the multi-turn tool
  loop of training (TRL's response parser and tool-suffix tokens), 96 episodes at a time. The prompt is one
  user message, and the `python` tool schema is training's.
- **Sampling:** the model's generation-config defaults (temperature 0.7, top-p 0.8, top-k 20, min-p 0), at
  most 8,192 tokens per episode, at most 10 tool calls. Sample j of an item uses seed 1000 + j.
- **Tool:** training's sandboxed session (20 s per call, 2,000 output characters, the worker as `nobody` with
  a 2 GB address-space limit). Task files are readable only by root.
- **Answers:** the last "Answer:" line. A reply that stops at the token limit scores 0.
  - v3 tasks and bnRep: `reasoning.grading.grade` in the task's format (numbers within 0.10).
  - QRData: the rule of its `eval.py` (`dataenv/qrdata.py`): a choice is right if the normalized answer
    starts with the gold choice; a number if its first number lies strictly within 3% of gold.
- **Records:** every reply records the model, revision, adapter path and digest, token limit, sampling, vLLM
  and package versions, the evaluation code digest, and its full message list, whose first message must
  equal the registered prompt.
- **Before collection:** deploy the registration commit, run `isolation_main` (the model's code cannot read
  the volume, the task files, the code or the parent process) and `files_check_main` on both QRData files
  (every file set loads in a locked-down session; it prints the code digest).
- **Collection:** calls are spawned on the deployed app and reattached on rerun. Each command below is
  followed by `python scripts/confirm_v3.py record <arm> <set>` and a commit of
  `docs/preregistration_v3_records.json`, before any score is computed. A set is never collected twice.

  ```bash
  R=cdbee75f17c01a7cc42f958dc650907174af0554
  F=outputs/dataenv/transfer/qrdata_source/data
  for s in iid ood_wording ood_size ood_vocab; do
    uv run --with modal modal run scripts/modal_dataenv.py::eval_main --tasks outputs/dataenv/v3/${s}_test.jsonl \
        --name v3-base-${s}_test --revision $R
  done
  for s in bnrep_real bnrep_placeholder; do
    uv run --with modal modal run scripts/modal_dataenv.py::eval_main --tasks outputs/dataenv/transfer/$s.jsonl \
        --name v3-base-$s --revision $R --samples 2
  done
  for s in qrdata_causal qrdata_statistics; do
    uv run --with modal modal run scripts/modal_dataenv.py::eval_main --tasks outputs/dataenv/transfer/$s.jsonl \
        --name v3-base-$s --revision $R --samples 4 --files-dir $F
  done
  ```

  `v3-trained` and `v3-control` use names `v3-trained-…` and `v3-control-…` and add
  `--adapter data-pilot/adapter` or `--adapter data-control/adapter`.

## Evaluation sets

Item-list hashes come from `python scripts/confirm_v3.py hash` (SHA-256 of each item's id, prompt, answer and
answer format, in id order), and the analysis asserts them.

| Set | Role | Items | Clusters | Samples | Score |
| --- | --- | ---: | ---: | ---: | --- |
| `iid_test` | manipulation check | 500 | items | 1 | accuracy |
| `ood_wording_test` | endpoint: held-out wording | 500 | items | 1 | accuracy |
| `ood_size_test` | endpoint: held-out graph sizes (7–8 variables) | 500 | items | 1 | accuracy |
| `ood_vocab_test` | endpoint: held-out concept names | 500 | items | 1 | accuracy |
| `qrdata_causal` | endpoint: far transfer, real data | 269 | 139 file sets | 4 | cell-balanced accuracy |
| `bnrep_real` | endpoint: near transfer, published networks | 192 | items | 2 | accuracy |
| `qrdata_statistics` | exploratory | 142 | 51 file sets | 4 | accuracy |
| `bnrep_placeholder` | exploratory | 192 | items | 2 | accuracy |

```text
iid_test           7d761a225299c6caadd9e2efbf72bbee8d147185f356b94f56097b85f8c1116e
ood_wording_test   28ac5e146b57af787082be7ab76b4398e9c65fc5b30ae2727907b617f44027ba
ood_size_test      83a539bb87d20f800dbfcb4085dc8fa5462d96e9091838101aa07b0abd6b6fb7
ood_vocab_test     1e5717b94f20777880bc431b091356872bae4e303626189709197cce119b57d1
qrdata_causal      d8c36a59640a2534742fb0158d396dcc020d9715783465d3479fb11b6ee919b7
bnrep_real         45bef4f8664f8ebc195638cd727ce1df887381d4f3eee47b31230835b68c4849
qrdata_statistics  6ddd936604bf048492df4d0914c2dff478e64a3836b9c1b3a65b6383d29eeabe
bnrep_placeholder  ccc1dcd6ec0e89655de0d715912b629a9f21911f0efd8d8ea25c8833f8e8c02a
```

- **v3 test splits:** `outputs/dataenv/v3/{split}_test.jsonl` (seed 200, `scripts/build_dataenv.py`). No test
  prompt appears in train or dev. Labels are balanced, and every single-source shortcut and constant answer
  in the audit is near chance (`docs/environment_v3.md`). The column-order rule is not (see the revisions).
- **QRData causal** (Liu et al., 2024, github.com/xxxiaol/QRData at commit `de450af`; `QRData.json` SHA-256
  `e076d960…`, `data.zip` `3c3a97cc…`; CC BY-NC 4.0, kept local): every question whose keywords include
  "Causality", 205 multiple-choice and 64 numerical.
  - **Prompt:** the benchmark's data description, where the files are (`df`, and `dfs` for the 2 causal and
    14 statistics questions with several files), the question, training's tool sentence, and the final-line
    instruction listing the item's choices.
  - **Why cell-balanced:** gold answers are skewed. "No causal relationship exists" is gold on 92 of the 150
    full-graph questions. A constant letter chosen per question group scores 0.615 over the 205
    multiple-choice items (one constant letter for all of them scores at most 0.366). A trained model that
    says "none" more often would gain official accuracy without item-level skill.
  - **Score:** multiple-choice items fall in groups that share a question type and choice set: full graph
    with 3 options (100) or 4 options (50), pairwise (50), randomized experiment (4), and one observational
    item. Within a group every gold answer's items weigh the same in total, so no habit that ignores the
    item scores more than 1/K of the group, K its number of distinct gold answers. Groups weigh by size, and
    numerical items weigh 1/269 each.
  - **Concentration:** questions on `arctic.csv` and `flow.csv` carry 0.186 and 0.160 of the score, and the
    effective number of clusters (1 / Σ cluster weight²) is 15.5. The heaviest single items (gold B among
    the 4-option questions) weigh 0.023 each.
  - **Clusters:** questions that share data files form one cluster (139), which is conservative under the
    fixed-item estimand.
  - Official accuracy is reported too, for comparison with published numbers.
- **bnRep** (`data/bnrep`, `scripts/build_dataenv_transfer.py --per-family 12`; `dataenv/bnrep.py`): v1's four
  published Gaussian networks (algal2, expenditure, lexical, liquefaction; 8–12 variables), standardized,
  nodes in a random order.
  - **Items:** 12 per network and family, 192 in all, each with its own 2,000 simulated rows, distinct
    queries and the training families' data checks. Wording follows the training templates. The names
    sentence says the names are the published study's (`bnrep_real`) or placeholders (`bnrep_placeholder`,
    `V1`…`Vn` in random order). Item i of one file is item i of the other.
  - **Labels:** each item's target is drawn once before its search: independent with probability 1/2,
    causally related with probability 2/3 (directions balanced by a swap), each adjustment set valid with
    probability 1/2, a nonzero effect with probability 0.9. Answer counts: independence 23 yes / 25 no,
    structure letters 20 / 17 / 11, effect 44 nonzero / 4 zero. The best constant answer scores 0.53 on
    yes/no items, 0.42 on letters, 0.38 on 1/2/Both/Neither and 0.10 on effects (zero).
  - **Shortcuts are strong:** the published networks have little confounding. The unadjusted regression is
    right on 41 of 48 effect items (0.85), the confounder rule on 0.84 of single-set adjustment items, and the
    marginal rule on 0.67 of independence items.

## Statistics

- **Estimand:** each arm's expected score on these fixed items. Replies are random; items are not resampled.
- **Per item:** mean reward over its samples. **Per set:** the weighted mean above.
- **Test:** one-sided cluster sign-flip randomization test, 20,000 resamples, numpy seed 0. Each resample
  flips the sign of every cluster's weighted score difference with probability 1/2;
  p = (1 + number of resampled differences ≥ observed) / (1 + 20,000).
- **Interval:** 95% paired cluster-bootstrap percentile interval with fixed weights, 20,000 resamples. It is
  descriptive, and it decides the scope conditions below.
- **Endpoint p-value:** the largest over its comparators (`v3-base`, and `v3-control` once it exists).
- **Multiplicity:** Holm's step-down over the five endpoints at one-sided 0.025. The manipulation check
  compares `v3-trained` with `v3-base` alone, at one-sided 0.025.
- **Inputs:** `confirm_v3.py` refuses to analyze unless all of these hold:
  - every arm has exactly the registered replies (each item × seed once), and each answered its registered
    prompt;
  - each arm has one configuration across sets, with the registered model, revision, adapter digest, token
    limit, sampling and vLLM version;
  - all arms share one evaluation code digest and package set;
  - every stored reward equals a re-grade of its reply;
  - the replies match the committed manifest.

## Decisions and scope

1. **Manipulation check:** if `iid_test` fails, training did not work in distribution on test; confirmed
   endpoints still stand but carry that caveat.
2. **Held-out generalization:** each confirmed endpoint among the three held-out test splits.
3. **Transfer:** `bnrep_real` (near: same question families on published structures) and `qrdata_causal`
   (far: real data, other question formats).

A confirmed endpoint means the trained adapter beats the base model and the same training with shuffled
rewards. There is one training run per arm, so claims are about these adapters. An endpoint not confirmed is
reported as "not confirmed" with its interval, not as evidence of no effect.

**Scope conditions** decide how a confirmed endpoint is described; they never confirm or unconfirm one. Each
uses the 95% bootstrap interval of the trained-minus-comparator difference, against every comparator:
- **Analysis, not finishing:** recompute the score on the items where every sample of every compared arm
  ended with an answer line within the budget. If the interval lies above zero, the gain is described as
  better analysis; otherwise as a gain that finishing within the budget may explain.
- **QRData, not formatting:** recompute `qrdata_causal` with `strict_choice` for multiple-choice items. If the
  interval lies above zero, the gain is described as holding under strict answer parsing; otherwise as one
  the benchmark's prefix rule may explain.
- **bnRep, not shortcuts:** bnRep's shortcut-wrong cells are too small to test (for example 5 effect items
  where both regressions are wrong). A confirmed `bnrep_real` gain is described as a gain on published
  networks that the single-source shortcuts could explain, with the per-cell numbers shown.

## Exploratory analyses

None supports a claim on its own:
- every set by family and answer format, and the test splits by shortcut cell, as in phase C: in particular,
  effect items where both the unadjusted and adjust-for-everything regressions are wrong, and where the answer
  lands (truth, unadjusted value, adjust-for-everything value);
- structure items on the test splits split by whether "the earlier column causes the later" is right;
- truncation, no-answer, tool-timeout, restart and startup-failure rates, tool calls and reply tokens per
  arm and set;
- QRData: official accuracy, by question type, the rate of answering "No causal relationship", replies right
  only under the prefix rule, numerical items alone with errors split by rounding (a reply that is right
  after rounding gold to the reply's decimals), and the statistics subset;
- bnRep: placeholder names, the real-minus-placeholder gap per arm (do names help, and does training change
  that?), and results per network;
- the gap between each held-out split and `iid_test`, per arm.

## Power (estimates)

- **Test splits:** on dev the paired difference had a standard error of about 3.6 points with 200 items, so
  about 2.3 points with 500. The smallest effect with 80% power is about 8 points at Holm's strictest level
  (0.025 / 5) and about 6.5 points at 0.025. Dev gains were 8–19 points.
- **bnRep:** 192 items × 2 samples, a standard error of roughly 4 points: about 14 points at the strictest
  level.
- **QRData causal:** with an effective 15.5 clusters, the standard error may reach 5 points or more; small
  gains will be missed.

## Budget

| Item | Cost (approximate) |
| --- | ---: |
| Evaluation of `v3-base` and `v3-trained` (about 3,500 episodes each on L40S) | $8–12 |
| If the gate opens: `data-control` training (about 4.3 h on an H100) and evaluation | $22–28 |

## Known limitations

- **One run per arm:** run-to-run variation is not measured.
- **The trained arm was chosen on dev:** the pilot was kept because it passed the dev gate; test, QRData and
  bnRep are untouched by that choice.
- **Column order in training data:** see the revisions. The trained arm may use it on structure questions.
- **Shortcuts:** bnRep's networks reward the unadjusted regression, which the trained arm uses on effect
  questions.
- **QRData:** CC BY-NC 4.0, local only. It dates from 2024 and may be in Qwen3's training data, which affects
  every arm alike. Some analyses (PC on the 222-column neuropathic files) may exceed the 20-second tool limit
  for every arm.
- **Reproducibility:** vLLM's batched arithmetic is not bit-reproducible; results replicate in distribution.
