# Environment v2 training study: pre-registration

Does GRPO on the causal-reasoning environment (`docs/environment_v2.md`) teach skills that hold beyond its
training distribution? This file fixes the training runs, evaluation sets, protocol, statistics and decision
rules before any trained model is trained or evaluated. The code that applies it is `scripts/confirm_v2.py`,
committed with this file.

## Revisions before any training or evaluation

The first version (commit `9933b98`) went through two independent reviews, and the revision after them
(commit `e839066`) through a third. Every fix below was made before any adapter of this study existed and before
any reply to a test or transfer item was collected.

| Problem | Fix |
| --- | --- |
| Every CLadder mediator/collider item keys "do not adjust", so never adjusting scores 1.0. Answering "no" every time beats base (0.347) by 16 points. | The CLadder endpoint adds the held-out confounding and diamondcut items, where adjusting is keyed. It is scored by balanced accuracy over four cells (whether adjusting is keyed × label). |
| Base is below chance on Corr2Cause (balanced accuracy 0.466), and may be on d-separation, so guessing would beat it. | On the three yes/no transfer sets the trained arm must also beat chance. |
| The cluster bootstrap is anti-conservative with 24 clusters: in simulation it rejected at 0.032 for nominal 0.025. | Decisions use a cluster sign-flip randomization test; the bootstrap interval is descriptive. |
| The control was neither implemented nor format-matched. | A knights-and-knaves control was built, format-matched and calibrated. Then it was dropped (next row). |
| The knights-and-knaves control taught only brevity. Whenever the base model finished a puzzle it was right 97–99% of the time, at every size from 3 to 12 inhabitants. Its failures were truncations (38% at the size that matched the pass rate). | The control is now the same causal training with shuffled rewards, which matches prompts, formats, sampling and compute. |
| Inputs could silently differ from the registration: items, base model, adapter weights, vLLM version, resumed training data, or resampled replies. | Item hashes (including the answer format) are asserted. Every reply records the model, adapter weight hash, token limit, sampling and vLLM version, and the analysis requires one registered configuration per arm. Adapter hashes are committed after training. Each arm's replies are hashed into a committed manifest right after collection. A run refuses to resume on different data or arguments. |
| The disclosure omitted earlier trained-model replies. | Corrected below. |

Third review:

| Problem | Fix |
| --- | --- |
| Corr2Cause's label base rates differ widely across its hypothesis wordings (for example 65 yes / 323 no for "X directly causes Y", 18 / 175 for the confounder wording). Balancing over labels alone lets a yes-rate shift concentrated on some wordings beat base and chance with no item-level skill. | `c2c_test` is scored by balanced accuracy over ten cells, hypothesis wording × label, so any answering habit that depends only on the wording scores exactly 0.5. |
| The shuffled-reward control was described as matching reply lengths and truncation. It does not: the trained run is rewarded for finishing within 4,096 tokens and writing the answer line; the control is not. | The claim is corrected, and truncation and no-answer rates per arm, plus a re-analysis on items where both arms finished, are pre-registered as exploratory analyses next to each confirmed endpoint. |
| The document said a truncated reply scores 0, but the grader scored any answer line in it. | `transfer_eval.judge` now treats a reply that stopped at the token limit as no answer. Base scores computed earlier do not change at the reported precision, except Qwen3-4B-Instruct-2507's 4,096-token `ood_wording` dev accuracy (0.61 to 0.60). |
| Sampling was checked for consistency within an arm but not against the registered values; the model revision was not pinned; the adapter hash left out its config; training recorded the script name, not its code; replies did not record which prompt they answered. | The analysis asserts the sampling parameters and model revision. Evaluation and training load revision `cdbee75`. The adapter digest covers config and weights. `run.json` holds a hash of the training code. Each reply carries its prompt's SHA-256, which the analysis checks against the registered items. |
| The manipulation check was decided by the bootstrap interval, which is otherwise descriptive. | It is decided by the sign-flip test at one-sided 0.025. |

## What has been seen before registration

- **Environment v2:** base models on the four dev splits only (`docs/environment_v2.md`, base-model ladder).
  No model has answered a v2 test item. The test splits were enlarged from 400 to 1,000 items per split
  before any evaluation; the first 400 items of each are the original files.
- **CLadder and Corr2Cause:** the earlier tool-use study (`docs/transfer_preregistration.md`) evaluated, at
  4,096 tokens:
  - its base model, its trained adapter (step 120) and its shuffled-reward control on Corr2Cause test and on
    all of CLadder `backadj` and `collider_bias`;
  - its step-40 and step-80 checkpoints on CLadder only.
  - Those adapters were trained on a different environment, the tool-use benchmark.
  - Their replies cover every item of `c2c_test` and `cladder_adjust`.
  - The CLadder endpoint was designed after those results were known (research notes,
    `research_notes/Transfer target selection/notes.md`, section 3.1).
  - Base balanced accuracy on `cladder_adjust`, from those replies, is 0.621; on `c2c_test`, under the
    ten-cell score, 0.467 (0.466 balanced over labels).
  - Every reply in this study is newly sampled under a different protocol and seeds.
- **d-separation:** no model has answered it.
- **This study's adapters:** none exists yet. The pilot adapter `reason60` was evaluated only outside this
  study:
  - the v1 reasoning dev set;
  - CLadder dev stories;
  - 100 Corr2Cause dev items, in an evaluator smoke test.

## Arms

All three arms are Qwen3-4B-Instruct-2507 at Hugging Face revision `cdbee75f17c01a7cc42f958dc650907174af0554` (the latest since
2025-09-17; earlier results loaded the latest revision, so this one).

| Name | What it is |
| --- | --- |
| `v2-base` | The untrained model |
| `v2-trained` | LoRA adapter after 60 GRPO steps on the v2 training set (run `reason-v2`) |
| `v2-control` | The same run with each step's rewards shuffled across its 128 rollouts (run `control-v2`); trained only if the gate opens |

**Training configuration (both runs).** `scripts/train_reasoning.py` via
`scripts/modal_grpo.py::reasoning_main` on one H100:
- 60 steps, 16 tasks × 8 replies per step;
- learning rate 2e-5 (constant), LoRA rank 32 on all linear layers;
- at most 4,096 new tokens, temperature 1.0;
- DAPO loss, β = 0, rewards scaled by the batch standard deviation (`scale_rewards="batch"`), truncated
  replies masked from the loss;
- vLLM colocated with 35% of GPU memory, seed 0.

The training file is uploaded as `<first 16 hex digits of its SHA-256>-<name>`. The run directory's
`run.json` records the data, script, arguments and a hash of the training code, and a run never resumes with
different ones. The adapter at step 60 is the arm; no checkpoint is selected. Right after each run, its
`modal_eval.adapter_digest` (over `adapter_config.json` and `adapter_model.safetensors`) is committed to
`confirm_v2.ADAPTER_SHA256`, before any evaluation of it.

```bash
uv run --with modal modal run scripts/modal_grpo.py::reasoning_main --dataset outputs/reasoning/v2/train.jsonl \
    --run reason-v2 --max-steps 60 --extra "--revision cdbee75f17c01a7cc42f958dc650907174af0554"
uv run --with modal modal run scripts/modal_grpo.py::reasoning_main --dataset outputs/reasoning/v2/train.jsonl \
    --run control-v2 --max-steps 60 --extra "--revision cdbee75f17c01a7cc42f958dc650907174af0554 --shuffle-rewards"
```

**Training record.**
- `reason-v2` (2026-10-29): the first call stopped during step 60 when the local Modal client lost the app
  (`APP_STATE_STOPPED`), before the adapter was saved. The same command, with `modal run --detach`, resumed
  from `checkpoint-50` (steps 51–60 rerun, their earlier log records dropped) and finished. After saving the
  adapter and `log_history.json`, the process segfaulted at interpreter exit, in vLLM's sleep-mode memory
  pool destructor. The saved adapter is byte-identical to `checkpoint-60`'s. Adapter digest
  `b0f0b32c0ecb5d344465dff9ebab703bb7234aa3077f909f5cec3c213b71c2af`.

**Training data.** `outputs/reasoning/v2/train.jsonl`:
- **Contents:** 4,000 tasks, 1,000 per family, built by
  `scripts/build_reasoning.py --out-dir outputs/reasoning/v2 --train 4000 --dev 400 --test 1000`.
- **SHA-256:** `35ae51803fa4a0437b950e8624aa2e79e0735e951a8c33c592670fd4b8ee13f6`.
- **Formats:** 75.5% yes/no (50.1% yes), 12.5% choice and 12% number.
- **Adjustment dose:** in 43% of its adjustment tasks a candidate set holds a descendant of the treatment.
  The pilot had about 10–14% of adjustment queries with a mediator.

**Control: shuffled rewards** (`--shuffle-rewards`). Each step's 128 rewards are permuted across all of the
step's rollouts, with a generator seeded by the run seed and the step.
- **What it keeps:** prompts, formats, sampling, each step's reward distribution, and compute.
- **What it does not keep:** the reward for finishing within 4,096 tokens and writing the answer line. The
  trained run is rewarded for both, since a truncated or unanswered reply scores 0; in the control no reply
  property is rewarded. So reply lengths and truncation may drift apart.
- **Whole-step check:** the reward function raises an error unless it sees the whole step at once.
- **What it rules out:** that a gain comes from running GRPO on these prompts with uninformative rewards,
  for example through drift in style. It does not rule out that a gain comes from learning to finish or to
  follow the answer format; the exploratory analyses below report this. Nor does it test whether generic
  non-causal reasoning RL would also help.
- **Earlier candidate:** a format-matched knights-and-knaves control was built and calibrated first, then
  dropped, as described above.

**Gate.** `v2-trained` is first compared with `v2-base` and, on the yes/no transfer sets, with chance. The
control is trained and evaluated only if Holm's procedure, applied to those p-values, passes at least one
endpoint. Adding a comparator can only raise an endpoint's p-value, so if none passes, none can be confirmed,
and the study reports no confirmed effect and stops. If the gate opens but the control cannot be trained
after resuming, no endpoint is confirmed.

## Evaluation protocol

- **Generation:** `scripts/modal_eval.py`, offline vLLM 0.31.0 on one L40S. Each prompt is one user message,
  with no system prompt and no tools.
- **Model:** revision `cdbee75` for weights and tokenizer (`--revision`).
- **Sampling:** the model's generation-config defaults (temperature 0.7, top-p 0.8, top-k 20, min-p 0) and at most
  **8,192 new tokens**. The base-model ladder showed that 4,096 tokens truncates a quarter of `ood_size`
  replies, which would reward brevity rather than reasoning.
- **Seeds:** sample j of every item uses seed 1000 + j, so dev evaluations (seed 0) are never reused.
  vLLM's batched arithmetic is not bit-reproducible: rerunning a seed reproduces a reply's opening, then
  diverges. So results replicate in distribution only.
- **Adapters:** served with vLLM LoRA, rank 32. A greedy check confirmed that the LoRA path changes outputs:
  9 of 24 continuations matched base with the adapter, against 21 of 24 between two base runs.
- **Records:** every reply records:
  - the base model, its revision and the adapter path;
  - the adapter digest (config and weights);
  - the token limit, the sampling parameters and the vLLM version;
  - the SHA-256 of the prompt it answered.
- **Answer rules:**
  - v2 test splits: `causalagentbench.reasoning.grading.grade`, the last "Answer:" line in the task's
    format.
  - Transfer sets: `transfer_eval.extract`, the last "answer … yes/no", else a final line that is only yes
    or no.
  - No answer scores 0. A reply that stopped at the token limit counts as no answer, even if it already
    contains an answer line.
- **Collection:** for each arm, the same three calls, then `python scripts/confirm_v2.py --record <arm>` and
  a commit of `docs/preregistration_v2_records.json`. That file holds the hash of each arm's replies on each
  set. It refuses to overwrite an entry, and the analysis refuses replies that do not match it. Replies are
  not looked at before they are recorded; the manifest cannot detect replies collected, discarded and collected
  again before recording, so this rests on the commit history.

  ```bash
  uv run --with modal modal run scripts/modal_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --name v2-base \
      --sets iid_test,ood_wording_test,ood_vocab_test,ood_size_test,dsep --seed 1000 --max-tokens 8192 \
      --revision cdbee75f17c01a7cc42f958dc650907174af0554
  uv run --with modal modal run scripts/modal_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --name v2-base \
      --sets c2c_test --samples 2 --seed 1000 --max-tokens 8192 \
      --revision cdbee75f17c01a7cc42f958dc650907174af0554
  uv run --with modal modal run scripts/modal_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --name v2-base \
      --sets cladder_adjust --samples 4 --seed 1000 --max-tokens 8192 \
      --revision cdbee75f17c01a7cc42f958dc650907174af0554
  ```

  `v2-trained` and `v2-control` add `--adapter reason-v2/adapter` or `--adapter control-v2/adapter`.
  Before the first call, a smoke run with the same settings and the pilot adapter `reason60` on `pilot_c2c`
  (100 Corr2Cause dev items it has answered before), under the name `smoke-v2`, checks that the records carry
  the registered vLLM version, revision and sampling, and an adapter digest.

## Evaluation sets

Item-list hashes come from `python scripts/transfer_eval.py hash <set>`: the SHA-256 of each item's id,
prompt, label and answer format, in id order. `confirm_v2.py` asserts them.

| Set | Role | Items | Clusters | Samples | Metric |
| --- | --- | ---: | ---: | ---: | --- |
| `iid_test` | manipulation check | 1,000 | items | 1 | accuracy |
| `ood_wording_test` | held-out wording | 1,000 | items | 1 | accuracy |
| `ood_vocab_test` | held-out vocabulary | 1,000 | items | 1 | accuracy |
| `ood_size_test` | held-out graph sizes | 1,000 | items | 1 | accuracy |
| `dsep` | near transfer (independence) | 996 | 473 graphs | 1 | balanced accuracy over labels |
| `c2c_test` | near transfer (discovery) | 1,162 | items | 2 | balanced accuracy over 10 cells |
| `cladder_adjust` | far transfer (adjustment) | 832 | 29 stories | 4 | balanced accuracy over 4 cells |

```text
iid_test          9ca960158c0202b519ee0a10b0ff2918585428d51341951f5b502efd2f06e96c
ood_wording_test  f53cf292d94578d220ea6ed6b3f8d91209d5ef089cb02fcc856e1b02238e1a06
ood_vocab_test    57ede41b5f52608cfc84f075df4c2d089d2ca63233ad071a570f4f94cbffe615
ood_size_test     d81bee267be3679ae0e8246bd23f35d25bf9ef477694cfacfa7c5f17eb7ad335
dsep              daa0fc225333938fc4e3a4a8c87bfd834582157024b6495b9c6858dddad551b4
c2c_test          b81bdce10ce4a115ed382588969549f08af8c8baa9b4481aa6792cd3f99eb817
cladder_adjust    b84502525ddc570f19a6deec222906468c1f60ae02852de345f179d98b8ec452
```

- **v2 test splits:** `outputs/reasoning/v2/{split}_test.jsonl`, built as above (seed 200). No test prompt
  appears in train or dev.
  - Labels are balanced, and every constant answer and shortcut rule scores at chance (audit in
    `docs/environment_v2.md`).
  - So a shift in response bias cannot raise these scores.
  - `iid`, `ood_wording` and `ood_vocab` pose the same problems item by item.
- **d-separation** (Vashishtha et al., 2024, "Teaching Transformers Causal Reasoning through Axiomatic
  Training"):
  - **Source:** the 5- and 8-node files of `eval_dsep/dsep_eval_branching_factor14`, from
    github.com/AniketVashishtha/Causal_Axioms at commit `fbddd14`.
  - **Size:** 1,000 items, 250 yes and 250 no per file. 4 repeated questions are kept once, leaving 996
    (496 yes).
  - **Prompt:** the files' "Answer the following question with ONLY 'Yes' or 'No'." header is replaced by
    the final-line instruction used for CLadder and Corr2Cause.
  - **Labels:** every one agrees with d-separation in the graph re-parsed from the premise
    (`tests/test_transfer_scripts.py::test_dsep_labels_match_networkx`).
  - **Clustering:** a graph carries up to 9 questions, so the cluster is the graph.
  - **Excluded:** the 10- and 12-node files, to save cost.
  - **License:** the repository has none, so the files stay local.
- **Corr2Cause test:** `causal-nlp/corr2cause` at revision `42ba12c`.
  - **Size:** 1,162 items, 180 of them positive.
  - **Prompt:** the same as in `docs/transfer_preregistration.md`.
  - **Label errors:** 28 items (2.4%) have labels that disagree with exact Markov-equivalence semantics
    (research notes, section 3.4). They stay in. They bias every comparison toward zero, more so for a model
    that answers them correctly.
  - **Wording priors:** the labels' base rates differ by hypothesis wording. A rule that says "yes" for two
    templates and "no" otherwise scores 0.727 balanced over labels. Training could shift the yes-rate on some
    wordings without teaching item-level skill, since its labels are 50% yes and base says yes far less often
    here.
  - **Cells:** hypothesis wording (variable names masked) × label. The five wordings are collider,
    confounder, "directly causes" (shared by the parent and child templates), "causes something else
    which causes", and "a cause, but not a direct one". Positives per wording: 64, 18, 65, 22 and 11;
    negatives: 129, 175, 323, 173 and 182. Balanced accuracy averages the ten cells, so any answering habit
    that depends only on the wording scores exactly 0.5.
- **CLadder adjustment** (`cladder_adjust`): CLadder v1.5 at revision `18d684e`, query type `backadj`.
  - **Stories:** the 37 stories outside the 10 used for development. 29 of them contain the graphs below.
  - **Graphs:** chain, mediation, arrowhead, diamond and collision, where the adjustment set holds a
    mediator or a collider, so not adjusting is keyed (610 items, 24 stories). And confounding and
    diamondcut, where adjusting is keyed (222 items, 9 stories).
  - **Cells:** whether adjusting is keyed × label: 312, 298, 113 and 109 items. Balanced accuracy averages
    the four cells. So a model that never or always adjusts, or that always answers yes or always no, scores
    exactly 0.5.
  - **Labels:** they agree with both the back-door and the complete adjustment criterion.
  - **Excluded graphs, from a label audit that used no trained-model output:**
    - fork: both methods valid, yet one is keyed;
    - IV and frontdoor: they key adjustment for a variable the prompt calls unobserved;
    - `collider_bias`: labels follow the wording.

## Statistics

- **Estimand:** each arm's expected score on these fixed items. Replies are random; the items are not
  resampled. Claims are about these benchmarks' items and these adapters.
- **Per item:** mean correctness over its samples.
- **Per set and arm:**
  - test splits: the mean over items;
  - transfer sets: balanced accuracy, the unweighted mean of the accuracies within each cell (label for
    `dsep`, Corr2Cause's ten cells, CLadder's four cells).
- **Comparators:**
  - `v2-base` and `v2-control` on every endpoint;
  - on `dsep`, `c2c_test` and `cladder_adjust`, also chance: a score of 0.5 on every item.
- **Test:** a one-sided cluster sign-flip randomization test (`transfer_eval.permutation_p`).
  - Each of 20,000 resamples (numpy seed 0) swaps the two arms' scores in every cluster independently with
    probability 1/2.
  - p = (1 + number of resampled differences ≥ observed) / (1 + 20,000).
  - **Against an arm:** under the null that the two arms' replies have the same distribution within every
    cluster, the test is exact. In a simulation on `cladder_adjust`'s 29 unequal clusters with 4 samples per
    item, it rejected at 0.0235 for nominal 0.025 and at 0.004 for nominal 0.0042.
  - **Against chance:** the test assumes each cluster's score difference from 0.5 is symmetric about zero
    under the null. With the items fixed and only reply noise, a reviewer's simulation found it
    conservative. With 473 and 1,162 clusters, `dsep` and `c2c_test` are robust to this assumption. On
    `cladder_adjust` the base model is already at 0.621, so the comparison with base is the binding one.
- **Interval:** a 95% paired cluster-bootstrap percentile interval (20,000 resamples). It is descriptive and
  not used for decisions.
- **Endpoint p-value:** the largest over its comparators. This is an intersection-union test, so a confirmed
  endpoint beats every comparator.
- **Multiplicity:** Holm's step-down procedure over the six endpoints. Sort their p-values p_(1) ≤ … ≤ p_(6)
  and confirm endpoints in order while p_(k) ≤ 0.025 / (7 − k). The family-wise error is one-sided 0.025,
  the same as two-sided 0.05.
- **Inputs:** `confirm_v2.py` refuses to analyze unless all of these hold:
  - every arm has exactly the registered replies (each item × seed once);
  - every reply answered its item's registered prompt;
  - one configuration per arm across all sets, with the registered model and revision, adapter digest,
    8,192 tokens, sampling and vLLM 0.31.0;
  - replies that match the committed manifest.

## Decisions and scope

1. **Manipulation check.**
   - If Δ(`v2-trained` − `v2-base`) on `iid_test` fails the sign-flip test at one-sided 0.025, training did
     not work in distribution.
   - Confirmed endpoints still stand, since each test is valid on its own, but they are reported with that
     caveat.
   - A null is then uninformative about generalization.
2. **Held-out generalization:** each confirmed endpoint among `ood_wording_test`, `ood_vocab_test` and
   `ood_size_test`.
3. **Near transfer:** each confirmed endpoint among `dsep` and `c2c_test`.
4. **Far transfer:** `cladder_adjust` confirmed.

There is one training run per arm, so every claim is about these two adapters, not the training procedure
in general. A confirmed endpoint means the trained adapter beats the base model, beats the same training with
uninformative rewards, and, on yes/no transfer sets, beats chance. An endpoint that is not confirmed is
reported as "not confirmed", with its interval, not as evidence of no effect.

## Exploratory analyses

These are reported, and none supports a claim:
- each test split by family, answer format, variant and number of variables;
- reply length, truncation rate and no-answer rate per arm and set, reported next to each confirmed endpoint.
  A trained model may answer CLadder in the environment's choice format ("Answer: 1"), which scores as no
  answer;
- each confirmed endpoint re-analyzed on items where every sample of both arms finished with an answer, to
  separate reasoning from finishing;
- yes-rate on every yes/no set;
- `cladder_adjust` by cell, and the mediator/collider items alone;
- Corr2Cause yes-rate and balanced accuracy within each wording and template, over answered items only, and
  without the 28 audited items;
- `dsep` split by whether X and Y are adjacent. Adjacent pairs are never d-separated, and "yes if and only if
  not adjacent" scores 0.828 here (0.726 on the training independence questions), so a gain could come from
  that partly valid rule. Balanced accuracy on non-adjacent pairs alone (496 yes, 172 no) isolates the rest;
- the gap between each held-out split and `iid`, for every arm: does training widen it?
- training reward and accuracy by family over steps, for both runs;
- if budget remains after the confirmatory evaluation, with 2 samples each:
  - the remaining CLadder `backadj` graphs and `collider_bias` on held-out stories;
  - the 10- and 12-node d-separation files.

## Power (estimates)

- **Test splits:** with 1,000 single-sample items, the standard error of Δ is about 1.5 points if base and
  trained answers correlate across items as they did in the pilot, and 2.1 points if they were independent.
  The smallest effect with 80% power is then:
  - about 5–7 points at Holm's strictest level, 0.025 / 6;
  - about 4–6 points at its loosest, 0.025.
- **`dsep`:** about the same.
- **Corr2Cause:** only 180 positives, and the ten-cell score weights small cells (11, 18, 22 positives)
  heavily: its standard error is about 24% larger than balanced over labels, so about 7.5–10 points.
- **CLadder:** a simulation of the sign-flip test on `cladder_adjust` gave 71% power at the strictest level
  for a 4-point gain in balanced accuracy, and 99% for 7.5 points. It had no story-level variation in the
  effect, so real power is lower.

The pilot's in-distribution gain was 6.6 points, so smaller held-out or transfer effects may be missed.

## Budget

| Item | Cost (approximate) |
| --- | ---: |
| `reason-v2` training (60 steps × about 200 s on an H100) | $14 |
| Evaluation of `v2-base` and `v2-trained` (about 20M generated tokens each on an L40S) | $14 |
| If the gate opens: `control-v2` training and evaluation | $21 |
| **Total** | **$28, or $49 with the control** |

## Known limitations

- **One training run per arm:** claims are about these adapters, and run-to-run variation is not measured.
- **Control scope:** the control rules out GRPO on these prompts without informative rewards. It does not
  rule out that a non-causal reasoning task with informative rewards would help as much.
- **Contamination:** CLadder and Corr2Cause date from 2023. The d-separation repository dates from 2025 and
  may be in Qwen3's training data. Contamination affects every arm alike, but it compresses headroom.
- **Reproducibility:** replies are not bit-reproducible (see the evaluation protocol).
- **Effect family:** no independent near-transfer set exists for it that the base model hasn't already
  saturated.
- **Held out in wording, not in problems:** a reviewer found that 211 of the 1,000 items in each of
  `iid_test`, `ood_wording_test` and `ood_vocab_test` pose the same graph, query and answer as some training
  task under other names or wording. `ood_size_test` has none. Prompts are never repeated.
- **Token limits:** evaluation uses 8,192 tokens and training 4,096. Every arm is evaluated the same way.
- **CLadder design:** the endpoint was designed with base, and the earlier study's adapters', CLadder results
  in view. Its cell-balanced score removes the response-bias route those results revealed, but the design
  is not blind.
