# Environment v2 training study: pre-registration

Does GRPO on the causal-reasoning environment (`docs/environment_v2.md`) teach skills that hold beyond its
training distribution? This file fixes the training runs, evaluation sets, protocol, statistics and decision
rules before any trained model is evaluated on any of them. The code that applies it is
`scripts/confirm_v2.py`, committed with this file.

## What has been seen before registration

- **Environment v2:** base models on the four dev splits only (`docs/environment_v2.md`, base-model ladder).
  No model has answered a v2 test item. The test splits were enlarged from 400 to 1,000 items per split
  before any evaluation; the first 400 items of each are the original files.
- **Transfer sets:** the base model answered Corr2Cause test (2 samples) and all of CLadder `backadj` and
  `collider_bias` (4 samples) in the earlier tool-use study, at 4,096 tokens through `modal_serve.py`. The
  CLadder endpoint below was chosen after that, using a label audit and those base replies
  (`research_notes/Transfer target selection/notes.md`, section 3.1). Its base score (0.347 over 610 items)
  is therefore known. Every reply in this study is newly sampled, under a different protocol and seeds.
  No model has answered the d-separation set.
- **No trained model** of any kind has been evaluated on any set below. The pilot adapter `reason60` was
  evaluated only on sets outside this study: the v1 reasoning dev set, CLadder dev stories, and 100
  Corr2Cause dev items in a smoke test of the evaluator.

## Arms

All three arms are Qwen3-4B-Instruct-2507.

| Name | What it is |
| --- | --- |
| `v2-base` | The untrained model |
| `v2-trained` | LoRA adapter after 60 GRPO steps on the v2 training set (run `reason-v2`) |
| `v2-control` | The same configuration, run on knights-and-knaves logic puzzles (run `control-v2`); trained only if the gate below opens |

**Training configuration (both runs).** `scripts/train_reasoning.py` through
`scripts/modal_grpo.py::reasoning_main` on one H100:
- 60 steps, 16 tasks × 8 replies per step, learning rate 2e-5 (constant), LoRA rank 32 on all linear
  layers;
- at most 4,096 new tokens, temperature 1.0;
- DAPO loss, β = 0, truncated replies masked from the loss;
- vLLM colocated with 35% of GPU memory, seed 0.

The adapter at step 60 is the arm; no checkpoint is selected. A preempted run resumes from its last
checkpoint.

**Training data.** `outputs/reasoning/v2/train.jsonl`: 4,000 tasks, 1,000 per family, built by
`scripts/build_reasoning.py --out-dir outputs/reasoning/v2 --train 4000 --dev 400 --test 1000`. Its SHA-256
is `35ae51803fa4a0437b950e8624aa2e79e0735e951a8c33c592670fd4b8ee13f6`. In 43% of its adjustment tasks a
candidate set holds a descendant of the treatment (a mediator, a collider or a node downstream of them). The
research notes counted 10–14% of the pilot's adjustment queries with a mediator.

**Control task: knights and knaves.** The control asks whether a gain comes from training on causal
structure, or from any GRPO run with a verifiable reward, the same answer-line format and the same signal
density.
- **Puzzle:** n inhabitants of an island, each a knight (always truthful) or a knave (always lying), named
  with the environment's nonsense-name generator.
- **Statements:** each inhabitant makes one statement about one or two others, drawn uniformly from these
  forms:
  - "X is a knight" or "X is a knave";
  - "X and Y are both knights";
  - "X or Y is a knave";
  - "X and Y are the same kind" or "X and Y are different kinds";
  - "if X is a knight, Y is a knave".
- **Solutions:** puzzles without exactly one consistent assignment, found by enumerating all 2^n
  assignments, are redrawn.
- **Question and answer:** the question is "Which of them are knights?". The answer is the set of knights'
  names, or "None", graded as set equality on the last "Answer:" line.
- **Prompt:** the environment's instruction, "Reason step by step, then end with a final line "Answer:
  <answer>", where <answer> is …".
- **Training set:** 4,000 puzzles, seed 0, with n drawn uniformly from one of the ranges 3–5, 4–6 or 5–7.
- **Choosing the range:** before training, the base model answers 200 puzzles per range (seed 100, one
  sample, 4,096 tokens). The range whose base accuracy is closest to 0.62 is used: 0.62 is the base
  accuracy on v2 `iid` dev at 4,096 tokens. Within that rule the choice uses no outcome of any evaluation
  set.

**Gate.** `v2-trained` is evaluated against `v2-base` first. The control is trained and evaluated only if
Holm's procedure (below), applied to the p-values against base, confirms at least one endpoint. Each
endpoint's final p-value is the larger of its two p-values, so if no endpoint passes against base, none can
pass at all. Then the study reports no confirmed effect and stops.

## Evaluation protocol

- **Generation:** `scripts/modal_eval.py`, offline vLLM 0.31.0 on one L40S. Each prompt is one user
  message, with no system prompt and no tools.
- **Sampling:** the model's generation-config defaults (temperature 0.7, top-p 0.8, top-k 20) and at most
  **8,192 new tokens**. The base-model ladder showed that 4,096 tokens truncates a quarter of `ood_size`
  replies, which would reward brevity rather than reasoning.
- **Seeds:** sample j of every item uses seed 1000 + j, so dev evaluations (seed 0) are never reused.
  vLLM's batched arithmetic is not bit-reproducible: rerunning the same seed reproduces a reply's opening
  and then diverges. So results replicate in distribution only.
- **Adapters:** served with vLLM LoRA, rank 32. A greedy check confirmed that the LoRA path changes outputs:
  9 of 24 continuations matched base with the adapter, against 21 of 24 between two base runs.
- **Answer rules:**
  - v2 test splits: `causalagentbench.reasoning.grading.grade`, the last "Answer:" line in the task's
    format.
  - Transfer sets: `transfer_eval.extract`, the last "answer … yes/no", else a final line that is only yes
    or no.
  - No answer, including a truncated reply, scores 0.
- **Commands:** for each arm, one call covering every set at its sample count.

  ```bash
  uv run --with modal modal run scripts/modal_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --name v2-base \
      --sets iid_test,ood_wording_test,ood_vocab_test,ood_size_test,dsep --seed 1000 --max-tokens 8192
  uv run --with modal modal run scripts/modal_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --name v2-base \
      --sets c2c_test --samples 2 --seed 1000 --max-tokens 8192
  uv run --with modal modal run scripts/modal_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --name v2-base \
      --sets cladder_medcol --samples 4 --seed 1000 --max-tokens 8192
  ```

  `v2-trained` and `v2-control` use the same commands with `--adapter reason-v2/adapter` or
  `--adapter control-v2/adapter`.

## Evaluation sets

Item-list hashes are from `python scripts/transfer_eval.py hash <set>`: the SHA-256 of each item's id,
prompt and label, in id order.

| Set | Role | Items | Clusters | Samples | Metric | Hash (first 16) |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `iid_test` | manipulation check | 1,000 | items | 1 | accuracy | `3679151d10497dfc` |
| `ood_wording_test` | held-out wording | 1,000 | items | 1 | accuracy | `083469f477f71f2a` |
| `ood_vocab_test` | held-out vocabulary | 1,000 | items | 1 | accuracy | `c65c678c8bc5ad5d` |
| `ood_size_test` | held-out graph sizes | 1,000 | items | 1 | accuracy | `f00ad3690c09caf5` |
| `dsep` | near transfer (independence) | 996 | 473 graphs | 1 | accuracy | `47210b28d3a9ca4c` |
| `c2c_test` | near transfer (discovery) | 1,162 | items | 2 | balanced accuracy | `42fc523e63197706` |
| `cladder_medcol` | far transfer (adjustment) | 610 | 24 stories | 4 | accuracy | `d53b9075c80b3456` |

Full hashes:

```text
iid_test          3679151d10497dfca0ebfaf190608a76b9fdf0aa5e68580b82a6b550f81fbb34
ood_wording_test  083469f477f71f2a559d652bc294a970317c63d1fdd56fe383e2d9efe7d1b1c5
ood_vocab_test    c65c678c8bc5ad5d4c521a003a4c49cb94a05b53264d36e973fe0e93353d9163
ood_size_test     f00ad3690c09caf5e21ca1c7113861df87ebade8de4b26cce8da2cc101fb1a12
dsep              47210b28d3a9ca4c81a70e23150876d42ac644ce93190ed8e07eec31b921a2a8
c2c_test          42fc523e631977066a4c3c26ff21e9c54fb1d0a2af4e307396d4e049b78af4f6
cladder_medcol    d53b9075c80b3456e311d2aed5a04d5993cab4f313b1ee8ad0039139b59207e7
```

- **v2 test splits:** `outputs/reasoning/v2/{split}_test.jsonl`, built as above (seed 200). No test prompt
  appears in train or dev.
- **d-separation** (Vashishtha et al., 2024, "Teaching Transformers Causal Reasoning through Axiomatic
  Training"):
  - **Source and size:** the 5- and 8-node files of `eval_dsep/dsep_eval_branching_factor14` from
    github.com/AniketVashishtha/Causal_Axioms at commit `fbddd14`. That is 1,000 items, 250 yes and 250 no
    per file; 4 repeated questions are kept once, leaving 996.
  - **Prompt:** the files' "Answer the following question with ONLY 'Yes' or 'No'." header is replaced by
    the final-line instruction used for CLadder and Corr2Cause.
  - **Labels:** every one agrees with d-separation in the graph re-parsed from the premise
    (`tests/test_transfer_scripts.py::test_dsep_labels_match_networkx`).
  - **Clustering:** a graph carries up to 9 questions, so the bootstrap resamples graphs.
  - **Exclusions:** the 10- and 12-node files are left out to save cost.
  - **License:** the repository has none, so the files stay local.
- **Corr2Cause test:** `causal-nlp/corr2cause` at revision `42ba12c`, 1,162 items, 180 positive, with the
  same prompt as in `docs/transfer_preregistration.md`.
  - **Metric:** balanced accuracy, because the labels are unbalanced.
  - **Label errors:** 28 items (2.4%) have labels that disagree with exact Markov-equivalence semantics
    (research notes, section 3.4). They stay in, because errors affect every arm alike.
- **CLadder mediator/collider** (`cladder_medcol`): CLadder v1.5 at revision `18d684e`, query type
  `backadj`.
  - **Graphs:** chain, mediation, arrowhead, diamond and collision.
  - **Stories:** the 37 stories outside the 10 used for development, of which 24 have these graphs; 610
    items, 298 labelled yes.
  - **Labels:** in these graphs the adjustment set Z holds a mediator or a collider. Their labels agree with
    both the back-door criterion and the complete adjustment criterion.
  - **Excluded graphs, from a label audit that used no trained-model output:**
    - fork: both methods valid, yet one is keyed;
    - IV and frontdoor: they key adjustment for a variable the prompt calls unobserved;
    - `collider_bias`: labels follow the wording.
    - confounding and diamondcut: the base model already scores higher on confounder-type graphs (0.71 on
      held-out stories and 0.91 on dev, pooled with IV and frontdoor). They test whether training stops
      adjusting for confounders, a different question.
  - **Disclosure:** this subset was chosen after the pilot's CLadder null, with the base model's scores in
    view.

## Statistics

- **Per item:** mean correctness over its samples. Per set and arm: mean over items, or balanced accuracy
  (the mean of the accuracies on positive and on negative items) for Corr2Cause.
- **Difference:** Δ = `v2-trained` − comparator, with a paired bootstrap over clusters: 20,000 resamples,
  `random.Random(0)`. The 95% percentile interval is reported.
- **One-sided p-value:** (1 + number of resampled Δ ≤ 0) / (1 + 20,000).
- **Endpoint p-value:** the larger of the p-values against `v2-base` and against `v2-control`. This is an
  intersection-union test, so a confirmed endpoint beats both comparators.
- **Multiplicity:** Holm's step-down procedure over the six endpoints. Sort their p-values p_(1) ≤ … ≤ p_(6)
  and confirm endpoints in order while p_(k) ≤ 0.025 / (7 − k). The family-wise error is one-sided 0.025,
  the same as two-sided 0.05.
- **Completeness:** every arm must have exactly the pre-registered replies (each item × seed once);
  `confirm_v2.py` refuses to analyze otherwise.

## Decisions

1. **Manipulation check.** If Δ(`v2-trained` − `v2-base`) on `iid_test` has a 95% interval not above zero,
   training did not work. The endpoints are still reported, but a null is then uninformative about
   generalization.
2. **Held-out generalization:** each confirmed endpoint among `ood_wording_test`, `ood_vocab_test` and
   `ood_size_test`.
3. **Near transfer:** each confirmed endpoint among `dsep` and `c2c_test`.
4. **Far transfer:** `cladder_medcol` confirmed.

An endpoint that is not confirmed is reported as "not confirmed", with its interval, not as evidence of no
effect.

## Exploratory analyses

These are reported, and none supports a claim:
- each split by family, answer format, variant and number of variables;
- reply length and truncation rate per arm and set;
- yes-rate on every yes/no set;
- the gap between each held-out split and `iid` for every arm. Does training widen it?
- training reward and accuracy by family over steps;
- if budget remains after the confirmatory evaluation, with 2 samples each:
  - the remaining CLadder `backadj` graphs and `collider_bias` on held-out stories;
  - the 10- and 12-node d-separation files;
  - Corr2Cause without the 28 audited items.

## Power (estimates)

With 1,000 single-sample items, the standard error of Δ is about 1.5 points if base and trained answers
correlate across items as they did in the pilot, and 2.1 points if they were independent. The smallest
effect with 80% power is then about 5–7 points at Holm's strictest threshold, 0.025 / 6, and 4–6 points at
its loosest, 0.025. The same holds roughly for `dsep`. Corr2Cause has only 180 positives, so its threshold
is about 6–8 points. CLadder has 24 story clusters, so its threshold is about 5–8 points (research notes,
section 3.5). The pilot's in-distribution gain was 6.6 points, so smaller held-out or transfer effects may
be missed.

## Budget

| Item | Cost (approximate) |
| --- | ---: |
| `reason-v2` training (60 steps × about 200 s on an H100) | $14 |
| Evaluation of `v2-base` and `v2-trained` (about 18M generated tokens each on an L40S) | $13 |
| If the gate opens: control range check, training and evaluation | $21 |
| **Total** | **$27, or $48 with the control** |

## Known limitations

- **Seeds:** one training seed per arm, so run-to-run training variance is not measured.
- **Contamination:** CLadder and Corr2Cause date from 2023. The d-separation repository dates from 2025
  and may be in Qwen3's training data. Any contamination affects every arm alike, but it compresses
  headroom.
- **Reproducibility:** replies are not bit-reproducible (see the evaluation protocol).
- **Effect family:** no independent near-transfer set exists for it that the base model hasn't already
  saturated.
- **Training budget:** evaluation uses 8,192 tokens and training 4,096. Every arm is evaluated the same
  way, but a trained model's replies may differ in length from the ones it was trained on.
