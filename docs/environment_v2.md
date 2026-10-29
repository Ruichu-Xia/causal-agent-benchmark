# Causal-reasoning environment v2: design

Written before implementation, then updated with the decisions and measurements made while building it. It records what v2 changes, why, and how each change is verified.

## Why v2

The pilot (`docs/reasoning_pilot.md`) trained Qwen3-4B-Instruct-2507 for 60 GRPO steps on v1. In-domain
dev accuracy rose from 0.574 to 0.737, but CLadder did not move (decision: rethink). The diagnosis found
three problems in the environment itself:

1. **One wording per question.** The trained gain on CLadder's adjustment graphs was +7.6 points
   [+3.2, +12.8] when asked in our template and +2.7 [−1.0, +6.8] in CLadder's own wording. The skill
   was tied to our template.
2. **Shortcuts pay.** The base model follows a "condition on more" or "adjust for the confounder"
   heuristic. On randomly drawn adjustment sets that heuristic is often right, so reward did not push
   against it. In our own wording, both models still failed graphs where the heuristic is wrong (a back
   door blocked by a non-confounder: base 0.12, trained 0.04).
3. **Adjustment did not learn.** Its training accuracy was flat over 60 steps.

The environment's claims are now: correct, shortcut-resistant rewards; calibrated difficulty; and
generalization to held-out wordings, graph sizes and vocabulary, then to independent benchmarks.

## Changes

### 1. Wording: many templates per slot, some held out

Every phrase the model reads that is not a variable name comes from a slot with several templates:
the graph description style, the closed-world sentence, each family's question and each method
description. Templates within a slot are exact paraphrases: every one states the estimand explicitly
(the total causal effect under intervention, or statistical dependence). None is ambiguous in the way
CLadder's "to understand how X affects Y" is.

In every slot, about a third of the templates are held out. Training and i.i.d. test data use only the
others; the held-out-wording split uses only the held-out ones.

No template contains phrases specific to the far-transfer benchmark ("case by case", "Method 1", "more
correct", "self-contained, hypothetical world"). A test enforces this, so CLadder stays far transfer.

### 2. Shortcut balance

Each family names the most plausible shortcut for its question (`reasoning/shortcuts.py`):

| Family | Shortcut | Where it fails |
| --- | --- | --- |
| independence | undirected separation: X and Y are independent given Z if and only if every path in the skeleton passes through Z | colliders and their descendants |
| adjustment | confounder rule: a set is valid if and only if it contains every common ancestor of X and Y and no descendant of X | back doors blocked by non-confounders, M-bias, harmless descendants of X |
| effect (causal) | report the observed difference as the causal effect | confounding |
| effect (sign) | the causal effect has the sign of the observed difference | confounding that reverses the sign |
| discovery | a fixed answer per question type (e.g. "must" → no) | everywhere, once labels are balanced within each type |

Sampling draws the label and whether the shortcut agrees with it independently, by rejection
sampling. The label is a fair coin and agreement has probability `Spec.agree` (0.5), so each of the
four cells has probability 1/4 and the shortcut scores 0.5. For adjustment the cells apply to each
queried set. Numeric effect questions balance two things: the observed-difference shortcut, and a
zero answer, which is held at 10% (`P_NONZERO`) so that answering "0" is not a shortcut either.

Uniform cells change the graph distribution on purpose. The rarest cell, an invalid set that the
confounder rule accepts (M-bias-like), occurs in 0.8% of random training-size graphs but makes up a
quarter of adjustment queries. Rejection sampling is exact, so its budget (`MAX_TRIES` = 100,000) is
set far above the measured worst case.

### 3. Splits

| Split | Graph sizes | Wording | Concept names |
| --- | --- | --- | --- |
| train | training range | training templates | training concepts |
| iid | training range | training templates | training concepts |
| ood_wording | training range | held-out templates | training concepts |
| ood_size | larger | training templates | training concepts |
| ood_vocab | training range | training templates | held-out concepts |

Two random streams drive generation: one draws the problem (graph, query, answer), the other the
surface (templates, statement order, names). Built with the same seed, `iid`, `ood_wording` and
`ood_vocab` therefore pose the same problems item by item, so their accuracy differences measure
wording and vocabulary alone, as paired comparisons.

Concept names are split by a stable hash: 80% train, 20% held out. Nonsense names are drawn fresh in
every split, so `ood_vocab` uses held-out concepts only. Names that would read ambiguously are
excluded: those containing "and" or "or" ("wear and tear" would break name lists), and any name containing a
word of the task text, derived from the templates so it stays correct when they change. Otherwise
"failure to do" makes "an arrow from {p} to {c}" parse two ways, and "climate change" collides with
"change the probability". The pool keeps 2,516 of the 2,642 extracted concepts. The concept pool ships with the package.

| Family | Training sizes | ood_size sizes |
| --- | --- | --- |
| independence | 3–6 | 7–8 |
| adjustment | 4–6 | 7–8 |
| effect | 3–4 (gap questions: 3) | 5 (gap questions: 4) |
| discovery | 3–4 | 5 |

Effect and discovery get one extra node only, because their prompts and replies grow exponentially
with graph size (CPT rows; listed independences).

`scripts/build_reasoning.py` writes `train.jsonl` (seed 0) and, for every evaluation split, a `dev` set
(seed 100) for development decisions and a `test` set (seed 200) reserved for the pre-registered
evaluation. An item whose prompt appeared in train, or in dev, is dropped from every split of its
role, which keeps the pairing.

### 4. Precise questions

v1 asked whether comparing units "within groups that share the same values of Z" gives an "unbiased
estimate". v2 states the procedure (compare within groups of Z, then average over the population
distribution of Z) and asks whether it recovers the causal effect with unlimited data. That is
exactly the adjustment formula, which the adjustment criterion characterizes. The criterion stays the
complete one (Perković et al.). A set containing a harmless descendant of X is valid. Pearl's back-door
criterion would call it invalid, so benchmarks that grade by the back-door criterion differ from us on
those items.

Adjustment prompts now also state that effects never cancel exactly. Without that, an invalid set
could be unbiased by coincidence, and "no" would mean only "not guaranteed".

The multiple-select adjustment variant ("which of these sets are valid?") was dropped. Its options
cannot each be balanced against the shortcut without heavy rejection, and the two-analyst comparison
already tests pairwise judgement. Answer formats are yes/no, a two-decimal number, and 1/2/Both/Neither.

### 5. Grader fix

The pilot's grader let whitespace after "Answer:" cross line breaks, so a reply ending
"Final answer:\n\nAnswer: Yes" scored 0. The fixed pattern requires the value on the answer line and
allows markdown emphasis around the label. The fix raised the pilot's base reasoning-dev accuracy
from 0.574 to 0.681 (`docs/reasoning_pilot.md`). Regression cases are in `tests/test_reasoning.py`.

## Verification

- **Brute force.** The adjustment criterion and d-separation agree with exact probability computation
  in random worlds of 3–5 and of 7 variables.
- **Answers depend only on the problem.** The paired-split test checks that, with one seed, `iid`,
  `ood_wording` and `ood_vocab` give identical problems and answers and different prompts.
- **Prompts state the problem exactly.** Round-trip tests parse every prompt back into data and compare
  with the stored problem:
  - the graph, under every description style;
  - the probability tables, recomputing every effect answer from the parsed tables;
  - the independence list, checking every d-separation is stated and nothing else;
  - the variables each question names.
- **Split hygiene.** Tests check that held-out template fragments appear in every `ood_wording` prompt
  and in no other split, that concept names fall on the right side of the hash split, and that no
  template contains a far-transfer benchmark phrase.
- **Audit** (`scripts/audit_reasoning.py`, 10,000 tasks per evaluation split, seed 11). It re-derives
  every label from the stored problem with the verifier functions, independently of the sampling code,
  and scores every shortcut and constant answer. Thresholds:
  - every shortcut and constant answer within [0.45, 0.55] on yes/no questions;
  - discovery's best constant within each statement type and mode at most 0.56;
  - best constant and shortcut at most 0.35 on choice questions;
  - best constant at most 0.15 on numeric questions.

  At 10,000 tasks a yes/no group has 1,250–2,500 items, so ±0.05 is at least 3.5 standard errors; a
  2,000-task audit failed one bound by chance at 2.1 standard errors. Result: 40,000 tasks, 0 label
  mismatches, all checks passed. Yes/no shortcuts and constants scored 0.49–0.51, discovery per cell
  at most 0.525, choice at most 0.27, and numeric best constant at most 0.144.
  (`outputs/reasoning/audit_v2.txt`).

## Verifiers environment

`environments/causal_reasoning` is a Verifiers 0.4.0 (`verifiers.v1`) taskset package: a `Taskset` that
generates one split from its config (`split`, `seed`, `num_tasks`, `families`); a `Task` with the
reward `correct` (the same `grade` function) and the metrics `answered` and `truncated`; and a
`NullHarness` subclass, so the default harness is plain chat with no tools. A model with a shell could
compute d-separation in Python, which would test a different skill.

`vf-eval` ran end to end against the Modal vLLM endpoint with a local subprocess runtime
(`--env.agent.runtime.type subprocess`). Rewards on completed rollouts matched an independent re-grade.
That run also found the grader bug above. Long streamed replies through the Modal web endpoint are cut
at about 150 seconds, and a cancelled input stops the serving container. Bulk evaluation therefore uses
`scripts/transfer_eval.py`, which does not stream, with the same prompts and grader.

## Base-model ladder (dev splits)

Run on 2026-10-28 with `scripts/modal_eval.py` (offline vLLM 0.31.0 on one L40S, each model's
generation-config sampling, seed 0, one sample per item, hybrid Qwen3 models with thinking off) and
summarized by `scripts/ladder_report.py`. Each split has 400 items, so a split's accuracy has a standard
error of about 2.4 points; a family-by-size cell has 20 to 100 items.

| Model | Max tokens | iid | ood_wording | ood_vocab | ood_size | Truncated (iid / ood_size) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-0.6B | 4096 | 0.45 | 0.43 | 0.46 | 0.41 | 0% / 0% |
| Qwen3-1.7B | 4096 | 0.52 | 0.49 | 0.52 | 0.51 | 0% / 0% |
| Qwen3-8B | 4096 | 0.62 | 0.62 | 0.60 | 0.54 | 1% / 3% |
| Qwen3-4B-Instruct-2507 | 4096 | 0.62 | 0.61 | 0.64 | 0.45 | 12% / 25% |
| Qwen3-4B-Instruct-2507 | 8192 | 0.67 | 0.63 | 0.69 | 0.64 | 0% / 2% |

Qwen3-4B-Instruct-2507 (the model we train) by family and number of variables, 8,192 tokens, `iid` and
`ood_size` pooled:

| Family | Accuracy by number of variables |
| --- | --- |
| effect | 3: 0.95, 4: 0.88, 5: 0.75 |
| independence | 3: 0.84, 4: 0.68, 5: 0.62, 6: 0.26, 7: 0.56, 8: 0.67 |
| adjustment | 4: 0.63, 5: 0.59, 6: 0.37, 7: 0.60, 8: 0.45 |
| discovery | 3: 0.62, 4: 0.65, 5: 0.66 |

Findings:

1. **The 4,096-token cap made `ood_size` look hard.** At 4,096 tokens a quarter of the 4B model's
   `ood_size` replies are cut off (36% in effect, 52% in discovery); with 8,192 tokens, 2% are, and
   accuracy rises from 0.45 to 0.64. GRPO shortens replies, so under the 4,096 cap a trained model would
   gain on `ood_size` partly by being brief. Held-out evaluation therefore uses 8,192 tokens for every
   model (95th percentile of reply length: 5,374 tokens on `iid`, 6,724 on `ood_size`); training keeps the
   4,096 cap.
2. **No split is saturated, and no split is at chance for the 4B model:** 0.63 to 0.69 on every split.
   The effect family is near ceiling at training sizes (0.86 to 0.97 per variant), so it carries the
   least training signal.
3. **Accuracy falls with size for effect, and for independence up to 6 variables; not for discovery or
   adjustment.** Independence at 7 and 8 variables (0.56, 0.67) is above 6 (0.26, 23 items); this may be
   noise or a property of how larger independence questions are sampled, and is unresolved. `ood_size`
   is therefore a generalization test for larger graphs, not a uniformly harder split.
4. **Wording and vocabulary barely move the base model:** `ood_wording` is 4 points below `iid` and
   `ood_vocab` 2 points above, both within noise. These splits test whether training overfits to the
   training templates and names.
5. **Smaller models are at chance on structure.** Qwen3-0.6B and 1.7B score 0.4 to 0.6 on every
   independence, adjustment and discovery cell, with balanced yes-rates; only effect questions are above
   chance. Qwen3-1.7B is not a viable cheaper training model: GRPO needs correct samples in a group.
6. **Format compliance is not a confound for the 4B model.** None of its untruncated replies lacks an
   answer line; the smaller models miss it in 2% (8B) to 6% (0.6B) of replies. The smoke test showed small models echoing the `<answer>` placeholder ("Answer: <answer> Yes"); the
   grader now strips those tags (regression tests added). It affected 5 of 3,200 pilot replies.

Training signal: with group size 8 and pass rate p, a group carries a gradient with probability
1 − p^8 − (1 − p)^8. From the 4B model's `iid` accuracy at 4,096 tokens, that is 0.99 for adjustment and
discovery, 0.96 for independence and 0.73 for effect. Item-level pass-rate filtering would need about
32,000 training-set samples, so we keep the balanced family mix instead.

Training throughput (4 GRPO steps of the pilot configuration on the v2 training set): about 200 s per step
on an H100 with vLLM given 35% or 70% of GPU memory (no difference), and about 380 s on an A100-80GB.
At Modal's prices the H100 is about 20% cheaper per step, so runs stay on the H100 at about $14 per
60 steps.

## Next steps

1. Run the pre-registered training study (`docs/preregistration_v2.md`). It evaluates held-out splits at
   8,192 tokens, with test splits enlarged to 1,000 items each, and near transfer to the axiomatic
   d-separation set and Corr2Cause test. Far transfer is CLadder backdoor adjustment on mediator and collider
   graphs in held-out stories. A knights-and-knaves GRPO control runs only if an endpoint passes against
   the base model. CLEAR was left out: its gradable yes/no and multiple-choice items on DAGs are too few.
2. Port the agentic CausalAgentBench environment to Verifiers.
