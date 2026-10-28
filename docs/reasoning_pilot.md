# Text-reasoning pilot: plan

Written before training. The pilot decides whether a text causal-reasoning environment is worth building
out; it is not the confirmatory study.

## Environment

`src/causalagentbench/reasoning`: small random causal worlds (3–6 binary variables) described in words,
named with nonsense words or random CauseNet concepts. Four families with exact answers: independence
(d-separation), adjustment (the adjustment criterion), effect (causal and observed effects from stated
probabilities) and discovery (what all independences imply about the structure). Wording, graph
description style and answer format vary; yes/no labels are balanced. The verifiers are checked against
brute-force probability computation (`tests/test_reasoning.py`).

## Data

- Train: `build_reasoning.py --n 4000 --seed 0`. Dev: `--n 400 --seed 1` (no prompt in common).
- CLadder dev: 10 of the 47 CLadder v1.5 stories, drawn with seed 0 (`transfer_eval.py`, set
  `cladder_dev`, 2,857 items, all query types). The other 37 stories are reserved for the confirmatory study
  and are not evaluated during development.

## Training

Qwen3-4B-Instruct-2507, LoRA r 32, GRPO (TRL 1.16): 16 tasks × 8 replies per step, learning rate 2e-5,
60 steps, reward 1 for a correct final answer and 0 otherwise.

## Decision

Evaluate base and trained on reasoning dev (4 samples) and CLadder dev (2 samples).

- **Go** if CLadder dev accuracy rises by at least 2 points overall, or by at least 5 points on the query
  types the families train (`backadj`, `collider_bias`, `exp_away`, `ate`, `correlation`, `marginal`),
  with the story-bootstrap 95% interval above zero.
- **Rethink** otherwise. In-domain gains alone do not justify building out the environment.

Counterfactual query types (`det-counterfactual`, `ett`, `nde`, `nie`) are not trained and serve as a
within-CLadder comparison.

## Result (added after the run)

Run `reason60`: 60 steps, adapter at `reason60/adapter` on the Modal volume. Evaluation: reasoning dev
with 4 samples, CLadder dev with 2 samples, story-bootstrap 95% intervals.

| Set | Base | Trained | Trained − base |
| --- | ---: | ---: | --- |
| CLadder dev, all query types | 0.771 | 0.778 | +0.7 [−0.2, +1.4] |
| CLadder dev, the six trained query types | | | +1.2 [+0.4, +2.0] |
| CLadder dev, counterfactual types | | | −0.3 [−1.8, +1.3] |
| Reasoning dev (corrected grader, see below) | 0.681 | 0.747 | +6.6 [+4.0, +9.2] |

**Decision: rethink.** Neither CLadder criterion was met.

**Grader correction.** The grader's answer pattern let whitespace after "Answer:" cross line breaks. A
reply ending "Final answer:\n\nAnswer: Yes" was therefore read as "Answer: Yes" and scored 0. This
affected the training reward and the reasoning-dev evaluation, but not CLadder, which uses its own
extractor. With the fix (in the v2 commit), base reasoning-dev accuracy is 0.681, not 0.574; the
trained adapter's is 0.747, not 0.737. Mean training reward was 0.704, not 0.679. Most of the reported
+16.3-point in-domain gain was the policy learning to avoid that harmless formatting. The real gain
is +6.6.

The diagnosis of the CLadder null, and the redesign it led to, are in `docs/environment_v2.md`.
