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
