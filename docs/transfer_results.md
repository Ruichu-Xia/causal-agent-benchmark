# Transfer evaluation: results

Results for the plan in [transfer_preregistration.md](transfer_preregistration.md), run as registered.
**No transfer endpoint meets the decision rule**: GRPO on CausalAgentBench lowers in-domain test error by
0.20, but this run does not improve the model on CLadder, Corr2Cause or published networks with their real
names, measured against both the base model and the shuffled-reward control.

## The control run

`control120` matches `grpo120` exactly at step 1 (same seed, same 64 rollouts), then diverges. With
rewards permuted within each step it degrades on the training task itself; training-rollout statistics,
per block of 20 steps:

| Steps | Trained: reward / valid | Control: reward / valid |
| --- | ---: | ---: |
| 1–20 | +0.15 / 85% | +0.04 / 78% |
| 41–60 | +0.25 / 96% | −0.16 / 68% |
| 81–100 | +0.31 / 98% | −0.78 / 18% |
| 101–120 | +0.33 / 98% | −0.70 / 31% |

So the control answers "does GRPO on this data help without an informative reward?" (no: it hurts), and
a trained − control difference on a task in the benchmark's format partly measures that damage.

## Pre-registered endpoints

Higher is better for accuracy, lower for error. Intervals are 95% for the primary, 97.5% for the
secondaries.

| Endpoint | Base | Trained | Control | Trained − base | Trained − control | Rule met |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| CLadder backadj + collider_bias, accuracy | 0.520 | 0.509 | 0.522 | −0.011 [−0.025, +0.003] | −0.014 [−0.028, +0.000] | no |
| bnRep, real names, error | 0.531 | 0.516 | 0.826 | −0.015 [−0.075, +0.045] | −0.310 [−0.401, −0.218] | no |
| Corr2Cause test, balanced accuracy | 0.466 | 0.500 | 0.480 | +0.034 [+0.009, +0.060] | +0.020 [−0.006, +0.046] | no |

- **CLadder.** The interval excludes any gain above 0.3 points over base. No reply lacked an answer.
- **bnRep.** The trained model beats the control, which answers validly on only 44% of tasks, but not the
  base model. On the 67 tasks all three answered validly, trained − base is −0.034 [−0.099, +0.031] (95%).
- **Corr2Cause.** All three models are at or below chance. The trained model's gain over base comes from
  shorter replies: median 1,710 tokens against 1,878 (control 2,112), so fewer hit the 4,096-token limit
  (7.7% against 10.8%; control 12.9%). Among answered replies, accuracy on true items is 0.125 (base),
  0.168 (trained) and 0.190 (control), and about 0.91 on false items for all three.

## Exploratory analyses

- **CLadder negative controls.** `correlation`: base 0.930, trained 0.928, control 0.930 (trained − base
  −0.002 [−0.007, +0.003]). `ate`: 0.924, 0.920, 0.917 (−0.004 [−0.013, +0.007]); flat in every property
  (anticommonsense, nonsense, easy, hard) to within 1.5 points.
- **Checkpoints on the primary set** (1 sample each): step 40 0.518 (−0.002 [−0.020, +0.017] against
  base), step 80 0.500 (−0.020 [−0.046, +0.006]), step 120 0.509. No trend with training.
- **Answer behaviour on the primary set.** Yes-rates on `backadj` are 0.60 (base), 0.61 (trained), 0.58
  (control); on `collider_bias` 0.04, 0.04 and 0.06. All three models nearly always answer no to collider
  questions.
- **bnRep with placeholder names.** Base 0.624, trained 0.533, control 0.775: trained − base −0.091
  [−0.143, −0.039], and −0.076 [−0.131, −0.021] on the 108 tasks all three answered validly. With
  placeholders the trained model is as accurate as with real names (0.533 against 0.516), while the base
  model is worse (0.624 against 0.531). Training used real names only, so this is the one out-of-domain
  gain the run shows, and it was not pre-registered as a confirmatory endpoint.
- **bnRep per network** (valid tasks, real names, trained − base): expenditure −0.131, lexical −0.073,
  algal2 −0.022, liquefaction +0.024; every interval includes zero.

## Reproduce

```bash
scripts/run_transfer_far.sh base trained control      # CLadder and Corr2Cause sets
scripts/run_transfer_bnrep.sh base trained control    # bnRep, real and placeholder names
python scripts/transfer_eval.py report primary --model trained --against base --against control
python scripts/transfer_eval.py report c2c_test --model trained --against base --against control --balanced --level 0.975
python scripts/transfer_bnrep_report.py trained base control
```

Models are served by `scripts/modal_serve.py` with
`ADAPTERS="trained=grpo120/adapter,trained40=grpo120/checkpoint-40,trained80=grpo120/checkpoint-80,control=control120/adapter"`.
The control was trained with
`modal run --detach scripts/modal_grpo.py::main --dataset train --run control120 --max-steps 120 --extra "--save-steps 10 --shuffle-rewards"`,
resumed once from `checkpoint-40` after the client lost its connection.
