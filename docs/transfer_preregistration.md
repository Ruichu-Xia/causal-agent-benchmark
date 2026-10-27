# Transfer evaluation: pre-registration

Does GRPO on CausalAgentBench improve a model beyond its own environment? This file fixes the models, item
sets, prompts, scoring and decision rules before any trained or control model sees a scored item. The pilot
used only items outside every scored set: 100 CLadder rung-3 items and 100 Corr2Cause dev items, base and
trained models, 4 samples each. It changed one thing: every reply without an answer had hit the 2,048-token
limit, so the limit was raised to 4,096.

## Models

All three are Qwen3-4B-Instruct-2507, served by `scripts/modal_serve.py` with the model's default sampling.

| Name | What it is |
| --- | --- |
| `base` | The untrained model |
| `trained` | LoRA adapter after 120 GRPO steps on the benchmark reward (`grpo120/adapter`) |
| `control` | The same run with each step's rewards permuted across its rollouts (`control120/adapter`, `train_grpo.py --shuffle-rewards`): same configuration, seed, tasks and steps, but the reward carries no information about which rollout earned it |

The control answers whether a gain comes from the reward signal or from running GRPO on Qwen at all.

## Primary endpoint: far transfer to CLadder adjustment questions

- **Items:** CLadder v1.5 (`causal-nlp/CLadder` at revision `18d684e`, `data/full_v1.5_default.csv`),
  `query_type` `backadj` or `collider_bias`: 1,738 items, 869 labeled yes, 47 stories. Item list hash
  (`scripts/transfer_eval.py`, set `primary`): `f7a3425214ff7f4f4d05d1e4dbaf9b5d6744f2a3b1177c6e083ec04f5ced7934`.
- **Prompt:** one user message, the dataset's `prompt` followed by a blank line and
  `Think it through, then end your reply with a final line that is exactly "Answer: Yes" or "Answer: No".`
  No system prompt, no tools.
- **Sampling:** 4 samples per item, seeds 0 to 3, at most 4,096 new tokens.
- **Answer rule:** the last match of `answer\W*(?:is\W*)?(yes|no)\b` (case-insensitive); otherwise a final
  non-empty line that is only yes or no; otherwise no answer, scored wrong.
- **Statistic:** per item, accuracy over its 4 samples; Δ = mean item accuracy of `trained` minus that of
  the comparator.
- **Interval:** 95% paired bootstrap over CLadder stories (10,000 resamples, seed 0).
- **Decision:** far transfer is claimed only if Δ(`trained` − `base`) and Δ(`trained` − `control`) both have
  95% intervals above zero. Requiring both keeps the error rate at 5% without splitting it.

## Confirmatory secondary endpoints

Each uses the same two comparisons and the same decision rule, with 97.5% intervals (Bonferroni over the two).

1. **Near transfer: published Gaussian networks as benchmark tasks.** 200 tasks, 50 from each of four bnRep
   networks with descriptive variable names (`expenditure`, `lexical`, `algal2`, `liquefaction`; exported to
   `data/bnrep` by `scripts/bnrep_export.R`, built by `scripts/transfer_bnrep.py --per-graph 50 --seed 0`).
   Dataset hashes: JSON `d73627724f30beab128e1a0130b1dc6af9f1757238c5cc6e48d45fb192bbed09`, NPZ
   `dd6ec23109e3488f47d12220786d50f54d2d705dc8ce330ef31e610a6ce38fbd`. Evaluated with
   `causalagentbench eval` under the in-domain protocol (lean-v2), real names. Statistic: paired difference in
   pooled capped error (`causalagentbench report --compare`), task bootstrap; lower is better.
2. **Far transfer: Corr2Cause.** The test set (1,162 items, 180 true; `causal-nlp/corr2cause` at revision
   `42ba12c`; set `c2c_test`, hash `947d5570cf62f5137c989bf8f47b4a92bae0ff2d65a6decb7242051a3614e49b`), 2
   samples per item, balanced accuracy (the mean of the accuracies on true and on false items), item bootstrap.
   Prompt: the premise, the hypothesis, then `Given the premise, is the hypothesis necessarily true?` and
   the same final-line instruction.

## Exploratory analyses

Reported, but not used for any claim of transfer:

- **CLadder `correlation` items (negative control).** 1,422 items, 2 samples each. They use the same stories
  and arithmetic but no adjustment, so a causal-specific gain should be smaller here than on the primary set.
- **CLadder `ate` items.** 1,422 items, 2 samples, reported by question property (commonsense, anticommonsense,
  nonsense).
- **bnRep with placeholder names**, and per-network results.
- **Answer behaviour.** Yes-rate and no-answer rate per model and query type.
- **Checkpoints.** Steps 40 and 80 of the trained run on the primary set (1 sample each).
- **Validity split for bnRep.** Validity rate, and error on tasks every model answered validly.

## Known limitations

- There is one training seed per arm.
- CLadder and Corr2Cause are public (2023) and may be in Qwen3's pretraining data; this affects every arm
  equally.
- The bnRep networks reuse the benchmark's prompt, which calls coefficients synthetic. Their coefficients are
  in fact the published fits, and their structures were learned from data or elicited from experts.
- The pilot found that Corr2Cause replies average about 1,650 tokens, so the Corr2Cause
  variable-refactorization set (2,246 items) was dropped from the exploratory analyses to save GPU time.
- QRData was planned as a second near-transfer set but is deferred: its questions do not fit the benchmark's
  answer format without a separate harness.
