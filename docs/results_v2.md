# Environment v2 training study: results

The registered design is `docs/preregistration_v2.md`. Replies were collected on 2026-10-29/30, their digests
committed in `ae1aaff` before any score was computed, and the analysis run with `python scripts/confirm_v2.py`.

## Confirmatory result: the gate did not open

No endpoint beat the base model, so under the registration the shuffled-reward control was not trained and
the study reports **no confirmed effect**. The manipulation check also failed: the trained adapter was not
reliably better than base even on held-out items from the training distribution.

Differences are trained − comparator in points, with the descriptive 95% cluster-bootstrap interval and the
one-sided sign-flip p-value. An endpoint's p-value is the larger of its comparators'; none is below Holm's
loosest threshold, 0.025.

| Set | Base | Trained | vs base | vs chance | Endpoint p |
| --- | ---: | ---: | --- | --- | ---: |
| `iid_test` (manipulation check) | 0.703 | 0.722 | +1.9 [−1.0, +4.8], p = 0.108 | | |
| `ood_wording_test` | 0.667 | 0.674 | +0.7 [−2.2, +3.6], p = 0.349 | | 0.349 |
| `ood_vocab_test` | 0.708 | 0.714 | +0.6 [−2.4, +3.7], p = 0.377 | | 0.377 |
| `ood_size_test` | 0.615 | 0.639 | +2.4 [−0.8, +5.6], p = 0.080 | | 0.080 |
| `dsep` (balanced) | 0.714 | 0.721 | +0.7 [−2.6, +4.0], p = 0.357 | +22.1 [+18.9, +25.1], p = 0.00005 | 0.357 |
| `c2c_test` (10 cells) | 0.548 | 0.569 | +2.1 [−1.0, +5.4], p = 0.097 | +6.9 [+4.6, +9.3], p = 0.001 | 0.097 |
| `cladder_adjust` (4 cells) | 0.615 | 0.607 | −0.8 [−3.0, +1.4], p = 0.761 | +10.7 [+5.9, +15.9], p = 0.082 | 0.761 |

Every point estimate but one is positive, but each is within the noise the study could resolve (its power
analysis put the smallest reliably detectable effect at about 4–7 points on the test splits). "Not
confirmed" is not evidence of no effect: the data are compatible with no change and with gains of a few
points.

## Exploratory (none supports a claim)

Raw accuracy, no-answer and truncation rates, reply length and yes-rate per arm:

| Set | Arm | Accuracy | No answer | Truncated | Mean tokens | Yes-rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `iid_test` | base | 0.703 | 0.003 | 0.003 | 2448 | 0.350 |
| | trained | 0.722 | 0.000 | 0.000 | 2106 | 0.330 |
| `ood_wording_test` | base | 0.667 | 0.001 | 0.001 | 2530 | 0.338 |
| | trained | 0.674 | 0.004 | 0.004 | 2176 | 0.340 |
| `ood_vocab_test` | base | 0.708 | 0.001 | 0.001 | 2356 | 0.327 |
| | trained | 0.714 | 0.000 | 0.000 | 2030 | 0.332 |
| `ood_size_test` | base | 0.615 | 0.015 | 0.015 | 3414 | 0.358 |
| | trained | 0.639 | 0.009 | 0.009 | 3082 | 0.352 |
| `dsep` | base | 0.714 | 0.001 | 0.001 | 1518 | 0.493 |
| | trained | 0.721 | 0.000 | 0.000 | 1317 | 0.474 |
| `c2c_test` | base | 0.793 | 0.000 | 0.000 | 2259 | 0.114 |
| | trained | 0.793 | 0.000 | 0.000 | 1980 | 0.128 |
| `cladder_adjust` | base | 0.478 | 0.000 | 0.000 | 473 | 0.601 |
| | trained | 0.467 | 0.000 | 0.000 | 358 | 0.615 |

Test-split accuracy by family, base → trained (250 items each):

| Split | Adjustment | Discovery | Effect | Independence |
| --- | --- | --- | --- | --- |
| `iid` | 0.53 → 0.54 | 0.64 → 0.65 | 0.93 → 0.94 | 0.72 → 0.75 |
| `ood_wording` | 0.48 → 0.51 | 0.60 → 0.64 | 0.89 → 0.84 | 0.70 → 0.71 |
| `ood_vocab` | 0.54 → 0.58 | 0.64 → 0.58 | 0.93 → 0.95 | 0.72 → 0.74 |
| `ood_size` | 0.54 → 0.50 | 0.52 → 0.57 | 0.81 → 0.79 | 0.60 → 0.69 |

Training rollouts of `reason-v2` (1,280 per 10-step block, a different 160 tasks in each block):

| Steps | Reward | No answer (mostly truncation at 4,096 tokens) | Accuracy among answered |
| --- | ---: | ---: | ---: |
| 1–10 | 0.617 | 0.152 | 0.728 |
| 11–20 | 0.627 | 0.127 | 0.717 |
| 21–30 | 0.603 | 0.120 | 0.686 |
| 31–40 | 0.688 | 0.090 | 0.755 |
| 41–50 | 0.662 | 0.116 | 0.748 |
| 51–60 | 0.733 | 0.084 | 0.800 |

Observations:
- **Truncation is no longer a factor at evaluation.** At 8,192 tokens both arms finish almost every reply,
  so the part of the training-reward gain that came from finishing within 4,096 tokens (no-answer rate
  0.15 → 0.08) cannot show up in evaluation. The trained model writes 10–24% shorter replies on every set.
- **No response-bias shift.** Yes-rates move by at most 2 points on every set.
- **Largest family movement:** independence on `ood_size` (+9 points on 250 items), alongside drops on other
  families; per-family differences of this size are within noise for 250 items.
- **Training accuracy among answered replies** rose from about 0.72 to 0.80 in the last block, but blocks
  hold different tasks, so this trend is noisy.

## Cost

About $27 of GPU time for the study (training about $16.5 including the resume; the six evaluation calls,
estimated, about $10), plus earlier calibration and pilots, for about $39 in total.
