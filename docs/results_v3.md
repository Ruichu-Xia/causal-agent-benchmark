# Environment v3 study: results

The registered design is `docs/preregistration_v3.md` (commit `d38dde6`). Replies were collected on
2026-10-31, and each set was recorded in `docs/preregistration_v3_records.json` and committed before any
score was computed (base and trained `35fbe58`…`a59e428`, control `553d8b2`…`aef9d1b`). The control adapter's
digest was committed in `b0e3d70` before its evaluation. The analysis:

```bash
uv run python scripts/confirm.py analyze --control v3-control
uv run python scripts/explore.py --control v3-control
```

## Confirmatory result: four of five endpoints confirmed

The trained adapter beats both the base model and the shuffled-reward control on the three held-out test
splits and on `bnrep_real`. It is not confirmed on `qrdata_causal`. The manipulation check passes.

Scores are each arm's weighted accuracy (cell-balanced on `qrdata_causal`). Differences are trained −
comparator in points, with the descriptive 95% cluster-bootstrap interval and the one-sided sign-flip
p-value. An endpoint's p-value is the larger of its two comparators' values. 0.00005 is the smallest p-value
20,000 resamples can give.

| Set | Base | Trained | Control | Trained − base | Trained − control | Endpoint p | Holm |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| `iid_test` (manipulation check) | 0.426 | 0.632 | 0.376 | +20.6 [+15.4, +25.8], p = 0.00005 | | | passes |
| `ood_wording_test` | 0.428 | 0.612 | 0.380 | +18.4 [+13.6, +23.2], p = 0.00005 | +23.2 [+18.2, +28.2], p = 0.00005 | 0.00005 | **confirmed** |
| `ood_size_test` | 0.440 | 0.614 | 0.360 | +17.4 [+12.0, +22.8], p = 0.00005 | +25.4 [+20.4, +30.2], p = 0.00005 | 0.00005 | **confirmed** |
| `ood_vocab_test` | 0.462 | 0.600 | 0.364 | +13.8 [+8.8, +18.8], p = 0.00005 | +23.6 [+18.4, +28.8], p = 0.00005 | 0.00005 | **confirmed** |
| `qrdata_causal` | 0.323 | 0.367 | 0.382 | +4.4 [+0.5, +8.8], p = 0.016 | −1.4 [−6.0, +3.1], p = 0.729 | 0.729 | not confirmed |
| `bnrep_real` | 0.443 | 0.615 | 0.380 | +17.2 [+11.5, +22.9], p = 0.00005 | +23.4 [+16.4, +30.2], p = 0.00005 | 0.00005 | **confirmed** |
| `qrdata_statistics` (exploratory) | 0.567 | 0.565 | 0.563 | | | | |
| `bnrep_placeholder` (exploratory) | 0.401 | 0.609 | 0.341 | | | | |

Holm's thresholds for the four smallest p-values are 0.005, 0.00625, 0.0083 and 0.0125. QRData's 0.729 is
compared with 0.025. There is one training run per arm, so these claims are about these adapters.

**Scope conditions** (registered; they decide how each confirmed endpoint is described):

| Set | Items every sample of all three arms answered | Trained − base | Trained − control |
| --- | ---: | --- | --- |
| `iid_test` | 355 | +17.7 [+11.8, +23.7] | +17.2 [+11.3, +23.1] |
| `ood_wording_test` | 367 | +15.3 [+9.8, +20.7] | +15.8 [+10.1, +21.5] |
| `ood_size_test` | 358 | +16.5 [+9.8, +22.9] | +18.2 [+12.3, +24.0] |
| `ood_vocab_test` | 371 | +10.8 [+5.1, +16.4] | +19.1 [+13.2, +25.1] |
| `bnrep_real` | 107 | +9.3 [+2.3, +16.4] | +9.3 [+0.5, +18.7] |

- **Held-out splits:** every interval lies above zero, so the gains are described as **better analysis, not
  only finishing within the budget**. The gains hold for held-out wording, for graphs with 7–8 variables
  (training used 4–6), and for held-out concept names.
- **bnRep:** the gain is described as **a gain on published networks that the single-source shortcuts could
  explain**. The shortcut-wrong cells are too small to test (trained / base / control):
  - independence where the marginal rule is wrong: 1.00 / 0.75 / 0.53 (16 items);
  - effects where both regressions are wrong: 0.00 / 0.10 / 0.20 (5 items);
  - adjustment where the confounder rule is wrong: 0.50 / 0.45 / 0.35 (10 items).
- **QRData:** not confirmed, so no scope condition applies. The gain over base (+4.4) also holds under strict
  choice parsing (+4.4 [+0.5, +8.8]; no reply was right only under the prefix rule). But shuffled-reward
  training reaches the same score (0.382 against 0.367), so the gain over base cannot be attributed to the
  reward signal. "Not confirmed" is not evidence of no effect. With an effective 15.5 clusters the interval
  against control is 9 points wide.

### What the control shows

GRPO with shuffled rewards made the model worse than base on every v3 and bnRep set. Replies grew longer and
truncation rose:

| Set | Truncated (base / trained / control) | Mean reply tokens | Tool calls |
| --- | --- | --- | --- |
| `iid_test` | 0.046 / 0.006 / 0.180 | 1,985 / 929 / 3,499 | 2.6 / 2.4 / 3.7 |
| `ood_size_test` | 0.052 / 0.002 / 0.182 | 2,254 / 892 / 3,509 | 3.0 / 2.4 / 3.7 |
| `bnrep_real` | 0.057 / 0.003 / 0.167 | 2,232 / 928 / 3,414 | 3.0 / 2.3 / 4.0 |
| `qrdata_causal` | 0.007 / 0.000 / 0.030 | 1,869 / 905 / 2,303 | 3.2 / 2.5 / 3.3 |

On the four confirmed endpoints, base is therefore the closer comparator, and it alone sets the endpoint
p-value. The control still does its registered job. Running the same GRPO on the same prompts and compute,
with uninformative rewards, does not produce the gains, so they come from the reward signal. The mean
training reward per 10-step block was:
- pilot: 0.447, 0.436, 0.439, 0.577, 0.600, 0.620;
- control: 0.452, 0.423, 0.388, 0.458, 0.506, 0.431.

Shuffling keeps each step's mean reward, so the control's figures are its true rewards.

## Exploratory (none supports a claim on its own)

**By family, base → trained** (control in parentheses):

| Family | `iid_test` | `ood_wording_test` | `ood_size_test` | `ood_vocab_test` | `bnrep_real` |
| --- | --- | --- | --- | --- | --- |
| Independence (yes/no) | 0.59 → 0.97 (0.61) | 0.70 → 0.95 (0.65) | 0.57 → 0.96 (0.67) | 0.66 → 0.94 (0.65) | 0.68 → 0.98 (0.57) |
| Structure (letter) | 0.45 → 0.70 (0.39) | 0.42 → 0.56 (0.34) | 0.49 → 0.70 (0.33) | 0.44 → 0.59 (0.28) | 0.34 → 0.53 (0.30) |
| Effect (number) | 0.29 → 0.39 (0.18) | 0.16 → 0.44 (0.16) | 0.29 → 0.34 (0.17) | 0.28 → 0.39 (0.15) | 0.41 → 0.55 (0.29) |
| Adjustment (yes/no) | 0.51 → 0.61 (0.46) | 0.47 → 0.65 (0.50) | 0.51 → 0.57 (0.35) | 0.59 → 0.55 (0.50) | 0.55 → 0.50 (0.47) |
| Adjustment (choice) | 0.18 → 0.25 (0.12) | 0.37 → 0.27 (0.20) | 0.27 → 0.27 (0.16) | 0.29 → 0.35 (0.20) | 0.21 → 0.33 (0.28) |

Test splits hold 125 items per family (yes/no adjustment 74, choice 51); `bnrep_real` holds 48 (19 and 29).

- **Independence is the clearest learned skill.** The trained model scores 0.91–0.97 where the marginal
  correlation misleads (base 0.46–0.73), and 0.92–0.97 whether or not the names suggest the right answer.
  It is near ceiling on bnRep with real or placeholder names.
- **Structure gains are not only the column-order shortcut.** Where "the earlier column causes the later"
  is wrong (25–26 items per split), trained against base scores 0.60 against 0.40 on iid, 0.40 against 0.40
  on wording, 0.54 against 0.35 on size and 0.56 against 0.28 on vocab. On bnRep, whose node order is random,
  structure still rises from 0.34 to 0.53. The gains appear in both the name-critical and data-critical
  cells.
- **Effect gains come from shortcuts.** The trained model's answers are near zero on the 35 test items per
  split where both the unadjusted and the adjust-for-everything regressions are wrong:
  - accuracy 0.00–0.06, against base 0.03–0.09;
  - its answers land on those regressions' values: adjust-for-everything on 14–20 of 35 items, unadjusted on
    4–10.

  It has not learned to choose a valid adjustment set, the weakness phase C found on dev.
- **Adjustment** moves little and in both directions.
- **Names do not help the trained model on bnRep.** Real names score higher than placeholders by +0.005 for
  trained, +0.042 for base and +0.039 for control. Its gain appears on all four networks, with real and with
  placeholder names.
- **QRData causal by type** (base / trained / control):
  - full-graph discovery 0.535 / 0.542 / 0.520 (150);
  - pairwise discovery 0.58 / 0.55 / 0.65 (50);
  - average treatment effect on the treated 0.11 / 0.35 / 0.11 (20);
  - average treatment effect 0.33 / 0.40 / 0.35 (10).

  Official accuracy is 0.454 / 0.479 / 0.464. "No causal relationship" answers are 473 / 465 / 416 of 600
  full-graph replies. The gain over base is in numerical treatment-effect items. Discovery, most of the score,
  does not move.
- **QRData statistics** is flat (0.567 / 0.565 / 0.563): no gain or loss on general data analysis.
- **Reply length:** the trained model writes 885–973 tokens on every set, against base 1,230–2,551, and makes
  fewer tool calls. Its scores on the held-out splits are 2–3 points below its `iid_test` score. Base and
  control show no consistent gap.

## Deviations and disclosures

- **Column order in training data** (registered as a known limitation): 81–85% of edges point from an
  earlier to a later column. Structure results are split by it above.
- **The control's pre-collection checks were incomplete.** The registration commit was redeployed for the
  control's evaluation. Before collection, only `files_check_main` on `qrdata_statistics` was run, as a
  digest check. It printed the registered digest `ad5579e6…` and failed 2 of 51 file sets, both
  `FLVoters.csv + names.csv` sets: "the Python session failed to start". `isolation_main` and the
  `qrdata_causal` file check were not rerun. Collection went ahead without code changes, so all arms share
  one evaluation code digest; `confirm.py` enforces that. After collection, against the same deployment,
  `isolation_main` passed in both images and all 139 `qrdata_causal` file sets loaded.
- **Sandbox startup failures on large QRData files.** "Failed to start" tool results, by arm (base / trained /
  control):
  - `qrdata_causal`: 32 / 25 / 11 of 1,076 episodes;
  - `qrdata_statistics`: 33 / 27 / 32 of 568 episodes;
  - v3 and bnRep sets: none.

  Every one is on a file set containing one of the three largest files: `social.csv`, `ak91.csv` and
  `names.csv`, 8–9 MB each. The model sees an error and may retry.
  - **Diagnosis** (a separate Modal app on the registered image): on their own, these sessions start in
    2.5 s with or without the 2 GB address-space limit and print nothing to stderr. With 96 sessions
    started at once in a container the size of the evaluation's (16 CPUs, 64 GB, no vLLM), all became ready
    within 31 s, peaking at about 450 MB each against 375 MB for small files.
  - **Cause:** not established. Likely candidates are memory pressure or CPU load in the real evaluation
    container (vLLM plus 96 sessions); the worker's stderr is discarded, which hid it.
  - **Effect on the result:** the failures touch 4 `qrdata_causal` items, 1.5% of its weight. Excluding
    those file sets (post hoc) changes nothing: trained − base +4.7 [+0.8, +9.0], trained − control −1.4
    [−6.0, +3.2].
- **Budget overrun:** see the cost section below.

The known limitations in the pre-registration still apply:
- one run per arm;
- the trained arm was chosen on dev;
- QRData may be in Qwen3's training data;
- vLLM is not bit-reproducible.

## Cost

Modal billing for the `causalagentbench-dataenv` app is about $108 for all of v3 (phases B to D′), against
the $55–86 planned:

| Phase | Planned | Actual |
| --- | ---: | ---: |
| B, calibration | | $7 |
| C, pilot training (4.3 h) and dev evaluation | | $26 |
| D, evaluation of base and trained (16 calls in parallel) | $8–12 | $22 |
| D′, control training (about 6 h on an H100, slower than the pilot) and evaluation | $22–28 | $52 |

The control's longer episodes, about 3,500 tokens each at evaluation, plausibly explain the slower
training. The diagnosis runs cost under $1.
