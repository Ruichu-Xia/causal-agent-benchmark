# Agent alpha-grid probe — gpt-5 (2026-07-21)

**Model:** `openai/gpt-5`, temp 0.2, standard `preview`/`correlate`/`regress` loop.
**Scope:** 5 SCM seeds (0-4) × 6 conditions (`{-1,-0.5,0,+0.5,+1}` + blind) = 30 runs, `n_samples=1000`.
Diagnostic probe, not the full set (`configs/testset_v1.yaml` → `n_scms=20`).

## Per-driver adjustment correctness (n=30)

| driver | correct | note |
|---|---|---|
| B | 30/30 (100%) | not orientation-ambiguous |
| Dz | 27/30 (90%) | ambiguous, but empirically diagnosable |
| A | 8/30 (27%) | genuinely unidentifiable from data alone (`backlog.md` §4) |

A by level: `blind 2/5, -1.0 0/5, -0.5 0/5, 0.0 1/5, +0.5 0/5, +1.0 5/5` — solved only at full alignment.

## Alpha vs. score, pooled per-instance (n=25 named, blind excluded)

| metric | Pearson r | Spearman r | perm. p |
|---|---|---|---|
| ranking_tau | +0.545 | +0.507 | 0.008 |
| adjustment_accuracy | +0.730 | +0.725 | 0.0001 |

- `adjustment_accuracy` is a near step function: 0.667 in 19/20 at α≠1.0, exactly 1.000 in 5/5 at α=+1.0.
- `ranking_tau`'s weaker r has an identified 2nd cause: effect-*magnitude* estimation noise (n=1000
  samples), independent of naming. 2 counterexamples at α=+1.0 (seeds 1, 3): adj_acc=1.00, τ=+0.333 —
  correct adjustment sets, swapped ranking order.

## Tool usage

Mean tool calls stay 17-39 across all levels; **blind is highest (38.6)**. Every run ≥7 calls. No
collapse toward name-only answers at high alignment.

## Mechanism (seed 0, α=-1.0 vs α=+1.0, same SCM/data)

- α=-1.0: 28 tool calls; mediator `M` has a plausible-confounder misaligned name → agent wrongly adjusts
  `A` for `{M,C}` (bad-control trap) → τ=-1.0.
- α=+1.0: 37 tool calls; explicitly compares `regress(A,[C])` vs `regress(A,[C,M])` → correctly picks
  `{C}` → τ=+1.0.

## Caveats

- Single model/config, n=5 seeds only.
- No naive-recitation baseline yet — "not pure recitation" rests on tool-call volume only.
- B's 100% may partly reflect that misaligned-trap design targets A/M/C/Dz, not B — less surprising than
  it looks.
- B invariant on adjustment validity only, not ranking position (swapped in both τ=+0.333 cases).
- Untouched: continuous-only data, fixed-topology memorization risk, soft description constraints
  (`backlog.md` §2-4).

## Reproducing

Ad hoc, uncommitted: `scripts/alpha_grid_probe.py` + `_continue.py` (needs `generation` extra + CauseNet +
LLM key). Raw 30-record JSON w/ full transcripts (~240KB) not committed.

## Naive-recitation baseline (added same day)

Same 30 conditions (fresh naming, same 5 seeds), `run_agent(instance, client, max_turns=1)` — forces
`tool_choice="none"` on the only turn, so the model answers from column names/descriptions/context alone,
never touching the data.

| | tool-using | naive-recitation |
|---|---|---|
| parse_rate | 1.00 | 0.33 |
| mean ranking_tau | +0.333 | +0.033 |
| mean adjustment_accuracy | 0.722 | 0.133 |
| B correct | 30/30 (100%) | 6/30 (20%) |
| Dz correct | 27/30 (90%) | 3/30 (10%) |
| A correct | 8/30 (27%) | 3/30 (10%) |

Blind baseline: parse_rate = 0/5 (zero names + zero data → the model can't even produce a well-formed
answer). Baseline never exceeds adj_acc=0.267 at any single alpha level, vs. 1.000 for tool-using at
+1.0.

This directly answers the earlier caveat "B's 100% may be a structural artifact, not proof of
robustness" — no, it isn't: the baseline, given the identical prompts, gets B wrong 80% of the time.
Tool use is doing real, substantial work; naming alone does not solve this task.

## Next

Weaker (non-frontier) model on the tool-using loop · scale to n_scms=20.
