# Multi-model baseline — testset_v1 (2026-07-22)

**Dataset:** `outputs/testset_v1` (12 SCMs × 6 conditions = 72 instances, `n_samples=1000`).
**Client:** `DatabricksClient`, `max_turns=20` + final-turn reminder (raised from 8 after finding several
models weren't converging in time - see commit history).
**Models:** claude-opus-4-8, claude-haiku-4-5, meta-llama-3-3-70b-instruct, meta-llama-3-1-8b-instruct,
gpt-oss-120b. (`llama-4-maverick` excluded - doesn't translate native tool-call syntax into the gateway's
`tool_calls` field, leaks raw chat-template text into content instead; not usable with this loop.)

## Overall

| model | parse_rate | mean τ | mean adj_acc |
|---|---|---|---|
| claude-opus-4-8 | 1.00 | +0.519 | 0.667 |
| claude-haiku-4-5 | 0.92 | +0.417 | 0.556 |
| meta-llama-3-3-70b-instruct | 1.00 | +0.241 | 0.417 |
| meta-llama-3-1-8b-instruct | 1.00 | +0.157 | 0.384 |
| gpt-oss-120b | 0.33 | +0.046 | 0.185 |

## Per-driver correctness

| model | A (ambiguous) | B (unambiguous) | Dz (ambiguous) |
|---|---|---|---|
| claude-opus-4-8 | 0.33 | 1.00 | 0.67 |
| claude-haiku-4-5 | 0.17 | 0.92 | 0.58 |
| meta-llama-3-3-70b | 0.07 | 1.00 | 0.18 |
| meta-llama-3-1-8b | 0.08 | 0.97 | 0.10 |
| gpt-oss-120b | 0.07 | 0.33 | 0.15 |

4/5 models replicate the single-model-probe finding (`B` near-invariant regardless of naming). gpt-oss-120b
breaks it (0.33 on `B`) - a qualitatively different, more basic failure than the others' shared struggle
with `A`, consistent with its low parse rate: general unreliability on this gateway, not narrowly a
naming-axis weakness.

## mean τ by alpha level

| model | blind | −1.0 | −0.5 | 0.0 | +0.5 | +1.0 |
|---|---|---|---|---|---|---|
| claude-opus-4-8 | +0.61 | −0.33 | +0.44 | +0.61 | +0.78 | +1.00 |
| claude-haiku-4-5 | +0.67 | −0.33 | +0.39 | +0.39 | +0.58 | +0.81 |
| meta-llama-3-3-70b | +0.39 | +0.17 | +0.17 | +0.17 | +0.11 | +0.44 |
| meta-llama-3-1-8b | +0.39 | +0.17 | +0.11 | +0.06 | +0.11 | +0.11 |
| gpt-oss-120b | −0.08 | +0.08 | −0.08 | −0.03 | +0.14 | +0.25 |

## Pooled α-vs-score correlation (named instances, n=60/model)

| model | r(α, τ) | r(α, adj_acc) |
|---|---|---|
| claude-opus-4-8 | +0.630 | +0.672 |
| claude-haiku-4-5 | +0.567 | +0.275 |
| gpt-oss-120b | +0.209 | +0.187 |
| meta-llama-3-3-70b | +0.143 | −0.051 |
| meta-llama-3-1-8b | −0.029 | −0.265 |

Opus tracks the naming axis far more strongly than any other model - the cleanest signal that it's doing
"verify what's identifiable, use naming to resolve the rest" more coherently. Both Llama models show
near-zero or negative correlation: their errors look like generic noise, not a naming-driven pattern.

## Methodology notes / things fixed mid-run

- `MAX_TURNS` raised 8→20 plus an explicit final-turn reminder message: several models (haiku, gpt-oss,
  partially opus) were still mid-investigation when force_final cut off tool access, then kept narrating
  next steps as plain text instead of answering - not a reasoning failure, a turn-budget one. First run at
  max_turns=8 showed inverted, unusable results (haiku/gpt-oss parse rates near zero, below the small
  Llama model) purely from this; discarded and rerun.
- `DatabricksClient` needed two live fixes before this run was trustworthy: model names require a
  `databricks-` prefix not shown in the bare model list; some models (claude-opus-4-8) reject the
  `temperature` param outright (retried without it); gpt-oss-120b sometimes returns `message.content` as
  a list of blocks (reasoning + text) instead of a string (extracted defensively, Databricks-only fix).
- Raw per-instance results (5 JSON files) not committed - scratchpad diagnostic data.

## Caveats

- n=72 (12 SCMs) - same scale caveats as the single-model probe apply.
- Single dataset build; naming content is fixed across all 5 models (fair comparison) but is itself one
  sample from the naming-generation pipeline, not averaged over multiple builds.
- gpt-oss-120b's low parse rate reflects this specific Databricks-served configuration, not necessarily
  the model in general.
