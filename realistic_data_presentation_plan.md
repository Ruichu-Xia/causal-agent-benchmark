# CausalAgentBench — Realistic Data Presentation Plan (v1)

> **Status: implemented** (`task/instance.py::build_prompt`, the standardization sentence). Converged
> 2026-07-24 after comparing two approaches; the rejected one is kept on record (§2) since the reasoning
> is a standing constraint on any future surface-realism work, not just a note on a past decision.
> Orthogonal to which scaffold generator is used, and orthogonal to the
> driver-effect-construction plan in `procedural_scaffold_plan.md` §5a - this covers what units/disclosure
> the agent sees; that one covers how large and well-separated the underlying ground-truth effects are.

---

## 0. Motivation

Every instance today presents standardized, unit-variance continuous data regardless of what a column is
named - a column called `patient_age_years` and one called `enrollment_batch_id` look statistically
identical. The question was how to make this feel more like real analytical data.

## 1. Decision: disclose the standardization, don't fake realistic ranges

Give the agent standardized data exactly as today, but say so explicitly in the prompt - one sentence,
stated once, universally, for every instance. No per-node realistic ranges, no rescaling.

This is a legitimate, common real-world scenario on its own terms (an analyst frequently receives
pre-standardized data from an upstream pipeline), not merely a fallback. It also has a concrete secondary
benefit: telling the agent magnitudes are directly comparable across variables reinforces correct use of
`regress()`'s output for the actual ranking task, rather than leaving the agent to infer that on its own.

## 2. The rejected alternative, and why - kept on record

The first draft of this plan (realistic per-node ranges, e.g. `patient_age_years` → `(20, 80)`, decided once
per node's true role and held fixed across every naming-alignment skin) was rejected for a real, not
cosmetic, reason: at a misaligned or neutral alpha level, the displayed range would stay tied to the node's
*true* role while the *name* shown is deliberately something else - so the range can look inconsistent with
whatever name is actually displayed (e.g. a plausible age-like range under a name like `annual_rainfall_mm`).
That's a new signal, unrelated to causal reasoning, that could let an agent shortcut the misaligned
condition by noticing a range/name mismatch rather than reasoning about structure at all - a real risk to
the exact experiment this benchmark exists to run. The standardize-and-
disclose approach removes this risk structurally: with no per-node range, there is nothing for any name to
be inconsistent with, at any alignment level. It also cuts the implementation surface from a labeler schema
change + a rescaling step + a `regress()` re-standardization fix down to one sentence of prompt text - lower
risk and lower cost together, not a trade-off between them.

## 3. What changes, concretely

Exactly one thing: `task/instance.py::build_prompt()` gains one sentence in the existing template, stating
that released columns are standardized (mean 0, standard deviation 1) and that this makes magnitudes
directly comparable across variables regardless of what they measure.

**Everything else is confirmed unchanged**, and worth stating explicitly since it's the point of choosing
this approach:
* `generation/labeler.py` - no schema changes, no new prompt content, no new pre-check.
* `task/labeling.py` (`Labeling`, `LabelProvenance`) - no new fields.
* `task/instance.py::build_instance` - `display_data` stays exactly as it is today (column-permuted, never
  rescaled).
* `agent/tools.py` (`preview`, `correlate`, `regress`) - no changes; there is no rescaling to compensate for.
* `scoring/` - never touched `Instance.data` in the first place; unaffected either way.

## 4. Explicitly deferred

* Genuine per-node realistic value ranges (surface realism - values that literally look like ages, incomes,
  etc.) - not merely deferred for later convenience, rejected on a specific risk (§2). Revisit only if a
  concrete future need for surface realism arises, and only with the naming-coherence risk addressed head
  -on, not waved away.
* Genuine categorical/binary node types - unrelated to this plan's scope; tracked as `backlog.md` §11,
  unchanged by this decision either way.
