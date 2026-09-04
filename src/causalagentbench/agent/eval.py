"""Batch the agent loop over many instances and aggregate scores by naming alignment.

The headline metric this benchmark measures is score *as a function of* naming alignment, on the drivers
where the data provably cannot settle the answer. ``run_evaluation`` produces one ``AnswerScore`` per
instance (each already carries its own ``alignment``).

``driver_cells`` crosses naming alignment with whether a driver's effect is identifiable from data at all.
That table is the experiment: naming can only matter where the data leaves the orientation open, so the
identifiable row is a built-in placebo. ``naming_table`` reduces it to the contrasts that carry the result,
each paired within SCM with an interval from ``paired_bootstrap_ci`` at the unit that is actually
independent (see ``scm_group_key``).

The contrast that settles the benchmark's claim is **aligned minus misaligned** (``NamingResult.correctness``)
- the only one that holds "these columns carry domain names" fixed and varies solely whether the names are
true. Report every metric separately: ``overall`` averages identification and estimation, and measured live
those moved in opposite directions under the same manipulation, so their mean hid the effect.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from causalagentbench.agent.loop import (
    MAX_TOKEN_BUDGET,
    MAX_TOOL_CALLS,
    MAX_TURNS,
    ToolCallingClient,
    TurnTrace,
    TurnTranscript,
    run_agent,
)
from causalagentbench.scoring.score import AnswerScore, score_output
from causalagentbench.task.instance import Instance, InstanceKey

logger = logging.getLogger(__name__)


def run_evaluation(
    instances: Iterable[tuple[Instance, InstanceKey]],
    client: ToolCallingClient,
    *,
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_turns: int = MAX_TURNS,
    max_token_budget: int | None = MAX_TOKEN_BUDGET,
    on_instance: Callable[[int, AnswerScore], None] | None = None,
    on_turn: Callable[[int, TurnTrace], None] | None = None,
    on_transcript: Callable[[int, TurnTranscript], None] | None = None,
    on_instance_error: Callable[[int, Exception], None] | None = None,
) -> list[AnswerScore]:
    """Run the agent on every instance and score each reply; one ``AnswerScore`` per instance.

    A single instance whose agent run raises (a client/provider error the tool loop itself can't turn into
    text) is logged and scored as a parse failure rather than aborting the whole batch - one bad instance
    should never cost every other instance's already-computed score.

    ``on_instance``, if given, is called with (0-based index, that instance's score) right after each one
    finishes - optional because a batch over many instances against a slow model (an extended-thinking
    model needing a dozen-plus turns each) can otherwise run for a long time with zero visibility until the
    entire batch returns; the default (``None``) keeps this function silent, as before.

    ``on_turn``, if given, is called with (0-based instance index, ``TurnTrace``) for every turn of every
    instance - the same instance-index tagging as ``on_instance``, one level more granular. Passed straight
    through to each ``run_agent`` call.

    ``on_transcript`` similarly receives an instance index and full ``TurnTranscript`` for each model call.
    It is opt-in because transcripts include prompts, tool results, and unredacted model output.

    ``on_instance_error`` distinguishes a provider/client failure from an unparseable model answer. The
    returned list retains a zero score for compatibility with existing callers; evaluation runners should use
    this callback to record and exclude transport failures from benchmark aggregates.
    """
    scores = []
    for i, (instance, key) in enumerate(instances):
        instance_on_turn = (lambda trace, i=i: on_turn(i, trace)) if on_turn is not None else None
        instance_on_transcript = (
            (lambda transcript, i=i: on_transcript(i, transcript)) if on_transcript is not None else None
        )
        try:
            reply = run_agent(
                instance,
                client,
                max_tool_calls=max_tool_calls,
                max_turns=max_turns,
                max_token_budget=max_token_budget,
                on_turn=instance_on_turn,
                on_transcript=instance_on_transcript,
            )
        except Exception as exc:
            logger.exception("agent run failed; scoring this instance as a parse failure")
            if on_instance_error is not None:
                on_instance_error(i, exc)
            reply = ""
        score = score_output(reply, key)
        scores.append(score)
        if on_instance is not None:
            on_instance(i, score)
    return scores


@dataclass(frozen=True)
class DriverCell:
    """One (identifiability x naming) cell, pooled over driver observations.

    ``n`` counts *driver observations*, not instances - one instance contributes as many rows here as it has
    candidates. Decoys are included: every candidate is graded, so excluding them would make the cell
    incomparable with the instance-level numbers.

    This table is the experiment. Naming is expected to move ``estimation`` in the ``identifiable=False``
    row and not in the ``identifiable=True`` row, which doubles as a placebo - a nuisance effect (prompt
    length, domain difficulty, anything that travels with the skin) would move both rows, while a real
    semantic effect can only move the row where the data leaves the orientation open.
    """

    identifiable: bool
    alignment: float | None
    n: int  # driver observations in this cell, NOT instances
    identification: float  # 1 - mean induced adjustment bias
    estimation: float  # 1 - mean normalized effect error; the quantity Gain/Harm are computed on


def driver_cells(scores: list[AnswerScore]) -> list[DriverCell]:
    """Pool every driver observation into (identifiability, alignment) cells; name-blind last."""
    groups: dict[tuple[bool, float | None], list[tuple[float, float]]] = {}
    for score in scores:
        for driver, identifiable in score.identifiable.items():
            groups.setdefault((identifiable, score.alignment), []).append(
                (score.adjustment_bias[driver], score.effect_error[driver])
            )
    cells = [
        DriverCell(
            identifiable=identifiable,
            alignment=alignment,
            n=len(rows),
            identification=1.0 - sum(bias for bias, _ in rows) / len(rows),
            estimation=1.0 - sum(error for _, error in rows) / len(rows),
        )
        for (identifiable, alignment), rows in groups.items()
    ]
    return sorted(cells, key=lambda c: (not c.identifiable, c.alignment is None, c.alignment))


class IdentifiabilityF1(NamedTuple):
    """Pooled precision/recall/F1 of the "is this settled by data alone?" judgement.

    F1 rather than accuracy, following AbstentionBench: accuracy rewards a blanket answer, and ours had a
    floor of 0.605 from class balance alone. Positive class is *unidentifiable* - the judgement that costs
    something to get right.

    Pooled from raw counts rather than averaged over per-instance F1s, which does not aggregate correctly.
    A ``None`` judgement (the agent did not answer the field) counts as a miss, so omitting it is never
    cheaper than getting it wrong.
    """

    precision: float
    recall: float
    f1: float
    n_positive: int  # drivers that really are unidentifiable - the recall denominator


def identifiability_f1(scores: list[AnswerScore]) -> IdentifiabilityF1:
    """Pooled identifiability F1 over every driver of every score.

    Report it beside the mechanical PC baseline (F1 0.650 on the built set). Gold identifiability *is* the
    population CPDAG, so a perfect discovery tool reads it straight off and this number is not evidence of
    reasoning on its own.
    """
    tp = fp = fn = 0
    for score in scores:
        for driver, identifiable in score.identifiable.items():
            said = score.identifiability_judgment[driver]
            said_unidentifiable = said is False  # None (no answer) is never a positive
            truly_unidentifiable = not identifiable
            tp += said_unidentifiable and truly_unidentifiable
            fp += said_unidentifiable and not truly_unidentifiable
            fn += (not said_unidentifiable) and truly_unidentifiable
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return IdentifiabilityF1(precision, recall, f1, tp + fn)


def driver_metric(score: AnswerScore, metric: str, *, identifiable: bool | None = None) -> float | None:
    """One answer's score on ``metric``, optionally restricted to one identifiability class.

    ``identifiable=None`` is the whole instance and supports every metric. Restricting to a driver class
    supports only the two per-driver metrics: ``ranking_tau`` is a property of the ordering over *all*
    candidates and ``overall`` is built from both axes, so neither has a meaning on a subset - they return
    ``None`` rather than a number computed on a different population than its name implies.

    ``None`` is also returned when the instance has no driver of the requested class, so a caller can drop
    that SCM from a pairing rather than average in a value that was never measured.
    """
    if identifiable is None:
        return float(getattr(score, metric)) if hasattr(score, metric) else None
    if metric not in ("identification", "estimation"):
        return None
    source = score.adjustment_bias if metric == "identification" else score.effect_error
    values = [v for driver, v in source.items() if score.identifiable[driver] is identifiable]
    return 1.0 - sum(values) / len(values) if values else None


class Contrast(NamedTuple):
    """One paired difference with a bootstrap interval over SCMs."""

    delta: float
    ci: tuple[float, float]
    n_scms: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval clears zero. NaN (nothing paired) is never significant."""
        low, high = self.ci
        return low == low and not (low <= 0.0 <= high)


class NamingResult(NamedTuple):
    """All three naming contrasts for one metric on one driver population.

    ``correctness`` = aligned - misaligned is **the decisive one**, and it is reported first because it is
    the only contrast that holds "the columns carry domain names" fixed and varies solely whether those
    names are *true*. ``gain`` = aligned - blind conflates two different claims - that names help, and that
    *correct* names help - and measured live it moved (+0.164) while ``correctness`` on the same drivers did
    not (+0.102, interval spanning zero). A result read off ``gain`` alone would have claimed the benchmark's
    central finding on evidence that does not support it.

    **Pre-registered reading, fixed before the first full run.** Computed on the unidentifiable drivers this
    is the result; on the identifiable ones it is a negative control, because identifiability is a property
    of the graph and is therefore identical across an SCM's three skins. The rival explanation the control
    exists to rule out is that misaligned skins are simply *harder* - stranger names, more confusion, worse
    scores on everything - which predicts equal damage in both halves, while a genuine structural effect
    predicts it concentrated where the data left the orientation open. So:

    * the contrast on unidentifiable drivers **substantially exceeds** the one on identifiable drivers ->
      the effect tracks identifiability; the naming claim holds.
    * the two are **comparable** -> whatever the names are doing is not specific to the edges data cannot
      settle. That is general semantic contamination *or* a nuisance confound, and this design cannot
      separate them; report it as such rather than as the naming result.
    * the identifiable contrast **exceeds** the unidentifiable one -> suspect something travelling with the
      skin, and make no naming claim.

    Read as a pair, ``gain`` and ``harm`` separate three behaviours a single contrast cannot: a model that
    reasons about whether names can be trusted shows gain > 0 with harm ~ 0; one that trusts names blindly
    shows both positive; one that ignores names entirely shows both ~ 0.

    The control is "should be smaller", **not** "should be zero". Zero is the right expectation only for a
    name-blind policy - ``scoring/scripted_agent.py`` measures exactly 0.0000 - because a real model uses
    names as a general heuristic and some leakage into the identifiable half is expected and normal.

    Restricting to a driver subset costs power: it drops from ~5 drivers per instance to ~2.4, and measured
    live a contrast significant over all drivers (estimation +0.081, CI [+0.007, +0.154]) was not
    significant on either subset alone. Absence of a subset result is therefore weak evidence about
    concentration, not evidence against it.
    """

    metric: str
    subset: str  # "all" | "identifiable" | "unidentifiable"
    correctness: Contrast  # aligned - misaligned; the decisive contrast
    gain: Contrast  # aligned - name-blind
    harm: Contrast  # name-blind - misaligned


def naming_result(
    paired: Sequence[tuple[InstanceKey, AnswerScore]],
    metric: str,
    *,
    identifiable: bool | None = None,
    seed: int = 0,
) -> NamingResult:
    """The three naming contrasts for ``metric``, paired within SCM.

    Paired because the three skins share every structural parameter, so differencing them removes the
    instance-level variance that otherwise swamps the effect. The interval resamples SCMs, never skins or
    drivers - treating those as independent understated the standard error enough to turn noise into a
    reported effect once already.
    """
    by_scm: dict[str, dict[float | None, float]] = {}
    for key, score in paired:
        value = driver_metric(score, metric, identifiable=identifiable)
        if value is not None:
            by_scm.setdefault(scm_group_key(key), {})[score.alignment] = value

    def contrast(minuend: float | None, subtrahend: float | None) -> Contrast:
        diffs = [c[minuend] - c[subtrahend] for c in by_scm.values() if minuend in c and subtrahend in c]
        if not diffs:
            return Contrast(float("nan"), (float("nan"), float("nan")), 0)
        return Contrast(sum(diffs) / len(diffs), paired_bootstrap_ci([[d] for d in diffs], seed=seed), len(diffs))

    subset = "all" if identifiable is None else ("identifiable" if identifiable else "unidentifiable")
    return NamingResult(
        metric=metric,
        subset=subset,
        correctness=contrast(1.0, -1.0),
        gain=contrast(1.0, None),
        harm=contrast(None, -1.0),
    )


#: Reported for every model. ``overall`` is the leaderboard number but the **worst** place to read a naming
#: effect from: it averages identification and estimation, which respond differently to the manipulation -
#: measured live, estimation moved +0.081 (CI clear of zero) while identification moved -0.023 (null), and
#: their mean came out at +0.029 (null). Always read the naming result per metric.
CONTRAST_METRICS: tuple[str, ...] = ("overall", "identification", "estimation", "ranking_tau")


def naming_table(paired: Sequence[tuple[InstanceKey, AnswerScore]], *, seed: int = 0) -> list[NamingResult]:
    """Every contrast worth reporting: all metrics over all drivers, then the two per-driver subsets."""
    rows = [naming_result(paired, m, seed=seed) for m in CONTRAST_METRICS]
    for identifiable in (False, True):
        rows += [
            naming_result(paired, m, identifiable=identifiable, seed=seed) for m in ("identification", "estimation")
        ]
    return rows


def scm_group_key(key: InstanceKey) -> str:
    """Stable id for the *SCM* behind an instance, shared by all of its naming skins.

    The resampling unit. An instance is not an independent observation: the grid builds three skins of one
    SCM, and those skins share every structural parameter, so they succeed and fail together. Treating each
    skin - or worse, each driver - as independent understates the standard error by roughly the square root
    of the multiplicity. It happened here: a per-driver mean showed a naming gap that a paired SCM-level
    bootstrap put at [-0.222, +0.025], straddling zero.

    Derived from the structural parameters rather than a stored id because that is what "same SCM" means,
    and it needs no change to the on-disk format: skins differ only in ``node_to_label``.
    """
    scaffold = key.scm.scaffold
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repr(sorted(scaffold.edges)).encode())
    for array in (key.scm.W, key.scm.noise_var):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()


def paired_bootstrap_ci(
    groups: Sequence[Sequence[float]], *, n_resamples: int = 20000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile CI for the grand mean, resampling whole ``groups`` (SCMs) rather than observations.

    Each element of ``groups`` is one SCM's values. Resampling groups keeps the within-SCM correlation
    intact, which is exactly what the naive interval throws away.
    """
    means = np.array([np.mean(values) for values in groups if len(values) > 0], dtype=np.float64)
    if means.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, means.size, size=(n_resamples, means.size))
    resampled = means[draws].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(resampled, [tail, 1.0 - tail])
    return (float(low), float(high))
