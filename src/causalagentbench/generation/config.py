"""Versioned build configuration - the few knobs a user must set to build a data set.

Deliberately minimal: only what genuinely varies per build. The alignment level is NOT here - it is the
fixed grid ``{-1,0,+1}``, constructed per instance. Internal tuning (temperatures, grounding
thresholds, sample size) stays as fixed code defaults. Secrets and the CauseNet dump path live in
``.env``. Committing the config makes a shipped set reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BuildConfig(BaseModel):
    """Immutable, validated configuration for a test-set / train-data build."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gen_model: str  # LiteLLM model id for the generator
    critic_model: str | None = None  # correctness critic (None skips it); same or a different model
    max_revisions: int = Field(default=2, ge=0)  # propose->critique->revise iterations before skipping
    n_scms: int = Field(default=20, gt=0)  # base SCM seeds; each yields all grid levels (+ name-blind)
    n_drivers: int | None = Field(default=8, ge=2)
    """Non-decoy driver count, fixed for every instance in the build.

    A build knob rather than a code default because it is a property *of the produced set*: task size is
    not comparable across instances (ranking agreement at 3 candidates is a much coarser, noisier quantity
    than at 9), so a set that mixes sizes cannot be pooled without stratifying. Pinning it makes the
    committed config fully describe the set, which is the reason this file exists.

    ``8`` is the top of ``DRIVER_COUNT_RANGE`` and the largest count the gates still accept affordably
    (16.4%, vs 8.6% at 9 and 4.8% at 10 - see ``variance_budget_plan.md`` §6 for why that ceiling exists
    and what raising it is coupled to). ``None`` restores the free per-seed draw, which is fine for a smoke
    probe but should not be used for a set anything is measured on."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> BuildConfig:
        """Load and validate a build config from a YAML file."""
        import yaml

        data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
