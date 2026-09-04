"""Versioned build configuration - the few knobs a user must set to build a data set.

Deliberately minimal: only what genuinely varies per build. The alignment level is NOT here - it is the
fixed grid ``{-1,+1}``, constructed per instance. Internal tuning (temperatures, grounding
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
    n_candidates: int | None = Field(default=7, ge=4)
    """Total candidate columns per instance - drivers *and* decoys - fixed for every instance in the build.

    A build knob rather than a code default because it is a property *of the produced set*: task size is
    not comparable across instances (ranking agreement at 4 candidates is a much coarser, noisier quantity
    than at 9), so a set that mixes sizes cannot be pooled without stratifying. Pinning it makes the
    committed config fully describe the set, which is the reason this file exists.

    ``7`` from measurement, not preference. Larger tasks are worse on both axes that matter. The oracle
    itself stops recovering the true ranking as driver count rises, because effect gaps shrink as
    ``1/sqrt(N)`` while sampling noise at ``n_samples`` does not: 59% oracle recovery at 5 drivers against
    23% at 8, which is what made large sets expensive to generate. And blind policies get *closer* to the
    truth as the task grows - best-blind estimation error 31% at 5 drivers, 24% at 8 - so a bigger task
    discriminates less. ``None`` restores the free per-seed draw, fine for a smoke probe but not for a set
    anything is measured on."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> BuildConfig:
        """Load and validate a build config from a YAML file."""
        import yaml

        data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
