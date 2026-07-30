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

    @classmethod
    def from_yaml(cls, path: str | Path) -> BuildConfig:
        """Load and validate a build config from a YAML file."""
        import yaml

        data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
