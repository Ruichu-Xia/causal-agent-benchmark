"""Generator configuration.

Stage 0 has one fixed difficulty, so the only user-facing knob is the data-table size. The gate
thresholds that define a valid instance live as constants in ``gates`` and will migrate here when
tunable difficulty is introduced.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GeneratorConfig(BaseModel):
    """Immutable generator configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_samples: int = Field(
        default=2000,
        gt=0,
        description="Rows per instance; large enough that finite-sample noise is negligible.",
    )
