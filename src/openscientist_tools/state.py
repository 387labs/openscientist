"""Per-job state for the standalone tools MCP server.

State is bound from ``OPENSCIENTIST_*`` environment variables at
module import. Missing required vars crash the server at start-up
via a Pydantic ``ValidationError`` that names the missing field.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ToolServerState(BaseSettings):
    """Env-var-bound state mirroring the in-process ``ToolContext``."""

    model_config = SettingsConfigDict(
        env_prefix="OPENSCIENTIST_",
        case_sensitive=False,
        extra="ignore",
    )

    job_id: str
    job_dir: Path
    data_file: Path | None = None
    data_files: tuple[Path, ...] = ()
    use_hypotheses: bool = False

    @field_validator("data_files", mode="before")
    @classmethod
    def _split_pathsep(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(Path(p) for p in value.split(os.pathsep) if p)
        return value


STATE = ToolServerState()
