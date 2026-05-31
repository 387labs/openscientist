"""Direct OpenAI API provider (drives the Codex agent).

Routes the Codex agent at OpenAI's default endpoint. Authentication is
either an ``OPENAI_API_KEY`` (API auth) or the ChatGPT OAuth login that
the codex CLI stores in ``~/.codex/auth.json``. ``CodexAgent`` provisions
whichever is available into the per-job ``CODEX_HOME``.
"""

from __future__ import annotations

import os
from pathlib import Path

from openscientist.providers.base import CodexCompatible, CostInfo
from openscientist.settings import get_settings


def _codex_auth_json() -> Path:
    """Path to the codex CLI's stored OAuth login (default codex home)."""
    return Path.home() / ".codex" / "auth.json"


class OpenAIDirectProvider(CodexCompatible):
    """OpenAI's API as a Codex backend (Codex's default endpoint)."""

    @property
    def id(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI API"

    def validate_required_config(self) -> list[str]:
        if os.environ.get("OPENAI_API_KEY") or _codex_auth_json().exists():
            return []
        return [
            "OpenAI provider needs auth: set OPENAI_API_KEY, or log in with "
            "the codex CLI ('codex login') so ~/.codex/auth.json exists."
        ]

    def get_cost_info(self, lookback_hours: int = 24) -> CostInfo:
        # OpenAI exposes no simple per-key spend endpoint, so report unavailable.
        return CostInfo(
            provider_name=self.display_name,
            total_spend_usd=None,
            recent_spend_usd=None,
            recent_period_hours=lookback_hours,
            data_lag_note="OpenAI per-key cost tracking is not available.",
        )

    def codex_config_overrides(self) -> list[str]:
        # Codex ships a built-in "openai" model_providers entry pointing at
        # the default endpoint, so no config.toml override is needed.
        return []

    def codex_model_name(self) -> str | None:
        # No forced default: codex uses its account/config default unless the
        # operator sets OPENSCIENTIST_MODEL (some accounts reject explicit ids).
        return get_settings().provider.model

    def codex_model_provider_id(self) -> str:
        return "openai"

    def codex_sdk_env(self) -> dict[str, str]:
        key = os.environ.get("OPENAI_API_KEY")
        return {"OPENAI_API_KEY": key} if key else {}
