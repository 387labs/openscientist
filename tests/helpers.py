"""Shared test utilities.

This module contains helper functions used across multiple test files.
These are separated from conftest.py so they can be imported without
mypy resolution issues.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from openscientist.providers.base import ClaudeCompatible, CostInfo


class StubClaudeProvider(ClaudeCompatible):
    """Minimal concrete `ClaudeCompatible` for tests that need a provider
    instance but do not exercise real provider behavior. Implements every
    abstract member with inert defaults; subclass and override as needed."""

    @property
    def id(self) -> str:
        return "stub"

    @property
    def display_name(self) -> str:
        return "Stub"

    def validate_required_config(self) -> list[str]:
        return []

    def setup_environment(self) -> None:
        return None

    def get_cost_info(self, lookback_hours: int = 24) -> CostInfo:
        return CostInfo(
            provider_name=self.display_name,
            total_spend_usd=None,
            recent_spend_usd=None,
            recent_period_hours=lookback_hours,
        )

    def claude_sdk_env(self) -> dict[str, str]:
        return {}

    def claude_model_name(self) -> str:
        return "stub-model"


async def enable_rls(session: AsyncSession) -> None:
    """Switch session from admin role to app role (enables RLS enforcement).

    Use this when you need to test RLS behavior within the same session
    that created fixture data. Switches from openscientist_admin to openscientist_app.
    """
    await session.execute(text("SET ROLE openscientist_app"))


def fake_admin_session(session_obj: Any) -> Any:
    """Build an async context manager that yields the provided session.

    Useful for monkeypatching ``get_admin_session`` in tests so that the
    test's own database session is used instead of creating a new one.
    """

    @asynccontextmanager
    async def _ctx() -> AsyncIterator[Any]:
        yield session_obj

    return _ctx
