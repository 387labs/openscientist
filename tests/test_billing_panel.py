"""Tests for the billing panel's cost-record rendering.

The panel is the only reader of ``cost_records``, and it went stale once already:
it showed input and output while ``cost_usd`` priced all five buckets, so the
displayed cost per token was incoherent and the dominant bucket was invisible.
The buckets it surfaces are pinned here.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

from nicegui import ui

from openscientist.database.models import CostRecord
from openscientist.webapp_components.pages import billing

_PRICED_BUCKETS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
)


def _record(**buckets: int) -> CostRecord:
    return CostRecord(
        id=uuid4(),
        job_id=uuid4(),
        created_at=datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
        updated_at=datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
        iteration=None,
        operation_type="discovery",
        provider="anthropic",
        model="claude-opus-4-6",
        cost_usd=21.75,
        **buckets,
    )


class _Scalars:
    def __init__(self, records: Sequence[CostRecord]) -> None:
        self._records = records

    def all(self) -> Sequence[CostRecord]:
        return self._records


class _Result:
    def __init__(self, records: Sequence[CostRecord]) -> None:
        self._records = records

    def scalars(self) -> _Scalars:
        return _Scalars(self._records)


class _Session:
    def __init__(self, records: Sequence[CostRecord]) -> None:
        self._records = records

    async def execute(self, _statement: object) -> _Result:
        return _Result(self._records)


@contextlib.asynccontextmanager
async def _admin_session(records: Sequence[CostRecord]) -> AsyncIterator[_Session]:
    yield _Session(records)


async def _render(
    records: Sequence[CostRecord],
) -> tuple[list[tuple[str, str | int, str]], MagicMock]:
    """Run the real render, capturing the badges and the table it builds."""
    badges: list[tuple[str, str | int, str]] = []
    table = MagicMock()
    with (
        patch.object(billing, "get_admin_session", lambda: _admin_session(records)),
        patch.object(billing, "render_stat_badges", lambda stats: badges.extend(stats)),
        patch.object(ui, "table", table),
    ):
        await billing._render_db_cost_section()
    return badges, table


async def test_totals_badges_cover_every_priced_bucket() -> None:
    """A badge row that stops at input and output cannot explain ``cost_usd``."""
    badges, _ = await _render(
        [
            _record(
                input_tokens=500_000,
                output_tokens=100_000,
                cache_read_tokens=2_000_000,
                cache_write_tokens=200_000,
                reasoning_tokens=40,
            ),
            _record(
                input_tokens=1,
                output_tokens=2,
                cache_read_tokens=3,
                cache_write_tokens=4,
                reasoning_tokens=5,
            ),
        ]
    )

    assert dict((label, value) for label, value, _color in badges) == {
        "Total Cost": "$43.5000",
        "Input Tokens": "500,001",
        "Output Tokens": "100,002",
        "Cache Read": "2,000,003",
        "Cache Write": "200,004",
        "Reasoning": "45",
    }


async def test_per_record_table_carries_every_priced_bucket() -> None:
    record = _record(
        input_tokens=500_000,
        output_tokens=100_000,
        cache_read_tokens=2_000_000,
        cache_write_tokens=200_000,
        reasoning_tokens=40,
    )

    _, table = await _render([record])

    kwargs = table.call_args.kwargs
    assert [column["field"] for column in kwargs["columns"]] == [
        "job_id",
        "model",
        "provider",
        *_PRICED_BUCKETS,
        "cost_usd",
        "created_at",
    ]
    row = kwargs["rows"][0]
    assert [row[bucket] for bucket in _PRICED_BUCKETS] == [
        500_000,
        100_000,
        2_000_000,
        200_000,
        40,
    ]
