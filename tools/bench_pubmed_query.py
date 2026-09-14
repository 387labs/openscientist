"""Time unrelated-table query latency, to run before and after the PubMed load.

uv run python tools/bench_pubmed_query.py
"""

from __future__ import annotations

import asyncio
import statistics
import time

import asyncpg  # type: ignore[import-untyped]

from openscientist.pubmed_mirror.ingest import to_asyncpg_dsn
from openscientist.settings import get_settings

# Ordinary application queries whose latency should be unaffected by the size
# of pubmed_articles. Kept read-only and independent of row counts.
_QUERIES: list[tuple[str, str]] = [
    ("jobs count", "SELECT count(*) FROM jobs"),
    ("recent jobs", "SELECT id FROM jobs ORDER BY created_at DESC LIMIT 20"),
    ("literature count", "SELECT count(*) FROM literature"),
    ("skills count", "SELECT count(*) FROM skills"),
    ("users by email", "SELECT id FROM users WHERE email = 'nobody@example.org'"),
]

_REPEATS = 50


async def _time_query(conn: asyncpg.Connection, sql: str) -> list[float]:
    timings: list[float] = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        await conn.fetch(sql)
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


async def main() -> None:
    dsn = to_asyncpg_dsn(get_settings().database.effective_admin_database_url)
    conn = await asyncpg.connect(dsn)
    try:
        pubmed_rows = await conn.fetchval("SELECT count(*) FROM pubmed_articles")
        print(f"pubmed_articles rows: {pubmed_rows}\n")
        print(f"{'query':<20}{'median (ms)':>14}{'p95 (ms)':>12}")
        print("-" * 46)
        for label, sql in _QUERIES:
            timings = sorted(await _time_query(conn, sql))
            median = statistics.median(timings)
            p95 = timings[int(len(timings) * 0.95) - 1]
            print(f"{label:<20}{median:>14.3f}{p95:>12.3f}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
