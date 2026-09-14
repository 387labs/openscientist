"""Bulk-load parsed MEDLINE records into Postgres via asyncpg COPY."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from openscientist.pubmed_mirror.parse import iter_articles

logger = logging.getLogger(__name__)

_COLUMNS = ("pmid", "title", "abstract", "authors", "journal", "year")
_BASELINE_YEAR_RE = re.compile(r"pubmed(\d{2})n\d+\.xml")

# Must match the migration; rebuilt here once after the bulk COPY.
_INDEX_NAME = "ix_pubmed_articles_search_vector"
_CREATE_INDEX = (
    f"CREATE INDEX IF NOT EXISTS {_INDEX_NAME} ON pubmed_articles USING gin (search_vector)"
)

_ArticleRow = tuple[int, str, str | None, str | None, str | None, int | None]


def to_asyncpg_dsn(url: str) -> str:
    """Strip the SQLAlchemy driver suffix so asyncpg accepts the URL."""
    return re.sub(r"^([a-z]+)\+[a-z0-9]+://", r"\1://", url)


def baseline_year_from_files(files: Sequence[Path]) -> int | None:
    for path in files:
        match = _BASELINE_YEAR_RE.search(path.name)
        if match:
            return 2000 + int(match.group(1))
    return None


class _Counter:
    """Counts rows as they stream through COPY (which returns only a status)."""

    def __init__(self, max_articles: int | None) -> None:
        self.count = 0
        self._max = max_articles

    def rows(self, files: Sequence[Path]) -> Iterator[_ArticleRow]:
        with tqdm(total=self._max, unit="article", desc="ingesting") as bar:
            for path in files:
                bar.set_postfix_str(path.name, refresh=False)
                for article in iter_articles(path):
                    if self._max is not None and self.count >= self._max:
                        return
                    self.count += 1
                    bar.update(1)
                    yield (
                        article["pmid"],
                        article["title"],
                        article["abstract"],
                        article["authors"],
                        article["journal"],
                        article["year"],
                    )


async def ingest_files(
    files: Sequence[Path],
    *,
    dsn: str,
    baseline_year: int | None = None,
    release_date: str | None = None,
    fresh: bool = True,
    max_articles: int | None = None,
) -> int:
    """Load ``files`` into ``pubmed_articles``, record provenance, return the count.

    ``fresh`` TRUNCATEs first for an idempotent rebuild; COPY has no upsert, so
    appending across baselines is intentionally opt-out.
    """
    year = baseline_year or baseline_year_from_files(files) or 0
    release = release_date or f"baseline-{year}"

    conn = await asyncpg.connect(to_asyncpg_dsn(dsn))
    try:
        if fresh:
            await conn.execute("TRUNCATE pubmed_articles")

        # Build the reverse index once after the load: dropping it first turns
        # per-row GIN maintenance into a single bulk build, far faster at scale.
        await conn.execute(f"DROP INDEX IF EXISTS {_INDEX_NAME}")

        counter = _Counter(max_articles)
        copy_start = time.perf_counter()
        await conn.copy_records_to_table(
            "pubmed_articles", records=counter.rows(files), columns=list(_COLUMNS)
        )
        logger.info("Copied %d rows in %.1fs", counter.count, time.perf_counter() - copy_start)

        index_start = time.perf_counter()
        await conn.execute(_CREATE_INDEX)
        logger.info("Built index in %.1fs", time.perf_counter() - index_start)

        await conn.execute(
            "INSERT INTO pubmed_corpus_meta (baseline_year, release_date, article_count)"
            " VALUES ($1, $2, $3)",
            year,
            release,
            counter.count,
        )
        await conn.execute("ANALYZE pubmed_articles")
    finally:
        await conn.close()

    logger.info("Loaded %d articles (baseline %s)", counter.count, year)
    return counter.count
