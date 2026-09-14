"""Local full-text search against Postgres (uses the test database)."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from openscientist.database.models.pubmed_article import PubMedArticle
from openscientist.pubmed_mirror.query import run_search


@pytest.mark.asyncio
async def test_run_search_ranks_and_shapes_results(db_session: AsyncSession) -> None:
    db_session.add_all(
        [
            PubMedArticle(
                pmid=1,
                title="CRISPR gene editing in mice",
                abstract="A study of CRISPR off-target effects.",
                authors="Doe",
                journal="Nature",
                year=2020,
            ),
            PubMedArticle(
                pmid=2,
                title="Weather patterns over the Pacific",
                abstract="Unrelated climate observations.",
                authors="Roe",
                journal="Science",
                year=2019,
            ),
        ]
    )
    # flush so Postgres computes the generated search_vector for the query.
    await db_session.flush()

    papers = await run_search(db_session, "crispr", max_results=5)

    assert [p["pmid"] for p in papers] == ["1"]
    paper = papers[0]
    assert paper["title"] == "CRISPR gene editing in mice"
    assert paper["authors"] == "Doe"
    assert paper["year"] == "2020"


@pytest.mark.asyncio
async def test_run_search_blank_query_returns_empty(db_session: AsyncSession) -> None:
    assert await run_search(db_session, "   ", max_results=5) == []
