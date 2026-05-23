"""In-process tests for the standalone `search_pubmed` tool."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent
from sqlalchemy import delete

from openscientist.database import AsyncSessionLocal
from openscientist.database.models.job import Job
from openscientist.knowledge_state import KnowledgeState
from openscientist_tools.pubmed import search_pubmed
from openscientist_tools.state import STATE

_ESEARCH_TWO_HITS: dict[str, Any] = {"esearchresult": {"idlist": ["12345678", "87654321"]}}
_ESEARCH_EMPTY: dict[str, Any] = {"esearchresult": {"idlist": []}}
_EFETCH_TWO_PAPERS = """<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Carnosine in cold-adapted cells</ArticleTitle>
        <Abstract><AbstractText>Elevated carnosine was observed.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Smith</LastName></Author></AuthorList>
        <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>87654321</PMID>
      <Article>
        <ArticleTitle>Thermal stress response</ArticleTitle>
        <Abstract><AbstractText>Heat shock pathway activation.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Jones</LastName></Author></AuthorList>
        <Journal><JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def _fake_requests_get(
    *, esearch_payload: dict[str, Any], efetch_xml: str
) -> Callable[..., MagicMock]:
    def _get(url: str, **_kw: Any) -> MagicMock:
        response = MagicMock()
        if "esearch" in url:
            response.json.return_value = esearch_payload
            response.text = ""
        else:
            response.text = efetch_xml
            response.json.return_value = {}
        response.raise_for_status = MagicMock()
        return response

    return _get


@pytest.fixture(autouse=True)
def _state_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(STATE, "job_id", "test-job-uuid")


def test_returns_formatted_markdown(
    monkeypatch: pytest.MonkeyPatch,
    patched_ks_persistence: KnowledgeState,
) -> None:
    monkeypatch.setattr(
        "openscientist.literature.requests.get",
        _fake_requests_get(esearch_payload=_ESEARCH_TWO_HITS, efetch_xml=_EFETCH_TWO_PAPERS),
    )

    result = search_pubmed(query="cold adapted carnosine", max_results=10)

    assert result.startswith("Found 2 papers for query: 'cold adapted carnosine'")
    assert "PMID: 12345678" in result
    assert "PMID: 87654321" in result
    assert "Carnosine in cold-adapted cells" in result


def test_writes_literature_and_log_to_knowledge_state(
    monkeypatch: pytest.MonkeyPatch,
    patched_ks_persistence: KnowledgeState,
) -> None:
    monkeypatch.setattr(
        "openscientist.literature.requests.get",
        _fake_requests_get(esearch_payload=_ESEARCH_TWO_HITS, efetch_xml=_EFETCH_TWO_PAPERS),
    )

    search_pubmed(query="cold adapted carnosine", description="why")

    assert len(patched_ks_persistence.data["literature"]) == 2
    pmids = {entry["pmid"] for entry in patched_ks_persistence.data["literature"]}
    assert pmids == {"12345678", "87654321"}

    last_log = patched_ks_persistence.data["analysis_log"][-1]
    assert last_log["action"] == "search_pubmed"
    assert last_log["query"] == "cold adapted carnosine"
    assert last_log["results_count"] == 2
    assert last_log["description"] == "why"


def test_empty_results(
    monkeypatch: pytest.MonkeyPatch,
    patched_ks_persistence: KnowledgeState,
) -> None:
    monkeypatch.setattr(
        "openscientist.literature.requests.get",
        _fake_requests_get(esearch_payload=_ESEARCH_EMPTY, efetch_xml=""),
    )

    result = search_pubmed(query="nonexistent topic xyzzy")

    assert result == "No papers found for query: 'nonexistent topic xyzzy'"
    assert patched_ks_persistence.data["literature"] == []
    last_log = patched_ks_persistence.data["analysis_log"][-1]
    assert last_log["action"] == "search_pubmed"
    assert last_log["results_count"] == 0


async def test_subprocess_smoke_real_ncbi(
    tmp_path: Path,
    server_env: Callable[..., dict[str, str]],
    server_params: Callable[[dict[str, str]], StdioServerParameters],
    test_database_url: str,
    _apply_migrations_once: None,
) -> None:
    """End-to-end: subprocess MCP server hits real NCBI and writes to real DB."""
    job_id = uuid4()
    async with AsyncSessionLocal(thread_safe=True) as session:
        session.add(
            Job(
                id=job_id,
                research_question="CRISPR smoke",
                description="pubmed smoke test",
                llm_provider="mock",
                llm_config={"model": "mock-model-v1"},
                status="pending",
            )
        )
        await session.commit()

    try:
        env = server_env(tmp_path, OPENSCIENTIST_JOB_ID=str(job_id))
        env["DATABASE_URL"] = test_database_url
        env["OPENSCIENTIST_SECRET_KEY"] = os.environ["OPENSCIENTIST_SECRET_KEY"]

        async with stdio_client(server_params(env)) as (read, write):
            async with ClientSession(read, write) as mcp_session:
                await mcp_session.initialize()
                result = await mcp_session.call_tool(
                    "search_pubmed", {"query": "CRISPR Cas9", "max_results": 1}
                )
                (block,) = result.content
                assert isinstance(block, TextContent)
                assert block.text.startswith("Found ") and "PMID:" in block.text

        reloaded = KnowledgeState.load_from_database_sync(str(job_id))
        assert len(reloaded.data["literature"]) >= 1
        last_log = reloaded.data["analysis_log"][-1]
        assert last_log["action"] == "search_pubmed"
        assert last_log["query"] == "CRISPR Cas9"
    finally:
        async with AsyncSessionLocal(thread_safe=True) as session:
            await session.execute(delete(Job).where(Job.id == job_id))
            await session.commit()
