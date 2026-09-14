"""Unit tests for the MEDLINE baseline parser (no DB, no network)."""

from __future__ import annotations

import gzip
from pathlib import Path

from openscientist.pubmed_mirror.parse import iter_articles

_SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal>
          <Title>Journal of Testing</Title>
          <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>CRISPR editing of <i>genes</i> in mice</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">First part.</AbstractText>
          <AbstractText Label="RESULTS">Second part.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Doe</LastName></Author>
          <Author><LastName>Roe</LastName></Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>222</PMID>
      <Article>
        <Journal>
          <JournalIssue><PubDate><MedlineDate>2011 Nov-Dec</MedlineDate></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>No abstract here</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_iter_articles_extracts_fields(tmp_path: Path) -> None:
    path = tmp_path / "sample.xml"
    path.write_text(_SAMPLE_XML)

    articles = list(iter_articles(path))

    assert len(articles) == 2
    first, second = articles

    assert first["pmid"] == 12345678
    assert first["title"] == "CRISPR editing of genes in mice"
    assert first["abstract"] == "First part. Second part."
    assert first["authors"] == "Doe, Roe"
    assert first["journal"] == "Journal of Testing"
    assert first["year"] == 2021

    # Missing abstract/authors collapse to None; year falls back to MedlineDate.
    assert second["pmid"] == 222
    assert second["abstract"] is None
    assert second["authors"] is None
    assert second["year"] == 2011


def test_iter_articles_reads_gzip(tmp_path: Path) -> None:
    path = tmp_path / "sample.xml.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(_SAMPLE_XML)

    pmids = [article["pmid"] for article in iter_articles(path)]

    assert pmids == [12345678, 222]
