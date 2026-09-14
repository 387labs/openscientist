"""Stream-parse NCBI MEDLINE baseline XML into article records."""

from __future__ import annotations

import gzip
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import IO, TypedDict


class Article(TypedDict):
    pmid: int
    title: str
    abstract: str | None
    authors: str | None
    journal: str | None
    year: int | None


_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b")


def _open(path: Path) -> IO[bytes]:
    if path.suffix == ".gz":
        return gzip.open(path, "rb")  # type: ignore[return-value]
    return path.open("rb")


def _node_text(elem: ET.Element | None) -> str:
    """Flatten an element's text, including nested inline markup."""
    return "".join(elem.itertext()).strip() if elem is not None else ""


def _extract_year(article: ET.Element) -> int | None:
    year = article.find(".//JournalIssue/PubDate/Year")
    if year is not None and year.text and year.text.strip().isdigit():
        return int(year.text.strip())
    # Fall back to free-form MedlineDate strings like "2011 Nov-Dec".
    medline = article.find(".//JournalIssue/PubDate/MedlineDate")
    if medline is not None and medline.text:
        match = _YEAR_RE.search(medline.text)
        if match:
            return int(match.group(1))
    return None


def _parse_article(article: ET.Element) -> Article | None:
    pmid = article.find("./MedlineCitation/PMID")
    if pmid is None or not (pmid.text or "").strip().isdigit():
        return None

    abstract = " ".join(
        text for elem in article.findall(".//Abstract/AbstractText") if (text := _node_text(elem))
    )
    authors = ", ".join(
        name
        for elem in article.findall(".//AuthorList/Author/LastName")
        if (name := (elem.text or "").strip())
    )

    return Article(
        pmid=int(pmid.text.strip()),  # type: ignore[union-attr]
        title=_node_text(article.find(".//ArticleTitle")),
        abstract=abstract or None,
        authors=authors or None,
        journal=_node_text(article.find(".//Journal/Title")) or None,
        year=_extract_year(article),
    )


def iter_articles(path: Path) -> Iterator[Article]:
    """Yield article records from one baseline file (plain or gzipped XML)."""
    with _open(path) as handle:
        for _event, elem in ET.iterparse(handle, events=("end",)):
            if elem.tag != "PubmedArticle":
                continue
            article = _parse_article(elem)
            if article is not None:
                yield article
            elem.clear()
