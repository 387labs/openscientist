"""Provenance for the local PubMed mirror: one row per ingest."""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class PubMedCorpusMeta(Base):
    """Snapshot metadata for one ingest of the MEDLINE baseline."""

    __tablename__ = "pubmed_corpus_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    baseline_year: Mapped[int] = mapped_column(Integer, nullable=False)
    release_date: Mapped[str] = mapped_column(String(64), nullable=False)
    article_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
