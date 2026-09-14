"""PubMed article table for the optional local literature mirror."""

from sqlalchemy import BigInteger, Computed, Integer, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base

# Two-arg to_tsvector pins the config so the expression is IMMUTABLE and thus
# valid inside a STORED generated column. Title weighted above abstract.
SEARCH_VECTOR_EXPRESSION = (
    "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
    "setweight(to_tsvector('english', coalesce(abstract, '')), 'B')"
)


class PubMedArticle(Base):
    """A MEDLINE record keyed by PMID. Global reference data, no RLS."""

    __tablename__ = "pubmed_articles"

    pmid: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)
    journal: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed(SEARCH_VECTOR_EXPRESSION, persisted=True), nullable=True
    )
