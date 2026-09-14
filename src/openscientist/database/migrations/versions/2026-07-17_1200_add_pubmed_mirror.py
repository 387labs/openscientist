"""Add the optional local PubMed mirror tables.

Revision ID: add_pubmed_mirror
Revises: rename_title_to_rq
Create Date: 2026-07-17 12:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import TSVECTOR

# revision identifiers, used by Alembic.
revision: str = "add_pubmed_mirror"
down_revision: str | None = "rename_title_to_rq"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Mirrors PubMedArticle.SEARCH_VECTOR_EXPRESSION (kept self-contained).
_SEARCH_VECTOR_EXPRESSION = (
    "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
    "setweight(to_tsvector('english', coalesce(abstract, '')), 'B')"
)


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "pubmed_articles",
        sa.Column("pmid", sa.BigInteger(), primary_key=True, autoincrement=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("authors", sa.Text(), nullable=True),
        sa.Column("journal", sa.Text(), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column(
            "search_vector",
            TSVECTOR(),
            sa.Computed(_SEARCH_VECTOR_EXPRESSION, persisted=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_pubmed_articles_search_vector",
        "pubmed_articles",
        ["search_vector"],
        postgresql_using="gin",
    )

    op.create_table(
        "pubmed_corpus_meta",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("baseline_year", sa.Integer(), nullable=False),
        sa.Column("release_date", sa.String(length=64), nullable=False),
        sa.Column("article_count", sa.BigInteger(), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
    )

    # Global reference data: readable by the app role, no per-user RLS. Guarded
    # so a database provisioned without the app role still migrates cleanly.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'openscientist_app') THEN
                GRANT SELECT ON pubmed_articles TO openscientist_app;
                GRANT SELECT ON pubmed_corpus_meta TO openscientist_app;
            END IF;
        END
        $$
        """
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index("ix_pubmed_articles_search_vector", table_name="pubmed_articles")
    op.drop_table("pubmed_corpus_meta")
    op.drop_table("pubmed_articles")
