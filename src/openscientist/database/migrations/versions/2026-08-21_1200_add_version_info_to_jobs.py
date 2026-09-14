"""Add version_info provenance column to jobs.

Revision ID: add_version_info
Revises: add_token_buckets
Create Date: 2026-08-21 12:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "add_version_info"
down_revision: str | None = "add_token_buckets"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column(
            "version_info",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Runtime provenance (claude_code_version, claude_agent_sdk_version, etc.)",
        ),
    )


def downgrade() -> None:
    op.drop_column("jobs", "version_info")
