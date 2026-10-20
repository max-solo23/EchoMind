"""cache_persona_hash_drop_tfidf

The per-request TF-IDF vectors were never read; similar-question matching now indexes the
question text in memory. persona_hash scopes those matches to the current persona.

Revision ID: d7a3b5e19c42
Revises: c41e9a7d2f60
Create Date: 2026-10-20
"""

from alembic import op
import sqlalchemy as sa

revision = "d7a3b5e19c42"
down_revision = "c41e9a7d2f60"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("cached_answers", sa.Column("persona_hash", sa.String(64), nullable=True))
    op.drop_column("cached_answers", "tfidf_vector")


def downgrade() -> None:
    op.add_column(
        "cached_answers",
        sa.Column("tfidf_vector", sa.Text, nullable=False, server_default="[]"),
    )
    op.drop_column("cached_answers", "persona_hash")
