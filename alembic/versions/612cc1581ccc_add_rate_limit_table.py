"""add_rate_limit_table
to generate id: python -c "import secrets; print(secrets.token_hex(6))"

Revision ID: 612cc1581ccc
Revises: b3dc51b73835
Create Date: 2026-01-22
"""

from alembic import op
import sqlalchemy as sa

revision = "612cc1581ccc"
down_revision = "b3dc51b73835"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rate_limit",
        sa.Column("key", sa.String(500), primary_key=True),
        sa.Column("count", sa.Integer, nullable=False, default=0),
        sa.Column("expiry", sa.Integer, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("rate_limit")
