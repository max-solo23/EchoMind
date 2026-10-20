"""add_app_settings_table

Revision ID: c41e9a7d2f60
Revises: 612cc1581ccc
Create Date: 2026-10-20
"""

from alembic import op
import sqlalchemy as sa

revision = "c41e9a7d2f60"
down_revision = "612cc1581ccc"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("value", sa.JSON, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("app_settings")
