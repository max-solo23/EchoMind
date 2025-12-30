"""add_context_aware_cache_with_ttl

Revision ID: b3dc51b73835
Revises: 9d127d350d71
Create Date: 2025-12-30 10:17:50.155481

This migration adds context-aware caching with TTL support:
- cache_key: SHA256 hash of (context + question) for unique lookups
- cache_type: "knowledge" (30 day TTL) or "conversational" (24 hour TTL)
- expires_at: When the cache entry expires
- context_preview: Truncated context for debugging

Fixes the issue where short inputs like "ok" would return cached responses
from unrelated conversations.
"""
from typing import Sequence, Union
import hashlib

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3dc51b73835'
down_revision: Union[str, Sequence[str], None] = '9d127d350d71'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add context-aware caching columns to cached_answers table.

    Steps:
    1. Add new columns (nullable initially)
    2. Backfill existing rows with generated cache_key and default TTL
    3. Make cache_key non-nullable and add unique constraint
    """
    # Step 1: Add new columns (nullable for now)
    op.add_column('cached_answers', sa.Column('cache_key', sa.String(64), nullable=True))
    op.add_column('cached_answers', sa.Column('context_preview', sa.String(200), nullable=True))
    op.add_column('cached_answers', sa.Column('cache_type', sa.String(20), nullable=True))
    op.add_column('cached_answers', sa.Column('expires_at', sa.DateTime(), nullable=True))

    # Step 2: Backfill existing rows
    # - cache_key = SHA256 hash of question (no context for existing entries)
    # - cache_type = 'knowledge' (assume existing are knowledge-type)
    # - expires_at = 30 days from now
    connection = op.get_bind()

    # Get all existing cached answers
    result = connection.execute(sa.text("SELECT id, question FROM cached_answers"))
    rows = result.fetchall()

    for row in rows:
        cache_id = row[0]
        question = row[1]
        # Generate cache_key from question only (no context for legacy entries)
        cache_key = hashlib.sha256(f"||{question}".encode()).hexdigest()

        connection.execute(
            sa.text("""
                UPDATE cached_answers
                SET cache_key = :cache_key,
                    cache_type = 'knowledge',
                    expires_at = NOW() + INTERVAL '30 days'
                WHERE id = :id
            """),
            {"cache_key": cache_key, "id": cache_id}
        )

    # Step 3: Make cache_key non-nullable and add constraints
    op.alter_column('cached_answers', 'cache_key', nullable=False)
    op.alter_column('cached_answers', 'cache_type', nullable=False, server_default='knowledge')

    # Add indexes
    op.create_index('ix_cached_answers_cache_key', 'cached_answers', ['cache_key'], unique=True)
    op.create_index('ix_cached_answers_expires_at', 'cached_answers', ['expires_at'])
    op.create_index('ix_cached_answers_cache_type', 'cached_answers', ['cache_type'])


def downgrade() -> None:
    """Remove context-aware caching columns."""
    # Drop indexes first
    op.drop_index('ix_cached_answers_cache_type', table_name='cached_answers')
    op.drop_index('ix_cached_answers_expires_at', table_name='cached_answers')
    op.drop_index('ix_cached_answers_cache_key', table_name='cached_answers')

    # Drop columns
    op.drop_column('cached_answers', 'expires_at')
    op.drop_column('cached_answers', 'cache_type')
    op.drop_column('cached_answers', 'context_preview')
    op.drop_column('cached_answers', 'cache_key')
