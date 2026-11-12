"""Alembic env: URL берётся из настроек приложения (async-драйвер заменяется на sync)."""
from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool

from backend.core.config import get_settings
from backend.db.models import Base

target_metadata = Base.metadata


def sync_url() -> str:
    return get_settings().db_url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg2")


def run_migrations_offline() -> None:
    context.configure(url=sync_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(sync_url(), poolclass=pool.NullPool)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
