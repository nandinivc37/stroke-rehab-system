"""SQLAlchemy connection foundation for the staged database migration.

The existing application data-access layer remains SQLite-specific. This
module is infrastructure for new ORM-backed repositories and must not be
mistaken for a drop-in replacement for src.database.db.
"""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.paths import DB_PATH


class Base(DeclarativeBase):
    """Base class for ORM models introduced during repository migration."""


def configured_database_url() -> str:
    """Return the configured SQLAlchemy URL.

    Defaults to the existing local SQLite database. PostgreSQL deployments
    should set NARS_DATABASE_URL to a psycopg SQLAlchemy URL, for example
    postgresql+psycopg://user:password@host:5432/nars.
    """
    configured = os.environ.get("NARS_DATABASE_URL", "").strip()
    if configured:
        return configured
    return "sqlite:///" + Path(DB_PATH).resolve().as_posix()


def create_database_engine(url: str | None = None) -> Engine:
    """Create an engine without creating or changing any database tables."""
    database_url = url or configured_database_url()
    options = {"pool_pre_ping": True}
    if database_url.startswith("sqlite:"):
        options["connect_args"] = {"check_same_thread": False}
    return create_engine(database_url, **options)


def create_session_factory(engine: Engine):
    """Create a SQLAlchemy session factory for an application repository."""
    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
