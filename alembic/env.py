"""Alembic environment for the staged SQLAlchemy model migration."""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.database.sqlalchemy_support import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def database_url():
    return os.environ.get(
        "NARS_DATABASE_URL",
        config.get_main_option("sqlalchemy.url"),
    )


def run_migrations_offline():
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = database_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    try:
        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
