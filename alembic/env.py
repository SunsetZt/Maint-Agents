from __future__ import annotations

from logging.config import fileConfig
from typing import Any

from alembic import context
from alembic.ddl.postgresql import PostgresqlImpl
from sqlalchemy import (
    Column,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    engine_from_config,
    inspect,
    pool,
    text,
)
from sqlalchemy.engine import Connection

from industrial_ops_agent.persistence.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
_VERSION_COLUMN_LENGTH = 128


class _IndustrialOpsPostgresqlImpl(PostgresqlImpl):
    """Keep Alembic revision identifiers wider than its 32-byte default."""

    __dialect__ = "postgresql"

    def version_table_impl(
        self,
        *,
        version_table: str,
        version_table_schema: str | None,
        version_table_pk: bool,
        **kw: Any,
    ) -> Table:
        table = Table(
            version_table,
            MetaData(),
            Column("version_num", String(_VERSION_COLUMN_LENGTH), nullable=False),
            schema=version_table_schema,
        )
        if version_table_pk:
            table.append_constraint(
                PrimaryKeyConstraint("version_num", name=f"{version_table}_pkc")
            )
        return table


def _ensure_version_column_capacity(connection: Connection) -> None:
    inspector = inspect(connection)
    if not inspector.has_table("alembic_version"):
        return
    version_column = next(
        column
        for column in inspector.get_columns("alembic_version")
        if column["name"] == "version_num"
    )
    current_length = getattr(version_column["type"], "length", None)
    if current_length is not None and current_length < _VERSION_COLUMN_LENGTH:
        connection.execute(
            text(
                "ALTER TABLE alembic_version ALTER COLUMN version_num "
                f"TYPE VARCHAR({_VERSION_COLUMN_LENGTH})"
            )
        )


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    # Introspection starts an implicit SQLAlchemy transaction. Keep that
    # transaction separate so Alembic owns and commits its migration transaction.
    with connectable.begin() as connection:
        _ensure_version_column_capacity(connection)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
