"""Database session factory that installs tenant context per transaction."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from industrial_ops_agent.persistence.tenant import TenantContext


class Database:
    def __init__(self, url: str, *, pool_timeout_seconds: int = 5) -> None:
        self.engine: Engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_timeout=pool_timeout_seconds,
        )
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    @classmethod
    def from_engine(cls, engine: Engine) -> Database:
        """Build a database boundary around an externally managed engine."""

        database = cls.__new__(cls)
        database.engine = engine
        database._sessions = sessionmaker(engine, expire_on_commit=False)
        return database

    @contextmanager
    def transaction(self, context: TenantContext) -> Iterator[Session]:
        with self._sessions.begin() as session:
            session.info["tenant_id"] = context.tenant_id
            if session.bind is not None and session.bind.dialect.name == "postgresql":
                session.execute(
                    text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
                    {"tenant_id": context.tenant_id},
                )
            yield session

    def dispose(self) -> None:
        self.engine.dispose()
