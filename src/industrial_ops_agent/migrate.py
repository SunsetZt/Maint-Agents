"""Vault-aware Alembic upgrade entrypoint for Compose Lite."""

from __future__ import annotations

from alembic import command
from alembic.config import Config

from industrial_ops_agent.config import get_settings
from industrial_ops_agent.secrets import SecretName, build_secret_provider


def main() -> None:
    settings = get_settings()
    database_url = build_secret_provider(settings).get(SecretName.DATABASE_URL)
    configuration = Config("alembic.ini")
    configuration.set_main_option("sqlalchemy.url", database_url.reveal())
    command.upgrade(configuration, "head")
    print("M1 database migration is current.")


if __name__ == "__main__":
    main()
