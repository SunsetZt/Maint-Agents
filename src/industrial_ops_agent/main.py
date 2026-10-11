"""Production API process entrypoint."""

from __future__ import annotations

import uvicorn

from industrial_ops_agent.api.app import create_app
from industrial_ops_agent.config import get_settings
from industrial_ops_agent.runtime import create_runtime_app

app = create_app()


def run() -> None:
    """Run the API process using typed configuration."""

    settings = get_settings()
    runtime_app = create_runtime_app(settings)
    uvicorn.run(
        runtime_app,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_config=None,
    )


if __name__ == "__main__":
    run()
