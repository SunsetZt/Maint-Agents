FROM python:3.12.10-slim-bookworm AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
WORKDIR /build
RUN python -m venv /opt/venv
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install '.[local-ai,multimodal-video]' \
    && docling-tools models download layout tableformer -o /opt/docling-artifacts

FROM python:3.12.10-slim-bookworm AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    IOAP_DOCLING_ARTIFACTS_PATH=/opt/docling-artifacts \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/docling-artifacts /opt/docling-artifacts
USER app
CMD ["python", "-m", "industrial_ops_agent.orchestration.runner"]
