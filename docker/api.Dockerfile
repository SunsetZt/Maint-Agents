FROM python:3.12.10-slim-bookworm AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
WORKDIR /build
RUN python -m venv /opt/venv
COPY pyproject.toml README.md ./
COPY src ./src
COPY .docker-cache/m3/wheels /opt/wheels
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-index --find-links=/opt/wheels .

FROM python:3.12.10-slim-bookworm AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app alembic ./alembic
USER app
EXPOSE 8000
CMD ["industrial-ops-api"]
