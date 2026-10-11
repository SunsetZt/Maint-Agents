FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        'aiortc>=1.13,<2' \
        'av>=14,<17' \
        'fastapi>=0.116,<1' \
        'httpx>=0.28,<1' \
        'pydantic-settings>=2.10,<3' \
        'uvicorn>=0.35,<1' \
    && python -m pip install --no-deps .

USER 10001:10001
EXPOSE 7880
CMD ["industrial-ops-media-server"]
