FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ARG IOAP_BUILD_GIT_COMMIT=""
ARG IOAP_SOURCE_REPOSITORY=""
LABEL org.opencontainers.image.title="industrial-ops-embedding-runtime" \
      org.opencontainers.image.source="${IOAP_SOURCE_REPOSITORY}" \
      org.opencontainers.image.revision="${IOAP_BUILD_GIT_COMMIT}"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    IOAP_IMAGE_GIT_COMMIT=${IOAP_BUILD_GIT_COMMIT}

WORKDIR /build
COPY pyproject.toml README.md ./
COPY docker/training-requirements.lock /opt/training-requirements.lock
COPY src ./src
COPY .docker-cache/m3/wheels/pyspark-4.0.1-py2.py3-none-any.whl /opt/wheels/
RUN pip install --no-cache-dir /opt/wheels/pyspark-4.0.1-py2.py3-none-any.whl \
    && pip install --no-cache-dir --require-hashes -r /opt/training-requirements.lock \
    && pip install --no-cache-dir --no-deps .

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app \
    && mkdir -p /app /mnt/models/embedding \
    && chown -R app:app /app /mnt/models/embedding

WORKDIR /app
USER app
ENTRYPOINT ["industrial-ops-embedding-runtime"]
