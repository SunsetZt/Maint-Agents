FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /build
COPY pyproject.toml README.md ./
COPY docker/training-requirements.lock /opt/training-requirements.lock
COPY .docker-cache/m3/wheels/pyspark-4.0.1-py2.py3-none-any.whl /opt/wheels/
RUN pip install --no-cache-dir /opt/wheels/pyspark-4.0.1-py2.py3-none-any.whl \
    && pip install --no-cache-dir --require-hashes -r /opt/training-requirements.lock

COPY src ./src
ARG IOAP_BUILD_GIT_COMMIT=""
LABEL org.opencontainers.image.title="industrial-ops-m4-training-worker" \
      org.opencontainers.image.revision="${IOAP_BUILD_GIT_COMMIT}"
ENV IOAP_IMAGE_GIT_COMMIT=${IOAP_BUILD_GIT_COMMIT}
RUN pip install --no-cache-dir --no-deps .

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app \
    && mkdir -p /app /models/huggingface \
    && chown -R app:app /app /models/huggingface

WORKDIR /app
USER app
CMD ["industrial-ops-training-worker"]
