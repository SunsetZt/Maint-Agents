FROM ubuntu:24.04 AS llama-cpp-builder

ARG LLAMA_CPP_COMMIT=7ba604f1cb61cd14898138e9abc0b4ff2601f180
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git \
    && rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    for retry_delay in 0 3 8; do \
        if [ "${retry_delay}" != "0" ]; then sleep "${retry_delay}"; fi; \
        rm -rf /src/llama.cpp; \
        git init /src/llama.cpp; \
        git -C /src/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git; \
        if git -c http.version=HTTP/1.1 -C /src/llama.cpp fetch --depth 1 origin "${LLAMA_CPP_COMMIT}" \
            && git -C /src/llama.cpp checkout --detach FETCH_HEAD; then \
            break; \
        fi; \
    done; \
    test "$(git -C /src/llama.cpp rev-parse HEAD)" = "${LLAMA_CPP_COMMIT}"; \
    cmake -S /src/llama.cpp -B /src/llama.cpp/build \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_CURL=OFF \
    && cmake --build /src/llama.cpp/build --target llama-quantize llama-cli --parallel 2

FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

ENV VIRTUAL_ENV=/opt/ioap-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRITON_CACHE_DIR=/tmp/triton \
    TOKENIZERS_PARALLELISM=false \
    IOAP_LLAMA_CPP_CONVERT_SCRIPT=/opt/llama.cpp/convert_hf_to_gguf.py \
    IOAP_LLAMA_CPP_QUANTIZE_BINARY=/opt/llama.cpp/build/bin/llama-quantize \
    IOAP_LLAMA_CPP_CLI_BINARY=/opt/llama.cpp/build/bin/llama-cli

WORKDIR /build
COPY docker/quantization-requirements.lock /opt/quantization-requirements.lock
COPY .docker-cache/m3/wheels/pyspark-4.0.1-py2.py3-none-any.whl /opt/wheels/
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m venv --without-pip --system-site-packages "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/python" -m pip install \
        /opt/wheels/pyspark-4.0.1-py2.py3-none-any.whl \
    && "${VIRTUAL_ENV}/bin/python" -m pip install \
        --require-hashes -r /opt/quantization-requirements.lock

COPY --from=llama-cpp-builder /src/llama.cpp /opt/llama.cpp

COPY pyproject.toml README.md ./
COPY src ./src
RUN "${VIRTUAL_ENV}/bin/python" -m pip install --no-deps .

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app \
    && mkdir -p /app /models/huggingface \
    && chown -R app:app /app /models/huggingface

ARG IOAP_BUILD_GIT_COMMIT=""
LABEL org.opencontainers.image.title="industrial-ops-quantization-worker" \
      org.opencontainers.image.revision="${IOAP_BUILD_GIT_COMMIT}"
ENV IOAP_IMAGE_GIT_COMMIT=${IOAP_BUILD_GIT_COMMIT}

WORKDIR /app
USER app
CMD ["industrial-ops-quantization-worker"]
