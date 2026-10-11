FROM ubuntu:24.04 AS builder

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
    && cmake --build /src/llama.cpp/build --target llama-server --parallel 2

FROM ubuntu:24.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app \
    && mkdir -p /app /mnt/models/gguf \
    && chown -R app:app /app /mnt/models/gguf

COPY --from=builder /src/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

ARG IOAP_BUILD_GIT_COMMIT=""
LABEL org.opencontainers.image.title="industrial-ops-llama-cpp-runtime" \
      org.opencontainers.image.revision="${IOAP_BUILD_GIT_COMMIT}" \
      io.industrial-ops.llama-cpp.commit="7ba604f1cb61cd14898138e9abc0b4ff2601f180"

WORKDIR /app
USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/llama-server"]
CMD ["--help"]
