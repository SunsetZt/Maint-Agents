FROM docker.io/vllm/vllm-openai:v0.26.0@sha256:770fe65b2c73ee74a5c42165cf3433de4048cc2cd9c57a937ca4e35aba5aa87b

ARG IOAP_BUILD_GIT_COMMIT=""
ARG IOAP_SOURCE_REPOSITORY=""
LABEL org.opencontainers.image.title="industrial-ops-model-runtime" \
      org.opencontainers.image.source="${IOAP_SOURCE_REPOSITORY}" \
      org.opencontainers.image.revision="${IOAP_BUILD_GIT_COMMIT}" \
      org.opencontainers.image.version="vllm-0.26.0"
