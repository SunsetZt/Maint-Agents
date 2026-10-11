FROM docker.io/library/python:3.12.10-slim-bookworm@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db

ARG IOAP_BUILD_GIT_COMMIT=""
ARG IOAP_SOURCE_REPOSITORY=""
LABEL org.opencontainers.image.title="industrial-ops-project-staging-model-router" \
      org.opencontainers.image.source="${IOAP_SOURCE_REPOSITORY}" \
      org.opencontainers.image.revision="${IOAP_BUILD_GIT_COMMIT}"

WORKDIR /opt/industrial-ops
COPY src/industrial_ops_agent/model_gateway/router_runtime.py ./router_runtime.py
COPY src/industrial_ops_agent/model_gateway/owned_runtime.py ./owned_runtime.py

USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["python3", "/opt/industrial-ops/router_runtime.py"]
