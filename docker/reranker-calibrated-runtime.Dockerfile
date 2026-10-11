FROM industrial-ops/m4-training-worker@sha256:7bcd50262cf62f33f8a3728a53f49ee36cbcc206f2bc0970776442f0b105effe

ARG IOAP_BUILD_GIT_COMMIT=""
ARG IOAP_RUNTIME_SOURCE_SHA256=""
LABEL org.opencontainers.image.title="industrial-ops-calibrated-reranker-runtime" \
      org.opencontainers.image.revision="${IOAP_BUILD_GIT_COMMIT}" \
      io.industrial-ops.runtime.component="reranker" \
      io.industrial-ops.runtime.profile="industrial-reranker-evidence-calibration-v1" \
      io.industrial-ops.runtime.source-sha256="${IOAP_RUNTIME_SOURCE_SHA256}"

USER root
RUN mkdir -p /opt/ioap-runtime \
    && chown app:app /opt/ioap-runtime
COPY --chown=app:app src/industrial_ops_agent/deployment/reranker_runtime.py \
    /opt/ioap-runtime/reranker_runtime.py

USER app
ENTRYPOINT ["python", "-B", "/opt/ioap-runtime/reranker_runtime.py"]
