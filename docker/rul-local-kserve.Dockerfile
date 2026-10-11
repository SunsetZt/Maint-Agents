ARG IOAP_BASE_IMAGE=industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7
FROM ${IOAP_BASE_IMAGE}

ARG IOAP_BUILD_GIT_COMMIT=""
ARG IOAP_RUL_MODEL_SOURCE="artifacts/rul-promotion-lab/invalid/model"
LABEL org.opencontainers.image.title="industrial-ops-rul-local-kserve-runtime" \
      org.opencontainers.image.revision="${IOAP_BUILD_GIT_COMMIT}" \
      ioap.openai.com.classification="SIMULATED_NON_PRODUCTION"

USER root
COPY --chown=10001:10001 src/industrial_ops_agent /opt/ioap-rul/industrial_ops_agent
COPY --chown=10001:10001 ${IOAP_RUL_MODEL_SOURCE}/ /mnt/models/rul/

ENV PYTHONPATH=/opt/ioap-rul \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    IOAP_ENVIRONMENT_CLASSIFICATION=SIMULATED_NON_PRODUCTION

USER 10001:10001
ENTRYPOINT ["python", "-B", "-m", "industrial_ops_agent.predictive_maintenance.rul_runtime"]
