ARG BASE_IMAGE=industrial-ops/m4-training-worker:local
FROM ${BASE_IMAGE}

USER root
COPY --chown=app:app base/ /models/base/
COPY --chown=app:app adapter/ /models/adapter/
COPY --chown=app:app identity.json /models/identity.json
COPY --chown=app:app gpu_lab_runtime.py /app/gpu_lab_runtime.py

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    IOAP_GPU_LAB_BASE_MODEL_PATH=/models/base \
    IOAP_GPU_LAB_ADAPTER_PATH=/models/adapter \
    IOAP_GPU_LAB_IDENTITY_PATH=/models/identity.json \
    IOAP_GPU_LAB_PORT=8080

USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["python", "-B", "/app/gpu_lab_runtime.py"]
