ARG IOAP_BASE_IMAGE=industrial-ops/gpu-promotion-runtime:ac9f90725e26-4844463141b6-8567fe9114e7
FROM ${IOAP_BASE_IMAGE}

USER root
COPY docker/enterprise-candidate-runtime-requirements.lock /opt/enterprise-candidate-runtime-requirements.lock
RUN pip install --no-cache-dir --only-binary=:all: --require-hashes \
    -r /opt/enterprise-candidate-runtime-requirements.lock

USER 10001:10001
