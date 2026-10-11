FROM python:3.12.10-slim-bookworm AS java

ARG TEMURIN_JRE_SHA256=ef491a51a46ef90cc47fbc4abb219fde32483ff91be5ec66ddc896df43524b27
COPY .docker-cache/m3/OpenJDK17U-jre_x64_linux_hotspot_17.0.20_8.tar.gz /tmp/temurin-jre.tar.gz
RUN echo "${TEMURIN_JRE_SHA256}  /tmp/temurin-jre.tar.gz" | sha256sum --check --strict \
    && mkdir -p /opt/java \
    && tar --extract --gzip --file /tmp/temurin-jre.tar.gz --directory /opt/java --strip-components=1 \
    && rm /tmp/temurin-jre.tar.gz

FROM python:3.12.10-slim-bookworm AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
WORKDIR /build
RUN python -m venv /opt/venv
COPY pyproject.toml README.md ./
COPY src ./src
COPY .docker-cache/m3/wheels /opt/wheels
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked pip install --no-index --find-links=/opt/wheels '.[data-pipeline]' \
    "apache-airflow==3.3.0" \
    "apache-airflow-providers-standard>=1.10,<2"

FROM python:3.12.10-slim-bookworm AS runtime

ENV PATH=/opt/venv/bin:/opt/java/bin:/opt/venv/lib/python3.12/site-packages/pyspark/bin:$PATH \
    JAVA_HOME=/opt/java \
    SPARK_HOME=/opt/venv/lib/python3.12/site-packages/pyspark \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /opt/airflow app \
    && mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/spark/shared \
    && chown -R app:app /opt/airflow /opt/spark/shared
COPY --from=build /opt/venv /opt/venv
COPY --from=java /opt/java /opt/java
COPY --chown=app:app airflow/dags /opt/airflow/dags
WORKDIR /opt/airflow
USER app
CMD ["airflow", "standalone"]
