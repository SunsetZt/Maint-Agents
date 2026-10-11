"""Backfillable governed dataset snapshot DAG."""

from __future__ import annotations

from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

with DAG(
    dag_id="build_dataset_snapshot",
    description="Build immutable governed Parquet snapshots and confirm OpenLineage",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=True,
    max_active_runs=1,
    default_args={"owner": "industrial-ops-data", "retries": 2},
    tags=["m3", "governed-data", "spark", "openlineage"],
) as dag:
    build_dataset_snapshot = BashOperator(
        task_id="build_dataset_snapshot",
        bash_command=(
            "industrial-ops-data-pipeline build "
            "--tenant-id \"${IOAP_PIPELINE_TENANT_ID}\" "
            "--window-start \"{{ data_interval_start.isoformat() }}\" "
            "--window-end \"{{ data_interval_end.isoformat() }}\" "
            "--engine spark "
            "--code-version \"${IOAP_CODE_VERSION:-runtime}\""
        ),
        append_env=True,
    )
