"""Daily governed supervised RUL dataset curation with catch-up support."""

from __future__ import annotations

from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

with DAG(
    dag_id="build_rul_dataset_snapshot",
    description="Freeze observed failure lead-times as governed RUL Transformer snapshots",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=True,
    max_active_runs=1,
    default_args={"owner": "industrial-ops-ml", "retries": 2},
    tags=["m7", "predictive-maintenance", "rul", "spark", "openlineage"],
) as dag:
    build_rul_dataset_snapshot = BashOperator(
        task_id="build_rul_dataset_snapshot",
        bash_command=(
            "industrial-ops-rul-dataset build "
            '--tenant-id "${IOAP_PIPELINE_TENANT_ID}" '
            '--window-start "{{ data_interval_start.isoformat() }}" '
            '--window-end "{{ data_interval_end.isoformat() }}" '
            "--engine spark "
            '--code-version "${IOAP_CODE_VERSION:-runtime}"'
        ),
    )
