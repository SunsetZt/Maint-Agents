"""Daily KB-003 expiry discovery and durable deletion dispatch."""

from __future__ import annotations

from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

with DAG(
    dag_id="retention_and_deletion",
    description="Revoke expired knowledge and dispatch full deletion propagation",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "industrial-ops-data-governance", "retries": 2},
    tags=["kb-003", "governance", "retention", "temporal"],
) as dag:
    sweep_expired_knowledge = BashOperator(
        task_id="sweep_expired_knowledge",
        bash_command=(
            "industrial-ops-retention "
            "--limit \"${IOAP_RETENTION_SWEEP_LIMIT:-100}\" "
            "--reason \"scheduled tenant retention policy expiry sweep\""
        ),
        append_env=True,
    )
