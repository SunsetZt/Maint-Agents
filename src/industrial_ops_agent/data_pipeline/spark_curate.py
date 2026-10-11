"""PySpark execution of the shared governed-dataset transformation contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from industrial_ops_agent.data_pipeline.contracts import prepare_rows


def build_spark_snapshot(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    spark: Any | None = None,
) -> Path:
    """Validate, transform and write split-partitioned Parquet through Spark."""

    from pyspark.sql import SparkSession

    prepared = prepare_rows(rows)
    owns_session = spark is None
    session = spark or (
        SparkSession.builder.appName("industrial-ops-dataset-curation")
        .master(os.environ.get("IOAP_SPARK_MASTER_URL", "local[*]"))
        .config(
            "spark.driver.host",
            os.environ.get("IOAP_SPARK_DRIVER_HOST", "localhost"),
        )
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    try:
        frame = session.createDataFrame(prepared)
        frame.write.mode("errorifexists").partitionBy("split").parquet(str(output_path))
    finally:
        if owns_session:
            session.stop()
    return output_path
