#!/usr/bin/env bash
set -euo pipefail

cache_dir=.docker-cache/m3
jre_archive=$cache_dir/OpenJDK17U-jre_x64_linux_hotspot_17.0.20_8.tar.gz
jre_sha256=ef491a51a46ef90cc47fbc4abb219fde32483ff91be5ec66ddc896df43524b27
pyspark_wheel=$cache_dir/pyspark-4.0.1-py2.py3-none-any.whl
wheelhouse=$cache_dir/wheels
wheelhouse_stamp=$wheelhouse/.complete-v2

mkdir -p "$cache_dir"
if [[ ! -f "$jre_archive" ]] || ! echo "$jre_sha256  $jre_archive" | sha256sum --check --status; then
  curl --fail --location --silent --show-error \
    --output "$jre_archive" \
    'https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.20%2B8/OpenJDK17U-jre_x64_linux_hotspot_17.0.20_8.tar.gz'
fi
echo "$jre_sha256  $jre_archive" | sha256sum --check --status

if [[ ! -f "$pyspark_wheel" ]]; then
  site_packages=$(
    .venv/bin/python - <<'PY'
import importlib.metadata
import pathlib
import pyspark

if importlib.metadata.version("pyspark") != "4.0.1":
    raise SystemExit("the locked local environment must contain pyspark 4.0.1")
print(pathlib.Path(pyspark.__file__).resolve().parent.parent)
PY
  )
  (
    cd "$site_packages"
    zip -q -0 -r "$OLDPWD/$pyspark_wheel" \
      pyspark \
      pyspark-4.0.1.data \
      pyspark-4.0.1.dist-info
  )
fi

[[ -s "$pyspark_wheel" ]] || {
  echo "Failed to prepare the locked PySpark wheel." >&2
  exit 1
}

if [[ ! -f "$wheelhouse_stamp" ]]; then
  mkdir -p "$wheelhouse"
  .venv/bin/python -m pip download \
    --only-binary=:all: \
    --dest "$wheelhouse" \
    --find-links "$cache_dir" \
    '.[data-pipeline]'
  .venv/bin/python -m pip download \
    --only-binary=:all: \
    --dest "$wheelhouse" \
    --find-links "$wheelhouse" \
    'apache-airflow==3.3.0' \
    'apache-airflow-providers-standard>=1.10,<2' \
    'hatchling>=1.27,<2'
  touch "$wheelhouse_stamp"
fi

compgen -G "$wheelhouse/apache_airflow-3.3.0-*.whl" >/dev/null || {
  echo "Failed to prepare the Airflow wheelhouse." >&2
  exit 1
}
compgen -G "$wheelhouse/pyarrow-*.whl" >/dev/null || {
  echo "Failed to prepare the project wheelhouse." >&2
  exit 1
}
