#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$ROOT"

runtime=${M1_CONTAINER_CLI:-docker}
runtime_env=${M1_RUNTIME_ENV_FILE:-.env.m1.local}
runtime_binding=${IOAP_REAL_MODEL_BINDING_FILE:-.env.real-model.local}
accepted_pump_image=artifacts/m7-vlm-checkpoint-continuation-rescored-lab/object-store/tenant/tenant-simulation-lab/datasets/snapshots/dataset-sim-m7-eval-a5d68beddfc298953a61d8a2/published/media/test/m7-sim-evaluation-003-pump_seal_leak.png
accepted_pump_sha256=3e87a180daaed2f7cd268f0ec14e5675276d69fc01630c4c0266fd464f499e0a
live_owned_workspace=

runtime_compose() {
  local -a env_files=(--env-file "$runtime_env")
  [[ ! -f "$runtime_binding" || -L "$runtime_binding" ]] || env_files+=(--env-file "$runtime_binding")
  M1_RUNTIME_ENV_FILE="$runtime_env" "$runtime" compose "${env_files[@]}" \
    -f compose.lite.yaml -f compose.real-model.yaml "$@"
}

env_value() {
  python3 - "$1" "$runtime_env" "$runtime_binding" <<'PY'
import os, sys
from pathlib import Path
key = sys.argv[1]
value = os.environ.get(key, "")
for raw in sys.argv[2:]:
    path = Path(raw)
    if not path.is_file() or path.is_symlink():
        continue
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(key + "="):
            value = line.split("=", 1)[1].strip()
print(value)
PY
}

create_live_workspace() {
  local workspace
  workspace=$(mktemp -d /tmp/ioap-project-staging-live-gpu.XXXXXX)
  chmod 700 "$workspace"
  printf '%s\n' "$workspace"
}

cleanup_live_workspace() {
  local workspace=${live_owned_workspace:-} resolved
  [[ -n "$workspace" ]] || return 0
  resolved=$(realpath -- "$workspace" 2>/dev/null || true)
  case "$resolved" in /tmp/ioap-project-staging-live-gpu.*) ;; *) return 3 ;; esac
  [[ -d "$resolved" && ! -L "$resolved" && "$(stat -c '%u' "$resolved")" == "$(id -u)" ]] || return 3
  rm -rf -- "$resolved"
  live_owned_workspace=
}

prepare_live_requests() {
  local workspace=$1 diagnosis_id=$2 vlm_id=$3
  [[ "$(sha256sum -- "$accepted_pump_image" | cut -d' ' -f1)" == "$accepted_pump_sha256" ]]
  python3 - "$workspace" "$accepted_pump_image" "$diagnosis_id" "$vlm_id" <<'PY'
import base64, json, os, sys
from pathlib import Path

workspace, image = Path(sys.argv[1]), Path(sys.argv[2]); os.chmod(workspace, 0o711)
diagnosis_id, vlm_id = sys.argv[3:5]
common = {"type": "object", "additionalProperties": False}
diagnosis_schema = {
    **common,
    "properties": {
        "fault": {"type": "string"},
        "recommended_action": {"type": "string"},
        "risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
    },
    "required": ["fault", "recommended_action", "risk"],
}
vlm_schema = {
    **common,
    "properties": {
        "leak_detected": {"type": "boolean"},
        "component": {"type": "string"},
        "observation": {"type": "string"},
    },
    "required": ["leak_detected", "component", "observation"],
}
diagnosis = {
    "model": diagnosis_id,
    "messages": [
        {"role": "system", "content": "Return only the requested maintenance JSON."},
        {"role": "user", "content": "A PUMP-X100 seal leaks 8-10 drops per minute and discharge pressure varies from 0.52 to 0.58 MPa. Diagnose the fault and recommend a safe next action."},
    ],
    "temperature": 0,
    "max_tokens": 160,
    "response_format": {"type": "json_schema", "json_schema": {"name": "project_staging_diagnosis", "strict": True, "schema": diagnosis_schema}},
}
data_url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode("ascii")
vlm = {
    "model": vlm_id,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Inspect this accepted synthetic pump image for visible leakage. Return only the requested JSON."},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}],
    "temperature": 0,
    "max_tokens": 128,
    "response_format": {"type": "json_schema", "json_schema": {"name": "project_staging_vlm", "strict": True, "schema": vlm_schema}},
}
for name, payload in (("diagnosis-request.json", diagnosis), ("vlm-request.json", vlm)):
    path = workspace / name
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.chmod(path, 0o444)
PY
}

run_router_requests() {
  local workspace=$1 round=$2
  runtime_compose run --rm --no-deps -T --entrypoint python3 \
    --volume "$workspace:/verify:ro" model-serving-gateway \
    - /verify/diagnosis-request.json /verify/vlm-request.json \
    >"$workspace/$round-summary.json" <<'PY'
import json, stat, sys, urllib.error, urllib.request
from pathlib import Path

def fail(reason):
    raise SystemExit(reason)

def get_json(url, payload=None, credential=None):
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
        headers["Authorization"] = "Bearer " + credential
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(urllib.request.Request(url, data=data, headers=headers), timeout=300) as response:
            if response.status != 200:
                fail("live_model_http_status_invalid")
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError):
        fail("live_model_request_failed")

def schema_valid(value, schema):
    kind = schema.get("type")
    if kind == "object":
        properties = schema.get("properties", {})
        if not isinstance(value, dict) or any(key not in properties for key in value):
            return False
        if any(key not in value for key in schema.get("required", [])):
            return False
        return all(schema_valid(item, properties[key]) for key, item in value.items())
    checks = {
        "string": lambda item: isinstance(item, str) and bool(item.strip()),
        "boolean": lambda item: isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
    }
    if kind not in checks or not checks[kind](value):
        return False
    return "enum" not in schema or value in schema["enum"]

key_path = Path("/run/secrets/model-gateway-api-key")
try:
    mode = key_path.stat().st_mode
    credential = key_path.read_text(encoding="utf-8").strip()
except OSError:
    fail("router_credential_unavailable")
if not stat.S_ISREG(mode) or not credential:
    fail("router_credential_invalid")

requests = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]]
expected = [payload["model"] for payload in requests]
for upstream, model_id in zip(("diagnosis-model", "vlm-model"), expected, strict=True):
    listing = get_json(f"http://{upstream}:8000/v1/models")
    if model_id not in {row.get("id") for row in listing.get("data", []) if isinstance(row, dict)}:
        fail("upstream_model_identity_missing")
ready = get_json("http://model-serving-gateway:8080/health/ready")
if ready.get("status") != "READY" or set(ready.get("models", {}).values()) != set(expected):
    fail("router_readiness_identity_mismatch")

summaries = []
for payload in requests:
    body = get_json("http://model-serving-gateway:8080/v1/chat/completions", payload, credential)
    try:
        choice = body["choices"][0]
        content = json.loads(choice["message"]["content"])
        usage = body["usage"]
        schema = payload["response_format"]["json_schema"]["schema"]
        token_total = int(usage["total_tokens"])
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        fail("live_model_response_invalid")
    if body.get("model") != payload["model"] or not schema_valid(content, schema):
        fail("live_model_schema_or_identity_mismatch")
    if min(token_total, prompt_tokens, completion_tokens) <= 0 or token_total < prompt_tokens + completion_tokens:
        fail("live_model_usage_invalid")
    summaries.append({"model": body["model"], "finish_reason": choice.get("finish_reason"), "usage": usage})
print(json.dumps({"status": "PASSED", "requests": summaries}, sort_keys=True))
PY
  python3 - "$workspace/$round-summary.json" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["status"] == "PASSED" and len(payload["requests"]) == 2
PY
}

snapshot_identity() {
  local workspace=$1 name=$2 diagnosis_id=$3 vlm_id=$4
  local inspect_file="$workspace/$name-inspect.json"
  local -a ids
  mapfile -t ids < <(runtime_compose ps -q diagnosis-model vlm-model model-serving-gateway)
  [[ ${#ids[@]} -eq 3 ]]
  "$runtime" inspect "${ids[@]}" >"$inspect_file"
  runtime_compose exec -T diagnosis-model cat /models/runtime/ioap-package-manifest.json >"$workspace/$name-diagnosis-manifest.json"
  runtime_compose exec -T vlm-model cat /models/runtime/ioap-package-manifest.json >"$workspace/$name-vlm-manifest.json"
  python3 - "$inspect_file" "$workspace/$name-diagnosis-manifest.json" "$workspace/$name-vlm-manifest.json" \
    "$workspace/$name-identity.json" "$diagnosis_id" "$vlm_id" <<'PY'
import json, os, sys
sys.path.insert(0, "src/industrial_ops_agent/model_gateway")
from owned_runtime import _model_arguments, RouterConfigurationError
docs=json.load(open(sys.argv[1], encoding="utf-8"))
manifests={name:json.load(open(path, encoding="utf-8")) for name,path in zip(("diagnosis-model","vlm-model"),sys.argv[2:4],strict=True)}
expected={
    "diagnosis-model":{"component":"diagnosis","model_id":sys.argv[5]},
    "vlm-model":{"component":"vlm","model_id":sys.argv[6]},
}
identity={}
for row in docs:
    service=row["Config"]["Labels"].get("com.docker.compose.service")
    if service not in {"diagnosis-model","vlm-model","model-serving-gateway"}:
        continue
    mounts=sorted({"type":m["Type"],"source":m["Name"] if m["Type"]=="volume" else m["Source"],"destination":m["Destination"],"rw":m["RW"]} for m in row["Mounts"] if m["Destination"]=="/models/runtime")
    identity[service]={"image":row["Image"],"entrypoint":row["Config"]["Entrypoint"],"command":row["Config"]["Cmd"],"model_mounts":mounts}
if set(identity) != {"diagnosis-model","vlm-model","model-serving-gateway"}:
    raise SystemExit("runtime_identity_incomplete")
def option_value(command, name):
    if not isinstance(command, list) or command.count(name) != 1:
        raise SystemExit("runtime_model_command_invalid")
    index=command.index(name)
    if index + 1 >= len(command) or not isinstance(command[index + 1], str):
        raise SystemExit("runtime_model_command_invalid")
    return command[index + 1]
for service, expected_identity in expected.items():
    model_id=expected_identity["model_id"]
    manifest=manifests[service]
    packaged_model_id=manifest.get("packaged_model_id", manifest.get("served_model_id"))
    if (
        manifest.get("classification") != "PROJECT_STAGING_REAL"
        or manifest.get("component") != expected_identity["component"]
        or packaged_model_id != model_id
        or not manifest.get("package_digest")
        or not manifest.get("base_model_id")
        or not manifest.get("base_revision")
        or not manifest.get("runtime_image")
    ):
        raise SystemExit("package_manifest_identity_invalid")
    command=identity[service]["command"]
    try:
        # Reuse the real owner's closed argument checks, not a second VLM parser.
        if service == "vlm-model":
            if manifest.get("schema_version") != 2:
                raise SystemExit("runtime_base_vlm_required")
            _model_arguments(command, model_id, manifest)
        else:
            if (manifest.get("schema_version") != 1 or manifest["base_model_id"] != "Qwen/Qwen3-0.6B"
                    or option_value(command, "--served-model-name") != manifest["base_model_id"]):
                raise SystemExit("package_manifest_identity_invalid")
            if option_value(command, "--lora-modules") != f"{model_id}=/models/runtime/adapter":
                raise SystemExit("runtime_route_identity_invalid")
            _model_arguments(command, model_id)
    except RouterConfigurationError:
        raise SystemExit("runtime_route_identity_invalid") from None
    if not manifest["runtime_image"].endswith("@" + identity[service]["image"]):
        raise SystemExit("runtime_image_identity_invalid")
    if identity[service]["model_mounts"] == [] or any(mount["rw"] for mount in identity[service]["model_mounts"]):
        raise SystemExit("model_mount_not_read_only")
identity["package_manifests"]=manifests
with open(sys.argv[4],"w",encoding="utf-8") as stream:
    json.dump(identity,stream,sort_keys=True,separators=(",",":")); stream.write("\n")
os.chmod(sys.argv[4],0o600)
PY
}

verify_secret_safe_logs() {
  local workspace=$1 since=$2
  runtime_compose logs --no-color --since "$since" diagnosis-model vlm-model model-serving-gateway >"$workspace/runtime.log"
  python3 - "$workspace/runtime.log" "$runtime_env" "$runtime_binding" "$workspace/diagnosis-request.json" "$workspace/vlm-request.json" <<'PY'
import json, re, sys
from pathlib import Path
log=Path(sys.argv[1]).read_bytes()
values={}
for raw in sys.argv[2:4]:
    path=Path(raw)
    if path.is_file() and not path.is_symlink():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key,value=line.split("=",1); values[key.strip()]=value.strip()
secret=values.get("IOAP_MODEL_GATEWAY_API_KEY","").encode()
if secret and secret in log: raise SystemExit("router_secret_exposed_in_logs")
if re.search(br"(?i)bearer\s+\S+|data:image/",log): raise SystemExit("credential_or_media_exposed_in_logs")
for raw in sys.argv[4:]:
    payload=json.load(open(raw,encoding="utf-8"))
    for message in payload["messages"]:
        content=message["content"]
        texts=[content] if isinstance(content,str) else [item.get("text","") for item in content if item.get("type")=="text"]
        if any(text.encode() in log for text in texts if text): raise SystemExit("prompt_exposed_in_logs")
PY
}

live_gpu() {
  local workspace diagnosis_id vlm_id target_environment log_since
  umask 077
  command -v "$runtime" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1
  [[ -f "$runtime_env" && ! -L "$runtime_env" && -f "$accepted_pump_image" && ! -L "$accepted_pump_image" ]]
  workspace=$(create_live_workspace)
  case "$workspace" in /tmp/ioap-project-staging-live-gpu.*) live_owned_workspace=$workspace; trap cleanup_live_workspace EXIT INT TERM HUP ;; esac
  target_environment=$(env_value IOAP_MODEL_GATEWAY_REQUIRED_ENVIRONMENT)
  [[ -z "$target_environment" || "${target_environment^^}" == STAGING ]] || { printf '%s\n' 'live GPU acceptance is staging-only' >&2; return 3; }
  diagnosis_id=$(env_value IOAP_DIAGNOSIS_MODEL_ID); vlm_id=$(env_value IOAP_VLM_MODEL_ID)
  diagnosis_id=${diagnosis_id:-project-staging-diagnosis-dpo-d1efc0c036cf09516ec9}
  vlm_id=${vlm_id:-project-staging-vlm-34d369c52a2caa67ed8e27f8}
  [[ "$diagnosis_id" != "$vlm_id" ]] || fail "live_model_identities_must_be_distinct"
  export IOAP_DIAGNOSIS_MODEL_ID="$diagnosis_id" IOAP_VLM_MODEL_ID="$vlm_id"
  export IOAP_MODEL_GATEWAY_REQUIRED_ENVIRONMENT=STAGING
  export IOAP_MODEL_GATEWAY_ALIAS=industrial-diagnosis-staging IOAP_VLM_MODEL_ALIAS=industrial-diagnosis-staging
  log_since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  scripts/real_model_business_loop.sh package
  runtime_compose up --detach --wait gpu-probe diagnosis-model vlm-model model-serving-gateway || { [[ "$("$runtime" inspect --format '{{.State.Status}}:{{.State.ExitCode}}' "$(runtime_compose ps --all --quiet gpu-probe)")" == exited:0 ]] && runtime_compose up --detach --wait --no-deps diagnosis-model vlm-model model-serving-gateway || return 1; }
  prepare_live_requests "$workspace" "$diagnosis_id" "$vlm_id"
  snapshot_identity "$workspace" before "$diagnosis_id" "$vlm_id"
  run_router_requests "$workspace" first
  runtime_compose restart diagnosis-model vlm-model
  runtime_compose up --detach --wait diagnosis-model vlm-model model-serving-gateway
  snapshot_identity "$workspace" after "$diagnosis_id" "$vlm_id"
  cmp --silent "$workspace/before-identity.json" "$workspace/after-identity.json"
  run_router_requests "$workspace" second
  verify_secret_safe_logs "$workspace" "$log_since"
  printf 'PROJECT_STAGING_GPU_RUNTIME_OK diagnosis=%s vlm=%s rounds=2\n' "$diagnosis_id" "$vlm_id"
  cleanup_live_workspace
  trap - EXIT INT TERM HUP
}

build_images() {
  python3 -B - "$runtime" <<'PY'
import hashlib, io, json, os, re, subprocess, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path

def build_local_images(cli):
    """Refresh only source layers on qualified local images; never pull or install."""
    reviewed_legacy_base = "sha256:2af2dfde967f428473668a8fdc517f11ead81e5ded818c672ceca2a427a0e4a6"
    def run(*args, **kwargs):
        return subprocess.run([cli, *args], check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              timeout=kwargs.pop("timeout", 30), **kwargs).stdout

    def inspect(reference, model, allow_reviewed_base=False):
        row, = json.loads(run("image", "inspect", reference))
        config = row.get("Config", {})
        labels = config.get("Labels") or {}
        name = "owned_runtime.py" if model else "router_runtime.py"
        title = "industrial-ops-model-runtime" if model else "industrial-ops-project-staging-model-router"
        legacy_entrypoint = (allow_reviewed_base and model and row.get("Id") == reviewed_legacy_base
                             and config.get("Entrypoint") == ["vllm", "serve"])
        if (not re.fullmatch(r"sha256:[0-9a-f]{64}", row.get("Id", ""))
            or config.get("Volumes")
            or (config.get("Entrypoint") != ["python3", "/opt/industrial-ops/" + name] and not legacy_entrypoint)
            or labels.get("org.opencontainers.image.title") != title
            or (not model and config.get("User") != "65532:65532")
            or (model and (labels.get("org.opencontainers.image.version") != "vllm-0.26.0"
                           or labels.get("ai.vllm.image.tag") != "vllm/vllm-openai:v0.26.0"))):
            raise RuntimeError("local_runtime_base_identity_invalid")
        return row

    local_hosts = {"unix:///var/run/docker.sock", "unix:///run/docker.sock"}
    endpoint = json.loads(run("context", "inspect", "--format", "{{json .Endpoints.docker.Host}}"))
    if (endpoint not in local_hosts or os.environ.get("DOCKER_HOST", endpoint) not in local_hosts
        or ["driver-type", "io.containerd.snapshotter.v1"] not in json.loads(run("info", "--format", "{{json .DriverStatus}}"))):
        raise RuntimeError("local_containerd_image_store_required")

    def valid_descriptor(desc, metadata=False):
        if (not isinstance(desc, dict) or not re.fullmatch(r"sha256:[0-9a-f]{64}", desc.get("digest", ""))
            or type(desc.get("size")) is not int or desc["size"] <= 0 or desc.get("urls")
            or (metadata and desc["size"] > 1024**2)):
            raise RuntimeError("local_image_descriptor_invalid")

    def metadata(desc, kind):
        valid_descriptor(desc, True)
        allowed = {"manifest": {"application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"},
                   "config": {"application/vnd.oci.image.config.v1+json", "application/vnd.docker.container.image.v1+json"}}
        if desc.get("mediaType") not in allowed[kind]:
            raise RuntimeError("local_image_metadata_type_invalid")
        # Read only content-addressed JSON, never dependency blobs or store files.
        raw = subprocess.run(["sudo", "-n", "ctr", "--address", "/run/containerd/containerd.sock",
                              "--namespace", "moby", "content", "get", desc["digest"]],
                             check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout
        if len(raw) != desc["size"] or "sha256:" + hashlib.sha256(raw).hexdigest() != desc["digest"]:
            raise RuntimeError("local_image_metadata_identity_mismatch")
        return json.loads(raw)

    def parent(row):
        manifest = metadata(row["Descriptor"], "manifest")
        config = metadata(manifest["config"], "config")
        layers, rootfs = manifest["layers"], config["rootfs"]
        if (manifest.get("schemaVersion") != 2 or not isinstance(layers, list) or not layers
            or config.get("os") != row.get("Os") or config.get("architecture") != row.get("Architecture")
            or config.get("os") != "linux" or config.get("architecture") != "amd64"
            or rootfs.get("type") != "layers" or rootfs.get("diff_ids") != row["RootFS"]["Layers"]
            or len(rootfs["diff_ids"]) != len(layers)
            or sum(not item.get("empty_layer", False) for item in config["history"]) != len(layers)
            or any(config["config"].get(key) != row["Config"].get(key)
                   for key in ("Entrypoint", "Labels", "User", "Env", "Cmd", "WorkingDir"))):
            raise RuntimeError("local_image_parent_inconsistent")
        for layer, diff_id in zip(layers, rootfs["diff_ids"]):
            valid_descriptor(layer)
            if (not re.fullmatch(r"sha256:[0-9a-f]{64}", diff_id)
                or layer.get("mediaType") not in {"application/vnd.oci.image.layer.v1.tar",
                    "application/vnd.oci.image.layer.v1.tar+gzip", "application/vnd.oci.image.layer.v1.tar+zstd",
                    "application/vnd.docker.image.rootfs.diff.tar.gzip"}):
                raise RuntimeError("local_image_layer_invalid")
        return manifest, config

    repositories = ["project-staging-model-router", "project-staging-model-runtime"]
    # Validate BOTH bases before any build/container. Missing local dependencies
    # require explicit provisioning, not an implicit multi-GB registry download.
    rows = [inspect(name + ":local", bool(index), True) for index, name in enumerate(repositories)]
    parents = [parent(row) for row in rows]  # BOTH JSON chains verified before any import.
    identities = [row["Id"] for row in rows]
    sources = {name: (Path("src/industrial_ops_agent/model_gateway") / name).read_bytes()
               for name in ("owned_runtime.py", "router_runtime.py")}
    expected = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    source_id = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()

    def tar_bytes(files):
        if sum(map(len, files.values())) > 3 * 1024**2:
            raise RuntimeError("local_source_archive_too_large")
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            for name, raw in files.items():
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode = len(raw), 0o644
                archive.addfile(entry, io.BytesIO(raw))
        if output.tell() > 4 * 1024**2:
            raise RuntimeError("local_source_archive_too_large")
        return output.getvalue()

    source_layer = tar_bytes({"opt/industrial-ops/" + name: raw for name, raw in sources.items()})

    def matches(identity):
        try:
            output = run("run", "--rm", "--network", "none", "--pull", "never", "--read-only",
                         "--cpus", "1", "--memory", "256m", "--memory-swap", "256m", "--pids-limit", "32",
                         "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
                         "--env", "NVIDIA_VISIBLE_DEVICES=void", "--entrypoint", "sha256sum", identity,
                         *("/opt/industrial-ops/" + name for name in sources)).decode().splitlines()
        except subprocess.CalledProcessError as error:
            if error.returncode == 1:  # sha256sum missing/unreadable source, not daemon failure
                return False
            raise
        return output == [expected[name] + "  /opt/industrial-ops/" + name for name in sources]

    for index, (repository, identity) in enumerate(zip(repositories, identities)):
        if identity != reviewed_legacy_base and matches(identity):
            continue
        # Reference immutable local layers; only code + JSON cross the Docker API.
        # No walking diff, dependency export, image-store write or build container.
        files = {"oci-layout": b'{"imageLayoutVersion":"1.0.0"}'}
        def blob(raw, media):
            digest = "sha256:" + hashlib.sha256(raw).hexdigest()
            files["blobs/sha256/" + digest.split(":")[1]] = raw
            return {"mediaType": "application/vnd.oci.image." + media, "digest": digest, "size": len(raw)}

        manifest, config = parents[index]
        layer = blob(source_layer, "layer.v1.tar")
        config["rootfs"]["diff_ids"].append(layer["digest"])
        config["created"] = datetime.now(timezone.utc).isoformat()
        config["history"].append({"created": config["created"], "created_by": "industrial-ops bounded source layer " + source_id})
        config["config"]["Entrypoint"] = ["python3", "/opt/industrial-ops/" + ("owned_runtime.py" if index else "router_runtime.py")]
        config["config"]["Labels"].pop("io.industrial-ops.source-image-build", None)
        config["config"]["Labels"].update({"io.industrial-ops.local-base": identity, "io.industrial-ops.source-sha256": source_id})
        manifest = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "config": blob(json.dumps(config).encode(), "config.v1+json"), "layers": manifest["layers"] + [layer]}
        candidate = blob(json.dumps(manifest).encode(), "manifest.v1+json")
        files["index.json"] = json.dumps({"schemaVersion": 2, "manifests": [candidate]}).encode()
        payload = tar_bytes(files)
        print(f"LOCAL_RUNTIME_SOURCE_LAYER_IMPORT {repository} bytes={len(payload)}", flush=True)
        run("load", "--quiet", input=payload, timeout=180)
        identities[index] = inspect(candidate["digest"], bool(index))["Id"]
        if identities[index] != candidate["digest"]:
            raise RuntimeError("imported_runtime_identity_mismatch")
        if not matches(identities[index]):
            raise RuntimeError("built_runtime_source_mismatch")

    # Publish local convenience tags only after both immutable results pass.
    # Existing releases/bindings continue to point to their previous image IDs.
    for repository, identity in zip(repositories, identities):
        run("tag", identity, repository + ":local")
        run("tag", identity, "industrial-ops/" + repository + ":local")
    print("PROJECT_STAGING_IMAGES_READY router=industrial-ops/" + repositories[0] + "@" + identities[0]
          + " runtime=industrial-ops/" + repositories[1] + "@" + identities[1])

if __name__ == "__main__":
    try:
        build_local_images(sys.argv[1])
    except subprocess.TimeoutExpired:
        raise SystemExit("local_runtime_image_build_timed_out_no_remote_fallback")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, subprocess.SubprocessError):
        raise SystemExit("local_runtime_image_build_failed_no_remote_fallback")
PY
}

case "${1:-}" in
  build-images)
    build_images
    ;;
  live-gpu)
    live_gpu
    ;;
  *)
    printf '%s\n' \
      'usage: project_staging_model_runtime_verify.sh build-images|live-gpu' >&2
    exit 2
    ;;
esac
