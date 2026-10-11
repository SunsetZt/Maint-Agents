"""Package an approved local STAGING Adapter; never reuse old DPO acceptance."""
from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys

def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def rows(root):
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("local_adapter_symlink")
        if path.is_file():
            h = sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    h.update(chunk)
            result.append({"path":path.relative_to(root).as_posix(), "size_bytes":path.stat().st_size, "sha256":h.hexdigest()})
    return result

def build_local_plan(root, release, values):
    # The owner is also a standalone stdlib entrypoint. Packaging must not import
    # the application package (and its server dependencies) in the host Python.
    sys.path.insert(0, str(root / "src/industrial_ops_agent/model_gateway"))
    from owned_runtime import _verify_adapter_identity

    manifest = release["manifest"]
    local = manifest.get("local_staging_adapter")
    if (local is not True or release["target_environment"] != "STAGING" or manifest.get("prompt_evaluation_only")
            or release.get("approval", {}).get("status") != "APPROVED"
            or release["approval"]["manifest_hash"] != release["manifest_hash"]
            or digest(manifest) != release["manifest_hash"]):
        raise ValueError("local_release_not_approved")
    # Use the prepared target-tenant candidate, never a source-tenant record ID.
    output = root / "artifacts/staging-smoke-business-rollout"
    spec = json.loads((output / "candidate-runtime.json").read_text())
    preview = json.loads((output / "preview-release.json").read_text())["data"]
    ignored = {"prompt_bundle", "prompt_bundle_id", "rollback", "prompt_evaluation_only"}
    context = lambda item: {k:v for k,v in item.items() if k not in ignored}
    if context(manifest) != context(preview["manifest"]):
        raise ValueError("local_candidate_model_context_changed")
    folder = Path(spec["folder"])
    if not folder.resolve().is_relative_to(output.resolve()) or folder.is_symlink():
        raise ValueError("local_candidate_folder_invalid")
    package = json.loads((folder / "ioap-package-manifest.json").read_text())
    if (_verify_adapter_identity(package, folder) != manifest["adapter"]["content_hash"]
            or package["base_model_id"] != manifest["base_model"]["id"]):
        raise ValueError("local_candidate_adapter_changed")
    for name, expected in spec["source_files"].items():
        if sha256((root / "src/industrial_ops_agent/model_gateway" / name).read_bytes()).hexdigest() != expected:
            raise ValueError("local_candidate_serving_source_changed")
    base_volume = spec["base_volume"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,127}", base_volume):
        raise ValueError("local_base_volume_invalid")
    # CPU-only reader verifies copied base bytes; it never starts a model.
    inspection = """import hashlib,json
from pathlib import Path
root=Path('/source')
manifest=json.loads((root/'ioap-package-manifest.json').read_text())
rows=[]
for path in sorted((root/'base').rglob('*')):
 if path.is_symlink(): raise SystemExit(1)
 if path.is_file():
  h=hashlib.sha256()
  with path.open('rb') as f:
   while chunk:=f.read(1024*1024): h.update(chunk)
  rows.append({'path':path.relative_to(root/'base').as_posix(),'size_bytes':path.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps({'manifest':manifest,'rows':rows}))"""
    run = subprocess.run([os.environ.get("M1_CONTAINER_CLI","docker"), "run", "--rm", "--network", "none",
        "--pull", "never", "--cpus", "1", "--memory", "512m", "--read-only",
        "--mount", f"type=volume,src={base_volume},dst=/source,readonly",
        "--entrypoint", "python3", spec["image"], "-c", inspection], text=True, capture_output=True)
    if run.returncode:
        raise ValueError("local_base_read_failed")
    retained = json.loads(run.stdout)
    if ({row["path"]:row["sha256"] for row in retained["rows"]} != package["base_files"]
            or retained["manifest"]["base_revision"] != package["base_revision"]):
        raise ValueError("local_base_bytes_changed")
    vlm = manifest["specialized_components"]["vlm"]["base_checkpoint"]
    source_path = root / "infra/model-runtime/smolvlm-500m.json"
    source = json.loads(source_path.read_text())
    if (sha256(source_path.read_bytes()).hexdigest() != vlm["source_manifest_sha256"]
            or source["model_id"] != vlm["base_model_id"] or source["revision"] != vlm["base_revision"]):
        raise ValueError("local_vlm_source_changed")
    adapter_rows = rows(folder / "adapter")
    package_hash = digest({"release_id":release["release_id"], "manifest_hash":release["manifest_hash"],
        "base":retained["rows"], "adapter":adapter_rows, "adapter_bundle":manifest["adapter"]["content_hash"],
        "serving_source_files":spec["source_files"]})
    return {
        "schema_version":2, "vlm_artifact_kind":"BASE_CHECKPOINT", "vlm_source_manifest_sha256":vlm["source_manifest_sha256"],
        "diagnosis_source_image":spec["image"], "diagnosis_source_digest":spec["image"].rsplit("@",1)[1],
        "diagnosis_source_volume":base_volume, "diagnosis_source_package_digest":retained["manifest"]["package_digest"],
        "diagnosis_adapter_source":str(folder / "adapter"),
        "diagnosis_adapter_bundle":str(folder / "ioap-adapter-bundle.tar.gz"),
        "diagnosis_adapter_content_hash":manifest["adapter"]["content_hash"],
        "diagnosis_serving_source_files":spec["source_files"],
        "vlm_cache_source":str(Path(values.get("IOAP_VLM_BASE_CACHE_PATH",
            str(Path.home() / ".cache/industrial-ops/models/smolvlm-500m" / source["revision"])))),
        "vlm_revision":source["revision"],
        "diagnosis_volume":"industrial-ops-project-staging-diagnosis-" + package_hash[:24],
        "vlm_volume":values["IOAP_VLM_MODEL_VOLUME"],
        "diagnosis_base_model_id":package["base_model_id"], "vlm_base_model_id":source["model_id"],
        "diagnosis_model_id":release["release_id"], "vlm_model_id":vlm["runtime_model_id"],
        "diagnosis_package_digest":package_hash, "vlm_package_digest":vlm["package_digest"],
        "diagnosis_base_files":retained["rows"], "diagnosis_adapter_files":adapter_rows,
        "vlm_base_files":source["files"], "vlm_adapter_files":[],
        "diagnosis_base_revision":package["base_revision"], "vlm_base_revision":source["revision"],
        "gpu_probe_image":values["IOAP_GPU_PROBE_IMAGE"],
    }
