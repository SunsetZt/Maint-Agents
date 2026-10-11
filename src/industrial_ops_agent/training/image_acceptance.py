"""Executable, content-bound acceptance contract for the NVIDIA training image."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from packaging.specifiers import SpecifierSet

GPU_IMAGE_ACCEPTANCE_SCHEMA = "industrial-gpu-image-acceptance-v1"
CUDA_VERSION_PREFIX = "13.0"
PACKAGE_CONTRACT = {
    "accelerate": ">=1.10,<2",
    "bitsandbytes": ">=0.48,<1",
    "datasets": ">=4,<5",
    "deepspeed": ">=0.18,<1",
    "diffusers": ">=0.39,<0.40",
    "peft": ">=0.18,<1",
    "Pillow": ">=11,<13",
    "safetensors": ">=0.6,<1",
    "sentence-transformers": ">=5,<6",
    "soundfile": ">=0.13,<1",
    "torch": "==2.10.0",
    "transformers": ">=5,<6",
    "trl": ">=0.28,<1",
}


class GpuImageAcceptanceError(RuntimeError):
    """The image or attached GPU does not satisfy the executable contract."""


@dataclass(frozen=True, slots=True)
class AcceptanceCheck:
    name: str
    status: Literal["PASSED", "FAILED"]
    reason_code: str | None
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason_code": self.reason_code,
            "details": self.details,
        }


class _CheckFailure(RuntimeError):
    def __init__(self, reason_code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.details = details or {}


def run_gpu_image_acceptance(
    *,
    image_digest: str,
    git_commit: str,
    precision: Literal["bfloat16", "float16"] = "bfloat16",
    minimum_gpu_memory_bytes: int = 0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run imports plus real CUDA and NF4 kernels, returning a sealed JSON report."""

    checks = [
        _capture("runtime_identity", lambda: _runtime_identity(image_digest, git_commit)),
        _capture("dependency_versions", _dependency_versions),
        _capture("training_import_graph", _training_import_graph),
        _capture(
            "gpu_runtime",
            lambda: _gpu_runtime(precision, minimum_gpu_memory_bytes),
        ),
        _capture("cuda_tensor_kernel", lambda: _cuda_tensor_kernel(precision)),
        _capture("bitsandbytes_nf4_kernel", lambda: _bitsandbytes_nf4_kernel(precision)),
        _capture("nvidia_driver_visibility", _nvidia_driver_visibility),
    ]
    dependencies: dict[str, Any] = next(
        (item.details.get("resolved", {}) for item in checks if item.name == "dependency_versions"),
        {},
    )
    gpu: dict[str, Any] = next(
        (item.details for item in checks if item.name == "gpu_runtime"),
        {},
    )
    payload: dict[str, Any] = {
        "schema_version": GPU_IMAGE_ACCEPTANCE_SCHEMA,
        "status": "PASSED" if all(item.status == "PASSED" for item in checks) else "FAILED",
        "generated_at": (now or datetime.now(UTC)).isoformat(),
        "image_digest": image_digest,
        "git_commit": git_commit,
        "precision": precision,
        "minimum_gpu_memory_bytes": minimum_gpu_memory_bytes,
        "dependencies": dependencies,
        "gpu": gpu,
        "checks": [item.as_dict() for item in checks],
    }
    payload["report_digest"] = gpu_acceptance_report_digest(payload)
    return payload


def gpu_acceptance_report_digest(report: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in report.items() if key != "report_digest"}
    encoded = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return f"sha256:{sha256(encoded).hexdigest()}"


def verify_gpu_acceptance_report(
    report: Any,
    *,
    expected_image_digest: str,
    expected_git_commit: str,
    expected_visible_gpu_count: int | None = None,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise GpuImageAcceptanceError("gpu_image_acceptance_report_is_missing")
    if report.get("schema_version") != GPU_IMAGE_ACCEPTANCE_SCHEMA:
        raise GpuImageAcceptanceError("gpu_image_acceptance_schema_is_not_supported")
    if report.get("report_digest") != gpu_acceptance_report_digest(report):
        raise GpuImageAcceptanceError("gpu_image_acceptance_report_digest_mismatch")
    if report.get("image_digest") != expected_image_digest:
        raise GpuImageAcceptanceError("gpu_image_acceptance_image_digest_mismatch")
    if report.get("git_commit") != expected_git_commit:
        raise GpuImageAcceptanceError("gpu_image_acceptance_git_commit_mismatch")
    checks = report.get("checks")
    if (
        report.get("status") != "PASSED"
        or not isinstance(checks, list)
        or not checks
        or any(not isinstance(item, dict) or item.get("status") != "PASSED" for item in checks)
    ):
        raise GpuImageAcceptanceError("gpu_image_acceptance_did_not_pass")
    gpu = report.get("gpu")
    if expected_visible_gpu_count is not None and (
        not isinstance(gpu, dict)
        or gpu.get("visible_gpu_count") != expected_visible_gpu_count
    ):
        raise GpuImageAcceptanceError("gpu_image_acceptance_visible_gpu_count_mismatch")
    return report


def gpu_acceptance_artifact_metadata(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "schema_version",
            "status",
            "generated_at",
            "image_digest",
            "git_commit",
            "precision",
            "dependencies",
            "gpu",
            "report_digest",
        )
    }


def write_gpu_acceptance_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_gpu_acceptance_report(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _capture(name: str, operation: Callable[[], dict[str, Any]]) -> AcceptanceCheck:
    try:
        return AcceptanceCheck(name, "PASSED", None, operation())
    except _CheckFailure as exc:
        return AcceptanceCheck(name, "FAILED", exc.reason_code, exc.details)
    except Exception:
        return AcceptanceCheck(name, "FAILED", f"{name}_unexpected_failure", {})


def _runtime_identity(image_digest: str, git_commit: str) -> dict[str, Any]:
    if not _sha256_digest(image_digest):
        raise _CheckFailure("container_digest_must_be_sha256")
    if not _git_commit(git_commit):
        raise _CheckFailure("git_commit_must_be_full_sha")
    baked_git_commit = os.getenv("IOAP_IMAGE_GIT_COMMIT", "")
    if baked_git_commit != git_commit:
        raise _CheckFailure(
            "image_git_commit_mismatch",
            {"baked_git_commit": baked_git_commit, "requested_git_commit": git_commit},
        )
    return {
        "image_digest": image_digest,
        "git_commit": git_commit,
        "baked_git_commit": baked_git_commit,
    }


def _dependency_versions() -> dict[str, Any]:
    resolved: dict[str, str] = {}
    failures: dict[str, str] = {}
    for distribution, requirement in PACKAGE_CONTRACT.items():
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            failures[distribution] = "missing"
            continue
        resolved[distribution] = installed
        if installed not in SpecifierSet(requirement):
            failures[distribution] = f"{installed} not in {requirement}"
    if failures:
        raise _CheckFailure(
            "training_dependency_contract_failed",
            {"resolved": resolved, "failures": failures},
        )
    return {"resolved": resolved, "contract": PACKAGE_CONTRACT}


def _training_import_graph() -> dict[str, Any]:
    required_symbols = {
        "datasets": ("Dataset",),
        "deepspeed": (),
        "diffusers": ("AutoPipelineForInpainting",),
        "peft": ("LoraConfig", "get_peft_model", "prepare_model_for_kbit_training"),
        "sentence_transformers": (
            "SentenceTransformer",
            "SentenceTransformerTrainer",
            "SentenceTransformerTrainingArguments",
        ),
        "transformers": (
            "AutoModelForCausalLM",
            "AutoModelForImageTextToText",
            "AutoModelForSequenceClassification",
            "AutoModelForSpeechSeq2Seq",
            "AutoProcessor",
            "AutoTokenizer",
            "BitsAndBytesConfig",
            "Seq2SeqTrainer",
        ),
        "trl": ("DPOConfig", "DPOTrainer", "GRPOConfig", "GRPOTrainer", "SFTTrainer"),
        "trl.experimental.ppo": ("PPOConfig", "PPOTrainer"),
    }
    loaded: list[str] = []
    for module_name, symbols in required_symbols.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise _CheckFailure(
                "training_dependency_import_failed", {"module": module_name}
            ) from exc
        for symbol in symbols:
            if getattr(module, symbol, None) is None:
                raise _CheckFailure(
                    "training_dependency_symbol_is_missing",
                    {"module": module_name, "symbol": symbol},
                )
            loaded.append(f"{module_name}.{symbol}")
    return {"symbols": loaded}


def _gpu_runtime(
    precision: Literal["bfloat16", "float16"], minimum_gpu_memory_bytes: int
) -> dict[str, Any]:
    torch = _torch()
    if not torch.cuda.is_available():
        raise _CheckFailure("cuda_gpu_is_not_available")
    visible_gpu_count = int(torch.cuda.device_count())
    if visible_gpu_count < 1:
        raise _CheckFailure(
            "at_least_one_visible_gpu_is_required",
            {"visible_gpu_count": visible_gpu_count},
        )
    if not str(torch.version.cuda or "").startswith(CUDA_VERSION_PREFIX):
        raise _CheckFailure(
            "cuda_runtime_version_mismatch",
            {"expected_prefix": CUDA_VERSION_PREFIX, "actual": torch.version.cuda},
        )
    if not torch.backends.cudnn.is_available() or not torch.backends.cudnn.version():
        raise _CheckFailure("cudnn_is_not_available")
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise _CheckFailure("gpu_does_not_support_bfloat16")
    device = torch.cuda.get_device_properties(0)
    total_memory = int(device.total_memory)
    if minimum_gpu_memory_bytes > 0 and total_memory < minimum_gpu_memory_bytes:
        raise _CheckFailure(
            "gpu_memory_is_below_acceptance_minimum",
            {"actual_bytes": total_memory, "minimum_bytes": minimum_gpu_memory_bytes},
        )
    return {
        "name": str(device.name),
        "total_memory_bytes": total_memory,
        "compute_capability": f"{device.major}.{device.minor}",
        "visible_gpu_count": visible_gpu_count,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version()),
        "bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def _cuda_tensor_kernel(precision: Literal["bfloat16", "float16"]) -> dict[str, Any]:
    torch = _torch()
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    verified_devices: list[int] = []
    try:
        for device_index in range(int(torch.cuda.device_count())):
            device = f"cuda:{device_index}"
            left = torch.randn((64, 64), device=device, dtype=dtype)
            right = torch.randn((64, 64), device=device, dtype=dtype)
            output = left @ right
            torch.cuda.synchronize(device_index)
            if not bool(torch.isfinite(output).all().item()):
                raise _CheckFailure(
                    "cuda_tensor_kernel_returned_non_finite_values",
                    {"device_index": device_index},
                )
            verified_devices.append(device_index)
    except Exception as exc:
        if isinstance(exc, _CheckFailure):
            raise
        raise _CheckFailure("cuda_tensor_kernel_failed") from exc
    return {
        "operation": "matmul",
        "shape": [64, 64],
        "dtype": str(dtype),
        "verified_device_indices": verified_devices,
    }


def _bitsandbytes_nf4_kernel(
    precision: Literal["bfloat16", "float16"],
) -> dict[str, Any]:
    torch = _torch()
    verified_devices: list[int] = []
    try:
        bitsandbytes = importlib.import_module("bitsandbytes")
        dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
        for device_index in range(int(torch.cuda.device_count())):
            device = f"cuda:{device_index}"
            layer = bitsandbytes.nn.Linear4bit(
                64,
                32,
                bias=False,
                compute_dtype=dtype,
                compress_statistics=True,
                quant_type="nf4",
            ).to(device)
            inputs = torch.randn((2, 64), device=device, dtype=dtype)
            outputs = layer(inputs)
            torch.cuda.synchronize(device_index)
            if not bool(torch.isfinite(outputs).all().item()) or tuple(outputs.shape) != (2, 32):
                raise _CheckFailure(
                    "bitsandbytes_nf4_kernel_output_is_invalid",
                    {"device_index": device_index},
                )
            verified_devices.append(device_index)
    except Exception as exc:
        if isinstance(exc, _CheckFailure):
            raise
        raise _CheckFailure("bitsandbytes_nf4_kernel_failed") from exc
    return {
        "quant_type": "nf4",
        "output_shape": [2, 32],
        "dtype": str(dtype),
        "verified_device_indices": verified_devices,
    }


def _nvidia_driver_visibility() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _CheckFailure("nvidia_smi_query_failed") from exc
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    expected_count = int(_torch().cuda.device_count())
    if len(rows) != expected_count:
        raise _CheckFailure(
            "nvidia_smi_gpu_count_mismatch",
            {"nvidia_smi_count": len(rows), "torch_visible_gpu_count": expected_count},
        )
    devices: list[dict[str, str]] = []
    for row in rows:
        values = [value.strip() for value in row.split(",")]
        if len(values) != 4 or any(not value for value in values):
            raise _CheckFailure("nvidia_smi_output_is_invalid")
        devices.append(
            {
                "driver_version": values[0],
                "gpu_name": values[1],
                "memory_mib": values[2],
                "compute_capability": values[3],
            }
        )
    return {
        "visible_gpu_count": expected_count,
        "devices": devices,
    }


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise _CheckFailure("torch_is_not_installed") from exc


def _sha256_digest(value: str) -> bool:
    return (
        value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _git_commit(value: str) -> bool:
    return len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )
