"""Private, standard-library collector for real_model_business_loop.sh inspect.

Metadata is evidence of presence, never permission to deploy or proof of inference.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time
from datetime import datetime, timezone

_MIB = 1024 * 1024
_ROOT = Path(__file__).resolve().parents[1]
_MODELS = {"diagnosis": ("Qwen/Qwen3-0.6B", "qwen3"),
           "vlm": ("Qwen/Qwen2-VL-2B-Instruct", "qwen2_vl")}
_SMOL_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
_SMOL_REVISION = "a7da5b986cb59b408707209984f360a5f4ad7e47"
_SMOL_SOURCE_SHA256 = "276d9a77513b54202bcb3db3753203ca7a1e9016bdf4ea6b243da324671ac9de"
_RECORDS = {
    "diagnosis": "artifacts/m7-dpo-post-training-lab/runs/dpo-d1efc0c036cf09516ec9/acceptance.json",
    "source": "artifacts/m7-vlm-checkpoint-continuation-rescored-lab/acceptance.json",
    "adoption": "artifacts/m7-vlm-enterprise-staging/acceptance.json",
}


class _Fault(Exception):
    """Only fixed internal reason codes cross the disclosure boundary."""


def _require(condition, code="METADATA_INVALID"):
    if not condition:
        raise _Fault(code)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, "Real-model inspection: INVALID_ARGUMENTS\n")


def _arguments(argv):
    parser = _Parser(prog="real_model_business_loop.sh inspect", allow_abbrev=False)
    for name in ("diagnosis", "vlm", "ocr", "evidence"):
        parser.add_argument(f"--{name}-root", required=True)
    args = parser.parse_args(argv)
    for name in ("diagnosis", "vlm", "ocr", "evidence"):
        raw = getattr(args, name + "_root")
        if (len(raw) > 4096 or not raw.startswith("/") or ".." in raw.split("/")
                or any(ord(c) < 32 for c in raw)):
            parser.error("path")
        setattr(args, name + "_root", Path(raw))
    return args


class _Inspector:
    def __init__(self):
        self.deadline = time.monotonic() + 30
        self.entries = 0
        self.read_bytes = 0
        self.reasons = []
        self.metadata = {}

    def check(self):
        _require(time.monotonic() < self.deadline, "DEADLINE_EXCEEDED")

    def attempt(self, label, operation):
        try:
            self.check()
            return operation()
        except (OSError, _Fault, ValueError, TypeError, KeyError, RecursionError) as error:
            if isinstance(error, _Fault):
                code = str(error)
                if code in ("INTERRUPTED", "DEADLINE_EXCEEDED"):
                    raise
            elif isinstance(error, PermissionError):
                code = "PERMISSION_DENIED"
            elif isinstance(error, FileNotFoundError):
                code = "MISSING"
            elif isinstance(error, OSError):
                code = "UNSAFE_PATH" if error.errno in (errno.ELOOP, errno.ENOTDIR) else "IO_UNAVAILABLE"
            else:
                code = "METADATA_INVALID"
            reason = label + "_" + code
            if reason not in self.reasons:
                self.reasons.append(reason)
            return None

    def parent(self, path):
        self.check()
        _require(path.is_absolute() and ".." not in path.parts and len(path.parts) <= 128, "UNSAFE_PATH")
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in path.parts[1:-1]:
                self.check()
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                os.close(fd)
                fd = child
            return fd
        except BaseException as error:
            os.close(fd)
            if isinstance(error, OSError) and error.errno in (errno.ELOOP, errno.ENOTDIR):
                raise _Fault("UNSAFE_PATH") from None
            raise

    def file_stat(self, path):
        self.check()
        _require(self.entries < 128, "ENTRY_LIMIT")
        self.entries += 1
        fd = self.parent(path)
        try:
            info = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
            _require(stat.S_ISREG(info.st_mode), "UNSAFE_FILE")
            return info
        finally:
            os.close(fd)

    def read_json(self, path):
        info = self.file_stat(path)
        _require(info.st_size <= _MIB and self.read_bytes + info.st_size <= 16 * _MIB, "METADATA_LIMIT")
        fd = self.parent(path)
        try:
            file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        finally:
            os.close(fd)
        with os.fdopen(file_fd, "rb") as stream:
            current = os.fstat(stream.fileno())
            _require(stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) ==
                     (info.st_dev, info.st_ino), "UNSAFE_FILE")
            data = stream.read(_MIB + 1)
        self.check()
        self.read_bytes += len(data)
        _require(len(data) <= _MIB and self.read_bytes <= 16 * _MIB, "METADATA_LIMIT")
        def pairs(items):
            _require(len(dict(items)) == len(items), "DUPLICATE_JSON_KEY")
            return dict(items)
        def invalid(value):
            raise _Fault("NONFINITE_JSON")
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
        stack, count = [(value, 0)], 0
        while stack:
            node, depth = stack.pop()
            count += 1
            _require(depth <= 16 and count <= 16384, "JSON_STRUCTURE_LIMIT")
            if isinstance(node, float):
                _require(math.isfinite(node), "NONFINITE_JSON")
            if isinstance(node, (dict, list)):
                _require(len(node) <= 4096, "JSON_STRUCTURE_LIMIT")
                stack.extend((item, depth + 1) for item in (node.values() if isinstance(node, dict) else node))
        _require(isinstance(value, dict))
        self.metadata[str(path)] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        return value

    def history(self, evidence, kind, package, root):
        def read_record(name):
            record = self.read_json(evidence / _RECORDS[name])
            claimed = record.get("evidence_chain_sha256")
            _require(claimed == _digest({k: v for k, v in record.items() if k != "evidence_chain_sha256"}),
                     "HISTORY_CHAIN_MISMATCH")
            _require(record.get("production_claim") is False)
            return record
        def compare(rows, name, actual):
            _require(isinstance(rows, list) and len(rows) <= 128)
            selected = [row for row in rows if isinstance(row, dict) and row.get("path") == name]
            _require(len(selected) == 1, "HISTORY_METADATA_MISMATCH")
            observed = self.metadata[str(actual)]
            _require(all(selected[0].get(key) == observed[key] for key in ("sha256", "size_bytes")),
                     "HISTORY_METADATA_MISMATCH")
        if kind == "diagnosis":
            record = read_record("diagnosis")
            _require(record.get("schema_version") == "enterprise-dpo-post-training-lab/v1")
            _require(record.get("classification") == "LOCAL_STAGING_PROJECT_AUTHORIZED")
            _require(record.get("status") == "DPO_ACTUAL_GPU_POST_TRAINING_PASSED")
            base = record["base_model"]
            _require(base["model_id"] == package["base_model_id"] and base["revision"] == package["base_revision"],
                     "HISTORY_BASE_MISMATCH")
            compare(base["snapshot_files"], "config.json", root / "base/config.json")
            compare(record["adapter"]["files"], "adapter_config.json", root / "adapter/adapter_config.json")
            classification = record["classification"]
        else:
            source, adoption = read_record("source"), read_record("adoption")
            _require(source.get("classification") == "SIMULATED_NON_PRODUCTION")
            _require(source.get("decision") == "SIMULATION_CONTINUATION_ELIGIBLE")
            _require(adoption.get("schema_version") == "enterprise-vlm-staging-adoption/v1")
            _require(adoption.get("classification") == "LOCAL_STAGING_PROJECT_AUTHORIZED")
            _require(adoption.get("status") == "ENTERPRISE_VLM_STAGING_DEFAULT_ADOPTED")
            candidate = adoption["candidate"]
            _require(candidate["model_id"] == source["model_id"] == package["base_model_id"] and
                     candidate["model_revision"] == source["model_revision"] == package["base_revision"],
                     "HISTORY_BASE_MISMATCH")
            _require(candidate["source_acceptance_sha256"] == self.metadata[str(evidence / _RECORDS["source"])]["sha256"]
                     and candidate["source_evidence_chain_sha256"] == source["evidence_chain_sha256"],
                     "HISTORY_SOURCE_MISMATCH")
            compare(source["evidence_files"], "adapters/lora-continuation/adapter_config.json",
                    root / "adapter/adapter_config.json")
            classification = source["classification"]
        return {"metadata_match": True, "source_classification": classification,
                "scope": "HISTORICAL_METADATA_ONLY", "weights_verified": False}

    def source_manifest(self, path):
        source = self.read_json(path)
        _require(self.metadata[str(path)]["sha256"] == _SMOL_SOURCE_SHA256,
                 "SOURCE_MANIFEST_CHANGED")
        return source

    def base_checkpoint(self, root, package):
        _require(type(package.get("schema_version")) is int and package["schema_version"] == 2
                 and package.get("component") == "vlm"
                 and package.get("classification") == "PROJECT_STAGING_REAL"
                 and package.get("artifact_kind") == "BASE_CHECKPOINT"
                 and package.get("base_model_id") == _SMOL_MODEL
                 and package.get("base_revision") == _SMOL_REVISION
                 and package.get("source_manifest_sha256") == _SMOL_SOURCE_SHA256,
                 "BASE_VLM_IDENTITY_MISMATCH")
        for field, pattern in (("package_digest", r"[a-f0-9]{64}"),
                               ("runtime_image", r"[A-Za-z0-9_./:-]{1,180}@sha256:[a-f0-9]{64}"),
                               ("packaged_model_id", r"[A-Za-z0-9][A-Za-z0-9._/+-]{0,511}")):
            _require(isinstance(package.get(field), str) and re.fullmatch(pattern, package[field]))
        _require(not os.path.lexists(root / "adapter"), "BASE_VLM_ADAPTER_FORBIDDEN")
        source = self.source_manifest(root / "ioap-source-manifest.json")
        expected = {row["path"]: row for row in source["files"]}
        directory = self.parent(root / "base/config.json")
        try:
            _require(set(os.listdir(directory)) == set(expected), "SOURCE_FILE_SET_CHANGED")
        finally:
            os.close(directory)
        rows = []
        for name, row in expected.items():
            info = self.file_stat(root / "base" / name)
            _require(info.st_size == row["size_bytes"], "SOURCE_FILE_SIZE_CHANGED")
            rows.append({"file": "base/" + name, "size_bytes": info.st_size, "content_verified": False})
        base = self.read_json(root / "base/config.json")
        _require(base.get("model_type") == source["model_type"]
                 and base.get("architectures") == [source["architecture"]], "MODEL_TYPE_MISMATCH")
        for name in ("preprocessor_config.json", "processor_config.json"):
            proc = self.read_json(root / "base" / name)
            _require(proc.get("processor_class") == source["processor_class"], "PROCESSOR_MISMATCH")
        for name in ("config.json", "preprocessor_config.json", "processor_config.json"):
            actual = self.metadata[str(root / "base" / name)]
            _require(all(actual[key] == expected[name][key] for key in ("size_bytes", "sha256")),
                     "SOURCE_METADATA_CHANGED")
        return {"base_model_id": _SMOL_MODEL, "base_revision": _SMOL_REVISION,
                "package_digest_declared": package["package_digest"], "image_declared": package["runtime_image"],
                "image_verified": False, "adapter": None, "adapter_metadata": None,
                "processor": {"class": source["processor_class"]}, "files": rows, "historical": None,
                "base_metadata": self.metadata[str(root / "base/config.json")],
                "source_integrity": {"metadata_match": True, "scope": "SOURCE_METADATA_ONLY",
                                     "manifest_sha256": _SMOL_SOURCE_SHA256, "weights_verified": False}}

    def model(self, root, kind, evidence):
        package = self.read_json(root / "ioap-package-manifest.json")
        if kind == "vlm" and package.get("schema_version") == 2:
            return self.base_checkpoint(root, package)
        mid, model_type = _MODELS[kind]
        _require(package.get("schema_version") == 1 and package.get("component") == kind)
        _require(package.get("classification") == "PROJECT_STAGING_REAL")
        _require(package.get("base_model_id") == mid, "MODEL_ID_MISMATCH")
        for field, pattern in (("base_revision", r"[a-f0-9]{40}"), ("package_digest", r"[a-f0-9]{64}"),
                               ("runtime_image", r"[A-Za-z0-9_./:-]{1,180}@sha256:[a-f0-9]{64}")):
            _require(isinstance(package.get(field), str) and re.fullmatch(pattern, package[field]))
        base = self.read_json(root / "base/config.json")
        _require(base.get("model_type") == model_type, "MODEL_TYPE_MISMATCH")
        adapter = self.read_json(root / "adapter/adapter_config.json")
        _require(adapter.get("peft_type") == "LORA")
        reference = adapter.get("base_model_name_or_path")
        _require(reference in (mid, "/models/base") and (kind == "diagnosis" or reference == mid),
                 "ADAPTER_BASE_MISMATCH")
        _require(adapter.get("revision") in (None, package["base_revision"]), "ADAPTER_REVISION_MISMATCH")
        _require(type(adapter.get("r")) is int and 0 < adapter["r"] <= 1024)
        _require(type(adapter.get("lora_alpha")) in (int, float) and 0 < adapter["lora_alpha"] <= 65536)
        files = ["base/tokenizer.json", "adapter/adapter_model.safetensors"]
        try:
            self.file_stat(root / "base/model.safetensors.index.json")
        except FileNotFoundError:
            files.append("base/model.safetensors")
        else:
            index = self.read_json(root / "base/model.safetensors.index.json")
            mapping = index["weight_map"]
            _require(isinstance(mapping, dict) and mapping)
            shards = set(mapping.values())
            _require(len(shards) <= 32 and all(isinstance(s, str) and
                     re.fullmatch(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", s) for s in shards))
            files.extend("base/" + s for s in sorted(shards))
        rows = []
        for file in files:
            info = self.file_stat(root / file)
            _require(info.st_size > 0, "EMPTY_ARTIFACT")
            rows.append({"file": file, "size_bytes": info.st_size, "content_verified": False})
        processor = None
        if kind == "vlm":
            proc = self.read_json(root / "base/preprocessor_config.json")
            _require(proc.get("processor_class") == "Qwen2VLProcessor")
            low, high = proc.get("min_pixels"), proc.get("max_pixels")
            _require(type(low) is int and type(high) is int and 0 < low <= high <= 2**31)
            processor = {"class": "Qwen2VLProcessor", "min_pixels": low, "max_pixels": high}
            try:
                self.file_stat(root / "adapter/processor_config.json")
            except FileNotFoundError:
                pass
            else:
                extra = self.read_json(root / "adapter/processor_config.json")
                _require(extra.get("processor_class") == "Qwen2VLProcessor", "PROCESSOR_MISMATCH")
                processor["adapter_metadata_sha256"] = self.metadata[str(root / "adapter/processor_config.json")]["sha256"]
        history = self.attempt(kind.upper() + "_HISTORY", lambda: self.history(evidence, kind, package, root))
        return {"base_model_id": mid, "base_revision": package["base_revision"],
                "package_digest_declared": package["package_digest"], "image_declared": package["runtime_image"],
                "image_verified": False, "adapter": {"base_reference": reference, "rank": adapter["r"],
                "alpha": adapter["lora_alpha"]}, "processor": processor, "files": rows, "historical": history,
                "base_metadata": self.metadata[str(root / "base/config.json")],
                "adapter_metadata": self.metadata[str(root / "adapter/adapter_config.json")]}

    def ocr(self, root):
        rows = []
        for model in ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"):
            path = root / "official_models" / model
            self.read_json(path / "inference.json")
            info = self.file_stat(path / "inference.pdiparams")
            _require(info.st_size > 0, "EMPTY_ARTIFACT")
            rows.append({"model": model, "parameters_bytes": info.st_size,
                         "metadata": self.metadata[str(path / "inference.json")]})
        return {"models": rows, "device": None, "release_binding": "NOT_CHECKED"}


def _main(argv):
    args = _arguments(argv)
    check = _Inspector()
    report = {"schema_version": "ioap-real-model-inspection/v1",
              "observed_at": datetime.now(timezone.utc).isoformat(), "status": "INCOMPLETE",
              "inspection_only": True, "trial_authorized": False, "artifact_content_verified": False,
              "components": {}, "resources": None, "connections": None, "reason_codes": check.reasons}
    def interrupt(signum, frame):
        raise _Fault("DEADLINE_EXCEEDED" if signum == signal.SIGALRM else "INTERRUPTED")
    prior = {s: signal.signal(s, interrupt) for s in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        for kind in _MODELS:
            report["components"][kind] = check.attempt(kind.upper(), lambda k=kind: check.model(
                getattr(args, k + "_root"), k, args.evidence_root))
        report["components"]["ocr"] = check.attempt("OCR", lambda: check.ocr(args.ocr_root))
        check.check()
    except _Fault as error:
        check.reasons.append(str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in prior.items():
            signal.signal(sig, handler)
    report["status"] = "INCOMPLETE" if check.reasons else "OBSERVED"
    print(json.dumps(report, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 3 if check.reasons else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
