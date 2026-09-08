#!/usr/bin/env python3
"""Collect fail-closed physical evidence from one exact candidate dispatcher.

The collector owns the dispatcher process, challenges every response with a
fresh nonce, verifies the package-bound Ed25519 signature, and accepts
placement only from provider-specific live runtime properties.  Static
``cfetch_execution`` prose is identity metadata; it is never placement proof.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import statistics
import struct
import subprocess
import tempfile
import threading
import time
from typing import Any
import urllib.parse

from admission_evidence import (
    DIMENSIONS,
    DOCUMENT_PREFIX,
    EVIDENCE_IDENTITY_FIELDS,
    MAX_TOKENS,
    QUERY_PREFIX,
    SEQUENCE_BUCKETS,
    SEQUENCE_SEMANTIC_FIXTURE_ID,
    SEQUENCE_SEMANTIC_FIXTURE_SHA256,
    SUPPORTED_MAX_BATCH_SIZE,
    WIRE_BATCH_INPUT_SELECTION,
    ordered_input_json_sha256,
    parse_evidence_json,
    sequence_semantic_probe_inputs,
    utf8_sha256,
    validate_evidence_reports,
)
from export_adapter_cache import (
    ADMISSION_POLICY_SHA256,
    ATTESTATION_DOMAIN,
    ATTESTATION_NONCE_HEADER,
    ATTESTATION_SIGNATURE_HEADER,
    MODEL,
    MODEL_REVISION,
    PROFILE_ID,
    PROFILE_MANIFEST_SHA256,
)
from scifact_contract import DATASET, DATASET_REVISION
from physical_checkpoint import Attempt, Checkpoint, CheckpointError, MAX_FILES, rename_new


MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_READINESS_BYTES = 4096
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_RAW_EVIDENCE_BYTES = 16 * 1024 * 1024
SCOPE_ID_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
SIGNATURE_RE = re.compile(r"[0-9a-f]{128}")
DEVICE_FOR_CLASS = {"npu": "NPU", "gpu": "GPU", "cpu": "CPU"}
OPENVINO_SCOPE_FIELDS = {
    "scope_id",
    "transport",
    "backend",
    "runtime",
    "compiler",
    "package_target",
    "artifact_source",
    "artifact_sha256",
    "internal_precision",
    "device_class",
    "device",
    "openvino_device",
    "openvino_compile_config",
    "required_openvino_properties",
    "required_execution_devices",
    "required_host",
    "placement_evidence_sha256",
    "supported_max_tokens",
    "supported_sequence_buckets",
    "supported_max_batch_size",
    "sequence_capability_evidence_sha256",
    "performance_evidence_sha256",
    "compatibility_report_sha256",
    "attestation_public_key",
    "attestation_private_key_file",
    "accelerated_placement",
}
PACKAGE_MANIFEST_FIELDS = {
    "schema_version",
    "package_state",
    "profile_id",
    "profile_manifest_sha256",
    "admission_policy_sha256",
    "model",
    "model_revision",
    "artifact_manifest",
    "artifact_manifest_sha256",
    "runtime_manifest_sha256",
    "dependency_versions",
    "scopes",
}


class EvidenceError(ValueError):
    """Physical evidence is absent, ambiguous, or not bound to the candidate."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"cannot encode canonical evidence JSON: {error}") from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str, maximum: int | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{label} must be a regular non-symlink file")
    size = path.stat().st_size
    if size < 1 or (maximum is not None and size > maximum):
        bound = f"1..{maximum}" if maximum is not None else "at least 1"
        raise EvidenceError(f"{label} must contain {bound} bytes")
    return path.resolve()


def _read_json_file(path: Path, label: str, maximum: int) -> tuple[dict[str, Any], bytes]:
    resolved = _regular_file(path, label, maximum)
    raw = resolved.read_bytes()
    if len(raw) > maximum:
        raise EvidenceError(f"{label} changed while it was read")
    try:
        return parse_evidence_json(raw, label), raw
    except ValueError as error:
        raise EvidenceError(str(error)) from error


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise EvidenceError(f"{label} must be a nonempty string without NUL bytes")
    return value


def _same_json_value(actual: object, expected: object) -> bool:
    """Compare JSON values without Python's bool/int equality shortcut."""

    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            _same_json_value(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _same_json_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


@dataclass(frozen=True)
class ScopeContract:
    scope_id: str
    document: Mapping[str, Any]
    identity: Mapping[str, Any]
    public_key_hex: str
    openvino_device: str
    required_execution_devices: tuple[str, ...]
    required_openvino_properties: Mapping[str, str | int]
    required_host: Mapping[str, Any]
    package_state: str

    @property
    def supported_sequence_buckets(self) -> list[int]:
        return list(self.document["supported_sequence_buckets"])

    def expected_execution(self) -> dict[str, Any]:
        fields = {
            "package_state",
            "scope_id",
            "transport",
            "backend",
            "runtime",
            "compiler",
            "package_target",
            "artifact_source",
            "artifact_sha256",
            "internal_precision",
            "device_class",
            "device",
            "placement_evidence_sha256",
            "supported_max_tokens",
            "supported_sequence_buckets",
            "supported_max_batch_size",
            "sequence_capability_evidence_sha256",
            "performance_evidence_sha256",
            "accelerated_placement",
        }
        result = {
            field: self.package_state if field == "package_state" else self.document[field]
            for field in fields
        }
        # Every phase carries all four bindings explicitly.  In a physical
        # probe they are JSON null, which prevents an omitted field from
        # disguising an unbound or differently bound package.
        result["compatibility_report_sha256"] = self.document[
            "compatibility_report_sha256"
        ]
        return result


@dataclass(frozen=True)
class CandidatePackage:
    manifest_path: Path
    manifest_sha256: str
    runtime_manifest_sha256: str
    ordered_scope_ids: tuple[str, ...]
    scope: ScopeContract


def load_candidate_package(
    manifest_path: Path,
    expected_manifest_sha256: str,
    scope_id: str,
) -> CandidatePackage:
    expected_manifest_sha256 = _digest(
        expected_manifest_sha256, "package manifest digest"
    )
    document, raw = _read_json_file(
        manifest_path, "candidate package manifest", MAX_MANIFEST_BYTES
    )
    if raw != _canonical_json(document) + b"\n":
        raise EvidenceError(
            "physical-probe package manifest must use canonical JSON plus one newline"
        )
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != expected_manifest_sha256:
        raise EvidenceError(
            f"candidate package manifest has sha256 {actual_digest}, "
            f"expected {expected_manifest_sha256}"
        )
    if set(document) != PACKAGE_MANIFEST_FIELDS:
        raise EvidenceError("candidate package manifest has an unexpected schema")
    fixed = {
        "schema_version": 1,
        "package_state": "physical-probe",
        "profile_id": PROFILE_ID,
        "profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
        "admission_policy_sha256": ADMISSION_POLICY_SHA256,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
    }
    for field, expected in fixed.items():
        if not _same_json_value(document.get(field), expected):
            raise EvidenceError(
                f"candidate package manifest {field} does not match the frozen profile"
            )
    scopes = document.get("scopes")
    if not isinstance(scopes, list) or len(scopes) != 3:
        raise EvidenceError("candidate OpenVINO package must contain exactly three scopes")
    ordered_scope_ids: list[str] = []
    selected: dict[str, Any] | None = None
    expected_classes = ["npu", "gpu", "cpu"]
    for index, (entry, expected_class) in enumerate(
        zip(scopes, expected_classes, strict=True)
    ):
        label = f"candidate package scopes[{index}]"
        if not isinstance(entry, dict) or set(entry) != OPENVINO_SCOPE_FIELDS:
            raise EvidenceError(f"{label} has an unexpected schema")
        candidate_id = entry.get("scope_id")
        if (
            not isinstance(candidate_id, str)
            or len(candidate_id) > 128
            or SCOPE_ID_RE.fullmatch(candidate_id) is None
            or candidate_id in ordered_scope_ids
        ):
            raise EvidenceError(f"{label}.scope_id is invalid or duplicated")
        ordered_scope_ids.append(candidate_id)
        if entry.get("device_class") != expected_class:
            raise EvidenceError("candidate scopes must be ordered NPU, GPU, CPU")
        if entry.get("openvino_device") != DEVICE_FOR_CLASS[expected_class]:
            raise EvidenceError(f"{label} does not select one exact OpenVINO device")
        if entry.get("backend") != "openvino":
            raise EvidenceError(f"{label}.backend must be openvino")
        if entry.get("transport") != "supervised-local":
            raise EvidenceError(f"{label}.transport must be supervised-local")
        if entry.get("supported_max_tokens") != MAX_TOKENS:
            raise EvidenceError(f"{label} does not support {MAX_TOKENS} tokens")
        if entry.get("supported_sequence_buckets") != SEQUENCE_BUCKETS:
            raise EvidenceError(f"{label} does not support all frozen sequence buckets")
        if entry.get("supported_max_batch_size") != SUPPORTED_MAX_BATCH_SIZE:
            raise EvidenceError(f"{label} does not support wire batches through 64")
        if entry.get("accelerated_placement") is not True:
            raise EvidenceError(f"{label} does not claim accelerated placement")
        for field in ("artifact_sha256", "attestation_public_key"):
            _digest(entry.get(field), f"{label}.{field}")
        for field in (
            "placement_evidence_sha256",
            "sequence_capability_evidence_sha256",
            "performance_evidence_sha256",
            "compatibility_report_sha256",
        ):
            if entry.get(field) is not None:
                raise EvidenceError(
                    f"physical-probe {label}.{field} must be explicitly null"
                )
        if candidate_id == scope_id:
            selected = entry
    if selected is None:
        raise EvidenceError(f"scope {scope_id!r} is absent from the candidate package")

    identity = {
        field: selected.get(field) for field in EVIDENCE_IDENTITY_FIELDS
    }
    for field, value in identity.items():
        _nonempty(value, f"selected scope {field}")
    execution_devices = selected.get("required_execution_devices")
    if (
        not isinstance(execution_devices, list)
        or len(execution_devices) != 1
        or any(not isinstance(value, str) for value in execution_devices)
    ):
        raise EvidenceError("selected scope needs one exact required execution device")
    properties = selected.get("required_openvino_properties")
    if not isinstance(properties, dict) or not properties:
        raise EvidenceError("selected scope needs required OpenVINO properties")
    if any(type(value) not in {str, int} for value in properties.values()):
        raise EvidenceError("required OpenVINO property values must be strings or integers")
    host = selected.get("required_host")
    if not isinstance(host, dict) or set(host) != {
        "system",
        "machine",
        "kernel_release",
        "files",
    }:
        raise EvidenceError("selected scope needs an exact required_host binding")
    files = host.get("files")
    if not isinstance(files, list) or not files:
        raise EvidenceError("selected scope required_host.files must be nonempty")
    for index, row in enumerate(files):
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise EvidenceError(f"required_host.files[{index}] has an unexpected schema")
        _nonempty(row["path"], f"required_host.files[{index}].path")
        _digest(row["sha256"], f"required_host.files[{index}].sha256")
    return CandidatePackage(
        manifest_path=manifest_path.resolve(),
        manifest_sha256=actual_digest,
        runtime_manifest_sha256=_digest(
            document.get("runtime_manifest_sha256"), "runtime manifest digest"
        ),
        ordered_scope_ids=tuple(ordered_scope_ids),
        scope=ScopeContract(
            scope_id=scope_id,
            document=selected,
            identity=identity,
            public_key_hex=selected["attestation_public_key"],
            openvino_device=selected["openvino_device"],
            required_execution_devices=tuple(execution_devices),
            required_openvino_properties=properties,
            required_host=host,
            package_state="physical-probe",
        ),
    )


class _StderrCapture:
    def __init__(self, stream, maximum: int = 64 * 1024) -> None:
        self._stream = stream
        self._maximum = maximum
        self._data = bytearray()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._drain, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _drain(self) -> None:
        while True:
            chunk = self._stream.read(4096)
            if not chunk:
                return
            with self._lock:
                self._data.extend(chunk)
                if len(self._data) > self._maximum:
                    del self._data[: len(self._data) - self._maximum]

    def finish(self) -> None:
        self._thread.join(timeout=2)

    def text(self) -> str:
        with self._lock:
            return bytes(self._data).decode("utf-8", errors="replace")


class DispatcherSession:
    """Own one exact dispatcher from startup through parent-EOF shutdown."""

    def __init__(
        self,
        dispatcher: Path,
        dispatcher_sha256: str,
        package: CandidatePackage,
        startup_timeout_seconds: float,
    ) -> None:
        self.dispatcher = dispatcher.resolve()
        self.dispatcher_sha256 = _digest(dispatcher_sha256, "dispatcher digest")
        self.package = package
        self.startup_timeout_seconds = startup_timeout_seconds
        self.process: subprocess.Popen[bytes] | None = None
        self.bearer: str | None = None
        self.endpoint: str | None = None
        self._stderr: _StderrCapture | None = None
        self._startup_sampler: RssSampler | None = None
        self.startup_peak_rss_bytes: int | None = None
        self.startup_rss_sample_count = 0

    def _verify_inputs(self) -> None:
        _regular_file(self.dispatcher, "candidate dispatcher")
        if self.dispatcher.parent != self.package.manifest_path.parent:
            raise EvidenceError(
                "candidate dispatcher and package manifest must be exact siblings"
            )
        actual = _file_sha256(self.dispatcher)
        if actual != self.dispatcher_sha256:
            raise EvidenceError(
                f"candidate dispatcher has sha256 {actual}, expected {self.dispatcher_sha256}"
            )
        if _file_sha256(self.package.manifest_path) != self.package.manifest_sha256:
            raise EvidenceError("candidate package manifest changed before launch")

    def __enter__(self) -> "DispatcherSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.stop()
        except Exception:
            if exc_type is None:
                raise

    def start(self) -> None:
        if self.process is not None:
            raise EvidenceError("dispatcher session was started twice")
        if self.startup_timeout_seconds <= 0:
            raise EvidenceError("dispatcher startup timeout must be positive")
        self._verify_inputs()
        self.bearer = secrets.token_hex(32)
        process = subprocess.Popen(
            [
                str(self.dispatcher),
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--auth-stdin",
            ],
            cwd=self.dispatcher.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        self.process = process
        try:
            self._startup_sampler = RssSampler(process.pid)
            self._startup_sampler.start()
        except BaseException:
            self._abort()
            raise
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        self._stderr = _StderrCapture(process.stderr)
        self._stderr.start()
        try:
            process.stdin.write(
                json.dumps({"bearer": self.bearer}, separators=(",", ":")).encode()
                + b"\n"
            )
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            detail = self.stderr_text()
            self._abort()
            raise EvidenceError(
                "candidate dispatcher rejected its stdin authentication boundary"
                + (f": {detail}" if detail else "")
            ) from error
        result: queue.Queue[bytes | BaseException] = queue.Queue(maxsize=1)

        def read_ready() -> None:
            try:
                result.put(process.stdout.readline(MAX_READINESS_BYTES + 1))
            except BaseException as error:  # pragma: no cover - OS pipe failure
                result.put(error)

        reader = threading.Thread(target=read_ready, daemon=True)
        try:
            reader.start()
        except RuntimeError:
            self._abort()
            raise
        try:
            ready_value = result.get(timeout=self.startup_timeout_seconds)
        except queue.Empty as error:
            self._abort()
            raise EvidenceError("candidate dispatcher readiness timed out") from error
        if isinstance(ready_value, BaseException):
            self._abort()
            raise EvidenceError(f"cannot read candidate dispatcher readiness: {ready_value}")
        if (
            not ready_value
            or len(ready_value) > MAX_READINESS_BYTES
            or not ready_value.endswith(b"\n")
        ):
            detail = self.stderr_text()
            self._abort()
            raise EvidenceError(
                "candidate dispatcher did not emit one bounded readiness line"
                + (f": {detail}" if detail else "")
            )
        try:
            ready = parse_evidence_json(ready_value, "dispatcher readiness")
        except ValueError as error:
            self._abort()
            raise EvidenceError(str(error)) from error
        if set(ready) != {"schema_version", "url", "scope_ids"}:
            self._abort()
            raise EvidenceError("dispatcher readiness has an unexpected schema")
        if ready["schema_version"] != 1 or ready["scope_ids"] != list(
            self.package.ordered_scope_ids
        ):
            self._abort()
            raise EvidenceError("dispatcher readiness does not bind the ordered package scopes")
        url = ready.get("url")
        if not isinstance(url, str):
            self._abort()
            raise EvidenceError("dispatcher readiness URL must be a string")
        parsed = urllib.parse.urlsplit(url)
        try:
            port = parsed.port
        except ValueError as error:
            self._abort()
            raise EvidenceError("dispatcher readiness URL has an invalid port") from error
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or port is None
            or port <= 0
            or parsed.path != "/v1"
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
        ):
            self._abort()
            raise EvidenceError("dispatcher readiness URL is not an exact loopback /v1 URL")
        if process.poll() is not None:
            detail = self.stderr_text()
            self._abort()
            raise EvidenceError(
                "candidate dispatcher exited after readiness"
                + (f": {detail}" if detail else "")
            )
        try:
            self._finish_startup_sampler(required=True)
        except EvidenceError:
            self._abort()
            raise
        self.endpoint = f"http://127.0.0.1:{port}/v1/embeddings"

    def stderr_text(self) -> str:
        return self._stderr.text().strip() if self._stderr is not None else ""

    def _finish_startup_sampler(self, *, required: bool) -> None:
        sampler = self._startup_sampler
        if sampler is None:
            return
        self._startup_sampler = None
        try:
            peak, count = sampler.finish()
        except EvidenceError:
            if required:
                raise
            return
        self.startup_peak_rss_bytes = peak
        self.startup_rss_sample_count = count

    def _abort(self) -> None:
        process = self.process
        if process is None:
            return
        self._finish_startup_sampler(required=False)
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if self._stderr is not None:
            self._stderr.finish()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        self._finish_startup_sampler(required=True)
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired as error:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            raise EvidenceError("candidate dispatcher ignored parent EOF shutdown") from error
        if self._stderr is not None:
            self._stderr.finish()
        if return_code != 0:
            detail = self.stderr_text()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            raise EvidenceError(
                f"candidate dispatcher exited with status {return_code}"
                + (f": {detail}" if detail else "")
            )
        assert process.stdout is not None
        try:
            if process.stdout.read(1):
                raise EvidenceError("candidate dispatcher wrote data after readiness")
        finally:
            process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        self._verify_inputs()
        self.process = None


def _proc_status_value(pid: int, name: str) -> int | None:
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    prefix = name + ":"
    for line in lines:
        if line.startswith(prefix):
            parts = line.split()
            if len(parts) == 3 and parts[2] == "kB" and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def _process_children(pid: int) -> set[int]:
    children: set[int] = set()
    task_root = Path(f"/proc/{pid}/task")
    try:
        tasks = list(task_root.iterdir())
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return children
    for task in tasks:
        try:
            raw = (task / "children").read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        for value in raw.split():
            if value.isdigit():
                children.add(int(value))
    return children


def process_tree_rss_bytes(root_pid: int) -> int | None:
    pending = [root_pid]
    seen: set[int] = set()
    total = 0
    measured = False
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        rss = _proc_status_value(pid, "VmRSS")
        if rss is not None:
            total += rss
            measured = True
        pending.extend(_process_children(pid) - seen)
    return total if measured else None


class RssSampler:
    def __init__(self, root_pid: int, interval_seconds: float = 0.002) -> None:
        if not Path("/proc").is_dir():
            raise EvidenceError("physical RSS collection requires Linux /proc")
        self.root_pid = root_pid
        self.interval_seconds = interval_seconds
        self.values: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _take(self) -> None:
        value = process_tree_rss_bytes(self.root_pid)
        if value is not None and value > 0:
            self.values.append(value)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._take()

    def start(self) -> None:
        self._take()
        self._thread.start()

    def finish(self) -> tuple[int, int]:
        self._take()
        self._stop.set()
        self._thread.join(timeout=1)
        self._take()
        if not self.values:
            raise EvidenceError("dispatcher RSS could not be measured from Linux /proc")
        return max(self.values), len(self.values)


def _f32(value: float) -> float:
    try:
        return struct.unpack("!f", struct.pack("!f", value))[0]
    except (OverflowError, struct.error) as error:
        raise EvidenceError("embedding component is not representable as float32") from error


def canonical_i8_bytes(vector: Sequence[float]) -> bytes:
    """Apply the frozen float32 codec without requiring NumPy on the collector."""

    if len(vector) != DIMENSIONS:
        raise EvidenceError(f"embedding must contain exactly {DIMENSIONS} components")
    values = [_f32(float(value)) for value in vector]
    if any(not math.isfinite(value) for value in values):
        raise EvidenceError("embedding contains a non-finite float32 component")
    maximum = max(abs(value) for value in values)
    if maximum <= 0.0:
        raise EvidenceError("embedding is all zero")
    encoded: list[int] = []
    for value in values:
        divided = _f32(value / maximum)
        scaled = _f32(divided * _f32(127.0))
        clipped = min(_f32(127.0), max(_f32(-127.0), scaled))
        quantized = int(round(clipped))
        if not -127 <= quantized <= 127:
            raise EvidenceError("canonical codec produced an out-of-range component")
        encoded.append(quantized)
    if max(abs(value) for value in encoded) != 127:
        raise EvidenceError("canonical codec output has no +/-127 extremum")
    return bytes(value & 0xFF for value in encoded)


def _signed_components(raw: bytes) -> list[int]:
    return [value if value < 128 else value - 256 for value in raw]


def exact_i8_relevant_precedes(query: bytes, relevant: bytes, irrelevant: bytes) -> bool:
    q = _signed_components(query)
    r = _signed_components(relevant)
    i = _signed_components(irrelevant)
    relevant_dot = sum(left * right for left, right in zip(q, r, strict=True))
    irrelevant_dot = sum(left * right for left, right in zip(q, i, strict=True))
    relevant_norm = sum(value * value for value in r)
    irrelevant_norm = sum(value * value for value in i)
    relevant_sign = (relevant_dot > 0) - (relevant_dot < 0)
    irrelevant_sign = (irrelevant_dot > 0) - (irrelevant_dot < 0)
    if relevant_sign != irrelevant_sign:
        return relevant_sign > irrelevant_sign
    if relevant_sign == 0:
        return False
    relevant_squared = relevant_dot * relevant_dot * irrelevant_norm
    irrelevant_squared = irrelevant_dot * irrelevant_dot * relevant_norm
    return (
        relevant_squared > irrelevant_squared
        if relevant_sign > 0
        else relevant_squared < irrelevant_squared
    )


@dataclass(frozen=True)
class ResponseRow:
    token_count: int
    sequence_bucket: int
    canonical: bytes


@dataclass(frozen=True)
class SignedTransaction:
    nonce_hex: str
    signature_hex: str
    request_body: bytes
    response_body: bytes
    elapsed_ns: int
    peak_rss_bytes: int | None
    rss_sample_count: int
    rows: tuple[ResponseRow, ...]
    runtime_evidence: Mapping[str, Any]

    def raw_document(self) -> dict[str, Any]:
        return {
            "nonce_hex": self.nonce_hex,
            "signature_hex": self.signature_hex,
            "request_body_base64": base64.b64encode(self.request_body).decode("ascii"),
            "response_body_base64": base64.b64encode(self.response_body).decode("ascii"),
            "request_body_sha256": hashlib.sha256(self.request_body).hexdigest(),
            "response_body_sha256": hashlib.sha256(self.response_body).hexdigest(),
            "elapsed_ns": self.elapsed_ns,
            "peak_rss_bytes": self.peak_rss_bytes,
            "rss_sample_count": self.rss_sample_count,
        }


class OpenVinoLiveEvidenceValidator:
    """Validate live, signed OpenVINO properties against the candidate scope."""

    def __init__(self, scope: ScopeContract) -> None:
        self.scope = scope

    def validate(self, value: object, executed_buckets: Sequence[int]) -> None:
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "provider",
            "scope_id",
            "host",
            "host_source",
            "bucket_results",
        }:
            raise EvidenceError(
                "signed response lacks the exact OpenVINO live-runtime evidence schema"
            )
        fixed = {
            "schema_version": 1,
            "provider": "openvino",
            "scope_id": self.scope.scope_id,
            "host_source": "platform-and-sha256",
        }
        for field, expected in fixed.items():
            if not _same_json_value(value.get(field), expected):
                raise EvidenceError(f"live OpenVINO evidence {field} is not exact")
        if not _same_json_value(value.get("host"), self.scope.required_host):
            raise EvidenceError(
                "signed live host/kernel/file evidence does not match the package scope"
            )
        records = value.get("bucket_results")
        expected_buckets = sorted(set(executed_buckets))
        if (
            not isinstance(records, list)
            or [row.get("bucket") if isinstance(row, dict) else None for row in records]
            != expected_buckets
        ):
            raise EvidenceError(
                "signed live OpenVINO evidence does not cover the executed static buckets"
            )
        fields = {
            "bucket",
            "requested_device",
            "execution_devices",
            "execution_devices_source",
            "device_properties",
            "device_properties_source",
        }
        for row in records:
            if not isinstance(row, dict) or set(row) != fields:
                raise EvidenceError("live OpenVINO bucket evidence has an unexpected schema")
            if row["requested_device"] != self.scope.openvino_device:
                raise EvidenceError("live OpenVINO evidence requested another device")
            if row["execution_devices_source"] != (
                "compiled_model.get_property(EXECUTION_DEVICES)"
            ):
                raise EvidenceError(
                    "placement is an echoed claim, not compiled-model EXECUTION_DEVICES"
                )
            if row["device_properties_source"] != "core.get_property":
                raise EvidenceError(
                    "device properties are an echoed claim, not live core.get_property values"
                )
            if not _same_json_value(
                row["execution_devices"], list(self.scope.required_execution_devices)
            ):
                raise EvidenceError("compiled model executed on an unexpected device")
            if not _same_json_value(
                row["device_properties"], dict(self.scope.required_openvino_properties)
            ):
                raise EvidenceError("live OpenVINO device properties changed from the scope")


def _request_body(scope: ScopeContract, texts: Sequence[str]) -> bytes:
    return json.dumps(
        {"model": MODEL, "dimensions": DIMENSIONS, "input": list(texts),
         "cfetch_requested_scope_id": scope.scope_id},
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")


def _positive_measurement(value: object, label: str) -> int:
    if type(value) is not int or not 0 < value <= (1 << 63) - 1:
        raise EvidenceError(f"checkpoint {label} must be a positive bounded integer")
    return value


def _restore_transaction(
    scope: ScopeContract, raw: dict, texts: Sequence[str], *, measure_rss: bool,
) -> SignedTransaction:
    """Reverify original signed bytes; no live dispatcher or new measurement."""
    fields = {"nonce_hex", "signature_hex", "request_body_base64", "response_body_base64",
              "request_body_sha256", "response_body_sha256", "elapsed_ns",
              "peak_rss_bytes", "rss_sample_count"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise EvidenceError("checkpoint signed transaction schema changed")
    nonce = bytes.fromhex(_digest(raw["nonce_hex"], "checkpoint nonce"))
    bodies = []
    for field in ("request_body", "response_body"):
        encoded = raw[f"{field}_base64"]
        if not isinstance(encoded, str):
            raise EvidenceError("checkpoint body must be canonical base64")
        try:
            body = base64.b64decode(encoded, validate=True)
        except (ValueError, UnicodeError) as error:
            raise EvidenceError("checkpoint body must be canonical base64") from error
        if not 1 <= len(body) <= MAX_RESPONSE_BYTES or (
                base64.b64encode(body).decode("ascii") != encoded
                or hashlib.sha256(body).hexdigest() != raw[f"{field}_sha256"]):
            raise EvidenceError("checkpoint body digest or size differs")
        bodies.append(body)
    request_body, response_body = bodies
    if request_body != _request_body(scope, texts):
        raise EvidenceError("checkpoint request input order or identity changed")
    if not isinstance(raw["signature_hex"], str):
        raise EvidenceError("checkpoint signature must be hexadecimal")
    # Use precisely the live client's signature and response validators without
    # constructing its network/process session.
    verifier = object.__new__(SignedAdapterClient)
    verifier.scope = scope
    verifier._live_validator = OpenVinoLiveEvidenceValidator(scope)
    verifier._verify_signature(nonce, request_body, response_body, raw["signature_hex"])
    rows, runtime = verifier._validate_response(
        parse_evidence_json(response_body, "checkpoint signed response"), len(texts)
    )
    elapsed = _positive_measurement(raw["elapsed_ns"], "elapsed_ns")
    if measure_rss:
        _positive_measurement(raw["peak_rss_bytes"], "peak_rss_bytes")
        _positive_measurement(raw["rss_sample_count"], "rss_sample_count")
    elif raw["peak_rss_bytes"] is not None or type(raw["rss_sample_count"]) is not int or raw["rss_sample_count"] != 0:
        raise EvidenceError("wire checkpoint unexpectedly claims RSS measurements")
    return SignedTransaction(nonce.hex(), raw["signature_hex"], request_body,
                             response_body, elapsed, raw["peak_rss_bytes"],
                             raw["rss_sample_count"], tuple(rows), runtime)


class SignedAdapterClient:
    def __init__(
        self,
        session: DispatcherSession,
        timeout_seconds: float,
        nonce_registry: set[bytes] | None = None,
        checkpoint_attempt: Attempt | None = None,
    ) -> None:
        if session.process is None or session.endpoint is None or session.bearer is None:
            raise EvidenceError("signed adapter client needs a running dispatcher")
        self.session = session
        self.scope = session.package.scope
        self.timeout_seconds = timeout_seconds
        self._nonces = nonce_registry if nonce_registry is not None else set()
        self._live_validator = OpenVinoLiveEvidenceValidator(self.scope)
        self._checkpoint_attempt = checkpoint_attempt

    def _verify_signature(
        self,
        nonce: bytes,
        request_body: bytes,
        response_body: bytes,
        signature_hex: str,
    ) -> None:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        if SIGNATURE_RE.fullmatch(signature_hex) is None:
            raise EvidenceError("adapter signature is not one lowercase Ed25519 signature")
        message = b"".join(
            (
                ATTESTATION_DOMAIN,
                nonce,
                hashlib.sha256(request_body).digest(),
                hashlib.sha256(response_body).digest(),
            )
        )
        try:
            key = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(self.scope.public_key_hex)
            )
            key.verify(bytes.fromhex(signature_hex), message)
        except (InvalidSignature, ValueError) as error:
            raise EvidenceError(
                "adapter response failed its package-bound Ed25519 signature"
            ) from error

    def request(self, texts: Sequence[str], measure_rss: bool = True) -> SignedTransaction:
        if not texts or len(texts) > SUPPORTED_MAX_BATCH_SIZE:
            raise EvidenceError("adapter request must contain 1..64 inputs")
        if any(not isinstance(text, str) or not text for text in texts):
            raise EvidenceError("adapter request inputs must be nonempty strings")
        body = _request_body(self.scope, texts)
        nonce = secrets.token_bytes(32)
        if len(nonce) != 32 or nonce in self._nonces:
            raise EvidenceError("attestation nonce source repeated or returned a wrong size")
        self._nonces.add(nonce)
        if self._checkpoint_attempt is not None:
            self._checkpoint_attempt.reserve(nonce, body)
        parsed = urllib.parse.urlsplit(self.session.endpoint)
        assert parsed.port is not None
        sampler: RssSampler | None = None
        assert self.session.process is not None
        if measure_rss:
            sampler = RssSampler(self.session.process.pid)
            sampler.start()
        started = time.perf_counter_ns()
        connection = http.client.HTTPConnection(
            "127.0.0.1", parsed.port, timeout=self.timeout_seconds
        )
        peak_rss: int | None = None
        rss_samples = 0
        request_error: BaseException | None = None
        try:
            connection.request(
                "POST",
                parsed.path,
                body=body,
                headers={
                    "Authorization": f"Bearer {self.session.bearer}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    ATTESTATION_NONCE_HEADER: nonce.hex(),
                },
            )
            response = connection.getresponse()
            content_length = response.getheader("Content-Length")
            try:
                declared_length = int(content_length) if content_length is not None else -1
            except ValueError as error:
                raise EvidenceError("adapter returned an invalid Content-Length") from error
            if not 1 <= declared_length <= MAX_RESPONSE_BYTES:
                raise EvidenceError("adapter response Content-Length is missing or out of bounds")
            response_body = response.read(MAX_RESPONSE_BYTES + 1)
            elapsed_ns = time.perf_counter_ns() - started
            if len(response_body) != declared_length:
                raise EvidenceError("adapter response body does not match Content-Length")
            if response.status != 200:
                detail = response_body[:512].decode("utf-8", errors="replace")
                raise EvidenceError(f"adapter returned HTTP {response.status}: {detail}")
            if response.getheader("Content-Type", "").split(";", 1)[0].strip().lower() != (
                "application/json"
            ):
                raise EvidenceError("adapter response Content-Type is not application/json")
            signatures = response.headers.get_all(ATTESTATION_SIGNATURE_HEADER, failobj=[])
            if len(signatures) != 1:
                raise EvidenceError("adapter response must contain exactly one signature")
            signature_hex = signatures[0]
        except BaseException as error:
            request_error = error
            raise
        finally:
            connection.close()
            if sampler is not None:
                try:
                    peak_rss, rss_samples = sampler.finish()
                except BaseException:
                    if request_error is None:
                        raise
        self._verify_signature(nonce, body, response_body, signature_hex)
        payload = parse_evidence_json(response_body, "signed adapter response")
        rows, runtime_evidence = self._validate_response(payload, len(texts))
        transaction = SignedTransaction(
            nonce_hex=nonce.hex(),
            signature_hex=signature_hex,
            request_body=body,
            response_body=response_body,
            elapsed_ns=elapsed_ns,
            peak_rss_bytes=peak_rss,
            rss_sample_count=rss_samples,
            rows=tuple(rows),
            runtime_evidence=runtime_evidence,
        )
        if self._checkpoint_attempt is not None:
            self._checkpoint_attempt.save(transaction)
        return transaction

    def _validate_response(
        self, payload: dict[str, Any], expected_items: int
    ) -> tuple[list[ResponseRow], Mapping[str, Any]]:
        fields = {
            "model",
            "cfetch_profile",
            "cfetch_profile_manifest_sha256",
            "cfetch_admission_policy_sha256",
            "cfetch_model_revision",
            "cfetch_execution",
            "cfetch_runtime_evidence",
            "data",
        }
        if set(payload) != fields:
            raise EvidenceError("signed adapter response has an unexpected top-level schema")
        fixed = {
            "model": MODEL,
            "cfetch_profile": PROFILE_ID,
            "cfetch_profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
            "cfetch_admission_policy_sha256": ADMISSION_POLICY_SHA256,
            "cfetch_model_revision": MODEL_REVISION,
        }
        for field, expected in fixed.items():
            if payload.get(field) != expected:
                raise EvidenceError(f"signed adapter response {field} is not frozen")
        expected_execution = self.scope.expected_execution()
        if not _same_json_value(payload.get("cfetch_execution"), expected_execution):
            raise EvidenceError("signed adapter execution identity differs from the package")
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != expected_items:
            raise EvidenceError("signed adapter returned the wrong number of rows")
        ordered: list[ResponseRow | None] = [None] * expected_items
        row_fields = {
            "index",
            "cfetch_scope_id",
            "token_count",
            "sequence_bucket",
            "truncated",
            "embedding",
        }
        for row in data:
            if not isinstance(row, dict) or set(row) != row_fields:
                raise EvidenceError("signed adapter row has an unexpected schema")
            index = row["index"]
            if type(index) is not int or not 0 <= index < expected_items:
                raise EvidenceError("signed adapter row index is out of range")
            if ordered[index] is not None:
                raise EvidenceError("signed adapter returned a duplicate row index")
            if row["cfetch_scope_id"] != self.scope.scope_id:
                raise EvidenceError("signed adapter row came from another scope")
            token_count = row["token_count"]
            bucket = row["sequence_bucket"]
            if type(token_count) is not int or not 1 <= token_count <= MAX_TOKENS:
                raise EvidenceError("signed adapter row has an invalid token count")
            expected_bucket = next(
                (value for value in SEQUENCE_BUCKETS if value >= token_count), None
            )
            if bucket != expected_bucket or row["truncated"] is not False:
                raise EvidenceError("signed adapter row selected a wrong bucket or truncated")
            embedding = row["embedding"]
            if (
                not isinstance(embedding, list)
                or len(embedding) != DIMENSIONS
                or any(type(value) not in {int, float} for value in embedding)
            ):
                raise EvidenceError("signed adapter row has an invalid embedding")
            canonical = canonical_i8_bytes(embedding)
            ordered[index] = ResponseRow(token_count, bucket, canonical)
        if any(row is None for row in ordered):
            raise EvidenceError("signed adapter response omitted a row")
        complete = [row for row in ordered if row is not None]
        runtime_evidence = payload["cfetch_runtime_evidence"]
        self._live_validator.validate(
            runtime_evidence,
            [row.sequence_bucket for row in complete],
        )
        return complete, runtime_evidence


def load_wire_inputs(path: Path) -> list[str]:
    document, _raw = _read_json_file(path, "wire probe inputs", MAX_MANIFEST_BYTES)
    expected_fields = {
        "schema_version",
        "dataset",
        "dataset_revision",
        "selection",
        "inputs",
    }
    if set(document) != expected_fields:
        raise EvidenceError("wire probe input manifest has an unexpected schema")
    fixed = {
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "selection": WIRE_BATCH_INPUT_SELECTION,
    }
    for field, expected in fixed.items():
        if document.get(field) != expected:
            raise EvidenceError(f"wire probe input manifest {field} is not pinned")
    inputs = document.get("inputs")
    if (
        not isinstance(inputs, list)
        or len(inputs) != SUPPORTED_MAX_BATCH_SIZE
        or any(not isinstance(value, str) or not value for value in inputs)
    ):
        raise EvidenceError("wire probe input manifest must contain exactly 64 strings")
    if any(not value.startswith(QUERY_PREFIX) for value in inputs[:32]):
        raise EvidenceError("wire probe inputs 0..31 must use the pinned query prefix")
    if any(not value.startswith(DOCUMENT_PREFIX) for value in inputs[32:]):
        raise EvidenceError("wire probe inputs 32..63 must use the pinned document prefix")
    return inputs


def _store_raw(raw_root: Path, document: Mapping[str, Any]) -> str:
    raw = _canonical_json(document) + b"\n"
    if len(raw) > MAX_RAW_EVIDENCE_BYTES:
        raise EvidenceError("one raw evidence record exceeds the 16 MiB bound")
    digest = hashlib.sha256(raw).hexdigest()
    destination = raw_root / f"{digest}.bin"
    if destination.exists():
        if destination.read_bytes() != raw:
            raise EvidenceError(f"raw evidence digest collision at {destination}")
    else:
        with destination.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    return digest


def _write_summary(path: Path, document: Mapping[str, Any]) -> str:
    raw = _canonical_json(document) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _latency_percentile_ns(values: Sequence[int], percentile: float) -> int:
    if not values:
        raise EvidenceError("latency percentile needs at least one sample")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _check_bucket_semantics(bucket: int, first: SignedTransaction, repeat: SignedTransaction):
    if any(row.token_count != bucket for row in (*first.rows, *repeat.rows)):
        raise EvidenceError(f"semantic probe inputs did not tokenize to exact bucket {bucket}")
    first_outputs = [row.canonical for row in first.rows]
    if first_outputs != [row.canonical for row in repeat.rows]:
        raise EvidenceError(f"bucket {bucket} canonical output is not repeatable")
    if not exact_i8_relevant_precedes(*first_outputs):
        raise EvidenceError(f"bucket {bucket} semantic probe did not rank relevant before irrelevant")
    return first_outputs


def _restore_attempt(attempt: Attempt, scope: ScopeContract, requests: Sequence[Sequence[str]],
                     *, measure_rss: bool) -> list[SignedTransaction]:
    if len(attempt.reservations) > len(requests) or (attempt.completion is not None and
            len(attempt.transactions) != len(requests)):
        raise EvidenceError("checkpoint attempt has the wrong request count")
    for reservation, texts in zip(attempt.reservations, requests):
        if reservation["request_body_sha256"] != hashlib.sha256(_request_body(scope, texts)).hexdigest():
            raise EvidenceError("checkpoint reserved request order changed")
    return [_restore_transaction(scope, raw, texts, measure_rss=measure_rss)
            for raw, texts in zip(attempt.transactions, requests)]


def _bucket_requests(bucket: int, warmup_count: int, sample_count: int):
    texts = sequence_semantic_probe_inputs(bucket)
    return [texts, texts] + [[texts[0]]] * (warmup_count + sample_count)


def _run_bucket(
    dispatcher: Path,
    dispatcher_sha256: str,
    package: CandidatePackage,
    startup_timeout_seconds: float,
    request_timeout_seconds: float,
    warmup_count: int,
    sample_count: int,
    energy_not_measured_reason: str,
    bucket: int,
    raw_root: Path,
    nonce_registry: set[bytes],
    checkpoint: Checkpoint | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    attempt = checkpoint.completed(f"bucket-{bucket}") if checkpoint else None
    if attempt is not None:
        transactions = _restore_attempt(attempt, package.scope,
                                        _bucket_requests(bucket, warmup_count, sample_count),
                                        measure_rss=True)
        metadata = attempt.completion["metadata"]
    else:
        attempt = checkpoint.start(f"bucket-{bucket}") if checkpoint else None
        with DispatcherSession(dispatcher, dispatcher_sha256, package,
                               startup_timeout_seconds) as session:
            client = SignedAdapterClient(session, request_timeout_seconds, nonce_registry,
                                         checkpoint_attempt=attempt)
            texts = sequence_semantic_probe_inputs(bucket)
            first, repeat = client.request(texts), client.request(texts)
            _check_bucket_semantics(bucket, first, repeat)
            warmups = [client.request([texts[0]]) for _ in range(warmup_count)]
            samples = [client.request([texts[0]]) for _ in range(sample_count)]
            transactions = [first, repeat, *warmups, *samples]
        metadata = {
            "collected_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "startup_peak_rss_bytes": session.startup_peak_rss_bytes,
            "startup_rss_sample_count": session.startup_rss_sample_count,
        }
    result = _summarize_bucket(dispatcher_sha256, package, warmup_count, sample_count,
                               energy_not_measured_reason, bucket, raw_root, transactions, metadata)
    if attempt is not None:
        if attempt.completion is None:
            attempt.complete(metadata, list(result))
        elif not _same_json_value(attempt.completion["summary"], list(result)):
            raise EvidenceError("checkpoint bucket summary differs from revalidated transactions")
    return result


def _summarize_bucket(dispatcher_sha256, package, warmup_count, sample_count,
                      energy_not_measured_reason, bucket, raw_root, transactions, metadata):
    if not isinstance(metadata, dict) or set(metadata) != {
            "collected_at_utc", "startup_peak_rss_bytes", "startup_rss_sample_count"}:
        raise EvidenceError("checkpoint bucket metadata schema changed")
    _positive_measurement(metadata["startup_peak_rss_bytes"], "startup_peak_rss_bytes")
    _positive_measurement(metadata["startup_rss_sample_count"], "startup_rss_sample_count")
    collected_at = metadata["collected_at_utc"]
    if not isinstance(collected_at, str) or not collected_at.endswith("Z"):
        raise EvidenceError("checkpoint collection timestamp must be UTC")
    try:
        datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError("checkpoint collection timestamp is invalid") from error
    if len(transactions) != 2 + warmup_count + sample_count:
        raise EvidenceError("checkpoint performance attempt is incomplete")
    first, repeat = transactions[:2]
    warmups = transactions[2:2 + warmup_count]
    samples = transactions[2 + warmup_count:]
    texts = sequence_semantic_probe_inputs(bucket)
    first_outputs = _check_bucket_semantics(bucket, first, repeat)
    placement_raw = {
        "schema_version": 1,
        "kind": "cfetch-openvino-live-placement-v1",
        "collected_at_utc": collected_at,
        "scope_id": package.scope.scope_id,
        "bucket": bucket,
        "dispatcher_sha256": dispatcher_sha256,
        "probe_package_manifest_sha256": package.manifest_sha256,
        "runtime_manifest_sha256": package.runtime_manifest_sha256,
        "placement_source": "signed-live-openvino-runtime-evidence",
        "transactions": [first.raw_document(), repeat.raw_document()],
    }
    profiler_digest = _store_raw(raw_root, placement_raw)
    benchmark_raw = {
        "schema_version": 1,
        "kind": "cfetch-loopback-end-to-end-benchmark-v1",
        "collected_at_utc": collected_at,
        "scope_id": package.scope.scope_id,
        "bucket": bucket,
        "dispatcher_sha256": dispatcher_sha256,
        "probe_package_manifest_sha256": package.manifest_sha256,
        "runtime_manifest_sha256": package.runtime_manifest_sha256,
        "latency_clock": "time.perf_counter_ns",
        "latency_boundary": "signed-loopback-http-request-response",
        "peak_memory_method": "2ms-sampled-linux-proc-process-tree-vmrss",
        "startup_peak_rss_bytes": metadata["startup_peak_rss_bytes"],
        "startup_rss_sample_count": metadata["startup_rss_sample_count"],
        "warmup_count": warmup_count,
        "sample_count": sample_count,
        "warmups": [transaction.raw_document() for transaction in warmups],
        "samples": [transaction.raw_document() for transaction in samples],
        "initialization_peak_rss_bytes": max(
            value
            for value in (
                metadata["startup_peak_rss_bytes"],
                first.peak_rss_bytes,
                repeat.peak_rss_bytes,
            )
            if value is not None
        ),
        "energy_measurement": "not_measured",
        "energy_not_measured_reason": energy_not_measured_reason,
    }
    benchmark_digest = _store_raw(raw_root, benchmark_raw)
    latency_values = [transaction.elapsed_ns for transaction in samples]
    rss_values = [
        transaction.peak_rss_bytes
        for transaction in (first, repeat, *warmups, *samples)
        if transaction.peak_rss_bytes is not None
    ]
    if metadata["startup_peak_rss_bytes"] is not None:
        rss_values.append(metadata["startup_peak_rss_bytes"])
    if not rss_values:
        raise EvidenceError(f"bucket {bucket} has no honest RSS measurement")
    sequence_row = {
        "bucket": bucket,
        "requested_tokens": bucket,
        "tokenized_tokens": bucket,
        "executed_shape_tokens": bucket,
        "output_dimensions": DIMENSIONS,
        "finite_output": True,
        "nonzero_output": True,
        "truncated": False,
        "semantic_probe": {
            "fixture_id": SEQUENCE_SEMANTIC_FIXTURE_ID,
            "fixture_sha256": SEQUENCE_SEMANTIC_FIXTURE_SHA256,
            "query_input_utf8_sha256": utf8_sha256(texts[0]),
            "relevant_document_input_utf8_sha256": utf8_sha256(texts[1]),
            "irrelevant_document_input_utf8_sha256": utf8_sha256(texts[2]),
            "query_token_count": bucket,
            "relevant_document_token_count": bucket,
            "irrelevant_document_token_count": bucket,
            "query_canonical_output_bytes_sha256": hashlib.sha256(
                first_outputs[0]
            ).hexdigest(),
            "relevant_document_canonical_output_bytes_sha256": hashlib.sha256(
                first_outputs[1]
            ).hexdigest(),
            "irrelevant_document_canonical_output_bytes_sha256": hashlib.sha256(
                first_outputs[2]
            ).hexdigest(),
            "canonical_repeatability": True,
            "self_relevant_before_irrelevant": True,
        },
    }
    placement_row = {
        "bucket": bucket,
        "accelerator_execution_confirmed": True,
        "fallback_disclosure_complete": True,
        "unexpected_fallback_detected": False,
        "fallback_summary": (
            "No fallback configured; signed live OpenVINO EXECUTION_DEVICES "
            f"equalled {list(package.scope.required_execution_devices)!r}."
        ),
        "profiler_output_sha256": profiler_digest,
        "provider_evidence": {
            "schema_version": 1,
            "provider": "openvino",
            "requested_device": package.scope.openvino_device,
            "expected_execution_devices": list(
                package.scope.required_execution_devices
            ),
            "actual_execution_devices": first.runtime_evidence["bucket_results"][
                0
            ]["execution_devices"],
            "execution_devices_source": first.runtime_evidence["bucket_results"][
                0
            ]["execution_devices_source"],
            "expected_device_properties": dict(
                package.scope.required_openvino_properties
            ),
            "actual_device_properties": first.runtime_evidence["bucket_results"][
                0
            ]["device_properties"],
            "device_properties_source": first.runtime_evidence["bucket_results"][
                0
            ]["device_properties_source"],
        },
    }
    performance_row = {
        "bucket": bucket,
        "sample_count": sample_count,
        "benchmark_output_sha256": benchmark_digest,
        "latency_ms_p50": statistics.median(latency_values) / 1_000_000.0,
        "latency_ms_p95": _latency_percentile_ns(latency_values, 0.95)
        / 1_000_000.0,
        "peak_memory_bytes": max(rss_values),
        "energy_measurement": "not_measured",
        "energy_not_measured_reason": energy_not_measured_reason,
    }
    return sequence_row, placement_row, performance_row, first.runtime_evidence


def _wire_requests(inputs: Sequence[str], batch_size: int):
    return [inputs[start:start + batch_size] for start in range(0, len(inputs), batch_size)]


def _summarize_wire(package, inputs, batch_size, raw_root, transactions):
    input_digest = ordered_input_json_sha256(inputs)
    complete = b"".join(row.canonical for transaction in transactions for row in transaction.rows)
    response_count = sum(len(transaction.rows) for transaction in transactions)
    if len(transactions) != math.ceil(len(inputs) / batch_size) or response_count != len(inputs):
        raise EvidenceError("wire checkpoint has incomplete request or response coverage")
    signed_transactions_digest = _store_raw(raw_root, {
        "schema_version": 1,
        "kind": "wire-grouping-signed-transactions",
        "scope_id": package.scope.scope_id,
        "batch_size": batch_size,
        "ordered_input_json_sha256": input_digest,
        "transactions": [transaction.raw_document() for transaction in transactions],
    })
    return {
        "batch_size": batch_size,
        "input_count": SUPPORTED_MAX_BATCH_SIZE,
        "request_count": len(transactions),
        "response_row_count": response_count,
        "ordered_input_json_sha256": input_digest,
        "canonical_output_bytes_sha256": hashlib.sha256(complete).hexdigest(),
        "signed_transactions_sha256": signed_transactions_digest,
    }, complete


def _wire_grouping_results(
    dispatcher: Path,
    dispatcher_sha256: str,
    package: CandidatePackage,
    startup_timeout_seconds: float,
    request_timeout_seconds: float,
    inputs: Sequence[str],
    raw_root: Path,
    nonce_registry: set[bytes],
    checkpoint: Checkpoint | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    baseline: bytes | None = None
    with ExitStack() as stack:
        session = None
        for batch_size in range(1, SUPPORTED_MAX_BATCH_SIZE + 1):
            attempt = checkpoint.completed(f"wire-{batch_size}") if checkpoint else None
            requests = _wire_requests(inputs, batch_size)
            if attempt is not None:
                transactions = _restore_attempt(attempt, package.scope, requests, measure_rss=False)
            else:
                attempt = checkpoint.start(f"wire-{batch_size}") if checkpoint else None
                if session is None:
                    session = stack.enter_context(DispatcherSession(
                        dispatcher, dispatcher_sha256, package, startup_timeout_seconds))
                client = SignedAdapterClient(session, request_timeout_seconds, nonce_registry,
                                             checkpoint_attempt=attempt)
                transactions = [client.request(texts, measure_rss=False) for texts in requests]
            result, complete = _summarize_wire(package, inputs, batch_size, raw_root, transactions)
            if baseline is None:
                baseline = complete
            elif complete != baseline:
                raise EvidenceError(f"wire grouping size {batch_size} changed canonical output bytes")
            if attempt is not None:
                if attempt.completion is None:
                    attempt.complete({}, result)
                elif attempt.completion["metadata"] != {} or not _same_json_value(
                        attempt.completion["summary"], result):
                    raise EvidenceError("checkpoint wire summary differs from revalidated transactions")
            results.append(result)
    return results


def _checkpoint_identity(dispatcher, package, wire_inputs_path, wire_inputs,
                         startup_timeout_seconds, request_timeout_seconds,
                         warmup_count, sample_count, energy_not_measured_reason):
    directory = Path(__file__).resolve().parent
    implementations = ("physical_evidence.py", "physical_checkpoint.py", "admission_evidence.py",
                       "export_adapter_cache.py", "scifact_contract.py", "profile_identity.py",
                       "requirements-lock.txt")
    return {
        "schema_version": 1,
        "kind": "cfetch-physical-evidence-checkpoint-v1",
        "dispatcher_sha256": _file_sha256(dispatcher),
        "probe_package_manifest_sha256": _file_sha256(package.manifest_path),
        "runtime_manifest_sha256": package.runtime_manifest_sha256,
        "scope": dict(package.scope.document),
        "wire_inputs_file_sha256": _file_sha256(wire_inputs_path),
        "wire_inputs_sha256": ordered_input_json_sha256(wire_inputs),
        "sequence_inputs_sha256": {
            str(bucket): ordered_input_json_sha256(sequence_semantic_probe_inputs(bucket))
            for bucket in SEQUENCE_BUCKETS
        },
        "startup_timeout_seconds": startup_timeout_seconds,
        "request_timeout_seconds": request_timeout_seconds,
        "warmup_count": warmup_count,
        "sample_count": sample_count,
        "energy_not_measured_reason": energy_not_measured_reason,
        "implementation_sha256": {
            name: _file_sha256(_regular_file(directory / name, "collector implementation", MAX_MANIFEST_BYTES))
            for name in implementations
        },
    }


def _audit_checkpoint(checkpoint, package, wire_inputs, dispatcher_sha256,
                      warmup_count, sample_count, energy_not_measured_reason, raw_root):
    """Reject all corrupt history before launching any new dispatcher."""
    for attempt in checkpoint.attempts.values():
        kind, number = attempt.key.split("-")
        number = int(number)
        if attempt.key != f"{kind}-{number}" or (kind == "bucket" and number not in SEQUENCE_BUCKETS) or (
                kind == "wire" and not 1 <= number <= SUPPORTED_MAX_BATCH_SIZE):
            raise EvidenceError("checkpoint attempt is outside the frozen request plan")
        requests = (_bucket_requests(number, warmup_count, sample_count) if kind == "bucket"
                    else _wire_requests(wire_inputs, number))
        transactions = _restore_attempt(attempt, package.scope, requests, measure_rss=kind == "bucket")
        if attempt.completion is None:
            continue
        if kind == "bucket":
            summary = list(_summarize_bucket(dispatcher_sha256, package, warmup_count, sample_count,
                           energy_not_measured_reason, number, raw_root, transactions,
                           attempt.completion["metadata"]))
        else:
            if attempt.completion["metadata"] != {}:
                raise EvidenceError("checkpoint wire metadata schema changed")
            summary, _ = _summarize_wire(package, wire_inputs, number, raw_root, transactions)
        if not _same_json_value(attempt.completion["summary"], summary):
            raise EvidenceError("checkpoint summary differs from revalidated transactions")


def collect_physical_evidence(
    dispatcher: Path,
    dispatcher_sha256: str,
    package_manifest: Path,
    package_manifest_sha256: str,
    scope_id: str,
    wire_inputs_path: Path,
    output_directory: Path,
    startup_timeout_seconds: float,
    request_timeout_seconds: float,
    warmup_count: int,
    sample_count: int,
    energy_not_measured_reason: str,
    checkpoint_directory: Path | None = None,
) -> dict[str, Any]:
    if os.path.lexists(output_directory):
        raise EvidenceError("output directory must not already exist")
    if type(warmup_count) is not int or warmup_count < 1:
        raise EvidenceError("warmup count must be at least 1")
    if type(sample_count) is not int or sample_count < 20:
        raise EvidenceError("sample count must be at least 20 for a meaningful p95")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in (startup_timeout_seconds, request_timeout_seconds)):
        raise EvidenceError("timeouts must be positive and finite")
    if checkpoint_directory is not None:
        if 2 * len(SEQUENCE_BUCKETS) * (warmup_count + sample_count + 2) > MAX_FILES - 1000:
            raise EvidenceError("requested trial exceeds checkpoint record bound")
        checkpoint_path = Path(os.path.abspath(checkpoint_directory))
        output_path = Path(os.path.abspath(output_directory))
        if checkpoint_path.is_relative_to(output_path) or output_path.is_relative_to(checkpoint_path):
            raise EvidenceError("checkpoint and public output directories must be separate")
    if not energy_not_measured_reason.strip():
        raise EvidenceError("energy-not-measured reason must be nonempty")
    dispatcher = _regular_file(dispatcher, "candidate dispatcher")
    package = load_candidate_package(
        package_manifest, package_manifest_sha256, scope_id
    )
    if dispatcher.parent != package.manifest_path.parent:
        raise EvidenceError("dispatcher must be the sibling bound by the candidate package")
    if _file_sha256(dispatcher) != _digest(dispatcher_sha256, "dispatcher digest"):
        raise EvidenceError("candidate dispatcher digest does not match")
    wire_inputs = load_wire_inputs(wire_inputs_path)
    checkpoint_identity = (
        _checkpoint_identity(dispatcher, package, wire_inputs_path, wire_inputs,
                             startup_timeout_seconds, request_timeout_seconds,
                             warmup_count, sample_count, energy_not_measured_reason)
        if checkpoint_directory is not None else None
    )
    if checkpoint_identity is not None and (
            checkpoint_identity["dispatcher_sha256"] != dispatcher_sha256 or
            checkpoint_identity["probe_package_manifest_sha256"] != package.manifest_sha256):
        raise EvidenceError("checkpoint candidate identity changed during validation")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}-", dir=output_directory.parent
        )
    )
    try:
        with (Checkpoint(checkpoint_directory, checkpoint_identity)
              if checkpoint_directory is not None else nullcontext(None)) as checkpoint:
            raw_root = temporary / "raw"
            raw_root.mkdir()
            sequence_rows: list[dict[str, Any]] = []
            placement_rows: list[dict[str, Any]] = []
            performance_rows: list[dict[str, Any]] = []
            live_runtime_evidence: list[Mapping[str, Any]] = []
            nonce_registry: set[bytes] = set(checkpoint.nonces) if checkpoint else set()
            if checkpoint is not None:
                _audit_checkpoint(checkpoint, package, wire_inputs, dispatcher_sha256,
                                  warmup_count, sample_count, energy_not_measured_reason, raw_root)
            for bucket in SEQUENCE_BUCKETS:
                sequence, placement, performance, live_evidence = _run_bucket(
                    dispatcher,
                    dispatcher_sha256,
                    package,
                    startup_timeout_seconds,
                    request_timeout_seconds,
                    warmup_count,
                    sample_count,
                    energy_not_measured_reason,
                    bucket,
                    raw_root,
                    nonce_registry,
                    checkpoint,
                )
                sequence_rows.append(sequence)
                placement_rows.append(placement)
                performance_rows.append(performance)
                live_runtime_evidence.append(live_evidence)
            wire_results = _wire_grouping_results(
                dispatcher,
                dispatcher_sha256,
                package,
                startup_timeout_seconds,
                request_timeout_seconds,
                wire_inputs,
                raw_root,
                nonce_registry,
                checkpoint,
            )
            identity = dict(package.scope.identity)
            sequence_report = {
                **identity,
                "supported_max_tokens": MAX_TOKENS,
                "supported_sequence_buckets": list(SEQUENCE_BUCKETS),
                "supported_max_batch_size": SUPPORTED_MAX_BATCH_SIZE,
                "wire_batch_results": wire_results,
                "grouping_invariance": {
                    "batch_sizes": list(range(1, SUPPORTED_MAX_BATCH_SIZE + 1)),
                    "input_selection": WIRE_BATCH_INPUT_SELECTION,
                    "same_inputs_in_same_order": True,
                    "canonical_output_bytes_equal": True,
                },
                "bucket_results": sequence_rows,
            }
            placement_report = {
                **identity,
                "accelerated_placement": True,
                "accelerator_execution_confirmed": True,
                "fallback_disclosure_complete": True,
                "unexpected_fallback_detected": False,
                "provider_binding": {
                    "schema_version": 1,
                    "provider": "openvino",
                    "dispatcher_sha256": dispatcher_sha256,
                    "probe_package_manifest_sha256": package.manifest_sha256,
                    "runtime_manifest_sha256": package.runtime_manifest_sha256,
                    "openvino_compile_config": dict(
                        package.scope.document["openvino_compile_config"]
                    ),
                    "expected_host": dict(package.scope.required_host),
                    "actual_host": live_runtime_evidence[0]["host"],
                    "host_source": live_runtime_evidence[0]["host_source"],
                },
                "bucket_results": placement_rows,
            }
            performance_report = {**identity, "bucket_results": performance_rows}
            validate_evidence_reports(
                package.scope.document,
                sequence_report,
                placement_report,
                performance_report,
            )
            sequence_digest = _write_summary(
                temporary / "sequence-capability.json", sequence_report
            )
            placement_digest = _write_summary(
                temporary / "placement.json", placement_report
            )
            performance_digest = _write_summary(
                temporary / "performance.json", performance_report
            )
            if checkpoint_identity is not None and not _same_json_value(
                    checkpoint_identity,
                    _checkpoint_identity(dispatcher, package, wire_inputs_path, wire_inputs,
                                         startup_timeout_seconds, request_timeout_seconds,
                                         warmup_count, sample_count, energy_not_measured_reason)):
                raise EvidenceError("checkpoint experiment files changed during collection")
            _fsync_directory(raw_root)
            _fsync_directory(temporary)
            rename_new(temporary, output_directory)
            _fsync_directory(output_directory.parent)
            return {
                "schema_version": 1,
                "scope_id": scope_id,
                "sequence_capability_evidence": str(
                    output_directory / "sequence-capability.json"
                ),
                "sequence_capability_evidence_sha256": sequence_digest,
                "placement_evidence": str(output_directory / "placement.json"),
                "placement_evidence_sha256": placement_digest,
                "performance_evidence": str(output_directory / "performance.json"),
                "performance_evidence_sha256": performance_digest,
                "raw_measurements": str(output_directory / "raw"),
            }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dispatcher", required=True, type=Path)
    result.add_argument("--dispatcher-sha256", required=True)
    result.add_argument("--package-manifest", required=True, type=Path)
    result.add_argument("--package-manifest-sha256", required=True)
    result.add_argument("--scope-id", required=True)
    result.add_argument("--wire-inputs", required=True, type=Path)
    result.add_argument("--output-directory", required=True, type=Path)
    result.add_argument("--checkpoint-directory", type=Path,
                        help="Linux working journal; parent must exist; resumes only complete verified trials")
    result.add_argument("--startup-timeout-seconds", type=_positive_float, default=30.0)
    result.add_argument("--request-timeout-seconds", type=_positive_float, default=180.0)
    result.add_argument("--warmup-count", type=_positive_integer, default=2)
    result.add_argument("--sample-count", type=_positive_integer, default=20)
    result.add_argument(
        "--energy-not-measured-reason",
        required=True,
        help="exact reason no physical device-scoped energy meter was available",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = collect_physical_evidence(
            args.dispatcher,
            args.dispatcher_sha256,
            args.package_manifest,
            args.package_manifest_sha256,
            args.scope_id,
            args.wire_inputs,
            args.output_directory,
            args.startup_timeout_seconds,
            args.request_timeout_seconds,
            args.warmup_count,
            args.sample_count,
            args.energy_not_measured_reason,
            args.checkpoint_directory,
        )
    except (EvidenceError, CheckpointError, OSError, RuntimeError) as error:
        print(f"physical evidence collection refused: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
