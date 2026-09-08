#!/usr/bin/env python3
"""Strict loopback adapter for an ordered manifest-selected OpenVINO plan."""

from __future__ import annotations

import argparse
from collections.abc import Sequence as RuntimeSequence
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.metadata
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import platform
import re
import sys
import threading
from typing import Any, BinaryIO, Callable, Mapping, Protocol, Sequence

if __package__:
    from .inference_governor import from_installation
    from .native_deadline import NativeDeadline
    from .manifest import (
        ADMISSION_POLICY_SHA256,
        DIMENSIONS,
        MAX_TOKENS,
        MAX_WIRE_BATCH_SIZE,
        MODEL,
        MODEL_REVISION,
        PROFILE_ID,
        PROFILE_MANIFEST_SHA256,
        DEVICE_FOR_CLASS,
        HOST_FILE_PREFIXES,
        GOVERNOR_POLICY_PATH,
        REQUIRED_OPENVINO_PROPERTY_TYPES,
        SEQUENCE_BUCKETS,
        Artifact,
        ManifestError,
        PackageManifest,
        Scope,
        load_artifact,
        load_package_manifest,
        read_bounded_file,
    )
    from .runtime_bundle import (
        DISPATCHER,
        RuntimeBundleError,
        load_and_verify as load_runtime_bundle,
    )
    from .package_inventory import (
        INVENTORY_NAME,
        InventoryError,
        inventory_digest_from_environment,
        verify_bound as verify_package_inventory,
    )
else:  # Direct execution is the shipped sibling-executable form.
    from inference_governor import from_installation  # type: ignore[no-redef]
    from native_deadline import NativeDeadline  # type: ignore[no-redef]
    from manifest import (  # type: ignore[no-redef]
        ADMISSION_POLICY_SHA256,
        DIMENSIONS,
        MAX_TOKENS,
        MAX_WIRE_BATCH_SIZE,
        MODEL,
        MODEL_REVISION,
        PROFILE_ID,
        PROFILE_MANIFEST_SHA256,
        DEVICE_FOR_CLASS,
        HOST_FILE_PREFIXES,
        GOVERNOR_POLICY_PATH,
        REQUIRED_OPENVINO_PROPERTY_TYPES,
        SEQUENCE_BUCKETS,
        Artifact,
        ManifestError,
        PackageManifest,
        Scope,
        load_artifact,
        load_package_manifest,
        read_bounded_file,
    )
    from runtime_bundle import (  # type: ignore[no-redef]
        DISPATCHER,
        RuntimeBundleError,
        load_and_verify as load_runtime_bundle,
    )
    from package_inventory import (  # type: ignore[no-redef]
        INVENTORY_NAME,
        InventoryError,
        inventory_digest_from_environment,
        verify_bound as verify_package_inventory,
    )


ATTESTATION_NONCE_HEADER = "X-Cfetch-Attestation-Nonce"
ATTESTATION_SIGNATURE_HEADER = "X-Cfetch-Attestation-Signature"
ATTESTATION_DOMAIN = b"cfetch-embedding-response-attestation-v1\0"
MAX_AUTH_LINE_BYTES = 512
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
LOWER_HEX_32_RE = re.compile(r"[0-9a-f]{64}")
RUNTIME_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
MAX_PREFLIGHT_CONFIG_BYTES = 64 * 1024
MAX_PREFLIGHT_OUTPUT_BYTES = 1024 * 1024
PREFLIGHT_PURPOSE = "physical-probe-scope-config-input-not-admission-evidence"


def _normalize_execution_devices(raw: Any) -> tuple[str, ...]:
    """Normalize OpenVINO's scalar-or-sequence execution-device property."""

    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        raise TypeError("EXECUTION_DEVICES must not be bytes")
    elif isinstance(raw, RuntimeSequence):
        values = tuple(raw)
    else:
        raise TypeError("EXECUTION_DEVICES must be a string or sequence of strings")
    if not values:
        raise ValueError("EXECUTION_DEVICES must not be empty")
    if any(not isinstance(value, str) for value in values):
        raise TypeError("EXECUTION_DEVICES values must be strings")
    if any(not value for value in values):
        raise ValueError("EXECUTION_DEVICES values must not be empty")
    return values


class RequestError(ValueError):
    """An HTTP request cannot be interpreted as the frozen adapter contract."""


class ScopeUnavailableError(RuntimeError):
    """One exact local scope cannot initialize or execute on this device."""

    def __init__(self, scope_id: str) -> None:
        super().__init__(scope_id)
        self.scope_id = scope_id


class Tokenizer(Protocol):
    pad_token_id: int

    def encode(self, text: str) -> Sequence[int]: ...


class InferenceEngine(Protocol):
    def embed(
        self, input_ids: Sequence[int], attention_mask: Sequence[int], bucket: int
    ) -> Sequence[float]: ...

    def runtime_evidence(self, bucket: int) -> Mapping[str, Any]: ...

    def host_evidence(self) -> Mapping[str, Any]: ...


class Signer(Protocol):
    public_key_hex: str

    def sign(self, message: bytes) -> bytes: ...


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RequestError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _compact_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"adapter produced a non-JSON response: {error}") from error


def attestation_message(
    nonce: bytes, request_body: bytes, response_body: bytes
) -> bytes:
    if len(nonce) != 32:
        raise ValueError("attestation nonce must contain exactly 32 bytes")
    return b"".join(
        (
            ATTESTATION_DOMAIN,
            nonce,
            hashlib.sha256(request_body).digest(),
            hashlib.sha256(response_body).digest(),
        )
    )


def parse_auth_line(stream: BinaryIO) -> str:
    raw = stream.readline(MAX_AUTH_LINE_BYTES + 1)
    if not raw:
        raise RequestError("auth-stdin closed before the bearer credential arrived")
    if len(raw) > MAX_AUTH_LINE_BYTES or not raw.endswith(b"\n"):
        raise RequestError(
            "auth-stdin must provide one newline-terminated JSON line of at most "
            f"{MAX_AUTH_LINE_BYTES} bytes"
        )
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RequestError(f"auth-stdin did not contain valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != {"bearer"}:
        raise RequestError('auth-stdin must contain exactly {"bearer":"<32-byte-lowercase-hex>"}')
    bearer = value["bearer"]
    if not isinstance(bearer, str) or LOWER_HEX_32_RE.fullmatch(bearer) is None:
        raise RequestError("auth-stdin bearer must be exactly 64 lowercase hexadecimal characters")
    return bearer


def parse_embedding_request(
    raw: bytes, allowed_scope_ids: Mapping[str, Scope]
) -> tuple[list[str], Scope]:
    if not raw or len(raw) > MAX_REQUEST_BYTES:
        raise RequestError(f"request body must contain 1..{MAX_REQUEST_BYTES} bytes")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RequestError(f"request body is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise RequestError("request body must be a JSON object")
    required = {"model", "dimensions", "input", "cfetch_requested_scope_id"}
    if set(value) != required:
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - required)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise RequestError(
            "request fields do not match the local adapter contract: "
            + ", ".join(details)
        )
    if value["model"] != MODEL:
        raise RequestError(f"model must be {MODEL!r}")
    if value["dimensions"] != DIMENSIONS or type(value["dimensions"]) is not int:
        raise RequestError(f"dimensions must be the integer {DIMENSIONS}")
    requested_scope_id = value["cfetch_requested_scope_id"]
    if (
        not isinstance(requested_scope_id, str)
        or requested_scope_id not in allowed_scope_ids
    ):
        raise RequestError(
            "cfetch_requested_scope_id must name an exact scope in this target package"
        )
    inputs = value["input"]
    if (
        not isinstance(inputs, list)
        or not 1 <= len(inputs) <= MAX_WIRE_BATCH_SIZE
        or any(not isinstance(text, str) for text in inputs)
    ):
        raise RequestError(
            f"input must be an array containing 1..{MAX_WIRE_BATCH_SIZE} strings"
        )
    return inputs, allowed_scope_ids[requested_scope_id]


def smallest_bucket(token_count: int) -> int:
    for bucket in SEQUENCE_BUCKETS:
        if token_count <= bucket:
            return bucket
    raise RequestError(
        f"prefixed input contains {token_count} tokens; the profile limit is "
        f"{MAX_TOKENS} and truncation is forbidden"
    )


class EmbeddingService:
    def __init__(
        self,
        package: PackageManifest,
        tokenizer: Tokenizer,
        engine_factory: Callable[[PackageManifest, Scope], InferenceEngine],
        signers: Mapping[str, Signer],
    ) -> None:
        self.package = package
        self.tokenizer = tokenizer
        self.engine_factory = engine_factory
        self.signers = signers
        self._engines: dict[str, InferenceEngine] = {}
        self._unavailable: set[str] = set()

    def close(self) -> None:
        """Release native engines before CPython begins interpreter teardown."""

        engines = tuple(self._engines.values())
        self._engines.clear()
        for engine in engines:
            close = getattr(engine, "close", None)
            if callable(close):
                close()

    def _engine_for(self, scope: Scope) -> InferenceEngine:
        if scope.scope_id in self._unavailable:
            raise ScopeUnavailableError(scope.scope_id)
        engine = self._engines.get(scope.scope_id)
        if engine is not None:
            return engine
        try:
            engine = self.engine_factory(self.package, scope)
        except Exception as error:
            self._unavailable.add(scope.scope_id)
            print(
                f"OpenVINO scope {scope.scope_id} failed initialization: {error}",
                file=sys.stderr,
                flush=True,
            )
            raise ScopeUnavailableError(scope.scope_id) from error
        self._engines[scope.scope_id] = engine
        return engine

    def response_for(self, request_body: bytes) -> tuple[bytes, Signer]:
        texts, scope = parse_embedding_request(request_body, self.package.scopes)
        prepared: list[tuple[int, int, list[int], list[int]]] = []
        for text in texts:
            ids = list(self.tokenizer.encode(text))
            if not ids or any(type(item) is not int or item < 0 for item in ids):
                raise RuntimeError("tokenizer returned an invalid token sequence")
            token_count = len(ids)
            bucket = smallest_bucket(token_count)
            padding = bucket - token_count
            input_ids = ids + [self.tokenizer.pad_token_id] * padding
            attention_mask = [1] * token_count + [0] * padding
            prepared.append((token_count, bucket, input_ids, attention_mask))
        engine = self._engine_for(scope)
        rows: list[dict[str, Any]] = []
        bucket_evidence: dict[int, Mapping[str, Any]] = {}
        for index, (token_count, bucket, input_ids, attention_mask) in enumerate(
            prepared
        ):
            try:
                output = engine.embed(input_ids, attention_mask, bucket)
                vector = [float(component) for component in output]
                if len(vector) != DIMENSIONS:
                    raise RuntimeError(
                        f"OpenVINO graph returned {len(vector)} components, expected {DIMENSIONS}"
                    )
                if not all(math.isfinite(component) for component in vector):
                    raise RuntimeError("OpenVINO graph returned a non-finite embedding")
                if not any(component != 0.0 for component in vector):
                    raise RuntimeError("OpenVINO graph returned an all-zero embedding")
            except Exception as error:
                self._engines.pop(scope.scope_id, None)
                self._unavailable.add(scope.scope_id)
                print(
                    f"OpenVINO scope {scope.scope_id} failed execution: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                raise ScopeUnavailableError(scope.scope_id) from error
            rows.append(
                {
                    "index": index,
                    "cfetch_scope_id": scope.scope_id,
                    "token_count": token_count,
                    "sequence_bucket": bucket,
                    "truncated": False,
                    "embedding": vector,
                }
            )
            bucket_evidence[bucket] = engine.runtime_evidence(bucket)
        response = {
            "model": MODEL,
            "cfetch_profile": PROFILE_ID,
            "cfetch_profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
            "cfetch_admission_policy_sha256": ADMISSION_POLICY_SHA256,
            "cfetch_model_revision": MODEL_REVISION,
            "cfetch_execution": scope.execution_document(),
            "cfetch_runtime_evidence": {
                "schema_version": 1,
                "provider": "openvino",
                "scope_id": scope.scope_id,
                "host": engine.host_evidence(),
                "host_source": "platform-and-sha256",
                "bucket_results": [
                    bucket_evidence[bucket] for bucket in sorted(bucket_evidence)
                ],
            },
            "data": rows,
        }
        raw = _compact_json(response)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise RuntimeError(
                f"adapter response exceeds the {MAX_RESPONSE_BYTES}-byte safety limit"
            )
        try:
            signer = self.signers[scope.scope_id]
        except KeyError as error:
            raise RuntimeError("selected scope has no package-bound signer") from error
        return raw, signer


class HuggingFaceTokenizer:
    def __init__(self, artifact: Artifact) -> None:
        from tokenizers import Tokenizer as RuntimeTokenizer

        tokenizer = RuntimeTokenizer.from_file(str(artifact.tokenizer_json))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        expected_tokens = {
            "<pad>": artifact.pad_token_id,
            "<bos>": artifact.bos_token_id,
            "<eos>": artifact.eos_token_id,
        }
        for token, expected_id in expected_tokens.items():
            if tokenizer.token_to_id(token) != expected_id:
                raise RuntimeError(
                    f"frozen tokenizer does not map {token} to token ID {expected_id}"
                )
        self._tokenizer = tokenizer
        self.pad_token_id = artifact.pad_token_id
        self._bos_token_id = artifact.bos_token_id
        self._eos_token_id = artifact.eos_token_id

    def encode(self, text: str) -> Sequence[int]:
        # Apply the frozen tokenizer_config explicitly instead of depending on
        # a runtime's reconstruction of GemmaTokenizer's post-processor.  The
        # query/document prompt is already part of text and is not added here.
        pieces = self._tokenizer.encode(text, add_special_tokens=False).ids
        return [self._bos_token_id, *pieces, self._eos_token_id]


class OpenVinoEngine:
    def __init__(self, package: PackageManifest, scope: Scope) -> None:
        # Refuse an unprovisioned or faulted host before loading native code.
        self._governor = from_installation()
        if not any(
            file.path == Path(GOVERNOR_POLICY_PATH) and file.sha256 == self._governor.policy_sha256
            for file in scope.required_host.files
        ):
            raise RuntimeError("admitted scope does not bind the installed inference governor policy")
        import numpy as np
        import openvino as ov

        self._np = np
        artifact = package.artifact
        self._host_evidence = validate_host_binding(scope)
        core = ov.Core()
        self._openvino_properties = validate_openvino_properties(core, scope)
        self._model = core.read_model(
            model=str(artifact.graph_xml), weights=str(artifact.graph_bin)
        )
        self._core = core
        self._compiled: dict[int, Any] = {}
        self._execution_devices: dict[int, tuple[str, ...]] = {}
        self._artifact = artifact
        self._scope = scope

    def close(self) -> None:
        """Destroy compiled models and the plugin core in dependency order."""

        self._compiled.clear()
        self._execution_devices.clear()
        self._model = None
        self._core = None

    def _compiled_bucket(self, bucket: int):
        compiled = self._compiled.get(bucket)
        if compiled is not None:
            return compiled
        if bucket not in SEQUENCE_BUCKETS:
            raise RuntimeError(f"OpenVINO bucket {bucket} is not in the frozen profile")
        bucket_model = self._model.clone()
        bucket_model.reshape(
            {
                self._artifact.input_ids_name: [1, bucket],
                self._artifact.attention_mask_name: [1, bucket],
            }
        )
        # Every compiled request shape is static.  Device selection remains
        # the exact manifest value; AUTO/MULTI/HETERO never enter this call.
        with self._governor.operation("compile", bucket) as lease, NativeDeadline(lease.deadline_ns):
            compiled = self._core.compile_model(
                bucket_model,
                self._scope.openvino_device,
                dict(self._scope.openvino_compile_config),
            )
        try:
            execution_devices = _normalize_execution_devices(
                compiled.get_property("EXECUTION_DEVICES")
            )
        except Exception as error:
            raise RuntimeError(
                "compiled OpenVINO model did not expose EXECUTION_DEVICES"
            ) from error
        if execution_devices != self._scope.required_execution_devices:
            raise RuntimeError(
                "compiled OpenVINO model execution devices did not match the admitted scope"
            )
        self._execution_devices[bucket] = execution_devices
        self._compiled[bucket] = compiled
        return compiled

    def embed(
        self, input_ids: Sequence[int], attention_mask: Sequence[int], bucket: int
    ) -> Sequence[float]:
        if len(input_ids) != bucket or len(attention_mask) != bucket:
            raise RuntimeError("OpenVINO input does not match its static sequence bucket")
        compiled = self._compiled_bucket(bucket)
        inputs = {
            self._artifact.input_ids_name: self._np.asarray([input_ids], dtype=self._np.int64),
            self._artifact.attention_mask_name: self._np.asarray([attention_mask], dtype=self._np.int64),
        }
        # One lease per actual invocation, after final padding. A wire batch
        # never pays for only one of its native calls.
        with self._governor.operation("inference", bucket) as lease, NativeDeadline(lease.deadline_ns):
            result = compiled(inputs)
        output = result[compiled.output(self._artifact.output_name)]
        array = self._np.asarray(output)
        if array.shape != (1, DIMENSIONS):
            raise RuntimeError(
                f"OpenVINO graph returned shape {array.shape}, expected (1, {DIMENSIONS})"
            )
        return array[0].astype(self._np.float32, copy=False).tolist()

    def runtime_evidence(self, bucket: int) -> Mapping[str, Any]:
        try:
            execution_devices = self._execution_devices[bucket]
        except KeyError as error:
            raise RuntimeError(
                "runtime evidence was requested before the static bucket executed"
            ) from error
        return {
            "bucket": bucket,
            "requested_device": self._scope.openvino_device,
            "execution_devices": list(execution_devices),
            "execution_devices_source": "compiled_model.get_property(EXECUTION_DEVICES)",
            "device_properties": self._openvino_properties,
            "device_properties_source": "core.get_property",
        }

    def host_evidence(self) -> Mapping[str, Any]:
        return self._host_evidence


def validate_openvino_properties(core: Any, scope: Scope) -> dict[str, str | int]:
    """Bind a scope to the exact physical/runtime properties it admitted."""

    property_device = scope.required_execution_devices[0]
    try:
        supported = {
            str(name)
            for name in core.get_property(
                property_device, "SUPPORTED_PROPERTIES"
            )
        }
    except Exception as error:
        raise RuntimeError(
            f"OpenVINO device {property_device} did not expose "
            "SUPPORTED_PROPERTIES"
        ) from error
    actual_properties: dict[str, str | int] = {}
    for property_name, expected in scope.required_openvino_properties.items():
        if property_name not in supported:
            raise RuntimeError(
                f"OpenVINO device {property_device} does not support required "
                f"property {property_name}"
            )
        try:
            actual = core.get_property(property_device, property_name)
        except Exception as error:
            raise RuntimeError(
                f"OpenVINO device {property_device} did not expose required "
                f"property {property_name}"
            ) from error
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError(
                f"OpenVINO device property {property_name} did not match the "
                f"admitted {scope.scope_id} scope"
            )
        actual_properties[property_name] = actual
    return actual_properties


def parse_preflight_compile_config(raw: str) -> dict[str, str | int | float | bool]:
    try:
        encoded = raw.encode("utf-8")
    except UnicodeEncodeError as error:
        raise RuntimeError(
            "host-preflight compile config is not valid UTF-8 text"
        ) from error
    if not encoded or len(encoded) > MAX_PREFLIGHT_CONFIG_BYTES:
        raise RuntimeError(
            "host-preflight compile config must contain 1..65536 UTF-8 bytes"
        )
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, RequestError) as error:
        raise RuntimeError(
            f"host-preflight compile config is not strict JSON: {error}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError("host-preflight compile config must be a JSON object")
    result: dict[str, str | int | float | bool] = {}
    for key in sorted(value):
        item = value[key]
        if not isinstance(key, str) or not key or len(key) > 128:
            raise RuntimeError(
                "host-preflight compile config keys must be 1..128 character strings"
            )
        if type(item) not in {str, int, float, bool}:
            raise RuntimeError(
                f"host-preflight compile config {key} is not a JSON primitive"
            )
        if type(item) is float and not math.isfinite(item):
            raise RuntimeError(
                f"host-preflight compile config {key} is not finite"
            )
        result[key] = item
    canonical = json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if raw != canonical:
        raise RuntimeError(
            "host-preflight compile config must be canonical compact sorted JSON"
        )
    return result


def _host_file_sha256(path: Path) -> str:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise RuntimeError(f"required host file path contains a symlink: {path}")
    try:
        metadata = path.stat()
    except OSError as error:
        raise RuntimeError(f"cannot inspect required host file {path}") from error
    if not path.is_file() or metadata.st_size < 1 or metadata.st_size > 1024**3:
        raise RuntimeError(
            f"required host file must be a 1..1073741824-byte regular file: {path}"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise RuntimeError(f"cannot read required host file {path}") from error
    return digest.hexdigest()


def _preflight_host_binding(host_files: Sequence[Path]) -> dict[str, Any]:
    if not 1 <= len(host_files) <= 16:
        raise RuntimeError("host-preflight requires 1..16 explicit host files")
    system = platform.system()
    machine = platform.machine()
    kernel_release = platform.release()
    if system != "Linux" or machine != "x86_64":
        raise RuntimeError("host-preflight requires exactly Linux x86_64")
    if (
        not kernel_release
        or len(kernel_release) > 256
        or any(character in kernel_release for character in ("\x00", "\r", "\n"))
    ):
        raise RuntimeError("host-preflight kernel release is invalid")
    normalized: list[Path] = []
    seen: set[str] = set()
    for path in host_files:
        text = str(path)
        pure = PurePosixPath(text)
        if (
            not pure.is_absolute()
            or str(pure) != text
            or len(text) > 512
            or any(part in ("", ".", "..") for part in pure.parts)
            or (pure != GOVERNOR_POLICY_PATH
                and not any(pure.is_relative_to(prefix) for prefix in HOST_FILE_PREFIXES))
        ):
            raise RuntimeError(
                "host-preflight host files must be normalized absolute paths under "
                "an allowlisted system library directory"
            )
        if text in seen:
            raise RuntimeError("host-preflight host files must be distinct")
        seen.add(text)
        normalized.append(Path(text))
    normalized.sort(key=str)
    return {
        "system": system,
        "machine": machine,
        "kernel_release": kernel_release,
        "files": [
            {"path": str(path), "sha256": _host_file_sha256(path)}
            for path in normalized
        ],
    }


def _preflight_openvino_properties(
    core: Any, device_class: str, property_device: str
) -> dict[str, str | int]:
    try:
        supported = {
            str(name)
            for name in core.get_property(property_device, "SUPPORTED_PROPERTIES")
        }
    except Exception as error:
        raise RuntimeError(
            f"host-preflight {property_device} did not expose SUPPORTED_PROPERTIES"
        ) from error
    properties: dict[str, str | int] = {}
    for name, expected_type in REQUIRED_OPENVINO_PROPERTY_TYPES[device_class].items():
        if name not in supported:
            raise RuntimeError(
                f"host-preflight {property_device} does not support required property {name}"
            )
        try:
            actual = core.get_property(property_device, name)
        except Exception as error:
            raise RuntimeError(
                f"host-preflight {property_device} did not expose required property {name}"
            ) from error
        if type(actual) is not expected_type:
            raise RuntimeError(
                f"host-preflight property {name} has type {type(actual).__name__}, "
                f"expected {expected_type.__name__}"
            )
        if expected_type is str:
            if (
                not actual
                or len(actual) > 4096
                or any(character in actual for character in ("\x00", "\r", "\n"))
            ):
                raise RuntimeError(f"host-preflight property {name} is invalid")
        elif not -(1 << 63) <= actual < (1 << 64):
            raise RuntimeError(
                f"host-preflight property {name} is outside the integer range"
            )
        properties[name] = actual
    return properties


def collect_host_preflight(
    artifact: Artifact,
    runtime: Mapping[str, Any],
    device_class: str,
    openvino_device: str,
    compile_config: Mapping[str, str | int | float | bool],
    host_files: Sequence[Path],
    *,
    core: Any | None = None,
) -> dict[str, Any]:
    """Compile every bucket with the frozen runtime and emit manifest inputs only."""

    if device_class not in DEVICE_FOR_CLASS:
        raise RuntimeError("host-preflight device class must be npu, gpu, or cpu")
    if openvino_device != DEVICE_FOR_CLASS[device_class]:
        raise RuntimeError(
            "host-preflight device must exactly match its NPU/GPU/CPU class"
        )
    runtime_digest = runtime.get("runtime_manifest_sha256")
    dependencies = runtime.get("dependency_versions")
    if (
        not isinstance(runtime_digest, str)
        or LOWER_HEX_32_RE.fullmatch(runtime_digest) is None
        or not isinstance(dependencies, dict)
        or set(dependencies) != {"cryptography", "numpy", "openvino", "tokenizers"}
        or any(not isinstance(value, str) or not value for value in dependencies.values())
    ):
        raise RuntimeError("host-preflight runtime identity is incomplete")
    if artifact.conversion_versions.get("openvino") != dependencies["openvino"]:
        raise RuntimeError(
            "host-preflight artifact and frozen runtime use different OpenVINO versions"
        )
    governor = from_installation()
    # Existing host-file evidence binds the exact safety policy alongside
    # driver/runtime files. No separate producer receipt or wire field.
    selected_host_files = tuple(host_files)
    if Path(GOVERNOR_POLICY_PATH) not in selected_host_files:
        selected_host_files += (Path(GOVERNOR_POLICY_PATH),)
    host_before = _preflight_host_binding(selected_host_files)
    if not any(file["path"] == str(GOVERNOR_POLICY_PATH) and file["sha256"] == governor.policy_sha256
               for file in host_before["files"]):
        raise RuntimeError("host-preflight inference governor policy changed")
    if core is None:
        import openvino as ov

        core = ov.Core()
    try:
        available_devices = [str(device) for device in core.available_devices]
    except Exception as error:
        raise RuntimeError("host-preflight could not enumerate OpenVINO devices") from error
    if not any(
        re.fullmatch(rf"{re.escape(openvino_device)}(?:\.[0-9]+)?", device)
        is not None
        for device in available_devices
    ):
        raise RuntimeError(
            f"host-preflight frozen OpenVINO does not expose a physical {openvino_device}"
        )
    try:
        model = core.read_model(
            model=str(artifact.graph_xml), weights=str(artifact.graph_bin)
        )
    except Exception as error:
        raise RuntimeError("host-preflight could not read the verified OpenVINO graph") from error
    required_execution_devices: tuple[str, ...] | None = None
    properties_before: dict[str, str | int] | None = None
    bucket_results: list[dict[str, Any]] = []
    for bucket in SEQUENCE_BUCKETS:
        try:
            bucket_model = model.clone()
            bucket_model.reshape(
                {
                    artifact.input_ids_name: [1, bucket],
                    artifact.attention_mask_name: [1, bucket],
                }
            )
            with governor.operation("compile", bucket) as lease, NativeDeadline(lease.deadline_ns):
                compiled = core.compile_model(
                    bucket_model,
                    openvino_device,
                    dict(compile_config),
                )
            execution_devices = _normalize_execution_devices(
                compiled.get_property("EXECUTION_DEVICES")
            )
        except Exception as error:
            raise RuntimeError(
                f"host-preflight could not compile bucket {bucket} or query EXECUTION_DEVICES"
            ) from error
        if (
            len(execution_devices) != 1
            or re.fullmatch(
                rf"{re.escape(openvino_device)}(?:\.[0-9]+)?",
                execution_devices[0],
            )
            is None
        ):
            raise RuntimeError(
                f"host-preflight bucket {bucket} did not execute on one exact "
                f"{openvino_device} device"
            )
        if required_execution_devices is None:
            required_execution_devices = execution_devices
            properties_before = _preflight_openvino_properties(
                core, device_class, execution_devices[0]
            )
        elif execution_devices != required_execution_devices:
            raise RuntimeError(
                "host-preflight EXECUTION_DEVICES changed between static buckets"
            )
        bucket_results.append(
            {"bucket": bucket, "execution_devices": list(execution_devices)}
        )
    if required_execution_devices is None:  # pragma: no cover - frozen tuple is nonempty
        raise RuntimeError("host-preflight compiled no sequence buckets")
    if properties_before is None:  # pragma: no cover - first bucket sets both values
        raise RuntimeError("host-preflight queried no physical device properties")
    properties_after = _preflight_openvino_properties(
        core, device_class, required_execution_devices[0]
    )
    if properties_after != properties_before:
        raise RuntimeError(
            "host-preflight physical device properties changed during compilation"
        )
    host_after = _preflight_host_binding(selected_host_files)
    if host_after != host_before:
        raise RuntimeError("host-preflight host binding changed during compilation")
    return {
        "schema_version": 1,
        "purpose": PREFLIGHT_PURPOSE,
        "runtime_manifest_sha256": runtime_digest,
        "dependency_versions": dict(sorted(dependencies.items())),
        "artifact_manifest_sha256": artifact.manifest_sha256,
        "device_class": device_class,
        "openvino_device": openvino_device,
        "openvino_property_device": required_execution_devices[0],
        "openvino_compile_config": dict(sorted(compile_config.items())),
        "required_openvino_properties": properties_before,
        "required_execution_devices": list(required_execution_devices),
        "required_host": host_before,
        "host_binding_source": (
            "operator-selected-paths-sha256-before-and-after-compilation"
        ),
        "device_properties_source": "core.get_property",
        "execution_devices_source": "compiled_model.get_property(EXECUTION_DEVICES)",
        "bucket_results": bucket_results,
    }


def canonical_preflight_output(document: Mapping[str, Any]) -> bytes:
    try:
        raw = (
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"host-preflight produced invalid JSON: {error}") from error
    if not 1 <= len(raw) <= MAX_PREFLIGHT_OUTPUT_BYTES:
        raise RuntimeError("host-preflight output exceeds its byte bound")
    return raw


def validate_host_binding(scope: Scope) -> dict[str, Any]:
    binding = scope.required_host
    actual = (platform.system(), platform.machine(), platform.release())
    expected = (binding.system, binding.machine, binding.kernel_release)
    if actual != expected:
        raise RuntimeError(
            f"host identity did not match the admitted {scope.scope_id} scope"
        )
    actual_files: list[dict[str, Any]] = []
    for file_binding in binding.files:
        actual_sha256 = _host_file_sha256(file_binding.path)
        if not hmac.compare_digest(actual_sha256, file_binding.sha256):
            raise RuntimeError(
                f"host file {file_binding.path} did not match the admitted "
                f"{scope.scope_id} scope"
            )
        actual_files.append(
            {"path": str(file_binding.path), "sha256": actual_sha256}
        )
    return {
        "system": actual[0],
        "machine": actual[1],
        "kernel_release": actual[2],
        "files": actual_files,
    }


class Ed25519Signer:
    def __init__(self, private_key_file: Path, expected_public_key_hex: str) -> None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        raw = read_bounded_file(
            private_key_file, 65, "attestation private key file"
        )
        if raw.endswith(b"\n"):
            raw = raw[:-1]
        if re.fullmatch(rb"[0-9a-f]{64}", raw) is None:
            raise RuntimeError(
                "attestation private key file must contain exactly 64 lowercase "
                "hexadecimal characters and an optional newline"
            )
        private_key = Ed25519PrivateKey.from_private_bytes(
            bytes.fromhex(raw.decode("ascii"))
        )
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.public_key_hex = public_key.hex()
        if not hmac.compare_digest(self.public_key_hex, expected_public_key_hex):
            raise RuntimeError(
                "attestation private key does not match the public key pinned for this scope"
            )
        self._private_key = private_key

    def sign(self, message: bytes) -> bytes:
        return self._private_key.sign(message)


def verify_dependency_versions(package: PackageManifest) -> None:
    for distribution, expected in package.dependency_versions.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"target package dependency {distribution}=={expected} is not installed"
            ) from error
        if actual != expected:
            raise RuntimeError(
                f"target package requires {distribution}=={expected}, found {actual}"
            )


def package_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def package_bound_files(package: PackageManifest) -> list[str]:
    root = package.path.parent.resolve()
    files = {
        package.path.resolve(),
        package.artifact.manifest_path.resolve(),
        *(path.resolve() for path in package.artifact.files),
        *(scope.attestation_private_key_file.resolve() for scope in package.scopes.values()),
        (root / INVENTORY_NAME).resolve(),
    }
    try:
        return sorted(path.relative_to(root).as_posix() for path in files)
    except ValueError as error:
        raise RuntimeError("package-bound file escapes the dispatcher directory") from error


def runtime_self_check(
    expected_manifest_sha256: str | None = None,
    allowed_package_files: Sequence[str] = (),
) -> dict[str, Any]:
    path = package_root() / "runtime-manifest.json"
    document = load_runtime_bundle(
        package_root(), expected_manifest_sha256, allowed_package_files
    )
    raw = read_bounded_file(path, RUNTIME_MANIFEST_MAX_BYTES, "runtime manifest")
    if (
        document.get("schema_version") != 1
        or document.get("target") != "linux-x86_64-glibc"
        or document.get("python_abi") != "cp312"
        or document.get("dispatcher") != DISPATCHER
    ):
        raise RuntimeError("runtime manifest target identity is invalid")
    if (
        sys.platform != "linux"
        or platform.machine() not in ("x86_64", "AMD64")
        or sys.implementation.name != "cpython"
        or sys.version_info[:2] != (3, 12)
    ):
        raise RuntimeError("frozen dispatcher is not running on Linux x86_64 CPython 3.12")
    minimum_glibc = document.get("minimum_glibc")
    libc_name, libc_version = platform.libc_ver()
    try:
        minimum_tuple = tuple(int(piece) for piece in minimum_glibc.split("."))
        actual_tuple = tuple(int(piece) for piece in libc_version.split("."))
    except (AttributeError, ValueError) as error:
        raise RuntimeError("runtime manifest or host glibc version is invalid") from error
    if libc_name != "glibc" or actual_tuple < minimum_tuple:
        raise RuntimeError(
            f"frozen dispatcher requires glibc>={minimum_glibc}, found {libc_name} {libc_version}"
        )
    expected_versions = document.get("dependency_versions")
    if not isinstance(expected_versions, dict) or set(expected_versions) != {
        "cryptography",
        "numpy",
        "openvino",
        "tokenizers",
    }:
        raise RuntimeError("runtime manifest dependency versions are incomplete")
    actual_versions: dict[str, str] = {}
    for distribution, expected in expected_versions.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"frozen runtime dependency {distribution} is not installed"
            ) from error
        if actual != expected:
            raise RuntimeError(
                f"frozen runtime requires {distribution}=={expected}, found {actual}"
            )
        actual_versions[distribution] = actual

    import cryptography
    import numpy as np
    import openvino as ov
    import tokenizers
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    del cryptography, tokenizers
    if np.asarray([1], dtype=np.int64).tolist() != [1]:
        raise RuntimeError("bundled NumPy failed its integer-array self-check")
    key = Ed25519PrivateKey.generate()
    message = b"cfetch-openvino-runtime-self-check"
    key.public_key().verify(key.sign(message), message)
    devices = list(ov.Core().available_devices)
    if "CPU" not in devices:
        raise RuntimeError("bundled OpenVINO runtime did not expose its CPU plugin")
    return {
        "schema_version": 1,
        "target": "linux-x86_64-glibc",
        "runtime_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "dependency_versions": actual_versions,
        "openvino_devices": devices,
    }


class AdapterServer(HTTPServer):
    allow_reuse_address = False

    def __init__(
        self,
        address: tuple[str, int],
        service: EmbeddingService,
        bearer: str,
    ) -> None:
        self.service = service
        self.bearer = bearer
        super().__init__(address, AdapterHandler, bind_and_activate=True)


class AdapterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "cfetch-openvino-adapter/1"
    sys_version = ""

    @property
    def adapter_server(self) -> AdapterServer:
        return self.server  # type: ignore[return-value]

    def _headers(self, name: str) -> list[str]:
        return self.headers.get_all(name, failobj=[])

    def _error(self, status: int, message: str) -> None:
        raw = _compact_json({"error": message})
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            if self.path != "/v1/embeddings":
                self._error(404, "not found")
                return
            auth_values = self._headers("Authorization")
            expected_auth = f"Bearer {self.adapter_server.bearer}"
            if len(auth_values) != 1 or not hmac.compare_digest(
                auth_values[0], expected_auth
            ):
                self._error(401, "unauthorized")
                return
            if self._headers("Transfer-Encoding"):
                raise RequestError("chunked or transformed request bodies are not accepted")
            lengths = self._headers("Content-Length")
            if len(lengths) != 1:
                raise RequestError("exactly one Content-Length header is required")
            try:
                length = int(lengths[0], 10)
            except ValueError as error:
                raise RequestError("Content-Length must be a decimal integer") from error
            if not 1 <= length <= MAX_REQUEST_BYTES:
                raise RequestError(
                    f"Content-Length must be in 1..{MAX_REQUEST_BYTES}"
                )
            content_types = self._headers("Content-Type")
            if (
                len(content_types) != 1
                or content_types[0].split(";", 1)[0].strip().lower()
                != "application/json"
            ):
                raise RequestError("Content-Type must be application/json")
            nonce_values = self._headers(ATTESTATION_NONCE_HEADER)
            if (
                len(nonce_values) != 1
                or LOWER_HEX_32_RE.fullmatch(nonce_values[0]) is None
            ):
                raise RequestError(
                    f"{ATTESTATION_NONCE_HEADER} must appear once as 64 lowercase "
                    "hexadecimal characters"
                )
            request_body = self.rfile.read(length)
            if len(request_body) != length:
                raise RequestError("request body ended before Content-Length bytes arrived")
            response_body, signer = self.adapter_server.service.response_for(request_body)
            signature = signer.sign(
                attestation_message(
                    bytes.fromhex(nonce_values[0]), request_body, response_body
                )
            )
            if len(signature) != 64:
                raise RuntimeError("Ed25519 signer did not return a 64-byte signature")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header(ATTESTATION_SIGNATURE_HEADER, signature.hex())
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(response_body)
            self.close_connection = True
        except RequestError as error:
            self._error(400, str(error))
        except ScopeUnavailableError as error:
            raw = _compact_json(
                {
                    "error": {
                        "code": "scope_unavailable",
                        "scope_id": error.scope_id,
                        "message": "requested admitted scope could not initialize or execute",
                    }
                }
            )
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
            self.close_connection = True
        except Exception as error:  # Keep implementation details off the wire.
            print(f"embedding request failed: {error}", file=sys.stderr, flush=True)
            self._error(500, "embedding inference failed")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._error(404, "not found")

    def log_message(self, format: str, *args: object) -> None:
        print(
            f"{self.address_string()} - {format % args}",
            file=sys.stderr,
            flush=True,
        )


def readiness_document(server: AdapterServer, package: PackageManifest) -> dict[str, Any]:
    host, port = server.server_address[:2]
    if host != "127.0.0.1" or not isinstance(port, int) or port <= 0:
        raise RuntimeError("adapter did not bind an ephemeral IPv4 loopback port")
    return {
        "schema_version": 1,
        "url": f"http://127.0.0.1:{port}/v1",
        "scope_ids": list(package.scopes),
    }


def _shutdown_on_parent_eof(stream: BinaryIO, server: AdapterServer) -> None:
    try:
        while stream.read(4096):
            pass
    finally:
        server.shutdown()


def serve(
    package: PackageManifest,
    bearer: str,
    auth_stream: BinaryIO,
    tokenizer: Tokenizer,
    engine_factory: Callable[[PackageManifest, Scope], InferenceEngine],
    signers: Mapping[str, Signer],
) -> None:
    service = EmbeddingService(package, tokenizer, engine_factory, signers)
    server = AdapterServer(
        ("127.0.0.1", 0),
        service,
        bearer,
    )
    monitor = threading.Thread(
        target=_shutdown_on_parent_eof,
        args=(auth_stream, server),
        name="cfetch-parent-lifetime",
        daemon=True,
    )
    monitor.start()
    # Stdout is a machine channel: exactly this one bounded readiness line.
    sys.stdout.buffer.write(_compact_json(readiness_document(server, package)) + b"\n")
    sys.stdout.buffer.flush()
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        monitor.join(timeout=5)
        if monitor.is_alive():
            raise RuntimeError("parent-lifetime monitor did not finish after EOF")
        service.close()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subcommands = root.add_subparsers(dest="command", required=True)
    serve_command = subcommands.add_parser(
        "serve", help="serve the package's ordered exact OpenVINO scopes"
    )
    serve_command.add_argument("--host", required=True)
    serve_command.add_argument("--port", required=True, type=int)
    serve_command.add_argument("--auth-stdin", action="store_true", required=True)
    subcommands.add_parser(
        "runtime-check", help="verify the frozen Python and OpenVINO runtime payload"
    )
    preflight = subcommands.add_parser(
        "host-preflight",
        help="emit frozen-runtime inputs for one later physical-probe scope",
    )
    preflight.add_argument(
        "--runtime-manifest-sha256",
        required=True,
        help="externally pinned SHA-256 of the raw frozen runtime manifest",
    )
    preflight.add_argument("--artifact-dir", required=True, type=Path)
    preflight.add_argument("--artifact-manifest-sha256", required=True)
    preflight.add_argument(
        "--device-class", required=True, choices=tuple(DEVICE_FOR_CLASS)
    )
    preflight.add_argument(
        "--device", required=True, choices=tuple(DEVICE_FOR_CLASS.values())
    )
    preflight.add_argument(
        "--compile-config-json",
        required=True,
        help="canonical compact sorted JSON object, for example {}",
    )
    preflight.add_argument(
        "--host-file",
        action="append",
        required=True,
        type=Path,
        help="repeat for 1..16 explicit driver/library files",
    )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    # `runtime-check` is the build-only dependency/inventory check. Serving
    # and physical preflight inspect host safety state before native imports.
    if args.command in {"serve", "host-preflight"}:
        try:
            from_installation()
        except (RuntimeError, OSError) as error:
            print(f"cfetch OpenVINO host governor refused startup: {error}", file=sys.stderr)
            return 1
    has_package_inventory = os.environ.get("CFETCH_PACKAGE_INVENTORY_SHA256") is not None
    raw_runtime: dict[str, Any] | None = None
    if has_package_inventory:
        try:
            inventory_sha256 = inventory_digest_from_environment()
            verify_package_inventory(package_root(), inventory_sha256)
        except (InventoryError, OSError) as error:
            print(
                f"cfetch OpenVINO package integrity check failed: {error}",
                file=sys.stderr,
            )
            return 1
    elif args.command in {"runtime-check", "host-preflight"}:
        try:
            expected_runtime_sha256 = (
                args.runtime_manifest_sha256
                if args.command == "host-preflight"
                else None
            )
            raw_runtime = runtime_self_check(expected_runtime_sha256)
        except (
            ImportError,
            ManifestError,
            RuntimeBundleError,
            RuntimeError,
            OSError,
        ) as error:
            print(f"cfetch OpenVINO raw runtime check failed: {error}", file=sys.stderr)
            return 1
    else:
        print(
            "cfetch OpenVINO package integrity check failed: serve requires the "
            "inventory-bound native launcher",
            file=sys.stderr,
        )
        return 1
    if args.command == "runtime-check":
        try:
            if raw_runtime is not None:
                runtime = raw_runtime
            else:
                manifest_path = package_root() / "package-manifest.json"
                if manifest_path.exists():
                    package = load_package_manifest(manifest_path)
                    runtime = runtime_self_check(
                        package.runtime_manifest_sha256, package_bound_files(package)
                    )
                else:
                    runtime = runtime_self_check(
                        allowed_package_files=(INVENTORY_NAME,)
                    )
            print(json.dumps(runtime, separators=(",", ":")))
        except (
            ImportError,
            ManifestError,
            RuntimeBundleError,
            RuntimeError,
            OSError,
        ) as error:
            print(f"cfetch OpenVINO runtime check failed: {error}", file=sys.stderr)
            return 1
        return 0
    if args.command == "host-preflight":
        if raw_runtime is None:
            print(
                "cfetch OpenVINO host-preflight requires the raw verified frozen "
                "runtime dispatcher",
                file=sys.stderr,
            )
            return 2
        try:
            if args.artifact_dir.is_symlink() or not args.artifact_dir.is_dir():
                raise RuntimeError(
                    "host-preflight artifact directory must be a real directory"
                )
            artifact = load_artifact(
                args.artifact_dir,
                "artifact-manifest.json",
                args.artifact_manifest_sha256,
            )
            compile_config = parse_preflight_compile_config(
                args.compile_config_json
            )
            result = collect_host_preflight(
                artifact,
                raw_runtime,
                args.device_class,
                args.device,
                compile_config,
                args.host_file,
            )
            verified_artifact = load_artifact(
                args.artifact_dir,
                "artifact-manifest.json",
                args.artifact_manifest_sha256,
            )
            if verified_artifact.manifest_sha256 != artifact.manifest_sha256:
                raise RuntimeError("host-preflight artifact changed during collection")
            verified_runtime = runtime_self_check(args.runtime_manifest_sha256)
            if (
                verified_runtime["runtime_manifest_sha256"]
                != raw_runtime["runtime_manifest_sha256"]
                or verified_runtime["dependency_versions"]
                != raw_runtime["dependency_versions"]
            ):
                raise RuntimeError("host-preflight runtime changed during collection")
            sys.stdout.buffer.write(canonical_preflight_output(result))
            sys.stdout.buffer.flush()
        except (
            ImportError,
            ManifestError,
            RuntimeBundleError,
            RuntimeError,
            OSError,
        ) as error:
            print(f"cfetch OpenVINO host-preflight failed: {error}", file=sys.stderr)
            return 1
        return 0
    if args.command != "serve":
        raise AssertionError("argparse accepted an unknown command")
    if args.host != "127.0.0.1" or args.port != 0:
        print("adapter must be launched with --host 127.0.0.1 --port 0", file=sys.stderr)
        return 2
    try:
        package = load_package_manifest(
            package_root() / "package-manifest.json"
        )
        runtime_self_check(
            package.runtime_manifest_sha256, package_bound_files(package)
        )
        verify_dependency_versions(package)
        bearer = parse_auth_line(sys.stdin.buffer)
        signers = {
            scope.scope_id: Ed25519Signer(
                scope.attestation_private_key_file, scope.attestation_public_key
            )
            for scope in package.scopes.values()
        }
        tokenizer = HuggingFaceTokenizer(package.artifact)
        serve(
            package,
            bearer,
            sys.stdin.buffer,
            tokenizer,
            OpenVinoEngine,
            signers,
        )
    except (
        ManifestError,
        RequestError,
        RuntimeBundleError,
        RuntimeError,
        OSError,
    ) as error:
        print(f"cfetch OpenVINO adapter refused to start: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
