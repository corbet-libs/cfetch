#!/usr/bin/env python3
"""Run one hermetic, nonempty backend admission transaction.

The stage phase is intentionally offline.  It consumes a complete local cohort,
builds deterministic content-addressed assets, and writes a proposed registry.
It never edits the repository or uploads a release.  The activate phase still
does not publish anything: it creates a release-ready activation bundle only
after exact final-package conformance receipts cover every package/scope pair.
The run command owns the complete local sequence and emits an immutable
publication plan without uploading assets or changing the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from packages.openvino.package_inventory import (  # noqa: E402
    INVENTORY_NAME as PACKAGE_INVENTORY_NAME,
    LAUNCHER as PACKAGE_LAUNCHER,
    InventoryError as PackageInventoryError,
    RebindingProjection,
    project_package_manifest_rebinding,
    verify_bound as verify_bound_package_inventory,
)

from admission_evidence import REMOTE_ATTESTED_TRANSPORT, SUPERVISED_LOCAL_TRANSPORT
from cross_backend_eval import (
    ADMISSION_IMPLEMENTATION_BUNDLE_SHA256,
    ADMISSION_POLICY_SHA256,
    MAX_ADMISSION_CACHE_BYTES,
    MAX_ADMISSION_COHORT_BYTES,
    MAX_ADMISSION_REGISTRY_BYTES,
    MAX_ADMISSION_REPORT_BYTES,
    MAX_ADMITTED_SCOPES,
    MODEL,
    MODEL_REVISION,
    PROFILE_ID,
    PROFILE_MANIFEST_SHA256,
    REGISTRY_CACHE_BINDING_FIELDS,
    REPORT_BACKEND_BINDING_FIELDS,
    REQUIRED_CLASSES,
    build_compatibility_report,
    file_sha256,
    load_cache,
    load_embedded_evidence_reports,
    load_sequence_probe_cache,
    scope_id_value,
    validate_admission_cache_container,
    validate_loaded_scope_bindings,
    validate_measurement_bundle,
    verify_implementation_bundle,
    verify_release_registry,
)
from measurement_bundle import build_measurement_bundle
from scifact_contract import load_scifact_contract


SCHEMA_VERSION = 1
MAX_TRANSACTION_MANIFEST_BYTES = 1024 * 1024
MAX_STAGE_PLAN_BYTES = 4 * 1024 * 1024
MAX_PACKAGE_MANIFEST_BYTES = 1024 * 1024
MAX_PACKAGE_FILES = 4096
MAX_PACKAGE_BYTES = 2 * 1024 * 1024 * 1024
MAX_PACKAGES = 64
MAX_RECEIPT_BYTES = 64 * 1024
MAX_ACTIVATION_BYTES = 4 * 1024 * 1024 * 1024
MAX_PUBLICATION_PLAN_BYTES = 4 * 1024 * 1024
EXPECTED_SIGNED_REQUESTS = sum((64 + size - 1) // size for size in range(1, 65)) + 14
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SLUG_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
RELEASE_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
PLATFORM_VALUE_RE = re.compile(r"[a-z0-9]+(?:[._+-][a-z0-9]+)*")
DEVICE_CLASS_ORDER = {"npu": 0, "gpu": 1, "cpu": 2}
PACKAGE_FORMAT_SUFFIX = {"zip": ".zip"}
REQUIRED_SELECTION = (
    "first available admitted scope in NPU, GPU, accelerated CPU order; each "
    "signed request and response is bound to the requested scope id"
)
PROFILE_SOURCE_PATH = "src/embedding_profile.rs"
PROFILE_SOURCE_MAX_BYTES = 4 * 1024 * 1024
PROFILE_STATUS_CANDIDATE_TEXT = 'pub const PROFILE_STATUS: &str = "candidate";'
PROFILE_STATUS_ACTIVE_TEXT = 'pub const PROFILE_STATUS: &str = "active";'
RELEASE_REPOSITORY = "corbet-libs/cfetch"


class TransactionError(ValueError):
    """The proposed admission transaction is incomplete or unsafe."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TransactionError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise TransactionError(f"{label} must be a regular non-symlink file")
    with path.open("rb") as source:
        size = os.fstat(source.fileno()).st_size
        if size < 1 or size > maximum:
            raise TransactionError(f"{label} must be 1..{maximum} bytes")
        data = source.read(maximum + 1)
    if len(data) != size:
        raise TransactionError(f"{label} changed while it was read")
    return data


def _parse_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TransactionError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise TransactionError(f"{label} must contain one JSON object")
    return value


def _read_json(path: Path, maximum: int, label: str) -> tuple[dict[str, Any], bytes]:
    data = _read_bounded(path, maximum, label)
    return _parse_json_bytes(data, label), data


def _canonical_json(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        text = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return (text + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise TransactionError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _validate_profile_source_promotion_claim(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise TransactionError("profile_source_promotion must be an object")
    _exact_keys(
        value,
        {"path", "base_sha256", "active_sha256", "candidate_text", "active_text"},
        set(),
        "profile_source_promotion",
    )
    if value["path"] != PROFILE_SOURCE_PATH:
        raise TransactionError(
            f"profile_source_promotion.path must be {PROFILE_SOURCE_PATH!r}"
        )
    _digest(value["base_sha256"], "profile_source_promotion.base_sha256")
    _digest(value["active_sha256"], "profile_source_promotion.active_sha256")
    if (
        value["candidate_text"] != PROFILE_STATUS_CANDIDATE_TEXT
        or value["active_text"] != PROFILE_STATUS_ACTIVE_TEXT
    ):
        raise TransactionError(
            "profile_source_promotion must bind the exact candidate-to-active constant"
        )
    return value


def _profile_source_promotion(repository: Path) -> dict[str, str]:
    source = repository / PROFILE_SOURCE_PATH
    base = _read_bounded(source, PROFILE_SOURCE_MAX_BYTES, "embedding profile source")
    candidate = PROFILE_STATUS_CANDIDATE_TEXT.encode("utf-8")
    active = PROFILE_STATUS_ACTIVE_TEXT.encode("utf-8")
    if base.count(candidate) != 1 or base.count(active) != 0:
        raise TransactionError(
            "embedding profile source must contain exactly one candidate status constant "
            "and no active status constant"
        )
    promoted = base.replace(candidate, active, 1)
    return {
        "path": PROFILE_SOURCE_PATH,
        "base_sha256": _sha256_bytes(base),
        "active_sha256": _sha256_bytes(promoted),
        "candidate_text": PROFILE_STATUS_CANDIDATE_TEXT,
        "active_text": PROFILE_STATUS_ACTIVE_TEXT,
    }


def _slug(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 128
        or SLUG_RE.fullmatch(value) is None
    ):
        raise TransactionError(f"{label} must be a canonical lowercase slug")
    return value


def _platform_value(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 64
        or PLATFORM_VALUE_RE.fullmatch(value) is None
    ):
        raise TransactionError(f"{label} must be a canonical platform value")
    return value


def _unique_strings(
    value: object, label: str, *, sorted_values: bool = False
) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise TransactionError(f"{label} must be a nonempty array of unique strings")
    if sorted_values and value != sorted(value):
        raise TransactionError(f"{label} must be sorted")
    return list(value)


def _exact_keys(
    value: Mapping[str, object], required: set[str], optional: set[str], label: str
) -> None:
    missing = sorted(required.difference(value))
    unknown = sorted(set(value).difference(required).difference(optional))
    if missing:
        raise TransactionError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise TransactionError(f"{label} has unknown fields: {', '.join(unknown)}")


def _relative_parts(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise TransactionError(f"{label} must be a bounded relative path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise TransactionError(f"{label} must be a normalized relative path")
    return pure.parts


def _resolve_input(
    root: Path, value: object, label: str, *, directory: bool
) -> Path:
    parts = _relative_parts(value, label)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise TransactionError(f"{label} contains a symlink")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise TransactionError(f"{label} escapes or is absent from the manifest root") from error
    if directory:
        if not resolved.is_dir():
            raise TransactionError(f"{label} must name a directory")
    elif not resolved.is_file():
        raise TransactionError(f"{label} must name a regular file")
    return resolved


def _safe_output_parent(output: Path) -> tuple[Path, Path]:
    if output.exists() or output.is_symlink():
        raise TransactionError(f"refusing to overwrite output: {output}")
    parent = output.parent.resolve(strict=True)
    if output.parent.is_symlink():
        raise TransactionError("output parent must not be a symlink")
    return parent, parent / output.name


def _copy_exclusive(source: Path, destination: Path, expected_digest: str) -> int:
    if destination.exists() or destination.is_symlink():
        raise TransactionError(f"refusing to overwrite staged asset: {destination}")
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        while chunk := input_stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
            output_stream.write(chunk)
    if size < 1 or digest.hexdigest() != expected_digest:
        destination.unlink(missing_ok=True)
        raise TransactionError(f"staged bytes for {source.name} changed during copying")
    return size


def _load_base_registry(path: Path, expected_digest: str) -> dict[str, Any]:
    registry, raw = _read_json(path, MAX_ADMISSION_REGISTRY_BYTES, "base registry")
    if _sha256_bytes(raw) != expected_digest:
        raise TransactionError("base registry bytes do not match base_registry_sha256")
    if registry.get("schema_version") != 1 or registry.get("profile_id") != PROFILE_ID:
        raise TransactionError("base registry does not match the cfetch profile")
    if registry.get("shared_identity", {}).get(
        "profile_manifest_sha256"
    ) != PROFILE_MANIFEST_SHA256:
        raise TransactionError("base registry has another semantic profile identity")
    admission = registry.get("admission")
    if not isinstance(admission, dict):
        raise TransactionError("base registry has no admission policy")
    if admission.get("policy_manifest_sha256") != ADMISSION_POLICY_SHA256:
        raise TransactionError("base registry has another admission policy")
    if (
        admission.get("implementation_bundle_sha256")
        != ADMISSION_IMPLEMENTATION_BUNDLE_SHA256
    ):
        raise TransactionError("base registry has another admission implementation")
    entries = registry.get("admitted_backends")
    packages = registry.get("local_packages")
    if not isinstance(entries, list) or not isinstance(packages, list):
        raise TransactionError("base registry backend/package lists are invalid")
    if len(entries) > MAX_ADMITTED_SCOPES or len(packages) > MAX_PACKAGES:
        raise TransactionError("base registry exceeds admission transaction bounds")
    return registry


def _parse_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    if path.is_symlink():
        raise TransactionError("transaction manifest must not be a symlink")
    manifest, _ = _read_json(
        path.resolve(strict=True),
        MAX_TRANSACTION_MANIFEST_BYTES,
        "transaction manifest",
    )
    _exact_keys(
        manifest,
        {
            "schema_version",
            "base_registry",
            "base_registry_sha256",
            "base_variants",
            "base_variants_sha256",
            "release_tag",
            "receipt_attestation_public_key",
            "candidate_scopes",
            "scopes",
            "packages",
        },
        {"scifact_snapshot"},
        "transaction manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise TransactionError("transaction manifest schema_version must be 1")
    release_tag = manifest["release_tag"]
    if (
        not isinstance(release_tag, str)
        or RELEASE_TAG_RE.fullmatch(release_tag) is None
        or release_tag.lower() in {"latest", "current", "draft"}
    ):
        raise TransactionError("release_tag must be a fixed, canonical release tag")
    _digest(manifest["base_registry_sha256"], "base_registry_sha256")
    _digest(manifest["base_variants_sha256"], "base_variants_sha256")
    _digest(
        manifest["receipt_attestation_public_key"],
        "receipt_attestation_public_key",
    )
    candidates = _unique_strings(manifest["candidate_scopes"], "candidate_scopes")
    for index, scope_id in enumerate(candidates):
        _slug(scope_id, f"candidate_scopes[{index}]")
    if not isinstance(manifest["scopes"], list) or not manifest["scopes"]:
        raise TransactionError("scopes must be a nonempty array")
    if len(manifest["scopes"]) > MAX_ADMITTED_SCOPES:
        raise TransactionError("scopes exceeds the admitted cohort bound")
    if not isinstance(manifest["packages"], list) or not manifest["packages"]:
        raise TransactionError("packages must be a nonempty array")
    if len(manifest["packages"]) > MAX_PACKAGES:
        raise TransactionError("packages exceeds the transaction bound")
    return manifest, path.resolve(strict=True).parent


def _transaction_scifact_snapshot(
    manifest: Mapping[str, object], root: Path
) -> Path:
    relative = manifest.get("scifact_snapshot")
    if relative is None:
        raise TransactionError(
            "run requires scifact_snapshot as a manifest-relative input"
        )
    snapshot = _resolve_input(
        root, relative, "scifact_snapshot", directory=True
    )
    try:
        load_scifact_contract(snapshot_directory=snapshot)
    except (OSError, ValueError) as error:
        raise TransactionError(
            f"manifest-bound pinned SciFact snapshot is invalid: {error}"
        ) from error
    return snapshot


def _load_variant_catalog(path: Path, expected_digest: str) -> dict[str, dict[str, Any]]:
    raw = _read_bounded(path, MAX_TRANSACTION_MANIFEST_BYTES, "release variant catalog")
    if _sha256_bytes(raw) != expected_digest:
        raise TransactionError(
            "release variant catalog bytes do not match base_variants_sha256"
        )
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TransactionError("release variant catalog is not valid UTF-8 JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "variants"}
        or value["schema_version"] != 1
        or not isinstance(value["variants"], list)
        or not value["variants"]
    ):
        raise TransactionError(
            "release variant catalog must contain schema_version 1 and a nonempty variants array"
        )
    variants: dict[str, dict[str, Any]] = {}
    for index, variant in enumerate(value["variants"]):
        if not isinstance(variant, dict):
            raise TransactionError(f"release variants[{index}] must be an object")
        for field in ("id", "os", "arch", "binary", "backend"):
            if not isinstance(variant.get(field), str) or not variant[field]:
                raise TransactionError(f"release variants[{index}] needs {field}")
        variant_id = _slug(variant["id"], f"release variants[{index}].id")
        if variant_id in variants:
            raise TransactionError("release variant catalog contains a duplicate id")
        variants[variant_id] = variant
    return variants


def _validate_scope_rows(
    manifest: dict[str, Any], root: Path, base_registry: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], dict[str, Path]]:
    existing_entries = base_registry["admitted_backends"]
    existing: dict[str, dict[str, Any]] = {}
    for entry in existing_entries:
        if not isinstance(entry, dict):
            raise TransactionError("base registry admitted_backends entries must be objects")
        scope_id = _slug(entry.get("scope_id"), "base registry scope_id")
        if scope_id in existing:
            raise TransactionError("base registry has duplicate admitted scope ids")
        existing[scope_id] = entry

    candidates = set(manifest["candidate_scopes"])
    if candidates.intersection(existing):
        raise TransactionError("candidate_scopes must not already be admitted")
    rows: dict[str, dict[str, Any]] = {}
    cache_paths: dict[str, Path] = {}
    raw_paths: dict[str, Path] = {}
    cohort_bytes = 0
    for index, item in enumerate(manifest["scopes"]):
        label = f"scopes[{index}]"
        if not isinstance(item, dict):
            raise TransactionError(f"{label} must be an object")
        _exact_keys(
            item,
            {
                "scope_id",
                "admission_cache",
                "admission_cache_sha256",
                "raw_measurements",
            },
            set(),
            label,
        )
        scope_id = _slug(item["scope_id"], f"{label}.scope_id")
        if scope_id in rows:
            raise TransactionError(f"duplicate scope row {scope_id!r}")
        expected_digest = _digest(
            item["admission_cache_sha256"], f"{label}.admission_cache_sha256"
        )
        cache_path = _resolve_input(
            root, item["admission_cache"], f"{label}.admission_cache", directory=False
        )
        if cache_path.is_symlink():
            raise TransactionError(f"{label}.admission_cache must not be a symlink")
        size = cache_path.stat().st_size
        cohort_bytes += size
        if size < 1 or size > MAX_ADMISSION_CACHE_BYTES:
            raise TransactionError(f"{label}.admission_cache exceeds its byte bound")
        if cohort_bytes > MAX_ADMISSION_COHORT_BYTES:
            raise TransactionError("complete admission cohort exceeds its byte bound")
        if file_sha256(cache_path) != expected_digest:
            raise TransactionError(f"{label}.admission_cache_sha256 does not match bytes")
        validate_admission_cache_container(cache_path)
        raw_path = _resolve_input(
            root, item["raw_measurements"], f"{label}.raw_measurements", directory=True
        )
        rows[scope_id] = item
        cache_paths[scope_id] = cache_path
        raw_paths[scope_id] = raw_path

    required_scopes = set(existing).union(candidates)
    if set(rows) != required_scopes:
        missing = sorted(required_scopes.difference(rows))
        unexpected = sorted(set(rows).difference(required_scopes))
        raise TransactionError(
            f"scope rows must equal retained plus candidate cohort; missing={missing}, "
            f"unexpected={unexpected}"
        )
    return rows, cache_paths, raw_paths


def _load_cohort(
    cache_paths: Mapping[str, Path], base_registry: dict[str, Any]
) -> tuple[
    dict[str, tuple[dict[str, object], Any, Any]],
    dict[str, tuple[Any, Any, Any]],
]:
    loaded = {scope_id: load_cache(path) for scope_id, path in cache_paths.items()}
    probes = {
        scope_id: load_sequence_probe_cache(path)
        for scope_id, path in cache_paths.items()
    }
    for scope_id, (metadata, _, _) in loaded.items():
        if metadata.get("scope_id") != scope_id:
            raise TransactionError(
                f"cache scope {metadata.get('scope_id')!r} does not equal row {scope_id!r}"
            )
    retained = {
        entry["scope_id"]: entry for entry in base_registry["admitted_backends"]
    }
    validate_loaded_scope_bindings(loaded, retained)
    classes = {metadata["device_class"] for metadata, _, _ in loaded.values()}
    if classes != REQUIRED_CLASSES:
        raise TransactionError(
            "complete cohort must contain admitted npu, gpu, and accelerated cpu scopes"
        )
    return loaded, probes


def _parent_binding(base_registry: dict[str, Any]) -> tuple[str | None, str | None]:
    bindings = {
        (entry.get("compatibility_report"), entry.get("compatibility_report_sha256"))
        for entry in base_registry["admitted_backends"]
        if isinstance(entry, dict)
    }
    if not bindings:
        return None, None
    if len(bindings) != 1:
        raise TransactionError("base registry does not have one global report binding")
    reference, digest = next(iter(bindings))
    if (
        not isinstance(reference, str)
        or not isinstance(digest, str)
        or reference != f"release/admission/{digest}.json"
        or SHA256_RE.fullmatch(digest) is None
    ):
        raise TransactionError("base registry parent report binding is not content-addressed")
    return reference, digest


def _build_report(
    cache_paths: dict[str, Path],
    loaded: dict[str, tuple[dict[str, object], Any, Any]],
    probes: dict[str, tuple[Any, Any, Any]],
    base_registry: dict[str, Any],
    candidate_scopes: set[str],
) -> tuple[dict[str, Any], bytes, str]:
    admitted = {
        entry["scope_id"] for entry in base_registry["admitted_backends"]
    }
    parent_reference, parent_digest = _parent_binding(base_registry)
    report = build_compatibility_report(
        cache_paths,
        loaded,
        probes,
        admitted,
        candidate_scopes,
        parent_reference,
        parent_digest,
    )
    if report.get("admission_gate", {}).get("passed") is not True:
        raise TransactionError("global cohort admission gate did not pass")
    report_bytes = _canonical_json(report, pretty=True)
    if len(report_bytes) > MAX_ADMISSION_REPORT_BYTES:
        raise TransactionError("compatibility report exceeds its byte bound")
    return report, report_bytes, _sha256_bytes(report_bytes)


def _walk_package_files(root: Path) -> list[tuple[str, Path, int]]:
    if root.is_symlink() or not root.is_dir():
        raise TransactionError("package_directory must be a real directory")
    result: list[tuple[str, Path, int]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TransactionError(f"package contains a symlink: {path}")
        mode = path.stat().st_mode
        if path.is_dir():
            continue
        if not stat.S_ISREG(mode):
            raise TransactionError(f"package contains a non-regular file: {path}")
        relative = path.relative_to(root).as_posix()
        _relative_parts(relative, f"package member {relative}")
        size = path.stat().st_size
        if size < 1:
            raise TransactionError(f"package contains an empty file: {relative}")
        total += size
        if total > MAX_PACKAGE_BYTES:
            raise TransactionError("package expands beyond its byte bound")
        result.append((relative, path, mode))
        if len(result) > MAX_PACKAGE_FILES:
            raise TransactionError("package contains too many files")
    if not result:
        raise TransactionError("package_directory contains no files")
    return result


def _package_tree_digest(root: Path) -> str:
    digest = hashlib.sha256(b"cfetch-target-package-source-v1\0")
    for relative, path, mode in _walk_package_files(root):
        name = relative.encode("utf-8")
        size_before = path.stat().st_size
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(stat.S_IMODE(mode).to_bytes(4, "big"))
        digest.update(size_before.to_bytes(8, "big"))
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        if path.stat().st_size != size_before:
            raise TransactionError(f"package file changed while hashing: {relative}")
    return digest.hexdigest()


def _resolve_package_member(root: Path, value: object, label: str) -> Path:
    parts = _relative_parts(value, label)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise TransactionError(f"{label} contains a symlink")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise TransactionError(f"{label} escapes the package") from error
    if not resolved.is_file():
        raise TransactionError(f"{label} must name a package file")
    return resolved


def _validate_artifact_manifest(
    package_root: Path, package_document: dict[str, Any]
) -> None:
    artifact_path = _resolve_package_member(
        package_root, package_document.get("artifact_manifest"), "artifact_manifest"
    )
    expected = _digest(
        package_document.get("artifact_manifest_sha256"),
        "artifact_manifest_sha256",
    )
    artifact, raw = _read_json(
        artifact_path, MAX_PACKAGE_MANIFEST_BYTES, "artifact manifest"
    )
    if _sha256_bytes(raw) != expected:
        raise TransactionError("artifact manifest digest does not match package manifest")
    files = artifact.get("files")
    if not isinstance(files, list) or not files:
        raise TransactionError("artifact manifest must bind a nonempty files array")
    seen: set[str] = set()
    for index, entry in enumerate(files):
        label = f"artifact manifest files[{index}]"
        if not isinstance(entry, dict):
            raise TransactionError(f"{label} must be an object")
        relative = entry.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise TransactionError(f"{label}.path must be unique")
        seen.add(relative)
        path = _resolve_package_member(artifact_path.parent, relative, f"{label}.path")
        digest = _digest(entry.get("sha256"), f"{label}.sha256")
        byte_count = entry.get("bytes")
        if type(byte_count) is not int or byte_count < 1:
            raise TransactionError(f"{label}.bytes must be a positive integer")
        if path.stat().st_size != byte_count or file_sha256(path) != digest:
            raise TransactionError(f"{label} does not match the artifact bytes")


def _load_and_bind_package_manifest(
    package_root: Path,
    manifest_relative: object,
    ordered_scope_ids: Sequence[str],
    loaded: Mapping[str, tuple[dict[str, object], Any, Any]],
    evidence_reports: Mapping[str, Mapping[str, dict[str, object]]],
    dispatcher: Mapping[str, str],
    report_digest: str,
) -> tuple[str, dict[str, Any], bytes, RebindingProjection]:
    manifest_parts = _relative_parts(manifest_relative, "package_manifest")
    manifest_name = PurePosixPath(*manifest_parts).as_posix()
    if manifest_name != "package-manifest.json":
        raise TransactionError("OpenVINO package manifest must be at the package root")
    if dispatcher.get("binary") != PACKAGE_LAUNCHER:
        raise TransactionError(
            f"OpenVINO package dispatcher must be {PACKAGE_LAUNCHER!r}"
        )
    manifest_path = _resolve_package_member(
        package_root, manifest_name, "package_manifest"
    )
    document, manifest_raw = _read_json(
        manifest_path, MAX_PACKAGE_MANIFEST_BYTES, "package manifest"
    )
    if manifest_raw != _canonical_json(document):
        raise TransactionError("candidate package manifest must use canonical JSON bytes")
    inventory_path = _resolve_package_member(
        package_root, PACKAGE_INVENTORY_NAME, "package inventory"
    )
    candidate_inventory_sha256 = file_sha256(inventory_path)
    try:
        verify_bound_package_inventory(package_root, candidate_inventory_sha256)
    except (OSError, PackageInventoryError) as error:
        raise TransactionError(
            f"candidate package inventory/launcher binding is invalid: {error}"
        ) from error
    expected_fixed = {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
        "admission_policy_sha256": ADMISSION_POLICY_SHA256,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
    }
    for field, expected in expected_fixed.items():
        if document.get(field) != expected:
            raise TransactionError(f"package manifest {field} is not the frozen identity")
    if document.get("package_state") != "candidate":
        raise TransactionError(
            "admission package manifest package_state must be candidate"
        )
    _validate_artifact_manifest(package_root, document)
    scopes = document.get("scopes")
    if not isinstance(scopes, list) or any(not isinstance(item, dict) for item in scopes):
        raise TransactionError("package manifest scopes must be an array of objects")
    by_scope: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(scopes):
        scope_id = _slug(entry.get("scope_id"), f"package scopes[{index}].scope_id")
        if scope_id in by_scope:
            raise TransactionError("package manifest contains a duplicate scope")
        by_scope[scope_id] = entry
    if list(by_scope) != list(ordered_scope_ids):
        raise TransactionError(
            "package manifest scope order must equal package ordered_scope_ids"
        )

    # Project the exact probe package without changing the candidate. Because
    # the candidate tree and launcher binding were verified above, replacing
    # only this canonical manifest entry also reconstructs the probe inventory
    # and the exact launcher bytes seen by the physical collector.
    probe_document = json.loads(json.dumps(document))
    probe_document["package_state"] = "physical-probe"
    for entry in probe_document["scopes"]:
        entry["placement_evidence_sha256"] = None
        entry["sequence_capability_evidence_sha256"] = None
        entry["performance_evidence_sha256"] = None
        entry["compatibility_report_sha256"] = None
    probe_bytes = _canonical_json(probe_document)
    probe_digest = _sha256_bytes(probe_bytes)
    try:
        probe_projection = project_package_manifest_rebinding(
            package_root, candidate_inventory_sha256, probe_bytes
        )
    except (OSError, PackageInventoryError) as error:
        raise TransactionError(
            f"candidate package cannot reconstruct its physical probe: {error}"
        ) from error

    for scope_id in ordered_scope_ids:
        metadata = loaded[scope_id][0]
        entry = by_scope[scope_id]
        for field in REGISTRY_CACHE_BINDING_FIELDS:
            if entry.get(field) != metadata.get(field):
                raise TransactionError(
                    f"package scope {scope_id!r} {field} does not match admission cache"
                )
        compatibility = entry.get("compatibility_report_sha256")
        if compatibility is not None:
            raise TransactionError(
                f"candidate package scope {scope_id!r} compatibility report must be null"
            )
        try:
            placement = evidence_reports[scope_id]["placement"]
        except KeyError as error:
            raise TransactionError(
                f"scope {scope_id!r} has no embedded placement provider binding"
            ) from error
        binding = placement.get("provider_binding")
        if not isinstance(binding, dict) or binding.get("provider") != "openvino":
            raise TransactionError(
                f"scope {scope_id!r} placement provider binding is not OpenVINO"
            )
        package_binding = {
            "dispatcher_sha256": probe_projection.launcher_sha256,
            "runtime_manifest_sha256": document.get("runtime_manifest_sha256"),
            "openvino_compile_config": entry.get("openvino_compile_config"),
            "expected_host": entry.get("required_host"),
            "actual_host": entry.get("required_host"),
        }
        for field, expected in package_binding.items():
            if binding.get(field) != expected:
                raise TransactionError(
                    f"package scope {scope_id!r} placement {field} does not "
                    "match the candidate manifest"
                )
        bucket_results = placement.get("bucket_results")
        if not isinstance(bucket_results, list):
            raise TransactionError(
                f"scope {scope_id!r} placement bucket_results must be an array"
            )
        if [row.get("bucket") for row in bucket_results if isinstance(row, dict)] != (
            metadata.get("supported_sequence_buckets")
        ):
            raise TransactionError(
                f"scope {scope_id!r} placement buckets do not match the candidate"
            )
        expected_provider = {
            "requested_device": entry.get("openvino_device"),
            "expected_execution_devices": entry.get("required_execution_devices"),
            "actual_execution_devices": entry.get("required_execution_devices"),
            "expected_device_properties": entry.get("required_openvino_properties"),
            "actual_device_properties": entry.get("required_openvino_properties"),
        }
        for row in bucket_results:
            if not isinstance(row, dict) or not isinstance(
                row.get("provider_evidence"), dict
            ):
                raise TransactionError(
                    f"scope {scope_id!r} placement bucket lacks provider evidence"
                )
            provider = row["provider_evidence"]
            for field, expected in expected_provider.items():
                if provider.get(field) != expected:
                    raise TransactionError(
                        f"package scope {scope_id!r} bucket {row.get('bucket')} "
                        f"placement {field} does not match the candidate manifest"
                    )

    for scope_id in ordered_scope_ids:
        expected_probe_digest = evidence_reports[scope_id]["placement"][
            "provider_binding"
        ].get("probe_package_manifest_sha256")
        if expected_probe_digest != probe_digest:
            raise TransactionError(
                f"candidate package cannot reconstruct measured probe manifest for "
                f"scope {scope_id!r}"
            )

    document["package_state"] = "release"
    for entry in scopes:
        entry["compatibility_report_sha256"] = report_digest
    final_bytes = _canonical_json(document)
    if len(final_bytes) > MAX_PACKAGE_MANIFEST_BYTES:
        raise TransactionError("final package manifest exceeds its byte bound")
    try:
        release_projection = project_package_manifest_rebinding(
            package_root, candidate_inventory_sha256, final_bytes
        )
    except (OSError, PackageInventoryError) as error:
        raise TransactionError(
            f"candidate package cannot project its release integrity binding: {error}"
        ) from error
    return manifest_name, document, final_bytes, release_projection


def _zip_info(name: str, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    permissions = (
        0o755
        if mode & 0o111
        else 0o600
        if stat.S_IMODE(mode) & 0o077 == 0
        else 0o644
    )
    info.external_attr = (stat.S_IFREG | permissions) << 16
    return info


def _stream_zip_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, path: Path
) -> None:
    with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as output:
        while chunk := source.read(1024 * 1024):
            output.write(chunk)


def _build_package_zip(
    package_root: Path,
    manifest_name: str,
    final_manifest_bytes: bytes,
    destination_directory: Path,
    replacement_files: Mapping[str, bytes] | None = None,
) -> tuple[Path, int]:
    files = _walk_package_files(package_root)
    names = {name for name, _, _ in files}
    if manifest_name not in names:
        raise TransactionError("package manifest disappeared during packaging")
    replacements = {manifest_name: final_manifest_bytes}
    if replacement_files is not None:
        overlap = set(replacements).intersection(replacement_files)
        if overlap:
            raise TransactionError(
                f"package replacement files overlap the manifest: {sorted(overlap)}"
            )
        replacements.update(replacement_files)
    missing_replacements = set(replacements).difference(names)
    if missing_replacements:
        raise TransactionError(
            f"package replacement files are absent: {sorted(missing_replacements)}"
        )
    if any(not isinstance(raw, bytes) or not raw for raw in replacements.values()):
        raise TransactionError("package replacement files must be nonempty bytes")
    temporary = destination_directory / f".package-{os.getpid()}-{len(files)}.zip"
    if temporary.exists():
        raise TransactionError("temporary package output already exists")
    try:
        with zipfile.ZipFile(
            temporary,
            "x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
            strict_timestamps=True,
        ) as archive:
            for name, path, mode in files:
                info = _zip_info(name, mode)
                if name in replacements:
                    archive.writestr(info, replacements[name])
                else:
                    _stream_zip_member(archive, info, path)
        digest = file_sha256(temporary)
        destination = destination_directory / f"{digest}.zip"
        if destination.exists():
            raise TransactionError(f"refusing to overwrite package asset {destination}")
        os.replace(temporary, destination)
        return destination, destination.stat().st_size
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_dispatcher(
    package_root: Path,
    value: object,
    os_name: str,
    cfetch_binary: str,
    label: str,
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise TransactionError(f"{label} must be an object")
    _exact_keys(value, {"binary", "sha256"}, set(), label)
    binary = value["binary"]
    if (
        not isinstance(binary, str)
        or not binary
        or len(binary) > 128
        or binary in {".", "..", cfetch_binary}
        or "/" in binary
        or "\\" in binary
        or any(character.isspace() or ord(character) < 32 for character in binary)
        or PurePosixPath(binary).name != binary
    ):
        raise TransactionError(
            f"{label}.binary must be a plain package-root sibling basename distinct from cfetch"
        )
    if (os_name == "win") != binary.endswith(".exe"):
        raise TransactionError(f"{label}.binary extension does not match package OS")
    path = package_root / binary
    if path.is_symlink() or not path.is_file():
        raise TransactionError(f"{label}.binary must be a regular non-symlink package file")
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode) or mode & 0o111 == 0:
        raise TransactionError(f"{label}.binary must be executable")
    expected = _digest(value["sha256"], f"{label}.sha256")
    if file_sha256(path) != expected:
        raise TransactionError(f"{label}.sha256 does not match the dispatcher bytes")
    return {"binary": binary, "sha256": expected}


def _validate_packaged_dispatcher(
    package_path: Path, dispatcher: Mapping[str, str]
) -> None:
    try:
        with zipfile.ZipFile(package_path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or dispatcher["binary"] not in names:
                raise TransactionError(
                    "final package archive does not contain one exact dispatcher member"
                )
            info = archive.getinfo(dispatcher["binary"])
            permissions = (info.external_attr >> 16) & 0o777
            if info.is_dir() or permissions & 0o111 == 0:
                raise TransactionError(
                    "final package archive dispatcher is not an executable regular member"
                )
            digest = hashlib.sha256()
            with archive.open(info) as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != dispatcher["sha256"]:
                raise TransactionError(
                    "final package archive dispatcher bytes changed during packaging"
                )
    except zipfile.BadZipFile as error:
        raise TransactionError("final target package is not a valid ZIP") from error


def _validate_package_rows(
    manifest: dict[str, Any],
    root: Path,
    loaded: Mapping[str, tuple[dict[str, object], Any, Any]],
    evidence_reports: Mapping[str, Mapping[str, dict[str, object]]],
    report_digest: str,
    release_tag: str,
    variant_catalog: Mapping[str, Mapping[str, object]],
    assets_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    local_packages: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    package_ids: set[str] = set()
    release_variant_ids: set[str] = set()
    covered_scopes: set[str] = set()
    for index, item in enumerate(manifest["packages"]):
        label = f"packages[{index}]"
        if not isinstance(item, dict):
            raise TransactionError(f"{label} must be an object")
        _exact_keys(
            item,
            {
                "package_id",
                "release_variant_id",
                "os",
                "arch",
                "device_families",
                "ordered_scope_ids",
                "package_directory",
                "package_manifest",
                "package_format",
                "dispatcher",
            },
            set(),
            label,
        )
        package_id = _slug(item["package_id"], f"{label}.package_id")
        if package_id in package_ids:
            raise TransactionError(f"duplicate package_id {package_id!r}")
        package_ids.add(package_id)
        variant_id = _slug(item["release_variant_id"], f"{label}.release_variant_id")
        if variant_id in release_variant_ids:
            raise TransactionError(
                f"release variant {variant_id!r} has more than one local package"
            )
        release_variant_ids.add(variant_id)
        os_name = _platform_value(item["os"], f"{label}.os")
        arch = _platform_value(item["arch"], f"{label}.arch")
        if os_name not in {"linux", "mac", "win"} or arch not in {
            "x86_64",
            "aarch64",
        }:
            raise TransactionError(f"{label} has an unsupported package target")
        variant = variant_catalog.get(variant_id)
        if variant is None:
            raise TransactionError(f"{label} references an unknown release variant")
        if variant.get("backend") != "local":
            raise TransactionError(f"{label} cannot bind an endpoint-only release variant")
        if variant.get("os") != os_name or variant.get("arch") != arch:
            raise TransactionError(f"{label} target does not match its release variant")
        cfetch_binary = variant.get("binary")
        if not isinstance(cfetch_binary, str) or not cfetch_binary:
            raise TransactionError(f"{label} release variant has no cfetch binary")
        families = _unique_strings(item["device_families"], f"{label}.device_families")
        for family_index, family in enumerate(families):
            _platform_value(family, f"{label}.device_families[{family_index}]")
        ordered = _unique_strings(item["ordered_scope_ids"], f"{label}.ordered_scope_ids")
        for scope_index, scope_id in enumerate(ordered):
            _slug(scope_id, f"{label}.ordered_scope_ids[{scope_index}]")
            if scope_id not in loaded:
                raise TransactionError(f"{label} references unknown scope {scope_id!r}")
            if loaded[scope_id][0]["transport"] != SUPERVISED_LOCAL_TRANSPORT:
                raise TransactionError(
                    f"{label} local scope {scope_id!r} must use "
                    f"{SUPERVISED_LOCAL_TRANSPORT} transport"
                )
        class_order = [DEVICE_CLASS_ORDER[loaded[scope][0]["device_class"]] for scope in ordered]
        if class_order != sorted(class_order):
            raise TransactionError(f"{label} scope order must be NPU, GPU, accelerated CPU")
        if set(class_order) != set(DEVICE_CLASS_ORDER.values()):
            raise TransactionError(
                f"{label} must contain NPU, GPU, and accelerated CPU fallbacks"
            )
        expected_families = {str(loaded[scope][0]["device"]) for scope in ordered}
        if set(families) != expected_families:
            raise TransactionError(
                f"{label}.device_families must exactly name its admitted scope devices"
            )
        covered_scopes.update(ordered)
        package_format = item["package_format"]
        if package_format not in PACKAGE_FORMAT_SUFFIX:
            raise TransactionError(f"{label}.package_format must be zip")
        package_root = _resolve_input(
            root, item["package_directory"], f"{label}.package_directory", directory=True
        )
        package_tree_digest = _package_tree_digest(package_root)
        dispatcher = _validate_dispatcher(
            package_root,
            item["dispatcher"],
            os_name,
            cfetch_binary,
            f"{label}.dispatcher",
        )
        (
            manifest_name,
            package_document,
            final_manifest,
            release_projection,
        ) = _load_and_bind_package_manifest(
            package_root,
            item["package_manifest"],
            ordered,
            loaded,
            evidence_reports,
            dispatcher,
            report_digest,
        )
        release_dispatcher = {
            "binary": dispatcher["binary"],
            "sha256": release_projection.launcher_sha256,
        }
        package_manifest_digest = _sha256_bytes(final_manifest)
        package_path, package_bytes = _build_package_zip(
            package_root,
            manifest_name,
            final_manifest,
            assets_directory,
            {
                PACKAGE_INVENTORY_NAME: release_projection.inventory_bytes,
                PACKAGE_LAUNCHER: release_projection.launcher_bytes,
            },
        )
        if _package_tree_digest(package_root) != package_tree_digest:
            raise TransactionError(f"{label}.package_directory changed during packaging")
        _validate_packaged_dispatcher(package_path, release_dispatcher)
        package_digest = file_sha256(package_path)
        package_url = (
            "https://github.com/corbet-libs/cfetch/releases/download/"
            f"{release_tag}/{package_digest}{PACKAGE_FORMAT_SUFFIX[package_format]}"
        )
        recipes = {
            scope_id: {
                "artifact_sha256": loaded[scope_id][0]["artifact_sha256"],
                "install_source": f"sibling:{package_document['artifact_manifest']}",
            }
            for scope_id in ordered
        }
        local_packages.append(
            {
                "package_id": package_id,
                "release_variant_id": variant_id,
                "os": os_name,
                "arch": arch,
                "device_families": families,
                "ordered_scope_ids": ordered,
                "artifact_recipes": recipes,
                "package_url": package_url,
                "package_sha256": package_digest,
                "package_format": package_format,
                "dispatcher": release_dispatcher,
                "package_manifest_sha256": package_manifest_digest,
                "selection": REQUIRED_SELECTION,
                "remote_fallback": "none",
            }
        )
        assets.append(
            {
                "kind": "target-package",
                "owner_id": package_id,
                "filename": package_path.name,
                "sha256": package_digest,
                "bytes": package_bytes,
                "format": package_format,
            }
        )
    expected_packaged_scopes = {
        scope_id
        for scope_id, (metadata, _, _) in loaded.items()
        if metadata["transport"] == SUPERVISED_LOCAL_TRANSPORT
    }
    unexpected_transport_scopes = {
        scope_id
        for scope_id, (metadata, _, _) in loaded.items()
        if metadata["transport"]
        not in {SUPERVISED_LOCAL_TRANSPORT, REMOTE_ATTESTED_TRANSPORT}
    }
    if unexpected_transport_scopes:
        raise TransactionError(
            "admission scopes have invalid transport values: "
            + ", ".join(sorted(unexpected_transport_scopes))
        )
    if covered_scopes != expected_packaged_scopes:
        raise TransactionError(
            "target packages must cover every supervised-local execution scope"
        )
    return local_packages, assets


def _registry_entry(
    metadata: Mapping[str, object],
    cache_digest: str,
    measurement_digest: str,
    report_digest: str,
    release_tag: str,
) -> dict[str, object]:
    entry = {field: metadata[field] for field in REPORT_BACKEND_BINDING_FIELDS}
    base = f"https://github.com/corbet-libs/cfetch/releases/download/{release_tag}"
    entry.update(
        {
            "admission_cache_url": f"{base}/{cache_digest}.npz",
            "admission_cache_sha256": cache_digest,
            "measurement_evidence_url": f"{base}/{measurement_digest}.zip",
            "measurement_evidence_sha256": measurement_digest,
            "compatibility_report": f"release/admission/{report_digest}.json",
            "compatibility_report_sha256": report_digest,
        }
    )
    return entry


def _stage_claim(plan: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in plan.items() if key != "stage_id"}


def _stage_id(plan: Mapping[str, object]) -> str:
    return _sha256_bytes(_canonical_json(_stage_claim(plan)))


def stage_transaction(manifest_path: Path, output: Path) -> Path:
    """Build an immutable staged cohort without editing or uploading anything."""
    verify_implementation_bundle()
    repository = Path(__file__).resolve().parents[2]
    profile_source_promotion = _profile_source_promotion(repository)
    manifest, root = _parse_manifest(manifest_path)
    registry_path = _resolve_input(
        root, manifest["base_registry"], "base_registry", directory=False
    )
    variants_path = _resolve_input(
        root, manifest["base_variants"], "base_variants", directory=False
    )
    base_registry = _load_base_registry(
        registry_path, manifest["base_registry_sha256"]
    )
    variant_catalog = _load_variant_catalog(
        variants_path, manifest["base_variants_sha256"]
    )
    scope_rows, cache_paths, raw_paths = _validate_scope_rows(
        manifest, root, base_registry
    )
    loaded, probes = _load_cohort(cache_paths, base_registry)
    report, report_bytes, report_digest = _build_report(
        cache_paths,
        loaded,
        probes,
        base_registry,
        set(manifest["candidate_scopes"]),
    )
    cache_digests = {
        scope_id: scope_rows[scope_id]["admission_cache_sha256"]
        for scope_id in scope_rows
    }
    if report.get("cache_sha256_by_scope") != cache_digests:
        raise TransactionError(
            "compatibility report cache digests do not match the exact transaction inputs"
        )
    for scope_id, path in cache_paths.items():
        if file_sha256(path) != cache_digests[scope_id]:
            raise TransactionError(
                f"admission cache {scope_id!r} changed during global gate replay"
            )

    parent, target = _safe_output_parent(output)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=parent))
    try:
        assets_directory = temporary / "assets"
        report_directory = temporary / "proposed/release/admission"
        registry_directory = temporary / "proposed/release"
        assets_directory.mkdir()
        report_directory.mkdir(parents=True)
        report_path = report_directory / f"{report_digest}.json"
        report_path.write_bytes(report_bytes)

        assets: list[dict[str, Any]] = []
        measurements: dict[str, str] = {}
        evidence_reports_by_scope: dict[
            str, Mapping[str, dict[str, object]]
        ] = {}
        for scope_id in sorted(cache_paths):
            cache_digest = cache_digests[scope_id]
            cache_destination = assets_directory / f"{cache_digest}.npz"
            cache_bytes = _copy_exclusive(
                cache_paths[scope_id], cache_destination, cache_digest
            )
            measurement_path = build_measurement_bundle(
                cache_paths[scope_id], raw_paths[scope_id], assets_directory
            )
            if file_sha256(cache_paths[scope_id]) != cache_digest:
                raise TransactionError(
                    f"admission cache {scope_id!r} changed while evidence was staged"
                )
            measurement_digest = file_sha256(measurement_path)
            evidence_reports = load_embedded_evidence_reports(cache_paths[scope_id])
            evidence_reports_by_scope[scope_id] = evidence_reports
            validate_measurement_bundle(
                measurement_path,
                scope_id,
                {
                    "sequence_capability_evidence_sha256": loaded[scope_id][0][
                        "sequence_capability_evidence_sha256"
                    ],
                    "placement_evidence_sha256": loaded[scope_id][0][
                        "placement_evidence_sha256"
                    ],
                    "performance_evidence_sha256": loaded[scope_id][0][
                        "performance_evidence_sha256"
                    ],
                },
                evidence_reports,
            )
            measurements[scope_id] = measurement_digest
            assets.extend(
                (
                    {
                        "kind": "admission-cache",
                        "owner_id": scope_id,
                        "filename": cache_destination.name,
                        "sha256": cache_digest,
                        "bytes": cache_bytes,
                        "format": "npz",
                    },
                    {
                        "kind": "measurement-evidence",
                        "owner_id": scope_id,
                        "filename": measurement_path.name,
                        "sha256": measurement_digest,
                        "bytes": measurement_path.stat().st_size,
                        "format": "zip",
                    },
                )
            )

        local_packages, package_assets = _validate_package_rows(
            manifest,
            root,
            loaded,
            evidence_reports_by_scope,
            report_digest,
            manifest["release_tag"],
            variant_catalog,
            assets_directory,
        )
        assets.extend(package_assets)
        assets.sort(key=lambda row: (row["kind"], row["owner_id"]))

        proposed = json.loads(json.dumps(base_registry))
        proposed["profile_status"] = "active"
        proposed["admitted_backends"] = [
            _registry_entry(
                loaded[scope_id][0],
                cache_digests[scope_id],
                measurements[scope_id],
                report_digest,
                manifest["release_tag"],
            )
            for scope_id in sorted(loaded)
        ]
        proposed["local_packages"] = local_packages
        registry_bytes = _canonical_json(proposed, pretty=True)
        if len(registry_bytes) > MAX_ADMISSION_REGISTRY_BYTES:
            raise TransactionError("proposed registry exceeds its byte bound")
        registry_path_out = registry_directory / "inference-backends.json"
        registry_path_out.write_bytes(registry_bytes)
        registry_digest = _sha256_bytes(registry_bytes)

        expected_receipts = []
        package_by_id = {row["package_id"]: row for row in local_packages}
        for package_id in sorted(package_by_id):
            package = package_by_id[package_id]
            for scope_id in package["ordered_scope_ids"]:
                expected_receipts.append(
                    {
                        "package_id": package_id,
                        "package_sha256": package["package_sha256"],
                        "scope_id": scope_id,
                        "cache_sha256": cache_digests[scope_id],
                        "compatibility_report_sha256": report_digest,
                    }
                )

        plan: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "release_tag": manifest["release_tag"],
            "receipt_attestation_public_key": manifest[
                "receipt_attestation_public_key"
            ],
            "base_registry_sha256": manifest["base_registry_sha256"],
            "base_variants_sha256": manifest["base_variants_sha256"],
            "profile_source_promotion": profile_source_promotion,
            "admission_implementation_bundle_sha256": (
                ADMISSION_IMPLEMENTATION_BUNDLE_SHA256
            ),
            "candidate_scopes": sorted(manifest["candidate_scopes"]),
            "cohort_scopes": sorted(loaded),
            "compatibility_report": f"proposed/release/admission/{report_digest}.json",
            "compatibility_report_sha256": report_digest,
            "proposed_registry": "proposed/release/inference-backends.json",
            "proposed_registry_sha256": registry_digest,
            "assets": assets,
            "expected_conformance_receipts": expected_receipts,
        }
        plan["stage_id"] = _stage_id(plan)
        plan_bytes = _canonical_json(plan, pretty=True)
        if len(plan_bytes) > MAX_STAGE_PLAN_BYTES:
            raise TransactionError("stage plan exceeds its byte bound")
        (temporary / "stage-plan.json").write_bytes(plan_bytes)
        os.replace(temporary, target)
        return target / "stage-plan.json"
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


RECEIPT_CLAIM_FIELDS = {
    "schema_version",
    "passed",
    "stage_id",
    "scope_id",
    "package_id",
    "package_sha256",
    "package_bytes",
    "cache_sha256",
    "compatibility_report_sha256",
    "admission_implementation_bundle_sha256",
    "wire_groupings",
    "sequence_buckets",
    "signed_requests",
}
RECEIPT_ENVELOPE_FIELDS = {"schema_version", "receipt", "signature"}


def _load_stage_plan(path: Path) -> tuple[dict[str, Any], Path]:
    plan, _ = _read_json(path.resolve(strict=True), MAX_STAGE_PLAN_BYTES, "stage plan")
    required = {
        "schema_version",
        "stage_id",
        "release_tag",
        "receipt_attestation_public_key",
        "base_registry_sha256",
        "base_variants_sha256",
        "profile_source_promotion",
        "admission_implementation_bundle_sha256",
        "candidate_scopes",
        "cohort_scopes",
        "compatibility_report",
        "compatibility_report_sha256",
        "proposed_registry",
        "proposed_registry_sha256",
        "assets",
        "expected_conformance_receipts",
    }
    _exact_keys(plan, required, set(), "stage plan")
    if plan["schema_version"] != SCHEMA_VERSION:
        raise TransactionError("stage plan schema_version must be 1")
    _validate_profile_source_promotion_claim(plan["profile_source_promotion"])
    if _stage_id(plan) != plan["stage_id"]:
        raise TransactionError("stage plan identity does not match its exact claim")
    return plan, path.resolve(strict=True).parent


def _validate_stage_bytes(plan: dict[str, Any], root: Path) -> None:
    report_path = _resolve_input(
        root, plan["compatibility_report"], "compatibility_report", directory=False
    )
    if file_sha256(report_path) != plan["compatibility_report_sha256"]:
        raise TransactionError("staged compatibility report bytes changed")
    registry_path = _resolve_input(
        root, plan["proposed_registry"], "proposed_registry", directory=False
    )
    if file_sha256(registry_path) != plan["proposed_registry_sha256"]:
        raise TransactionError("staged proposed registry bytes changed")
    assets = plan["assets"]
    if not isinstance(assets, list) or not assets:
        raise TransactionError("stage plan assets must be nonempty")
    total = 0
    filenames: set[str] = set()
    for index, asset in enumerate(assets):
        label = f"stage assets[{index}]"
        if not isinstance(asset, dict):
            raise TransactionError(f"{label} must be an object")
        _exact_keys(
            asset,
            {"kind", "owner_id", "filename", "sha256", "bytes", "format"},
            set(),
            label,
        )
        digest = _digest(asset["sha256"], f"{label}.sha256")
        suffix = "." + asset["format"]
        if asset["filename"] != digest + suffix or asset["filename"] in filenames:
            raise TransactionError(f"{label} is not uniquely content-addressed")
        filenames.add(asset["filename"])
        asset_path = _resolve_input(
            root, f"assets/{asset['filename']}", f"{label}.filename", directory=False
        )
        if type(asset["bytes"]) is not int or asset["bytes"] < 1:
            raise TransactionError(f"{label}.bytes must be positive")
        if asset_path.stat().st_size != asset["bytes"] or file_sha256(asset_path) != digest:
            raise TransactionError(f"{label} exact bytes changed")
        total += asset["bytes"]
        if total > MAX_ACTIVATION_BYTES:
            raise TransactionError("staged transaction exceeds activation byte bound")
    assets_root = _resolve_input(root, "assets", "staged assets", directory=True)
    actual_names: set[str] = set()
    for path in assets_root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise TransactionError("staged assets contains an unplanned or unsafe entry")
        actual_names.add(path.name)
    if actual_names != filenames:
        raise TransactionError("staged assets do not exactly equal the stage plan inventory")


def _load_receipts(
    paths: Iterable[Path], stage_id: str, public_key_hex: str
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], tuple[Path, str]],
]:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
    except ValueError as error:
        raise TransactionError("stage receipt attestation public key is invalid") from error
    receipts: dict[tuple[str, str], dict[str, Any]] = {}
    sources: dict[tuple[str, str], tuple[Path, str]] = {}
    for path in paths:
        if path.is_symlink() or path.resolve() != path.absolute():
            raise TransactionError("receipt must be a regular non-symlink path")
        envelope, raw = _read_json(path.resolve(strict=True), MAX_RECEIPT_BYTES, "receipt")
        if set(envelope) != RECEIPT_ENVELOPE_FIELDS or envelope["schema_version"] != 1:
            raise TransactionError("receipt envelope has an invalid schema")
        receipt = envelope["receipt"]
        signature = envelope["signature"]
        if not isinstance(receipt, dict) or set(receipt) != RECEIPT_CLAIM_FIELDS:
            raise TransactionError("receipt claim has an invalid schema")
        if (
            not isinstance(signature, str)
            or re.fullmatch(r"[0-9a-f]{128}", signature) is None
        ):
            raise TransactionError("receipt signature must be a lowercase Ed25519 signature")
        try:
            public_key.verify(bytes.fromhex(signature), _canonical_json(receipt))
        except InvalidSignature as error:
            raise TransactionError("receipt signature is invalid") from error
        digest = _sha256_bytes(raw)
        if path.name != f"{digest}.json":
            raise TransactionError("receipt filename must equal its exact SHA-256")
        if (
            receipt["schema_version"] != SCHEMA_VERSION
            or receipt["passed"] is not True
            or receipt["stage_id"] != stage_id
            or receipt["wire_groupings"] != 64
            or receipt["sequence_buckets"] != 7
            or receipt["signed_requests"] != EXPECTED_SIGNED_REQUESTS
            or type(receipt["package_bytes"]) is not int
            or receipt["package_bytes"] < 1
        ):
            raise TransactionError("receipt did not pass the complete final conformance gate")
        for field in (
            "stage_id",
            "package_sha256",
            "cache_sha256",
            "compatibility_report_sha256",
            "admission_implementation_bundle_sha256",
        ):
            _digest(receipt[field], f"receipt {field}")
        key = (
            _slug(receipt["package_id"], "receipt package_id"),
            _slug(receipt["scope_id"], "receipt scope_id"),
        )
        if key in receipts:
            raise TransactionError(f"duplicate final conformance receipt for {key}")
        receipts[key] = receipt
        sources[key] = (path.resolve(strict=True), digest)
    return receipts, sources


def activate_transaction(
    stage_plan_path: Path, receipt_paths: Sequence[Path], output: Path
) -> Path:
    """Create a release-ready bundle only when final package receipts are complete."""
    verify_implementation_bundle()
    if (
        stage_plan_path.is_symlink()
        or stage_plan_path.resolve() != stage_plan_path.absolute()
    ):
        raise TransactionError("stage plan must be a regular non-symlink path")
    plan, stage_root = _load_stage_plan(stage_plan_path)
    if (
        plan["admission_implementation_bundle_sha256"]
        != ADMISSION_IMPLEMENTATION_BUNDLE_SHA256
    ):
        raise TransactionError("stage plan uses another admission implementation")
    repository = Path(__file__).resolve().parents[2]
    if (
        file_sha256(repository / "release/inference-backends.json")
        != plan["base_registry_sha256"]
        or file_sha256(repository / "release/variants.json")
        != plan["base_variants_sha256"]
    ):
        raise TransactionError(
            "stage plan no longer matches the checked-out base registry and variants"
        )
    current_promotion = _profile_source_promotion(repository)
    if current_promotion != plan["profile_source_promotion"]:
        raise TransactionError(
            "stage plan no longer matches the checked-out embedding profile source"
        )
    _validate_stage_bytes(plan, stage_root)
    receipts, receipt_sources = _load_receipts(
        receipt_paths,
        plan["stage_id"],
        plan["receipt_attestation_public_key"],
    )
    expected_rows = plan["expected_conformance_receipts"]
    if not isinstance(expected_rows, list) or not expected_rows:
        raise TransactionError("stage plan has no expected final conformance receipts")
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(expected_rows):
        if not isinstance(row, dict) or set(row) != {
            "package_id",
            "package_sha256",
            "scope_id",
            "cache_sha256",
            "compatibility_report_sha256",
        }:
            raise TransactionError(f"expected receipt {index} has an invalid schema")
        key = (row["package_id"], row["scope_id"])
        if key in expected:
            raise TransactionError("stage plan repeats an expected receipt")
        expected[key] = row
    if set(receipts) != set(expected):
        missing = sorted(set(expected).difference(receipts))
        unexpected = sorted(set(receipts).difference(expected))
        raise TransactionError(
            f"final conformance receipt coverage is incomplete; missing={missing}, "
            f"unexpected={unexpected}"
        )
    for key, row in expected.items():
        receipt = receipts[key]
        for field in (
            "package_id",
            "scope_id",
            "package_sha256",
            "cache_sha256",
            "compatibility_report_sha256",
        ):
            if receipt[field] != row[field]:
                raise TransactionError(f"receipt {key} {field} does not match the stage")
        if (
            receipt["admission_implementation_bundle_sha256"]
            != plan["admission_implementation_bundle_sha256"]
        ):
            raise TransactionError(f"receipt {key} uses another admission implementation")
        package_asset = next(
            (
                asset
                for asset in plan["assets"]
                if asset["kind"] == "target-package"
                and asset["owner_id"] == key[0]
            ),
            None,
        )
        if package_asset is None or receipt["package_bytes"] != package_asset["bytes"]:
            raise TransactionError(f"receipt {key} does not bind the exact package size")

    parent, target = _safe_output_parent(output)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=parent))
    try:
        activation_assets = temporary / "assets"
        activation_release = temporary / "release"
        activation_admission = activation_release / "admission"
        activation_assets.mkdir()
        activation_admission.mkdir(parents=True)
        activation_asset_inventory = []
        for asset in plan["assets"]:
            source = _resolve_input(
                stage_root,
                f"assets/{asset['filename']}",
                "staged activation asset",
                directory=False,
            )
            copied = activation_assets / asset["filename"]
            copied_bytes = _copy_exclusive(source, copied, asset["sha256"])
            if copied_bytes != asset["bytes"] or file_sha256(copied) != asset["sha256"]:
                raise TransactionError("activation asset changed while it was copied")
            activation_asset_inventory.append(
                {**asset, "path": f"assets/{asset['filename']}"}
            )
        report_source = _resolve_input(
            stage_root,
            plan["compatibility_report"],
            "staged compatibility report",
            directory=False,
        )
        report_destination = activation_admission / report_source.name
        report_bytes = _copy_exclusive(
            report_source,
            report_destination,
            plan["compatibility_report_sha256"],
        )
        registry_source = _resolve_input(
            stage_root,
            plan["proposed_registry"],
            "staged proposed registry",
            directory=False,
        )
        registry_destination = activation_release / "inference-backends.json"
        registry_bytes = _copy_exclusive(
            registry_source,
            registry_destination,
            plan["proposed_registry_sha256"],
        )
        receipt_directory = temporary / "receipts"
        receipt_directory.mkdir()
        receipt_inventory = []
        for key in sorted(receipts):
            source, receipt_digest = receipt_sources[key]
            destination = receipt_directory / source.name
            receipt_bytes = _copy_exclusive(source, destination, receipt_digest)
            receipt_inventory.append(
                {
                    "package_id": key[0],
                    "scope_id": key[1],
                    "receipt": f"receipts/{source.name}",
                    "receipt_sha256": receipt_digest,
                    "receipt_bytes": receipt_bytes,
                }
            )
        activation = {
            "schema_version": SCHEMA_VERSION,
            "stage_id": plan["stage_id"],
            "release_tag": plan["release_tag"],
            "base_registry_sha256": plan["base_registry_sha256"],
            "base_variants_sha256": plan["base_variants_sha256"],
            "admission_implementation_bundle_sha256": plan[
                "admission_implementation_bundle_sha256"
            ],
            "profile_source_promotion": plan["profile_source_promotion"],
            "compatibility_report": (
                f"release/admission/{report_source.name}"
            ),
            "compatibility_report_sha256": plan[
                "compatibility_report_sha256"
            ],
            "compatibility_report_bytes": report_bytes,
            "registry": "release/inference-backends.json",
            "registry_sha256": plan["proposed_registry_sha256"],
            "registry_bytes": registry_bytes,
            "assets": activation_asset_inventory,
            "receipts": receipt_inventory,
            "status": "release-ready-not-published",
        }
        activation_bytes = _canonical_json(activation, pretty=True)
        activation_digest = _sha256_bytes(activation_bytes)
        (temporary / f"{activation_digest}.activation.json").write_bytes(
            activation_bytes
        )
        os.replace(temporary, target)
        return target / f"{activation_digest}.activation.json"
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_receipt_private_key(path: Path, expected_public_hex: str) -> Path:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise TransactionError(
            "receipt attestation private key must be a regular non-symlink file"
        )
    private_bytes = _read_bounded(
        resolved, 32, "receipt attestation private key"
    )
    if len(private_bytes) != 32:
        raise TransactionError(
            "receipt attestation private key must contain exactly 32 raw bytes"
        )
    try:
        private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
    except ValueError as error:
        raise TransactionError("receipt attestation private key is invalid") from error
    public_hex = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    if public_hex != expected_public_hex:
        raise TransactionError(
            "receipt attestation private key does not match the transaction manifest"
        )
    return resolved


def _asset_for(
    plan: Mapping[str, object], kind: str, owner_id: str
) -> Mapping[str, object]:
    assets = plan.get("assets")
    if not isinstance(assets, list):
        raise TransactionError("stage plan assets must be an array")
    matches = [
        asset
        for asset in assets
        if isinstance(asset, dict)
        and asset.get("kind") == kind
        and asset.get("owner_id") == owner_id
    ]
    if len(matches) != 1:
        raise TransactionError(
            f"stage plan must contain one {kind} asset for {owner_id!r}"
        )
    return matches[0]


def _receipt_execution_rows(
    plan: dict[str, Any], stage_root: Path
) -> list[dict[str, object]]:
    registry_path = _resolve_input(
        stage_root, plan["proposed_registry"], "proposed_registry", directory=False
    )
    registry, _ = _read_json(
        registry_path, MAX_ADMISSION_REGISTRY_BYTES, "proposed registry"
    )
    packages = registry.get("local_packages")
    if not isinstance(packages, list) or not packages:
        raise TransactionError("proposed registry has no local packages")
    packages_by_id: dict[str, dict[str, object]] = {}
    for index, package in enumerate(packages):
        if not isinstance(package, dict):
            raise TransactionError(f"local_packages[{index}] must be an object")
        package_id = _slug(package.get("package_id"), f"local_packages[{index}].package_id")
        if package_id in packages_by_id:
            raise TransactionError("proposed registry repeats a local package id")
        packages_by_id[package_id] = package

    expected = plan.get("expected_conformance_receipts")
    if not isinstance(expected, list) or not expected:
        raise TransactionError("stage plan has no expected conformance receipts")
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(expected):
        label = f"expected_conformance_receipts[{index}]"
        if not isinstance(row, dict):
            raise TransactionError(f"{label} must be an object")
        _exact_keys(
            row,
            {
                "package_id",
                "package_sha256",
                "scope_id",
                "cache_sha256",
                "compatibility_report_sha256",
            },
            set(),
            label,
        )
        package_id = _slug(row["package_id"], f"{label}.package_id")
        scope_id = _slug(row["scope_id"], f"{label}.scope_id")
        pair = (package_id, scope_id)
        if pair in seen:
            raise TransactionError("stage plan repeats a package/scope receipt")
        seen.add(pair)
        package_digest = _digest(row["package_sha256"], f"{label}.package_sha256")
        cache_digest = _digest(row["cache_sha256"], f"{label}.cache_sha256")
        if row["compatibility_report_sha256"] != plan["compatibility_report_sha256"]:
            raise TransactionError(f"{label} does not bind the newest cohort report")
        package_asset = _asset_for(plan, "target-package", package_id)
        cache_asset = _asset_for(plan, "admission-cache", scope_id)
        if (
            package_asset.get("sha256") != package_digest
            or package_asset.get("format") != "zip"
            or cache_asset.get("sha256") != cache_digest
            or cache_asset.get("format") != "npz"
        ):
            raise TransactionError(f"{label} does not bind its exact staged assets")
        package = packages_by_id.get(package_id)
        if package is None:
            raise TransactionError(f"{label} names an unknown local package")
        ordered = package.get("ordered_scope_ids")
        dispatcher = package.get("dispatcher")
        if (
            not isinstance(ordered, list)
            or scope_id not in ordered
            or not isinstance(dispatcher, dict)
            or not isinstance(dispatcher.get("binary"), str)
            or PurePosixPath(dispatcher["binary"]).name != dispatcher["binary"]
        ):
            raise TransactionError(f"{label} has no valid staged dispatcher binding")
        rows.append(
            {
                "package_id": package_id,
                "scope_id": scope_id,
                "package_sha256": package_digest,
                "cache_sha256": cache_digest,
                "package_asset": _resolve_input(
                    stage_root,
                    f"assets/{package_asset['filename']}",
                    f"{label} package asset",
                    directory=False,
                ),
                "cache_asset": _resolve_input(
                    stage_root,
                    f"assets/{cache_asset['filename']}",
                    f"{label} cache asset",
                    directory=False,
                ),
                "dispatcher": dispatcher["binary"],
            }
        )
    return rows


def _run_final_conformance_receipt(
    stage_plan: Path,
    row: Mapping[str, object],
    private_key: Path,
    receipt_directory: Path,
    scifact_snapshot: Path,
) -> Path:
    before = set(receipt_directory.iterdir())
    plan, _ = _load_stage_plan(stage_plan)
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "experiments/embedding-profile/final_package_conformance.py"),
        "--cache",
        str(row["cache_asset"]),
        "--cache-sha256",
        str(row["cache_sha256"]),
        "--compatibility-report-sha256",
        str(plan["compatibility_report_sha256"]),
        "--stage-id",
        str(plan["stage_id"]),
        "--stage-plan",
        str(stage_plan),
        "--package-id",
        str(row["package_id"]),
        "--package-asset",
        str(row["package_asset"]),
        "--package-sha256",
        str(row["package_sha256"]),
        "--dispatcher",
        str(row["dispatcher"]),
        "--receipt-attestation-private-key",
        str(private_key),
        "--receipt-directory",
        str(receipt_directory),
        "--scifact-snapshot",
        str(scifact_snapshot),
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise TransactionError(
            "final package conformance failed for "
            f"{row['package_id']!r}/{row['scope_id']!r}"
        )
    created = set(receipt_directory.iterdir()).difference(before)
    receipt = next(iter(created)) if len(created) == 1 else None
    if receipt is None or receipt.is_symlink() or not receipt.is_file():
        raise TransactionError(
            "final package conformance did not create exactly one receipt"
        )
    return receipt


def _build_publication_plan(
    activation_manifest: Path, base_registry_replay: Mapping[str, object]
) -> tuple[dict[str, object], bytes, str]:
    activation, raw = _read_json(
        activation_manifest,
        MAX_STAGE_PLAN_BYTES,
        "activation manifest",
    )
    activation_digest = _sha256_bytes(raw)
    if activation_manifest.name != f"{activation_digest}.activation.json":
        raise TransactionError(
            "activation manifest filename does not match its exact bytes"
        )
    if activation.get("status") != "release-ready-not-published":
        raise TransactionError("activation bundle is not release ready")
    release_tag = activation.get("release_tag")
    if (
        not isinstance(release_tag, str)
        or RELEASE_TAG_RE.fullmatch(release_tag) is None
        or release_tag.lower() in {"latest", "current", "draft"}
    ):
        raise TransactionError("activation release tag is invalid")
    assets = activation.get("assets")
    if not isinstance(assets, list) or not assets:
        raise TransactionError("activation bundle has no release assets")
    release_assets = []
    for index, asset in enumerate(assets):
        label = f"activation assets[{index}]"
        if not isinstance(asset, dict):
            raise TransactionError(f"{label} must be an object")
        digest = _digest(asset.get("sha256"), f"{label}.sha256")
        size = asset.get("bytes")
        filename = asset.get("filename")
        source = asset.get("path")
        if type(size) is not int or size < 1 or not isinstance(filename, str):
            raise TransactionError(f"{label} has invalid size or filename")
        if source != f"assets/{filename}" or filename != f"{digest}.{asset.get('format')}":
            raise TransactionError(f"{label} is not canonically content-addressed")
        source_path = _resolve_input(
            activation_manifest.parent, source, label, directory=False
        )
        if source_path.stat().st_size != size or file_sha256(source_path) != digest:
            raise TransactionError(f"{label} bytes do not match the activation")
        release_assets.append(
            {
                "kind": asset.get("kind"),
                "owner_id": asset.get("owner_id"),
                "source": f"activation/{source}",
                "filename": filename,
                "sha256": digest,
                "bytes": size,
                "url": (
                    f"https://github.com/{RELEASE_REPOSITORY}/releases/download/"
                    f"{release_tag}/{filename}"
                ),
            }
        )

    report_source = _resolve_input(
        activation_manifest.parent,
        activation["compatibility_report"],
        "activation compatibility report",
        directory=False,
    )
    registry_source = _resolve_input(
        activation_manifest.parent,
        activation["registry"],
        "activation registry",
        directory=False,
    )
    if (
        file_sha256(report_source) != activation["compatibility_report_sha256"]
        or file_sha256(registry_source) != activation["registry_sha256"]
    ):
        raise TransactionError("activation repository bytes changed")
    publication: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready-not-published",
        "repository": RELEASE_REPOSITORY,
        "release_tag": release_tag,
        "stage_id": activation["stage_id"],
        "activation_manifest": {
            "path": f"activation/{activation_manifest.name}",
            "sha256": activation_digest,
            "bytes": len(raw),
        },
        "base_registry_replay": dict(base_registry_replay),
        "release_assets": release_assets,
        "repository_activation": {
            "compatibility_report": {
                "source": f"activation/{activation['compatibility_report']}",
                "destination": activation["compatibility_report"],
                "sha256": activation["compatibility_report_sha256"],
                "bytes": activation["compatibility_report_bytes"],
            },
            "registry": {
                "source": f"activation/{activation['registry']}",
                "destination": activation["registry"],
                "sha256": activation["registry_sha256"],
                "bytes": activation["registry_bytes"],
            },
            "profile_source_promotion": activation["profile_source_promotion"],
        },
        "required_order": [
            "publish every release asset to the fixed tag",
            "download every asset through credential-free GitHub HTTPS and verify exact bytes",
            "apply the bound repository activation",
            "replay the nonempty release registry",
        ],
    }
    publication_bytes = _canonical_json(publication, pretty=True)
    if len(publication_bytes) > MAX_PUBLICATION_PLAN_BYTES:
        raise TransactionError("publication plan exceeds its byte bound")
    return publication, publication_bytes, _sha256_bytes(publication_bytes)


def run_transaction(
    manifest_path: Path, private_key_path: Path, output: Path
) -> Path:
    """Run the complete local transaction without publishing or editing the checkout."""
    verify_implementation_bundle()
    manifest, root = _parse_manifest(manifest_path)
    scifact_snapshot = _transaction_scifact_snapshot(manifest, root)
    private_key = _validate_receipt_private_key(
        private_key_path, manifest["receipt_attestation_public_key"]
    )
    current_registry = REPOSITORY_ROOT / "release/inference-backends.json"
    current_variants = REPOSITORY_ROOT / "release/variants.json"
    manifest_registry = _resolve_input(
        root, manifest["base_registry"], "base_registry", directory=False
    )
    manifest_variants = _resolve_input(
        root, manifest["base_variants"], "base_variants", directory=False
    )
    if (
        file_sha256(manifest_registry) != file_sha256(current_registry)
        or manifest["base_registry_sha256"] != file_sha256(current_registry)
        or file_sha256(manifest_variants) != file_sha256(current_variants)
        or manifest["base_variants_sha256"] != file_sha256(current_variants)
    ):
        raise TransactionError(
            "one-command transaction inputs must bind the current checkout registry and variants"
        )
    base_replay = verify_release_registry()

    parent, target = _safe_output_parent(output)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=parent))
    try:
        stage_plan = stage_transaction(manifest_path, temporary / "stage")
        plan, stage_root = _load_stage_plan(stage_plan)
        _validate_stage_bytes(plan, stage_root)
        rows = _receipt_execution_rows(plan, stage_root)
        receipts_directory = temporary / "receipts"
        receipts_directory.mkdir()
        receipts = [
            _run_final_conformance_receipt(
                stage_plan,
                row,
                private_key,
                receipts_directory,
                scifact_snapshot,
            )
            for row in rows
        ]
        activation_manifest = activate_transaction(
            stage_plan, receipts, temporary / "activation"
        )
        _, publication_bytes, publication_digest = _build_publication_plan(
            activation_manifest, base_replay
        )
        publication_name = f"{publication_digest}.publication.json"
        (temporary / publication_name).write_bytes(publication_bytes)
        os.replace(temporary, target)
        return target / publication_name
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def generate_receipt_attestation_key(output: Path) -> str:
    """Create one raw private key and return its manifest-safe public key hex."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    parent, target = _safe_output_parent(output)
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    descriptor = os.open(
        parent / target.name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(private_bytes)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return public.hex()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    keygen = commands.add_parser(
        "keygen", help="create a dedicated raw Ed25519 receipt attestation key"
    )
    keygen.add_argument("--output", required=True, type=Path)
    stage = commands.add_parser("stage", help="build an offline immutable stage")
    stage.add_argument("--manifest", required=True, type=Path)
    stage.add_argument("--output", required=True, type=Path)
    activate = commands.add_parser(
        "activate", help="build a release-ready bundle after final conformance"
    )
    activate.add_argument("--stage-plan", required=True, type=Path)
    activate.add_argument("--receipt", action="append", required=True, type=Path)
    activate.add_argument("--output", required=True, type=Path)
    run = commands.add_parser(
        "run",
        help=(
            "replay the current registry, stage the cohort, run every exact "
            "package receipt, and build an immutable publication plan"
        ),
    )
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument(
        "--receipt-attestation-private-key", required=True, type=Path
    )
    run.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "keygen":
            result = generate_receipt_attestation_key(args.output)
        elif args.command == "stage":
            result = stage_transaction(args.manifest, args.output)
        elif args.command == "activate":
            result = activate_transaction(args.stage_plan, args.receipt, args.output)
        else:
            result = run_transaction(
                args.manifest,
                args.receipt_attestation_private_key,
                args.output,
            )
    except (OSError, RuntimeError, TransactionError, ValueError) as error:
        print(f"admission transaction refused: {error}")
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
