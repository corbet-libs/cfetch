"""Strict, dependency-free validation for an OpenVINO target package.

The package manifest is the only authority that maps a cfetch execution scope
to an OpenVINO device.  In particular, callers cannot pass a free-form device
name to the adapter and OpenVINO AUTO/MULTI/HETERO selection is never used.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

if __package__:
    from .legal import LEGAL_FILES, PINNED_LEGAL_SHA256
else:
    from legal import LEGAL_FILES, PINNED_LEGAL_SHA256  # type: ignore[no-redef]


PROFILE_ID = "cfetch-embedding-v1"
PROFILE_MANIFEST_SHA256 = (
    "59210a333494f788eb8e607fe38cabb6af1a7aa7cdf604ddf52e3fa6004b5afb"
)
ADMISSION_POLICY_SHA256 = (
    "36375d4e6fb5f6a48e5ecb7036ccc141c2314c2f2609c37063a6f79aa367fcc2"
)
MODEL = "google/embeddinggemma-300m"
MODEL_REVISION = "57c266a740f537b4dc058e1b0cda161fd15afa75"
# The canonical Google repository is gated at the transport layer.  This
# immutable public mirror commit contains the exact same required bytes; every
# file is still accepted only by the canonical SHA-256 allowlist below.
SOURCE_MIRROR = "unsloth/embeddinggemma-300m"
SOURCE_MIRROR_REVISION = "bfa3c846ac738e62aa61806ef9112d34acb1dc5a"
DIMENSIONS = 768
MAX_TOKENS = 2048
MAX_WIRE_BATCH_SIZE = 64
SEQUENCE_BUCKETS = (32, 64, 128, 257, 512, 1024, 2048)
PINNED_SOURCE_FILE_SHA256 = {
    "model.safetensors": "cbf5a78393b6a033e0b8a63a57549964f7ed5c6fbeb4ba0694214f36123f2fd2",
    "2_Dense/model.safetensors": "c327f2acb00149676ade24a75e11eb6ebbd367f9ee050267ba56829d2979f702",
    "3_Dense/model.safetensors": "ffb6cc5162e11e2ce6bc2367e121ee3bbbc4e82e1ee26826bd7573d4948d81b8",
    "tokenizer.json": "6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e",
    "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
    "tokenizer_config.json": "9076840490613047bc9115963ee96b7702018b0d26ba644240bf856efda93118",
    "config.json": "8f863f76e2d9c710cc833dc92efa898c9adfd41031c786507cc6b0e49c2e3e68",
    "special_tokens_map.json": "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397",
    "modules.json": "5b5649645fb756dad1a8e2efe7872d3bb32bc00b93c95f276dd17f474eedccdc",
    "sentence_bert_config.json": "5ea26221ce733ace29a3897360e7c6ac8816b2ca0f7306657d69e594fece7325",
    "1_Pooling/config.json": "35bbd47d7fdf1e378db6130bcc668b09d1aa67a7bbf7c8f89a9c71f4cc8ebcc6",
    "2_Dense/config.json": "0661e5e0b67b8f8408ab31ab5d073a78972fc1dc24a49992a64796557e4f9e53",
    "3_Dense/config.json": "8c4575c49353d63fb907878856ba94384635c3b2711fd5b7439e7f71888c66fc",
}

SCOPE_ID_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
LOWER_HEX_32_RE = re.compile(r"[0-9a-f]{64}")
DEVICE_FOR_CLASS = {"npu": "NPU", "gpu": "GPU", "cpu": "CPU"}
REQUIRED_OPENVINO_PROPERTY_TYPES = {
    "npu": {
        "FULL_DEVICE_NAME": str,
        "DEVICE_ARCHITECTURE": str,
        "NPU_DRIVER_VERSION": int,
        "NPU_COMPILER_VERSION": int,
    },
    "gpu": {
        "FULL_DEVICE_NAME": str,
        "DEVICE_ARCHITECTURE": str,
        "GPU_UARCH_VERSION": str,
        "GPU_DEVICE_ID": str,
    },
    "cpu": {
        "FULL_DEVICE_NAME": str,
        "DEVICE_ARCHITECTURE": str,
    },
}
MAX_JSON_BYTES = 1024 * 1024
HOST_FILE_PREFIXES = (
    PurePosixPath("/usr/lib"),
    PurePosixPath("/usr/lib64"),
    PurePosixPath("/lib"),
    PurePosixPath("/lib64"),
    PurePosixPath("/opt/intel"),
)
GOVERNOR_POLICY_PATH = PurePosixPath("/var/lib/cfetch/inference/policy.json")


class ManifestError(ValueError):
    """A target package does not implement its declared immutable contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def read_bounded_file(path: Path, limit: int, label: str) -> bytes:
    if limit < 1:
        raise ValueError("bounded file limit must be positive")
    try:
        metadata = path.stat()
    except OSError as error:
        raise ManifestError(f"cannot inspect {label}: {error}") from error
    if path.is_symlink() or not path.is_file():
        raise ManifestError(f"{label} must be a regular non-symlink file")
    if metadata.st_size < 1 or metadata.st_size > limit:
        raise ManifestError(
            f"{label} must contain 1..{limit} bytes"
        )
    try:
        with path.open("rb") as source:
            raw = source.read(limit + 1)
    except OSError as error:
        raise ManifestError(f"cannot read {label}: {error}") from error
    if not raw or len(raw) > limit:
        raise ManifestError(f"{label} changed size while it was read")
    return raw


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = read_bounded_file(path, MAX_JSON_BYTES, str(path))
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"{path} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return value, raw


def _require_exact_keys(
    value: Mapping[str, Any], required: set[str], optional: set[str], label: str
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ManifestError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ManifestError(f"{label} has unknown fields: {', '.join(unknown)}")


def _string(value: Any, label: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ManifestError(f"{label} must be a non-empty string of at most {maximum} characters")
    if "\x00" in value or "\r" in value or "\n" in value:
        raise ManifestError(f"{label} must be a single line without NUL bytes")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or LOWER_HEX_32_RE.fullmatch(value) is None:
        raise ManifestError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _relative_file(root: Path, value: Any, label: str) -> Path:
    text = _string(value, label, 512)
    pure = PurePosixPath(text)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise ManifestError(f"{label} must be a normalized package-relative path")
    root = root.resolve()
    unresolved = root / Path(*pure.parts)
    if unresolved.is_symlink():
        raise ManifestError(f"{label} must not be a symlink")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ManifestError(f"{label} escapes the target package") from error
    if not resolved.is_file():
        raise ManifestError(f"{label} does not name a regular package file: {text}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _primitive_config(value: Any, label: str) -> dict[str, str | int | float | bool]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    result: dict[str, str | int | float | bool] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ManifestError(f"{label} keys must be non-empty strings of at most 128 characters")
        if not isinstance(item, (str, int, float, bool)) or item is None:
            raise ManifestError(f"{label}.{key} must be a string, number, or boolean")
        if isinstance(item, float) and not math.isfinite(item):
            raise ManifestError(f"{label}.{key} must be finite")
        result[key] = item
    return result


def _required_openvino_properties(
    value: Any, device_class: str, label: str
) -> dict[str, str | int]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    property_types = REQUIRED_OPENVINO_PROPERTY_TYPES[device_class]
    _require_exact_keys(value, set(property_types), set(), label)
    result: dict[str, str | int] = {}
    for name, expected_type in property_types.items():
        item = value[name]
        if type(item) is not expected_type:
            raise ManifestError(
                f"{label}.{name} must be a {expected_type.__name__}"
            )
        if expected_type is str:
            result[name] = _string(item, f"{label}.{name}", 4096)
        else:
            if not -(1 << 63) <= item < (1 << 64):
                raise ManifestError(f"{label}.{name} is outside the supported integer range")
            result[name] = item
    return result


@dataclass(frozen=True)
class HostFileBinding:
    path: Path
    sha256: str


@dataclass(frozen=True)
class HostBinding:
    system: str
    machine: str
    kernel_release: str
    files: tuple[HostFileBinding, ...]


def _host_binding(value: Any, label: str) -> HostBinding:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    _require_exact_keys(
        value, {"system", "machine", "kernel_release", "files"}, set(), label
    )
    if value["system"] != "Linux" or value["machine"] != "x86_64":
        raise ManifestError(f"{label} must target exactly Linux x86_64")
    kernel_release = _string(
        value["kernel_release"], f"{label}.kernel_release", 256
    )
    file_documents = value["files"]
    if not isinstance(file_documents, list) or not 1 <= len(file_documents) <= 16:
        raise ManifestError(f"{label}.files must contain 1..16 host-file bindings")
    files: list[HostFileBinding] = []
    seen: set[str] = set()
    for index, entry in enumerate(file_documents):
        file_label = f"{label}.files[{index}]"
        if not isinstance(entry, dict):
            raise ManifestError(f"{file_label} must be a JSON object")
        _require_exact_keys(entry, {"path", "sha256"}, set(), file_label)
        text = _string(entry["path"], f"{file_label}.path", 512)
        pure = PurePosixPath(text)
        if (
            not pure.is_absolute()
            or str(pure) != text
            or any(part in ("", ".", "..") for part in pure.parts)
            or (pure != GOVERNOR_POLICY_PATH
                and not any(pure.is_relative_to(prefix) for prefix in HOST_FILE_PREFIXES))
        ):
            raise ManifestError(
                f"{file_label}.path must be a normalized absolute file under an "
                "allowlisted system library directory"
            )
        if text in seen:
            raise ManifestError(f"{label}.files contains duplicate path {text!r}")
        seen.add(text)
        files.append(
            HostFileBinding(
                path=Path(text),
                sha256=_digest(entry["sha256"], f"{file_label}.sha256"),
            )
        )
    bound_names = {binding.path.name for binding in files}
    required_cxx_runtime = {
        "libstdc++.so.6": re.compile(r"libstdc\+\+\.so\.6(?:\.[0-9]+)*"),
        "libgcc_s.so.1": re.compile(r"libgcc_s\.so\.1(?:\.[0-9]+)*"),
    }
    missing = [
        soname
        for soname, pattern in required_cxx_runtime.items()
        if not any(pattern.fullmatch(name) is not None for name in bound_names)
    ]
    if missing:
        raise ManifestError(
            f"{label}.files must bind the resolved regular target file for "
            + " and ".join(missing)
        )
    return HostBinding(
        system="Linux",
        machine="x86_64",
        kernel_release=kernel_release,
        files=tuple(files),
    )


@dataclass(frozen=True)
class Artifact:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    graph_xml: Path
    graph_bin: Path
    tokenizer_json: Path
    input_ids_name: str
    attention_mask_name: str
    output_name: str
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    files: tuple[Path, ...]
    conversion_versions: Mapping[str, str]


@dataclass(frozen=True)
class Scope:
    package_state: str
    scope_id: str
    backend: str
    transport: str
    runtime: str
    compiler: str
    package_target: str
    artifact_source: str
    artifact_sha256: str
    internal_precision: str
    device_class: str
    device: str
    openvino_device: str
    openvino_compile_config: Mapping[str, str | int | float | bool]
    required_openvino_properties: Mapping[str, str | int]
    required_execution_devices: tuple[str, ...]
    required_host: HostBinding
    placement_evidence_sha256: str | None
    sequence_capability_evidence_sha256: str | None
    performance_evidence_sha256: str | None
    compatibility_report_sha256: str | None
    attestation_public_key: str
    attestation_private_key_file: Path
    accelerated_placement: bool

    def execution_document(self) -> dict[str, Any]:
        execution: dict[str, Any] = {
            "package_state": self.package_state,
            "scope_id": self.scope_id,
            "backend": self.backend,
            "transport": self.transport,
            "runtime": self.runtime,
            "compiler": self.compiler,
            "package_target": self.package_target,
            "artifact_source": self.artifact_source,
            "device_class": self.device_class,
            "device": self.device,
            "artifact_sha256": self.artifact_sha256,
            "internal_precision": self.internal_precision,
            "placement_evidence_sha256": self.placement_evidence_sha256,
            "supported_max_tokens": MAX_TOKENS,
            "supported_sequence_buckets": list(SEQUENCE_BUCKETS),
            "supported_max_batch_size": MAX_WIRE_BATCH_SIZE,
            "sequence_capability_evidence_sha256": self.sequence_capability_evidence_sha256,
            "performance_evidence_sha256": self.performance_evidence_sha256,
            "compatibility_report_sha256": self.compatibility_report_sha256,
            "accelerated_placement": self.accelerated_placement,
        }
        return execution


@dataclass(frozen=True)
class PackageManifest:
    path: Path
    artifact: Artifact
    scopes: Mapping[str, Scope]
    dependency_versions: Mapping[str, str]
    runtime_manifest_sha256: str
    package_state: str

    def scope(self, scope_id: str) -> Scope:
        try:
            return self.scopes[scope_id]
        except KeyError as error:
            raise ManifestError(
                f"scope {scope_id!r} is not present in the exact target package manifest"
            ) from error


def load_artifact(root: Path, relative_manifest: Any, expected_sha256: Any) -> Artifact:
    manifest_path = _relative_file(root, relative_manifest, "artifact_manifest")
    document, raw = _read_json(manifest_path)
    actual_manifest_sha256 = hashlib.sha256(raw).hexdigest()
    expected_manifest_sha256 = _digest(expected_sha256, "artifact_manifest_sha256")
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ManifestError(
            "artifact manifest digest mismatch: "
            f"expected {expected_manifest_sha256}, found {actual_manifest_sha256}"
        )
    _require_exact_keys(
        document,
        {
            "schema_version",
            "artifact_format",
            "source",
            "semantic_pipeline",
            "sequence_buckets",
            "graph",
            "tokenizer",
            "legal",
            "files",
            "conversion",
        },
        set(),
        "artifact manifest",
    )
    if document["schema_version"] != 1:
        raise ManifestError("artifact manifest schema_version must be 1")
    if document["artifact_format"] != "openvino-ir-dynamic-sequence-static-buckets-v1":
        raise ManifestError("artifact manifest has an unsupported artifact_format")
    source = document["source"]
    if not isinstance(source, dict):
        raise ManifestError("artifact manifest source must be an object")
    _require_exact_keys(
        source,
        {"model", "revision", "files", "acquisition"},
        set(),
        "artifact source",
    )
    if source["model"] != MODEL or source["revision"] != MODEL_REVISION:
        raise ManifestError("artifact source does not match the frozen model and revision")
    if source["files"] != PINNED_SOURCE_FILE_SHA256:
        raise ManifestError("artifact source file digests do not match the pinned exact revision")
    if source["acquisition"] != {
        "repository": SOURCE_MIRROR,
        "revision": SOURCE_MIRROR_REVISION,
        "mode": "public-byte-identical-mirror",
    }:
        raise ManifestError("artifact source acquisition is not the pinned byte mirror")
    pipeline = document["semantic_pipeline"]
    expected_pipeline = {
        "dimensions": DIMENSIONS,
        "pooling": "attention-mask-weighted-mean-include-prompt",
        "dense_2": "linear-768x3072-identity",
        "dense_3": "linear-3072x768-identity",
        "normalization": "l2",
        "truncation": "disabled",
        "padding": "right-attention-mask-excludes-padding",
    }
    if pipeline != expected_pipeline:
        raise ManifestError("artifact semantic_pipeline is not the frozen cfetch pipeline")
    if document["sequence_buckets"] != list(SEQUENCE_BUCKETS):
        raise ManifestError("artifact must declare all seven frozen sequence buckets")
    conversion = document["conversion"]
    if not isinstance(conversion, dict):
        raise ManifestError("artifact conversion must be an object")
    _require_exact_keys(
        conversion,
        {
            "recipe",
            "export",
            "weight_storage",
            "openvino",
            "safetensors",
            "torch",
            "transformers",
        },
        set(),
        "artifact conversion",
    )
    if conversion["recipe"] != "packages/openvino/convert.py":
        raise ManifestError("artifact conversion recipe is not the pinned OpenVINO recipe")
    if conversion["export"] != (
        "torch-export-bounded-dynamic-sequence-1-to-2048-"
        "unit-reduction-matmul-rewrite-v1"
    ):
        raise ManifestError("artifact conversion did not use the bounded dynamic export")
    if conversion["weight_storage"] not in ("f16", "f32"):
        raise ManifestError("artifact conversion weight_storage must be f16 or f32")
    for dependency in ("openvino", "safetensors", "torch", "transformers"):
        _string(conversion[dependency], f"artifact conversion.{dependency}", 128)

    files = document["files"]
    if not isinstance(files, list) or not files:
        raise ManifestError("artifact files must be a non-empty array")
    verified: dict[str, Path] = {}
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise ManifestError(f"artifact files[{index}] must be an object")
        _require_exact_keys(entry, {"path", "sha256", "bytes"}, set(), f"artifact files[{index}]")
        path_text = _string(entry["path"], f"artifact files[{index}].path", 512)
        if path_text in verified:
            raise ManifestError(f"artifact files contains duplicate path {path_text!r}")
        path = _relative_file(manifest_path.parent, path_text, f"artifact files[{index}].path")
        expected_digest = _digest(entry["sha256"], f"artifact files[{index}].sha256")
        if type(entry["bytes"]) is not int or entry["bytes"] < 1:
            raise ManifestError(f"artifact files[{index}].bytes must be a positive integer")
        if path.stat().st_size != entry["bytes"]:
            raise ManifestError(f"artifact file size mismatch for {path_text}")
        if _sha256_file(path) != expected_digest:
            raise ManifestError(f"artifact file digest mismatch for {path_text}")
        verified[path_text] = path

    graph = document["graph"]
    if not isinstance(graph, dict):
        raise ManifestError("artifact graph must be an object")
    _require_exact_keys(
        graph,
        {"xml", "bin", "input_ids", "attention_mask", "output"},
        set(),
        "artifact graph",
    )
    tokenizer = document["tokenizer"]
    if not isinstance(tokenizer, dict):
        raise ManifestError("artifact tokenizer must be an object")
    _require_exact_keys(
        tokenizer,
        {
            "json",
            "sha256",
            "pad_token",
            "pad_token_id",
            "bos_token",
            "bos_token_id",
            "eos_token",
            "eos_token_id",
            "add_bos_token",
            "add_eos_token",
        },
        set(),
        "artifact tokenizer",
    )
    tokenizer_digest = _digest(tokenizer["sha256"], "artifact tokenizer.sha256")
    if tokenizer_digest != "6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e":
        raise ManifestError("artifact tokenizer digest is not the frozen profile tokenizer")
    expected_tokens = {
        "pad_token": "<pad>",
        "pad_token_id": 0,
        "bos_token": "<bos>",
        "bos_token_id": 2,
        "eos_token": "<eos>",
        "eos_token_id": 1,
        "add_bos_token": True,
        "add_eos_token": True,
    }
    for field, expected in expected_tokens.items():
        if tokenizer[field] != expected or type(tokenizer[field]) is not type(expected):
            raise ManifestError(
                f"artifact tokenizer {field} does not match the frozen tokenizer"
            )

    legal = document["legal"]
    expected_legal = {
        "terms_url": "https://ai.google.dev/gemma/terms",
        "prohibited_use_policy_url": "https://ai.google.dev/gemma/prohibited_use_policy",
        "terms_file": "GEMMA_TERMS.txt",
        "prohibited_use_policy_file": "GEMMA_PROHIBITED_USE_POLICY.txt",
        "use_restrictions_file": "MODEL_USE_RESTRICTIONS.txt",
        "modifications_file": "MODEL_MODIFICATIONS.txt",
        "notice_file": "NOTICE",
    }
    if legal != expected_legal:
        raise ManifestError("artifact legal payload does not match the Gemma distribution contract")

    def referenced_file(field: Any, label: str) -> Path:
        value = _string(field, label, 512)
        try:
            return verified[value]
        except KeyError as error:
            raise ManifestError(f"{label} is not bound by artifact files[]") from error

    graph_xml = referenced_file(graph["xml"], "artifact graph.xml")
    graph_bin = referenced_file(graph["bin"], "artifact graph.bin")
    tokenizer_json = referenced_file(tokenizer["json"], "artifact tokenizer.json")
    legal_paths = {
        name: referenced_file(name, f"artifact legal {name}")
        for name in LEGAL_FILES
    }
    if (
        len(verified) != 3 + len(LEGAL_FILES)
        or len({graph_xml, graph_bin, tokenizer_json, *legal_paths.values()})
        != 3 + len(LEGAL_FILES)
    ):
        raise ManifestError(
            "artifact files must contain exactly distinct graph XML, graph BIN, "
            "tokenizer JSON, and required Gemma legal files"
        )
    if _sha256_file(tokenizer_json) != tokenizer_digest:
        raise ManifestError("artifact tokenizer.json digest does not match tokenizer.sha256")
    for name, path in legal_paths.items():
        if _sha256_file(path) != PINNED_LEGAL_SHA256[name]:
            raise ManifestError(f"artifact legal file digest mismatch for {name}")
    return Artifact(
        root=manifest_path.parent,
        manifest_path=manifest_path,
        manifest_sha256=actual_manifest_sha256,
        graph_xml=graph_xml,
        graph_bin=graph_bin,
        tokenizer_json=tokenizer_json,
        input_ids_name=_string(graph["input_ids"], "artifact graph.input_ids", 128),
        attention_mask_name=_string(graph["attention_mask"], "artifact graph.attention_mask", 128),
        output_name=_string(graph["output"], "artifact graph.output", 128),
        pad_token_id=tokenizer["pad_token_id"],
        bos_token_id=tokenizer["bos_token_id"],
        eos_token_id=tokenizer["eos_token_id"],
        files=tuple(verified.values()),
        conversion_versions={
            dependency: conversion[dependency]
            for dependency in ("openvino", "safetensors", "torch", "transformers")
        },
    )


def load_package_manifest(path: Path) -> PackageManifest:
    path = path.resolve()
    document, _raw = _read_json(path)
    _require_exact_keys(
        document,
        {
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
        },
        set(),
        "package manifest",
    )
    if document["schema_version"] != 1:
        raise ManifestError("package manifest schema_version must be 1")
    package_state = document["package_state"]
    if package_state not in ("physical-probe", "candidate", "release"):
        raise ManifestError(
            "package manifest package_state must be physical-probe, candidate, or release"
        )
    fixed = {
        "profile_id": PROFILE_ID,
        "profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
        "admission_policy_sha256": ADMISSION_POLICY_SHA256,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
    }
    for field, expected in fixed.items():
        if document[field] != expected:
            raise ManifestError(f"package manifest {field} does not match the frozen profile")

    dependencies = document["dependency_versions"]
    if not isinstance(dependencies, dict) or set(dependencies) != {
        "cryptography",
        "numpy",
        "openvino",
        "tokenizers",
    }:
        raise ManifestError(
            "dependency_versions must contain exactly cryptography, numpy, openvino, and tokenizers"
        )
    dependency_versions = {
        name: _string(value, f"dependency_versions.{name}", 128)
        for name, value in dependencies.items()
    }
    artifact = load_artifact(
        path.parent,
        document["artifact_manifest"],
        document["artifact_manifest_sha256"],
    )
    if artifact.conversion_versions["openvino"] != dependency_versions["openvino"]:
        raise ManifestError(
            "artifact conversion and frozen runtime must use the same OpenVINO version"
        )

    scope_documents = document["scopes"]
    if not isinstance(scope_documents, list) or not scope_documents:
        raise ManifestError("package manifest scopes must be a non-empty array")
    scopes: dict[str, Scope] = {}
    key_files: set[Path] = set()
    public_keys: set[str] = set()
    previous_device_rank = -1
    required = {
        "scope_id",
        "backend",
        "transport",
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
    for index, entry in enumerate(scope_documents):
        label = f"package manifest scopes[{index}]"
        if not isinstance(entry, dict):
            raise ManifestError(f"{label} must be an object")
        _require_exact_keys(entry, required, set(), label)
        scope_id = _string(entry["scope_id"], f"{label}.scope_id", 128)
        if SCOPE_ID_RE.fullmatch(scope_id) is None:
            raise ManifestError(f"{label}.scope_id is not a canonical lowercase scope slug")
        if scope_id in scopes:
            raise ManifestError(f"package manifest contains duplicate scope {scope_id!r}")
        device_class = entry["device_class"]
        if device_class not in DEVICE_FOR_CLASS:
            raise ManifestError(f"{label}.device_class must be npu, gpu, or cpu")
        device_rank = ("npu", "gpu", "cpu").index(device_class)
        if device_rank < previous_device_rank:
            raise ManifestError(
                "package manifest scopes must be ordered NPU, then GPU, then accelerated CPU"
            )
        previous_device_rank = device_rank
        openvino_device = entry["openvino_device"]
        if openvino_device != DEVICE_FOR_CLASS[device_class]:
            raise ManifestError(
                f"{label}.openvino_device must be exactly {DEVICE_FOR_CLASS[device_class]} "
                f"for device_class={device_class}; aggregate/fallback device names are forbidden"
            )
        if entry["backend"] != "openvino":
            raise ManifestError(f"{label}.backend must be openvino")
        if entry["transport"] != "supervised-local":
            raise ManifestError(f"{label}.transport must be supervised-local")
        if entry["artifact_sha256"] != artifact.manifest_sha256:
            raise ManifestError(
                f"{label}.artifact_sha256 must equal the verified artifact manifest digest"
            )
        if entry["supported_max_tokens"] != MAX_TOKENS:
            raise ManifestError(f"{label}.supported_max_tokens must be {MAX_TOKENS}")
        if entry["supported_sequence_buckets"] != list(SEQUENCE_BUCKETS):
            raise ManifestError(f"{label} must support all seven exact sequence buckets")
        if entry["supported_max_batch_size"] != MAX_WIRE_BATCH_SIZE:
            raise ManifestError(
                f"{label}.supported_max_batch_size must be {MAX_WIRE_BATCH_SIZE}"
            )
        if entry["accelerated_placement"] is not True:
            raise ManifestError(f"{label}.accelerated_placement must be true")
        required_execution_devices = entry["required_execution_devices"]
        if (
            not isinstance(required_execution_devices, list)
            or len(required_execution_devices) != 1
            or not isinstance(required_execution_devices[0], str)
            or re.fullmatch(
                rf"{re.escape(openvino_device)}(?:\.[0-9]+)?",
                required_execution_devices[0],
            )
            is None
        ):
            raise ManifestError(
                f"{label}.required_execution_devices must contain exactly the "
                f"physical {openvino_device} device selected by this scope"
            )
        evidence_values: dict[str, str | None] = {}
        for field in (
            "placement_evidence_sha256",
            "sequence_capability_evidence_sha256",
            "performance_evidence_sha256",
            "compatibility_report_sha256",
        ):
            value = entry[field]
            evidence_values[field] = (
                None if value is None else _digest(value, f"{label}.{field}")
            )
        physical_values = tuple(
            evidence_values[field]
            for field in (
                "placement_evidence_sha256",
                "sequence_capability_evidence_sha256",
                "performance_evidence_sha256",
            )
        )
        compatibility = evidence_values["compatibility_report_sha256"]
        if package_state == "physical-probe":
            if any(value is not None for value in (*physical_values, compatibility)):
                raise ManifestError(
                    f"{label} physical-probe evidence and report bindings must be null"
                )
        elif package_state == "candidate":
            if any(value is None for value in physical_values) or compatibility is not None:
                raise ManifestError(
                    f"{label} candidate requires three evidence digests and a null report"
                )
        elif any(value is None for value in (*physical_values, compatibility)):
            raise ManifestError(
                f"{label} release requires three evidence digests and a report digest"
            )
        private_key_file = _relative_file(
            path.parent,
            entry["attestation_private_key_file"],
            f"{label}.attestation_private_key_file",
        )
        public_key = _digest(entry["attestation_public_key"], f"{label}.attestation_public_key")
        if private_key_file in key_files or public_key in public_keys:
            raise ManifestError("every execution scope must use a globally unique attestation key")
        key_files.add(private_key_file)
        public_keys.add(public_key)
        scope = Scope(
            package_state=package_state,
            scope_id=scope_id,
            backend="openvino",
            transport="supervised-local",
            runtime=_string(entry["runtime"], f"{label}.runtime"),
            compiler=_string(entry["compiler"], f"{label}.compiler"),
            package_target=_string(entry["package_target"], f"{label}.package_target"),
            artifact_source=_string(entry["artifact_source"], f"{label}.artifact_source"),
            artifact_sha256=artifact.manifest_sha256,
            internal_precision=_string(entry["internal_precision"], f"{label}.internal_precision"),
            device_class=device_class,
            device=_string(entry["device"], f"{label}.device"),
            openvino_device=openvino_device,
            openvino_compile_config=_primitive_config(
                entry["openvino_compile_config"], f"{label}.openvino_compile_config"
            ),
            required_openvino_properties=_required_openvino_properties(
                entry["required_openvino_properties"],
                device_class,
                f"{label}.required_openvino_properties",
            ),
            required_execution_devices=tuple(required_execution_devices),
            required_host=_host_binding(entry["required_host"], f"{label}.required_host"),
            placement_evidence_sha256=evidence_values[
                "placement_evidence_sha256"
            ],
            sequence_capability_evidence_sha256=evidence_values[
                "sequence_capability_evidence_sha256"
            ],
            performance_evidence_sha256=evidence_values[
                "performance_evidence_sha256"
            ],
            compatibility_report_sha256=compatibility,
            attestation_public_key=public_key,
            attestation_private_key_file=private_key_file,
            accelerated_placement=True,
        )
        scopes[scope_id] = scope
    if {scope.device_class for scope in scopes.values()} != {"npu", "gpu", "cpu"}:
        raise ManifestError(
            "an Intel target package must contain NPU, GPU, and accelerated CPU scopes"
        )
    return PackageManifest(
        path=path,
        artifact=artifact,
        scopes=scopes,
        dependency_versions=dependency_versions,
        runtime_manifest_sha256=_digest(
            document["runtime_manifest_sha256"], "runtime_manifest_sha256"
        ),
        package_state=package_state,
    )
