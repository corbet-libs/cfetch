#!/usr/bin/env python3
"""Gate vector compatibility across every required ordered backend pairing."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import tempfile
import urllib.parse
import urllib.request
import zipfile
from functools import cmp_to_key
from pathlib import Path
from typing import TYPE_CHECKING

from admission_evidence import (
    DOCUMENT_PREFIX,
    MAX_TOKENS,
    QUERY_PREFIX,
    SEQUENCE_BUCKETS,
    SEQUENCE_SEMANTIC_FIXTURE_ID,
    SEQUENCE_SEMANTIC_FIXTURE_SHA256,
    SUPPORTED_MAX_BATCH_SIZE,
    TRANSPORTS,
    ordered_input_json_sha256,
    parse_evidence_json,
    sequence_semantic_fixture_sha256,
    sequence_semantic_probe_inputs,
    validate_evidence_reports,
    validate_wire_batch_evidence,
    wire_batch_inputs,
)
from scifact_contract import (
    DATASET,
    DATASET_REVISION,
    EXPECTED_DOCUMENTS,
    load_scifact_contract,
)
from profile_identity import (
    ADMISSION_IMPLEMENTATION_BUNDLE_SHA256,
    ADMISSION_POLICY_SHA256,
)

if TYPE_CHECKING:
    import numpy as np

PROFILE_ID = "cfetch-embedding-v1"
PROFILE_MANIFEST_SHA256 = (
    "59210a333494f788eb8e607fe38cabb6af1a7aa7cdf604ddf52e3fa6004b5afb"
)
EVIDENCE_REPLAY_POLICY = (
    "durable-content-addressed-cache-and-measurement-bundle-strict-schema-"
    "ci-full-gate-replay"
)
MODEL = "google/embeddinggemma-300m"
MODEL_REVISION = "57c266a740f537b4dc058e1b0cda161fd15afa75"
VECTOR_ENCODING = "signed-int8x768"
REQUIRED_CLASSES = {"npu", "gpu", "cpu"}
SCOPE_ID_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
SEQUENCE_SEMANTIC_GATE = (
    "every-profile-sequence-bucket-global-ordered-query-document-scope-plus-"
    "adversarial-relevant-minimum-irrelevant-maximum-exact-int8-strict-ranking"
)
SEQUENCE_PROBE_PRIMARY_NAMES = (
    "sequence_probe_queries",
    "sequence_probe_relevant_documents",
    "sequence_probe_irrelevant_documents",
)
SEQUENCE_PROBE_REPEAT_NAMES = tuple(
    f"{name}_repeat" for name in SEQUENCE_PROBE_PRIMARY_NAMES
)
MAX_EXACT_I8_DOT_OR_NORM_SQ = 768 * 127 * 127
MAX_ADMISSION_CACHE_BYTES = 64 * 1024 * 1024
MAX_ADMISSION_CACHE_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_ADMISSION_COHORT_BYTES = 512 * 1024 * 1024
MAX_ADMITTED_SCOPES = 64
MAX_ADMISSION_REGISTRY_BYTES = 1024 * 1024
MAX_ADMISSION_REPORT_BYTES = 32 * 1024 * 1024
MAX_ADMISSION_REPORT_LINEAGE_BYTES = 256 * 1024 * 1024
MAX_MEASUREMENT_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_MEASUREMENT_BUNDLE_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_MEASUREMENT_BUNDLE_MEMBERS = 128
MAX_EMBEDDED_EVIDENCE_BYTES = 8 * 1024 * 1024
MAX_CACHE_METADATA_BYTES = 256 * 1024
IMPLEMENTATION_BUNDLE_DOMAIN = b"cfetch-admission-implementation-bundle-v1\0"
IMPLEMENTATION_BUNDLE_FILES = tuple(
    sorted(
        (
            "experiments/embedding-profile/cross_backend_eval.py",
            "experiments/embedding-profile/admission_evidence.py",
            "experiments/embedding-profile/admission_transaction.py",
            "experiments/embedding-profile/export_adapter_cache.py",
            "experiments/embedding-profile/final_package_conformance.py",
            "experiments/embedding-profile/measurement_bundle.py",
            "experiments/embedding-profile/physical_evidence.py",
            "experiments/embedding-profile/physical_checkpoint.py",
            "experiments/embedding-profile/scifact_contract.py",
            "experiments/embedding-profile/requirements-lock.txt",
            "experiments/embedding-profile/requirements-test.txt",
            "packages/openvino/package_inventory.py",
            "scripts/apply_admission_activation.py",
        )
    )
)
ADMISSION_CACHE_ARRAY_NAMES = {
    "metadata",
    "queries",
    "documents",
    "queries_repeat",
    "documents_repeat",
    *SEQUENCE_PROBE_PRIMARY_NAMES,
    *SEQUENCE_PROBE_REPEAT_NAMES,
    "sequence_capability_evidence_bytes",
    "placement_evidence_bytes",
    "performance_evidence_bytes",
    "wire_batch_outputs",
}
ADMISSION_CACHE_ZIP_MEMBERS = {
    f"{name}.npy" for name in ADMISSION_CACHE_ARRAY_NAMES
}

# These are fixed quality requirements for every ordered query/document backend
# pairing. No backend supplies or changes the floor for another backend.
ABSOLUTE_MINIMUM = {
    "ndcg_at_10": 0.767907905520953,
    "recall_at_100": 0.970,
    "mrr_at_10": 0.7305529100529101,
}
EXACT_INT8_RANKING = (
    "exact-signed-int8-cosine-query-norm-cancels-sign-branches-"
    "squared-cross-multiplication"
)
EXACT_RANKING_TIE_BREAK = (
    "pinned-corpus-insertion-index-ascending-as-evaluation-block-id-order"
)
ADVERSARIAL_MIXED_DOCUMENT_SELECTION = (
    "per-query-relevant-minimum-irrelevant-maximum-exact-cosine-"
    "across-document-scopes"
)
RANKING_SEMANTICS = {
    "ranking": EXACT_INT8_RANKING,
    "tie_break": EXACT_RANKING_TIE_BREAK,
    "adversarial_mixed_document_selection": ADVERSARIAL_MIXED_DOCUMENT_SELECTION,
}


class ExactI8Scores:
    """Exact integer cosine inputs for one query/document matrix.

    A document norm may be shared by every query (the normal all-pairs case)
    or selected independently for each query/document cell (the adversarial
    mixed-producer case).
    """

    def __init__(self, dots: np.ndarray, document_norms_sq: np.ndarray) -> None:
        if dots.ndim != 2:
            raise ValueError("INT8 dot products must be a rank-two array")
        if document_norms_sq.ndim == 1:
            if document_norms_sq.shape[0] != dots.shape[1]:
                raise ValueError("document norms do not match the score matrix")
        elif document_norms_sq.shape != dots.shape:
            raise ValueError("per-query document norms do not match the score matrix")
        self.dots = dots
        self.document_norms_sq = document_norms_sq

    def norms_for_row(self, row: int) -> np.ndarray:
        if self.document_norms_sq.ndim == 1:
            return self.document_norms_sq
        return self.document_norms_sq[row]


def scope_id_value(value: str) -> str:
    if len(value) > 128 or SCOPE_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "scope id must be at most 128 lowercase slug characters "
            "(letters, digits, and single '.', '_', or '-' separators)"
        )
    return value


def parse_backend(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("backend must be LABEL=PATH")
    return scope_id_value(label), Path(raw_path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_bundle_sha256(repository: Path | None = None) -> str:
    """Hash the exact admission implementation bytes with unambiguous framing."""
    root = repository or Path(__file__).resolve().parents[2]
    digest = hashlib.sha256(IMPLEMENTATION_BUNDLE_DOMAIN)
    for relative_path in IMPLEMENTATION_BUNDLE_FILES:
        path_bytes = relative_path.encode("utf-8")
        data = (root / relative_path).read_bytes()
        digest.update(len(path_bytes).to_bytes(4, "big"))
        digest.update(path_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def verify_requirements_lock(repository: Path | None = None) -> dict[str, object]:
    """Require an exact, hash-complete transitive environment for admission."""
    root = repository or Path(__file__).resolve().parents[2]
    direct_path = root / "experiments/embedding-profile/requirements-test.txt"
    lock_path = root / "experiments/embedding-profile/requirements-lock.txt"
    direct: dict[str, str] = {}
    for line in direct_path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s\\]+)", value)
        if match is None:
            raise ValueError(
                f"{direct_path}: direct admission requirements must be exact pins"
            )
        direct[match.group(1).lower().replace("_", "-")] = match.group(2)

    raw_lock = lock_path.read_text(encoding="utf-8")
    starts = list(
        re.finditer(r"(?m)^([A-Za-z0-9_.-]+)==([^\s\\]+)(?:\s*\\)?$", raw_lock)
    )
    if not starts:
        raise ValueError(f"{lock_path}: transitive requirements lock is empty")
    locked: dict[str, str] = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(raw_lock)
        entry = raw_lock[match.start() : end]
        name = match.group(1).lower().replace("_", "-")
        if name in locked:
            raise ValueError(f"{lock_path}: duplicate locked package {name}")
        if re.search(r"--hash=sha256:[0-9a-f]{64}(?:\s|\\|$)", entry) is None:
            raise ValueError(f"{lock_path}: locked package {name} has no SHA-256")
        locked[name] = match.group(2)
    for name, version in direct.items():
        if locked.get(name) != version:
            raise ValueError(
                f"{lock_path}: direct pin {name}=={version} is absent or changed"
            )
    return {"direct_packages": len(direct), "locked_packages": len(locked)}


def verify_implementation_bundle() -> dict[str, object]:
    lock = verify_requirements_lock()
    actual = implementation_bundle_sha256()
    if actual != ADMISSION_IMPLEMENTATION_BUNDLE_SHA256:
        raise ValueError(
            "admission implementation bundle digest is stale: "
            f"computed {actual}, expected {ADMISSION_IMPLEMENTATION_BUNDLE_SHA256}"
        )
    repository = Path(__file__).resolve().parents[2]
    registry_path = repository / "release/inference-backends.json"
    registry = parse_evidence_json(
        read_bounded_local_file(
            registry_path, MAX_ADMISSION_REGISTRY_BYTES, "release registry"
        ),
        str(registry_path),
    )
    if registry.get("admission", {}).get("implementation_bundle_sha256") != actual:
        raise ValueError(
            "release registry admission implementation bundle digest is stale"
        )
    return {
        "implementation_bundle_sha256": actual,
        **lock,
        "status": "passed",
    }


def read_bounded_local_file(path: Path, maximum_bytes: int, context: str) -> bytes:
    """Read one already-resolved local file through a single bounded handle."""
    with path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if size < 1 or size > maximum_bytes:
            raise ValueError(f"{context} must be 1..{maximum_bytes} bytes")
        data = handle.read(maximum_bytes + 1)
    if len(data) != size:
        raise ValueError(f"{context} changed while it was being read")
    return data


def write_new_report(path: Path, serialized_report: str) -> None:
    """Create a report exclusively so content-addressed history is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized_report)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite existing report: {path}") from error


def validate_admission_report_reference(report_reference: str, digest: str) -> None:
    expected_reference = f"release/admission/{digest}.json"
    if (
        re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or report_reference != expected_reference
    ):
        raise ValueError(
            "compatibility report references must be content-addressed as "
            "release/admission/<sha256>.json"
        )


def admission_report_path(report_reference: str, digest: str) -> Path:
    """Resolve one digest-named committed report without symlink escape."""
    validate_admission_report_reference(report_reference, digest)
    repository = Path(__file__).resolve().parents[2].resolve()
    declared_release_root = repository / "release"
    release_root = declared_release_root.resolve()
    declared_admission_root = declared_release_root / "admission"
    admission_root = declared_admission_root.resolve()
    if (
        declared_release_root.is_symlink()
        or release_root.parent != repository
        or declared_admission_root.is_symlink()
        or admission_root.parent != release_root
    ):
        raise ValueError("release/admission escaped the repository release directory")
    report_path = (repository / report_reference).resolve()
    if report_path.parent != admission_root or not report_path.is_file():
        raise ValueError(
            f"content-addressed compatibility report is missing: {report_reference}"
        )
    return report_path


def read_content_addressed_report(
    report_reference: str, digest: str
) -> tuple[Path, dict[str, object], int]:
    path = admission_report_path(report_reference, digest)
    data = read_bounded_local_file(
        path, MAX_ADMISSION_REPORT_BYTES, f"compatibility report {report_reference}"
    )
    actual_digest = hashlib.sha256(data).hexdigest()
    if actual_digest != digest:
        raise ValueError(
            f"compatibility report {report_reference} bytes do not match its sha256"
        )
    return path, parse_evidence_json(data, str(path)), len(data)


def validate_admission_cache_container(path: Path) -> None:
    """Bound and allowlist the NPZ before NumPy materializes any member."""
    size = path.stat().st_size
    if size < 1 or size > MAX_ADMISSION_CACHE_BYTES:
        raise ValueError(
            f"{path}: admission cache must be 1..{MAX_ADMISSION_CACHE_BYTES} bytes"
        )
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = {member.filename for member in members}
            if len(members) != len(names) or names != ADMISSION_CACHE_ZIP_MEMBERS:
                raise ValueError(
                    f"{path}: admission cache must contain exactly the canonical NPZ members"
                )
            if any(
                member.flag_bits & 0x1
                or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or member.file_size < 1
                for member in members
            ):
                raise ValueError(
                    f"{path}: admission cache contains an unsupported ZIP member"
                )
            expanded = sum(member.file_size for member in members)
            if expanded > MAX_ADMISSION_CACHE_EXPANDED_BYTES:
                raise ValueError(
                    f"{path}: admission cache expands beyond "
                    f"{MAX_ADMISSION_CACHE_EXPANDED_BYTES} bytes"
                )
            for member in members:
                validate_npy_member_header(path, archive, member)
    except zipfile.BadZipFile as error:
        raise ValueError(f"{path}: admission cache is not a valid NPZ archive") from error


def validate_npy_member_header(
    path: Path, archive: zipfile.ZipFile, member: zipfile.ZipInfo
) -> None:
    """Validate dtype, dimensions, and payload size without allocating an array."""
    import numpy as np

    name = member.filename.removesuffix(".npy")
    try:
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                header_reader = np.lib.format.read_array_header_1_0
            elif version == (2, 0):
                header_reader = np.lib.format.read_array_header_2_0
            else:
                raise ValueError(f"unsupported canonical NPY version {version}")
            shape, fortran_order, dtype = header_reader(
                stream, max_header_size=16 * 1024
            )
            header_bytes = stream.tell()
    except (EOFError, OSError, ValueError) as error:
        raise ValueError(
            f"{path}: admission cache member {member.filename} has an invalid NPY header"
        ) from error
    if fortran_order or dtype.hasobject:
        raise ValueError(
            f"{path}: admission cache member {member.filename} has an unsafe array layout"
        )

    element_count = math.prod(shape)
    payload_bytes = element_count * dtype.itemsize
    if header_bytes + payload_bytes != member.file_size:
        raise ValueError(
            f"{path}: admission cache member {member.filename} payload size mismatches "
            "its NPY header"
        )
    if name == "metadata":
        valid = shape == () and dtype.kind == "U" and 0 < payload_bytes <= MAX_CACHE_METADATA_BYTES
    elif name in {
        "sequence_capability_evidence_bytes",
        "placement_evidence_bytes",
        "performance_evidence_bytes",
    }:
        valid = (
            len(shape) == 1
            and dtype == np.dtype(np.uint8)
            and 0 < payload_bytes <= MAX_EMBEDDED_EVIDENCE_BYTES
        )
    elif name in {*SEQUENCE_PROBE_PRIMARY_NAMES, *SEQUENCE_PROBE_REPEAT_NAMES}:
        valid = shape == (len(SEQUENCE_BUCKETS), 768) and dtype == np.dtype(np.int8)
    elif name == "wire_batch_outputs":
        valid = shape == (64, 64, 768) and dtype == np.dtype(np.int8)
    else:
        valid = (
            len(shape) == 2
            and 0 < shape[0] <= EXPECTED_DOCUMENTS
            and shape[1] == 768
            and dtype == np.dtype(np.int8)
        )
    if not valid:
        raise ValueError(
            f"{path}: admission cache member {member.filename} has an invalid bounded shape or dtype"
        )


def validate_canonical_records(path: Path, name: str, array: np.ndarray) -> None:
    """Reject byte patterns the profile's max-absolute codec cannot emit."""
    import numpy as np

    if np.any(array == -128):
        raise ValueError(f"{path}: {name} contains forbidden signed INT8 value -128")
    has_canonical_extremum = np.any((array == -127) | (array == 127), axis=1)
    if not has_canonical_extremum.all():
        raise ValueError(
            f"{path}: every {name} row must contain at least one -127 or +127 component"
        )


def _sequence_probe_arrays_from_cache(
    cached: object, path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the two-run semantic fixture for every graph bucket."""
    import numpy as np

    names = (*SEQUENCE_PROBE_PRIMARY_NAMES, *SEQUENCE_PROBE_REPEAT_NAMES)
    missing = set(names).difference(cached.files)
    if missing:
        raise ValueError(f"{path}: missing sequence semantic probe arrays {sorted(missing)}")
    arrays = {name: np.asarray(cached[name]) for name in names}
    expected_shape = (len(SEQUENCE_BUCKETS), 768)
    for name, array in arrays.items():
        if array.dtype != np.dtype(np.int8):
            raise ValueError(
                f"{path}: {name} must be stored as signed int8, found {array.dtype}"
            )
        if array.shape != expected_shape:
            raise ValueError(
                f"{path}: {name} must have shape {expected_shape}, found {array.shape}"
            )
        validate_canonical_records(path, name, array)
        if not np.any(array, axis=1).all():
            raise ValueError(f"{path}: {name} contains an all-zero vector")
    for primary, repeat in zip(
        SEQUENCE_PROBE_PRIMARY_NAMES, SEQUENCE_PROBE_REPEAT_NAMES, strict=True
    ):
        if not np.array_equal(arrays[primary], arrays[repeat]):
            raise ValueError(
                f"{path}: {primary} is not byte-repeatable on the same "
                "runtime/artifact/device"
            )
    return tuple(arrays[name] for name in SEQUENCE_PROBE_PRIMARY_NAMES)


def validate_sequence_probe_evidence(
    path: Path,
    evidence_bytes: np.ndarray,
    probes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Bind cached probe vectors to the pinned inputs and signed evidence bytes."""
    import numpy as np

    report = parse_evidence_json(
        evidence_bytes.tobytes(), f"{path}: sequence evidence"
    )
    if report.get("supported_sequence_buckets") != SEQUENCE_BUCKETS:
        raise ValueError(
            f"{path}: sequence evidence must cover every profile bucket"
        )
    records = report.get("bucket_results")
    if (
        not isinstance(records, list)
        or any(not isinstance(row, dict) for row in records)
        or [row.get("bucket") for row in records] != SEQUENCE_BUCKETS
    ):
        raise ValueError(
            f"{path}: sequence evidence bucket_results must follow profile bucket order"
        )
    probe_fields = {
        "fixture_id",
        "fixture_sha256",
        "query_input_utf8_sha256",
        "relevant_document_input_utf8_sha256",
        "irrelevant_document_input_utf8_sha256",
        "query_token_count",
        "relevant_document_token_count",
        "irrelevant_document_token_count",
        "query_canonical_output_bytes_sha256",
        "relevant_document_canonical_output_bytes_sha256",
        "irrelevant_document_canonical_output_bytes_sha256",
        "canonical_repeatability",
        "self_relevant_before_irrelevant",
    }
    labels = ("query", "relevant_document", "irrelevant_document")
    for index, (bucket, row) in enumerate(zip(SEQUENCE_BUCKETS, records, strict=True)):
        evidence = row.get("semantic_probe")
        if not isinstance(evidence, dict) or set(evidence) != probe_fields:
            raise ValueError(
                f"{path}: sequence evidence bucket {bucket} has an invalid semantic_probe"
            )
        if (
            evidence["fixture_id"] != SEQUENCE_SEMANTIC_FIXTURE_ID
            or evidence["fixture_sha256"] != SEQUENCE_SEMANTIC_FIXTURE_SHA256
        ):
            raise ValueError(
                f"{path}: sequence evidence bucket {bucket} does not use the pinned fixture"
            )
        for label, text, array in zip(
            labels, sequence_semantic_probe_inputs(bucket), probes, strict=True
        ):
            input_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if evidence[f"{label}_input_utf8_sha256"] != input_digest:
                raise ValueError(
                    f"{path}: sequence evidence bucket {bucket} {label} input digest "
                    "does not match the pinned fixture"
                )
            token_count = evidence[f"{label}_token_count"]
            selected_bucket = next(
                (item for item in SEQUENCE_BUCKETS if item >= token_count), None
            ) if type(token_count) is int and token_count > 0 else None
            if selected_bucket != bucket:
                raise ValueError(
                    f"{path}: sequence evidence bucket {bucket} {label} token count "
                    "does not select that graph"
                )
            output_digest = hashlib.sha256(
                np.ascontiguousarray(array[index]).tobytes()
            ).hexdigest()
            if evidence[f"{label}_canonical_output_bytes_sha256"] != output_digest:
                raise ValueError(
                    f"{path}: sequence evidence bucket {bucket} {label} output digest "
                    "does not match the cached vector"
                )
        if evidence["canonical_repeatability"] is not True:
            raise ValueError(
                f"{path}: sequence evidence bucket {bucket} does not attest repeatability"
            )
        query = probes[0][index].astype(np.int64)
        relevant = probes[1][index].astype(np.int64)
        irrelevant = probes[2][index].astype(np.int64)
        exact_self_passed = (
            exact_i8_cosine_desc(
                int(query @ relevant),
                int(relevant @ relevant),
                int(query @ irrelevant),
                int(irrelevant @ irrelevant),
            )
            < 0
        )
        if evidence["self_relevant_before_irrelevant"] is not True or not exact_self_passed:
            raise ValueError(
                f"{path}: sequence evidence bucket {bucket} does not pass exact self ranking"
            )


def load_sequence_probe_cache(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the query/relevant/irrelevant bucket probes from one validated cache."""
    import numpy as np

    validate_admission_cache_container(path)
    with np.load(path, allow_pickle=False) as cached:
        probes = _sequence_probe_arrays_from_cache(cached, path)
        if "sequence_capability_evidence_bytes" not in cached.files:
            raise ValueError(f"{path}: missing sequence capability evidence bytes")
        evidence = np.asarray(cached["sequence_capability_evidence_bytes"])
    validate_sequence_probe_evidence(path, evidence, probes)
    return probes


def load_cache(path: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    import numpy as np

    validate_admission_cache_container(path)
    with np.load(path, allow_pickle=False) as cached:
        required = {
            "metadata",
            "queries",
            "documents",
            "queries_repeat",
            "documents_repeat",
            "sequence_capability_evidence_bytes",
            "placement_evidence_bytes",
            "performance_evidence_bytes",
            "wire_batch_outputs",
            *SEQUENCE_PROBE_PRIMARY_NAMES,
            *SEQUENCE_PROBE_REPEAT_NAMES,
        }
        missing = required.difference(cached.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        metadata = parse_evidence_json(
            str(cached["metadata"].item()).encode("utf-8"),
            f"{path}: metadata",
        )
        arrays = {
            "queries": np.asarray(cached["queries"]),
            "documents": np.asarray(cached["documents"]),
            "queries_repeat": np.asarray(cached["queries_repeat"]),
            "documents_repeat": np.asarray(cached["documents_repeat"]),
        }
        evidence = {
            "sequence_capability": np.asarray(
                cached["sequence_capability_evidence_bytes"]
            ),
            "placement": np.asarray(cached["placement_evidence_bytes"]),
            "performance": np.asarray(cached["performance_evidence_bytes"]),
        }
        sequence_probes = _sequence_probe_arrays_from_cache(cached, path)

    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata must be a JSON object")
    for name, array in arrays.items():
        if array.dtype != np.dtype(np.int8):
            raise ValueError(
                f"{path}: {name} must be stored as signed int8, found {array.dtype}"
            )
    queries = arrays["queries"]
    documents = arrays["documents"]
    queries_repeat = arrays["queries_repeat"]
    documents_repeat = arrays["documents_repeat"]

    expected = {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
        "admission_policy_sha256": ADMISSION_POLICY_SHA256,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "vector_encoding": VECTOR_ENCODING,
        "supported_max_tokens": MAX_TOKENS,
        "supported_sequence_buckets": SEQUENCE_BUCKETS,
        "supported_max_batch_size": SUPPORTED_MAX_BATCH_SIZE,
        "sequence_semantic_fixture_id": SEQUENCE_SEMANTIC_FIXTURE_ID,
        "sequence_semantic_fixture_sha256": SEQUENCE_SEMANTIC_FIXTURE_SHA256,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{path}: metadata {key}={metadata.get(key)!r}, expected {value!r}")
    for key in (
        "scope_id",
        "transport",
        "backend",
        "runtime",
        "compiler",
        "package_target",
        "artifact_source",
        "artifact_sha256",
        "attestation_public_key",
        "internal_precision",
        "sequence_capability_evidence",
        "sequence_capability_evidence_sha256",
        "device",
        "placement_evidence",
        "placement_evidence_sha256",
        "performance_evidence",
        "performance_evidence_sha256",
    ):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"{path}: metadata {key} must be a non-empty string")
    if metadata["transport"] not in TRANSPORTS:
        raise ValueError(
            f"{path}: metadata transport must be supervised-local or remote-attested"
        )
    if re.fullmatch(r"[0-9a-f]{64}", metadata["artifact_sha256"]) is None:
        raise ValueError(f"{path}: metadata artifact_sha256 must be 64 lowercase hexadecimal characters")
    if re.fullmatch(r"[0-9a-f]{64}", metadata["attestation_public_key"]) is None:
        raise ValueError(
            f"{path}: metadata attestation_public_key must be a 64-character "
            "lowercase hexadecimal Ed25519 public key"
        )
    for field in (
        "placement_evidence_sha256",
        "sequence_capability_evidence_sha256",
        "performance_evidence_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", metadata[field]) is None:
            raise ValueError(
                f"{path}: metadata {field} must be 64 lowercase hexadecimal characters"
            )
    for name, data in evidence.items():
        if data.dtype != np.dtype(np.uint8) or data.ndim != 1 or not data.size:
            raise ValueError(f"{path}: {name} evidence must be non-empty raw bytes")
        expected_digest = metadata[f"{name}_evidence_sha256"]
        actual_digest = hashlib.sha256(data.tobytes()).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                f"{path}: {name} evidence has sha256 {actual_digest}, "
                f"metadata requires {expected_digest}"
            )
    evidence_reports = {
        name: parse_evidence_json(data.tobytes(), f"{path}: {name} evidence")
        for name, data in evidence.items()
    }
    validate_evidence_reports(
        metadata,
        evidence_reports["sequence_capability"],
        evidence_reports["placement"],
        evidence_reports["performance"],
    )
    validate_sequence_probe_evidence(
        path, evidence["sequence_capability"], sequence_probes
    )
    if metadata.get("device_class") not in REQUIRED_CLASSES:
        raise ValueError(f"{path}: device_class must be npu, gpu, or cpu")
    if metadata.get("accelerated_placement") is not True:
        raise ValueError(f"{path}: accelerated_placement must be true")
    if queries.ndim != 2 or documents.ndim != 2:
        raise ValueError(f"{path}: vectors must be rank-two arrays")
    if queries.shape[1] != 768 or documents.shape[1] != 768:
        raise ValueError(f"{path}: vectors must have 768 components")
    if queries_repeat.shape != queries.shape or documents_repeat.shape != documents.shape:
        raise ValueError(f"{path}: repeat arrays have different shapes")
    for name, array in arrays.items():
        validate_canonical_records(path, name, array)
    if not np.array_equal(queries, queries_repeat) or not np.array_equal(
        documents, documents_repeat
    ):
        raise ValueError(f"{path}: backend is not byte-repeatable on the same runtime/artifact/device")
    if not np.any(queries, axis=1).all() or not np.any(documents, axis=1).all():
        raise ValueError(f"{path}: cache contains an all-zero vector")
    return metadata, queries, documents


def load_embedded_evidence_reports(path: Path) -> dict[str, dict[str, object]]:
    """Read the already bounded evidence JSON used for measurement-bundle binding."""
    import numpy as np

    validate_admission_cache_container(path)
    with np.load(path, allow_pickle=False) as cached:
        return {
            "sequence": parse_evidence_json(
                np.asarray(cached["sequence_capability_evidence_bytes"]).tobytes(),
                f"{path}: sequence evidence",
            ),
            "placement": parse_evidence_json(
                np.asarray(cached["placement_evidence_bytes"]).tobytes(),
                f"{path}: placement evidence",
            ),
            "performance": parse_evidence_json(
                np.asarray(cached["performance_evidence_bytes"]).tobytes(),
                f"{path}: performance evidence",
            ),
        }


def validate_wire_batch_output_cache(
    path: Path, expected_input_digest: str
) -> None:
    """Replay evidence digests from retained grouping outputs and pinned inputs."""
    import numpy as np

    validate_admission_cache_container(path)
    with np.load(path, allow_pickle=False) as cached:
        outputs = np.asarray(cached["wire_batch_outputs"])
        sequence_report = parse_evidence_json(
            np.asarray(cached["sequence_capability_evidence_bytes"]).tobytes(),
            f"{path}: sequence evidence",
        )
    validate_wire_batch_evidence(sequence_report)
    if outputs.dtype != np.dtype(np.int8) or outputs.shape != (64, 64, 768):
        raise ValueError(
            f"{path}: wire_batch_outputs must have exact signed-int8 shape (64, 64, 768)"
        )
    validate_canonical_records(path, "wire_batch_outputs", outputs.reshape(-1, 768))
    if any(not np.array_equal(outputs[0], item) for item in outputs[1:]):
        raise ValueError(
            f"{path}: retained wire-batch grouping outputs are not byte-identical"
        )
    for index, row in enumerate(sequence_report["wire_batch_results"]):
        batch_size = index + 1
        if row["ordered_input_json_sha256"] != expected_input_digest:
            raise ValueError(
                f"{path}: sequence evidence batch {batch_size} ordered-input digest "
                "does not match the pinned SciFact probe"
            )
        output_digest = hashlib.sha256(
            np.ascontiguousarray(outputs[index]).tobytes()
        ).hexdigest()
        if row["canonical_output_bytes_sha256"] != output_digest:
            raise ValueError(
                f"{path}: sequence evidence batch {batch_size} output digest does "
                "not match retained wire_batch_outputs"
            )


def scores(queries: np.ndarray, documents: np.ndarray) -> ExactI8Scores:
    """Compute exact signed-INT8 dots and document squared norms.

    The query norm is constant across a query row, so it cancels from the
    production cosine ordering just as it does in Rust.
    """
    import numpy as np

    query_int = queries.astype(np.int64)
    document_int = documents.astype(np.int64)
    dots = query_int @ document_int.T
    document_norms_sq = np.sum(
        document_int * document_int, axis=1, dtype=np.int64
    )
    if np.any(document_norms_sq <= 0):
        raise ValueError("cannot score an all-zero INT8 document")
    return ExactI8Scores(dots, document_norms_sq)


def exact_i8_cosine_desc(
    dot_a: int,
    norm_a_sq: int,
    dot_b: int,
    norm_b_sq: int,
) -> int:
    """Compare two cosine scores in descending production order.

    The branches intentionally mirror Rust's `i8_cosine_desc`. Python's
    arbitrary-size integers provide the same overflow-free cross
    multiplication as Rust's u128 for canonical INT8x768 records.
    """
    if norm_a_sq <= 0 or norm_b_sq <= 0:
        raise ValueError("document squared norms must be positive")
    sign_a = (dot_a > 0) - (dot_a < 0)
    sign_b = (dot_b > 0) - (dot_b < 0)
    if sign_a != sign_b:
        return -1 if sign_a > sign_b else 1
    if sign_a == 0:
        return 0

    left = abs(dot_a) ** 2 * norm_b_sq
    right = abs(dot_b) ** 2 * norm_a_sq
    if sign_a > 0:
        return (right > left) - (right < left)
    return (left > right) - (left < right)


def ranked_document_indices(
    exact_scores: ExactI8Scores, row: int, limit: int
) -> list[int]:
    """Rank a query exactly, tying by SciFact corpus insertion index.

    The evaluation inserts documents in the pinned corpus order, so this
    index is the gate's deterministic equivalent of production's block-id
    tie-break.
    """
    dots = exact_scores.dots[row].tolist()
    norms = exact_scores.norms_for_row(row).tolist()

    def compare(left: int, right: int) -> int:
        ordered = exact_i8_cosine_desc(
            dots[left], norms[left], dots[right], norms[right]
        )
        if ordered:
            return ordered
        return (left > right) - (left < right)

    return heapq.nsmallest(
        min(limit, len(dots)),
        range(len(dots)),
        key=cmp_to_key(compare),
    )


def metrics(
    similarities: ExactI8Scores,
    query_ids: list[str],
    document_ids: list[str],
    qrels: dict[str, set[str]],
) -> dict[str, float]:
    import numpy as np

    document_index = {document_id: index for index, document_id in enumerate(document_ids)}
    ndcg10: list[float] = []
    recall100: list[float] = []
    mrr10: list[float] = []
    for row, query_id in enumerate(query_ids):
        relevant = {document_index[item] for item in qrels[query_id] if item in document_index}
        order = ranked_document_indices(similarities, row, 100)
        gains = np.asarray([1.0 if index in relevant else 0.0 for index in order[:10]])
        discounts = 1.0 / np.log2(np.arange(2, 2 + len(gains)))
        dcg = float(np.sum(gains * discounts))
        ideal = float(np.sum(discounts[: min(len(relevant), 10)]))
        ndcg10.append(dcg / ideal if ideal else 0.0)
        recall100.append(len(relevant.intersection(order)) / len(relevant) if relevant else 0.0)
        first = next((rank for rank, index in enumerate(order[:10], 1) if index in relevant), None)
        mrr10.append(1.0 / first if first is not None else 0.0)
    return {
        "ndcg_at_10": float(np.mean(ndcg10)),
        "recall_at_100": float(np.mean(recall100)),
        "mrr_at_10": float(np.mean(mrr10)),
    }


def adversarial_mixed_document_scores(
    query_vectors: np.ndarray,
    document_vectors_by_backend: list[np.ndarray],
    query_ids: list[str],
    document_ids: list[str],
    qrels: dict[str, set[str]],
) -> ExactI8Scores:
    """A lower bound for every possible per-document producer mixture.

    Relevant documents receive their minimum score across document producers;
    irrelevant documents receive their maximum. This query-dependent choice is
    stricter than any real shared store, whose producer is fixed per document.
    Producer selection uses the same exact signed-INT8 comparison as ranking.
    """
    import numpy as np

    if not document_vectors_by_backend:
        raise ValueError("adversarial mixed-document scoring needs a backend")
    backend_scores = [
        scores(query_vectors, documents)
        for documents in document_vectors_by_backend
    ]
    adversarial_dots = backend_scores[0].dots.copy()
    first_norms = backend_scores[0].document_norms_sq
    if first_norms.ndim != 1:
        raise ValueError("ordinary backend document norms must be query-independent")
    adversarial_norms = np.broadcast_to(
        first_norms, adversarial_dots.shape
    ).copy()
    document_index = {document_id: index for index, document_id in enumerate(document_ids)}
    for row, query_id in enumerate(query_ids):
        relevant = {
            document_index[item]
            for item in qrels[query_id]
            if item in document_index
        }
        for backend in backend_scores[1:]:
            backend_norms = backend.document_norms_sq
            if backend_norms.ndim != 1:
                raise ValueError(
                    "ordinary backend document norms must be query-independent"
                )
            for column in range(len(document_ids)):
                candidate_dot = int(backend.dots[row, column])
                candidate_norm = int(backend_norms[column])
                ordered = exact_i8_cosine_desc(
                    candidate_dot,
                    candidate_norm,
                    int(adversarial_dots[row, column]),
                    int(adversarial_norms[row, column]),
                )
                choose_candidate = ordered > 0 if column in relevant else ordered < 0
                if choose_candidate:
                    adversarial_dots[row, column] = candidate_dot
                    adversarial_norms[row, column] = candidate_norm
    return ExactI8Scores(adversarial_dots, adversarial_norms)


def pair_key(query_label: str, document_label: str) -> str:
    return f"{query_label}__queries--{document_label}__documents"


def sequence_semantic_pair_result(
    query_vectors: np.ndarray,
    relevant_document_vectors: np.ndarray,
    irrelevant_document_vectors: np.ndarray,
) -> dict[str, object]:
    """Record sufficient integers to replay every bucket's exact rank decision."""
    import numpy as np

    expected_shape = (len(SEQUENCE_BUCKETS), 768)
    arrays = (query_vectors, relevant_document_vectors, irrelevant_document_vectors)
    if any(array.dtype != np.dtype(np.int8) for array in arrays):
        raise ValueError("sequence semantic probes must be signed INT8 arrays")
    if any(array.shape != expected_shape for array in arrays):
        raise ValueError(
            f"sequence semantic probes must all have shape {expected_shape}"
        )
    query_int = query_vectors.astype(np.int64)
    relevant_int = relevant_document_vectors.astype(np.int64)
    irrelevant_int = irrelevant_document_vectors.astype(np.int64)
    relevant_dots = np.sum(query_int * relevant_int, axis=1, dtype=np.int64)
    irrelevant_dots = np.sum(query_int * irrelevant_int, axis=1, dtype=np.int64)
    relevant_norms = np.sum(relevant_int * relevant_int, axis=1, dtype=np.int64)
    irrelevant_norms = np.sum(irrelevant_int * irrelevant_int, axis=1, dtype=np.int64)
    if np.any(relevant_norms <= 0) or np.any(irrelevant_norms <= 0):
        raise ValueError("sequence semantic probe documents must be non-zero")
    return {
        "buckets": {
            str(bucket): {
                "relevant_dot": int(relevant_dots[index]),
                "relevant_document_norm_sq": int(relevant_norms[index]),
                "irrelevant_dot": int(irrelevant_dots[index]),
                "irrelevant_document_norm_sq": int(irrelevant_norms[index]),
            }
            for index, bucket in enumerate(SEQUENCE_BUCKETS)
        }
    }


def evaluate_sequence_semantic_gate(
    labels: list[str],
    expected_scopes: set[str],
    pair_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Evaluate every ordered pair and worst mixed store at every bucket.

    For each query scope and bucket, the mixed-store check compares the lowest
    relevant score from any document scope with the highest irrelevant score
    from any document scope. This is the sequence-fixture equivalent of the
    production admission gate's derive-once adversarial shared store.
    """
    expected_pairs = {
        pair_key(query_label, document_label)
        for query_label in expected_scopes
        for document_label in expected_scopes
    }
    expected_bucket_keys = {str(bucket) for bucket in SEQUENCE_BUCKETS}
    evaluated_pairs = set(pair_results)
    pair_bucket_checks: dict[str, dict[str, bool]] = {}
    evaluated_bucket_count = 0
    all_bucket_sets_complete = True
    for key, result in pair_results.items():
        buckets = result.get("buckets")
        if not isinstance(buckets, dict):
            all_bucket_sets_complete = False
            pair_bucket_checks[key] = {}
            continue
        evaluated_bucket_count += len(buckets)
        if set(buckets) != expected_bucket_keys:
            all_bucket_sets_complete = False
        checks: dict[str, bool] = {}
        for bucket in sorted(expected_bucket_keys, key=int):
            row = buckets.get(bucket)
            if not isinstance(row, dict):
                continue
            checks[bucket] = (
                exact_i8_cosine_desc(
                    row["relevant_dot"],
                    row["relevant_document_norm_sq"],
                    row["irrelevant_dot"],
                    row["irrelevant_document_norm_sq"],
                )
                < 0
            )
        pair_bucket_checks[key] = checks
    all_ordered_pairs_evaluated = evaluated_pairs == expected_pairs
    all_sequence_buckets_evaluated = (
        all_ordered_pairs_evaluated
        and all_bucket_sets_complete
        and all(
            set(pair_bucket_checks[key]) == expected_bucket_keys
            for key in expected_pairs
        )
    )
    adversarial_mixed_document_bucket_checks: dict[str, dict[str, bool]] = {}
    evaluated_adversarial_mixed_document_check_count = 0
    for query_scope in sorted(expected_scopes):
        checks: dict[str, bool] = {}
        for bucket in sorted(expected_bucket_keys, key=int):
            rows: list[tuple[str, dict[str, int]]] = []
            for document_scope in sorted(expected_scopes):
                result = pair_results.get(pair_key(query_scope, document_scope))
                buckets = result.get("buckets") if isinstance(result, dict) else None
                row = buckets.get(bucket) if isinstance(buckets, dict) else None
                if isinstance(row, dict):
                    rows.append((document_scope, row))
            if len(rows) != len(expected_scopes):
                continue

            # Sorted document-scope order is deterministic when exact scores tie;
            # the selected score itself is identical, so this does not affect the
            # final strict relevant-before-irrelevant decision.
            _, minimum_relevant = rows[0]
            _, maximum_irrelevant = rows[0]
            for _, candidate in rows[1:]:
                if (
                    exact_i8_cosine_desc(
                        candidate["relevant_dot"],
                        candidate["relevant_document_norm_sq"],
                        minimum_relevant["relevant_dot"],
                        minimum_relevant["relevant_document_norm_sq"],
                    )
                    > 0
                ):
                    minimum_relevant = candidate
                if (
                    exact_i8_cosine_desc(
                        candidate["irrelevant_dot"],
                        candidate["irrelevant_document_norm_sq"],
                        maximum_irrelevant["irrelevant_dot"],
                        maximum_irrelevant["irrelevant_document_norm_sq"],
                    )
                    < 0
                ):
                    maximum_irrelevant = candidate
            checks[bucket] = (
                exact_i8_cosine_desc(
                    minimum_relevant["relevant_dot"],
                    minimum_relevant["relevant_document_norm_sq"],
                    maximum_irrelevant["irrelevant_dot"],
                    maximum_irrelevant["irrelevant_document_norm_sq"],
                )
                < 0
            )
            evaluated_adversarial_mixed_document_check_count += 1
        adversarial_mixed_document_bucket_checks[query_scope] = checks
    expected_adversarial_mixed_document_check_count = (
        len(expected_scopes) * len(SEQUENCE_BUCKETS)
    )
    all_adversarial_mixed_document_checks_evaluated = (
        evaluated_adversarial_mixed_document_check_count
        == expected_adversarial_mixed_document_check_count
        and all(
            set(checks) == expected_bucket_keys
            for checks in adversarial_mixed_document_bucket_checks.values()
        )
    )
    supplied_scopes = set(labels)
    passed = (
        supplied_scopes == expected_scopes
        and all_sequence_buckets_evaluated
        and all(
            all(checks.values())
            for key, checks in pair_bucket_checks.items()
            if key in expected_pairs
        )
        and all_adversarial_mixed_document_checks_evaluated
        and all(
            all(checks.values())
            for checks in adversarial_mixed_document_bucket_checks.values()
        )
    )
    return {
        "fixture_id": SEQUENCE_SEMANTIC_FIXTURE_ID,
        "fixture_sha256": SEQUENCE_SEMANTIC_FIXTURE_SHA256,
        "ranking": SEQUENCE_SEMANTIC_GATE,
        "expected_scopes": sorted(expected_scopes),
        "evaluated_scopes": sorted(supplied_scopes),
        "missing_scopes": sorted(expected_scopes.difference(supplied_scopes)),
        "unexpected_scopes": sorted(supplied_scopes.difference(expected_scopes)),
        "all_expected_scopes_present": supplied_scopes == expected_scopes,
        "expected_ordered_pair_count": len(expected_pairs),
        "evaluated_ordered_pair_count": len(evaluated_pairs),
        "missing_ordered_pairs": sorted(expected_pairs.difference(evaluated_pairs)),
        "unexpected_ordered_pairs": sorted(evaluated_pairs.difference(expected_pairs)),
        "all_ordered_pairs_evaluated": all_ordered_pairs_evaluated,
        "expected_bucket_check_count": len(expected_pairs) * len(SEQUENCE_BUCKETS),
        "evaluated_bucket_check_count": evaluated_bucket_count,
        "all_sequence_buckets_evaluated": all_sequence_buckets_evaluated,
        "pair_bucket_checks": pair_bucket_checks,
        "expected_adversarial_mixed_document_check_count": (
            expected_adversarial_mixed_document_check_count
        ),
        "evaluated_adversarial_mixed_document_check_count": (
            evaluated_adversarial_mixed_document_check_count
        ),
        "all_adversarial_mixed_document_checks_evaluated": (
            all_adversarial_mixed_document_checks_evaluated
        ),
        "adversarial_mixed_document_bucket_checks": (
            adversarial_mixed_document_bucket_checks
        ),
        "passed": passed,
    }


def evaluate_compatibility_gate(
    labels: list[str],
    expected_scopes: set[str],
    classes: set[str],
    pair_metrics: dict[str, dict[str, object]],
    mixed_document_metrics: dict[str, dict[str, float]],
) -> dict[str, object]:
    expected_pairs = {
        pair_key(query_label, document_label)
        for query_label in expected_scopes
        for document_label in expected_scopes
    }
    supplied_scopes = set(labels)
    evaluated_pairs = set(pair_metrics)
    pair_checks = {
        key: {
            metric: result[metric] >= minimum
            for metric, minimum in ABSOLUTE_MINIMUM.items()
        }
        for key, result in pair_metrics.items()
    }
    mixed_document_checks = {
        scope_id: {
            metric: result[metric] >= minimum
            for metric, minimum in ABSOLUTE_MINIMUM.items()
        }
        for scope_id, result in mixed_document_metrics.items()
    }
    complete_classes = REQUIRED_CLASSES.issubset(classes)
    all_ordered_pairs_evaluated = evaluated_pairs == expected_pairs
    passed = (
        supplied_scopes == expected_scopes
        and complete_classes
        and all_ordered_pairs_evaluated
        and all(all(checks.values()) for checks in pair_checks.values())
        and set(mixed_document_metrics) == expected_scopes
        and all(all(checks.values()) for checks in mixed_document_checks.values())
    )
    return {
        "required_device_classes": sorted(REQUIRED_CLASSES),
        "present_device_classes": sorted(classes),
        "all_required_device_classes_present": complete_classes,
        "expected_scopes": sorted(expected_scopes),
        "supplied_scopes": sorted(supplied_scopes),
        "missing_scopes": sorted(expected_scopes - supplied_scopes),
        "unexpected_scopes": sorted(supplied_scopes - expected_scopes),
        "all_expected_scopes_present": supplied_scopes == expected_scopes,
        "absolute_minimum_for_every_ordered_pair": ABSOLUTE_MINIMUM,
        "expected_ordered_pair_count": len(expected_pairs),
        "evaluated_ordered_pair_count": len(evaluated_pairs),
        "missing_ordered_pairs": sorted(expected_pairs - evaluated_pairs),
        "unexpected_ordered_pairs": sorted(evaluated_pairs - expected_pairs),
        "all_ordered_pairs_evaluated": all_ordered_pairs_evaluated,
        "pair_checks": pair_checks,
        "adversarial_mixed_document_scopes_complete": (
            set(mixed_document_metrics) == expected_scopes
        ),
        "adversarial_mixed_document_checks": mixed_document_checks,
        "passed": passed,
    }


def evaluate_admission_gate(
    vector_compatibility_gate: dict[str, object],
    sequence_semantic_gate: dict[str, object],
) -> dict[str, bool]:
    vector_passed = vector_compatibility_gate.get("passed") is True
    sequence_passed = sequence_semantic_gate.get("passed") is True
    return {
        "vector_compatibility_passed": vector_passed,
        "sequence_semantic_conformance_passed": sequence_passed,
        "passed": vector_passed and sequence_passed,
    }


REPORT_BACKEND_BINDING_FIELDS = (
    "profile_manifest_sha256",
    "admission_policy_sha256",
    "scope_id",
    "transport",
    "backend",
    "runtime",
    "compiler",
    "package_target",
    "artifact_source",
    "artifact_sha256",
    "internal_precision",
    "device",
    "device_class",
    "placement_evidence_sha256",
    "supported_max_tokens",
    "supported_sequence_buckets",
    "supported_max_batch_size",
    "sequence_capability_evidence_sha256",
    "performance_evidence_sha256",
    "attestation_public_key",
    "accelerated_placement",
)
REPORT_BACKEND_FIXED_IDENTITY = {
    "schema_version": 1,
    "profile_id": PROFILE_ID,
    "profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
    "admission_policy_sha256": ADMISSION_POLICY_SHA256,
    "model": MODEL,
    "model_revision": MODEL_REVISION,
    "vector_encoding": VECTOR_ENCODING,
    "supported_max_tokens": MAX_TOKENS,
    "supported_sequence_buckets": SEQUENCE_BUCKETS,
    "supported_max_batch_size": SUPPORTED_MAX_BATCH_SIZE,
    "sequence_semantic_fixture_id": SEQUENCE_SEMANTIC_FIXTURE_ID,
    "sequence_semantic_fixture_sha256": SEQUENCE_SEMANTIC_FIXTURE_SHA256,
    "dataset": DATASET,
    "dataset_revision": DATASET_REVISION,
}
REPORT_BACKEND_METADATA_FIELDS = tuple(
    dict.fromkeys((*REPORT_BACKEND_FIXED_IDENTITY, *REPORT_BACKEND_BINDING_FIELDS))
)
REPORT_TOP_LEVEL_FIELDS = {
    "schema_version",
    "profile_id",
    "profile_manifest_sha256",
    "admission_policy_sha256",
    "admission_implementation_bundle_sha256",
    "ranking_semantics",
    "model",
    "model_revision",
    "vector_encoding",
    "max_tokens",
    "sequence_buckets",
    "max_batch_size",
    "dataset",
    "dataset_revision",
    "sequence_semantic_fixture",
    "parent_report",
    "parent_report_sha256",
    "already_admitted_scopes",
    "candidate_scopes",
    "backends",
    "cache_sha256_by_scope",
    "pair_metrics",
    "adversarial_mixed_document_metrics",
    "sequence_semantic_pair_results",
    "vector_compatibility_gate",
    "sequence_semantic_gate",
    "admission_gate",
}


def report_backend_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Project cache metadata onto the public, registry-bound report schema."""
    missing = set(REPORT_BACKEND_METADATA_FIELDS).difference(metadata)
    if missing:
        raise ValueError(
            "backend cache cannot produce a compatibility report without "
            f"canonical metadata fields {sorted(missing)}"
        )
    return {field: metadata[field] for field in REPORT_BACKEND_METADATA_FIELDS}


def _scope_set(value: object, field: str) -> set[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) for item in value)
        or len(value) != len(set(value))
        or value != sorted(value)
    ):
        raise ValueError(
            f"compatibility report {field} must be a sorted array of unique scope ids"
        )
    for item in value:
        try:
            scope_id_value(item)
        except argparse.ArgumentTypeError as error:
            raise ValueError(
                f"compatibility report {field} contains an invalid scope id"
            ) from error
    return set(value)


def report_scope_partition(
    report: dict[str, object], context: str = "compatibility report"
) -> tuple[set[str], set[str], set[str]]:
    already = _scope_set(report.get("already_admitted_scopes"), "already_admitted_scopes")
    candidates = _scope_set(report.get("candidate_scopes"), "candidate_scopes")
    if already.intersection(candidates):
        raise ValueError(f"{context} admitted/candidate scope partitions overlap")
    if not candidates:
        raise ValueError(f"{context} must add at least one candidate scope")
    return already, candidates, already.union(candidates)


def validate_parent_report_declaration(
    report: dict[str, object], already: set[str]
) -> tuple[str, str] | None:
    parent_reference = report.get("parent_report")
    parent_digest = report.get("parent_report_sha256")
    if parent_reference is None and parent_digest is None:
        if already:
            raise ValueError(
                "genesis compatibility report must have no already-admitted scopes"
            )
        return None
    if not isinstance(parent_reference, str) or not isinstance(parent_digest, str):
        raise ValueError(
            "compatibility report parent_report and parent_report_sha256 must "
            "both be null or both be strings"
        )
    if not already:
        raise ValueError(
            "successor compatibility report must retain an already-admitted cohort"
        )
    validate_admission_report_reference(parent_reference, parent_digest)
    return parent_reference, parent_digest


def validate_report_lineage_edge(
    child: dict[str, object], parent: dict[str, object]
) -> None:
    child_already, child_candidates, child_total = report_scope_partition(
        child, "child compatibility report"
    )
    _, _, parent_total = report_scope_partition(
        parent, "parent compatibility report"
    )
    if child_already != parent_total:
        raise ValueError(
            "child already_admitted_scopes must equal the parent report's total cohort"
        )
    if child_candidates != child_total.difference(parent_total):
        raise ValueError(
            "child candidate_scopes must be exactly the scopes added after its parent"
        )
    for scope_id in parent_total:
        if (
            child["backends"][scope_id] != parent["backends"][scope_id]
            or child["cache_sha256_by_scope"][scope_id]
            != parent["cache_sha256_by_scope"][scope_id]
        ):
            raise ValueError(
                f"child compatibility report changed retained scope {scope_id}"
            )


def _validate_report_metrics(
    result: object, required_fields: set[str], context: str
) -> dict[str, object]:
    if not isinstance(result, dict) or set(result) != required_fields:
        raise ValueError(
            f"compatibility report {context} must contain exactly "
            f"{sorted(required_fields)}"
        )
    for metric in ABSOLUTE_MINIMUM:
        value = result[metric]
        if (
            type(value) not in {int, float}
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(
                f"compatibility report {context} {metric} must be finite and in 0..1"
            )
    return result


def validate_single_cohort_report_binding(
    scopes: dict[str, dict[str, object]],
) -> tuple[str, str] | None:
    """Require one atomic global report identity for the admitted cohort."""
    bindings: set[tuple[str, str]] = set()
    for scope_id, entry in scopes.items():
        report_path = entry.get("compatibility_report")
        report_digest = entry.get("compatibility_report_sha256")
        if not isinstance(report_path, str) or not isinstance(report_digest, str):
            raise ValueError(
                f"release cohort scope {scope_id} must bind a compatibility report"
            )
        validate_admission_report_reference(report_path, report_digest)
        bindings.add((report_path, report_digest))
    if len(bindings) > 1:
        raise ValueError(
            "all admitted backends must reference exactly the same compatibility "
            "report path and sha256"
        )
    return next(iter(bindings)) if bindings else None


def validate_stored_admission_report(
    report: object, scopes: dict[str, dict[str, object]]
) -> None:
    """Structurally preflight a committed report before full cache replay.

    This catches stale identities, malformed metrics, and forged aggregate
    verdicts without network access. It is not the admission proof: CI must
    also call ``verify_release_registry`` to recompute metrics from the exact
    content-addressed caches and verify retained measurement outputs.
    """
    if not isinstance(report, dict):
        raise ValueError("compatibility report must be a JSON object")
    if set(report) != REPORT_TOP_LEVEL_FIELDS:
        raise ValueError(
            "compatibility report must contain exactly the canonical public fields"
        )
    expected_header = {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
        "admission_policy_sha256": ADMISSION_POLICY_SHA256,
        "admission_implementation_bundle_sha256": (
            ADMISSION_IMPLEMENTATION_BUNDLE_SHA256
        ),
        "ranking_semantics": RANKING_SEMANTICS,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "vector_encoding": VECTOR_ENCODING,
        "max_tokens": MAX_TOKENS,
        "sequence_buckets": SEQUENCE_BUCKETS,
        "max_batch_size": SUPPORTED_MAX_BATCH_SIZE,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "sequence_semantic_fixture": {
            "id": SEQUENCE_SEMANTIC_FIXTURE_ID,
            "sha256": SEQUENCE_SEMANTIC_FIXTURE_SHA256,
            "bucket_order": SEQUENCE_BUCKETS,
            "ranking": SEQUENCE_SEMANTIC_GATE,
        },
    }
    for field, expected in expected_header.items():
        if report.get(field) != expected:
            raise ValueError(
                f"compatibility report {field}={report.get(field)!r}, expected {expected!r}"
            )

    expected_scopes = set(scopes)
    attestation_keys: list[str] = []
    for scope_id, entry in scopes.items():
        public_key = entry.get("attestation_public_key")
        if not isinstance(public_key, str) or re.fullmatch(
            r"[0-9a-f]{64}", public_key
        ) is None:
            raise ValueError(
                f"release cohort scope {scope_id} must bind a canonical Ed25519 public key"
            )
        attestation_keys.append(public_key)
    if len(attestation_keys) != len(set(attestation_keys)):
        raise ValueError(
            "release cohort must bind a unique Ed25519 attestation public key "
            "to every scope"
        )
    already_admitted, candidates, report_scopes = report_scope_partition(report)
    validate_parent_report_declaration(report, already_admitted)
    if report_scopes != expected_scopes:
        raise ValueError(
            "compatibility report admitted/candidate scope partition does not match "
            "the release cohort"
        )

    backends = report.get("backends")
    if not isinstance(backends, dict) or set(backends) != expected_scopes:
        raise ValueError(
            "compatibility report backends must contain the exact release cohort"
        )
    cache_digests = report.get("cache_sha256_by_scope")
    if not isinstance(cache_digests, dict) or set(cache_digests) != expected_scopes:
        raise ValueError(
            "compatibility report cache_sha256_by_scope must contain the exact cohort"
        )
    for scope_id, digest in cache_digests.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(
                f"compatibility report cache digest for {scope_id} must be a lowercase sha256"
            )
        if digest != scopes[scope_id].get("admission_cache_sha256"):
            raise ValueError(
                f"compatibility report cache digest for {scope_id} does not match "
                "the release registry"
            )

    for scope_id, entry in scopes.items():
        metadata = backends[scope_id]
        if not isinstance(metadata, dict):
            raise ValueError(
                f"compatibility report backend {scope_id} metadata must be an object"
            )
        if set(metadata) != set(REPORT_BACKEND_METADATA_FIELDS):
            raise ValueError(
                f"compatibility report backend {scope_id} metadata must contain "
                "only canonical public admission fields"
            )
        for field, expected in REPORT_BACKEND_FIXED_IDENTITY.items():
            if metadata.get(field) != expected:
                raise ValueError(
                    f"compatibility report backend {scope_id} {field} does not match "
                    "the frozen cache identity"
                )
        for field in REPORT_BACKEND_BINDING_FIELDS:
            if metadata.get(field) != entry.get(field):
                raise ValueError(
                    f"compatibility report backend {scope_id} {field} does not match "
                    "the release registry"
                )

    pair_metrics = report.get("pair_metrics")
    if not isinstance(pair_metrics, dict):
        raise ValueError("compatibility report pair_metrics must be an object")
    expected_pairs = {
        pair_key(query_scope, document_scope)
        for query_scope in expected_scopes
        for document_scope in expected_scopes
    }
    if set(pair_metrics) != expected_pairs:
        raise ValueError(
            "compatibility report pair_metrics must contain every exact ordered pair"
        )
    pair_fields = set(ABSOLUTE_MINIMUM).union(
        {"query_backend", "document_backend"}
    )
    for query_scope in expected_scopes:
        for document_scope in expected_scopes:
            key = pair_key(query_scope, document_scope)
            result = _validate_report_metrics(pair_metrics[key], pair_fields, key)
            if (
                result["query_backend"] != query_scope
                or result["document_backend"] != document_scope
            ):
                raise ValueError(
                    f"compatibility report pair {key} carries the wrong producer identities"
                )

    mixed_metrics = report.get("adversarial_mixed_document_metrics")
    if not isinstance(mixed_metrics, dict) or set(mixed_metrics) != expected_scopes:
        raise ValueError(
            "compatibility report adversarial metrics must contain every query scope"
        )
    for scope_id, result in mixed_metrics.items():
        _validate_report_metrics(
            result,
            set(ABSOLUTE_MINIMUM),
            f"adversarial mixed-document scope {scope_id}",
        )

    sequence_pair_results = report.get("sequence_semantic_pair_results")
    if (
        not isinstance(sequence_pair_results, dict)
        or set(sequence_pair_results) != expected_pairs
    ):
        raise ValueError(
            "compatibility report sequence semantic results must contain every "
            "exact ordered pair"
        )
    expected_bucket_keys = {str(bucket) for bucket in SEQUENCE_BUCKETS}
    bucket_result_fields = {
        "relevant_dot",
        "relevant_document_norm_sq",
        "irrelevant_dot",
        "irrelevant_document_norm_sq",
    }
    for query_scope in expected_scopes:
        for document_scope in expected_scopes:
            key = pair_key(query_scope, document_scope)
            result = sequence_pair_results[key]
            if not isinstance(result, dict) or set(result) != {
                "query_backend",
                "document_backend",
                "buckets",
            }:
                raise ValueError(
                    f"compatibility report sequence semantic pair {key} has an "
                    "invalid structure"
                )
            if (
                result["query_backend"] != query_scope
                or result["document_backend"] != document_scope
            ):
                raise ValueError(
                    f"compatibility report sequence semantic pair {key} carries "
                    "the wrong producer identities"
                )
            buckets = result["buckets"]
            if not isinstance(buckets, dict) or set(buckets) != expected_bucket_keys:
                raise ValueError(
                    f"compatibility report sequence semantic pair {key} must cover "
                    "every profile bucket"
                )
            for bucket, bucket_result in buckets.items():
                if (
                    not isinstance(bucket_result, dict)
                    or set(bucket_result) != bucket_result_fields
                ):
                    raise ValueError(
                        f"compatibility report sequence semantic pair {key} bucket "
                        f"{bucket} has an invalid structure"
                    )
                for field, value in bucket_result.items():
                    if type(value) is not int:
                        raise ValueError(
                            f"compatibility report sequence semantic pair {key} "
                            f"bucket {bucket} {field} must be an exact integer"
                        )
                    if field.endswith("norm_sq"):
                        valid = 0 < value <= MAX_EXACT_I8_DOT_OR_NORM_SQ
                    else:
                        valid = abs(value) <= MAX_EXACT_I8_DOT_OR_NORM_SQ
                    if not valid:
                        raise ValueError(
                            f"compatibility report sequence semantic pair {key} "
                            f"bucket {bucket} {field} is outside signed INT8x768 bounds"
                        )

    classes = {backends[scope_id]["device_class"] for scope_id in expected_scopes}
    recomputed_gate = evaluate_compatibility_gate(
        sorted(expected_scopes),
        expected_scopes,
        classes,
        pair_metrics,
        mixed_metrics,
    )
    if report.get("vector_compatibility_gate") != recomputed_gate:
        raise ValueError(
            "compatibility report gate structure/checks/counts do not match its metrics"
        )
    recomputed_sequence_gate = evaluate_sequence_semantic_gate(
        sorted(expected_scopes), expected_scopes, sequence_pair_results
    )
    if report.get("sequence_semantic_gate") != recomputed_sequence_gate:
        raise ValueError(
            "compatibility report sequence semantic gate structure/checks/counts "
            "do not match its exact integer results"
        )
    recomputed_admission_gate = evaluate_admission_gate(
        recomputed_gate, recomputed_sequence_gate
    )
    if report.get("admission_gate") != recomputed_admission_gate:
        raise ValueError(
            "compatibility report combined admission verdict does not match its gates"
        )
    if recomputed_admission_gate["passed"] is not True:
        raise ValueError("compatibility report does not pass the current global gate")


def load_validated_report_lineage(
    scopes: dict[str, dict[str, object]],
    root_reference: str,
    root_digest: str,
) -> list[tuple[Path, dict[str, object], set[str], set[str]]]:
    """Load and structurally validate the bounded newest-to-genesis chain."""
    nodes: list[tuple[Path, dict[str, object], set[str], set[str]]] = []
    visited: set[tuple[str, str]] = set()
    lineage_bytes = 0
    reference = root_reference
    digest = root_digest
    child_report: dict[str, object] | None = None
    registry_scopes = set(scopes)

    while True:
        if len(nodes) >= MAX_ADMITTED_SCOPES:
            raise ValueError("compatibility report lineage exceeds its depth bound")
        identity = (reference, digest)
        if identity in visited:
            raise ValueError("compatibility report lineage repeats a report")
        visited.add(identity)
        path, report, report_bytes = read_content_addressed_report(reference, digest)
        lineage_bytes += report_bytes
        if lineage_bytes > MAX_ADMISSION_REPORT_LINEAGE_BYTES:
            raise ValueError("compatibility report lineage exceeds its byte bound")

        already, candidates, total = report_scope_partition(report)
        if not nodes and total != registry_scopes:
            raise ValueError(
                "latest compatibility report cohort does not match the release registry"
            )
        unknown_scopes = total.difference(registry_scopes)
        if unknown_scopes:
            raise ValueError(
                "compatibility report lineage contains scopes outside the current registry"
            )
        report_scopes = {scope_id: scopes[scope_id] for scope_id in total}
        validate_stored_admission_report(report, report_scopes)
        if child_report is not None:
            validate_report_lineage_edge(child_report, report)
        nodes.append((path, report, already, candidates))

        parent = validate_parent_report_declaration(report, already)
        if parent is None:
            break
        child_report = report
        reference, digest = parent

    return nodes


def validate_release_asset_url(
    scope_id: str, field: str, url: str, digest: str, suffix: str
) -> None:
    """Require a content-addressed asset in cfetch's own GitHub release."""
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise ValueError(
            f"admitted backend {scope_id} {field} is invalid"
        ) from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.query
        or parsed.fragment
        or re.fullmatch(
            rf"/corbet-labs/cfetch/releases/download/"
            rf"[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}/"
            rf"{re.escape(digest)}{re.escape(suffix)}",
            parsed.path,
        )
        is None
    ):
        raise ValueError(
            f"admitted backend {scope_id} {field} must be a "
            f"content-addressed cfetch GitHub release URL ending in its sha256 plus {suffix}"
        )


def validate_admission_cache_url(scope_id: str, url: str, digest: str) -> None:
    validate_release_asset_url(scope_id, "admission_cache_url", url, digest, ".npz")


def validate_measurement_evidence_url(scope_id: str, url: str, digest: str) -> None:
    validate_release_asset_url(
        scope_id, "measurement_evidence_url", url, digest, ".zip"
    )


def admitted_scopes() -> dict[str, dict[str, object]]:
    repository = Path(__file__).resolve().parents[2].resolve()
    declared_release_root = repository / "release"
    release_root = declared_release_root.resolve()
    registry_path = (repository / "release/inference-backends.json").resolve()
    if (
        declared_release_root.is_symlink()
        or release_root.parent != repository
        or registry_path.parent != release_root
    ):
        raise ValueError("release registry escaped the repository release directory")
    registry = parse_evidence_json(
        read_bounded_local_file(
            registry_path, MAX_ADMISSION_REGISTRY_BYTES, "release registry"
        ),
        str(registry_path),
    )
    if registry.get("profile_id") != PROFILE_ID or registry.get("shared_identity", {}).get(
        "profile_manifest_sha256"
    ) != PROFILE_MANIFEST_SHA256:
        raise ValueError("release registry does not match this semantic profile")
    if registry.get("admission", {}).get("policy_manifest_sha256") != ADMISSION_POLICY_SHA256:
        raise ValueError("release registry does not match this admission policy")
    if (
        registry.get("admission", {}).get("implementation_bundle_sha256")
        != ADMISSION_IMPLEMENTATION_BUNDLE_SHA256
    ):
        raise ValueError(
            "release registry does not match this admission implementation bundle"
        )
    if registry.get("admission", {}).get("evidence_replay") != EVIDENCE_REPLAY_POLICY:
        raise ValueError("release registry does not match this evidence replay policy")
    entries = registry.get("admitted_backends")
    if not isinstance(entries, list):
        raise ValueError("release registry admitted_backends must be an array")
    if len(entries) > MAX_ADMITTED_SCOPES:
        raise ValueError(
            f"release registry exceeds the {MAX_ADMITTED_SCOPES}-scope replay bound"
        )
    if entries and registry.get("profile_status") != "active":
        raise ValueError("a candidate profile cannot contain production-admitted backends")
    scopes: dict[str, dict[str, object]] = {}
    attestation_public_keys: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("scope_id"), str):
            raise ValueError("every admitted backend must have a string scope_id")
        try:
            scope_id_value(entry["scope_id"])
        except argparse.ArgumentTypeError as error:
            raise ValueError(f"invalid admitted backend scope_id: {error}") from error
        if entry["scope_id"] in scopes:
            raise ValueError("admitted backend scope_ids must be non-empty and unique")
        for field in (
            "backend",
            "transport",
            "runtime",
            "compiler",
            "package_target",
            "artifact_source",
            "artifact_sha256",
            "internal_precision",
            "device",
            "device_class",
            "placement_evidence_sha256",
            "sequence_capability_evidence_sha256",
            "performance_evidence_sha256",
            "admission_cache_url",
            "admission_cache_sha256",
            "measurement_evidence_url",
            "measurement_evidence_sha256",
            "compatibility_report",
            "compatibility_report_sha256",
            "attestation_public_key",
        ):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ValueError(f"admitted backend {entry['scope_id']} needs {field}")
        if entry["transport"] not in TRANSPORTS:
            raise ValueError(
                f"admitted backend {entry['scope_id']} transport must be "
                "supervised-local or remote-attested"
            )
        for field in (
            "artifact_sha256",
            "placement_evidence_sha256",
            "sequence_capability_evidence_sha256",
            "performance_evidence_sha256",
            "admission_cache_sha256",
            "measurement_evidence_sha256",
            "compatibility_report_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", entry[field]) is None:
                raise ValueError(
                    f"admitted backend {entry['scope_id']} {field} must be a lowercase sha256"
                )
        validate_admission_cache_url(
            entry["scope_id"],
            entry["admission_cache_url"],
            entry["admission_cache_sha256"],
        )
        validate_measurement_evidence_url(
            entry["scope_id"],
            entry["measurement_evidence_url"],
            entry["measurement_evidence_sha256"],
        )
        if re.fullmatch(r"[0-9a-f]{64}", entry["attestation_public_key"]) is None:
            raise ValueError(
                f"admitted backend {entry['scope_id']} attestation_public_key must be "
                "a 64-character lowercase hexadecimal Ed25519 public key"
            )
        if entry["attestation_public_key"] in attestation_public_keys:
            raise ValueError(
                "admitted backend attestation_public_key values must be unique per scope"
            )
        attestation_public_keys.add(entry["attestation_public_key"])
        if entry.get("accelerated_placement") is not True:
            raise ValueError(
                f"admitted backend {entry['scope_id']} must record accelerated placement"
            )
        if entry.get("supported_max_tokens") != MAX_TOKENS:
            raise ValueError(
                f"admitted backend {entry['scope_id']} must support {MAX_TOKENS} tokens"
            )
        if entry.get("supported_sequence_buckets") != SEQUENCE_BUCKETS:
            raise ValueError(
                f"admitted backend {entry['scope_id']} must support every profile bucket"
            )
        if entry.get("supported_max_batch_size") != SUPPORTED_MAX_BATCH_SIZE:
            raise ValueError(
                f"admitted backend {entry['scope_id']} must support batch sizes "
                f"1 through {SUPPORTED_MAX_BATCH_SIZE}"
            )
        if entry.get("profile_manifest_sha256") != PROFILE_MANIFEST_SHA256:
            raise ValueError(
                f"admitted backend {entry['scope_id']} belongs to another semantic profile"
            )
        if entry.get("admission_policy_sha256") != ADMISSION_POLICY_SHA256:
            raise ValueError(
                f"admitted backend {entry['scope_id']} has stale admission evidence"
            )
        scopes[entry["scope_id"]] = entry
    report_binding = validate_single_cohort_report_binding(scopes)
    if report_binding is not None:
        load_validated_report_lineage(scopes, *report_binding)
    return scopes


REGISTRY_CACHE_BINDING_FIELDS = (
    "transport",
    "backend",
    "runtime",
    "compiler",
    "package_target",
    "artifact_source",
    "device_class",
    "device",
    "artifact_sha256",
    "attestation_public_key",
    "internal_precision",
    "placement_evidence_sha256",
    "supported_max_tokens",
    "supported_sequence_buckets",
    "supported_max_batch_size",
    "sequence_capability_evidence_sha256",
    "performance_evidence_sha256",
    "accelerated_placement",
)


def validate_loaded_scope_bindings(
    loaded: dict[str, tuple[dict[str, object], np.ndarray, np.ndarray]],
    scope_entries: dict[str, dict[str, object]],
) -> None:
    for scope_id, entry in scope_entries.items():
        cached = loaded.get(scope_id)
        if cached is None:
            continue
        metadata = cached[0]
        for field in REGISTRY_CACHE_BINDING_FIELDS:
            if metadata.get(field) != entry.get(field):
                raise ValueError(
                    f"admitted scope {scope_id!r} cache {field}={metadata.get(field)!r}, "
                    f"registry requires {entry.get(field)!r}"
                )


def build_compatibility_report(
    paths: dict[str, Path],
    loaded: dict[str, tuple[dict[str, object], np.ndarray, np.ndarray]],
    sequence_probes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    admitted_scope_ids: set[str],
    candidate_scopes: set[str],
    parent_report: str | None,
    parent_report_sha256: str | None,
) -> dict[str, object]:
    expected_scopes = admitted_scope_ids.union(candidate_scopes)
    classes = {metadata["device_class"] for metadata, _, _ in loaded.values()}

    contract = load_scifact_contract(QUERY_PREFIX, DOCUMENT_PREFIX)
    qrels = contract.qrels
    query_ids = contract.query_ids
    document_ids = contract.document_ids
    expected_wire_input_digest = ordered_input_json_sha256(
        wire_batch_inputs(contract.query_texts, contract.document_texts)
    )
    for path in paths.values():
        validate_wire_batch_output_cache(path, expected_wire_input_digest)
    expected_shape = (len(query_ids), len(document_ids))
    for label, (_, query_vectors, document_vectors) in loaded.items():
        if (len(query_vectors), len(document_vectors)) != expected_shape:
            raise ValueError(
                f"{label}: cache has {len(query_vectors)} queries/"
                f"{len(document_vectors)} documents; pinned SciFact requires "
                f"{expected_shape[0]}/{expected_shape[1]}"
            )

    pair_metrics: dict[str, dict[str, object]] = {}
    for query_label, (_, query_vectors, _) in loaded.items():
        for document_label, (_, _, document_vectors) in loaded.items():
            key = pair_key(query_label, document_label)
            pair_metrics[key] = {
                "query_backend": query_label,
                "document_backend": document_label,
                **metrics(
                    scores(query_vectors, document_vectors),
                    query_ids,
                    document_ids,
                    qrels,
                ),
            }

    all_document_vectors = [documents for _, _, documents in loaded.values()]
    mixed_document_metrics: dict[str, dict[str, float]] = {}
    for query_label, (_, query_vectors, _) in loaded.items():
        mixed_document_metrics[query_label] = metrics(
            adversarial_mixed_document_scores(
                query_vectors,
                all_document_vectors,
                query_ids,
                document_ids,
                qrels,
            ),
            query_ids,
            document_ids,
            qrels,
        )

    sequence_pair_results: dict[str, dict[str, object]] = {}
    for query_label, (probe_queries, _, _) in sequence_probes.items():
        for document_label, (
            _,
            probe_relevant_documents,
            probe_irrelevant_documents,
        ) in sequence_probes.items():
            key = pair_key(query_label, document_label)
            sequence_pair_results[key] = {
                "query_backend": query_label,
                "document_backend": document_label,
                **sequence_semantic_pair_result(
                    probe_queries,
                    probe_relevant_documents,
                    probe_irrelevant_documents,
                ),
            }

    compatibility_gate = evaluate_compatibility_gate(
        list(loaded),
        expected_scopes,
        classes,
        pair_metrics,
        mixed_document_metrics,
    )
    sequence_semantic_gate = evaluate_sequence_semantic_gate(
        list(sequence_probes), expected_scopes, sequence_pair_results
    )
    admission_gate = evaluate_admission_gate(
        compatibility_gate, sequence_semantic_gate
    )
    return {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
        "admission_policy_sha256": ADMISSION_POLICY_SHA256,
        "admission_implementation_bundle_sha256": (
            ADMISSION_IMPLEMENTATION_BUNDLE_SHA256
        ),
        "ranking_semantics": RANKING_SEMANTICS,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "vector_encoding": VECTOR_ENCODING,
        "max_tokens": MAX_TOKENS,
        "sequence_buckets": SEQUENCE_BUCKETS,
        "max_batch_size": SUPPORTED_MAX_BATCH_SIZE,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "sequence_semantic_fixture": {
            "id": SEQUENCE_SEMANTIC_FIXTURE_ID,
            "sha256": SEQUENCE_SEMANTIC_FIXTURE_SHA256,
            "bucket_order": SEQUENCE_BUCKETS,
            "ranking": SEQUENCE_SEMANTIC_GATE,
        },
        "parent_report": parent_report,
        "parent_report_sha256": parent_report_sha256,
        "already_admitted_scopes": sorted(admitted_scope_ids),
        "candidate_scopes": sorted(candidate_scopes),
        "backends": {
            label: report_backend_metadata(metadata)
            for label, (metadata, _, _) in loaded.items()
        },
        "cache_sha256_by_scope": {
            label: file_sha256(paths[label]) for label in sorted(paths)
        },
        "pair_metrics": pair_metrics,
        "adversarial_mixed_document_metrics": mixed_document_metrics,
        "sequence_semantic_pair_results": sequence_pair_results,
        "vector_compatibility_gate": compatibility_gate,
        "sequence_semantic_gate": sequence_semantic_gate,
        "admission_gate": admission_gate,
    }


QUALITY_REPLAY_ABS_TOLERANCE = 1e-12


def validate_replayed_report(
    stored_report: dict[str, object],
    replayed_report: dict[str, object],
    report_path: Path,
) -> None:
    """Compare all decisions exactly and allow only last-bit quality drift."""
    if set(stored_report) != set(replayed_report):
        raise ValueError(
            f"compatibility report {report_path} has a different top-level schema"
        )
    separately_compared = {
        "pair_metrics",
        "adversarial_mixed_document_metrics",
    }
    if any(
        stored_report[field] != replayed_report[field]
        for field in set(replayed_report).difference(separately_compared)
    ):
        raise ValueError(
            f"compatibility report {report_path} does not equal the full decision "
            "replay from its content-addressed caches"
        )

    for field in ("pair_metrics", "adversarial_mixed_document_metrics"):
        stored_metrics = stored_report[field]
        replayed_metrics = replayed_report[field]
        if (
            not isinstance(stored_metrics, dict)
            or not isinstance(replayed_metrics, dict)
            or set(stored_metrics) != set(replayed_metrics)
        ):
            raise ValueError(
                f"compatibility report {report_path} {field} does not match replay"
            )
        for scope_or_pair in replayed_metrics:
            stored_result = stored_metrics[scope_or_pair]
            replayed_result = replayed_metrics[scope_or_pair]
            if (
                not isinstance(stored_result, dict)
                or not isinstance(replayed_result, dict)
                or set(stored_result) != set(replayed_result)
            ):
                raise ValueError(
                    f"compatibility report {report_path} {field} "
                    f"{scope_or_pair} does not match replay"
                )
            for result_field, replayed_value in replayed_result.items():
                stored_value = stored_result[result_field]
                if result_field in ABSOLUTE_MINIMUM:
                    equal = (
                        type(stored_value) in {int, float}
                        and math.isclose(
                            float(stored_value),
                            float(replayed_value),
                            rel_tol=0.0,
                            abs_tol=QUALITY_REPLAY_ABS_TOLERANCE,
                        )
                    )
                else:
                    equal = stored_value == replayed_value
                if not equal:
                    raise ValueError(
                        f"compatibility report {report_path} {field} "
                        f"{scope_or_pair} {result_field} does not match replay"
                    )


def _validate_download_url(url: str) -> None:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("admission cache download redirected to an invalid URL") from error
    hostname = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or not (
            hostname == "github.com"
            or hostname.endswith(".githubusercontent.com")
        )
    ):
        raise ValueError(
            "admission cache download must remain on credential-free GitHub HTTPS"
        )


class _HttpsGithubRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> urllib.request.Request | None:
        _validate_download_url(new_url)
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


def download_release_asset(
    scope_id: str,
    field: str,
    url: str,
    expected_digest: str,
    destination: Path,
    maximum_bytes: int,
) -> int:
    opener = urllib.request.build_opener(_HttpsGithubRedirectHandler())
    request = urllib.request.Request(url, headers={"User-Agent": "cfetch-admission-replay/1"})
    digest = hashlib.sha256()
    size = 0
    with opener.open(request, timeout=60) as response, destination.open("xb") as output:
        _validate_download_url(response.geturl())
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as error:
                raise ValueError(
                    f"admitted backend {scope_id} {field} has an invalid Content-Length"
                ) from error
            if declared_size < 1 or declared_size > maximum_bytes:
                raise ValueError(
                    f"admitted backend {scope_id} {field} exceeds the bounded download"
                )
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > maximum_bytes:
                raise ValueError(
                    f"admitted backend {scope_id} {field} exceeds the bounded download"
                )
            digest.update(chunk)
            output.write(chunk)
    if size < 1 or digest.hexdigest() != expected_digest:
        raise ValueError(
            f"admitted backend {scope_id} {field} bytes do not match its sha256"
        )
    return size


def download_admission_cache(
    scope_id: str, entry: dict[str, object], destination: Path
) -> int:
    url = entry["admission_cache_url"]
    expected_digest = entry["admission_cache_sha256"]
    if not isinstance(url, str) or not isinstance(expected_digest, str):
        raise ValueError(f"admitted backend {scope_id} cache locator is invalid")
    validate_admission_cache_url(scope_id, url, expected_digest)
    size = download_release_asset(
        scope_id,
        "admission_cache_url",
        url,
        expected_digest,
        destination,
        MAX_ADMISSION_CACHE_BYTES,
    )
    validate_admission_cache_container(destination)
    return size


def expected_measurement_roles(
    evidence_reports: dict[str, dict[str, object]],
) -> dict[str, list[str]]:
    roles: dict[str, set[str]] = {}
    for row in evidence_reports["sequence"]["wire_batch_results"]:
        digest = row["signed_transactions_sha256"]
        roles.setdefault(digest, set()).add("wire-signed-transactions")
    for row in evidence_reports["placement"]["bucket_results"]:
        digest = row["profiler_output_sha256"]
        roles.setdefault(digest, set()).add("placement-profiler")
    for row in evidence_reports["performance"]["bucket_results"]:
        digest = row["benchmark_output_sha256"]
        roles.setdefault(digest, set()).add("performance-benchmark")
    return {digest: sorted(values) for digest, values in roles.items()}


def validate_measurement_bundle(
    path: Path,
    scope_id: str,
    entry: dict[str, object],
    evidence_reports: dict[str, dict[str, object]],
) -> None:
    """Bind retained raw profiler/benchmark files to the validated JSON summaries."""
    expected_roles = expected_measurement_roles(evidence_reports)
    size = path.stat().st_size
    if size < 1 or size > MAX_MEASUREMENT_BUNDLE_BYTES:
        raise ValueError(
            f"{path}: measurement bundle must be 1..{MAX_MEASUREMENT_BUNDLE_BYTES} bytes"
        )
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if (
                not 2 <= len(members) <= MAX_MEASUREMENT_BUNDLE_MEMBERS
                or len(names) != len(set(names))
                or "measurement-manifest.json" not in names
            ):
                raise ValueError(
                    f"{path}: measurement bundle has an invalid member set"
                )
            if any(
                member.is_dir()
                or member.flag_bits & 0x1
                or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or member.file_size < 1
                or member.filename.startswith("/")
                or "\\" in member.filename
                or ".." in Path(member.filename).parts
                for member in members
            ):
                raise ValueError(
                    f"{path}: measurement bundle contains an unsafe ZIP member"
                )
            if sum(member.file_size for member in members) > MAX_MEASUREMENT_BUNDLE_EXPANDED_BYTES:
                raise ValueError(f"{path}: measurement bundle expands beyond its limit")

            manifest_info = archive.getinfo("measurement-manifest.json")
            if manifest_info.file_size > MAX_CACHE_METADATA_BYTES:
                raise ValueError(f"{path}: measurement manifest exceeds its limit")
            manifest = parse_evidence_json(
                archive.read(manifest_info), f"{path}: measurement manifest"
            )
            if set(manifest) != {
                "schema_version",
                "scope_id",
                "sequence_capability_evidence_sha256",
                "placement_evidence_sha256",
                "performance_evidence_sha256",
                "files",
            }:
                raise ValueError(f"{path}: measurement manifest has an invalid schema")
            expected_header = {
                "schema_version": 1,
                "scope_id": scope_id,
                "sequence_capability_evidence_sha256": entry[
                    "sequence_capability_evidence_sha256"
                ],
                "placement_evidence_sha256": entry["placement_evidence_sha256"],
                "performance_evidence_sha256": entry["performance_evidence_sha256"],
            }
            for field, expected in expected_header.items():
                if manifest.get(field) != expected:
                    raise ValueError(
                        f"{path}: measurement manifest {field} does not match the admitted scope"
                    )
            files = manifest["files"]
            if not isinstance(files, list) or any(not isinstance(item, dict) for item in files):
                raise ValueError(f"{path}: measurement manifest files must be objects")
            manifest_roles: dict[str, list[str]] = {}
            manifest_paths: set[str] = set()
            for item in files:
                if set(item) != {"path", "sha256", "roles"}:
                    raise ValueError(f"{path}: measurement file entry has an invalid schema")
                digest = item["sha256"]
                roles = item["roles"]
                member_path = item["path"]
                if (
                    not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    or member_path != f"raw/{digest}.bin"
                    or not isinstance(roles, list)
                    or roles != sorted(set(roles))
                    or any(
                        role
                        not in {
                            "wire-signed-transactions",
                            "placement-profiler",
                            "performance-benchmark",
                        }
                        for role in roles
                    )
                    or not roles
                    or digest in manifest_roles
                ):
                    raise ValueError(f"{path}: measurement file entry is not canonical")
                manifest_roles[digest] = roles
                manifest_paths.add(member_path)
            if manifest_roles != expected_roles or set(names) != {
                "measurement-manifest.json",
                *manifest_paths,
            }:
                raise ValueError(
                    f"{path}: measurement bundle does not retain every referenced raw output"
                )
            for digest in manifest_roles:
                actual = hashlib.sha256()
                with archive.open(f"raw/{digest}.bin") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        actual.update(chunk)
                if actual.hexdigest() != digest:
                    raise ValueError(
                        f"{path}: retained measurement output {digest} has the wrong bytes"
                    )
    except zipfile.BadZipFile as error:
        raise ValueError(f"{path}: measurement evidence is not a valid ZIP bundle") from error


def download_measurement_evidence(
    scope_id: str,
    entry: dict[str, object],
    destination: Path,
    evidence_reports: dict[str, dict[str, object]],
) -> int:
    url = entry["measurement_evidence_url"]
    expected_digest = entry["measurement_evidence_sha256"]
    if not isinstance(url, str) or not isinstance(expected_digest, str):
        raise ValueError(
            f"admitted backend {scope_id} measurement evidence locator is invalid"
        )
    validate_measurement_evidence_url(scope_id, url, expected_digest)
    size = download_release_asset(
        scope_id,
        "measurement_evidence_url",
        url,
        expected_digest,
        destination,
        MAX_MEASUREMENT_BUNDLE_BYTES,
    )
    validate_measurement_bundle(destination, scope_id, entry, evidence_reports)
    return size


def verify_release_registry() -> dict[str, object]:
    """Replay every admitted scope from durable cache bytes; empty is a clean no-op."""
    scope_entries = admitted_scopes()
    if not scope_entries:
        return {
            "admitted_scopes": 0,
            "measurement_bundles_verified": 0,
            "reports_replayed": 0,
            "status": "empty-no-op",
        }

    with tempfile.TemporaryDirectory(prefix="cfetch-admission-replay-") as directory:
        temporary = Path(directory)
        paths: dict[str, Path] = {}
        cohort_bytes = 0
        for scope_id, entry in sorted(scope_entries.items()):
            destination = temporary / f"{scope_id}.npz"
            cohort_bytes += download_admission_cache(scope_id, entry, destination)
            if cohort_bytes > MAX_ADMISSION_COHORT_BYTES:
                raise ValueError("release admission cohort exceeds the bounded cache budget")
            paths[scope_id] = destination

        loaded = {scope_id: load_cache(path) for scope_id, path in paths.items()}
        sequence_probes = {
            scope_id: load_sequence_probe_cache(path)
            for scope_id, path in paths.items()
        }
        for scope_id, (metadata, _, _) in loaded.items():
            if metadata["scope_id"] != scope_id:
                raise ValueError(
                    f"registry scope {scope_id!r} does not match cache scope "
                    f"{metadata['scope_id']!r}"
                )
        validate_loaded_scope_bindings(loaded, scope_entries)
        for scope_id, entry in sorted(scope_entries.items()):
            evidence_reports = load_embedded_evidence_reports(paths[scope_id])
            measurement_path = temporary / f"{scope_id}-measurements.zip"
            cohort_bytes += download_measurement_evidence(
                scope_id,
                entry,
                measurement_path,
                evidence_reports,
            )
            if cohort_bytes > MAX_ADMISSION_COHORT_BYTES:
                raise ValueError(
                    "release admission cohort exceeds the bounded evidence budget"
                )

        report_binding = validate_single_cohort_report_binding(scope_entries)
        if report_binding is None:
            raise ValueError("nonempty release registry has no compatibility report")
        lineage = load_validated_report_lineage(scope_entries, *report_binding)
        for report_path, stored_report, already_admitted, candidates in lineage:
            report_scopes = already_admitted.union(candidates)
            report_paths = {
                scope_id: paths[scope_id] for scope_id in report_scopes
            }
            report_loaded = {
                scope_id: loaded[scope_id] for scope_id in report_scopes
            }
            report_sequence_probes = {
                scope_id: sequence_probes[scope_id] for scope_id in report_scopes
            }
            replayed_report = build_compatibility_report(
                report_paths,
                report_loaded,
                report_sequence_probes,
                already_admitted,
                candidates,
                stored_report["parent_report"],
                stored_report["parent_report_sha256"],
            )
            validate_replayed_report(stored_report, replayed_report, report_path)
    return {
        "admitted_scopes": len(scope_entries),
        "measurement_bundles_verified": len(scope_entries),
        "reports_replayed": len(lineage),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every ordered query-backend/document-backend pairing "
            "against one fixed absolute quality floor"
        )
    )
    parser.add_argument(
        "--verify-release-registry",
        action="store_true",
        help=(
            "replay every admitted scope from its content-addressed release cache; "
            "succeeds without network access when the registry is empty"
        ),
    )
    parser.add_argument(
        "--verify-implementation-bundle",
        action="store_true",
        help="recompute the exact-byte admission implementation bundle digest",
    )
    parser.add_argument(
        "--backend",
        action="append",
        type=parse_backend,
        default=[],
        metavar="LABEL=PATH",
        help="repeat for every NPU, GPU, and CPU cache",
    )
    parser.add_argument(
        "--candidate-scope",
        action="append",
        default=[],
        type=scope_id_value,
        metavar="SCOPE_ID",
        help="repeat for every new scope being evaluated for admission",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.verify_release_registry or args.verify_implementation_bundle:
        if args.backend or args.candidate_scope or args.output is not None:
            parser.error(
                "verification modes cannot be combined with backend evaluation arguments"
            )
        if args.verify_release_registry and args.verify_implementation_bundle:
            parser.error("choose exactly one verification mode")
        try:
            result = (
                verify_release_registry()
                if args.verify_release_registry
                else verify_implementation_bundle()
            )
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from error
        print(json.dumps(result, sort_keys=True))
        return

    if not args.backend or args.output is None:
        parser.error("backend evaluation requires --backend and --output")

    paths = dict(args.backend)
    if len(paths) != len(args.backend):
        raise SystemExit("backend labels must be unique")

    loaded = {label: load_cache(path) for label, path in paths.items()}
    sequence_probes = {
        label: load_sequence_probe_cache(path) for label, path in paths.items()
    }
    for label, (metadata, _, _) in loaded.items():
        if metadata["scope_id"] != label:
            raise SystemExit(
                f"backend label {label!r} must equal cache scope_id {metadata['scope_id']!r}"
            )
    admitted_scope_entries = admitted_scopes()
    admitted_scope_ids = set(admitted_scope_entries)
    candidate_scopes = set(args.candidate_scope)
    if len(candidate_scopes) != len(args.candidate_scope):
        raise SystemExit("candidate scope ids must be unique")
    if admitted_scope_ids.intersection(candidate_scopes):
        raise SystemExit("candidate scopes must not already be admitted")
    if not candidate_scopes:
        raise SystemExit("name at least one --candidate-scope for every new cohort")
    expected_scopes = admitted_scope_ids.union(candidate_scopes)
    if not expected_scopes:
        raise SystemExit("name at least one --candidate-scope when no backend is admitted")
    try:
        validate_loaded_scope_bindings(loaded, admitted_scope_entries)
        parent_binding = validate_single_cohort_report_binding(
            admitted_scope_entries
        )
        parent_report, parent_report_sha256 = (
            parent_binding if parent_binding is not None else (None, None)
        )
        report = build_compatibility_report(
            paths,
            loaded,
            sequence_probes,
            admitted_scope_ids,
            candidate_scopes,
            parent_report,
            parent_report_sha256,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    serialized_report = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if len(serialized_report.encode("utf-8")) > MAX_ADMISSION_REPORT_BYTES:
        raise SystemExit("compatibility report exceeds its bounded byte limit")
    try:
        write_new_report(args.output, serialized_report)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(serialized_report, end="")
    if report["admission_gate"]["passed"] is not True:
        raise SystemExit("global vector-space admission gate failed")


if __name__ == "__main__":
    main()
