#!/usr/bin/env python3
"""Export one local embedding adapter's canonical SciFact evidence cache."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Sequence
from contextlib import ExitStack
import hashlib
import ipaddress
import json
import os
import re
import secrets
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from admission_evidence import (
    DIMENSIONS,
    DOCUMENT_PREFIX,
    MAX_TOKENS,
    QUERY_PREFIX,
    SEQUENCE_BUCKETS,
    SEQUENCE_SEMANTIC_FIXTURE_ID,
    SEQUENCE_SEMANTIC_FIXTURE_SHA256,
    SUPERVISED_LOCAL_TRANSPORT,
    SUPPORTED_MAX_BATCH_SIZE,
    TRANSPORTS,
    WIRE_BATCH_INPUT_SELECTION,
    bucket_records,
    ordered_input_json_sha256,
    parse_evidence_json,
    selected_sequence_bucket,
    sequence_semantic_fixture_sha256,
    sequence_semantic_probe_inputs,
    utf8_sha256,
    validate_evidence_reports,
    validate_wire_batch_evidence,
    wire_batch_inputs,
)
from scifact_contract import DATASET, DATASET_REVISION, load_scifact_contract
from profile_identity import ADMISSION_POLICY_SHA256

if TYPE_CHECKING:
    import numpy as np


PROFILE_ID = "cfetch-embedding-v1"
PROFILE_MANIFEST_SHA256 = (
    "59210a333494f788eb8e607fe38cabb6af1a7aa7cdf604ddf52e3fa6004b5afb"
)
MODEL = "google/embeddinggemma-300m"
MODEL_REVISION = "57c266a740f537b4dc058e1b0cda161fd15afa75"
VECTOR_ENCODING = "signed-int8x768"
DEVICE_CLASSES = ("npu", "gpu", "cpu")
SCOPE_ID_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
MAX_EVIDENCE_BYTES = 8 * 1024 * 1024
# One 64x768 JSON embedding response is comfortably below this. Matching the
# embedded-evidence and Rust wire caps keeps a single adapter reply from
# becoming a larger allocation boundary than the evidence it is allowed to
# produce.
MAX_ADAPTER_RESPONSE_BYTES = 8 * 1024 * 1024
ATTESTATION_NONCE_HEADER = "X-Cfetch-Attestation-Nonce"
ATTESTATION_SIGNATURE_HEADER = "X-Cfetch-Attestation-Signature"
ATTESTATION_DOMAIN = b"cfetch-embedding-response-attestation-v1\0"
SEQUENCE_PROBE_ARRAY_NAMES = (
    "sequence_probe_queries",
    "sequence_probe_relevant_documents",
    "sequence_probe_irrelevant_documents",
    "sequence_probe_queries_repeat",
    "sequence_probe_relevant_documents_repeat",
    "sequence_probe_irrelevant_documents_repeat",
)
EVIDENCE_CACHE_LOCATORS = {
    "sequence_capability": "npz:sequence_capability_evidence_bytes",
    "placement": "npz:placement_evidence_bytes",
    "performance": "npz:performance_evidence_bytes",
}


class ExportCheckpoint:
    """An exclusively owned working directory of verified signed transactions.

    Trial names distinguish independent repeatability and grouping executions.
    Retained replies are reauthenticated and revalidated by request_embeddings;
    these working files never replace the final admission cache or its gates.
    """

    def __init__(self, directory: Path, identity: dict[str, object]) -> None:
        self.directory = directory
        self.identity = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self._lock = None

    def __enter__(self) -> ExportCheckpoint:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise ValueError("checkpoint directory must be a real directory")
        lock_path = self.directory / ".writer.lock"
        if lock_path.is_symlink():
            raise ValueError("checkpoint lock must not be a symlink")
        descriptor = os.open(
            lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        self._lock = os.fdopen(descriptor, "r+b")
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("checkpoint lock must be a regular file")
            if os.name == "nt":
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    self._lock.write(b"\0")
                    self._lock.flush()
                self._lock.seek(0)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as error:
            self._lock.close()
            self._lock = None
            raise ValueError("checkpoint directory already has a writer or cannot be locked") from error
        try:
            identity_path = self.directory / "identity.json"
            if identity_path.exists() or identity_path.is_symlink():
                if self._read(identity_path, MAX_EVIDENCE_BYTES) != self.identity:
                    raise ValueError("checkpoint identity does not match this export")
            else:
                # Never adopt pre-existing transactions without their identity.
                if any(path.name != ".writer.lock" for path in self.directory.iterdir()):
                    raise ValueError("checkpoint directory contains files without an identity")
                self._publish(identity_path, self.identity)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_args: object) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def _require_lock(self) -> None:
        if self._lock is None:
            raise ValueError("checkpoint directory is not locked")

    @staticmethod
    def _read(path: Path, maximum: int) -> bytes:
        if path.is_symlink():
            raise ValueError("checkpoint files must not be symlinks")
        # Reject special files after opening without letting a FIFO block first.
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
                raise ValueError("checkpoint file is not regular or exceeds its size bound")
            data = handle.read(maximum + 1)
        if len(data) != info.st_size:
            raise ValueError("checkpoint file changed while being read")
        return data

    def _publish(self, destination: Path, raw: bytes) -> None:
        self._require_lock()
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".checkpoint-", suffix=".tmp",
                dir=self.directory, delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(raw)
                temporary.flush()
                os.fsync(temporary.fileno())
            # Linking publishes atomically and refuses any existing destination.
            os.link(temporary_path, destination)
            if os.name != "nt":
                descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _transaction_path(self, key: str) -> Path:
        self._require_lock()
        if len(key) > 128 or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", key) is None:
            raise ValueError("invalid checkpoint transaction key")
        return self.directory / f"{key}.json"

    def read_transaction(
        self, key: str, request_body: bytes
    ) -> tuple[bytes, bytes, str] | None:
        path = self._transaction_path(key)
        if not path.exists() and not path.is_symlink():
            return None
        document = parse_evidence_json(
            self._read(path, 2 * MAX_ADAPTER_RESPONSE_BYTES), "checkpoint transaction"
        )
        if set(document) != {
            "schema_version", "request_body_sha256", "nonce", "signature", "response_base64"
        } or type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError("invalid checkpoint transaction schema")
        if document["request_body_sha256"] != hashlib.sha256(request_body).hexdigest():
            raise ValueError("checkpoint transaction input does not match")
        nonce = document["nonce"]
        signature = document["signature"]
        encoded = document["response_base64"]
        if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
            raise ValueError("invalid checkpoint attestation nonce")
        if not isinstance(signature, str) or re.fullmatch(r"[0-9a-f]{128}", signature) is None:
            raise ValueError("invalid checkpoint attestation signature")
        if not isinstance(encoded, str):
            raise ValueError("invalid checkpoint response encoding")
        try:
            response = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ValueError("invalid checkpoint response encoding") from error
        if not 0 < len(response) <= MAX_ADAPTER_RESPONSE_BYTES:
            raise ValueError("checkpoint response exceeds its size bound")
        return bytes.fromhex(nonce), response, signature

    def write_transaction(
        self, key: str, request_body: bytes, nonce: bytes, response: bytes, signature: str
    ) -> None:
        document = {
            "schema_version": 1,
            "request_body_sha256": hashlib.sha256(request_body).hexdigest(),
            "nonce": nonce.hex(),
            "signature": signature,
            "response_base64": base64.b64encode(response).decode("ascii"),
        }
        self._publish(
            self._transaction_path(key),
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not parsed > 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def supported_batch_size(value: str) -> int:
    parsed = positive_integer(value)
    if parsed > SUPPORTED_MAX_BATCH_SIZE:
        raise argparse.ArgumentTypeError(
            f"value must not exceed {SUPPORTED_MAX_BATCH_SIZE}"
        )
    return parsed


def sha256_value(value: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise argparse.ArgumentTypeError("value must be exactly 64 hexadecimal characters")
    return value.lower()


def ed25519_public_key_value(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError(
            "Ed25519 public key must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def nonempty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must not be empty")
    return value


def scope_id_value(value: str) -> str:
    if len(value) > 128 or SCOPE_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "scope id must be at most 128 lowercase slug characters "
            "(letters, digits, and single '.', '_', or '-' separators)"
        )
    return value


def evidence_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"evidence file does not exist: {path}")
    return path


def read_verified_evidence(path: Path, expected_sha256: str) -> bytes:
    with path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if size < 1:
            raise ValueError(f"evidence file is empty: {path}")
        if size > MAX_EVIDENCE_BYTES:
            raise ValueError(
                f"evidence file exceeds the {MAX_EVIDENCE_BYTES}-byte limit: {path}"
            )
        data = handle.read(MAX_EVIDENCE_BYTES + 1)
    if len(data) != size:
        raise ValueError(f"evidence file changed while it was being read: {path}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"evidence file {path} has sha256 {actual}, expected {expected_sha256}"
        )
    return data


def evidence_json(path: Path, data: bytes) -> dict[str, object]:
    return parse_evidence_json(data, f"evidence file {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local attested /embeddings adapter twice and export its "
            "canonical signed INT8x768 SciFact cache"
        )
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="loopback URL whose path ends in /embeddings",
    )
    parser.add_argument("--output", required=True, type=Path, help="new .npz cache path")
    parser.add_argument(
        "--checkpoint-directory", type=Path,
        help="opt-in exclusive working directory; resume only the identical export",
    )
    parser.add_argument("--scope-id", required=True, type=scope_id_value)
    parser.add_argument("--transport", required=True, choices=TRANSPORTS)
    parser.add_argument("--backend", required=True, type=nonempty)
    parser.add_argument("--runtime", required=True, type=nonempty)
    parser.add_argument("--compiler", required=True, type=nonempty)
    parser.add_argument("--package-target", required=True, type=nonempty)
    parser.add_argument("--artifact-source", required=True, type=nonempty)
    parser.add_argument("--artifact-sha256", required=True, type=sha256_value)
    parser.add_argument(
        "--attestation-public-key",
        required=True,
        type=ed25519_public_key_value,
        help=(
            "proposed package Ed25519 public key as 64 lowercase hexadecimal "
            "characters; every exporter response must verify under this key"
        ),
    )
    parser.add_argument("--internal-precision", required=True, type=nonempty)
    parser.add_argument("--supported-max-tokens", required=True, type=positive_integer)
    parser.add_argument(
        "--supported-sequence-bucket",
        action="append",
        required=True,
        type=positive_integer,
        metavar="TOKENS",
        help="repeat for every sequence bucket supported by this exact package",
    )
    parser.add_argument(
        "--sequence-capability-evidence", required=True, type=evidence_file
    )
    parser.add_argument(
        "--sequence-capability-evidence-sha256", required=True, type=sha256_value
    )
    parser.add_argument("--device", required=True, type=nonempty)
    parser.add_argument("--device-class", required=True, choices=DEVICE_CLASSES)
    parser.add_argument("--placement-evidence", required=True, type=evidence_file)
    parser.add_argument("--placement-evidence-sha256", required=True, type=sha256_value)
    parser.add_argument("--performance-evidence", required=True, type=evidence_file)
    parser.add_argument("--performance-evidence-sha256", required=True, type=sha256_value)
    parser.add_argument(
        "--accelerated-placement",
        action="store_true",
        required=True,
        help="explicitly attest that the recorded placement is accelerated",
    )
    parser.add_argument("--batch-size", type=supported_batch_size, default=32)
    parser.add_argument("--timeout-seconds", type=positive_float, default=120.0)
    parser.add_argument(
        "--scifact-snapshot",
        type=Path,
        help=(
            "explicit directory containing the exact pinned raw SciFact "
            "corpus.jsonl, queries.jsonl, and qrels/test.jsonl bytes; when absent, "
            "load the pinned revision through datasets"
        ),
    )
    parser.add_argument(
        "--bearer-token-env",
        metavar="NAME",
        help="read an optional loopback adapter bearer token from this environment variable",
    )
    return parser


def validate_loopback_endpoint(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("endpoint scheme must be http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint must not contain a query string or fragment")
    try:
        host = parsed.hostname
        parsed.port
    except ValueError as error:
        raise ValueError(f"endpoint has an invalid port: {error}") from error
    if host is None:
        raise ValueError("endpoint must contain a host")
    if host.lower() != "localhost":
        try:
            address = ipaddress.ip_address(host)
        except ValueError as error:
            raise ValueError("endpoint host must be localhost or a loopback address") from error
        if not address.is_loopback:
            raise ValueError("endpoint host must be localhost or a loopback address")
    if not parsed.path.rstrip("/").endswith("/embeddings"):
        raise ValueError("endpoint path must end in /embeddings")
    return value


def canonical_i8(vectors: np.ndarray) -> np.ndarray:
    """Apply cfetch's f32 max-absolute, round-ties-even signed INT8 codec."""
    import numpy as np

    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != DIMENSIONS:
        raise ValueError(
            f"embedding output must have shape (items, {DIMENSIONS}), found {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("embedding output contains a non-finite component")
    maximum = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(maximum <= np.float32(0.0)):
        raise ValueError("embedding output contains an all-zero vector")
    scaled = values / maximum * np.float32(127.0)
    return np.rint(
        np.clip(scaled, np.float32(-127.0), np.float32(127.0))
    ).astype(np.int8)


def validate_response(
    payload: object,
    expected_items: int,
    requested_scope_id: str,
    expected_execution: dict[str, object] | None = None,
    expected_row_metadata: Sequence[dict[str, object]] | None = None,
    expected_compatibility_report_sha256: str | None = None,
) -> np.ndarray:
    import numpy as np

    try:
        requested_scope_id = scope_id_value(requested_scope_id)
    except argparse.ArgumentTypeError as error:
        raise ValueError(str(error)) from error
    if not isinstance(payload, dict):
        raise ValueError("embeddings response must be a JSON object")
    attestation = {
        "model": MODEL,
        "cfetch_profile": PROFILE_ID,
        "cfetch_profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
        "cfetch_admission_policy_sha256": ADMISSION_POLICY_SHA256,
        "cfetch_model_revision": MODEL_REVISION,
    }
    for field, expected in attestation.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"embeddings response {field}={payload.get(field)!r}, expected {expected!r}"
            )
    execution = payload.get("cfetch_execution")
    if not isinstance(execution, dict):
        raise ValueError("embeddings response must contain a cfetch_execution object")
    for field in (
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
        "sequence_capability_evidence_sha256",
        "performance_evidence_sha256",
    ):
        if not isinstance(execution.get(field), str) or not execution[field]:
            raise ValueError(f"cfetch_execution {field} must be a non-empty string")
    if (
        len(execution["scope_id"]) > 128
        or SCOPE_ID_PATTERN.fullmatch(execution["scope_id"]) is None
    ):
        raise ValueError(
            "cfetch_execution scope_id must be at most 128 lowercase slug characters"
        )
    if execution["device_class"] not in DEVICE_CLASSES:
        raise ValueError("cfetch_execution device_class must be npu, gpu, or cpu")
    if execution["transport"] not in TRANSPORTS:
        raise ValueError(
            "cfetch_execution transport must be supervised-local or remote-attested"
        )
    if execution["scope_id"] != requested_scope_id:
        raise ValueError(
            "cfetch_execution scope_id="
            f"{execution['scope_id']!r}, requested {requested_scope_id!r}"
        )
    if re.fullmatch(r"[0-9a-f]{64}", execution["artifact_sha256"]) is None:
        raise ValueError(
            "cfetch_execution artifact_sha256 must be 64 lowercase hexadecimal characters"
        )
    for field in (
        "placement_evidence_sha256",
        "sequence_capability_evidence_sha256",
        "performance_evidence_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", execution[field]) is None:
            raise ValueError(
                f"cfetch_execution {field} must be 64 lowercase hexadecimal characters"
            )
    if expected_compatibility_report_sha256 is not None:
        try:
            expected_compatibility_report_sha256 = sha256_value(
                expected_compatibility_report_sha256
            )
        except argparse.ArgumentTypeError as error:
            raise ValueError(str(error)) from error
        if (
            execution.get("compatibility_report_sha256")
            != expected_compatibility_report_sha256
        ):
            raise ValueError(
                "cfetch_execution compatibility_report_sha256="
                f"{execution.get('compatibility_report_sha256')!r}, expected "
                f"{expected_compatibility_report_sha256!r}"
            )
    supported_max_tokens = execution.get("supported_max_tokens")
    if type(supported_max_tokens) is not int or supported_max_tokens < 1:
        raise ValueError(
            "cfetch_execution supported_max_tokens must be a positive integer"
        )
    if execution.get("supported_max_batch_size") != SUPPORTED_MAX_BATCH_SIZE:
        raise ValueError(
            "cfetch_execution supported_max_batch_size must be "
            f"{SUPPORTED_MAX_BATCH_SIZE}"
        )
    supported_buckets = execution.get("supported_sequence_buckets")
    if (
        not isinstance(supported_buckets, list)
        or not supported_buckets
        or any(type(value) is not int or value < 1 for value in supported_buckets)
        or supported_buckets != sorted(set(supported_buckets))
        or supported_buckets[-1] > supported_max_tokens
    ):
        raise ValueError(
            "cfetch_execution supported_sequence_buckets must be unique, sorted, "
            "positive, and no larger than supported_max_tokens"
        )
    if execution.get("accelerated_placement") is not True:
        raise ValueError("cfetch_execution accelerated_placement must be true")
    if expected_execution is not None:
        for field, expected in expected_execution.items():
            if field not in execution or execution[field] != expected:
                raise ValueError(
                    f"cfetch_execution {field}={execution.get(field)!r}, expected {expected!r}"
                )
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("embeddings response data must be an array")
    if len(rows) != expected_items:
        raise ValueError(
            f"embeddings response returned {len(rows)} rows for {expected_items} inputs"
        )
    if (
        expected_row_metadata is not None
        and len(expected_row_metadata) != expected_items
    ):
        raise ValueError("expected row metadata count does not match requested inputs")

    ordered: list[np.ndarray | None] = [None] * expected_items
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("embeddings response row must be a JSON object")
        index = row.get("index")
        if type(index) is not int:
            raise ValueError("embeddings response row index must be an integer")
        if index < 0 or index >= expected_items:
            raise ValueError(f"embeddings response row index {index} is out of range")
        if ordered[index] is not None:
            raise ValueError(f"embeddings response contains duplicate index {index}")
        if row.get("cfetch_scope_id") != execution["scope_id"]:
            raise ValueError(
                f"embeddings response index {index} cfetch_scope_id must match "
                "cfetch_execution scope_id"
            )
        token_count = row.get("token_count")
        if (
            type(token_count) is not int
            or token_count < 1
            or token_count > supported_max_tokens
        ):
            raise ValueError(
                f"embeddings response index {index} token_count must be in "
                f"1..{supported_max_tokens}"
            )
        selected_buckets = [
            bucket for bucket in supported_buckets if bucket >= token_count
        ]
        if not selected_buckets:
            raise ValueError(
                f"embeddings response index {index} token_count {token_count} "
                "does not fit a supported sequence bucket"
            )
        expected_bucket = selected_buckets[0]
        if row.get("sequence_bucket") != expected_bucket:
            raise ValueError(
                f"embeddings response index {index} sequence_bucket="
                f"{row.get('sequence_bucket')!r}, expected smallest supported bucket "
                f"{expected_bucket} for token_count {token_count}"
            )
        if row.get("truncated") is not False:
            raise ValueError(
                f"embeddings response index {index} must attest truncated=false"
            )
        if expected_row_metadata is not None:
            expected_metadata = expected_row_metadata[index]
            for field, actual in (
                ("token_count", token_count),
                ("sequence_bucket", row.get("sequence_bucket")),
            ):
                expected = expected_metadata.get(field)
                if actual != expected:
                    raise ValueError(
                        f"embeddings response index {index} {field}={actual!r}, "
                        f"sequence evidence requires {expected!r}"
                    )
        embedding = row.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != DIMENSIONS:
            found = len(embedding) if isinstance(embedding, list) else "non-array"
            raise ValueError(
                f"embeddings response index {index} has {found} components; expected {DIMENSIONS}"
            )
        if any(type(component) not in {int, float} for component in embedding):
            raise ValueError(
                f"embeddings response index {index} contains a non-numeric component"
            )
        try:
            vector = np.asarray(embedding, dtype=np.float32)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                f"embeddings response index {index} is not representable as float32"
            ) from error
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"embeddings response index {index} contains a non-finite component"
            )
        if not np.any(vector):
            raise ValueError(f"embeddings response index {index} is all zero")
        ordered[index] = vector

    if any(vector is None for vector in ordered):
        missing = [index for index, vector in enumerate(ordered) if vector is None]
        raise ValueError(f"embeddings response omitted indices {missing}")
    return np.stack(ordered)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        del request, file_pointer, code, message, headers, new_url
        return None


def local_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


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


def verify_response_signature(
    public_key_hex: str,
    signature_hex: object,
    nonce: bytes,
    request_body: bytes,
    response_body: bytes,
) -> None:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        public_key_hex = ed25519_public_key_value(public_key_hex)
    except argparse.ArgumentTypeError as error:
        raise ValueError(str(error)) from error
    if not isinstance(signature_hex, str) or re.fullmatch(
        r"[0-9a-f]{128}", signature_hex
    ) is None:
        raise ValueError(
            f"{ATTESTATION_SIGNATURE_HEADER} must be exactly 128 lowercase "
            "hexadecimal characters"
        )
    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        public_key.verify(
            bytes.fromhex(signature_hex),
            attestation_message(nonce, request_body, response_body),
        )
    except InvalidSignature as error:
        raise ValueError(
            "embedding response failed its proposed scope-key signature"
        ) from error
    except ValueError as error:
        raise ValueError("proposed Ed25519 public key is invalid") from error


def read_bounded_adapter_response(
    response, limit: int = MAX_ADAPTER_RESPONSE_BYTES
) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "embeddings endpoint returned an invalid Content-Length"
            ) from error
        if declared < 0:
            raise ValueError("embeddings endpoint returned a negative Content-Length")
        if declared > limit:
            raise ValueError(
                "embeddings endpoint Content-Length "
                f"{declared} exceeds the {limit}-byte response limit"
            )
    response_body = response.read(limit + 1)
    if len(response_body) > limit:
        raise ValueError(
            f"embeddings endpoint response exceeds the {limit}-byte limit"
        )
    return response_body


def request_embeddings(
    endpoint: str,
    requested_scope_id: str,
    texts: Sequence[str],
    timeout_seconds: float,
    bearer_token: str | None = None,
    opener: urllib.request.OpenerDirector | None = None,
    expected_execution: dict[str, object] | None = None,
    attestation_public_key: str | None = None,
    expected_row_metadata: Sequence[dict[str, object]] | None = None,
    expected_compatibility_report_sha256: str | None = None,
    used_attestation_nonces: set[bytes] | None = None,
    checkpoint: ExportCheckpoint | None = None,
    checkpoint_key: str | None = None,
) -> np.ndarray:
    validate_loopback_endpoint(endpoint)
    try:
        requested_scope_id = scope_id_value(requested_scope_id)
    except argparse.ArgumentTypeError as error:
        raise ValueError(str(error)) from error
    body = json.dumps(
        {
            "model": MODEL,
            "dimensions": DIMENSIONS,
            "input": list(texts),
            "cfetch_requested_scope_id": requested_scope_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    retained = None
    if checkpoint is not None:
        if checkpoint_key is None or attestation_public_key is None or used_attestation_nonces is None:
            raise ValueError("checkpoint requests require a trial key, scope key and nonce registry")
        retained = checkpoint.read_transaction(checkpoint_key, body)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    attestation_nonce: bytes | None = None
    if attestation_public_key is not None:
        try:
            ed25519_public_key_value(attestation_public_key)
        except argparse.ArgumentTypeError as error:
            raise ValueError(str(error)) from error
        attestation_nonce = retained[0] if retained is not None else secrets.token_bytes(32)
        if len(attestation_nonce) != 32:
            raise ValueError("attestation nonce source did not return exactly 32 bytes")
        if used_attestation_nonces is not None:
            if attestation_nonce in used_attestation_nonces:
                raise ValueError("attestation nonce source repeated a prior challenge")
            used_attestation_nonces.add(attestation_nonce)
        headers[ATTESTATION_NONCE_HEADER] = attestation_nonce.hex()
    if bearer_token is not None:
        headers["Authorization"] = f"Bearer {bearer_token}"
    if retained is not None:
        _, response_body, response_signature = retained
    else:
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        active_opener = opener if opener is not None else local_opener()
        try:
            with active_opener.open(request, timeout=timeout_seconds) as response:
                response_body = read_bounded_adapter_response(response)
                response_signature = (
                    response.headers.get(ATTESTATION_SIGNATURE_HEADER)
                    if attestation_public_key is not None
                    else None
                )
        except urllib.error.HTTPError as error:
            detail = error.read(512).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"embeddings endpoint returned HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"embeddings endpoint request failed: {error.reason}") from error
    if attestation_public_key is not None:
        verify_response_signature(
            attestation_public_key,
            response_signature,
            attestation_nonce,
            body,
            response_body,
        )
    payload = parse_evidence_json(
        response_body, "embeddings endpoint response"
    )
    vectors = validate_response(
        payload,
        len(texts),
        requested_scope_id,
        expected_execution,
        expected_row_metadata,
        expected_compatibility_report_sha256,
    )
    if checkpoint is not None and retained is None:
        assert checkpoint_key is not None and attestation_nonce is not None
        assert response_signature is not None
        checkpoint.write_transaction(
            checkpoint_key, body, attestation_nonce, response_body, response_signature
        )
    return vectors


RequestFunction = Callable[[str, str, Sequence[str], float, str | None], object]
TrialRequestFactory = Callable[[str], RequestFunction]
SequenceProbeRequestFunction = Callable[
    [Sequence[str], Sequence[dict[str, object]]], object
]
SequenceProbeRequestFactory = Callable[[str], SequenceProbeRequestFunction]


def embed_canonical(
    endpoint: str,
    requested_scope_id: str,
    texts: Sequence[str],
    batch_size: int,
    timeout_seconds: float,
    bearer_token: str | None,
    request_function: RequestFunction = request_embeddings,
) -> np.ndarray:
    import numpy as np

    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    if batch_size > SUPPORTED_MAX_BATCH_SIZE:
        raise ValueError(
            f"batch size must not exceed {SUPPORTED_MAX_BATCH_SIZE}"
        )
    if timeout_seconds <= 0:
        raise ValueError("timeout must be greater than zero")
    batches: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        floats = request_function(
            endpoint,
            requested_scope_id,
            texts[start : start + batch_size],
            timeout_seconds,
            bearer_token,
        )
        batches.append(canonical_i8(floats))
    if not batches:
        return np.empty((0, DIMENSIONS), dtype=np.int8)
    return np.concatenate(batches, axis=0)


def collect_cache_arrays(
    endpoint: str,
    requested_scope_id: str,
    query_inputs: Sequence[str],
    document_inputs: Sequence[str],
    batch_size: int,
    timeout_seconds: float,
    bearer_token: str | None = None,
    request_function: RequestFunction = request_embeddings,
    trial_request_factory: TrialRequestFactory | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    if not query_inputs or not document_inputs:
        raise ValueError("pinned SciFact inputs must contain queries and documents")
    if any(not text.startswith(QUERY_PREFIX) for text in query_inputs):
        raise ValueError("every query must already contain the pinned query prefix")
    if any(not text.startswith(DOCUMENT_PREFIX) for text in document_inputs):
        raise ValueError("every document must already contain the pinned document prefix")

    queries = embed_canonical(
        endpoint,
        requested_scope_id,
        query_inputs,
        batch_size,
        timeout_seconds,
        bearer_token,
        trial_request_factory("queries-first") if trial_request_factory else request_function,
    )
    documents = embed_canonical(
        endpoint,
        requested_scope_id,
        document_inputs,
        batch_size,
        timeout_seconds,
        bearer_token,
        trial_request_factory("documents-first") if trial_request_factory else request_function,
    )
    queries_repeat = embed_canonical(
        endpoint,
        requested_scope_id,
        query_inputs,
        batch_size,
        timeout_seconds,
        bearer_token,
        trial_request_factory("queries-repeat") if trial_request_factory else request_function,
    )
    documents_repeat = embed_canonical(
        endpoint,
        requested_scope_id,
        document_inputs,
        batch_size,
        timeout_seconds,
        bearer_token,
        trial_request_factory("documents-repeat") if trial_request_factory else request_function,
    )
    if not np.array_equal(queries, queries_repeat):
        raise ValueError("query vectors are not byte-repeatable in this adapter scope")
    if not np.array_equal(documents, documents_repeat):
        raise ValueError("document vectors are not byte-repeatable in this adapter scope")
    return queries, documents, queries_repeat, documents_repeat


def verify_wire_batch_contract(
    endpoint: str,
    requested_scope_id: str,
    texts: Sequence[str],
    timeout_seconds: float,
    bearer_token: str | None,
    sequence_report: dict[str, object],
    request_function: RequestFunction = request_embeddings,
    trial_request_factory: TrialRequestFactory | None = None,
) -> np.ndarray:
    import numpy as np

    validate_wire_batch_evidence(sequence_report)
    if len(texts) != SUPPORTED_MAX_BATCH_SIZE:
        raise ValueError(
            "wire-batch probe must contain exactly "
            f"{SUPPORTED_MAX_BATCH_SIZE} ordered canonical inputs"
        )
    expected_input_digest = ordered_input_json_sha256(texts)
    observed_outputs: list[np.ndarray] = []
    for row in sequence_report["wire_batch_results"]:
        batch_size = row["batch_size"]
        if row["ordered_input_json_sha256"] != expected_input_digest:
            raise ValueError(
                f"sequence evidence batch {batch_size} ordered inputs do not match "
                f"the pinned {WIRE_BATCH_INPUT_SELECTION} probe"
            )
        request_count = 0
        response_row_count = 0
        trial_request = (
            trial_request_factory(f"wire-{batch_size}")
            if trial_request_factory else request_function
        )

        def counted_request(
            request_endpoint: str,
            request_scope_id: str,
            request_texts: Sequence[str],
            request_timeout: float,
            request_token: str | None,
        ) -> object:
            nonlocal request_count, response_row_count
            request_count += 1
            response = trial_request(
                request_endpoint,
                request_scope_id,
                request_texts,
                request_timeout,
                request_token,
            )
            response_row_count += len(response)
            return response

        outputs = embed_canonical(
            endpoint,
            requested_scope_id,
            texts,
            batch_size,
            timeout_seconds,
            bearer_token,
            counted_request,
        )
        if request_count != row["request_count"]:
            raise ValueError(
                f"wire-batch probe size {batch_size} made {request_count} requests; "
                f"sequence evidence records {row['request_count']}"
            )
        if response_row_count != row["response_row_count"]:
            raise ValueError(
                f"wire-batch probe size {batch_size} returned {response_row_count} rows; "
                f"sequence evidence records {row['response_row_count']}"
            )
        output_digest = hashlib.sha256(
            np.ascontiguousarray(outputs).tobytes()
        ).hexdigest()
        if output_digest != row["canonical_output_bytes_sha256"]:
            raise ValueError(
                f"wire-batch probe size {batch_size} canonical output digest "
                "does not match sequence evidence"
            )
        observed_outputs.append(np.ascontiguousarray(outputs))
    if any(
        not np.array_equal(observed_outputs[0], outputs)
        for outputs in observed_outputs[1:]
    ):
        raise ValueError(
            "wire-batch probe sizes 1 through 64 produced different canonical outputs"
        )
    return np.stack(observed_outputs)


def exact_i8_relevant_precedes(
    query: np.ndarray, relevant: np.ndarray, irrelevant: np.ndarray
) -> bool:
    import numpy as np

    query_i64 = np.asarray(query, dtype=np.int64)
    relevant_i64 = np.asarray(relevant, dtype=np.int64)
    irrelevant_i64 = np.asarray(irrelevant, dtype=np.int64)
    relevant_dot = int(query_i64 @ relevant_i64)
    irrelevant_dot = int(query_i64 @ irrelevant_i64)
    relevant_norm = int(relevant_i64 @ relevant_i64)
    irrelevant_norm = int(irrelevant_i64 @ irrelevant_i64)
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


def collect_sequence_probe_arrays(
    sequence_report: dict[str, object],
    request_function: SequenceProbeRequestFunction,
    trial_request_factory: SequenceProbeRequestFactory | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    supported_buckets = sequence_report.get("supported_sequence_buckets")
    if not isinstance(supported_buckets, list):
        raise ValueError("sequence evidence supported_sequence_buckets must be an array")
    records = bucket_records(sequence_report, supported_buckets, "sequence")
    first_by_kind: list[list[np.ndarray]] = [[], [], []]
    repeat_by_kind: list[list[np.ndarray]] = [[], [], []]
    labels = ("query", "relevant_document", "irrelevant_document")
    for row in records:
        bucket = row["bucket"]
        probe = row.get("semantic_probe")
        if not isinstance(probe, dict):
            raise ValueError(
                f"sequence evidence bucket {bucket} omitted semantic_probe"
            )
        inputs = sequence_semantic_probe_inputs(bucket)
        expected_row_metadata = [
            {
                "token_count": probe[f"{label}_token_count"],
                "sequence_bucket": bucket,
            }
            for label in labels
        ]
        first_request = (
            trial_request_factory(f"sequence-{bucket}-first")
            if trial_request_factory else request_function
        )
        repeat_request = (
            trial_request_factory(f"sequence-{bucket}-repeat")
            if trial_request_factory else request_function
        )
        first = canonical_i8(first_request(inputs, expected_row_metadata))
        repeat = canonical_i8(repeat_request(inputs, expected_row_metadata))
        if not np.array_equal(first, repeat):
            raise ValueError(
                f"sequence semantic probe bucket {bucket} is not byte-repeatable"
            )
        for index, label in enumerate(labels):
            output_digest = hashlib.sha256(
                np.ascontiguousarray(first[index]).tobytes()
            ).hexdigest()
            if output_digest != probe[f"{label}_canonical_output_bytes_sha256"]:
                raise ValueError(
                    f"sequence semantic probe bucket {bucket} {label} canonical "
                    "output digest does not match sequence evidence"
                )
            first_by_kind[index].append(first[index])
            repeat_by_kind[index].append(repeat[index])
        if not exact_i8_relevant_precedes(first[0], first[1], first[2]):
            raise ValueError(
                f"sequence semantic probe bucket {bucket} did not rank its pinned "
                "relevant document before the irrelevant document"
            )
    return (
        np.stack(first_by_kind[0]),
        np.stack(first_by_kind[1]),
        np.stack(first_by_kind[2]),
        np.stack(repeat_by_kind[0]),
        np.stack(repeat_by_kind[1]),
        np.stack(repeat_by_kind[2]),
    )


def load_scifact_inputs(
    snapshot_directory: Path | None = None,
) -> tuple[list[str], list[str]]:
    contract = load_scifact_contract(
        QUERY_PREFIX, DOCUMENT_PREFIX, snapshot_directory
    )
    return contract.query_texts, contract.document_texts


def build_cache_metadata(args: argparse.Namespace) -> dict[str, object]:
    """Build public cache metadata without persisting host-private CLI paths."""
    return {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
        "admission_policy_sha256": ADMISSION_POLICY_SHA256,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "vector_encoding": VECTOR_ENCODING,
        "supported_max_tokens": args.supported_max_tokens,
        "supported_sequence_buckets": sorted(set(args.supported_sequence_bucket)),
        "supported_max_batch_size": SUPPORTED_MAX_BATCH_SIZE,
        "sequence_semantic_fixture_id": SEQUENCE_SEMANTIC_FIXTURE_ID,
        "sequence_semantic_fixture_sha256": SEQUENCE_SEMANTIC_FIXTURE_SHA256,
        "sequence_capability_evidence": EVIDENCE_CACHE_LOCATORS[
            "sequence_capability"
        ],
        "sequence_capability_evidence_sha256": (
            args.sequence_capability_evidence_sha256
        ),
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "scope_id": args.scope_id,
        "transport": args.transport,
        "backend": args.backend,
        "runtime": args.runtime,
        "compiler": args.compiler,
        "package_target": args.package_target,
        "artifact_source": args.artifact_source,
        "artifact_sha256": args.artifact_sha256,
        "attestation_public_key": args.attestation_public_key,
        "internal_precision": args.internal_precision,
        "device": args.device,
        "device_class": args.device_class,
        "placement_evidence": EVIDENCE_CACHE_LOCATORS["placement"],
        "placement_evidence_sha256": args.placement_evidence_sha256,
        "performance_evidence": EVIDENCE_CACHE_LOCATORS["performance"],
        "performance_evidence_sha256": args.performance_evidence_sha256,
        "accelerated_placement": args.accelerated_placement,
    }


def build_checkpoint_identity(
    args: argparse.Namespace, queries: Sequence[str], documents: Sequence[str]
) -> dict[str, object]:
    source_directory = Path(__file__).resolve().parent
    implementation_files = (
        "export_adapter_cache.py", "admission_evidence.py", "scifact_contract.py",
        "profile_identity.py", "requirements-lock.txt",
    )
    return {
        "schema_version": 1,
        "cache_metadata": build_cache_metadata(args),
        "batch_size": args.batch_size,
        "queries_sha256": ordered_input_json_sha256(queries),
        "documents_sha256": ordered_input_json_sha256(documents),
        "implementation_sha256": {
            name: hashlib.sha256((source_directory / name).read_bytes()).hexdigest()
            for name in implementation_files
        },
    }


def write_cache(
    output: Path,
    metadata: dict[str, object],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    sequence_probe_arrays: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    wire_batch_outputs: np.ndarray,
    evidence: dict[str, bytes],
) -> None:
    import numpy as np

    if output.suffix != ".npz":
        raise ValueError("output path must end in .npz")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(
                temporary,
                metadata=json.dumps(metadata, sort_keys=True),
                queries=arrays[0],
                documents=arrays[1],
                queries_repeat=arrays[2],
                documents_repeat=arrays[3],
                sequence_probe_queries=sequence_probe_arrays[0],
                sequence_probe_relevant_documents=sequence_probe_arrays[1],
                sequence_probe_irrelevant_documents=sequence_probe_arrays[2],
                sequence_probe_queries_repeat=sequence_probe_arrays[3],
                sequence_probe_relevant_documents_repeat=sequence_probe_arrays[4],
                sequence_probe_irrelevant_documents_repeat=sequence_probe_arrays[5],
                wire_batch_outputs=wire_batch_outputs,
                sequence_capability_evidence_bytes=np.frombuffer(
                    evidence["sequence_capability"], dtype=np.uint8
                ),
                placement_evidence_bytes=np.frombuffer(
                    evidence["placement"], dtype=np.uint8
                ),
                performance_evidence_bytes=np.frombuffer(
                    evidence["performance"], dtype=np.uint8
                ),
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(output)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = build_parser().parse_args()
    with ExitStack() as stack:
        run_export(args, stack)


def run_export(args: argparse.Namespace, stack: ExitStack) -> None:
    try:
        endpoint = validate_loopback_endpoint(args.endpoint)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.output.suffix != ".npz":
        raise SystemExit("--output must end in .npz")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing cache: {args.output}")
    try:
        evidence = {
            "sequence_capability": read_verified_evidence(
                args.sequence_capability_evidence,
                args.sequence_capability_evidence_sha256,
            ),
            "placement": read_verified_evidence(
                args.placement_evidence, args.placement_evidence_sha256
            ),
            "performance": read_verified_evidence(
                args.performance_evidence, args.performance_evidence_sha256
            ),
        }
        sequence_report = evidence_json(
            args.sequence_capability_evidence, evidence["sequence_capability"]
        )
        placement_report = evidence_json(args.placement_evidence, evidence["placement"])
        performance_report = evidence_json(
            args.performance_evidence, evidence["performance"]
        )
        validate_evidence_reports(
            args, sequence_report, placement_report, performance_report
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    bearer_token = None
    if args.bearer_token_env is not None:
        bearer_token = os.environ.get(args.bearer_token_env)
        if not bearer_token:
            raise SystemExit(
                f"environment variable {args.bearer_token_env!r} is missing or empty"
            )

    try:
        query_inputs, document_inputs = load_scifact_inputs(args.scifact_snapshot)
    except (OSError, ValueError) as error:
        raise SystemExit(f"pinned SciFact loading failed: {error}") from error
    expected_execution = {
        "scope_id": args.scope_id,
        "transport": args.transport,
        "backend": args.backend,
        "runtime": args.runtime,
        "compiler": args.compiler,
        "package_target": args.package_target,
        "artifact_source": args.artifact_source,
        "device_class": args.device_class,
        "device": args.device,
        "artifact_sha256": args.artifact_sha256,
        "internal_precision": args.internal_precision,
        "placement_evidence_sha256": args.placement_evidence_sha256,
        "supported_max_tokens": args.supported_max_tokens,
        "supported_sequence_buckets": sorted(set(args.supported_sequence_bucket)),
        "supported_max_batch_size": SUPPORTED_MAX_BATCH_SIZE,
        "sequence_capability_evidence_sha256": args.sequence_capability_evidence_sha256,
        "performance_evidence_sha256": args.performance_evidence_sha256,
        "accelerated_placement": args.accelerated_placement,
    }
    if args.transport == SUPERVISED_LOCAL_TRANSPORT:
        expected_execution.update(
            {
                "package_state": "candidate",
                "compatibility_report_sha256": None,
            }
        )
    checkpoint = (
        stack.enter_context(ExportCheckpoint(
            args.checkpoint_directory,
            build_checkpoint_identity(args, query_inputs, document_inputs),
        ))
        if args.checkpoint_directory is not None else None
    )
    used_attestation_nonces: set[bytes] = set()

    def scoped_requests(trial: str) -> RequestFunction:
        transaction_index = 0

        def request(
            request_endpoint: str,
            requested_scope_id: str,
            texts: Sequence[str],
            request_timeout: float,
            request_token: str | None,
        ) -> object:
            nonlocal transaction_index
            key = f"{trial}-{transaction_index}"
            transaction_index += 1
            return request_embeddings(
                request_endpoint, requested_scope_id, texts, request_timeout, request_token,
                expected_execution=expected_execution,
                attestation_public_key=args.attestation_public_key,
                used_attestation_nonces=used_attestation_nonces,
                checkpoint=checkpoint, checkpoint_key=key,
            )

        return request

    wire_batch_outputs = verify_wire_batch_contract(
        endpoint,
        args.scope_id,
        wire_batch_inputs(query_inputs, document_inputs),
        args.timeout_seconds,
        bearer_token,
        sequence_report,
        trial_request_factory=scoped_requests,
    )

    def sequence_probe_requests(trial: str) -> SequenceProbeRequestFunction:
        def request(
            texts: Sequence[str],
            expected_row_metadata: Sequence[dict[str, object]],
        ) -> object:
            return request_embeddings(
                endpoint, args.scope_id, texts, args.timeout_seconds, bearer_token,
                expected_execution=expected_execution,
                attestation_public_key=args.attestation_public_key,
                expected_row_metadata=expected_row_metadata,
                used_attestation_nonces=used_attestation_nonces,
                checkpoint=checkpoint, checkpoint_key=trial,
            )

        return request

    sequence_probe_arrays = collect_sequence_probe_arrays(
        sequence_report, sequence_probe_requests("sequence"), sequence_probe_requests
    )
    arrays = collect_cache_arrays(
        endpoint,
        args.scope_id,
        query_inputs,
        document_inputs,
        args.batch_size,
        args.timeout_seconds,
        bearer_token,
        trial_request_factory=scoped_requests,
    )
    metadata = build_cache_metadata(args)
    write_cache(
        args.output,
        metadata,
        arrays,
        sequence_probe_arrays,
        wire_batch_outputs,
        evidence,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "queries": len(arrays[0]),
                "documents": len(arrays[1]),
                "backend": args.backend,
                "device_class": args.device_class,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
