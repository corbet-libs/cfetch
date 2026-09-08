#!/usr/bin/env python3
"""Focused tests for fail-closed physical evidence collection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from admission_evidence import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    WIRE_BATCH_INPUT_SELECTION,
)
from physical_evidence import (
    DispatcherSession,
    EvidenceError,
    OpenVinoLiveEvidenceValidator,
    SignedAdapterClient,
    ScopeContract,
    ResponseRow,
    SignedTransaction,
    _store_raw,
    _wire_grouping_results,
    canonical_i8_bytes,
    exact_i8_relevant_precedes,
    load_candidate_package,
    load_wire_inputs,
)
from packages.openvino import adapter as openvino_adapter
from scifact_contract import DATASET, DATASET_REVISION


def host_binding() -> dict[str, object]:
    return {
        "system": "Linux",
        "machine": "x86_64",
        "kernel_release": "test-kernel",
        "files": [{"path": "/usr/lib/libtest.so", "sha256": "a" * 64}],
    }


def scope_document(scope_id: str, device_class: str) -> dict[str, object]:
    device = {"npu": "NPU", "gpu": "GPU", "cpu": "CPU"}[device_class]
    properties: dict[str, object] = {
        "FULL_DEVICE_NAME": f"Test {device}",
        "DEVICE_ARCHITECTURE": f"test-{device_class}",
    }
    if device_class == "npu":
        properties.update({"NPU_DRIVER_VERSION": 1, "NPU_COMPILER_VERSION": 2})
    if device_class == "gpu":
        properties.update({"GPU_UARCH_VERSION": "test", "GPU_DEVICE_ID": "0x0000"})
    return {
        "scope_id": scope_id,
        "transport": "supervised-local",
        "backend": "openvino",
        "runtime": "openvino-test",
        "compiler": "openvino-test-static-buckets",
        "package_target": "linux-x86_64-glibc",
        "artifact_source": "test-source@revision/model",
        "artifact_sha256": "1" * 64,
        "internal_precision": "target-native",
        "device_class": device_class,
        "device": f"test-{device_class}",
        "openvino_device": device,
        "openvino_compile_config": {},
        "required_openvino_properties": properties,
        "required_execution_devices": [device],
        "required_host": host_binding(),
        "placement_evidence_sha256": None,
        "supported_max_tokens": 2048,
        "supported_sequence_buckets": [32, 64, 128, 257, 512, 1024, 2048],
        "supported_max_batch_size": 64,
        "sequence_capability_evidence_sha256": None,
        "performance_evidence_sha256": None,
        "compatibility_report_sha256": None,
        "attestation_public_key": "2" * 64,
        "attestation_private_key_file": f"keys/{scope_id}.key",
        "accelerated_placement": True,
    }


def package_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "package_state": "physical-probe",
        "profile_id": "cfetch-embedding-v1",
        "profile_manifest_sha256": (
            "59210a333494f788eb8e607fe38cabb6af1a7aa7cdf604ddf52e3fa6004b5afb"
        ),
        "admission_policy_sha256": (
            "36375d4e6fb5f6a48e5ecb7036ccc141c2314c2f2609c37063a6f79aa367fcc2"
        ),
        "model": "google/embeddinggemma-300m",
        "model_revision": "57c266a740f537b4dc058e1b0cda161fd15afa75",
        "artifact_manifest": "artifact/artifact-manifest.json",
        "artifact_manifest_sha256": "1" * 64,
        "runtime_manifest_sha256": "3" * 64,
        "dependency_versions": {
            "cryptography": "test",
            "numpy": "test",
            "openvino": "test",
            "tokenizers": "test",
        },
        "scopes": [
            scope_document("test-npu", "npu"),
            scope_document("test-gpu", "gpu"),
            scope_document("test-cpu", "cpu"),
        ],
    }


def write_manifest(root: Path, document: dict[str, object]) -> tuple[Path, str]:
    raw = (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    path = root / "package-manifest.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def scope_contract() -> ScopeContract:
    document = scope_document("test-cpu", "cpu")
    identity_fields = (
        "scope_id",
        "transport",
        "backend",
        "artifact_source",
        "artifact_sha256",
        "internal_precision",
        "runtime",
        "compiler",
        "package_target",
        "device",
        "device_class",
    )
    return ScopeContract(
        scope_id="test-cpu",
        document=document,
        identity={field: document[field] for field in identity_fields},
        public_key_hex="2" * 64,
        openvino_device="CPU",
        required_execution_devices=("CPU",),
        required_openvino_properties=document["required_openvino_properties"],
        required_host=document["required_host"],
        package_state="physical-probe",
    )


def live_evidence() -> dict[str, object]:
    properties = scope_document("test-cpu", "cpu")["required_openvino_properties"]
    return {
        "schema_version": 1,
        "provider": "openvino",
        "scope_id": "test-cpu",
        "host": host_binding(),
        "host_source": "platform-and-sha256",
        "bucket_results": [
            {
                "bucket": 32,
                "requested_device": "CPU",
                "execution_devices": ["CPU"],
                "execution_devices_source": (
                    "compiled_model.get_property(EXECUTION_DEVICES)"
                ),
                "device_properties": properties,
                "device_properties_source": "core.get_property",
            }
        ],
    }


class PhysicalEvidenceTests(unittest.TestCase):
    def test_wire_groupings_retain_one_content_addressed_signed_record_each(
        self,
    ) -> None:
        inputs = [f"input-{index:02d}" for index in range(64)]

        class Session:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                pass

        class Client:
            def __init__(self, *_args, **_kwargs) -> None:
                self.counter = 0

            def request(self, values, *, measure_rss):
                self.counter += 1
                request = json.dumps(list(values), separators=(",", ":")).encode()
                rows = tuple(
                    ResponseRow(
                        token_count=1,
                        sequence_bucket=32,
                        canonical=bytes([inputs.index(value)]),
                    )
                    for value in values
                )
                return SignedTransaction(
                    nonce_hex=f"{self.counter:064x}",
                    signature_hex="ab" * 64,
                    request_body=request,
                    response_body=b"signed:" + request,
                    elapsed_ns=1,
                    peak_rss_bytes=None,
                    rss_sample_count=0,
                    rows=rows,
                    runtime_evidence={},
                )

        package = SimpleNamespace(scope=SimpleNamespace(scope_id="test-cpu"))
        with tempfile.TemporaryDirectory() as directory, patch(
            "physical_evidence.DispatcherSession", Session
        ), patch("physical_evidence.SignedAdapterClient", Client):
            raw_root = Path(directory)
            results = _wire_grouping_results(
                Path("cfetch-openvino-adapter"),
                "0" * 64,
                package,
                1.0,
                1.0,
                inputs,
                raw_root,
                set(),
            )
            self.assertEqual(len(results), 64)
            self.assertEqual(len({row["signed_transactions_sha256"] for row in results}), 64)
            self.assertEqual(len(list(raw_root.glob("*.bin"))), 64)
            for row in results:
                raw = (
                    raw_root / f"{row['signed_transactions_sha256']}.bin"
                ).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row["signed_transactions_sha256"])
                document = json.loads(raw)
                self.assertEqual(document["batch_size"], row["batch_size"])
                self.assertEqual(len(document["transactions"]), row["request_count"])
                self.assertEqual(document["kind"], "wire-grouping-signed-transactions")

    def test_float32_codec_and_exact_integer_ranking_need_no_numpy(self) -> None:
        query = canonical_i8_bytes([1.0, 0.0] + [0.0] * 766)
        relevant = canonical_i8_bytes([1.0, 0.5] + [0.0] * 766)
        irrelevant = canonical_i8_bytes([-1.0, 0.0] + [0.0] * 766)
        self.assertEqual(len(query), 768)
        self.assertEqual(query[0], 127)
        self.assertTrue(exact_i8_relevant_precedes(query, relevant, irrelevant))
        with self.assertRaisesRegex(EvidenceError, "all zero"):
            canonical_i8_bytes([0.0] * 768)

    def test_live_placement_requires_compiled_and_core_property_sources(self) -> None:
        validator = OpenVinoLiveEvidenceValidator(scope_contract())
        validator.validate(live_evidence(), [32])

        echoed = live_evidence()
        echoed["bucket_results"][0]["execution_devices_source"] = "manifest"
        with self.assertRaisesRegex(EvidenceError, "echoed claim"):
            validator.validate(echoed, [32])

        wrong_device = live_evidence()
        wrong_device["bucket_results"][0]["execution_devices"] = ["GPU"]
        with self.assertRaisesRegex(EvidenceError, "unexpected device"):
            validator.validate(wrong_device, [32])

        wrong_host = live_evidence()
        wrong_host["host"]["kernel_release"] = "another-kernel"
        with self.assertRaisesRegex(EvidenceError, "host/kernel/file"):
            validator.validate(wrong_host, [32])

    def test_signed_client_accepts_only_the_package_key_and_live_evidence(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        document = scope_document("test-cpu", "cpu")
        document["attestation_public_key"] = public_key.hex()
        identity_fields = (
            "scope_id",
            "transport",
            "backend",
            "artifact_source",
            "artifact_sha256",
            "internal_precision",
            "runtime",
            "compiler",
            "package_target",
            "device",
            "device_class",
        )
        contract = ScopeContract(
            scope_id="test-cpu",
            document=document,
            identity={field: document[field] for field in identity_fields},
            public_key_hex=public_key.hex(),
            openvino_device="CPU",
            required_execution_devices=("CPU",),
            required_openvino_properties=document["required_openvino_properties"],
            required_host=document["required_host"],
            package_state="physical-probe",
        )

        class Signer:
            def __init__(self, key: Ed25519PrivateKey) -> None:
                self.key = key

            def sign(self, message: bytes) -> bytes:
                return self.key.sign(message)

        class Service:
            signer = Signer(private_key)

            def response_for(self, request_body: bytes):
                request = json.loads(request_body)
                if set(request) != {
                    "model",
                    "dimensions",
                    "input",
                    "cfetch_requested_scope_id",
                }:
                    raise RuntimeError("collector request schema changed")
                response = {
                    "model": "google/embeddinggemma-300m",
                    "cfetch_profile": "cfetch-embedding-v1",
                    "cfetch_profile_manifest_sha256": (
                        "59210a333494f788eb8e607fe38cabb6af1a7aa7cdf604ddf52e3fa6004b5afb"
                    ),
                    "cfetch_admission_policy_sha256": (
                        "36375d4e6fb5f6a48e5ecb7036ccc141c2314c2f2609c37063a6f79aa367fcc2"
                    ),
                    "cfetch_model_revision": (
                        "57c266a740f537b4dc058e1b0cda161fd15afa75"
                    ),
                    "cfetch_execution": contract.expected_execution(),
                    "cfetch_runtime_evidence": live_evidence(),
                    "data": [
                        {
                            "index": 0,
                            "cfetch_scope_id": "test-cpu",
                            "token_count": 32,
                            "sequence_bucket": 32,
                            "truncated": False,
                            "embedding": [1.0] + [0.0] * 767,
                        }
                    ],
                }
                raw = json.dumps(response, separators=(",", ":")).encode()
                return raw, self.signer

        service = Service()
        bearer = "b" * 64
        server = openvino_adapter.AdapterServer(
            ("127.0.0.1", 0), service, bearer
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        session = SimpleNamespace(
            process=SimpleNamespace(pid=os.getpid()),
            endpoint=f"http://127.0.0.1:{port}/v1/embeddings",
            bearer=bearer,
            package=SimpleNamespace(scope=contract),
        )
        try:
            client = SignedAdapterClient(session, 2.0)
            transaction = client.request([f"{QUERY_PREFIX}signed"], measure_rss=False)
            self.assertEqual(transaction.rows[0].sequence_bucket, 32)
            self.assertEqual(transaction.runtime_evidence["provider"], "openvino")

            service.signer = Signer(Ed25519PrivateKey.generate())
            with self.assertRaisesRegex(EvidenceError, "Ed25519 signature"):
                client.request([f"{QUERY_PREFIX}tampered signer"], measure_rss=False)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_probe_manifest_requires_null_evidence_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, digest = write_manifest(root, package_document())
            package = load_candidate_package(path, digest, "test-cpu")
            self.assertEqual(package.scope.package_state, "physical-probe")
            self.assertEqual(package.runtime_manifest_sha256, "3" * 64)
            self.assertIsNone(
                package.scope.expected_execution()["placement_evidence_sha256"]
            )

            bound = package_document()
            bound["scopes"][2]["placement_evidence_sha256"] = "4" * 64
            path, digest = write_manifest(root, bound)
            with self.assertRaisesRegex(EvidenceError, "must be explicitly null"):
                load_candidate_package(path, digest, "test-cpu")

            candidate = package_document()
            candidate["package_state"] = "candidate"
            path, digest = write_manifest(root, candidate)
            with self.assertRaisesRegex(EvidenceError, "package_state"):
                load_candidate_package(path, digest, "test-cpu")

            raw = json.dumps(package_document(), indent=2).encode() + b"\n"
            path.write_bytes(raw)
            with self.assertRaisesRegex(EvidenceError, "canonical JSON"):
                load_candidate_package(
                    path, hashlib.sha256(raw).hexdigest(), "test-cpu"
                )

    def test_dispatcher_process_is_hash_bound_and_stops_on_parent_eof(self) -> None:
        script = """#!/usr/bin/env python3
import json
import sys
if sys.argv[1:] != ['serve', '--host', '127.0.0.1', '--port', '0', '--auth-stdin']:
    raise SystemExit(2)
auth = json.loads(sys.stdin.buffer.readline())
if sorted(auth) != ['bearer'] or len(auth['bearer']) != 64:
    raise SystemExit(3)
print(json.dumps({'schema_version': 1, 'url': 'http://127.0.0.1:32123/v1', 'scope_ids': ['test-npu', 'test-gpu', 'test-cpu']}, separators=(',', ':')), flush=True)
while sys.stdin.buffer.read(4096):
    pass
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest_digest = write_manifest(root, package_document())
            package = load_candidate_package(
                manifest_path, manifest_digest, "test-cpu"
            )
            dispatcher = root / "dispatcher"
            dispatcher.write_text(script)
            dispatcher.chmod(0o755)
            dispatcher_digest = hashlib.sha256(dispatcher.read_bytes()).hexdigest()
            with DispatcherSession(
                dispatcher, dispatcher_digest, package, 2.0
            ) as session:
                self.assertEqual(
                    session.endpoint, "http://127.0.0.1:32123/v1/embeddings"
                )
            self.assertIsNotNone(session.startup_peak_rss_bytes)
            self.assertGreater(session.startup_rss_sample_count, 0)

            dispatcher.write_text(script + "\n")
            dispatcher.chmod(0o755)
            with self.assertRaisesRegex(EvidenceError, "sha256"):
                DispatcherSession(
                    dispatcher, dispatcher_digest, package, 2.0
                ).start()

    def test_raw_records_are_content_addressed_and_do_not_store_bearers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = {"schema_version": 1, "kind": "test", "value": 7}
            first = _store_raw(root, document)
            second = _store_raw(root, document)
            self.assertEqual(first, second)
            raw = (root / f"{first}.bin").read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), first)
            self.assertNotIn(b"bearer", raw)

    def test_wire_inputs_require_the_frozen_shape_and_prefix_roles(self) -> None:
        document = {
            "schema_version": 1,
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "selection": WIRE_BATCH_INPUT_SELECTION,
            "inputs": [f"{QUERY_PREFIX}query {index}" for index in range(32)]
            + [f"{DOCUMENT_PREFIX}document {index}" for index in range(32)],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wire-inputs.json"
            path.write_bytes(
                (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
            self.assertEqual(load_wire_inputs(path), document["inputs"])

            document["inputs"][32] = f"{QUERY_PREFIX}wrong role"
            path.write_bytes(
                (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
            with self.assertRaisesRegex(EvidenceError, "document prefix"):
                load_wire_inputs(path)


if __name__ == "__main__":
    unittest.main()
