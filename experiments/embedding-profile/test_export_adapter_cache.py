#!/usr/bin/env python3
"""Focused tests for the backend-neutral adapter evidence exporter."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from cross_backend_eval import load_cache
from export_adapter_cache import (
    ADMISSION_POLICY_SHA256,
    DATASET,
    DATASET_REVISION,
    DIMENSIONS,
    DOCUMENT_PREFIX,
    ExportCheckpoint,
    MAX_ADAPTER_RESPONSE_BYTES,
    MAX_TOKENS,
    MODEL,
    MODEL_REVISION,
    PROFILE_ID,
    PROFILE_MANIFEST_SHA256,
    QUERY_PREFIX,
    SEQUENCE_BUCKETS,
    SEQUENCE_PROBE_ARRAY_NAMES,
    SEQUENCE_SEMANTIC_FIXTURE_ID,
    SEQUENCE_SEMANTIC_FIXTURE_SHA256,
    SUPPORTED_MAX_BATCH_SIZE,
    WIRE_BATCH_INPUT_SELECTION,
    attestation_message,
    build_cache_metadata,
    build_checkpoint_identity,
    build_parser,
    canonical_i8,
    collect_cache_arrays,
    collect_sequence_probe_arrays,
    embed_canonical,
    ordered_input_json_sha256,
    read_bounded_adapter_response,
    read_verified_evidence,
    request_embeddings,
    sequence_semantic_probe_inputs,
    utf8_sha256,
    validate_evidence_reports,
    validate_loopback_endpoint,
    validate_response,
    verify_response_signature,
    verify_wire_batch_contract,
    wire_batch_inputs,
    write_cache,
)


def attested_response(rows: list[dict[str, object]]) -> dict[str, object]:
    attested_rows = [
        {
            "cfetch_scope_id": "test-scope",
            "token_count": 5,
            "sequence_bucket": 32,
            "truncated": False,
            **row,
        }
        for row in rows
    ]
    return {
        "model": MODEL,
        "cfetch_profile": PROFILE_ID,
        "cfetch_profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
        "cfetch_admission_policy_sha256": ADMISSION_POLICY_SHA256,
        "cfetch_model_revision": MODEL_REVISION,
        "cfetch_execution": {
            "scope_id": "test-scope",
            "transport": "supervised-local",
            "backend": "test-adapter",
            "runtime": "test-runtime",
            "compiler": "test-compiler",
            "package_target": "test-target",
            "artifact_source": "test-source@revision/model",
            "device_class": "cpu",
            "device": "test-device",
            "artifact_sha256": "1" * 64,
            "internal_precision": "test-native",
            "placement_evidence_sha256": "2" * 64,
            "supported_max_tokens": MAX_TOKENS,
            "supported_sequence_buckets": SEQUENCE_BUCKETS,
            "supported_max_batch_size": SUPPORTED_MAX_BATCH_SIZE,
            "sequence_capability_evidence_sha256": "3" * 64,
            "performance_evidence_sha256": "4" * 64,
            "accelerated_placement": True,
        },
        "data": attested_rows,
    }


def vector(seed: float) -> list[float]:
    return [seed, -seed / 2.0] + [0.0] * (DIMENSIONS - 2)


def semantic_probe_fixture(bucket: int) -> dict[str, object]:
    query, relevant, irrelevant = sequence_semantic_probe_inputs(bucket)
    return {
        "fixture_id": SEQUENCE_SEMANTIC_FIXTURE_ID,
        "fixture_sha256": SEQUENCE_SEMANTIC_FIXTURE_SHA256,
        "query_input_utf8_sha256": utf8_sha256(query),
        "relevant_document_input_utf8_sha256": utf8_sha256(relevant),
        "irrelevant_document_input_utf8_sha256": utf8_sha256(irrelevant),
        "query_token_count": bucket,
        "relevant_document_token_count": bucket,
        "irrelevant_document_token_count": bucket,
        "query_canonical_output_bytes_sha256": "7" * 64,
        "relevant_document_canonical_output_bytes_sha256": "8" * 64,
        "irrelevant_document_canonical_output_bytes_sha256": "9" * 64,
        "canonical_repeatability": True,
        "self_relevant_before_irrelevant": True,
    }


def openvino_provider_binding() -> dict[str, object]:
    host = {
        "system": "Linux",
        "machine": "x86_64",
        "kernel_release": "test-kernel",
        "files": [{"path": "/usr/lib/libtest.so", "sha256": "a" * 64}],
    }
    return {
        "schema_version": 1,
        "provider": "openvino",
        "dispatcher_sha256": "b" * 64,
        "probe_package_manifest_sha256": "c" * 64,
        "runtime_manifest_sha256": "d" * 64,
        "openvino_compile_config": {},
        "expected_host": host,
        "actual_host": copy.deepcopy(host),
        "host_source": "platform-and-sha256",
    }


def openvino_provider_evidence() -> dict[str, object]:
    properties = {
        "FULL_DEVICE_NAME": "Test accelerated CPU",
        "DEVICE_ARCHITECTURE": "test-architecture",
    }
    return {
        "schema_version": 1,
        "provider": "openvino",
        "requested_device": "CPU",
        "expected_execution_devices": ["CPU"],
        "actual_execution_devices": ["CPU"],
        "execution_devices_source": (
            "compiled_model.get_property(EXECUTION_DEVICES)"
        ),
        "expected_device_properties": properties,
        "actual_device_properties": dict(properties),
        "device_properties_source": "core.get_property",
    }


def valid_evidence_fixture() -> tuple[
    argparse.Namespace,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    args = argparse.Namespace(
        scope_id="test-scope",
        transport="supervised-local",
        backend="test-adapter",
        artifact_source="test-source@revision/model",
        artifact_sha256="1" * 64,
        internal_precision="test-native",
        runtime="test-runtime",
        compiler="test-compiler",
        package_target="test-target",
        device="test-device",
        device_class="cpu",
        supported_max_tokens=64,
        supported_sequence_bucket=[32, 64],
    )
    identity = {
        "scope_id": args.scope_id,
        "transport": args.transport,
        "backend": args.backend,
        "artifact_source": args.artifact_source,
        "artifact_sha256": args.artifact_sha256,
        "internal_precision": args.internal_precision,
        "runtime": args.runtime,
        "compiler": args.compiler,
        "package_target": args.package_target,
        "device": args.device,
        "device_class": args.device_class,
    }
    sequence = {
        **identity,
        "supported_max_tokens": 64,
        "supported_sequence_buckets": [32, 64],
        "supported_max_batch_size": SUPPORTED_MAX_BATCH_SIZE,
        "wire_batch_results": [
            {
                "batch_size": batch_size,
                "input_count": SUPPORTED_MAX_BATCH_SIZE,
                "request_count": (
                    SUPPORTED_MAX_BATCH_SIZE + batch_size - 1
                )
                // batch_size,
                "response_row_count": SUPPORTED_MAX_BATCH_SIZE,
                "ordered_input_json_sha256": "5" * 64,
                "canonical_output_bytes_sha256": "5" * 64,
                "signed_transactions_sha256": hashlib.sha256(
                    f"wire-transactions-{batch_size}".encode()
                ).hexdigest(),
            }
            for batch_size in range(1, SUPPORTED_MAX_BATCH_SIZE + 1)
        ],
        "grouping_invariance": {
            "batch_sizes": list(range(1, SUPPORTED_MAX_BATCH_SIZE + 1)),
            "input_selection": WIRE_BATCH_INPUT_SELECTION,
            "same_inputs_in_same_order": True,
            "canonical_output_bytes_equal": True,
        },
        "bucket_results": [
            {
                "bucket": bucket,
                "requested_tokens": bucket,
                "tokenized_tokens": bucket,
                "executed_shape_tokens": bucket,
                "output_dimensions": DIMENSIONS,
                "finite_output": True,
                "nonzero_output": True,
                "truncated": False,
                "semantic_probe": semantic_probe_fixture(bucket),
            }
            for bucket in (32, 64)
        ],
    }
    placement = {
        **identity,
        "accelerated_placement": True,
        "accelerator_execution_confirmed": True,
        "fallback_disclosure_complete": True,
        "unexpected_fallback_detected": False,
        "provider_binding": openvino_provider_binding(),
        "bucket_results": [
            {
                "bucket": bucket,
                "accelerator_execution_confirmed": True,
                "fallback_disclosure_complete": True,
                "unexpected_fallback_detected": False,
                "fallback_summary": "none",
                "profiler_output_sha256": "2" * 64,
                "provider_evidence": openvino_provider_evidence(),
            }
            for bucket in (32, 64)
        ],
    }
    performance = {
        **identity,
        "bucket_results": [
            {
                "bucket": 32,
                "sample_count": 10,
                "benchmark_output_sha256": "6" * 64,
                "latency_ms_p50": 1.0,
                "latency_ms_p95": 2.0,
                "peak_memory_bytes": 1024,
                "energy_joules": 0.25,
                "average_power_watts": 1.5,
            },
            {
                "bucket": 64,
                "sample_count": 10,
                "benchmark_output_sha256": "6" * 64,
                "latency_ms_p50": 1.0,
                "latency_ms_p95": 2.0,
                "peak_memory_bytes": 1024,
                "energy_measurement": "not_measured",
                "energy_not_measured_reason": "meter unavailable",
            },
        ],
    }
    return args, sequence, placement, performance


class ExportAdapterCacheTests(unittest.TestCase):
    def test_evidence_schema_binds_scope_buckets_placement_and_performance(self) -> None:
        args, sequence, placement, performance = valid_evidence_fixture()
        validate_evidence_reports(args, sequence, placement, performance)

        cases = []
        changed = copy.deepcopy(sequence)
        changed["artifact_sha256"] = "9" * 64
        cases.append(("scope identity", changed, placement, performance, "artifact_sha256"))
        changed = copy.deepcopy(sequence)
        changed["bucket_results"][0]["truncated"] = True
        cases.append(("truncation", changed, placement, performance, "truncated"))
        changed = copy.deepcopy(sequence)
        changed["bucket_results"][0]["semantic_probe"][
            "canonical_repeatability"
        ] = False
        cases.append(
            ("bucket repeatability", changed, placement, performance, "repeatability")
        )
        changed = copy.deepcopy(sequence)
        changed["bucket_results"][1]["semantic_probe"]["query_token_count"] = 32
        cases.append(
            ("probe bucket selection", changed, placement, performance, "select this bucket")
        )
        changed = copy.deepcopy(sequence)
        changed["bucket_results"][0]["semantic_probe"][
            "query_input_utf8_sha256"
        ] = "0" * 64
        cases.append(
            ("probe input", changed, placement, performance, "pinned semantic probe")
        )
        changed = copy.deepcopy(sequence)
        changed["bucket_results"].pop()
        cases.append(("missing bucket", changed, placement, performance, "cover exactly"))
        changed = copy.deepcopy(sequence)
        changed["supported_max_batch_size"] = 32
        cases.append(("max batch", changed, placement, performance, "max_batch_size"))
        changed = copy.deepcopy(sequence)
        changed["wire_batch_results"][1]["canonical_output_bytes_sha256"] = "6" * 64
        cases.append(
            ("batch grouping", changed, placement, performance, "canonical outputs differ")
        )
        changed = copy.deepcopy(sequence)
        changed["wire_batch_results"][2]["request_count"] = 21
        cases.append(("ceil request count", changed, placement, performance, "must be 22"))
        changed = copy.deepcopy(sequence)
        changed["wire_batch_results"].pop(17)
        cases.append(("every batch size", changed, placement, performance, "cover exactly"))
        changed = copy.deepcopy(sequence)
        changed["wire_batch_results"][9]["ordered_input_json_sha256"] = "6" * 64
        cases.append(
            ("ordered batch inputs", changed, placement, performance, "different ordered inputs")
        )
        changed = copy.deepcopy(sequence)
        changed["grouping_invariance"]["same_inputs_in_same_order"] = False
        cases.append(
            ("input grouping", changed, placement, performance, "same_inputs_in_same_order")
        )
        changed_placement = copy.deepcopy(placement)
        changed_placement["bucket_results"][0]["unexpected_fallback_detected"] = True
        cases.append(
            ("unexpected fallback", sequence, changed_placement, performance, "unexpected fallback")
        )
        changed_placement = copy.deepcopy(placement)
        changed_placement["bucket_results"][0]["profiler_output_sha256"] = "not-a-digest"
        cases.append(("profiler digest", sequence, changed_placement, performance, "profiler"))
        changed_performance = copy.deepcopy(performance)
        changed_performance["bucket_results"][0]["latency_ms_p95"] = 0.5
        cases.append(("latency order", sequence, placement, changed_performance, "below p50"))
        changed_performance = copy.deepcopy(performance)
        del changed_performance["bucket_results"][0]["energy_joules"]
        cases.append(
            ("partial energy", sequence, placement, changed_performance, "energy_joules")
        )
        changed_performance = copy.deepcopy(performance)
        changed_performance["bucket_results"][1]["energy_not_measured_reason"] = " "
        cases.append(
            ("missing energy reason", sequence, placement, changed_performance, "nonempty")
        )
        for name, candidate_sequence, candidate_placement, candidate_performance, error in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, error):
                validate_evidence_reports(
                    args,
                    candidate_sequence,
                    candidate_placement,
                    candidate_performance,
                )

    def test_evidence_file_digest_is_checked_against_the_actual_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            data = b'{"evidence":true}\n'
            path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            self.assertEqual(read_verified_evidence(path, digest), data)
            with self.assertRaisesRegex(ValueError, "sha256"):
                read_verified_evidence(path, "0" * 64)

    def test_cache_metadata_never_persists_private_evidence_paths(self) -> None:
        args = argparse.Namespace(
            scope_id="test-scope",
            transport="supervised-local",
            backend="test-adapter",
            runtime="test-runtime",
            compiler="test-compiler",
            package_target="test-target",
            artifact_source="test-source@revision/model",
            artifact_sha256="1" * 64,
            attestation_public_key="5" * 64,
            internal_precision="test-native",
            supported_max_tokens=MAX_TOKENS,
            supported_sequence_bucket=SEQUENCE_BUCKETS,
            sequence_capability_evidence=Path(
                "/home/private-user/results/sequence.json"
            ),
            sequence_capability_evidence_sha256="2" * 64,
            device="test-device",
            device_class="cpu",
            placement_evidence=Path("/home/private-user/results/placement.json"),
            placement_evidence_sha256="3" * 64,
            performance_evidence=Path(
                "/home/private-user/results/performance.json"
            ),
            performance_evidence_sha256="4" * 64,
            accelerated_placement=True,
            scifact_snapshot=Path("/home/private-user/models/scifact-snapshot"),
        )

        metadata = build_cache_metadata(args)
        serialized = json.dumps(metadata, sort_keys=True)

        self.assertNotIn("/home/", serialized)
        self.assertEqual(
            metadata["sequence_capability_evidence"],
            "npz:sequence_capability_evidence_bytes",
        )
        self.assertEqual(
            metadata["placement_evidence"], "npz:placement_evidence_bytes"
        )
        self.assertEqual(
            metadata["performance_evidence"], "npz:performance_evidence_bytes"
        )

    def test_raw_scifact_snapshot_is_explicit_and_optional(self) -> None:
        action = next(
            item for item in build_parser()._actions if item.dest == "scifact_snapshot"
        )
        self.assertFalse(action.required)
        self.assertIsNone(action.default)
        self.assertEqual(action.type, Path)

    def test_canonical_codec_is_signed_int8_and_rounds_ties_to_even(self) -> None:
        values = np.zeros((1, DIMENSIONS), dtype=np.float32)
        values[0, :5] = [
            1.0,
            0.5 / 127.0,
            1.5 / 127.0,
            -0.5 / 127.0,
            -1.5 / 127.0,
        ]

        encoded = canonical_i8(values)

        self.assertEqual(encoded.dtype, np.dtype(np.int8))
        self.assertEqual(encoded[0, :5].tolist(), [127, 0, 2, 0, -2])

    def test_export_batch_size_cannot_exceed_wire_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed 64"):
            embed_canonical(
                "http://127.0.0.1/embeddings",
                "test-scope",
                [],
                SUPPORTED_MAX_BATCH_SIZE + 1,
                1.0,
                None,
            )

    def test_admission_export_requires_a_lowercase_ed25519_public_key(self) -> None:
        action = next(
            item
            for item in build_parser()._actions
            if item.dest == "attestation_public_key"
        )
        self.assertTrue(action.required)
        self.assertEqual(action.type("ab" * 32), "ab" * 32)
        for invalid in ("AB" * 32, "ab" * 31, "z0" * 32):
            with self.subTest(invalid=invalid), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                action.type(invalid)

        transport = next(
            item for item in build_parser()._actions if item.dest == "transport"
        )
        self.assertTrue(transport.required)
        self.assertEqual(
            tuple(transport.choices), ("supervised-local", "remote-attested")
        )

    def test_package_signature_binds_nonce_and_exact_request_response_bytes(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ).hex()
        nonce = bytes(range(32))
        request_body = b'{"input":["one"]}'
        response_body = b'{"data":[{"index":0}]}'
        signature = private_key.sign(
            attestation_message(nonce, request_body, response_body)
        ).hex()

        verify_response_signature(
            public_key, signature, nonce, request_body, response_body
        )
        for name, changed_nonce, changed_request, changed_response in (
            ("nonce", bytes(reversed(nonce)), request_body, response_body),
            ("request", nonce, request_body + b" ", response_body),
            ("response", nonce, request_body, response_body + b" "),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "scope-key signature"
            ):
                verify_response_signature(
                    public_key,
                    signature,
                    changed_nonce,
                    changed_request,
                    changed_response,
                )

    def test_signed_http_requests_use_fresh_nonces(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ).hex()

        class FakeResponse:
            def __init__(self, body: bytes, signature: str) -> None:
                self.body = body
                self.headers = {"X-Cfetch-Attestation-Signature": signature}

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                del args

            def read(self, amount: int = -1) -> bytes:
                return self.body[:amount] if amount >= 0 else self.body

        class SigningOpener:
            def __init__(self) -> None:
                self.nonces: list[bytes] = []
                self.requested_scopes: list[str] = []

            def open(self, request, timeout: float):
                del timeout
                headers = {
                    name.lower(): value for name, value in request.header_items()
                }
                nonce = bytes.fromhex(headers["x-cfetch-attestation-nonce"])
                self.nonces.append(nonce)
                request_payload = json.loads(request.data)
                self.requested_scopes.append(
                    request_payload["cfetch_requested_scope_id"]
                )
                rows = [
                    {"index": index, "embedding": vector(float(index + 1))}
                    for index, _ in enumerate(request_payload["input"])
                ]
                response_payload = attested_response(rows)
                response_payload["cfetch_execution"][
                    "compatibility_report_sha256"
                ] = "a" * 64
                response_body = json.dumps(
                    response_payload, separators=(",", ":")
                ).encode("utf-8")
                signature = private_key.sign(
                    attestation_message(nonce, request.data, response_body)
                ).hex()
                return FakeResponse(response_body, signature)

        opener = SigningOpener()
        used_nonces: set[bytes] = set()
        for _ in range(2):
            result = request_embeddings(
                "http://127.0.0.1:1234/embeddings",
                "test-scope",
                [QUERY_PREFIX + "signed"],
                7.0,
                opener=opener,
                attestation_public_key=public_key,
                expected_compatibility_report_sha256="a" * 64,
                used_attestation_nonces=used_nonces,
            )
            self.assertEqual(result.shape, (1, DIMENSIONS))
        self.assertEqual(len(opener.nonces), 2)
        self.assertEqual(len(opener.nonces[0]), 32)
        self.assertNotEqual(opener.nonces[0], opener.nonces[1])
        self.assertEqual(used_nonces, set(opener.nonces))
        self.assertEqual(opener.requested_scopes, ["test-scope", "test-scope"])
        with patch(
            "export_adapter_cache.secrets.token_bytes", return_value=opener.nonces[0]
        ), self.assertRaisesRegex(ValueError, "repeated a prior challenge"):
            request_embeddings(
                "http://127.0.0.1:1234/embeddings",
                "test-scope",
                [QUERY_PREFIX + "signed"],
                7.0,
                opener=opener,
                attestation_public_key=public_key,
                expected_compatibility_report_sha256="a" * 64,
                used_attestation_nonces=used_nonces,
            )

    def test_adapter_response_rejects_oversized_content_length_before_read(self) -> None:
        class FakeResponse:
            headers = {"Content-Length": str(MAX_ADAPTER_RESPONSE_BYTES + 1)}
            read_called = False

            def read(self, amount: int) -> bytes:
                del amount
                self.read_called = True
                return b""

        response = FakeResponse()
        with self.assertRaisesRegex(ValueError, "Content-Length.*response limit"):
            read_bounded_adapter_response(response)
        self.assertFalse(response.read_called, "oversized declared bodies must not be read")

    def test_chunked_adapter_response_reads_only_limit_plus_one(self) -> None:
        class FakeResponse:
            headers = {"Transfer-Encoding": "chunked"}
            requested: int | None = None

            def read(self, amount: int) -> bytes:
                self.requested = amount
                return b"x" * amount

        response = FakeResponse()
        with self.assertRaisesRegex(ValueError, "response exceeds.*byte limit"):
            read_bounded_adapter_response(response)
        self.assertEqual(response.requested, MAX_ADAPTER_RESPONSE_BYTES + 1)

    def test_exporter_replays_same_64_inputs_at_every_wire_batch_size(self) -> None:
        queries = [f"{QUERY_PREFIX}query-{index}" for index in range(32)]
        documents = [f"{DOCUMENT_PREFIX}document-{index}" for index in range(32)]
        inputs = wire_batch_inputs(queries, documents)
        args, sequence, _, _ = valid_evidence_fixture()
        del args

        def floats_for(texts: Sequence[str]) -> np.ndarray:
            values = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
            for index, text in enumerate(texts):
                values[index, 0] = 1.0
                values[index, 1] = (sum(text.encode("utf-8")) % 97) / 100.0
            return values

        expected_digest = hashlib.sha256(
            canonical_i8(floats_for(inputs)).tobytes()
        ).hexdigest()
        input_digest = ordered_input_json_sha256(inputs)
        for row in sequence["wire_batch_results"]:
            row["ordered_input_json_sha256"] = input_digest
            row["canonical_output_bytes_sha256"] = expected_digest

        calls: list[list[str]] = []
        trials: list[str] = []

        def fake_request(
            endpoint: str,
            requested_scope_id: str,
            texts: Sequence[str],
            timeout: float,
            token: str | None,
        ) -> np.ndarray:
            del endpoint, timeout, token
            self.assertEqual(requested_scope_id, "test-scope")
            calls.append(list(texts))
            return floats_for(texts)

        def trial_requests(trial: str):
            trials.append(trial)
            return fake_request

        outputs = verify_wire_batch_contract(
            "http://127.0.0.1/embeddings",
            "test-scope",
            inputs,
            1.0,
            None,
            sequence,
            fake_request,
            trial_requests,
        )

        self.assertEqual(outputs.shape, (64, 64, DIMENSIONS))
        self.assertEqual(trials, [f"wire-{size}" for size in range(1, 65)])
        self.assertTrue(
            all(np.array_equal(outputs[0], item) for item in outputs[1:])
        )
        self.assertEqual(
            {
                hashlib.sha256(np.ascontiguousarray(item).tobytes()).hexdigest()
                for item in outputs
            },
            {expected_digest},
        )
        self.assertEqual(
            len(calls),
            sum((64 + batch_size - 1) // batch_size for batch_size in range(1, 65)),
        )

    def test_every_sequence_bucket_runs_twice_and_binds_semantic_outputs(self) -> None:
        _, sequence, _, _ = valid_evidence_fixture()
        floats = np.zeros((3, DIMENSIONS), dtype=np.float32)
        floats[0, 0] = 1.0
        floats[1, :2] = [1.0, 0.5]
        floats[2, 0] = -1.0
        canonical = canonical_i8(floats)
        for row in sequence["bucket_results"]:
            probe = row["semantic_probe"]
            for index, label in enumerate(
                ("query", "relevant_document", "irrelevant_document")
            ):
                probe[f"{label}_canonical_output_bytes_sha256"] = hashlib.sha256(
                    canonical[index].tobytes()
                ).hexdigest()

        calls: list[tuple[list[str], list[dict[str, object]]]] = []
        trials: list[str] = []

        def fake_request(
            texts: Sequence[str], metadata: Sequence[dict[str, object]]
        ) -> np.ndarray:
            calls.append((list(texts), list(metadata)))
            return floats.copy()

        def trial_requests(trial: str):
            trials.append(trial)
            return fake_request

        arrays = collect_sequence_probe_arrays(sequence, fake_request, trial_requests)

        self.assertEqual(len(calls), 4)
        self.assertEqual(trials, [
            "sequence-32-first", "sequence-32-repeat",
            "sequence-64-first", "sequence-64-repeat",
        ])
        self.assertEqual(len(arrays), len(SEQUENCE_PROBE_ARRAY_NAMES))
        self.assertTrue(all(array.shape == (2, DIMENSIONS) for array in arrays))
        self.assertTrue(np.array_equal(arrays[0], arrays[3]))
        self.assertTrue(np.array_equal(arrays[1], arrays[4]))
        self.assertTrue(np.array_equal(arrays[2], arrays[5]))
        for offset, bucket in enumerate((32, 64)):
            expected_inputs = list(sequence_semantic_probe_inputs(bucket))
            first_call = calls[offset * 2]
            repeat_call = calls[offset * 2 + 1]
            self.assertEqual(first_call, repeat_call)
            self.assertEqual(first_call[0], expected_inputs)
            self.assertTrue(
                all(row["sequence_bucket"] == bucket for row in first_call[1])
            )

        call_index = 0

        def unstable_request(
            texts: Sequence[str], metadata: Sequence[dict[str, object]]
        ) -> np.ndarray:
            nonlocal call_index
            del texts, metadata
            call_index += 1
            changed = floats.copy()
            if call_index == 2:
                changed[0, :2] = [1.0, 0.25]
            return changed

        with self.assertRaisesRegex(ValueError, "not byte-repeatable"):
            collect_sequence_probe_arrays(sequence, unstable_request)

    def test_response_attestation_and_explicit_indices_are_enforced(self) -> None:
        payload = attested_response(
            [
                {"index": 1, "embedding": vector(2.0)},
                {"index": 0, "embedding": vector(1.0)},
            ]
        )

        ordered = validate_response(payload, 2, "test-scope")

        self.assertEqual(ordered[:, 0].tolist(), [1.0, 2.0])
        with self.subTest("next smallest sequence bucket"):
            boundary = attested_response(
                [
                    {
                        "index": 0,
                        "embedding": vector(1.0),
                        "token_count": 33,
                        "sequence_bucket": 64,
                    }
                ]
            )
            validate_response(boundary, 1, "test-scope")
            with self.assertRaisesRegex(ValueError, "sequence evidence requires"):
                validate_response(
                    boundary,
                    1,
                    "test-scope",
                    expected_row_metadata=[
                        {"token_count": 34, "sequence_bucket": 64}
                    ],
                )
        with self.subTest("wrong manifest"):
            invalid = dict(payload)
            invalid["cfetch_profile_manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "manifest"):
                validate_response(invalid, 2, "test-scope")
        with self.subTest("invalid transport"):
            invalid = copy.deepcopy(payload)
            invalid["cfetch_execution"]["transport"] = "loopback"
            with self.assertRaisesRegex(ValueError, "transport"):
                validate_response(invalid, 2, "test-scope")
        with self.subTest("duplicate index"):
            invalid = attested_response(
                [
                    {"index": 0, "embedding": vector(1.0)},
                    {"index": 0, "embedding": vector(2.0)},
                ]
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_response(invalid, 2, "test-scope")
        with self.subTest("non-finite vector"):
            invalid = attested_response([{"index": 0, "embedding": vector(float("nan"))}])
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_response(invalid, 1, "test-scope")
        with self.subTest("zero vector"):
            invalid = attested_response(
                [{"index": 0, "embedding": [0.0] * DIMENSIONS}]
            )
            with self.assertRaisesRegex(ValueError, "all zero"):
                validate_response(invalid, 1, "test-scope")
        for name, field, value, error in (
            ("wrong row scope", "cfetch_scope_id", "other-scope", "cfetch_scope_id"),
            ("zero token count", "token_count", 0, "token_count"),
            ("boolean token count", "token_count", True, "token_count"),
            ("overlength", "token_count", MAX_TOKENS + 1, "token_count"),
            ("larger bucket", "sequence_bucket", 64, "smallest supported bucket"),
            ("truncation", "truncated", True, "truncated=false"),
        ):
            with self.subTest(name=name):
                invalid = copy.deepcopy(payload)
                invalid["data"][0][field] = value
                with self.assertRaisesRegex(ValueError, error):
                    validate_response(invalid, 2, "test-scope")
        for name, field, error in (
            ("missing row scope", "cfetch_scope_id", "cfetch_scope_id"),
            ("missing token count", "token_count", "token_count"),
            ("missing bucket", "sequence_bucket", "sequence_bucket"),
            ("missing truncation", "truncated", "truncated=false"),
        ):
            with self.subTest(name=name):
                invalid = copy.deepcopy(payload)
                del invalid["data"][0][field]
                with self.assertRaisesRegex(ValueError, error):
                    validate_response(invalid, 2, "test-scope")
        with self.subTest("execution maximum batch"):
            invalid = copy.deepcopy(payload)
            invalid["cfetch_execution"]["supported_max_batch_size"] = 63
            with self.assertRaisesRegex(ValueError, "max_batch_size"):
                validate_response(invalid, 2, "test-scope")
        with self.subTest("token count not covered by configured buckets"):
            invalid = attested_response(
                [
                    {
                        "index": 0,
                        "embedding": vector(1.0),
                        "token_count": 64,
                        "sequence_bucket": 32,
                    }
                ]
            )
            invalid["cfetch_execution"]["supported_max_tokens"] = 64
            invalid["cfetch_execution"]["supported_sequence_buckets"] = [32]
            with self.assertRaisesRegex(ValueError, "does not fit"):
                validate_response(invalid, 1, "test-scope")
        with self.subTest("execution scope mismatch"):
            with self.assertRaisesRegex(ValueError, "scope_id"):
                validate_response(
                    payload,
                    2,
                    "test-scope",
                    {
                        "scope_id": "another-scope",
                        "transport": "supervised-local",
                        "backend": "test-adapter",
                        "runtime": "test-runtime",
                        "compiler": "test-compiler",
                        "package_target": "test-target",
                        "artifact_source": "test-source@revision/model",
                        "device_class": "cpu",
                        "device": "test-device",
                        "artifact_sha256": "1" * 64,
                        "internal_precision": "test-native",
                        "placement_evidence_sha256": "2" * 64,
                        "supported_max_tokens": MAX_TOKENS,
                        "supported_sequence_buckets": SEQUENCE_BUCKETS,
                        "supported_max_batch_size": SUPPORTED_MAX_BATCH_SIZE,
                        "sequence_capability_evidence_sha256": "3" * 64,
                        "performance_evidence_sha256": "4" * 64,
                        "accelerated_placement": True,
                    },
                )
        with self.subTest("supervised candidate lifecycle is explicit"):
            candidate = copy.deepcopy(payload)
            candidate["cfetch_execution"]["package_state"] = "candidate"
            candidate["cfetch_execution"]["compatibility_report_sha256"] = None
            lifecycle = {
                "package_state": "candidate",
                "compatibility_report_sha256": None,
            }
            validate_response(
                candidate,
                2,
                "test-scope",
                expected_execution=lifecycle,
            )
            missing = copy.deepcopy(candidate)
            del missing["cfetch_execution"]["compatibility_report_sha256"]
            with self.assertRaisesRegex(ValueError, "compatibility_report_sha256"):
                validate_response(
                    missing,
                    2,
                    "test-scope",
                    expected_execution=lifecycle,
                )
            release = copy.deepcopy(candidate)
            release["cfetch_execution"]["package_state"] = "release"
            with self.assertRaisesRegex(ValueError, "package_state"):
                validate_response(
                    release,
                    2,
                    "test-scope",
                    expected_execution=lifecycle,
                )
        with self.subTest("requested scope mismatch"):
            with self.assertRaisesRegex(ValueError, "requested 'another-scope'"):
                validate_response(payload, 2, "another-scope")
        with self.subTest("final compatibility report"):
            final = copy.deepcopy(payload)
            final["cfetch_execution"]["compatibility_report_sha256"] = "a" * 64
            validate_response(
                final,
                2,
                "test-scope",
                expected_compatibility_report_sha256="a" * 64,
            )
            with self.assertRaisesRegex(ValueError, "compatibility_report_sha256"):
                validate_response(
                    final,
                    2,
                    "test-scope",
                    expected_compatibility_report_sha256="b" * 64,
                )

    def test_only_loopback_embedding_urls_are_accepted(self) -> None:
        self.assertEqual(
            validate_loopback_endpoint("http://127.0.0.1:8080/v1/embeddings"),
            "http://127.0.0.1:8080/v1/embeddings",
        )
        self.assertEqual(
            validate_loopback_endpoint("http://[::1]:8080/embeddings"),
            "http://[::1]:8080/embeddings",
        )
        for endpoint in (
            "https://example.com/embeddings",
            "file:///embeddings",
            "http://localhost:8080/not-embeddings",
            "http://localhost:8080/embeddings?redirect=1",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_loopback_endpoint(endpoint)

    def test_local_adapter_is_batched_run_twice_and_writes_gate_schema(self) -> None:
        requests: list[dict[str, object]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                payload = json.loads(self.rfile.read(length))
                requests.append(payload)
                rows = [
                    {
                        "index": index,
                        "embedding": vector(float(len(text.encode("utf-8")) + 1)),
                    }
                    for index, text in reversed(list(enumerate(payload["input"])))
                ]
                response = json.dumps(attested_response(rows)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_port}/embeddings"
        query_inputs = [QUERY_PREFIX + "one", QUERY_PREFIX + "two"]
        document_inputs = [
            DOCUMENT_PREFIX + "alpha",
            DOCUMENT_PREFIX + "beta",
            DOCUMENT_PREFIX + "gamma",
        ]
        try:
            arrays = collect_cache_arrays(
                endpoint,
                "test-scope",
                query_inputs,
                document_inputs,
                batch_size=2,
                timeout_seconds=5.0,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5.0)

        self.assertEqual(len(requests), 6)
        self.assertEqual(
            [len(request["input"]) for request in requests],
            [2, 2, 1, 2, 2, 1],
        )
        self.assertTrue(
            all(request["model"] == MODEL for request in requests)
        )
        self.assertTrue(
            all(request["dimensions"] == DIMENSIONS for request in requests)
        )
        self.assertTrue(
            all(
                request["cfetch_requested_scope_id"] == "test-scope"
                for request in requests
            )
        )
        self.assertEqual(arrays[0].shape, (2, DIMENSIONS))
        self.assertEqual(arrays[1].shape, (3, DIMENSIONS))
        self.assertTrue(np.array_equal(arrays[0], arrays[2]))
        self.assertTrue(np.array_equal(arrays[1], arrays[3]))

        metadata = {
            "schema_version": 1,
            "profile_id": PROFILE_ID,
            "profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
            "admission_policy_sha256": ADMISSION_POLICY_SHA256,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "vector_encoding": "signed-int8x768",
            "supported_max_tokens": MAX_TOKENS,
            "supported_sequence_buckets": SEQUENCE_BUCKETS,
            "supported_max_batch_size": SUPPORTED_MAX_BATCH_SIZE,
            "sequence_semantic_fixture_id": SEQUENCE_SEMANTIC_FIXTURE_ID,
            "sequence_semantic_fixture_sha256": SEQUENCE_SEMANTIC_FIXTURE_SHA256,
            "sequence_capability_evidence": "npz:sequence_capability_evidence_bytes",
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "scope_id": "test-scope",
            "transport": "supervised-local",
            "backend": "test-adapter",
            "runtime": "test-runtime",
            "compiler": "test-compiler",
            "package_target": "test-target",
            "artifact_source": "test-source@revision/model",
            "artifact_sha256": "1" * 64,
            "attestation_public_key": "5" * 64,
            "internal_precision": "test-native",
            "device": "test-device",
            "device_class": "cpu",
            "placement_evidence": "npz:placement_evidence_bytes",
            "performance_evidence": "npz:performance_evidence_bytes",
            "accelerated_placement": True,
        }
        probe_queries = np.zeros((len(SEQUENCE_BUCKETS), DIMENSIONS), dtype=np.int8)
        probe_relevant = np.zeros_like(probe_queries)
        probe_irrelevant = np.zeros_like(probe_queries)
        probe_queries[:, 0] = 127
        probe_relevant[:, :2] = [127, 64]
        probe_irrelevant[:, 0] = -127
        sequence_probe_arrays = (
            probe_queries,
            probe_relevant,
            probe_irrelevant,
            probe_queries.copy(),
            probe_relevant.copy(),
            probe_irrelevant.copy(),
        )
        sequence_bucket_results = []
        for index, bucket in enumerate(SEQUENCE_BUCKETS):
            semantic_probe = semantic_probe_fixture(bucket)
            for label, array in zip(
                ("query", "relevant_document", "irrelevant_document"),
                (probe_queries, probe_relevant, probe_irrelevant),
                strict=True,
            ):
                semantic_probe[f"{label}_canonical_output_bytes_sha256"] = (
                    hashlib.sha256(
                        np.ascontiguousarray(array[index]).tobytes()
                    ).hexdigest()
                )
            sequence_bucket_results.append(
                {
                    "bucket": bucket,
                    "requested_tokens": bucket,
                    "tokenized_tokens": bucket,
                    "executed_shape_tokens": bucket,
                    "output_dimensions": DIMENSIONS,
                    "finite_output": True,
                    "nonzero_output": True,
                    "truncated": False,
                    "semantic_probe": semantic_probe,
                }
            )
        identity = {
            field: metadata[field]
            for field in (
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
        }
        sequence_evidence = (
            json.dumps(
                {
                    **identity,
                    "supported_max_tokens": MAX_TOKENS,
                    "supported_sequence_buckets": SEQUENCE_BUCKETS,
                    "supported_max_batch_size": SUPPORTED_MAX_BATCH_SIZE,
                    "wire_batch_results": [
                        {
                            "batch_size": batch_size,
                            "input_count": SUPPORTED_MAX_BATCH_SIZE,
                            "request_count": (
                                SUPPORTED_MAX_BATCH_SIZE + batch_size - 1
                            )
                            // batch_size,
                            "response_row_count": SUPPORTED_MAX_BATCH_SIZE,
                            "ordered_input_json_sha256": "6" * 64,
                            "canonical_output_bytes_sha256": "7" * 64,
                            "signed_transactions_sha256": hashlib.sha256(
                                f"wire-transactions-{batch_size}".encode()
                            ).hexdigest(),
                        }
                        for batch_size in range(1, SUPPORTED_MAX_BATCH_SIZE + 1)
                    ],
                    "grouping_invariance": {
                        "batch_sizes": list(
                            range(1, SUPPORTED_MAX_BATCH_SIZE + 1)
                        ),
                        "input_selection": WIRE_BATCH_INPUT_SELECTION,
                        "same_inputs_in_same_order": True,
                        "canonical_output_bytes_equal": True,
                    },
                    "bucket_results": sequence_bucket_results,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        placement_evidence = {
            **identity,
            "accelerated_placement": True,
            "accelerator_execution_confirmed": True,
            "fallback_disclosure_complete": True,
            "unexpected_fallback_detected": False,
            "provider_binding": openvino_provider_binding(),
            "bucket_results": [
                {
                    "bucket": bucket,
                    "accelerator_execution_confirmed": True,
                    "fallback_disclosure_complete": True,
                    "unexpected_fallback_detected": False,
                    "fallback_summary": "none",
                    "profiler_output_sha256": "8" * 64,
                    "provider_evidence": openvino_provider_evidence(),
                }
                for bucket in SEQUENCE_BUCKETS
            ],
        }
        performance_evidence = {
            **identity,
            "bucket_results": [
                {
                    "bucket": bucket,
                    "sample_count": 10,
                    "benchmark_output_sha256": "9" * 64,
                    "latency_ms_p50": 1.0,
                    "latency_ms_p95": 2.0,
                    "peak_memory_bytes": 1024,
                    "energy_measurement": "not_measured",
                    "energy_not_measured_reason": "meter unavailable",
                }
                for bucket in SEQUENCE_BUCKETS
            ],
        }
        evidence = {
            "sequence_capability": sequence_evidence,
            "placement": json.dumps(
                placement_evidence, sort_keys=True, separators=(",", ":")
            ).encode()
            + b"\n",
            "performance": json.dumps(
                performance_evidence, sort_keys=True, separators=(",", ":")
            ).encode()
            + b"\n",
        }
        wire_batch_outputs = np.zeros(
            (SUPPORTED_MAX_BATCH_SIZE, SUPPORTED_MAX_BATCH_SIZE, DIMENSIONS),
            dtype=np.int8,
        )
        wire_batch_outputs[:, :, 0] = 127
        metadata["sequence_capability_evidence_sha256"] = hashlib.sha256(
            evidence["sequence_capability"]
        ).hexdigest()
        metadata["placement_evidence_sha256"] = hashlib.sha256(
            evidence["placement"]
        ).hexdigest()
        metadata["performance_evidence_sha256"] = hashlib.sha256(
            evidence["performance"]
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "adapter.npz"
            write_cache(
                output,
                metadata,
                arrays,
                sequence_probe_arrays,
                wire_batch_outputs,
                evidence,
            )
            with np.load(output, allow_pickle=False) as cached:
                self.assertEqual(json.loads(str(cached["metadata"].item())), metadata)
                for name in (
                    "queries",
                    "documents",
                    "queries_repeat",
                    "documents_repeat",
                ):
                    self.assertEqual(cached[name].dtype, np.dtype(np.int8))
                for name, expected in zip(
                    SEQUENCE_PROBE_ARRAY_NAMES, sequence_probe_arrays, strict=True
                ):
                    self.assertTrue(np.array_equal(cached[name], expected))
            loaded_metadata, loaded_queries, loaded_documents = load_cache(output)
            self.assertEqual(loaded_metadata, metadata)
            self.assertTrue(np.array_equal(loaded_queries, arrays[0]))
            self.assertTrue(np.array_equal(loaded_documents, arrays[1]))

            incomplete = dict(metadata)
            incomplete["supported_max_tokens"] = 256
            incomplete["supported_sequence_buckets"] = [32, 64, 128, 256]
            incomplete_output = Path(directory) / "fixed-seq256.npz"
            write_cache(
                incomplete_output,
                incomplete,
                arrays,
                sequence_probe_arrays,
                wire_batch_outputs,
                evidence,
            )
            with self.assertRaisesRegex(ValueError, "supported_max_tokens"):
                load_cache(incomplete_output)

    def test_http_request_honors_response_index_order(self) -> None:
        class FakeResponse:
            headers: dict[str, str] = {}

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                del args

            def read(self, amount: int = -1) -> bytes:
                body = json.dumps(
                    attested_response(
                        [
                            {"index": 1, "embedding": vector(4.0)},
                            {"index": 0, "embedding": vector(3.0)},
                        ]
                    )
                ).encode("utf-8")
                return body[:amount] if amount >= 0 else body

        class FakeOpener:
            def open(self, request, timeout: float):
                self.request = request
                self.timeout = timeout
                return FakeResponse()

        opener = FakeOpener()
        result = request_embeddings(
            "http://127.0.0.1:1234/embeddings",
            "test-scope",
            [QUERY_PREFIX + "first", QUERY_PREFIX + "second"],
            7.0,
            opener=opener,
        )

        self.assertEqual(result[:, 0].tolist(), [3.0, 4.0])
        request_body = json.loads(opener.request.data)
        self.assertEqual(request_body["dimensions"], DIMENSIONS)
        self.assertEqual(request_body["cfetch_requested_scope_id"], "test-scope")
        self.assertEqual(opener.timeout, 7.0)

    def test_signed_response_rejects_duplicate_keys_and_nonfinite_numbers(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ).hex()
        valid_body = json.dumps(
            attested_response([{"index": 0, "embedding": vector(1.0)}]),
            separators=(",", ":"),
        ).encode()
        hostile_bodies = {
            "duplicate known field": b'{"model":"forged",' + valid_body[1:],
            "nonfinite embedding": valid_body.replace(b"1.0", b"NaN", 1),
        }

        class FakeResponse:
            def __init__(self, body: bytes, signature: str) -> None:
                self.body = body
                self.headers = {"X-Cfetch-Attestation-Signature": signature}

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                del args

            def read(self, amount: int = -1) -> bytes:
                return self.body[:amount] if amount >= 0 else self.body

        class SigningOpener:
            def __init__(self, response_body: bytes) -> None:
                self.response_body = response_body

            def open(self, request, timeout: float):
                del timeout
                headers = {
                    name.lower(): value for name, value in request.header_items()
                }
                nonce = bytes.fromhex(headers["x-cfetch-attestation-nonce"])
                signature = private_key.sign(
                    attestation_message(nonce, request.data, self.response_body)
                ).hex()
                return FakeResponse(self.response_body, signature)

        for name, response_body in hostile_bodies.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "must be UTF-8 JSON"
            ):
                request_embeddings(
                    "http://127.0.0.1:1234/embeddings",
                    "test-scope",
                    [QUERY_PREFIX + "hostile"],
                    7.0,
                    opener=SigningOpener(response_body),
                    attestation_public_key=public_key,
                )


class ExportCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.private_key = Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ).hex()
        self.inputs = [QUERY_PREFIX + "one", QUERY_PREFIX + "two"]
        self.requests: list[list[str]] = []
        self.fail_after: int | None = None
        self.changed_output = False
        self.bad_signature = False

    def open(self, request, timeout: float):
        del timeout
        if self.fail_after is not None and len(self.requests) >= self.fail_after:
            raise RuntimeError("interrupted request")
        texts = json.loads(request.data)["input"]
        self.requests.append(texts)
        headers = {name.lower(): value for name, value in request.header_items()}
        nonce = bytes.fromhex(headers["x-cfetch-attestation-nonce"])
        rows = []
        for index, _text in enumerate(texts):
            values = [1.0] + [0.0] * (DIMENSIONS - 1)
            if self.changed_output:
                values[1] = 0.5
            rows.append({"index": index, "embedding": values})
        payload = attested_response(rows)
        payload["cfetch_execution"]["compatibility_report_sha256"] = "a" * 64
        raw = json.dumps(payload, separators=(",", ":")).encode()
        signature = self.private_key.sign(attestation_message(nonce, request.data, raw)).hex()
        if self.bad_signature:
            signature = "0" * 128
        response = io.BytesIO(raw)
        response.headers = {"X-Cfetch-Attestation-Signature": signature}
        return response

    def request(self, checkpoint, key, texts, nonces):
        return request_embeddings(
            "http://127.0.0.1:1234/embeddings", "test-scope", texts, 1.0,
            opener=self, attestation_public_key=self.public_key,
            expected_compatibility_report_sha256="a" * 64,
            used_attestation_nonces=nonces,
            checkpoint=checkpoint, checkpoint_key=key,
        )

    def trial_requests(self, checkpoint, nonces):
        def factory(trial: str):
            position = 0

            def request(endpoint, scope, texts, timeout, token):
                del endpoint, scope, timeout, token
                nonlocal position
                key = f"{trial}-{position}"
                position += 1
                return self.request(checkpoint, key, texts, nonces)

            return request

        return factory

    def test_checkpoint_is_opt_in_and_rejects_another_process_writer(self) -> None:
        option = next(
            action for action in build_parser()._actions
            if action.dest == "checkpoint_directory"
        )
        self.assertIsNone(option.default)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "checkpoint"
            with ExportCheckpoint(directory, {"scope": "one"}):
                result = subprocess.run(
                    [sys.executable, "-c", (
                        "import sys\nfrom pathlib import Path\n"
                        "from export_adapter_cache import ExportCheckpoint\n"
                        "with ExportCheckpoint(Path(sys.argv[1]), {'scope': 'one'}):\n"
                        "    pass\n"
                    ), str(directory)],
                    cwd=Path(__file__).resolve().parent,
                    capture_output=True, text=True, timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("already has a writer", result.stderr)
            with ExportCheckpoint(directory, {"scope": "one"}):
                pass
            with self.assertRaisesRegex(ValueError, "identity does not match"):
                with ExportCheckpoint(directory, {"scope": "two"}):
                    pass

    def test_checkpoint_identity_binds_scope_evidence_inputs_and_implementation(self) -> None:
        args, _, _, _ = valid_evidence_fixture()
        args.attestation_public_key = self.public_key
        args.sequence_capability_evidence_sha256 = "2" * 64
        args.placement_evidence_sha256 = "3" * 64
        args.performance_evidence_sha256 = "4" * 64
        args.accelerated_placement = True
        args.batch_size = 1
        documents = [DOCUMENT_PREFIX + "document"]
        identity = build_checkpoint_identity(args, self.inputs, documents)
        for field, changed in (
            ("scope_id", "another-scope"), ("artifact_sha256", "8" * 64),
            ("runtime", "another-runtime"), ("compiler", "another-compiler"),
            ("attestation_public_key", "9" * 64),
            ("sequence_capability_evidence_sha256", "a" * 64),
            ("placement_evidence_sha256", "b" * 64),
            ("performance_evidence_sha256", "c" * 64), ("batch_size", 2),
        ):
            with self.subTest(field=field):
                changed_args = copy.copy(args)
                setattr(changed_args, field, changed)
                self.assertNotEqual(
                    identity, build_checkpoint_identity(changed_args, self.inputs, documents)
                )
        self.assertNotEqual(
            identity, build_checkpoint_identity(args, list(reversed(self.inputs)), documents)
        )
        implementation = identity["implementation_sha256"]
        self.assertEqual(
            implementation["export_adapter_cache.py"],
            hashlib.sha256((Path(__file__).parent / "export_adapter_cache.py").read_bytes()).hexdigest(),
        )
        self.assertIn("requirements-lock.txt", implementation)

    def test_interruption_resumes_missing_transactions_and_runs_independent_repeats(self) -> None:
        documents = [DOCUMENT_PREFIX + "one", DOCUMENT_PREFIX + "two"]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "checkpoint"
            self.fail_after = 3
            with ExportCheckpoint(directory, {"scope": self.public_key}) as checkpoint:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    collect_cache_arrays(
                        "http://127.0.0.1/embeddings", "test-scope", self.inputs,
                        documents, 1, 1.0,
                        trial_request_factory=self.trial_requests(checkpoint, set()),
                    )
            self.assertEqual(len(self.requests), 3)
            self.fail_after = None
            with ExportCheckpoint(directory, {"scope": self.public_key}) as checkpoint:
                arrays = collect_cache_arrays(
                    "http://127.0.0.1/embeddings", "test-scope", self.inputs,
                    documents, 1, 1.0,
                    trial_request_factory=self.trial_requests(checkpoint, set()),
                )
            self.assertEqual(len(self.requests), 8)
            self.assertTrue(np.array_equal(arrays[0], arrays[2]))
            self.assertTrue(np.array_equal(arrays[1], arrays[3]))
            self.assertEqual(
                {path.stem for path in directory.glob("*.json")} - {"identity"},
                {f"{trial}-{offset}" for trial in (
                    "queries-first", "documents-first", "queries-repeat", "documents-repeat"
                ) for offset in range(2)},
            )
            self.fail_after = 0
            with ExportCheckpoint(directory, {"scope": self.public_key}) as checkpoint:
                resumed = collect_cache_arrays(
                    "http://127.0.0.1/embeddings", "test-scope", self.inputs,
                    documents, 1, 1.0,
                    trial_request_factory=self.trial_requests(checkpoint, set()),
                )
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(arrays, resumed)))

    def test_resume_reauthenticates_and_rejects_changed_or_copied_transactions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "checkpoint"
            with ExportCheckpoint(directory, {}) as checkpoint:
                self.request(checkpoint, "first", self.inputs, set())
            path = directory / "first.json"
            original = path.read_bytes()
            document = json.loads(original)
            document["signature"] = "0" * 128
            path.write_text(json.dumps(document))
            with ExportCheckpoint(directory, {}) as checkpoint:
                with self.assertRaisesRegex(ValueError, "scope-key signature"):
                    self.request(checkpoint, "first", self.inputs, set())
            self.assertEqual(len(self.requests), 1)
            path.write_bytes(original)
            with ExportCheckpoint(directory, {}) as checkpoint:
                with self.assertRaisesRegex(ValueError, "input does not match"):
                    self.request(checkpoint, "first", list(reversed(self.inputs)), set())
            (directory / "repeat.json").write_bytes(original)
            with ExportCheckpoint(directory, {}) as checkpoint:
                nonces: set[bytes] = set()
                self.request(checkpoint, "first", self.inputs, nonces)
                with self.assertRaisesRegex(ValueError, "repeated a prior challenge"):
                    self.request(checkpoint, "repeat", self.inputs, nonces)
            path.write_bytes(b'{"schema_version":')
            with ExportCheckpoint(directory, {}) as checkpoint:
                with self.assertRaises(ValueError):
                    self.request(checkpoint, "first", self.inputs, set())
            self.assertEqual(path.read_bytes(), b'{"schema_version":')

    def test_resume_does_not_hide_repeatability_failure(self) -> None:
        documents = [DOCUMENT_PREFIX + "document"]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "checkpoint"
            self.fail_after = 3
            with ExportCheckpoint(directory, {}) as checkpoint:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    collect_cache_arrays(
                        "http://127.0.0.1/embeddings", "test-scope", self.inputs,
                        documents, 1, 1.0,
                        trial_request_factory=self.trial_requests(checkpoint, set()),
                    )
            self.fail_after = None
            self.changed_output = True
            with ExportCheckpoint(directory, {}) as checkpoint:
                with self.assertRaisesRegex(ValueError, "not byte-repeatable"):
                    collect_cache_arrays(
                        "http://127.0.0.1/embeddings", "test-scope", self.inputs,
                        documents, 1, 1.0,
                        trial_request_factory=self.trial_requests(checkpoint, set()),
                    )
            self.assertEqual(len(self.requests), 6)

    def test_failed_verification_or_publication_never_creates_a_completed_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "checkpoint"
            with ExportCheckpoint(directory, {}) as checkpoint:
                self.bad_signature = True
                with self.assertRaisesRegex(ValueError, "scope-key signature"):
                    self.request(checkpoint, "first", self.inputs, set())
                self.assertFalse((directory / "first.json").exists())
                self.bad_signature = False
                with patch("export_adapter_cache.os.link", side_effect=OSError("interrupted publish")):
                    with self.assertRaisesRegex(OSError, "interrupted publish"):
                        self.request(checkpoint, "first", self.inputs, set())
                self.assertFalse((directory / "first.json").exists())
                self.request(checkpoint, "first", self.inputs, set())
                original = (directory / "first.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    checkpoint._publish(directory / "first.json", b"changed")
                self.assertEqual((directory / "first.json").read_bytes(), original)

    @unittest.skipIf(os.name == "nt", "Windows symlink creation requires additional privileges")
    def test_checkpoint_rejects_symlinked_directory_and_transactions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = root / "actual"
            actual.mkdir()
            alias = root / "alias"
            alias.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "real directory"):
                with ExportCheckpoint(alias, {}):
                    pass
            with ExportCheckpoint(actual, {}) as checkpoint:
                (actual / "first.json").symlink_to(root / "missing")
                with self.assertRaisesRegex(ValueError, "symlinks"):
                    self.request(checkpoint, "first", self.inputs, set())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX named pipes")
    def test_checkpoint_rejects_fifo_identity_and_transaction_without_blocking(self) -> None:
        for filename in ("identity.json", "first.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "checkpoint"
                directory.mkdir()
                if filename != "identity.json":
                    with ExportCheckpoint(directory, {}):
                        pass
                os.mkfifo(directory / filename)
                result = subprocess.run(
                    [sys.executable, "-c", (
                        "import sys\nfrom pathlib import Path\n"
                        "from export_adapter_cache import ExportCheckpoint\n"
                        "try:\n"
                        "    with ExportCheckpoint(Path(sys.argv[1]), {}) as checkpoint:\n"
                        "        checkpoint.read_transaction('first', b'request')\n"
                        "except ValueError as error:\n"
                        "    if 'not regular' not in str(error):\n"
                        "        raise\n"
                        "else:\n"
                        "    raise RuntimeError('FIFO checkpoint was accepted')\n"
                    ), str(directory)],
                    cwd=Path(__file__).resolve().parent,
                    capture_output=True, text=True, timeout=5,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
