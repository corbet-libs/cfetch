#!/usr/bin/env python3
"""Unit tests for the backend-independent all-pairs admission policy."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cross_backend_eval import (
    ADMISSION_IMPLEMENTATION_BUNDLE_SHA256,
    ADMISSION_POLICY_SHA256,
    ABSOLUTE_MINIMUM,
    DATASET,
    DATASET_REVISION,
    ADVERSARIAL_MIXED_DOCUMENT_SELECTION,
    EXACT_INT8_RANKING,
    EXACT_RANKING_TIE_BREAK,
    EVIDENCE_REPLAY_POLICY,
    ExactI8Scores,
    IMPLEMENTATION_BUNDLE_FILES,
    MAX_TOKENS,
    MODEL,
    MODEL_REVISION,
    PROFILE_ID,
    PROFILE_MANIFEST_SHA256,
    RANKING_SEMANTICS,
    QUALITY_REPLAY_ABS_TOLERANCE,
    SEQUENCE_BUCKETS,
    SEQUENCE_SEMANTIC_FIXTURE_ID,
    SEQUENCE_SEMANTIC_FIXTURE_SHA256,
    SEQUENCE_SEMANTIC_GATE,
    SUPPORTED_MAX_BATCH_SIZE,
    VECTOR_ENCODING,
    adversarial_mixed_document_scores,
    exact_i8_cosine_desc,
    evaluate_admission_gate,
    evaluate_compatibility_gate,
    evaluate_sequence_semantic_gate,
    implementation_bundle_sha256,
    load_cache,
    load_sequence_probe_cache,
    metrics,
    pair_key,
    ranked_document_indices,
    read_bounded_local_file,
    read_content_addressed_report,
    report_backend_metadata,
    sequence_semantic_pair_result,
    sequence_semantic_fixture_sha256,
    sequence_semantic_probe_inputs,
    validate_admission_cache_url,
    validate_admission_report_reference,
    validate_measurement_bundle,
    validate_parent_report_declaration,
    validate_replayed_report,
    validate_report_lineage_edge,
    validate_single_cohort_report_binding,
    validate_stored_admission_report,
    validate_wire_batch_output_cache,
    verify_requirements_lock,
    verify_release_registry,
    write_new_report,
)


LABELS = ["npu-a", "gpu-b", "cpu-c"]
CLASSES = {"npu", "gpu", "cpu"}


def cache_evidence_identity() -> dict[str, object]:
    return {
        "scope_id": "test-scope",
        "transport": "supervised-local",
        "backend": "test-backend",
        "runtime": "test-runtime",
        "compiler": "test-compiler",
        "package_target": "test-target",
        "artifact_source": "test-source@revision/model",
        "artifact_sha256": "1" * 64,
        "internal_precision": "test-native",
        "device": "test-device",
        "device_class": "cpu",
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


def complete_pair_metrics() -> dict[str, dict[str, object]]:
    return {
        pair_key(query_label, document_label): {
            "query_backend": query_label,
            "document_backend": document_label,
            **ABSOLUTE_MINIMUM,
        }
        for query_label in LABELS
        for document_label in LABELS
    }


def complete_mixed_document_metrics() -> dict[str, dict[str, float]]:
    return {label: dict(ABSOLUTE_MINIMUM) for label in LABELS}


def complete_sequence_pair_results() -> dict[str, dict[str, object]]:
    return {
        pair_key(query_label, document_label): {
            "query_backend": query_label,
            "document_backend": document_label,
            "buckets": {
                str(bucket): {
                    "relevant_dot": 1,
                    "relevant_document_norm_sq": 1,
                    "irrelevant_dot": 0,
                    "irrelevant_document_norm_sq": 1,
                }
                for bucket in SEQUENCE_BUCKETS
            },
        }
        for query_label in LABELS
        for document_label in LABELS
    }


def sequence_evidence_bytes(
    queries: np.ndarray,
    relevant_documents: np.ndarray,
    irrelevant_documents: np.ndarray,
    wire_input_digest: str = "5" * 64,
    wire_output_digest: str = "6" * 64,
) -> bytes:
    arrays = (queries, relevant_documents, irrelevant_documents)
    labels = ("query", "relevant_document", "irrelevant_document")
    bucket_results = []
    for index, bucket in enumerate(SEQUENCE_BUCKETS):
        token_count = 1 if index == 0 else SEQUENCE_BUCKETS[index - 1] + 1
        semantic_probe = {
            "fixture_id": SEQUENCE_SEMANTIC_FIXTURE_ID,
            "fixture_sha256": SEQUENCE_SEMANTIC_FIXTURE_SHA256,
            "canonical_repeatability": True,
            "self_relevant_before_irrelevant": True,
        }
        for label, text, array in zip(
            labels, sequence_semantic_probe_inputs(bucket), arrays, strict=True
        ):
            semantic_probe[f"{label}_input_utf8_sha256"] = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()
            semantic_probe[f"{label}_token_count"] = token_count
            semantic_probe[f"{label}_canonical_output_bytes_sha256"] = (
                hashlib.sha256(np.ascontiguousarray(array[index]).tobytes()).hexdigest()
            )
        bucket_results.append(
            {
                "bucket": bucket,
                "requested_tokens": bucket,
                "tokenized_tokens": bucket,
                "executed_shape_tokens": bucket,
                "output_dimensions": 768,
                "finite_output": True,
                "nonzero_output": True,
                "truncated": False,
                "semantic_probe": semantic_probe,
            }
        )
    return (
        json.dumps(
            {
                **cache_evidence_identity(),
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
                        "ordered_input_json_sha256": wire_input_digest,
                        "canonical_output_bytes_sha256": wire_output_digest,
                        "signed_transactions_sha256": hashlib.sha256(
                            f"wire-transactions-{batch_size}".encode()
                        ).hexdigest(),
                    }
                    for batch_size in range(1, SUPPORTED_MAX_BATCH_SIZE + 1)
                ],
                "grouping_invariance": {
                    "batch_sizes": list(range(1, SUPPORTED_MAX_BATCH_SIZE + 1)),
                    "input_selection": (
                        "first-32-pinned-scifact-queries-then-first-32-"
                        "pinned-scifact-documents"
                    ),
                    "same_inputs_in_same_order": True,
                    "canonical_output_bytes_equal": True,
                },
                "bucket_results": bucket_results,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def placement_evidence_bytes() -> bytes:
    report = {
        **cache_evidence_identity(),
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
                "profiler_output_sha256": "7" * 64,
                "provider_evidence": openvino_provider_evidence(),
            }
            for bucket in SEQUENCE_BUCKETS
        ],
    }
    return json.dumps(report, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def performance_evidence_bytes() -> bytes:
    report = {
        **cache_evidence_identity(),
        "bucket_results": [
            {
                "bucket": bucket,
                "sample_count": 10,
                "benchmark_output_sha256": "8" * 64,
                "latency_ms_p50": 1.0,
                "latency_ms_p95": 2.0,
                "peak_memory_bytes": 1024,
                "energy_measurement": "not_measured",
                "energy_not_measured_reason": "meter unavailable",
            }
            for bucket in SEQUENCE_BUCKETS
        ],
    }
    return json.dumps(report, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def complete_admission_report() -> tuple[
    dict[str, object], dict[str, dict[str, object]]
]:
    class_by_scope = dict(zip(LABELS, ("npu", "gpu", "cpu"), strict=True))
    registry_entries: dict[str, dict[str, object]] = {}
    backend_metadata: dict[str, dict[str, object]] = {}
    for offset, scope_id in enumerate(LABELS, start=1):
        digit = str(offset)
        cache_digest = digit * 64
        measurement_digest = str(offset + 3) * 64
        entry = {
            "profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
            "admission_policy_sha256": ADMISSION_POLICY_SHA256,
            "scope_id": scope_id,
            "transport": "supervised-local",
            "backend": f"backend-{scope_id}",
            "runtime": f"runtime-{scope_id}",
            "compiler": f"compiler-{scope_id}",
            "package_target": f"target-{scope_id}",
            "artifact_source": f"source-{scope_id}@revision/model",
            "artifact_sha256": digit * 64,
            "attestation_public_key": chr(ord("a") + offset - 1) * 64,
            "internal_precision": "target-native",
            "device": f"device-{scope_id}",
            "device_class": class_by_scope[scope_id],
            "placement_evidence_sha256": digit * 64,
            "supported_max_tokens": MAX_TOKENS,
            "supported_sequence_buckets": SEQUENCE_BUCKETS,
            "supported_max_batch_size": SUPPORTED_MAX_BATCH_SIZE,
            "sequence_capability_evidence_sha256": digit * 64,
            "performance_evidence_sha256": digit * 64,
            "admission_cache_url": (
                "https://github.com/corbet-libs/cfetch/releases/download/"
                f"admission-v1/{cache_digest}.npz"
            ),
            "admission_cache_sha256": cache_digest,
            "measurement_evidence_url": (
                "https://github.com/corbet-libs/cfetch/releases/download/"
                f"admission-v1/{measurement_digest}.zip"
            ),
            "measurement_evidence_sha256": measurement_digest,
            "compatibility_report": f"release/admission/{'f' * 64}.json",
            "compatibility_report_sha256": "f" * 64,
            "accelerated_placement": True,
        }
        registry_entries[scope_id] = entry
        cache_metadata = {
            "schema_version": 1,
            "profile_id": PROFILE_ID,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "vector_encoding": VECTOR_ENCODING,
            "sequence_semantic_fixture_id": SEQUENCE_SEMANTIC_FIXTURE_ID,
            "sequence_semantic_fixture_sha256": SEQUENCE_SEMANTIC_FIXTURE_SHA256,
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            **entry,
        }
        backend_metadata[scope_id] = report_backend_metadata(cache_metadata)

    pair_results = complete_pair_metrics()
    mixed_results = complete_mixed_document_metrics()
    gate = evaluate_compatibility_gate(
        LABELS,
        set(LABELS),
        CLASSES,
        pair_results,
        mixed_results,
    )
    sequence_results = complete_sequence_pair_results()
    sequence_gate = evaluate_sequence_semantic_gate(
        LABELS, set(LABELS), sequence_results
    )
    report = {
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
        "parent_report": None,
        "parent_report_sha256": None,
        "already_admitted_scopes": [],
        "candidate_scopes": sorted(LABELS),
        "backends": backend_metadata,
        "cache_sha256_by_scope": {
            scope_id: str(offset) * 64
            for offset, scope_id in enumerate(LABELS, start=1)
        },
        "pair_metrics": pair_results,
        "adversarial_mixed_document_metrics": mixed_results,
        "sequence_semantic_pair_results": sequence_results,
        "vector_compatibility_gate": gate,
        "sequence_semantic_gate": sequence_gate,
        "admission_gate": evaluate_admission_gate(gate, sequence_gate),
    }
    return report, registry_entries


class AllPairsGateTests(unittest.TestCase):
    def test_adversarial_mix_uses_min_relevant_and_max_irrelevant_score(self) -> None:
        query = np.asarray([[127, 0]], dtype=np.int8)
        documents = [
            np.asarray([[127, 0], [0, 127]], dtype=np.int8),
            np.asarray([[127, 127], [127, 0]], dtype=np.int8),
        ]

        result = adversarial_mixed_document_scores(
            query, documents, ["q"], ["relevant", "irrelevant"], {"q": {"relevant"}}
        )

        # Relevant takes sqrt(1/2), the minimum; irrelevant takes 1, the maximum.
        self.assertEqual(result.dots.tolist(), [[16129, 16129]])
        self.assertEqual(result.document_norms_sq.tolist(), [[32258, 16129]])
        self.assertEqual(ranked_document_indices(result, 0, 2), [1, 0])

    def test_exact_cosine_comparator_matches_production_sign_branches(self) -> None:
        # Positive beats zero, which beats negative.
        self.assertLess(exact_i8_cosine_desc(1, 1, 0, 1), 0)
        self.assertLess(exact_i8_cosine_desc(0, 1, -1, 1), 0)
        # Positive scores use descending squared ratios; negative scores use
        # ascending squared ratios because a smaller magnitude is less negative.
        self.assertLess(exact_i8_cosine_desc(3, 9, 4, 25), 0)
        self.assertGreater(exact_i8_cosine_desc(-3, 9, -4, 25), 0)

        left = 12_000_000**2 * 12_000_001
        right = 12_000_001**2 * 12_000_000
        self.assertGreater(left, 2**64)
        self.assertLess(left, right)
        self.assertGreater(
            exact_i8_cosine_desc(12_000_000, 12_000_000, 12_000_001, 12_000_001),
            0,
        )

    def test_exact_ranking_ties_by_corpus_insertion_index(self) -> None:
        # Documents 0 and 1 have exactly equal cosine despite different dots
        # and norms. Corpus insertion index stands in for the production block
        # id assigned when this pinned evaluation corpus is inserted.
        exact = ExactI8Scores(
            np.asarray([[1, 2, 0, -1]], dtype=np.int64),
            np.asarray([1, 4, 1, 1], dtype=np.int64),
        )

        self.assertEqual(ranked_document_indices(exact, 0, 4), [0, 1, 2, 3])
        self.assertEqual(
            metrics(
                exact,
                ["q"],
                ["first", "second", "zero", "negative"],
                {"q": {"first"}},
            ),
            {"ndcg_at_10": 1.0, "recall_at_100": 1.0, "mrr_at_10": 1.0},
        )

    def test_noncanonical_int8_records_are_rejected(self) -> None:
        valid_probe_queries = np.zeros((len(SEQUENCE_BUCKETS), 768), dtype=np.int8)
        valid_probe_relevant = np.zeros_like(valid_probe_queries)
        valid_probe_irrelevant = np.zeros_like(valid_probe_queries)
        valid_probe_queries[:, 0] = 127
        valid_probe_relevant[:, 0] = 127
        valid_probe_irrelevant[:, 1] = 127
        wire_batch_outputs = np.zeros((64, 64, 768), dtype=np.int8)
        wire_batch_outputs[:, :, 0] = 127
        wire_output_digest = hashlib.sha256(
            np.ascontiguousarray(wire_batch_outputs[0]).tobytes()
        ).hexdigest()
        sequence_bytes = sequence_evidence_bytes(
            valid_probe_queries,
            valid_probe_relevant,
            valid_probe_irrelevant,
            wire_output_digest=wire_output_digest,
        )
        sequence_digest = hashlib.sha256(sequence_bytes).hexdigest()
        placement_bytes = placement_evidence_bytes()
        placement_digest = hashlib.sha256(placement_bytes).hexdigest()
        performance_bytes = performance_evidence_bytes()
        performance_digest = hashlib.sha256(performance_bytes).hexdigest()
        metadata = {
            "schema_version": 1,
            "profile_id": PROFILE_ID,
            "profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
            "admission_policy_sha256": ADMISSION_POLICY_SHA256,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "vector_encoding": VECTOR_ENCODING,
            "supported_max_tokens": MAX_TOKENS,
            "supported_sequence_buckets": SEQUENCE_BUCKETS,
            "supported_max_batch_size": 64,
            "sequence_semantic_fixture_id": SEQUENCE_SEMANTIC_FIXTURE_ID,
            "sequence_semantic_fixture_sha256": SEQUENCE_SEMANTIC_FIXTURE_SHA256,
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "scope_id": "test-scope",
            "transport": "supervised-local",
            "backend": "test-backend",
            "runtime": "test-runtime",
            "compiler": "test-compiler",
            "package_target": "test-target",
            "artifact_source": "test-source@revision/model",
            "artifact_sha256": "1" * 64,
            "attestation_public_key": "a" * 64,
            "internal_precision": "test-native",
            "sequence_capability_evidence": "sequence.json",
            "sequence_capability_evidence_sha256": sequence_digest,
            "device": "test-device",
            "device_class": "cpu",
            "placement_evidence": "placement.json",
            "placement_evidence_sha256": placement_digest,
            "performance_evidence": "performance.json",
            "performance_evidence_sha256": performance_digest,
            "accelerated_placement": True,
        }
        valid = np.zeros((1, 768), dtype=np.int8)
        valid[0, 0] = 127

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"

            def write_cache(
                cache_metadata: dict[str, object] | str,
                queries: np.ndarray,
                cache_placement_bytes: bytes = placement_bytes,
                cache_performance_bytes: bytes = performance_bytes,
                cache_sequence_bytes: bytes = sequence_bytes,
                cache_wire_batch_outputs: np.ndarray = wire_batch_outputs,
            ) -> None:
                serialized_metadata = (
                    cache_metadata
                    if isinstance(cache_metadata, str)
                    else json.dumps(cache_metadata, sort_keys=True)
                )
                np.savez_compressed(
                    path,
                    metadata=serialized_metadata,
                    queries=queries,
                    documents=valid,
                    queries_repeat=queries,
                    documents_repeat=valid,
                    sequence_capability_evidence_bytes=np.frombuffer(
                        cache_sequence_bytes, dtype=np.uint8
                    ),
                    placement_evidence_bytes=np.frombuffer(
                        cache_placement_bytes, dtype=np.uint8
                    ),
                    performance_evidence_bytes=np.frombuffer(
                        cache_performance_bytes, dtype=np.uint8
                    ),
                    sequence_probe_queries=valid_probe_queries,
                    sequence_probe_relevant_documents=valid_probe_relevant,
                    sequence_probe_irrelevant_documents=valid_probe_irrelevant,
                    sequence_probe_queries_repeat=valid_probe_queries,
                    sequence_probe_relevant_documents_repeat=valid_probe_relevant,
                    sequence_probe_irrelevant_documents_repeat=valid_probe_irrelevant,
                    wire_batch_outputs=cache_wire_batch_outputs,
                )

            cases = [
                (
                    "forbidden -128",
                    np.asarray([[-128, 127] + [0] * 766], dtype=np.int8),
                    "forbidden.*-128",
                ),
                (
                    "missing codec extremum",
                    np.asarray([[126, -126] + [0] * 766], dtype=np.int8),
                    r"-127 or \+127",
                ),
            ]
            for name, invalid, error in cases:
                with self.subTest(name=name):
                    write_cache(metadata, invalid)
                    with self.assertRaisesRegex(ValueError, error):
                        load_cache(path)

            wrong_width = np.zeros((1, 769), dtype=np.int8)
            wrong_width[0, 0] = 127
            write_cache(metadata, wrong_width)
            with self.assertRaisesRegex(ValueError, "invalid bounded shape or dtype"):
                load_cache(path)

            write_cache(metadata, valid)
            validate_wire_batch_output_cache(path, "5" * 64)
            with self.assertRaisesRegex(ValueError, "ordered-input digest"):
                validate_wire_batch_output_cache(path, "6" * 64)
            loaded_probes = load_sequence_probe_cache(path)
            self.assertTrue(
                all(
                    np.array_equal(actual, expected)
                    for actual, expected in zip(
                        loaded_probes,
                        (
                            valid_probe_queries,
                            valid_probe_relevant,
                            valid_probe_irrelevant,
                        ),
                        strict=True,
                    )
                )
            )

            with zipfile.ZipFile(path) as archive:
                canonical_members = {
                    member.filename: archive.read(member)
                    for member in archive.infolist()
                }
            for member_name, claimed_shape in (
                ("queries.npy", (10**12, 768)),
                ("wire_batch_outputs.npy", (10**12, 64, 768)),
            ):
                with self.subTest(hostile_npy_member=member_name):
                    forged_npy = io.BytesIO()
                    np.lib.format.write_array_header_1_0(
                        forged_npy,
                        {
                            "descr": np.dtype(np.int8).str,
                            "fortran_order": False,
                            "shape": claimed_shape,
                        },
                    )
                    forged_npy.write(b"\0")
                    forged_members = dict(canonical_members)
                    forged_members[member_name] = forged_npy.getvalue()
                    with zipfile.ZipFile(
                        path, "w", zipfile.ZIP_DEFLATED
                    ) as archive:
                        for cached_name, member_bytes in forged_members.items():
                            archive.writestr(cached_name, member_bytes)
                    with self.assertRaisesRegex(ValueError, "payload size mismatches"):
                        load_cache(path)

            write_cache(metadata, valid)

            changed_wire_outputs = wire_batch_outputs.copy()
            changed_wire_outputs[1, 0, 0] = -127
            write_cache(
                metadata,
                valid,
                cache_wire_batch_outputs=changed_wire_outputs,
            )
            with self.assertRaisesRegex(ValueError, "not byte-identical"):
                validate_wire_batch_output_cache(path, "5" * 64)

            changed_sequence_report = json.loads(sequence_bytes)
            for result in changed_sequence_report["wire_batch_results"]:
                result["canonical_output_bytes_sha256"] = "0" * 64
            changed_sequence_bytes = json.dumps(
                changed_sequence_report, sort_keys=True, separators=(",", ":")
            ).encode() + b"\n"
            changed_sequence_metadata = dict(metadata)
            changed_sequence_metadata["sequence_capability_evidence_sha256"] = (
                hashlib.sha256(changed_sequence_bytes).hexdigest()
            )
            write_cache(
                changed_sequence_metadata,
                valid,
                cache_sequence_bytes=changed_sequence_bytes,
            )
            with self.assertRaisesRegex(ValueError, "output digest"):
                validate_wire_batch_output_cache(path, "5" * 64)

            empty_evidence = b"{}\n"
            empty_placement_metadata = dict(metadata)
            empty_placement_metadata["placement_evidence_sha256"] = hashlib.sha256(
                empty_evidence
            ).hexdigest()
            write_cache(
                empty_placement_metadata,
                valid,
                cache_placement_bytes=empty_evidence,
            )
            with self.assertRaisesRegex(ValueError, "placement evidence scope_id"):
                load_cache(path)

            duplicate_key_evidence = (
                b'{"scope_id":"test-scope","scope_id":"forged"}\n'
            )
            duplicate_key_metadata = dict(metadata)
            duplicate_key_metadata["placement_evidence_sha256"] = hashlib.sha256(
                duplicate_key_evidence
            ).hexdigest()
            write_cache(
                duplicate_key_metadata,
                valid,
                cache_placement_bytes=duplicate_key_evidence,
            )
            with self.assertRaisesRegex(ValueError, "must be UTF-8 JSON"):
                load_cache(path)

            duplicate_metadata = json.dumps(metadata, sort_keys=True)
            duplicate_metadata = duplicate_metadata.replace(
                "{", '{"scope_id":"forged",', 1
            )
            write_cache(duplicate_metadata, valid)
            with self.assertRaisesRegex(ValueError, "must be UTF-8 JSON"):
                load_cache(path)

            for name, public_key, error in (
                ("missing attestation key", None, "attestation_public_key.*non-empty"),
                (
                    "noncanonical attestation key",
                    "A" * 64,
                    "attestation_public_key.*lowercase hexadecimal Ed25519",
                ),
            ):
                with self.subTest(name=name):
                    invalid_metadata = dict(metadata)
                    if public_key is None:
                        del invalid_metadata["attestation_public_key"]
                    else:
                        invalid_metadata["attestation_public_key"] = public_key
                    write_cache(invalid_metadata, valid)
                    with self.assertRaisesRegex(ValueError, error):
                        load_cache(path)

            invalid_transport = dict(metadata)
            invalid_transport["transport"] = "loopback"
            write_cache(invalid_transport, valid)
            with self.assertRaisesRegex(ValueError, "transport"):
                load_cache(path)

    def test_python_tools_and_release_registry_share_one_profile_identity(self) -> None:
        from export_adapter_cache import (
            ADMISSION_POLICY_SHA256 as EXPORT_ADMISSION_POLICY_SHA256,
            MODEL as EXPORT_MODEL,
            MODEL_REVISION as EXPORT_MODEL_REVISION,
            PROFILE_ID as EXPORT_PROFILE_ID,
            PROFILE_MANIFEST_SHA256 as EXPORT_PROFILE_MANIFEST_SHA256,
            SEQUENCE_SEMANTIC_FIXTURE_ID as EXPORT_SEQUENCE_FIXTURE_ID,
            SEQUENCE_SEMANTIC_FIXTURE_SHA256 as EXPORT_SEQUENCE_FIXTURE_SHA256,
            sequence_semantic_fixture_sha256 as export_sequence_fixture_sha256,
        )

        registry_path = Path(__file__).resolve().parents[2] / "release/inference-backends.json"
        registry = json.loads(registry_path.read_text())

        self.assertEqual(EXPORT_PROFILE_ID, PROFILE_ID)
        self.assertEqual(EXPORT_PROFILE_MANIFEST_SHA256, PROFILE_MANIFEST_SHA256)
        self.assertEqual(EXPORT_ADMISSION_POLICY_SHA256, ADMISSION_POLICY_SHA256)
        self.assertEqual(EXPORT_MODEL, MODEL)
        self.assertEqual(EXPORT_MODEL_REVISION, MODEL_REVISION)
        self.assertEqual(EXPORT_SEQUENCE_FIXTURE_ID, SEQUENCE_SEMANTIC_FIXTURE_ID)
        self.assertEqual(
            EXPORT_SEQUENCE_FIXTURE_SHA256, SEQUENCE_SEMANTIC_FIXTURE_SHA256
        )
        self.assertEqual(
            export_sequence_fixture_sha256(), SEQUENCE_SEMANTIC_FIXTURE_SHA256
        )
        self.assertEqual(
            sequence_semantic_fixture_sha256(), SEQUENCE_SEMANTIC_FIXTURE_SHA256
        )
        self.assertEqual(registry["profile_id"], PROFILE_ID)
        self.assertEqual(
            registry["shared_identity"]["profile_manifest_sha256"],
            PROFILE_MANIFEST_SHA256,
        )
        self.assertEqual(
            registry["admission"]["policy_manifest_sha256"],
            ADMISSION_POLICY_SHA256,
        )
        self.assertEqual(
            registry["admission"]["implementation_bundle_sha256"],
            ADMISSION_IMPLEMENTATION_BUNDLE_SHA256,
        )
        self.assertEqual(registry["model_candidate"]["source"], MODEL)
        self.assertEqual(registry["model_candidate"]["revision"], MODEL_REVISION)
        self.assertEqual(
            registry["decision_priority"],
            ["compatibility", "quality", "efficiency"],
        )
        self.assertEqual(registry["admission"]["dataset"], "mteb/scifact")
        self.assertEqual(
            registry["admission"]["dataset_revision"],
            "cf10ab6856b15b0e670ef8ae5dae4e266c12d035",
        )
        self.assertEqual(
            registry["admission"]["sequence_semantic_fixture"],
            {
                "id": SEQUENCE_SEMANTIC_FIXTURE_ID,
                "sha256": SEQUENCE_SEMANTIC_FIXTURE_SHA256,
                "buckets": SEQUENCE_BUCKETS,
            },
        )
        self.assertEqual(
            registry["admission"]["sequence_semantic_gate"],
            SEQUENCE_SEMANTIC_GATE,
        )
        self.assertEqual(
            registry["admission"]["quality_gate"],
            "global-ordered-all-pairs-plus-adversarial-mixed-document-plus-"
            "per-bucket-semantic-ranking",
        )
        self.assertEqual(
            registry["admission"]["absolute_minimums"], ABSOLUTE_MINIMUM
        )
        self.assertEqual(registry["admission"]["ranking"], EXACT_INT8_RANKING)
        self.assertEqual(
            registry["admission"]["tie_break"], EXACT_RANKING_TIE_BREAK
        )
        self.assertEqual(
            registry["admission"]["mixed_document_store"],
            ADVERSARIAL_MIXED_DOCUMENT_SELECTION,
        )
        self.assertEqual(
            registry["admission"]["evidence_replay"], EVIDENCE_REPLAY_POLICY
        )
        self.assertEqual(
            registry["admission"]["wire_batch_contract"],
            "one-execution-scope-per-response-supports-1-through-64-items-"
            "canonical-output-invariant-for-same-64-ordered-inputs-under-"
            "every-grouping-size-1-through-64",
        )
        self.assertIn(
            "wire-batch-1-through-64-grouping-invariance",
            registry["admission"]["required_evidence"],
        )

    def test_exact_admission_implementation_bundle_matches_registry(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        self.assertEqual(
            IMPLEMENTATION_BUNDLE_FILES,
            tuple(sorted(IMPLEMENTATION_BUNDLE_FILES)),
        )
        self.assertEqual(len(IMPLEMENTATION_BUNDLE_FILES), 13)
        self.assertIn(
            "experiments/embedding-profile/physical_checkpoint.py",
            IMPLEMENTATION_BUNDLE_FILES,
        )
        self.assertIn(
            "packages/openvino/package_inventory.py", IMPLEMENTATION_BUNDLE_FILES
        )
        self.assertIn("scripts/apply_admission_activation.py", IMPLEMENTATION_BUNDLE_FILES)
        computed = implementation_bundle_sha256(repository)
        registry = json.loads(
            (repository / "release/inference-backends.json").read_text()
        )

        self.assertEqual(computed, ADMISSION_IMPLEMENTATION_BUNDLE_SHA256)
        self.assertEqual(
            registry["admission"]["implementation_bundle_sha256"], computed
        )

    def test_admission_environment_is_fully_hashed_and_covers_direct_pins(self) -> None:
        result = verify_requirements_lock()
        self.assertEqual(result["direct_packages"], 3)
        self.assertGreater(result["locked_packages"], result["direct_packages"])

    def test_registry_replay_is_a_clean_no_op_for_an_empty_registry(self) -> None:
        with patch("cross_backend_eval.admitted_scopes", return_value={}):
            self.assertEqual(
                verify_release_registry(),
                {
                    "admitted_scopes": 0,
                    "measurement_bundles_verified": 0,
                    "reports_replayed": 0,
                    "status": "empty-no-op",
                },
            )

    def test_admitted_cohort_cannot_split_across_compatibility_reports(self) -> None:
        _, registry_entries = complete_admission_report()
        validate_single_cohort_report_binding(registry_entries)

        split_registry = copy.deepcopy(registry_entries)
        split_registry["cpu-c"]["compatibility_report"] = (
            f"release/admission/{'e' * 64}.json"
        )
        split_registry["cpu-c"]["compatibility_report_sha256"] = "e" * 64
        with self.assertRaisesRegex(
            ValueError,
            "exactly the same compatibility report path and sha256",
        ):
            validate_single_cohort_report_binding(split_registry)

    def test_report_lineage_rejects_false_genesis_reclassification_and_bad_parent(self) -> None:
        genesis, registry_entries = complete_admission_report()

        false_genesis = copy.deepcopy(genesis)
        false_genesis["already_admitted_scopes"] = ["cpu-c"]
        false_genesis["candidate_scopes"] = ["gpu-b", "npu-a"]
        with self.assertRaisesRegex(ValueError, "genesis.*no already-admitted"):
            validate_stored_admission_report(false_genesis, registry_entries)

        reclassified_child = {
            "already_admitted_scopes": [],
            "candidate_scopes": ["cpu-c", "gpu-b"],
        }
        parent = {
            "already_admitted_scopes": [],
            "candidate_scopes": ["cpu-c"],
        }
        with self.assertRaisesRegex(ValueError, "must equal the parent report"):
            validate_report_lineage_edge(reclassified_child, parent)

        half_parent = copy.deepcopy(genesis)
        half_parent["parent_report"] = f"release/admission/{'e' * 64}.json"
        with self.assertRaisesRegex(ValueError, "both be null or both be strings"):
            validate_parent_report_declaration(half_parent, set())

        with self.assertRaisesRegex(ValueError, "content-addressed"):
            validate_admission_report_reference(
                f"release/admission/{'e' * 64}.json", "f" * 64
            )

        with tempfile.TemporaryDirectory() as directory:
            wrong_parent = Path(directory) / "parent.json"
            wrong_parent.write_bytes(b"{}\n")
            with patch(
                "cross_backend_eval.admission_report_path",
                return_value=wrong_parent,
            ), self.assertRaisesRegex(ValueError, "bytes do not match"):
                read_content_addressed_report(
                    f"release/admission/{'f' * 64}.json", "f" * 64
                )

    def test_report_schema_rejects_unknown_local_payload_and_bounded_read_is_strict(self) -> None:
        report, registry_entries = complete_admission_report()
        report["local_cache_path"] = "/home/operator/private-cache.npz"
        with self.assertRaisesRegex(ValueError, "canonical public fields"):
            validate_stored_admission_report(report, registry_entries)

        with tempfile.TemporaryDirectory() as directory:
            oversized = Path(directory) / "oversized.json"
            oversized.write_bytes(b"x" * 17)
            with self.assertRaisesRegex(ValueError, "must be 1..16 bytes"):
                read_bounded_local_file(oversized, 16, "compatibility report")

            report_path = Path(directory) / "immutable-report.json"
            write_new_report(report_path, "{}\n")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                write_new_report(report_path, "forged\n")
            self.assertEqual(report_path.read_text(), "{}\n")

    def test_stored_report_binds_every_backend_to_the_release_cohort(self) -> None:
        report, registry_entries = complete_admission_report()
        validate_stored_admission_report(report, registry_entries)

        for scope_id in LABELS:
            for field, replacement in (
                ("runtime", "different-runtime"),
                ("attestation_public_key", "f" * 64),
            ):
                with self.subTest(scope_id=scope_id, field=field):
                    tampered = copy.deepcopy(report)
                    tampered["backends"][scope_id][field] = replacement
                    with self.assertRaisesRegex(ValueError, "release registry"):
                        validate_stored_admission_report(tampered, registry_entries)

        missing_scope = copy.deepcopy(report)
        missing_scope["candidate_scopes"] = sorted(LABELS[:-1])
        with self.assertRaisesRegex(ValueError, "release cohort"):
            validate_stored_admission_report(missing_scope, registry_entries)

        duplicate_key_entries = copy.deepcopy(registry_entries)
        duplicate_key_entries["cpu-c"]["attestation_public_key"] = (
            duplicate_key_entries["npu-a"]["attestation_public_key"]
        )
        with self.assertRaisesRegex(ValueError, "unique Ed25519"):
            validate_stored_admission_report(report, duplicate_key_entries)

        wrong_cache_digest = copy.deepcopy(registry_entries)
        wrong_cache_digest["npu-a"]["admission_cache_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "cache digest.*release registry"):
            validate_stored_admission_report(report, wrong_cache_digest)

    def test_admission_cache_locator_is_content_addressed_and_repo_scoped(self) -> None:
        digest = "a" * 64
        validate_admission_cache_url(
            "test-scope",
            "https://github.com/corbet-libs/cfetch/releases/download/"
            f"admission-v1/{digest}.npz",
            digest,
        )
        for url in (
            "https://github.com/corbet-labs/cfetch/releases/download/"
            f"admission-v1/{digest}.npz",
            f"https://example.com/{digest}.npz",
            "http://github.com/corbet-libs/cfetch/releases/download/"
            f"admission-v1/{digest}.npz",
            "https://github.com/corbet-libs/cfetch/releases/download/"
            f"admission-v1/{'b' * 64}.npz",
        ):
            with self.subTest(url=url), self.assertRaisesRegex(
                ValueError, "content-addressed cfetch GitHub release URL"
            ):
                validate_admission_cache_url("test-scope", url, digest)

    def test_measurement_bundle_retains_every_raw_signed_wire_profiler_and_benchmark(
        self,
    ) -> None:
        profiler_bytes = b"raw profiler output\n"
        benchmark_bytes = b"raw benchmark output\n"
        profiler_digest = hashlib.sha256(profiler_bytes).hexdigest()
        benchmark_digest = hashlib.sha256(benchmark_bytes).hexdigest()
        probe_vectors = np.ones((len(SEQUENCE_BUCKETS), 768), dtype=np.int8)
        sequence = json.loads(
            sequence_evidence_bytes(probe_vectors, probe_vectors, probe_vectors)
        )
        wire_files = {
            hashlib.sha256(
                f"wire-transactions-{batch_size}".encode()
            ).hexdigest(): f"wire-transactions-{batch_size}".encode()
            for batch_size in range(1, SUPPORTED_MAX_BATCH_SIZE + 1)
        }
        sequence_digest = hashlib.sha256(
            json.dumps(sequence, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        placement = json.loads(placement_evidence_bytes())
        performance = json.loads(performance_evidence_bytes())
        for row in placement["bucket_results"]:
            row["profiler_output_sha256"] = profiler_digest
        for row in performance["bucket_results"]:
            row["benchmark_output_sha256"] = benchmark_digest
        placement_digest = hashlib.sha256(
            json.dumps(placement, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        performance_digest = hashlib.sha256(
            json.dumps(performance, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        entry = {
            "sequence_capability_evidence_sha256": sequence_digest,
            "placement_evidence_sha256": placement_digest,
            "performance_evidence_sha256": performance_digest,
        }
        manifest = {
            "schema_version": 1,
            "scope_id": "test-scope",
            "sequence_capability_evidence_sha256": sequence_digest,
            "placement_evidence_sha256": placement_digest,
            "performance_evidence_sha256": performance_digest,
            "files": [
                {
                    "path": f"raw/{benchmark_digest}.bin",
                    "sha256": benchmark_digest,
                    "roles": ["performance-benchmark"],
                },
                {
                    "path": f"raw/{profiler_digest}.bin",
                    "sha256": profiler_digest,
                    "roles": ["placement-profiler"],
                },
                *[
                    {
                        "path": f"raw/{digest}.bin",
                        "sha256": digest,
                        "roles": ["wire-signed-transactions"],
                    }
                    for digest in sorted(wire_files)
                ],
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "measurements.zip"

            def write_bundle(
                manifest_bytes: bytes,
                include_profiler: bool = True,
                omitted_wire_digest: str | None = None,
            ) -> None:
                with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr("measurement-manifest.json", manifest_bytes)
                    archive.writestr(
                        f"raw/{benchmark_digest}.bin", benchmark_bytes
                    )
                    for digest, raw in wire_files.items():
                        if digest != omitted_wire_digest:
                            archive.writestr(f"raw/{digest}.bin", raw)
                    if include_profiler:
                        archive.writestr(
                            f"raw/{profiler_digest}.bin", profiler_bytes
                        )

            manifest_bytes = json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode()
            write_bundle(manifest_bytes)
            validate_measurement_bundle(
                bundle,
                "test-scope",
                entry,
                {
                    "sequence": sequence,
                    "placement": placement,
                    "performance": performance,
                },
            )

            write_bundle(manifest_bytes, include_profiler=False)
            with self.assertRaisesRegex(ValueError, "retain every referenced raw output"):
                validate_measurement_bundle(
                    bundle,
                    "test-scope",
                    entry,
                    {
                        "sequence": sequence,
                        "placement": placement,
                        "performance": performance,
                    },
                )

            write_bundle(
                manifest_bytes,
                omitted_wire_digest=next(iter(wire_files)),
            )
            with self.assertRaisesRegex(ValueError, "retain every referenced raw output"):
                validate_measurement_bundle(
                    bundle,
                    "test-scope",
                    entry,
                    {
                        "sequence": sequence,
                        "placement": placement,
                        "performance": performance,
                    },
                )

            duplicate_manifest = manifest_bytes.replace(
                b'{"files":', b'{"scope_id":"forged","files":', 1
            )
            write_bundle(duplicate_manifest)
            with self.assertRaisesRegex(ValueError, "must be UTF-8 JSON"):
                validate_measurement_bundle(
                    bundle,
                    "test-scope",
                    entry,
                    {
                        "sequence": sequence,
                        "placement": placement,
                        "performance": performance,
                    },
                )

    def test_replay_tolerates_only_last_bit_quality_metric_drift(self) -> None:
        replayed, _ = complete_admission_report()
        stored = copy.deepcopy(replayed)
        pair = pair_key("npu-a", "gpu-b")
        stored["pair_metrics"][pair]["ndcg_at_10"] += (
            QUALITY_REPLAY_ABS_TOLERANCE / 2
        )
        validate_replayed_report(stored, replayed, Path("report.json"))

        stored["pair_metrics"][pair]["ndcg_at_10"] += (
            QUALITY_REPLAY_ABS_TOLERANCE * 2
        )
        with self.assertRaisesRegex(ValueError, "ndcg_at_10 does not match replay"):
            validate_replayed_report(stored, replayed, Path("report.json"))

        tampered_gate = copy.deepcopy(replayed)
        tampered_gate["admission_gate"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "full decision replay"):
            validate_replayed_report(tampered_gate, replayed, Path("report.json"))

    def test_report_backend_projection_cannot_leak_cache_paths_or_extras(self) -> None:
        report, registry_entries = complete_admission_report()
        metadata = dict(report["backends"]["npu-a"])
        metadata.update(
            {
                "sequence_capability_evidence": "/home/operator/sequence.json",
                "placement_evidence": "/home/operator/placement.json",
                "performance_evidence": "/home/operator/performance.json",
                "private_note": "host-local detail",
            }
        )

        projected = report_backend_metadata(metadata)

        self.assertNotIn("sequence_capability_evidence", projected)
        self.assertNotIn("placement_evidence", projected)
        self.assertNotIn("performance_evidence", projected)
        self.assertNotIn("private_note", projected)
        self.assertNotIn("/home/", json.dumps(projected, sort_keys=True))

        report["backends"]["npu-a"] = metadata
        with self.assertRaisesRegex(ValueError, "canonical public admission fields"):
            validate_stored_admission_report(report, registry_entries)

    def test_stored_report_recomputes_gate_structure_checks_and_counts(self) -> None:
        report, registry_entries = complete_admission_report()
        pair = pair_key("npu-a", "gpu-b")
        tampered_reports = []

        wrong_passed = copy.deepcopy(report)
        wrong_passed["vector_compatibility_gate"]["passed"] = False
        tampered_reports.append(("passed", wrong_passed))

        wrong_count = copy.deepcopy(report)
        wrong_count["vector_compatibility_gate"]["expected_ordered_pair_count"] += 1
        tampered_reports.append(("count", wrong_count))

        wrong_check = copy.deepcopy(report)
        wrong_check["vector_compatibility_gate"]["pair_checks"][pair][
            "ndcg_at_10"
        ] = False
        tampered_reports.append(("check", wrong_check))

        wrong_structure = copy.deepcopy(report)
        del wrong_structure["vector_compatibility_gate"]["missing_ordered_pairs"]
        tampered_reports.append(("structure", wrong_structure))

        for name, tampered in tampered_reports:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError, "gate structure/checks/counts"
                ):
                    validate_stored_admission_report(tampered, registry_entries)

    def test_stored_report_cannot_admit_recomputed_failing_metrics(self) -> None:
        report, registry_entries = complete_admission_report()
        pair = pair_key("gpu-b", "npu-a")
        report["pair_metrics"][pair]["ndcg_at_10"] = (
            ABSOLUTE_MINIMUM["ndcg_at_10"] - 0.000001
        )
        report["vector_compatibility_gate"] = evaluate_compatibility_gate(
            LABELS,
            set(LABELS),
            CLASSES,
            report["pair_metrics"],
            report["adversarial_mixed_document_metrics"],
        )
        report["admission_gate"] = evaluate_admission_gate(
            report["vector_compatibility_gate"], report["sequence_semantic_gate"]
        )

        with self.assertRaisesRegex(ValueError, "does not pass"):
            validate_stored_admission_report(report, registry_entries)

    def test_every_bucket_uses_exact_ordered_cross_scope_semantic_ranking(self) -> None:
        query = np.zeros((len(SEQUENCE_BUCKETS), 768), dtype=np.int8)
        relevant = np.zeros_like(query)
        irrelevant = np.zeros_like(query)
        query[:, 0] = 127
        relevant[:, 0] = 127
        irrelevant[:, 1] = 127
        scopes = ["npu-a", "gpu-b"]
        results = {
            pair_key(query_scope, document_scope): {
                "query_backend": query_scope,
                "document_backend": document_scope,
                **sequence_semantic_pair_result(query, relevant, irrelevant),
            }
            for query_scope in scopes
            for document_scope in scopes
        }

        passing = evaluate_sequence_semantic_gate(
            scopes, set(scopes), results
        )
        self.assertTrue(passing["passed"])
        self.assertEqual(passing["expected_ordered_pair_count"], 4)
        self.assertEqual(
            passing["evaluated_bucket_check_count"], 4 * len(SEQUENCE_BUCKETS)
        )
        self.assertEqual(
            passing["evaluated_adversarial_mixed_document_check_count"],
            2 * len(SEQUENCE_BUCKETS),
        )
        self.assertTrue(
            passing["all_adversarial_mixed_document_checks_evaluated"]
        )
        self.assertTrue(
            passing["pair_bucket_checks"][pair_key("npu-a", "gpu-b")]["2048"]
        )

        tied_long_document = irrelevant.copy()
        tied_long_document[-1] = relevant[-1]
        failing_result = sequence_semantic_pair_result(
            query, relevant, tied_long_document
        )
        failing = evaluate_sequence_semantic_gate(
            ["npu-a"],
            {"npu-a"},
            {
                pair_key("npu-a", "npu-a"): {
                    "query_backend": "npu-a",
                    "document_backend": "npu-a",
                    **failing_result,
                }
            },
        )
        self.assertFalse(failing["pair_bucket_checks"][pair_key("npu-a", "npu-a")]["2048"])
        self.assertFalse(failing["passed"])

    def test_every_bucket_rejects_adversarial_cross_scope_semantic_mix(self) -> None:
        scopes = ["npu-a", "gpu-b"]

        def result(relevant_dot: int, irrelevant_dot: int) -> dict[str, object]:
            return {
                "buckets": {
                    str(bucket): {
                        "relevant_dot": relevant_dot,
                        "relevant_document_norm_sq": 100,
                        "irrelevant_dot": irrelevant_dot,
                        "irrelevant_document_norm_sq": 100,
                    }
                    for bucket in SEQUENCE_BUCKETS
                }
            }

        # Each ordinary pair passes: 6 > 0 and 10 > 8. The derive-once
        # adversarial store must still fail because its relevant minimum (6)
        # is below its irrelevant maximum (8), taken from different scopes.
        pair_results = {
            pair_key(query_scope, "npu-a"): {
                "query_backend": query_scope,
                "document_backend": "npu-a",
                **result(6, 0),
            }
            for query_scope in scopes
        }
        pair_results.update(
            {
                pair_key(query_scope, "gpu-b"): {
                    "query_backend": query_scope,
                    "document_backend": "gpu-b",
                    **result(10, 8),
                }
                for query_scope in scopes
            }
        )

        gate = evaluate_sequence_semantic_gate(
            scopes, set(scopes), pair_results
        )
        self.assertTrue(
            all(
                all(checks.values())
                for checks in gate["pair_bucket_checks"].values()
            )
        )
        self.assertTrue(gate["all_adversarial_mixed_document_checks_evaluated"])
        self.assertFalse(
            gate["adversarial_mixed_document_bucket_checks"]["npu-a"]["2048"]
        )
        self.assertFalse(gate["passed"])

    def test_stored_report_recomputes_and_enforces_sequence_semantic_gate(self) -> None:
        report, registry_entries = complete_admission_report()
        pair = pair_key("gpu-b", "cpu-c")
        report["sequence_semantic_pair_results"][pair]["buckets"]["2048"][
            "irrelevant_dot"
        ] = 2
        report["sequence_semantic_gate"] = evaluate_sequence_semantic_gate(
            LABELS,
            set(LABELS),
            report["sequence_semantic_pair_results"],
        )
        report["admission_gate"] = evaluate_admission_gate(
            report["vector_compatibility_gate"], report["sequence_semantic_gate"]
        )

        with self.assertRaisesRegex(ValueError, "does not pass"):
            validate_stored_admission_report(report, registry_entries)

    def test_stored_report_binds_pair_producer_identities(self) -> None:
        report, registry_entries = complete_admission_report()
        pair = pair_key("cpu-c", "gpu-b")
        report["pair_metrics"][pair]["document_backend"] = "npu-a"

        with self.assertRaisesRegex(ValueError, "wrong producer identities"):
            validate_stored_admission_report(report, registry_entries)

    def test_every_ordered_pair_at_absolute_floor_passes(self) -> None:
        result = evaluate_compatibility_gate(
            LABELS,
            set(LABELS),
            CLASSES,
            complete_pair_metrics(),
            complete_mixed_document_metrics(),
        )

        self.assertTrue(result["all_ordered_pairs_evaluated"])
        self.assertEqual(result["expected_ordered_pair_count"], 9)
        self.assertEqual(result["evaluated_ordered_pair_count"], 9)
        self.assertTrue(result["passed"])

    def test_one_asymmetric_cross_pair_below_floor_fails(self) -> None:
        pair_metrics = complete_pair_metrics()
        failing_key = pair_key("gpu-b", "npu-a")
        reverse_key = pair_key("npu-a", "gpu-b")
        pair_metrics[failing_key]["ndcg_at_10"] = (
            ABSOLUTE_MINIMUM["ndcg_at_10"] - 0.000001
        )

        result = evaluate_compatibility_gate(
            LABELS,
            set(LABELS),
            CLASSES,
            pair_metrics,
            complete_mixed_document_metrics(),
        )

        self.assertFalse(result["pair_checks"][failing_key]["ndcg_at_10"])
        self.assertTrue(result["pair_checks"][reverse_key]["ndcg_at_10"])
        self.assertFalse(result["passed"])

    def test_missing_ordered_pair_cannot_pass(self) -> None:
        pair_metrics = complete_pair_metrics()
        missing_key = pair_key("cpu-c", "gpu-b")
        del pair_metrics[missing_key]

        result = evaluate_compatibility_gate(
            LABELS,
            set(LABELS),
            CLASSES,
            pair_metrics,
            complete_mixed_document_metrics(),
        )

        self.assertEqual(result["missing_ordered_pairs"], [missing_key])
        self.assertFalse(result["all_ordered_pairs_evaluated"])
        self.assertFalse(result["passed"])

    def test_self_pair_uses_the_same_absolute_floor(self) -> None:
        pair_metrics = complete_pair_metrics()
        self_key = pair_key("npu-a", "npu-a")
        pair_metrics[self_key]["mrr_at_10"] = (
            ABSOLUTE_MINIMUM["mrr_at_10"] - 0.000001
        )

        result = evaluate_compatibility_gate(
            LABELS,
            set(LABELS),
            CLASSES,
            pair_metrics,
            complete_mixed_document_metrics(),
        )

        self.assertFalse(result["pair_checks"][self_key]["mrr_at_10"])
        self.assertFalse(result["passed"])

    def test_omitting_an_expected_scope_cannot_pass(self) -> None:
        expected = set(LABELS) | {"already-admitted-d"}
        result = evaluate_compatibility_gate(
            LABELS,
            expected,
            CLASSES,
            complete_pair_metrics(),
            complete_mixed_document_metrics(),
        )

        self.assertEqual(result["missing_scopes"], ["already-admitted-d"])
        self.assertFalse(result["all_expected_scopes_present"])
        self.assertFalse(result["passed"])

    def test_adversarial_mixed_store_below_floor_cannot_pass(self) -> None:
        mixed = complete_mixed_document_metrics()
        mixed["npu-a"]["recall_at_100"] = ABSOLUTE_MINIMUM["recall_at_100"] - 0.001

        result = evaluate_compatibility_gate(
            LABELS, set(LABELS), CLASSES, complete_pair_metrics(), mixed
        )

        self.assertFalse(
            result["adversarial_mixed_document_checks"]["npu-a"]["recall_at_100"]
        )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
