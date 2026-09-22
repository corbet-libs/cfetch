#!/usr/bin/env python3
"""Dependency-light tests for the two-phase admission transaction."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from admission_transaction import (
    ADMISSION_IMPLEMENTATION_BUNDLE_SHA256,
    ADMISSION_POLICY_SHA256,
    EXPECTED_SIGNED_REQUESTS,
    MODEL,
    MODEL_REVISION,
    PROFILE_ID,
    PROFILE_MANIFEST_SHA256,
    _build_package_zip,
    _build_publication_plan,
    _canonical_json,
    _load_and_bind_package_manifest,
    _load_variant_catalog,
    _profile_source_promotion,
    _run_final_conformance_receipt,
    _sha256_bytes,
    _stage_id,
    _validate_dispatcher,
    _validate_package_rows,
    _validate_packaged_dispatcher,
    _walk_package_files,
    activate_transaction,
    file_sha256,
    generate_receipt_attestation_key,
    run_transaction,
    stage_transaction,
)
from packages.openvino import package_inventory


def metadata(scope_id: str, device_class: str) -> dict[str, object]:
    openvino_device = {"npu": "NPU", "gpu": "GPU", "cpu": "CPU"}[device_class]
    return {
        "profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
        "admission_policy_sha256": ADMISSION_POLICY_SHA256,
        "scope_id": scope_id,
        "transport": "supervised-local",
        "backend": "synthetic-runtime",
        "runtime": "synthetic-runtime-1",
        "compiler": "synthetic-compiler-1",
        "package_target": "synthetic-target",
        "artifact_source": "synthetic/exact@1",
        "device_class": device_class,
        "device": f"synthetic-{device_class}",
        "artifact_sha256": "a" * 64,
        "attestation_public_key": ("b" if device_class == "npu" else "c") * 64,
        "internal_precision": "int8",
        "placement_evidence_sha256": "d" * 64,
        "supported_max_tokens": 2048,
        "supported_sequence_buckets": [32, 64, 128, 257, 512, 1024, 2048],
        "supported_max_batch_size": 64,
        "sequence_capability_evidence_sha256": "e" * 64,
        "performance_evidence_sha256": "f" * 64,
        "accelerated_placement": True,
        "openvino_device": openvino_device,
        "openvino_compile_config": {},
        "required_execution_devices": [openvino_device],
        "required_openvino_properties": {
            "FULL_DEVICE_NAME": f"Synthetic {openvino_device}",
            "DEVICE_ARCHITECTURE": f"synthetic-{device_class}",
        },
        "required_host": {
            "system": "Linux",
            "machine": "x86_64",
            "kernel_release": "synthetic-kernel",
            "files": [{"path": "/usr/lib/libsynthetic.so", "sha256": "1" * 64}],
        },
    }


def write_synthetic_package(root: Path, scopes: list[dict[str, object]]) -> Path:
    artifact = root / "artifact"
    artifact.mkdir(parents=True)
    payload = artifact / "model.bin"
    payload.write_bytes(b"exact native artifact\n")
    artifact_document = {
        "schema_version": 1,
        "files": [
            {
                "path": "model.bin",
                "sha256": file_sha256(payload),
                "bytes": payload.stat().st_size,
            }
        ],
    }
    artifact_bytes = _canonical_json(artifact_document)
    artifact_manifest = artifact / "artifact-manifest.json"
    artifact_manifest.write_bytes(artifact_bytes)
    artifact_digest = _sha256_bytes(artifact_bytes)
    for scope in scopes:
        scope["artifact_sha256"] = artifact_digest
        scope["compatibility_report_sha256"] = None
    (root / "adapter.py").write_text("#!/usr/bin/env python3\nprint('adapter')\n")
    os.chmod(root / "adapter.py", 0o755)
    launcher = root / package_inventory.LAUNCHER
    launcher_source = Path(package_inventory.__file__).with_name("launcher.c")
    subprocess.run(
        [
            "cc",
            "-std=c17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(launcher_source),
            "-o",
            str(launcher),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    runtime_dispatcher = root / package_inventory.RUNTIME_DISPATCHER
    runtime_dispatcher.write_bytes(b"#!/bin/sh\nexit 0\n")
    os.chmod(runtime_dispatcher, 0o755)
    launcher_template = launcher.read_bytes()
    placeholder = package_inventory.LAUNCHER_DIGEST_PLACEHOLDER.encode("ascii")
    runtime_manifest = root / "runtime-manifest.json"
    runtime_manifest.write_bytes(
        _canonical_json(
            {
                "launcher": package_inventory.LAUNCHER,
                "launcher_digest_offset": launcher_template.index(placeholder),
                "launcher_template_sha256": _sha256_bytes(launcher_template),
            }
        )
    )
    package_document = {
        "schema_version": 1,
        "package_state": "candidate",
        "profile_id": PROFILE_ID,
        "profile_manifest_sha256": PROFILE_MANIFEST_SHA256,
        "admission_policy_sha256": ADMISSION_POLICY_SHA256,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "artifact_manifest": "artifact/artifact-manifest.json",
        "artifact_manifest_sha256": artifact_digest,
        "runtime_manifest_sha256": file_sha256(runtime_manifest),
        "dependency_versions": {},
        "scopes": scopes,
    }
    package_manifest = root / "package-manifest.json"
    package_manifest.write_bytes(_canonical_json(package_document))
    _inventory, inventory_sha256 = package_inventory.create(root)
    package_inventory.patch_launcher(root, inventory_sha256)
    package_inventory.verify_bound(root, inventory_sha256)
    return package_manifest


def placement_reports_for_package(
    package_manifest: Path,
    scopes: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    document = json.loads(package_manifest.read_bytes())
    probe = json.loads(json.dumps(document))
    probe["package_state"] = "physical-probe"
    for entry in probe["scopes"]:
        entry["placement_evidence_sha256"] = None
        entry["sequence_capability_evidence_sha256"] = None
        entry["performance_evidence_sha256"] = None
        entry["compatibility_report_sha256"] = None
    probe_bytes = _canonical_json(probe)
    probe_digest = _sha256_bytes(probe_bytes)
    root = package_manifest.parent
    inventory_sha256 = file_sha256(root / package_inventory.INVENTORY_NAME)
    probe_projection = package_inventory.project_package_manifest_rebinding(
        root, inventory_sha256, probe_bytes
    )
    result: dict[str, dict[str, dict[str, object]]] = {}
    for scope in scopes:
        provider = {
            "requested_device": scope["openvino_device"],
            "expected_execution_devices": scope["required_execution_devices"],
            "actual_execution_devices": scope["required_execution_devices"],
            "expected_device_properties": scope["required_openvino_properties"],
            "actual_device_properties": scope["required_openvino_properties"],
        }
        result[str(scope["scope_id"])] = {
            "placement": {
                "provider_binding": {
                    "provider": "openvino",
                    "dispatcher_sha256": probe_projection.launcher_sha256,
                    "probe_package_manifest_sha256": probe_digest,
                    "runtime_manifest_sha256": document["runtime_manifest_sha256"],
                    "openvino_compile_config": scope["openvino_compile_config"],
                    "expected_host": scope["required_host"],
                    "actual_host": scope["required_host"],
                },
                "bucket_results": [
                    {"bucket": bucket, "provider_evidence": dict(provider)}
                    for bucket in scope["supported_sequence_buckets"]
                ],
            }
        }
    return result


class PackageStagingTests(unittest.TestCase):
    def test_generates_non_overwriting_receipt_attestation_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.key"
            public_hex = generate_receipt_attestation_key(path)
            self.assertEqual(len(path.read_bytes()), 32)
            self.assertEqual(len(public_hex), 64)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                generate_receipt_attestation_key(path)

    def test_loads_the_exact_release_variant_catalog_container(self) -> None:
        path = Path(__file__).resolve().parents[2] / "release/variants.json"
        catalog = _load_variant_catalog(path, file_sha256(path))
        self.assertEqual(catalog["linux-cfetch-cpu-x86_64"]["backend"], "cpu")

    def test_report_is_injected_before_deterministic_package_hashing(self) -> None:
        with (
            tempfile.TemporaryDirectory() as package_name,
            tempfile.TemporaryDirectory() as first_name,
            tempfile.TemporaryDirectory() as second_name,
        ):
            package_root = Path(package_name)
            scope = metadata("synthetic-npu", "npu")
            write_synthetic_package(package_root, [dict(scope)])
            # The cache binds the artifact-manifest digest, not a mutable graph path.
            artifact_digest = file_sha256(
                package_root / "artifact/artifact-manifest.json"
            )
            scope["artifact_sha256"] = artifact_digest
            loaded = {"synthetic-npu": (scope, None, None)}
            report_digest = "9" * 64
            dispatcher = {
                "binary": package_inventory.LAUNCHER,
                "sha256": file_sha256(
                    package_root / package_inventory.LAUNCHER
                ),
            }
            reports = placement_reports_for_package(
                package_root / "package-manifest.json", [scope]
            )

            manifest_name, _, final_manifest, projection = _load_and_bind_package_manifest(
                package_root,
                "package-manifest.json",
                ["synthetic-npu"],
                loaded,
                reports,
                dispatcher,
                report_digest,
            )
            first, _ = _build_package_zip(
                package_root,
                manifest_name,
                final_manifest,
                Path(first_name),
                {
                    package_inventory.INVENTORY_NAME: projection.inventory_bytes,
                    package_inventory.LAUNCHER: projection.launcher_bytes,
                },
            )
            second, _ = _build_package_zip(
                package_root,
                manifest_name,
                final_manifest,
                Path(second_name),
                {
                    package_inventory.INVENTORY_NAME: projection.inventory_bytes,
                    package_inventory.LAUNCHER: projection.launcher_bytes,
                },
            )
            self.assertEqual(first.name, second.name)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            extracted = Path(first_name) / "verified-release"
            extracted.mkdir()
            with zipfile.ZipFile(first) as archive:
                packaged = json.loads(archive.read("package-manifest.json"))
                for info in archive.infolist():
                    target = extracted / info.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(info))
                    os.chmod(target, (info.external_attr >> 16) & 0o777)
            package_inventory.verify_bound(
                extracted, projection.inventory_sha256
            )
            self.assertEqual(
                packaged["scopes"][0]["compatibility_report_sha256"],
                report_digest,
            )
            self.assertEqual(packaged["package_state"], "release")
            self.assertIsNone(
                json.loads((package_root / "package-manifest.json").read_bytes())[
                    "scopes"
                ][0]["compatibility_report_sha256"]
            )
            release_dispatcher = {
                "binary": package_inventory.LAUNCHER,
                "sha256": projection.launcher_sha256,
            }
            _validate_packaged_dispatcher(first, release_dispatcher)

            changed = json.loads(
                (package_root / "package-manifest.json").read_bytes()
            )
            changed["scopes"][0]["openvino_compile_config"] = {
                "PERFORMANCE_HINT": "THROUGHPUT"
            }
            old_inventory = file_sha256(
                package_root / package_inventory.INVENTORY_NAME
            )
            package_inventory.rebind_package_manifest(
                package_root, old_inventory, _canonical_json(changed)
            )
            dispatcher["sha256"] = file_sha256(
                package_root / package_inventory.LAUNCHER
            )
            changed_reports = placement_reports_for_package(
                package_root / "package-manifest.json", [scope]
            )
            with self.assertRaisesRegex(
                ValueError, "openvino_compile_config does not match"
            ):
                _load_and_bind_package_manifest(
                    package_root,
                    "package-manifest.json",
                    ["synthetic-npu"],
                    loaded,
                    changed_reports,
                    dispatcher,
                    report_digest,
                )

    def test_package_plan_emits_exact_dispatcher_and_fallback_policy(self) -> None:
        with (
            tempfile.TemporaryDirectory() as root_name,
            tempfile.TemporaryDirectory() as assets_name,
        ):
            root = Path(root_name)
            package_root = root / "package"
            scopes = [
                metadata("synthetic-npu", "npu"),
                metadata("synthetic-gpu", "gpu"),
                metadata("synthetic-cpu", "cpu"),
            ]
            write_synthetic_package(package_root, [dict(scope) for scope in scopes])
            artifact_digest = file_sha256(
                package_root / "artifact/artifact-manifest.json"
            )
            for scope in scopes:
                scope["artifact_sha256"] = artifact_digest
            loaded = {
                scope["scope_id"]: (scope, None, None) for scope in scopes
            }
            remote_scope = metadata("synthetic-remote", "cpu")
            remote_scope["transport"] = "remote-attested"
            loaded["synthetic-remote"] = (remote_scope, None, None)
            manifest = {
                "packages": [
                    {
                        "package_id": "synthetic-linux-package",
                        "release_variant_id": "linux-cfetch-local-x86_64",
                        "os": "linux",
                        "arch": "x86_64",
                        "device_families": [scope["device"] for scope in scopes],
                        "ordered_scope_ids": [scope["scope_id"] for scope in scopes],
                        "package_directory": "package",
                        "package_manifest": "package-manifest.json",
                        "package_format": "zip",
                        "dispatcher": {
                            "binary": package_inventory.LAUNCHER,
                            "sha256": file_sha256(
                                package_root / package_inventory.LAUNCHER
                            ),
                        },
                    }
                ]
            }
            evidence_reports = placement_reports_for_package(
                package_root / "package-manifest.json",
                scopes,
            )
            packages, _ = _validate_package_rows(
                manifest,
                root,
                loaded,
                evidence_reports,
                "9" * 64,
                "admission-v1",
                {
                    "linux-cfetch-local-x86_64": {
                        "id": "linux-cfetch-local-x86_64",
                        "os": "linux",
                        "arch": "x86_64",
                        "binary": "cfetch",
                        "backend": "local",
                    }
                },
                Path(assets_name),
            )
            plan = packages[0]
            self.assertEqual(
                plan["dispatcher"]["binary"], package_inventory.LAUNCHER
            )
            self.assertNotEqual(
                plan["dispatcher"]["sha256"],
                manifest["packages"][0]["dispatcher"]["sha256"],
            )
            with zipfile.ZipFile(Path(assets_name) / f"{plan['package_sha256']}.zip") as archive:
                packaged_manifest = archive.read("package-manifest.json")
                packaged_dispatcher = archive.read(package_inventory.LAUNCHER)
            self.assertEqual(
                plan["dispatcher"]["sha256"], _sha256_bytes(packaged_dispatcher)
            )
            self.assertEqual(
                plan["package_manifest_sha256"],
                _sha256_bytes(packaged_manifest),
            )
            self.assertIn("NPU, GPU, accelerated CPU", plan["selection"])
            self.assertEqual(plan["remote_fallback"], "none")
            loaded["synthetic-npu"][0]["transport"] = "remote-attested"
            with self.assertRaisesRegex(ValueError, "supervised-local"):
                _validate_package_rows(
                    manifest,
                    root,
                    loaded,
                    evidence_reports,
                    "9" * 64,
                    "admission-v1",
                    {
                        "linux-cfetch-local-x86_64": {
                            "id": "linux-cfetch-local-x86_64",
                            "os": "linux",
                            "arch": "x86_64",
                            "binary": "cfetch",
                            "backend": "local",
                        }
                    },
                    Path(assets_name),
                )
            loaded["synthetic-npu"][0]["transport"] = "supervised-local"
            endpoint_catalog = {
                "linux-cfetch-local-x86_64": {
                    "id": "linux-cfetch-local-x86_64",
                    "os": "linux",
                    "arch": "x86_64",
                    "binary": "cfetch",
                    "backend": "endpoint",
                }
            }
            with self.assertRaisesRegex(ValueError, "endpoint-only"):
                _validate_package_rows(
                    manifest,
                    root,
                    loaded,
                    evidence_reports,
                    "9" * 64,
                    "admission-v1",
                    endpoint_catalog,
                    Path(assets_name),
                )

    def test_dispatcher_must_be_executable_plain_package_root_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "cfetch-inference"
            path.write_bytes(b"not executable")
            with self.assertRaisesRegex(ValueError, "executable"):
                _validate_dispatcher(
                    root,
                    {"binary": path.name, "sha256": file_sha256(path)},
                    "linux",
                    "cfetch",
                    "dispatcher",
                )
            os.chmod(path, 0o755)
            with self.assertRaisesRegex(ValueError, "plain package-root"):
                _validate_dispatcher(
                    root,
                    {"binary": "bin/cfetch-inference", "sha256": file_sha256(path)},
                    "linux",
                    "cfetch",
                    "dispatcher",
                )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_package_tree_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real").write_text("payload")
            os.symlink(root / "real", root / "alias")
            with self.assertRaisesRegex(ValueError, "symlink"):
                _walk_package_files(root)


class FullStageTests(unittest.TestCase):
    @patch("admission_transaction.validate_measurement_bundle")
    @patch("admission_transaction.load_embedded_evidence_reports", return_value={})
    @patch("admission_transaction.validate_admission_cache_container")
    @patch("admission_transaction.verify_implementation_bundle")
    def test_stages_complete_synthetic_cohort_without_mutating_base_files(
        self,
        verify: object,
        validate_cache: object,
        load_evidence: object,
        validate_measurement: object,
    ) -> None:
        del verify, validate_cache, validate_measurement
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = Path(__file__).resolve().parents[2]
            base_registry = root / "inference-backends.json"
            base_registry.write_bytes(
                (repository / "release/inference-backends.json").read_bytes()
            )
            variants = root / "variants.json"
            variants.write_bytes(
                _canonical_json(
                    {
                        "schema_version": 1,
                        "variants": [
                            {
                                "id": "linux-cfetch-local-x86_64",
                                "os": "linux",
                                "arch": "x86_64",
                                "runner": "synthetic",
                                "target": "",
                                "binary": "cfetch",
                                "archive": "tar.gz",
                                "backend": "local",
                                "cargo_features": "",
                            }
                        ],
                    },
                    pretty=True,
                )
            )
            scopes = [
                metadata("synthetic-npu", "npu"),
                metadata("synthetic-gpu", "gpu"),
                metadata("synthetic-cpu", "cpu"),
            ]
            package = root / "package"
            write_synthetic_package(package, [dict(scope) for scope in scopes])
            artifact_digest = file_sha256(package / "artifact/artifact-manifest.json")
            for scope in scopes:
                scope["artifact_sha256"] = artifact_digest
            cache_paths: dict[str, Path] = {}
            scope_rows = []
            for index, scope in enumerate(scopes):
                cache = root / f"{scope['scope_id']}.npz"
                cache.write_bytes(f"synthetic-cache-{index}".encode("ascii"))
                cache_paths[str(scope["scope_id"])] = cache
                raw = root / f"{scope['scope_id']}-raw"
                raw.mkdir()
                (raw / "profiler.txt").write_text("synthetic physical evidence\n")
                scope_rows.append(
                    {
                        "scope_id": scope["scope_id"],
                        "admission_cache": cache.name,
                        "admission_cache_sha256": file_sha256(cache),
                        "raw_measurements": raw.name,
                    }
                )
            private_key = Ed25519PrivateKey.generate()
            public_hex = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            ).hex()
            transaction = {
                "schema_version": 1,
                "base_registry": base_registry.name,
                "base_registry_sha256": file_sha256(base_registry),
                "base_variants": variants.name,
                "base_variants_sha256": file_sha256(variants),
                "release_tag": "synthetic-admission-v1",
                "receipt_attestation_public_key": public_hex,
                "candidate_scopes": [scope["scope_id"] for scope in scopes],
                "scopes": scope_rows,
                "packages": [
                    {
                        "package_id": "synthetic-linux-package",
                        "release_variant_id": "linux-cfetch-local-x86_64",
                        "os": "linux",
                        "arch": "x86_64",
                        "device_families": [scope["device"] for scope in scopes],
                        "ordered_scope_ids": [scope["scope_id"] for scope in scopes],
                        "package_directory": package.name,
                        "package_manifest": "package-manifest.json",
                        "package_format": "zip",
                        "dispatcher": {
                            "binary": package_inventory.LAUNCHER,
                            "sha256": file_sha256(
                                package / package_inventory.LAUNCHER
                            ),
                        },
                    }
                ],
            }
            embedded_evidence = placement_reports_for_package(
                package / "package-manifest.json",
                scopes,
            )
            load_evidence.side_effect = lambda path: embedded_evidence[path.stem]
            manifest = root / "transaction.json"
            manifest.write_bytes(_canonical_json(transaction, pretty=True))
            loaded = {
                str(scope["scope_id"]): (scope, None, None) for scope in scopes
            }
            probes = {
                str(scope["scope_id"]): (None, None, None) for scope in scopes
            }
            report = {
                "admission_gate": {"passed": True},
                "cache_sha256_by_scope": {
                    scope_id: file_sha256(path)
                    for scope_id, path in cache_paths.items()
                },
            }
            report_bytes = _canonical_json(report, pretty=True)

            def measurement(
                cache: Path, raw: Path, output: Path
            ) -> Path:
                del raw
                payload = b"synthetic-measurement:" + cache.read_bytes()
                digest = _sha256_bytes(payload)
                result = output / f"{digest}.zip"
                result.write_bytes(payload)
                return result

            with (
                patch("admission_transaction.load_cache", side_effect=lambda path: loaded[path.stem]),
                patch(
                    "admission_transaction.load_sequence_probe_cache",
                    side_effect=lambda path: probes[path.stem],
                ),
                patch(
                    "admission_transaction._build_report",
                    return_value=(report, report_bytes, _sha256_bytes(report_bytes)),
                ),
                patch("admission_transaction.build_measurement_bundle", side_effect=measurement),
            ):
                stage_plan = stage_transaction(manifest, root / "stage")

            plan = json.loads(stage_plan.read_bytes())
            self.assertEqual(
                plan["profile_source_promotion"],
                _profile_source_promotion(repository),
            )
            proposed = json.loads(
                (root / "stage/proposed/release/inference-backends.json").read_bytes()
            )
            self.assertEqual(plan["cohort_scopes"], sorted(loaded))
            self.assertEqual(proposed["profile_status"], "active")
            self.assertEqual(len(proposed["admitted_backends"]), 3)
            self.assertEqual(
                proposed["local_packages"][0]["dispatcher"]["binary"],
                package_inventory.LAUNCHER,
            )
            package_asset = next(
                asset for asset in plan["assets"] if asset["kind"] == "target-package"
            )
            with zipfile.ZipFile(root / "stage/assets" / package_asset["filename"]) as archive:
                packaged_manifest = archive.read("package-manifest.json")
            self.assertEqual(
                proposed["local_packages"][0]["package_manifest_sha256"],
                _sha256_bytes(packaged_manifest),
            )
            self.assertEqual(
                file_sha256(base_registry), transaction["base_registry_sha256"]
            )


class ActivationTests(unittest.TestCase):
    def make_stage(self, root: Path) -> tuple[Path, dict[str, object], Path]:
        (root / "assets").mkdir()
        (root / "proposed/release/admission").mkdir(parents=True)
        report = root / "proposed/release/admission/report.json"
        report.write_bytes(b"{\"passed\":true}\n")
        registry = root / "proposed/release/inference-backends.json"
        registry.write_bytes(b"{\"profile_status\":\"active\"}\n")
        assets = []
        for kind, owner, suffix, payload in (
            ("admission-cache", "scope-one", "npz", b"cache"),
            ("measurement-evidence", "scope-one", "zip", b"measurements"),
            ("target-package", "package-one", "zip", b"package"),
        ):
            digest = _sha256_bytes(payload)
            path = root / "assets" / f"{digest}.{suffix}"
            path.write_bytes(payload)
            assets.append(
                {
                    "kind": kind,
                    "owner_id": owner,
                    "filename": path.name,
                    "sha256": digest,
                    "bytes": len(payload),
                    "format": suffix,
                }
            )
        expected = {
            "package_id": "package-one",
            "package_sha256": assets[2]["sha256"],
            "scope_id": "scope-one",
            "cache_sha256": assets[0]["sha256"],
            "compatibility_report_sha256": file_sha256(report),
        }
        private_key = Ed25519PrivateKey.generate()
        private_path = root / "receipt-attestation.key"
        private_path.write_bytes(
            private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        public_hex = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()
        plan: dict[str, object] = {
            "schema_version": 1,
            "release_tag": "admission-v1",
            "receipt_attestation_public_key": public_hex,
            "base_registry_sha256": file_sha256(
                Path(__file__).resolve().parents[2]
                / "release/inference-backends.json"
            ),
            "base_variants_sha256": file_sha256(
                Path(__file__).resolve().parents[2] / "release/variants.json"
            ),
            "profile_source_promotion": _profile_source_promotion(
                Path(__file__).resolve().parents[2]
            ),
            "admission_implementation_bundle_sha256": (
                ADMISSION_IMPLEMENTATION_BUNDLE_SHA256
            ),
            "candidate_scopes": ["scope-one"],
            "cohort_scopes": ["scope-one"],
            "compatibility_report": "proposed/release/admission/report.json",
            "compatibility_report_sha256": file_sha256(report),
            "proposed_registry": "proposed/release/inference-backends.json",
            "proposed_registry_sha256": file_sha256(registry),
            "assets": assets,
            "expected_conformance_receipts": [expected],
        }
        plan["stage_id"] = _stage_id(plan)
        stage_plan = root / "stage-plan.json"
        stage_plan.write_bytes(_canonical_json(plan, pretty=True))
        return stage_plan, plan, private_path

    def write_receipt(
        self, root: Path, plan: dict[str, object], private_path: Path
    ) -> Path:
        expected = plan["expected_conformance_receipts"][0]
        receipt = {
            "schema_version": 1,
            "passed": True,
            "stage_id": plan["stage_id"],
            **expected,
            "package_bytes": len(b"package"),
            "admission_implementation_bundle_sha256": (
                ADMISSION_IMPLEMENTATION_BUNDLE_SHA256
            ),
            "wire_groupings": 64,
            "sequence_buckets": 7,
            "signed_requests": EXPECTED_SIGNED_REQUESTS,
        }
        claim = _canonical_json(receipt)
        private_key = Ed25519PrivateKey.from_private_bytes(private_path.read_bytes())
        envelope = {
            "schema_version": 1,
            "receipt": receipt,
            "signature": private_key.sign(claim).hex(),
        }
        raw = _canonical_json(envelope)
        path = root / f"{_sha256_bytes(raw)}.json"
        path.write_bytes(raw)
        return path

    @patch("admission_transaction.verify_implementation_bundle")
    def test_activation_refuses_missing_receipt_then_accepts_exact_coverage(
        self, verify: object
    ) -> None:
        del verify
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_root = root / "stage"
            stage_root.mkdir()
            stage_plan, plan, private_path = self.make_stage(stage_root)
            with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
                activate_transaction(stage_plan, [], root / "blocked")

            receipt = self.write_receipt(root, plan, private_path)
            activation = activate_transaction(
                stage_plan, [receipt], root / "activation"
            )
            self.assertTrue(activation.is_file())
            self.assertEqual(
                json.loads(activation.read_bytes())["status"],
                "release-ready-not-published",
            )
            activation_document = json.loads(activation.read_bytes())
            self.assertEqual(
                activation_document["base_registry_sha256"],
                plan["base_registry_sha256"],
            )
            self.assertEqual(
                activation_document["base_variants_sha256"],
                plan["base_variants_sha256"],
            )
            self.assertEqual(
                activation_document["profile_source_promotion"],
                plan["profile_source_promotion"],
            )
            self.assertTrue(activation_document["assets"])
            self.assertGreater(
                activation_document["receipts"][0]["receipt_bytes"], 0
            )
            self.assertTrue(
                (root / "activation/release/inference-backends.json").is_file()
            )

    @patch("admission_transaction.verify_implementation_bundle")
    def test_activation_rejects_changed_staged_asset(self, verify: object) -> None:
        del verify
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_root = root / "stage"
            stage_root.mkdir()
            stage_plan, plan, private_path = self.make_stage(stage_root)
            receipt = self.write_receipt(root, plan, private_path)
            package = next(
                row
                for row in plan["assets"]
                if row["kind"] == "target-package"
            )
            (stage_root / "assets" / package["filename"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "exact bytes changed"):
                activate_transaction(stage_plan, [receipt], root / "blocked")

    @patch("admission_transaction.verify_implementation_bundle")
    def test_activation_rejects_profile_source_promotion_drift(
        self, verify: object
    ) -> None:
        del verify
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_root = root / "stage"
            stage_root.mkdir()
            stage_plan, plan, _ = self.make_stage(stage_root)
            plan["profile_source_promotion"]["base_sha256"] = "0" * 64
            plan["stage_id"] = _stage_id(plan)
            stage_plan.write_bytes(_canonical_json(plan, pretty=True))
            with self.assertRaisesRegex(ValueError, "embedding profile source"):
                activate_transaction(stage_plan, [], root / "blocked")

    @patch("admission_transaction.verify_implementation_bundle")
    def test_activation_rejects_a_content_addressed_but_forged_receipt(
        self, verify: object
    ) -> None:
        del verify
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_root = root / "stage"
            stage_root.mkdir()
            stage_plan, plan, private_path = self.make_stage(stage_root)
            valid = self.write_receipt(root, plan, private_path)
            forged = json.loads(valid.read_bytes())
            forged["receipt"]["package_bytes"] += 1
            forged_bytes = _canonical_json(forged)
            forged_path = root / f"{_sha256_bytes(forged_bytes)}.json"
            forged_path.write_bytes(forged_bytes)
            with self.assertRaisesRegex(ValueError, "signature is invalid"):
                activate_transaction(stage_plan, [forged_path], root / "blocked")


class OneCommandTransactionTests(unittest.TestCase):
    def transaction_inputs(
        self, root: Path
    ) -> tuple[Path, Path, dict[str, object]]:
        repository = Path(__file__).resolve().parents[2]
        registry = root / "inference-backends.json"
        variants = root / "variants.json"
        registry.write_bytes((repository / "release/inference-backends.json").read_bytes())
        variants.write_bytes((repository / "release/variants.json").read_bytes())
        private_key = Ed25519PrivateKey.generate()
        private_path = root / "receipt.key"
        private_path.write_bytes(
            private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        public_hex = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()
        (root / "scifact").mkdir()
        transaction: dict[str, object] = {
            "schema_version": 1,
            "base_registry": registry.name,
            "base_registry_sha256": file_sha256(registry),
            "base_variants": variants.name,
            "base_variants_sha256": file_sha256(variants),
            "release_tag": "synthetic-admission-v1",
            "receipt_attestation_public_key": public_hex,
            "scifact_snapshot": "scifact",
            "candidate_scopes": ["scope-npu"],
            "scopes": [{}],
            "packages": [{}],
        }
        manifest = root / "transaction.json"
        manifest.write_bytes(_canonical_json(transaction, pretty=True))
        return manifest, private_path, transaction

    @patch("admission_transaction._build_publication_plan")
    @patch("admission_transaction.activate_transaction")
    @patch("admission_transaction._run_final_conformance_receipt")
    @patch("admission_transaction._receipt_execution_rows")
    @patch("admission_transaction._validate_stage_bytes")
    @patch("admission_transaction._load_stage_plan")
    @patch("admission_transaction.stage_transaction")
    @patch("admission_transaction.verify_release_registry")
    @patch("admission_transaction.load_scifact_contract")
    @patch("admission_transaction.verify_implementation_bundle")
    def test_one_command_runs_replay_stage_all_receipts_activation_and_plan(
        self,
        verify_bundle: object,
        load_scifact: object,
        verify_registry: object,
        stage: object,
        load_plan: object,
        validate_stage: object,
        receipt_rows: object,
        conform: object,
        activate: object,
        build_publication: object,
    ) -> None:
        del verify_bundle, load_scifact, validate_stage
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, private_key, _ = self.transaction_inputs(root)
            replay = {
                "admitted_scopes": 0,
                "measurement_bundles_verified": 0,
                "reports_replayed": 0,
                "status": "empty-no-op",
            }
            verify_registry.return_value = replay
            plan = {
                "stage_id": "1" * 64,
                "compatibility_report_sha256": "2" * 64,
            }

            def fake_stage(source: Path, destination: Path) -> Path:
                self.assertEqual(source, manifest)
                destination.mkdir()
                result = destination / "stage-plan.json"
                result.write_bytes(b"{}\n")
                return result

            stage.side_effect = fake_stage
            load_plan.side_effect = lambda path: (plan, path.parent)
            rows = [
                {"package_id": "package-one", "scope_id": "scope-npu"},
                {"package_id": "package-one", "scope_id": "scope-gpu"},
            ]
            receipt_rows.return_value = rows

            def fake_conformance(
                stage_plan: Path,
                row: dict[str, object],
                key: Path,
                receipts: Path,
                scifact_snapshot: Path,
            ) -> Path:
                self.assertEqual(key, private_key.resolve())
                self.assertEqual(scifact_snapshot, root / "scifact")
                result = receipts / f"{row['scope_id']}.json"
                result.write_bytes(b"{}\n")
                return result

            conform.side_effect = fake_conformance

            def fake_activate(
                stage_plan: Path, receipts: list[Path], destination: Path
            ) -> Path:
                self.assertEqual(len(receipts), 2)
                destination.mkdir()
                result = destination / f"{'3' * 64}.activation.json"
                result.write_bytes(b"{}\n")
                return result

            activate.side_effect = fake_activate
            publication = {"schema_version": 1, "status": "ready-not-published"}
            publication_bytes = _canonical_json(publication, pretty=True)
            publication_digest = _sha256_bytes(publication_bytes)
            build_publication.return_value = (
                publication,
                publication_bytes,
                publication_digest,
            )

            result = run_transaction(manifest, private_key, root / "complete")

            self.assertEqual(
                result,
                root / "complete" / f"{publication_digest}.publication.json",
            )
            self.assertEqual(result.read_bytes(), publication_bytes)
            self.assertNotIn(str(root).encode("utf-8"), result.read_bytes())
            self.assertEqual(conform.call_count, 2)
            build_publication.assert_called_once()
            self.assertEqual(build_publication.call_args.args[1], replay)

    @patch("admission_transaction.load_scifact_contract")
    @patch("admission_transaction.verify_implementation_bundle")
    def test_one_command_refuses_a_receipt_key_not_bound_by_manifest(
        self, verify_bundle: object, load_scifact: object
    ) -> None:
        del verify_bundle, load_scifact
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _, _ = self.transaction_inputs(root)
            wrong = root / "wrong.key"
            wrong.write_bytes(b"x" * 32)
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_transaction(manifest, wrong, root / "blocked")
            self.assertFalse((root / "blocked").exists())

    @patch("admission_transaction.verify_implementation_bundle")
    def test_one_command_requires_a_valid_manifest_bound_snapshot(
        self, verify_bundle: object
    ) -> None:
        del verify_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, private_key, transaction = self.transaction_inputs(root)

            transaction.pop("scifact_snapshot")
            manifest.write_bytes(_canonical_json(transaction, pretty=True))
            with self.assertRaisesRegex(ValueError, "requires scifact_snapshot"):
                run_transaction(manifest, private_key, root / "missing")

            transaction["scifact_snapshot"] = "../outside"
            manifest.write_bytes(_canonical_json(transaction, pretty=True))
            with self.assertRaisesRegex(ValueError, "normalized relative path"):
                run_transaction(manifest, private_key, root / "escaped")

            transaction["scifact_snapshot"] = "scifact"
            manifest.write_bytes(_canonical_json(transaction, pretty=True))
            with self.assertRaisesRegex(ValueError, "snapshot is invalid"):
                run_transaction(manifest, private_key, root / "invalid")

            self.assertFalse((root / "missing").exists())
            self.assertFalse((root / "escaped").exists())
            self.assertFalse((root / "invalid").exists())

    @patch("admission_transaction._load_stage_plan")
    @patch("admission_transaction.subprocess.run")
    def test_final_conformance_receipt_receives_the_verified_snapshot(
        self, run: object, load_plan: object
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = root / "receipts"
            receipts.mkdir()
            snapshot = root / "scifact"
            snapshot.mkdir()
            stage_plan = root / "stage-plan.json"
            stage_plan.write_bytes(b"{}\n")
            private_key = root / "receipt.key"
            private_key.write_bytes(b"x" * 32)
            row = {
                "cache_asset": root / "cache.npz",
                "cache_sha256": "1" * 64,
                "package_id": "package-one",
                "package_asset": root / "package.zip",
                "package_sha256": "2" * 64,
                "dispatcher": "cfetch-inference",
                "scope_id": "scope-npu",
            }
            load_plan.return_value = (
                {
                    "stage_id": "3" * 64,
                    "compatibility_report_sha256": "4" * 64,
                },
                root,
            )

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
                del kwargs
                option = command.index("--scifact-snapshot")
                self.assertEqual(command[option + 1], str(snapshot))
                (receipts / "receipt.json").write_bytes(b"{}\n")
                return subprocess.CompletedProcess(command, 0)

            run.side_effect = fake_run
            result = _run_final_conformance_receipt(
                stage_plan, row, private_key, receipts, snapshot
            )
            self.assertEqual(result, receipts / "receipt.json")

    @patch("admission_transaction.verify_implementation_bundle")
    def test_publication_plan_binds_exact_release_and_repository_bytes(
        self, verify_bundle: object
    ) -> None:
        del verify_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_root = root / "stage"
            stage_root.mkdir()
            helper = ActivationTests()
            stage_plan, plan, private_path = helper.make_stage(stage_root)
            receipt = helper.write_receipt(root, plan, private_path)
            activation = activate_transaction(
                stage_plan, [receipt], root / "activation"
            )
            publication, raw, digest = _build_publication_plan(
                activation, {"status": "empty-no-op"}
            )
            self.assertEqual(digest, _sha256_bytes(raw))
            self.assertEqual(publication["status"], "ready-not-published")
            self.assertEqual(
                publication["activation_manifest"]["sha256"],
                file_sha256(activation),
            )
            for asset in publication["release_assets"]:
                self.assertTrue(asset["url"].endswith("/" + asset["filename"]))

            first_asset = publication["release_assets"][0]
            (root / first_asset["source"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "bytes do not match"):
                _build_publication_plan(activation, {"status": "empty-no-op"})


if __name__ == "__main__":
    unittest.main()
