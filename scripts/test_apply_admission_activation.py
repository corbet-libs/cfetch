#!/usr/bin/env python3
"""Dependency-light tests for explicit admission source promotion."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.apply_admission_activation import (
    ActivationError,
    PROFILE_STATUS_ACTIVE_TEXT,
    PROFILE_STATUS_CANDIDATE_TEXT,
    _release_asset_url,
    apply_activation,
    main,
)


def encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, final_url: str) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}
        self._final_url = final_url

    def geturl(self) -> str:
        return self._final_url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class FakePublicationOpener:
    def __init__(
        self, payloads: dict[str, bytes], final_url: str | None = None
    ) -> None:
        self.payloads = payloads
        self.final_url = final_url

    def open(self, request: object, timeout: int) -> FakeResponse:
        self.timeout = timeout
        url = request.full_url
        return FakeResponse(self.payloads[url], self.final_url or url)


class ActivationFixture:
    def __init__(self, root: Path) -> None:
        self.repository = root / "repository"
        self.bundle = root / "bundle"
        (self.repository / "src").mkdir(parents=True)
        (self.repository / "release").mkdir()
        (self.bundle / "assets").mkdir(parents=True)
        (self.bundle / "release/admission").mkdir(parents=True)
        (self.bundle / "receipts").mkdir()

        self.source_bytes = (
            "// frozen profile\n"
            f"{PROFILE_STATUS_CANDIDATE_TEXT}\n"
            "pub const PROFILE_ID: &str = \"cfetch-embedding-v1\";\n"
        ).encode("utf-8")
        (self.repository / "src/embedding_profile.rs").write_bytes(self.source_bytes)
        self.base_registry = {
            "schema_version": 1,
            "profile_id": "cfetch-embedding-v1",
            "profile_status": "candidate",
            "shared_identity": {"profile_manifest_sha256": "a" * 64},
            "admission": {"policy_manifest_sha256": "b" * 64},
            "admitted_backends": [],
            "local_packages": [],
        }
        self.base_registry_bytes = encoded(self.base_registry)
        (self.repository / "release/inference-backends.json").write_bytes(
            self.base_registry_bytes
        )
        self.variants_bytes = encoded(
            {
                "schema_version": 1,
                "variants": [{"id": "linux-cfetch-local-x86_64"}],
            }
        )
        (self.repository / "release/variants.json").write_bytes(self.variants_bytes)

        self.report = {"admission_gate": {"passed": True}}
        self.report_bytes = encoded(self.report)
        self.report_digest = sha256(self.report_bytes)
        self.report_relative = f"release/admission/{self.report_digest}.json"
        (self.bundle / self.report_relative).write_bytes(self.report_bytes)

        active_registry = copy.deepcopy(self.base_registry)
        active_registry["profile_status"] = "active"
        active_registry["admitted_backends"] = [
            {
                "scope_id": "scope-npu",
                "compatibility_report": self.report_relative,
                "compatibility_report_sha256": self.report_digest,
            }
        ]
        active_registry["local_packages"] = [{"package_id": "package-one"}]
        self.active_registry_bytes = encoded(active_registry)
        (self.bundle / "release/inference-backends.json").write_bytes(
            self.active_registry_bytes
        )

        asset_bytes = b"canonical cache bytes\n"
        asset_digest = sha256(asset_bytes)
        asset_name = f"{asset_digest}.npz"
        self.asset_bytes = asset_bytes
        self.asset_name = asset_name
        (self.bundle / "assets" / asset_name).write_bytes(asset_bytes)
        receipt_bytes = b'{"receipt":"verified upstream"}\n'
        receipt_digest = sha256(receipt_bytes)
        receipt_relative = f"receipts/{receipt_digest}.json"
        (self.bundle / receipt_relative).write_bytes(receipt_bytes)

        active_source = self.source_bytes.replace(
            PROFILE_STATUS_CANDIDATE_TEXT.encode(),
            PROFILE_STATUS_ACTIVE_TEXT.encode(),
            1,
        )
        self.activation: dict[str, object] = {
            "schema_version": 1,
            "stage_id": "1" * 64,
            "release_tag": "synthetic-admission-v1",
            "base_registry_sha256": sha256(self.base_registry_bytes),
            "base_variants_sha256": sha256(self.variants_bytes),
            "admission_implementation_bundle_sha256": "2" * 64,
            "profile_source_promotion": {
                "path": "src/embedding_profile.rs",
                "base_sha256": sha256(self.source_bytes),
                "active_sha256": sha256(active_source),
                "candidate_text": PROFILE_STATUS_CANDIDATE_TEXT,
                "active_text": PROFILE_STATUS_ACTIVE_TEXT,
            },
            "compatibility_report": self.report_relative,
            "compatibility_report_sha256": self.report_digest,
            "compatibility_report_bytes": len(self.report_bytes),
            "registry": "release/inference-backends.json",
            "registry_sha256": sha256(self.active_registry_bytes),
            "registry_bytes": len(self.active_registry_bytes),
            "assets": [
                {
                    "kind": "admission-cache",
                    "owner_id": "scope-npu",
                    "filename": asset_name,
                    "path": f"assets/{asset_name}",
                    "sha256": asset_digest,
                    "bytes": len(asset_bytes),
                    "format": "npz",
                }
            ],
            "receipts": [
                {
                    "package_id": "package-one",
                    "scope_id": "scope-npu",
                    "receipt": receipt_relative,
                    "receipt_sha256": receipt_digest,
                    "receipt_bytes": len(receipt_bytes),
                }
            ],
            "status": "release-ready-not-published",
        }
        self.manifest = self.rewrite_manifest()

    def publication_opener(self, payload: bytes | None = None) -> FakePublicationOpener:
        url = _release_asset_url(str(self.activation["release_tag"]), self.asset_name)
        return FakePublicationOpener(
            {url: self.asset_bytes if payload is None else payload}
        )

    def rewrite_manifest(self) -> Path:
        for old in self.bundle.glob("*.activation.json"):
            old.unlink()
        raw = encoded(self.activation)
        self.manifest = self.bundle / f"{sha256(raw)}.activation.json"
        self.manifest.write_bytes(raw)
        return self.manifest


class ApplyAdmissionActivationTests(unittest.TestCase):
    def test_publication_url_uses_only_the_canonical_repository(self) -> None:
        self.assertEqual(
            _release_asset_url("admission-v1", "a" * 64 + ".npz"),
            "https://github.com/corbet-libs/cfetch/releases/download/"
            "admission-v1/" + "a" * 64 + ".npz",
        )

    def test_applies_exact_registry_report_and_only_bound_source_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ActivationFixture(Path(directory))
            result = apply_activation(
                fixture.manifest, fixture.repository, fixture.publication_opener()
            )

            self.assertEqual(result["status"], "applied")
            self.assertEqual(
                (fixture.repository / "release/inference-backends.json").read_bytes(),
                fixture.active_registry_bytes,
            )
            self.assertEqual(
                (fixture.repository / fixture.report_relative).read_bytes(),
                fixture.report_bytes,
            )
            promoted = (fixture.repository / "src/embedding_profile.rs").read_bytes()
            expected = fixture.source_bytes.replace(
                PROFILE_STATUS_CANDIDATE_TEXT.encode(),
                PROFILE_STATUS_ACTIVE_TEXT.encode(),
                1,
            )
            self.assertEqual(promoted, expected)
            self.assertEqual(
                (fixture.repository / "release/variants.json").read_bytes(),
                fixture.variants_bytes,
            )
            self.assertEqual(result["published_assets_verified"], 1)
            self.assertEqual(
                result["published_asset_bytes_verified"], len(fixture.asset_bytes)
            )

    def test_refuses_unpublished_or_changed_release_asset_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ActivationFixture(Path(directory))
            with self.assertRaisesRegex(ActivationError, "Content-Length|downloaded bytes"):
                apply_activation(
                    fixture.manifest,
                    fixture.repository,
                    fixture.publication_opener(b"changed"),
                )
            self.assertEqual(
                (fixture.repository / "release/inference-backends.json").read_bytes(),
                fixture.base_registry_bytes,
            )
            self.assertFalse((fixture.repository / fixture.report_relative).exists())

    def test_refuses_release_asset_redirect_outside_github(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ActivationFixture(Path(directory))
            url = _release_asset_url(
                str(fixture.activation["release_tag"]), fixture.asset_name
            )
            opener = FakePublicationOpener(
                {url: fixture.asset_bytes}, "https://example.invalid/asset"
            )
            with self.assertRaisesRegex(ActivationError, "GitHub HTTPS"):
                apply_activation(fixture.manifest, fixture.repository, opener)
            self.assertFalse((fixture.repository / fixture.report_relative).exists())

    def test_refuses_manifest_filename_or_bundle_content_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ActivationFixture(Path(directory))
            wrong_name = fixture.bundle / f"{'0' * 64}.activation.json"
            wrong_name.write_bytes(fixture.manifest.read_bytes())
            with self.assertRaisesRegex(ActivationError, "filename"):
                apply_activation(wrong_name, fixture.repository)
            wrong_name.unlink()

            (fixture.bundle / fixture.report_relative).write_bytes(b"changed\n")
            with self.assertRaisesRegex(ActivationError, "size|hash"):
                apply_activation(fixture.manifest, fixture.repository)

    def test_refuses_current_registry_variants_or_source_drift_before_writes(self) -> None:
        for target, message in (
            ("release/inference-backends.json", "base registry digest"),
            ("release/variants.json", "release variants digest"),
            ("src/embedding_profile.rs", "profile source digest"),
        ):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                fixture = ActivationFixture(Path(directory))
                (fixture.repository / target).write_bytes(b"drift\n")
                with self.assertRaisesRegex(ActivationError, message):
                    apply_activation(fixture.manifest, fixture.repository)
                self.assertFalse((fixture.repository / fixture.report_relative).exists())

    def test_refuses_partial_or_extended_activation_schema(self) -> None:
        for mutation in ("missing", "unknown"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = ActivationFixture(Path(directory))
                if mutation == "missing":
                    del fixture.activation["base_variants_sha256"]
                else:
                    fixture.activation["unbound"] = True
                fixture.rewrite_manifest()
                with self.assertRaisesRegex(ActivationError, "schema mismatch"):
                    apply_activation(fixture.manifest, fixture.repository)

    def test_refuses_a_mutable_release_tag_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ActivationFixture(Path(directory))
            fixture.activation["release_tag"] = "latest"
            fixture.rewrite_manifest()
            with self.assertRaisesRegex(ActivationError, "release_tag"):
                apply_activation(fixture.manifest, fixture.repository)

    def test_refuses_nonexact_status_replacement_even_when_hash_claim_is_updated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ActivationFixture(Path(directory))
            duplicated = fixture.source_bytes + PROFILE_STATUS_CANDIDATE_TEXT.encode() + b"\n"
            (fixture.repository / "src/embedding_profile.rs").write_bytes(duplicated)
            active = duplicated.replace(
                PROFILE_STATUS_CANDIDATE_TEXT.encode(),
                PROFILE_STATUS_ACTIVE_TEXT.encode(),
                1,
            )
            promotion = fixture.activation["profile_source_promotion"]
            promotion["base_sha256"] = sha256(duplicated)
            promotion["active_sha256"] = sha256(active)
            fixture.rewrite_manifest()
            with self.assertRaisesRegex(ActivationError, "exactly the bound"):
                apply_activation(fixture.manifest, fixture.repository)

    def test_refuses_empty_active_registry_even_when_all_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ActivationFixture(Path(directory))
            empty_active = copy.deepcopy(fixture.base_registry)
            empty_active["profile_status"] = "active"
            active_bytes = encoded(empty_active)
            (fixture.bundle / "release/inference-backends.json").write_bytes(active_bytes)
            fixture.activation["registry_sha256"] = sha256(active_bytes)
            fixture.activation["registry_bytes"] = len(active_bytes)
            fixture.rewrite_manifest()
            with self.assertRaisesRegex(ActivationError, "nonempty active cohort"):
                apply_activation(fixture.manifest, fixture.repository)

    def test_cli_reports_success_and_failure_instead_of_silent_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ActivationFixture(Path(directory))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), patch(
                "scripts.apply_admission_activation._verify_published_assets",
                return_value={"assets": 1, "bytes": len(fixture.asset_bytes)},
            ):
                code = main(
                    [
                        "--activation-manifest",
                        str(fixture.manifest),
                        "--repository",
                        str(fixture.repository),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "applied")

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(
                [
                    "--activation-manifest",
                    "/does/not/exist.activation.json",
                    "--repository",
                    "/does/not/exist",
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("activation refused", stderr.getvalue())

    def test_mid_apply_failure_rolls_back_registry_report_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ActivationFixture(Path(directory))
            real_replace = os.replace
            calls = 0

            def fail_source_once(source: object, destination: object) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic source promotion failure")
                real_replace(source, destination)

            with patch(
                "scripts.apply_admission_activation.os.replace",
                side_effect=fail_source_once,
            ), self.assertRaisesRegex(OSError, "synthetic source"):
                apply_activation(
                    fixture.manifest,
                    fixture.repository,
                    fixture.publication_opener(),
                )

            self.assertEqual(
                (fixture.repository / "release/inference-backends.json").read_bytes(),
                fixture.base_registry_bytes,
            )
            self.assertEqual(
                (fixture.repository / "src/embedding_profile.rs").read_bytes(),
                fixture.source_bytes,
            )
            self.assertFalse((fixture.repository / fixture.report_relative).exists())


if __name__ == "__main__":
    unittest.main()
