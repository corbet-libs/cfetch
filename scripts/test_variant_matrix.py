#!/usr/bin/env python3
"""The build matrix keeps candidates; the release matrix requires admission."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "variant-matrix.sh"


class VariantMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = json.loads((ROOT / "release" / "variants.json").read_text())
        self.local = next(row for row in self.catalog["variants"] if row["backend"] == "local")
        self.registry = {
            "schema_version": 1,
            "profile_status": "candidate",
            "local_packages": [],
            "admitted_backends": [],
        }

    def run_matrix(self, release: bool = True, registry=None):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            catalog_path = directory / "catalog.json"
            registry_path = directory / "registry.json"
            catalog_path.write_text(json.dumps(self.catalog))
            registry_path.write_text(json.dumps(self.registry if registry is None else registry))
            arguments = ["bash", str(SCRIPT)]
            if release:
                arguments.append("--release")
            arguments.append(str(catalog_path))
            if release:
                arguments.append(str(registry_path))
            return subprocess.run(arguments, capture_output=True, text=True, timeout=10)

    def admit(self) -> None:
        digest = "1" * 64
        self.registry["profile_status"] = "active"
        self.registry["admitted_backends"] = [
            {
                "scope_id": f"scope-{device_class}",
                "device_class": device_class,
                "transport": "supervised-local",
                "accelerated_placement": True,
            }
            for device_class in ("npu", "gpu", "cpu")
        ]
        self.registry["local_packages"] = [{
            "package_id": "test-local-package",
            "release_variant_id": self.local["id"],
            "os": self.local["os"],
            "arch": self.local["arch"],
            "package_sha256": digest,
            "package_format": "tar.gz",
            "package_url": f"https://github.com/corbet-labs/cfetch/releases/download/evidence-v1/{digest}.tar.gz",
            "package_manifest_sha256": "2" * 64,
            "dispatcher": {"binary": "cfetch-inference", "sha256": "3" * 64},
            "ordered_scope_ids": ["scope-npu", "scope-gpu", "scope-cpu"],
        }]

    def assert_refused(self, registry, message: str) -> None:
        result = self.run_matrix(registry=registry)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn(message, result.stderr)

    def test_normal_ci_keeps_every_candidate_and_release_keeps_all_endpoints(self) -> None:
        build = self.run_matrix(release=False)
        self.assertEqual(build.returncode, 0, build.stderr)
        self.assertEqual(json.loads(build.stdout)["include"], self.catalog["variants"])
        release = self.run_matrix()
        self.assertEqual(release.returncode, 0, release.stderr)
        endpoints = [row for row in self.catalog["variants"] if row["backend"] == "endpoint"]
        self.assertEqual(json.loads(release.stdout)["include"], endpoints)
        self.assertEqual({row["os"] for row in endpoints}, {"linux", "mac", "win"})
        self.assertEqual({row["arch"] for row in endpoints}, {"x86_64", "aarch64"})

    def test_release_includes_admitted_local_target_and_retains_other_candidates_for_ci(self) -> None:
        candidate = {**self.local, "id": "linux-cfetch-local-second-x86_64"}
        self.catalog["variants"].append(candidate)
        self.admit()
        release = self.run_matrix()
        self.assertEqual(release.returncode, 0, release.stderr)
        self.assertEqual(json.loads(release.stdout)["include"], self.catalog["variants"][:-1])
        build = self.run_matrix(release=False)
        self.assertEqual(build.returncode, 0, build.stderr)
        self.assertEqual(json.loads(build.stdout)["include"][-1], candidate)

    def test_rejects_partial_or_malformed_activation(self) -> None:
        self.admit()
        changes = (
            ("schema_version", 2, "unsupported inference registry"),
            ("profile_status", "candidate", "profile status"),
            ("local_packages", [], "activate together"),
            ("admitted_backends", [], "activate together"),
            ("local_packages", None, "registry arrays"),
            ("admitted_backends", {}, "registry arrays"),
        )
        for field, value, message in changes:
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(self.registry)
                changed[field] = value
                self.assert_refused(changed, message)

    def test_rejects_unknown_endpoint_duplicate_and_mismatched_package_targets(self) -> None:
        self.admit()
        endpoint_id = next(row["id"] for row in self.catalog["variants"] if row["backend"] == "endpoint")
        for field, value, message in (
            ("release_variant_id", "unknown-target", "unknown variant"),
            ("release_variant_id", endpoint_id, "endpoint variant"),
            ("arch", "aarch64", "target does not match"),
            ("package_url", "https://example.invalid/payload.tar.gz", "content-addressed"),
            ("package_sha256", "bad", "SHA-256"),
            ("package_manifest_sha256", None, "SHA-256"),
            ("dispatcher", {}, "exact dispatcher"),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.registry)
                changed["local_packages"][0][field] = value
                self.assert_refused(changed, message)
        changed = copy.deepcopy(self.registry)
        changed["local_packages"].append(copy.deepcopy(changed["local_packages"][0]))
        self.assert_refused(changed, "exactly one target payload")
        changed["local_packages"][1]["package_id"] = "other-package"
        self.assert_refused(changed, "exactly one target payload")

    def test_rejects_unadmitted_unaccelerated_remote_or_unordered_local_scopes(self) -> None:
        self.admit()
        for scope_ids, message in (
            (["scope-npu", "scope-gpu", "unknown"], "unadmitted scope"),
            (["scope-npu", "scope-gpu"], "ordered NPU/GPU/CPU"),
            (["scope-cpu", "scope-gpu", "scope-npu"], "ordered NPU/GPU/CPU"),
            (["scope-npu", "scope-gpu", "scope-cpu", "scope-cpu"], "duplicate package scope"),
        ):
            with self.subTest(scope_ids=scope_ids):
                changed = copy.deepcopy(self.registry)
                changed["local_packages"][0]["ordered_scope_ids"] = scope_ids
                self.assert_refused(changed, message)
        for field, value, message in (
            ("transport", "remote-attested", "remote scope"),
            ("accelerated_placement", False, "not accelerated"),
        ):
            changed = copy.deepcopy(self.registry)
            changed["admitted_backends"][0][field] = value
            self.assert_refused(changed, message)

    def test_rejects_orphaned_and_duplicate_admitted_local_scopes(self) -> None:
        self.admit()
        changed = copy.deepcopy(self.registry)
        changed["admitted_backends"].append(copy.deepcopy(changed["admitted_backends"][0]))
        self.assert_refused(changed, "duplicate admitted scope")
        changed["admitted_backends"][-1]["scope_id"] = "orphan-npu"
        self.assert_refused(changed, "no release package")


if __name__ == "__main__":
    unittest.main()
