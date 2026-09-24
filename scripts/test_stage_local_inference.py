#!/usr/bin/env python3
"""Dependency-light tests for exact local inference release staging."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from scripts.stage_local_inference import StagingError, URL_RE, stage_archive


def zip_info(name: str, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.external_attr = (stat.S_IFREG | mode) << 16
    return info


class LocalInferenceStagingTests(unittest.TestCase):
    def test_package_origin_accepts_canonical_and_rejects_retired_owner(self) -> None:
        asset = "releases/download/admission-v1/" + "a" * 64 + ".tar.gz"
        self.assertIsNotNone(URL_RE.fullmatch("https://github.com/corbet-libs/cfetch/" + asset))
        self.assertIsNone(URL_RE.fullmatch("https://github.com/corbet-labs/cfetch/" + asset))

    def test_stages_only_exact_dispatcher_and_scope_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "payload.zip"
            dispatcher_bytes = b"#!/bin/sh\nexit 0\n"
            scopes = ["scope-npu", "scope-gpu", "scope-cpu"]
            manifest = json.dumps(
                {
                    "package_state": "release",
                    "scopes": [{"scope_id": scope} for scope in scopes],
                }
            ).encode()
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(zip_info("cfetch-inference", 0o755), dispatcher_bytes)
                output.writestr(zip_info("package-manifest.json", 0o644), manifest)
                output.writestr(zip_info("artifact/model.bin", 0o644), b"model")
            destination = root / "dist"
            destination.mkdir()
            plan = {
                "dispatcher": {
                    "binary": "cfetch-inference",
                    "sha256": hashlib.sha256(dispatcher_bytes).hexdigest(),
                },
                "package_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "ordered_scope_ids": scopes,
            }
            stage_archive(archive, "zip", plan, destination)
            self.assertEqual((destination / "artifact/model.bin").read_bytes(), b"model")
            self.assertTrue(os.access(destination / "cfetch-inference", os.X_OK))

    def test_rejects_non_release_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "payload.zip"
            dispatcher_bytes = b"#!/bin/sh\nexit 0\n"
            scopes = ["scope-npu", "scope-gpu", "scope-cpu"]
            manifest = json.dumps(
                {
                    "package_state": "candidate",
                    "scopes": [{"scope_id": scope} for scope in scopes],
                }
            ).encode()
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(zip_info("cfetch-inference", 0o755), dispatcher_bytes)
                output.writestr(zip_info("package-manifest.json", 0o644), manifest)
            destination = root / "dist"
            destination.mkdir()
            plan = {
                "dispatcher": {
                    "binary": "cfetch-inference",
                    "sha256": hashlib.sha256(dispatcher_bytes).hexdigest(),
                },
                "package_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "ordered_scope_ids": scopes,
            }
            with self.assertRaisesRegex(StagingError, "release state"):
                stage_archive(archive, "zip", plan, destination)

    def test_rejects_traversal_symlink_and_dispatcher_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {
                "dispatcher": {"binary": "run", "sha256": "0" * 64},
                "package_manifest_sha256": "1" * 64,
                "ordered_scope_ids": ["scope-npu", "scope-gpu", "scope-cpu"],
            }
            for index, (name, mode) in enumerate(
                (("../escape", 0o644), ("run", stat.S_IFLNK | 0o777))
            ):
                archive = root / f"bad-{index}.zip"
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.external_attr = mode << 16
                with zipfile.ZipFile(archive, "w") as output:
                    output.writestr(info, b"bad")
                destination = root / f"dist-{index}"
                destination.mkdir()
                with self.assertRaises(StagingError):
                    stage_archive(archive, "zip", plan, destination)

    def test_rejects_package_manifest_drift_even_when_dispatcher_matches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "payload.zip"
            dispatcher_bytes = b"#!/bin/sh\nexit 0\n"
            manifest = b'{"scopes":[{"scope_id":"scope-npu"},{"scope_id":"scope-gpu"},{"scope_id":"scope-cpu"}]}'
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(zip_info("run", 0o755), dispatcher_bytes)
                output.writestr(zip_info("package-manifest.json", 0o644), manifest)
            destination = root / "dist"
            destination.mkdir()
            plan = {
                "dispatcher": {
                    "binary": "run",
                    "sha256": hashlib.sha256(dispatcher_bytes).hexdigest(),
                },
                "package_manifest_sha256": "0" * 64,
                "ordered_scope_ids": ["scope-npu", "scope-gpu", "scope-cpu"],
            }
            with self.assertRaisesRegex(StagingError, "externally pinned"):
                stage_archive(archive, "zip", plan, destination)


if __name__ == "__main__":
    unittest.main()
