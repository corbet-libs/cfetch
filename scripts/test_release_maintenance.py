"""Maintenance behavior fixtures: temporary Git only; no remote writes or builds."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import release as r
from scripts import release_maintenance as m


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        fixture_environment = {"PATH": os.environ["PATH"]} if "PATH" in os.environ else {}
        fixture_environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        self.environment = patch.dict(os.environ, fixture_environment, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def metadata(self):
        catalog = {"schema_version": 1, "variants": [
            {"id": operating_system + "-cfetch-remote-" + architecture, "os": operating_system,
             "arch": architecture, "backend": "endpoint", "archive": "tar.gz", "binary": "cfetch"}
            for operating_system, architecture in m.PLATFORMS]}
        (self.base / "variants.json").write_text(json.dumps(catalog))
        checksums = {row["id"] + "-v0.9.9.tar.gz": "1" * 64 for row in catalog["variants"]}
        checksums["variants.json"] = r.sha(self.base / "variants.json")
        (self.base / "checksums_sha256.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
        (self.base / "release.json").write_text(json.dumps({"tagName": "v0.9.9", "isDraft": False,
            "isPrerelease": False, "assets": [{"name": name} for name in [*checksums, "checksums_sha256.txt"]]}))
        (self.base / "tag.json").write_text(json.dumps({"type": "commit", "sha": "2" * 40}))
        return catalog, checksums

    def test_version_rejects_shell_and_output_injection(self):
        for value in ('0.9.9"; exit 0; #', '$(touch marker)', '0.9.9\noutput=value', 'v0.9.9', '1.0.0'):
            with self.subTest(value=value), self.assertRaises(r.Failure):
                m.version(value)
        self.assertEqual(m.version("0.9.9"), "0.9.9")

    def test_legacy_public_formula_uses_exact_four_endpoint_checksums(self):
        self.metadata()
        payload, receipt = m.public_metadata(self.base, "0.9.9", "2" * 40)
        self.assertEqual(payload.count(b'      url "'), 4)
        self.assertEqual(payload.count(b'      sha256 "'), 4)
        self.assertIn(b'doc.install "THIRD-PARTY-LICENSES.txt"', payload)
        self.assertEqual(receipt["tag_commit"], "2" * 40)
        self.assertIn("not re-established", receipt["proof"])

    def test_draft_wrong_tag_and_incomplete_public_assets_are_rejected(self):
        self.metadata()
        path = self.base / "release.json"
        original = json.loads(path.read_text())
        for change in ({"isDraft": True}, {"isPrerelease": True}, {"tagName": "v0.9.8"}, {"assets": []}):
            path.write_text(json.dumps({**original, **change}))
            with self.subTest(change=change), self.assertRaises(r.Failure):
                m.public_metadata(self.base, "0.9.9")
        path.write_text(json.dumps(original))
        with self.assertRaisesRegex(r.Failure, "producing commit"):
            m.public_metadata(self.base, "0.9.9", "3" * 40)

    def test_catalog_bytes_and_unique_safe_variants_are_required(self):
        catalog, checksums = self.metadata()
        (self.base / "variants.json").write_text(json.dumps({**catalog, "changed": True}))
        with self.assertRaisesRegex(r.Failure, "checksum differs"):
            m.public_metadata(self.base, "0.9.9")
        catalog["variants"].append(catalog["variants"][0])
        with self.assertRaisesRegex(r.Failure, "Exactly one"):
            m.formula("0.9.9", catalog, checksums)
        catalog["variants"].pop()
        catalog["variants"][0]["id"] = '../escape"'
        with self.assertRaisesRegex(r.Failure, "Unsupported"):
            m.formula("0.9.9", catalog, checksums)

    def test_duplicate_checksums_and_changed_immutable_outputs_are_rejected(self):
        line = b"1" * 64 + b"  archive.tar.gz\n"
        with self.assertRaisesRegex(r.Failure, "duplicate"):
            m.checksum_inventory(line + line)
        path = self.base / "formula.rb"
        m.immutable(path, b"one")
        m.immutable(path, b"one")
        with self.assertRaisesRegex(r.Failure, "differs"):
            m.immutable(path, b"two")

    def test_duplicate_metadata_keys_and_asset_names_are_rejected(self):
        self.metadata()
        path = self.base / "release.json"
        value = json.loads(path.read_text())
        value["assets"].append(value["assets"][0])
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(r.Failure, "Duplicate public"):
            m.public_metadata(self.base, "0.9.9")
        path.write_text('{"isDraft":false,"isDraft":true}')
        with self.assertRaisesRegex(r.Failure, "Duplicate JSON"):
            m.public_metadata(self.base, "0.9.9")

    def test_missing_history_fails_before_source_or_version_changes(self):
        with patch.object(r, "current_source", side_effect=AssertionError("source must not be touched")):
            with self.assertRaisesRegex(r.Failure, "full Git history"):
                m.patch_plan(self.base)

    def test_patch_plan_uses_real_history_and_preserves_source_then_resumes(self):
        root = self.base / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "--quiet", str(root)], check=True)
        m.git(root, "config", "user.name", "Fixture")
        m.git(root, "config", "user.email", "fixture@example.invalid")
        files = {"Cargo.toml": '[package]\nname = "cfetch"\nversion = "0.9.9"\nrepository = "https://github.com/corbet-labs/cfetch"\n',
                 "Cargo.lock": '[[package]]\nname = "cfetch"\nversion = "0.9.9"\n',
                 "CHANGELOG.md": "## Unreleased\n", "packaging/arch/PKGBUILD": "pkgver=0.9.9\n",
                 "scripts/prepare-patch-release.sh": (r.ROOT / "scripts/prepare-patch-release.sh").read_text()}
        for name, payload in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
        m.git(root, "add", ".")
        m.git(root, "commit", "--quiet", "-m", "fixture")
        m.git(root, "tag", "v0.9.9")
        for attempt in (0, 1):
            bundle = self.base / f"history-{attempt}.bundle"
            m.git(root, "bundle", "create", str(bundle), "--all")
            source = self.base / f"source-{attempt}.tar"
            source.write_bytes(m.git(root, "archive", "--format=tar", "HEAD"))
            output = self.base / f"output-{attempt}"
            output.mkdir()
            before = {name: (root / name).read_bytes() for name in files}
            os.environ.update(MAINTENANCE_HISTORY_BUNDLE=str(bundle), MAINTENANCE_HISTORY_SHA256=r.sha(bundle),
                              SOURCE_ARCHIVE=str(source), SOURCE_SHA256=r.sha(source),
                              CI_COMMIT_SHA=m.git(root, "rev-parse", "HEAD").decode().strip())
            with patch.object(r, "ROOT", root):
                receipt = m.patch_plan(output)
            self.assertEqual(receipt["proposed_version"], "0.9.10")
            self.assertEqual(before, {name: (root / name).read_bytes() for name in files})
            self.assertEqual(m.git(root, "status", "--porcelain"), b"")
            if attempt == 0:
                m.git(root, "apply", str(output / "patch.diff"))
                m.git(root, "commit", "--quiet", "-am", "prepared fixture")
            else:
                self.assertEqual((output / "patch.diff").read_bytes(), b"")
            self.assertEqual(m.git(root, "tag", "--list").decode().splitlines(), ["v0.9.9"])


if __name__ == "__main__":
    unittest.main()
