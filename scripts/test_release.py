"""Release contract fixtures: no compiler, network, credentials or native execution."""
import hashlib
import contextlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import release as r


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.tool_environment = {key: os.environ[key] for key in ("CI_TOOL_ARCHIVE", "CI_TOOL_SHA256", "CFETCH_PUBLISHER_ARCHIVE", "CFETCH_PUBLISHER_SHA256") if key in os.environ}
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_native_host_spellings(self):
        # Native Windows ARM Python reports ARM64; Unix uses lower case.
        for system, machine, expected in [("Windows", "ARM64", ("win", "aarch64")),
                                           ("Windows", "AMD64", ("win", "x86_64")),
                                           ("Darwin", "arm64", ("mac", "aarch64")),
                                           ("Linux", "aarch64", ("linux", "aarch64")),
                                           ("Linux", "x86_64", ("linux", "x86_64"))]:
            with self.subTest(system=system, machine=machine), patch.object(r.platform, "system", return_value=system), patch.object(r.platform, "machine", return_value=machine):
                observed = r.host()
                self.assertEqual((observed["os"], observed["arch"]), expected)
        with patch.object(r.platform, "machine", return_value="i686"):
            with self.assertRaisesRegex(r.Failure, "Unsupported native release host"):
                r.host()

    def source(self, directory, entries=None):
        entries = entries or {"Cargo.toml": b'[package]\nname="cfetch"\nversion="0.9.9"\nrepository="https://github.com/corbet-libs/cfetch"\n',
                              "Cargo.lock": b"locked", **{name: name.encode() for name in r.METADATA.values()}}
        with tarfile.open(directory / "source.tar", "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": "1" * 40}) as archive:
            for name, payload in entries.items():
                item = tarfile.TarInfo(name)
                item.size = len(payload)
                item.mode = 0o644
                archive.addfile(item, io.BytesIO(payload))
        return r.identity(directory / "source.tar") if "Cargo.lock" in entries else None

    def fixture(self):
        tree = self.base / "tree"
        tree.mkdir()
        source, files = self.source(tree)
        row = {"id": "linux-cfetch-remote-x86_64", "os": "linux", "arch": "x86_64", "runner": "ubuntu-latest", "target": "", "binary": "cfetch", "archive": "tar.gz", "backend": "endpoint", "cargo_features": ""}
        r.durable(tree / "plan.json", {**source, "variants": [row], "checks": sorted(r.CHECKS), "rust_platforms": ["linux", "mac", "win"]})
        receipts = tree / "receipts"
        receipts.mkdir()
        artifacts = tree / "artifacts"
        artifacts.mkdir()
        runtime = {"python": "fixture", "cargo": "cargo stable", "rustc": "rustc stable", "build_jobs": "1", "test_threads": "1"}
        for selected, operating_system in [(name, "linux") for name in ("catalog", "licenses", "profile")] + [("rust", name) for name in ("linux", "mac", "win")]:
            r.durable(receipts / f"check-{selected}-{operating_system}.json", {**source, "kind": "check", "check": selected, "status": "success", "host": {"os": operating_system, "arch": "x86_64"}, "tools": runtime, "command": ["bash", "scripts/ci-check.sh", selected]})
        package = self.base / (row["id"] + "-v0.9.9")
        package.mkdir()
        for name, original in r.METADATA.items():
            (package / name).write_bytes(files[original])
        (package / "cfetch").write_bytes(b"native fixture")
        (package / "cfetch").chmod(0o755)
        archive = artifacts / (package.name + ".tar.gz")
        r.pack(package, archive, "tar.gz")
        native = {**source, "kind": "variant", "variant": row["id"], "row": row, "inventory": r.inventory(package, modes=True), "host": {"os": "linux", "arch": "x86_64"}, "tools": runtime,
                  "native_smoke": {"version": "cfetch 0.9.9", "build_variant": row["id"]}, "build_command": ["cargo", "build", "--release", "--locked", "--features", ""],
                  "artifact": {"name": archive.name, "sha256": r.sha(archive), "bytes": archive.stat().st_size}}
        r.durable(receipts / "variant-linux.json", native)
        crate = artifacts / "cfetch-0.9.9.crate"
        crate.write_bytes(b"crate fixture; exact Cargo parser covered in shared publisher fixtures")
        r.durable(receipts / "cargo.json", {**source, "kind": "cargo", "dry_run": True, "host": {"os": "linux", "arch": "x86_64"}, "tools": runtime,
                                           "artifact": {"name": crate.name, "sha256": r.sha(crate), "bytes": crate.stat().st_size}})
        core = SimpleNamespace(Bundle=SimpleNamespace(inspect_cargo=lambda *args: None))
        return tree, row, core

    def inspect(self, tree, row, core):
        with patch.object(r, "matrix", return_value=[row]):
            return r.inspect_tree(tree, core)

    def test_complete_fixture_requires_every_gate(self):
        tree, row, core = self.fixture()
        self.inspect(tree, row, core)
        (tree / "receipts/check-rust-mac.json").unlink()
        with self.assertRaisesRegex(r.Failure, "Incomplete.*gates"):
            self.inspect(tree, row, core)

    def test_wrong_native_architecture_rejected(self):
        tree, row, core = self.fixture()
        path = tree / "receipts/variant-linux.json"
        value = r.read_json(path)
        value["host"]["arch"] = "aarch64"
        path.write_bytes(r.json_bytes(value))
        with self.assertRaisesRegex(r.Failure, "Native variant identity"):
            self.inspect(tree, row, core)

    def test_receipt_cannot_relabel_source(self):
        tree, row, core = self.fixture()
        path = tree / "receipts/check-rust-win.json"
        value = r.read_json(path)
        value["source_commit"] = "2" * 40
        path.write_bytes(r.json_bytes(value))
        with self.assertRaisesRegex(r.Failure, "another producing source"):
            self.inspect(tree, row, core)

    def test_rust_gate_requires_actual_compiler(self):
        tree, row, core = self.fixture()
        path = tree / "receipts/check-rust-win.json"
        value = r.read_json(path)
        del value["tools"]["rustc"]
        path.write_bytes(r.json_bytes(value))
        with self.assertRaisesRegex(r.Failure, "actual compiler"):
            self.inspect(tree, row, core)

    def test_executable_mode_is_part_of_archive_identity(self):
        tree, row, core = self.fixture()
        archive = next((tree / "artifacts").glob("*.tar.gz"))
        package = self.base / (row["id"] + "-v0.9.9")
        (package / "cfetch").chmod(0o644)
        archive.unlink()
        r.pack(package, archive, "tar.gz")
        path = tree / "receipts/variant-linux.json"
        value = r.read_json(path)
        value["inventory"] = r.inventory(package, modes=True)
        value["artifact"].update(sha256=r.sha(archive), bytes=archive.stat().st_size)
        path.write_bytes(r.json_bytes(value))
        with self.assertRaisesRegex(r.Failure, "execution mode"):
            self.inspect(tree, row, core)

    def test_untracked_input_and_changed_source_rejected(self):
        root = self.base / "root"
        root.mkdir()
        self.source(self.base, {"Cargo.toml": b"source"})
        (root / "Cargo.toml").write_bytes(b"source")
        with patch.object(r, "ROOT", root):
            r.bind_source(self.base / "source.tar")
            (root / "extra.rs").write_text("untracked")
            with self.assertRaisesRegex(r.Failure, "Uncommitted"):
                r.bind_source(self.base / "source.tar")
            (root / "extra.rs").unlink()
            (root / ".cargo").symlink_to(self.base, target_is_directory=True)
            with self.assertRaisesRegex(r.Failure, "directory symlink"):
                r.bind_source(self.base / "source.tar")
            (root / ".cargo").unlink()
            (root / "Cargo.toml").write_text("changed")
            with self.assertRaisesRegex(r.Failure, "source bytes"):
                r.bind_source(self.base / "source.tar")

    def test_path_and_credentials_fail_before_work(self):
        for value in ("../escape", "/absolute", "a\\b", "a/../b"):
            with self.assertRaises(r.Failure):
                r.safe_name(value)
        os.environ["CARGO_REGISTRY_TOKEN"] = "fixture-only"
        with self.assertRaisesRegex(r.Failure, "Credentials"):
            r.no_credentials()

    def test_hosted_preparation_consumes_exact_shared_plan_source(self):
        root = self.base / "root"
        root.mkdir()
        output = self.base / "output"
        output.mkdir()
        identity, files = self.source(self.base)
        for name, payload in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        os.environ.update(RELEASE_SOURCE_ARCHIVE=str(self.base / "source.tar"), RELEASE_SOURCE_SHA256=identity["source_sha256"], GITHUB_SHA=identity["source_commit"])
        with patch.object(r, "ROOT", root), patch.object(r, "command", side_effect=AssertionError("No regenerated Git archive")):
            observed, _ = r.current_source(output)
        self.assertEqual(observed, identity)
        self.assertEqual((output / "source.tar").read_bytes(), (self.base / "source.tar").read_bytes())

    def test_cargo_requires_complete_public_github_release(self):
        data = {"artifacts": {"native.tar.gz": self.base / "native.tar.gz"}}
        remote = SimpleNamespace(release={"draft": True}, present=lambda *args: {"native.tar.gz"})
        with self.assertRaises(r.Failure):
            r.require_cargo_release(remote, data)
        remote.release["draft"] = False
        remote.present = lambda *args: set()
        with self.assertRaises(r.Failure):
            r.require_cargo_release(remote, data)
        remote.present = lambda *args: {"native.tar.gz"}
        r.require_cargo_release(remote, data)

    def test_ambiguous_create_is_never_retried(self):
        posts = []
        def request(method, url, **kwargs):
            if "/git/ref/tags/" in url:
                return {"object": {"type": "commit", "sha": "1" * 40}}
            if method == "POST":
                posts.append(url)
                raise r.Failure("unknown response")
            return []
        core = SimpleNamespace(Failure=r.Failure)
        remote = SimpleNamespace(api="https://api.github.com/repos/corbet-libs/cfetch", bundle=SimpleNamespace(manifest={"tag": "v0.9.9"}),
                                 http=SimpleNamespace(json=request), github_headers=lambda: {})
        os.environ["GH_TOKEN"] = "fixture-only"
        for _ in range(2):
            with self.assertRaises(r.Failure):
                r.release_object(core, remote, {"producing_commit": "1" * 40}, self.base, create=True)
        self.assertEqual(len(posts), 1)

    def test_duplicate_drafts_fail_before_mutation(self):
        def request(method, url, **kwargs):
            self.assertEqual(method, "GET")
            if "/git/ref/tags/" in url:
                return {"object": {"type": "commit", "sha": "1" * 40}}
            return [{"id": 1, "tag_name": "v0.9.9"}, {"id": 2, "tag_name": "v0.9.9"}]
        remote = SimpleNamespace(api="https://api.github.com/repos/corbet-libs/cfetch", bundle=SimpleNamespace(manifest={"tag": "v0.9.9"}),
                                 http=SimpleNamespace(json=request), github_headers=lambda: {})
        with self.assertRaisesRegex(r.Failure, "Multiple release objects"):
            r.release_object(SimpleNamespace(), remote, {"producing_commit": "1" * 40}, self.base, create=True)

    def test_unexpected_github_asset_blocks_release(self):
        class BaseRemote:
            def __init__(self, bundle):
                self.bundle = bundle
        remote = r.remote_for(SimpleNamespace(Remote=BaseRemote), SimpleNamespace())
        remote.asset_items = lambda: {"unexpected.exe": {}}
        with self.assertRaisesRegex(r.Failure, "Unexpected GitHub release asset"):
            remote.present("github", {"artifacts": {}})

    def test_shared_publisher_preserves_selected_draft(self):
        with patch.dict(os.environ, self.tool_environment):
            core = r.publisher()
        identity = {"producing_commit": "1" * 40}
        bundle = SimpleNamespace(repository=r.REPOSITORY, name="cfetch", version="0.9.9",
                                 manifest={"tag": "v0.9.9", "tag_commit": "1" * 40},
                                 channels={"github": {"identity": identity, "artifacts": {}}})
        calls = []
        def request(method, url, **kwargs):
            calls.append(url)
            self.assertEqual(method, "GET")
            self.assertNotIn("/releases/tags/", url)
            if "/git/ref/tags/" in url:
                return {"object": {"type": "commit", "sha": "1" * 40}}
            if "/assets?" in url:
                return []
            return [{"id": 7, "tag_name": "v0.9.9", "draft": True}]
        remote = r.remote_for(core, bundle)
        remote.http = SimpleNamespace(json=request)
        remote.release = {"id": 7, "tag_name": "v0.9.9", "draft": True}
        result = core.Publisher(bundle, remote, self.base).execute(["github"], publish=False)
        self.assertEqual(result, {"github": {"verified": [], "missing": []}})
        self.assertEqual(remote.release["id"], 7)
        self.assertTrue(remote.release["draft"])
        self.assertTrue(calls)

    def test_failed_upload_reconciles_without_losing_bundle_identity(self):
        with patch.dict(os.environ, self.tool_environment):
            core = r.publisher()
        (self.base / "artifacts").mkdir()
        source = {"version": "0.9.9", "tag": "v0.9.9", "source_commit": "1" * 40}
        unit = "cfetch-0.9.9.crate"
        cargo = {"identity": {"registry": "cargo"}, "artifacts": {unit: b"retained crate bytes"}}
        remote = Mock()
        remote.release = {"id": 7, "draft": False}
        remote.asset.return_value = None
        remote.upload.side_effect = core.Failure("Upload response unavailable")
        remote.present.side_effect = lambda registry, data: set(data["artifacts"]) if registry == "github" or remote.upload.called else set()
        os.environ["RELEASE_BUNDLE_SHA256"] = "a" * 64
        imported = contextlib.nullcontext((self.base, (source, {}, {}, cargo, {})))
        with patch.object(r, "imported", return_value=imported), patch.object(r, "remote_for", return_value=remote), \
                patch.object(r, "release_object", return_value=remote.release), patch("builtins.print"):
            r.publish(self.base, core, "publish", "cargo")
        remote.upload.assert_called_once()
        error = json.loads(next(self.base.rglob("*-error.json")).read_text())
        self.assertEqual(error["bundle_sha256"], "a" * 64)
        completion = json.loads(next(self.base.rglob("*-complete.json")).read_text())
        self.assertEqual(completion["status"], "download-verified")

    def test_matching_error_journal_is_reconciliable(self):
        with patch.dict(os.environ, self.tool_environment):
            core = r.publisher()
        identity = {"registry": "cargo", "producing_commit": "1" * 40}
        bundle = SimpleNamespace(repository=r.REPOSITORY, channels={
            "cargo": {"identity": identity, "artifacts": {"cfetch-0.9.9.crate": b"fixture"}}})
        remote = r.remote_for(core, bundle)
        remote.asset_items = lambda: {"publication-cargo-cfetch-0.9.9.crate-error.json": {}}
        receipt = {"identity": {**identity, "unit": "cfetch-0.9.9.crate"}}
        remote.asset = lambda name: receipt
        self.assertEqual(remote.present("github", {"artifacts": {}}), set())
        receipt["identity"]["producing_commit"] = "2" * 40
        with self.assertRaisesRegex(r.Failure, "Unexpected/conflicting publication journal"):
            remote.present("github", {"artifacts": {}})


if __name__ == "__main__":
    unittest.main()
