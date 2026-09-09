import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pinned_cache import prepare


class PinnedCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.blobs = self.source / "models--test--model" / "blobs"
        self.blobs.mkdir(parents=True)
        self.destination = self.root / "view"
        self.content = b"existing external weights"
        self.sha = hashlib.sha256(self.content).hexdigest()
        self.blob = self.blobs / self.sha
        self.blob.write_bytes(self.content)
        self.manifest = {"repository": "test/model", "revision": "1" * 40, "files": {
            "onnx/model.onnx_data": {"bytes": len(self.content), "sha256": self.sha}}}

    def test_contained_hardlink_reuses_cached_bytes_without_network(self):
        with patch("pinned_cache.subprocess.run", side_effect=AssertionError("network used")):
            observed = prepare(self.source, self.destination, self.manifest)
        repo = self.destination / "models--test--model"
        target = repo / "snapshots" / self.manifest["revision"] / "onnx/model.onnx_data"
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.stat().st_ino, self.blob.stat().st_ino)
        self.assertEqual(target.read_bytes(), self.content)
        self.assertEqual((repo / "refs/main").read_bytes(), b"1" * 40)
        self.assertTrue(target.resolve().is_relative_to(target.parent.resolve()))
        self.assertEqual(observed["files"]["onnx/model.onnx_data"]["sha256"], self.sha)

    def test_corrupt_cached_bytes_fail_without_redownload_or_ref(self):
        self.blob.write_bytes(b"x" * len(self.content))
        with patch("pinned_cache.subprocess.run", side_effect=AssertionError("network used")):
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                prepare(self.source, self.destination, self.manifest)
        self.assertFalse((self.destination / "models--test--model/refs/main").exists())

    def test_external_blob_symlink_is_refused(self):
        outside = self.root / "outside"
        self.blob.rename(outside)
        self.blob.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "escapes the source cache"):
            prepare(self.source, self.destination, self.manifest)

    def test_existing_view_is_never_replaced(self):
        self.destination.mkdir()
        sentinel = self.destination / "sentinel"
        sentinel.write_text("retain")
        with self.assertRaises(FileExistsError):
            prepare(self.source, self.destination, self.manifest)
        self.assertEqual(sentinel.read_text(), "retain")

    def test_missing_large_weights_are_not_downloaded(self):
        self.blob.unlink()
        self.manifest["files"]["onnx/model.onnx_data"]["bytes"] = 100 * 1024 * 1024
        with patch("pinned_cache.subprocess.run", side_effect=AssertionError("network used")):
            with self.assertRaisesRegex(RuntimeError, "retained model file is absent"):
                prepare(self.source, self.destination, self.manifest)

    def test_download_checks_git_blob_identity_before_linking(self):
        config = b'{"kind":"test"}'
        sha1 = hashlib.sha1(f"blob {len(config)}\0".encode() + config).hexdigest()
        self.manifest["files"]["config.json"] = {"bytes": len(config), "git_blob_sha1": sha1}

        def fetch(command, **kwargs):
            self.assertEqual(command[-1], f"https://huggingface.co/test/model/resolve/{'1' * 40}/config.json")
            Path(command[command.index("--output") + 1]).write_bytes(config)

        with patch("pinned_cache.subprocess.run", side_effect=fetch) as download:
            observed = prepare(self.source, self.destination, self.manifest)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(observed["files"]["config.json"]["sha256"], hashlib.sha256(config).hexdigest())
        self.assertEqual((self.blobs / sha1).read_bytes(), config)


if __name__ == "__main__":
    unittest.main()
