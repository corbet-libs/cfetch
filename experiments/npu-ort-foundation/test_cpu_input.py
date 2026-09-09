"""Exercise input rejection before any native runtime or model is loaded."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class CpuInputBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.probe = Path(os.environ["CFETCH_FOUNDATION_PROBE_DIR"]) / "cfetch-npu-ort-foundation"
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def refused(self, path, expected, device="CPU", model="EmbeddingGemma300M"):
        result = subprocess.run(
            [str(self.probe), "--ort", str(self.root / "absent-runtime"),
             "--device", device, "--run-model", model, "--cpu-input", str(path)],
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 1, result)
        self.assertIn(expected, result.stderr)
        self.assertNotIn("ort_version=", result.stdout)

    def test_npu_rejected_before_input_or_runtime_is_opened(self):
        self.refused(self.root / "absent-input", "requires CPU and EmbeddingGemma300M", device="NPU")

    def test_other_model_cannot_extend_fixed_canary(self):
        self.refused(self.root / "absent-input", "requires CPU and EmbeddingGemma300M", model="AllMiniLML6V2")

    def test_fifo_rejected_without_waiting_for_a_writer(self):
        path = self.root / "fifo"
        os.mkfifo(path)
        self.refused(path, "must be a regular file")

    def test_oversized_fixture_is_rejected(self):
        path = self.root / "oversized.json"
        path.write_bytes(b" " * 131073)
        self.refused(path, "exceeds 128 KiB")

    def test_out_of_range_token_count_is_rejected(self):
        path = self.root / "fixture.json"
        for tokens in (0, 2049):
            with self.subTest(tokens=tokens):
                path.write_text(json.dumps({"schema_version": 1, "id": "case", "text": "text",
                                           "tokens": tokens, "token_ids_sha256": "a" * 64}))
                self.refused(path, "1..2048 tokens")


if __name__ == "__main__":
    unittest.main()
