#!/usr/bin/env python3
"""Capture native variant Cargo commands without compiling or probing hardware."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent


class NativeVariantCommandTests(unittest.TestCase):
    def run_variants(self, rows, *, architecture="x86_64", cargo_exit=0):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            tools = root / "bin"
            tools.mkdir()
            shutil.copyfile(ROOT / "scripts/ci-check.sh", root / "scripts/ci-check.sh")
            # Isolate command routing from the separately tested matrix validator.
            (root / "scripts/variant-matrix.sh").write_text(
                '#!/bin/sh\nprintf \'%s\\n\' "$CFETCH_TEST_MATRIX"\n'
            )
            (tools / "uname").write_text(
                '#!/bin/sh\ncase "$1" in\n'
                '  -s) printf \'Linux\\n\' ;;\n'
                '  -m) printf \'%s\\n\' "$CFETCH_TEST_ARCH" ;;\n'
                '  *) exit 2 ;;\nesac\n'
            )
            (tools / "cargo").write_text(
                '#!/usr/bin/env python3\nimport json, os, sys\n'
                'with open(os.environ["CFETCH_TEST_CAPTURE"], "a") as stream:\n'
                '    stream.write(json.dumps({"variant": os.environ["CFETCH_VARIANT"], '
                '"args": sys.argv[1:]}) + "\\n")\n'
                'raise SystemExit(int(os.environ["CFETCH_TEST_CARGO_EXIT"]))\n'
            )
            for tool in tools.iterdir():
                tool.chmod(0o700)
            capture = root / "commands.jsonl"
            result = subprocess.run(
                ["bash", str(root / "scripts/ci-check.sh"), "variants"],
                env={**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                     "CFETCH_TEST_MATRIX": json.dumps({"include": rows}),
                     "CFETCH_TEST_ARCH": architecture,
                     "CFETCH_TEST_CAPTURE": str(capture),
                     "CFETCH_TEST_CARGO_EXIT": str(cargo_exit)},
                capture_output=True, text=True, timeout=10,
            )
            calls = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
            return result, calls

    def test_preserves_target_features_and_variant_for_native_rows_only(self):
        rows = [
            {"id": "native-explicit", "os": "linux", "arch": "x86_64",
             "target": "x86_64-unknown-linux-gnu", "cargo_features": "feature-one,feature-two"},
            {"id": "native-default", "os": "linux", "arch": "x86_64",
             "target": "", "cargo_features": ""},
            {"id": "other-arch", "os": "linux", "arch": "aarch64",
             "target": "aarch64-unknown-linux-gnu", "cargo_features": ""},
            {"id": "other-os", "os": "mac", "arch": "x86_64",
             "target": "x86_64-apple-darwin", "cargo_features": ""},
        ]
        result, calls = self.run_variants(rows)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [
            {"variant": "native-explicit", "args": ["check", "--release", "--locked",
             "--target", "x86_64-unknown-linux-gnu", "--features", "feature-one,feature-two"]},
            {"variant": "native-default", "args": ["check", "--release", "--locked", "--features", ""]},
        ])

    def test_no_matching_native_architecture_runs_no_cargo(self):
        result, calls = self.run_variants([
            {"id": "other-arch", "os": "linux", "arch": "aarch64", "target": "", "cargo_features": ""},
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No native Linux variant", result.stderr)
        self.assertEqual(calls, [])

    def test_failed_variant_stops_without_retrying_or_running_later_variants(self):
        rows = [{"id": name, "os": "linux", "arch": "x86_64", "target": "", "cargo_features": ""}
                for name in ("first", "second")]
        result, calls = self.run_variants(rows, cargo_exit=9)
        self.assertEqual(result.returncode, 9)
        self.assertEqual([call["variant"] for call in calls], ["first"])


if __name__ == "__main__":
    unittest.main()
