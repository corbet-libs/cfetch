"""Checkpoint behavior tests using only the standard library and a fake engine."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


class FakeTensor:
    shape = (1, 768)

    def astype(self, _dtype):
        return self

    def __pow__(self, _power):
        return self

    def __getitem__(self, _index):
        return types.SimpleNamespace(tolist=lambda: [1.0] + [0.0] * 767)


class FakeOptions:
    def add_session_config_entry(self, *_args):
        pass


def load_runner():
    numpy = types.ModuleType("numpy")
    numpy.asarray = lambda value, dtype: value
    numpy.int64 = "int64"
    numpy.float64 = "float64"
    numpy.isfinite = lambda _value: types.SimpleNamespace(all=lambda: True)
    numpy.sum = lambda _value: 1.0
    ort = types.ModuleType("onnxruntime")
    ort.get_build_info = lambda: "synthetic-test-engine"
    ort.SessionOptions = FakeOptions
    ort.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL="sequential")
    ort.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL="all")
    ort.InferenceSession = Mock()
    tokenizers = types.ModuleType("tokenizers")
    tokenizers.Tokenizer = types.SimpleNamespace(from_file=Mock())
    spec = importlib.util.spec_from_file_location(
        "memory_eval_cpu_under_test", Path(__file__).with_name("embed_cpu.py")
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "numpy": numpy, "onnxruntime": ort, "tokenizers": tokenizers,
    }):
        spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


class CpuCheckpointTests(unittest.TestCase):
    @contextmanager
    def experiment(self, root: Path, input_count: int = 1):
        model = root / "model"
        hashes = {}
        for name in RUNNER.FILES:
            path = model / name
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = b'{"pad_token_id": 0}' if name == "config.json" else name.encode()
            path.write_bytes(raw)
            hashes[name] = hashlib.sha256(raw).hexdigest()
        inputs = []
        for index in range(input_count):
            text = f"title: none | text: Synthetic statement {index}."
            identifier = hashlib.sha256(
                b"cfetch-evaluation-input-v1\0" + text.encode()
            ).hexdigest()
            inputs.append({"id": identifier, "kind": "document", "text": text})
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1, "dimensions": 768,
            "sequence_buckets": RUNNER.BUCKETS, "inputs": inputs,
        }))
        args = argparse.Namespace(
            manifest=manifest, model_root=model, artifact="fp32",
            checkpoints=root / "checkpoint", output=root / "output.json",
        )
        session = Mock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        session.run.return_value = [FakeTensor()]
        tokenizer = Mock()
        tokenizer.encode.return_value = types.SimpleNamespace(ids=[2, 7, 1])
        with (
            patch.object(RUNNER, "FILES", hashes),
            patch.object(RUNNER.ort, "InferenceSession", return_value=session) as engine,
            patch.object(RUNNER.Tokenizer, "from_file", return_value=tokenizer),
            patch.object(RUNNER.importlib.metadata, "version", return_value="test"),
            patch.object(RUNNER.platform, "platform", return_value="synthetic-test-platform"),
            patch.object(RUNNER.time, "sleep"),
            patch("builtins.print"),
        ):
            yield args, inputs, session, engine

    def test_reads_reject_fifo_without_blocking_and_reject_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fifo = root / "record.json"
            os.mkfifo(fifo)
            result = subprocess.run(
                [sys.executable, "-c", (
                    "from pathlib import Path\n"
                    "import sys\n"
                    "from test_embed_cpu import RUNNER\n"
                    "RUNNER.read_json(Path(sys.argv[1]))\n"
                ), str(fifo)],
                cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("bounded regular file", result.stderr)
            target = root / "target.json"
            target.write_text('{"original": true}')
            alias = root / "alias.json"
            alias.symlink_to(target)
            with self.assertRaises(OSError):
                RUNNER.read_json(alias)
            self.assertEqual(target.read_text(), '{"original": true}')

    def test_interrupted_publication_resumes_without_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "record.json"
            with patch.object(RUNNER.os, "link", side_effect=OSError("interrupted publish")):
                with self.assertRaisesRegex(OSError, "interrupted publish"):
                    RUNNER.publish(record, {"value": "unfinished"})
            self.assertFalse(record.exists())
            leftover = root / "record.json.part-interrupted"
            leftover.write_text('{"value":')
            RUNNER.publish(record, {"value": "complete"})
            original = record.read_bytes()
            self.assertEqual(RUNNER.read_json(record), {"value": "complete"})
            with self.assertRaises(FileExistsError):
                RUNNER.publish(record, {"value": "replacement"})
            self.assertEqual(record.read_bytes(), original)
            self.assertEqual(leftover.read_text(), '{"value":')

    def test_writer_lock_rejects_fifo_and_symlink_before_engine_creation(self):
        for kind in ("fifo", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.experiment(root) as (args, _inputs, _session, engine):
                    args.checkpoints.mkdir()
                    lock = args.checkpoints / "writer.lock"
                    if kind == "fifo":
                        os.mkfifo(lock)
                        expected = (ValueError, OSError)
                    else:
                        target = root / "unrelated-file"
                        target.write_text("original")
                        lock.symlink_to(target)
                        expected = OSError
                    with self.assertRaises(expected):
                        RUNNER.run(args)
                    engine.assert_not_called()
                    if kind == "symlink":
                        self.assertEqual(target.read_text(), "original")

    def test_completed_records_without_identity_are_not_adopted(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.experiment(Path(temporary)) as (args, inputs, session, _engine):
                RUNNER.run(args)
                record = args.checkpoints / (inputs[0]["id"] + ".json")
                original = record.read_bytes()
                (args.checkpoints / "identity.json").unlink()
                args.output = args.output.with_name("resumed.json")
                with self.assertRaisesRegex(ValueError, "lack their experiment identity"):
                    RUNNER.run(args)
                self.assertEqual(session.run.call_count, 1)
                self.assertEqual(record.read_bytes(), original)
                self.assertFalse(args.output.exists())

    def test_cross_experiment_identity_and_copied_record_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.experiment(root) as (args, inputs, session, _engine):
                RUNNER.run(args)
                first_record = (args.checkpoints / (inputs[0]["id"] + ".json")).read_bytes()
                args.artifact = "q4"
                args.output = root / "q4.json"
                with self.assertRaisesRegex(ValueError, "different exact experiment"):
                    RUNNER.run(args)
                self.assertEqual(session.run.call_count, 1)
                args.checkpoints = root / "q4-checkpoint"
                RUNNER.run(args)
                self.assertEqual(session.run.call_count, 2)
                q4_record = args.checkpoints / (inputs[0]["id"] + ".json")
                q4_record.write_bytes(first_record)
                args.output = root / "q4-resumed.json"
                with self.assertRaisesRegex(ValueError, "record belongs to a different exact experiment"):
                    RUNNER.run(args)
                self.assertEqual(session.run.call_count, 2)
                self.assertFalse(args.output.exists())

    def test_resume_reuses_completed_input_and_preserves_plain_numeric_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.experiment(Path(temporary), input_count=2) as (args, inputs, session, _engine):
                session.run.side_effect = [[FakeTensor()], RuntimeError("interrupted inference")]
                with self.assertRaisesRegex(RuntimeError, "interrupted inference"):
                    RUNNER.run(args)
                first_record = args.checkpoints / (inputs[0]["id"] + ".json")
                original = first_record.read_bytes()
                self.assertFalse((args.checkpoints / (inputs[1]["id"] + ".json")).exists())
                self.assertFalse(args.output.exists())
                session.run.side_effect = None
                RUNNER.run(args)
                self.assertEqual(session.run.call_count, 3)
                self.assertEqual(first_record.read_bytes(), original)
                bundle = RUNNER.read_json(args.output)
                self.assertEqual([row["id"] for row in bundle["outputs"]], [row["id"] for row in inputs])
                for row in bundle["outputs"]:
                    self.assertEqual(set(row), {"id", "token_count", "bucket", "vector", "error"})
                    self.assertEqual(row["vector"], [1.0] + [0.0] * 767)
                    self.assertEqual(row["token_count"], 3)
                    self.assertEqual(row["bucket"], 32)
                    self.assertIsNone(row["error"])


if __name__ == "__main__":
    unittest.main()
