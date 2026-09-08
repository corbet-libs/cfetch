"""Exercise the real native-call boundaries with a fake compiler and engine."""

from contextlib import contextmanager
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

from packages.openvino import adapter
from packages.openvino.tests.test_adapter import package, scope


class RuntimeGovernanceTests(unittest.TestCase):
    def test_compilation_and_every_invocation_get_their_own_final_bucket_lease(self):
        operations, native_calls = [], []
        active = []

        class Governor:
            @contextmanager
            def operation(self, kind, bucket):
                if active:
                    raise AssertionError("nested actual operations")
                active.append((kind, bucket))
                operations.append((kind, bucket))
                try:
                    yield SimpleNamespace(deadline_ns=time.monotonic_ns() + 5_000_000_000)
                finally:
                    active.clear()

        class Output:
            shape = (1, 768)

            def __getitem__(self, index):
                if index != 0:
                    raise AssertionError("expected one native row")
                return self

            def astype(self, *_args, **_kwargs):
                return self

            def tolist(self):
                return [1.0] + [0.0] * 767

        class Model:
            def clone(self):
                return Model()

            def reshape(self, shapes):
                widths = {shape[1] for shape in shapes.values()}
                if len(widths) != 1:
                    raise AssertionError("native inputs disagree")
                self.bucket = widths.pop()

        selected = scope()

        class Compiled:
            def __init__(self, bucket):
                self.bucket = bucket

            def get_property(self, name):
                if name != "EXECUTION_DEVICES":
                    raise AssertionError(name)
                return selected.required_execution_devices

            def output(self, name):
                return name

            def __call__(self, inputs):
                if active != [("inference", self.bucket)]:
                    raise AssertionError("native invocation escaped its lease")
                if any(len(value) != 1 or len(value[0]) != self.bucket for value in inputs.values()):
                    raise AssertionError("governor did not charge the final padded bucket")
                native_calls.append(self.bucket)
                return {"embedding": Output()}

        class Core:
            def compile_model(self, model, device, config):
                if active != [("compile", model.bucket)] or device != selected.openvino_device:
                    raise AssertionError("compilation escaped its separate lease")
                return Compiled(model.bucket)

        engine = adapter.OpenVinoEngine.__new__(adapter.OpenVinoEngine)
        engine._governor = Governor()
        engine._model, engine._core = Model(), Core()
        engine._scope, engine._artifact = selected, package(selected).artifact
        engine._compiled, engine._execution_devices = {}, {}
        engine._np = SimpleNamespace(asarray=lambda value, **_kwargs: value,
                                     int64="int64", float32="float32")
        for bucket in (32, 64, 32):
            vector = engine.embed([2, 7, 1] + [0] * (bucket - 3),
                                  [1] * 3 + [0] * (bucket - 3), bucket)
            self.assertEqual(len(vector), 768)
        self.assertEqual(native_calls, [32, 64, 32])
        self.assertEqual(operations, [("compile", 32), ("inference", 32),
                                      ("compile", 64), ("inference", 64), ("inference", 32)])

    def test_unprovisioned_engine_refuses_before_importing_native_runtime(self):
        with patch.object(adapter, "from_installation", side_effect=RuntimeError("unprovisioned host")):
            with self.assertRaisesRegex(RuntimeError, "unprovisioned host"):
                adapter.OpenVinoEngine(None, None)

    def test_scope_must_bind_the_exact_installed_governor_policy(self):
        selected = scope()
        with patch.object(adapter, "from_installation", return_value=SimpleNamespace(policy_sha256="1" * 64)):
            with self.assertRaisesRegex(RuntimeError, "does not bind"):
                adapter.OpenVinoEngine(package(selected), selected)


if __name__ == "__main__":
    unittest.main()
