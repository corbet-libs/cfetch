"""Governor mechanics for package CI: fake clocks and no native runtime."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from packages.openvino import inference_governor as governor

BOOT = "00000000-0000-0000-0000-000000000001"
OTHER_BOOT = "00000000-0000-0000-0000-000000000002"


class FakeClock:
    def __init__(self):
        self.now = 1_000_000_000
        self.boot = BOOT
        self.waits = []

    def boot_id(self):
        return self.boot

    def monotonic_ns(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += round(seconds * 1_000_000_000)


@unittest.skipUnless(sys.platform == "linux", "Linux governor mechanics")
class GovernorTests(unittest.TestCase):
    @contextmanager
    def provisioned(self, change_policy=None):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "governor"
            directory.mkdir(mode=0o750)
            limits = {"max_operations": 4, "max_charged_buckets": 4096,
                      "max_duration_ns": 100_000_000, "minimum_cooldown_ns": 5_000_000,
                      "cooldown_numerator": 2, "cooldown_denominator": 1}
            policy = {"schema_version": 1, "namespace": "cfetch-host-inference-v1",
                      "epoch_id": "synthetic-test-epoch", "state_directory": str(directory),
                      "lock_wait_ns": 1_000_000_000,
                      "operations": {kind: dict(limits) for kind in governor.KINDS}}
            if change_policy:
                change_policy(policy)
            raw = governor._encode(policy)
            digest = hashlib.sha256(raw).hexdigest()
            clock = FakeClock()
            state = governor.initial_state(digest, policy["epoch_id"], BOOT, clock.now)
            for name, data in (("policy.json", raw), ("operation.lock", b""),
                               ("state.json", governor._encode(state)), ("intent.json", b"")):
                path = directory / name
                path.write_bytes(data)
                path.chmod(0o640 if name == "policy.json" else 0o660)
            # Tests cannot provision root-owned /var state. Only ownership and
            # ancestor ownership/modes are mocked; actual no-follow opens,
            # regular-file checks, link counts, flock, writes and fsync remain.
            with patch.object(governor, "_root_owner"), patch.object(governor, "_directory_metadata"):
                yield directory, digest, clock, policy

    def instance(self, directory, digest, clock):
        return governor.InferenceGovernor(directory, digest, clock=clock)

    def state(self, directory):
        return json.loads((directory / "state.json").read_bytes())

    def test_real_metadata_platform_and_policy_validation_fail_closed(self):
        with self.assertRaisesRegex(governor.GovernorError, "installation-owned"):
            governor._root_owner(types.SimpleNamespace(st_uid=123))
        with self.assertRaisesRegex(governor.GovernorError, "group/world writable"):
            governor._directory_metadata(types.SimpleNamespace(st_uid=0, st_mode=stat.S_IFDIR | 0o777))
        with patch.object(governor.sys, "platform", "unsupported"):
            with self.assertRaisesRegex(governor.GovernorError, "Linux only"):
                governor.InferenceGovernor("/unused", "a" * 64)
        invalid = [
            lambda p: p.update(unrecognized=True),
            lambda p: p.update(lock_wait_ns=0),
            lambda p: p["operations"]["inference"].update(max_operations=True),
            lambda p: p["operations"]["compile"].update(cooldown_denominator=0),
            lambda p: p["operations"]["inference"].update(max_duration_ns=governor.MAX_INT),
        ]
        for change in invalid:
            with self.subTest(change=change), self.provisioned(change) as (directory, digest, clock, _policy):
                call = Mock()
                with self.assertRaises(governor.GovernorError):
                    with self.instance(directory, digest, clock).operation("inference", 32):
                        call()
                call.assert_not_called()
                self.assertEqual((directory / "intent.json").read_bytes(), b"")

    def test_compile_and_each_final_bucket_are_charged_once_with_persistent_cooldown(self):
        with self.provisioned() as (directory, digest, clock, _policy):
            calls = []
            for ordinal, (kind, bucket, duration) in enumerate(
                (("compile", 32, 10_000_000), ("inference", 257, 20_000_000), ("inference", 64, 1_000_000)), 1
            ):
                with self.instance(directory, digest, clock).operation(kind, bucket) as lease:
                    intent = json.loads((directory / "intent.json").read_bytes())
                    self.assertEqual((intent["kind"], intent["bucket"], intent["operation_id"]), (kind, bucket, ordinal))
                    self.assertEqual((lease.kind, lease.bucket, lease.operation_id), (kind, bucket, ordinal))
                    self.assertGreater(lease.timeout_seconds, 0)
                    calls.append((kind, bucket))
                    clock.now += duration
                state = self.state(directory)
                self.assertEqual(state["not_before_ns"], clock.now + 5_000_000 + duration * 2)
                self.assertEqual((directory / "intent.json").read_bytes(), b"")
            self.assertEqual(calls, [("compile", 32), ("inference", 257), ("inference", 64)])
            self.assertEqual(state["usage"]["compile"], {"operations": 1, "charged_buckets": 32})
            self.assertEqual(state["usage"]["inference"], {"operations": 2, "charged_buckets": 321})
            self.assertAlmostEqual(sum(clock.waits), 0.070)

    def test_factory_uses_only_fixed_installation_and_inspects_state_before_return(self):
        change = lambda p: p.update(state_directory=str(governor.INSTALLATION_DIRECTORY))
        with self.provisioned(change) as (directory, digest, clock, _policy):
            def open_directory(path):
                self.assertEqual(path, Path("/var/lib/cfetch/inference"))
                return os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

            with (patch.object(governor, "_open_directory", side_effect=open_directory),
                  patch.object(governor, "LinuxClock", return_value=clock),
                  patch.dict(os.environ, {"CFETCH_INFERENCE_DIRECTORY": "/untrusted/override"})):
                instance = governor.from_installation()
                self.assertEqual(instance.directory, Path("/var/lib/cfetch/inference"))
                self.assertEqual(instance.policy_sha256, digest)
                (directory / "state.json").unlink()
                with self.assertRaises(FileNotFoundError):
                    governor.from_installation()

    def test_epoch_budgets_survive_relaunch_and_large_clock_advance(self):
        for field, limit, bucket in (("max_operations", 1, 32), ("max_charged_buckets", 64, 64)):
            change = lambda p: p["operations"]["inference"].update({field: limit})
            with self.subTest(field=field), self.provisioned(change) as (directory, digest, clock, _policy):
                with self.instance(directory, digest, clock).operation("inference", bucket):
                    pass
                clock.now += 100 * 86_400 * 1_000_000_000
                with self.assertRaisesRegex(governor.GovernorError, "epoch budget exhausted"):
                    with self.instance(directory, digest, clock).operation("inference", bucket):
                        self.fail("exhausted epoch executed")
                with self.instance(directory, digest, clock).operation("compile", 32):
                    pass
                self.assertEqual(self.state(directory)["usage"]["inference"]["operations"], 1)

    def test_native_exception_overrun_and_backward_clock_leave_durable_intent(self):
        for failure in ("exception", "overrun", "backward", "nested"):
            with self.subTest(failure=failure), self.provisioned() as (directory, digest, clock, _policy):
                instance = self.instance(directory, digest, clock)
                with self.assertRaises((governor.GovernorError, RuntimeError)):
                    with instance.operation("inference", 512):
                        if failure == "exception":
                            raise RuntimeError("native operation failed")
                        if failure == "overrun":
                            clock.now += 100_000_001
                        if failure == "backward":
                            clock.now -= 1
                        if failure == "nested":
                            with instance.operation("inference", 32):
                                self.fail("nested operation executed")
                intent = (directory / "intent.json").read_bytes()
                self.assertTrue(intent)
                clock.now = 2_000_000_000
                with self.assertRaisesRegex(governor.GovernorError, "pending native intent"):
                    with self.instance(directory, digest, clock).operation("compile", 32):
                        self.fail("failed native intent was cleared by relaunch")
                self.assertEqual((directory / "intent.json").read_bytes(), intent)
                self.assertEqual(self.state(directory)["usage"]["inference"], {"operations": 1, "charged_buckets": 512})

    def test_boot_change_future_clock_and_corrupt_state_require_external_recovery(self):
        for failure in ("boot", "clock", "state", "intent", "policy", "missing"):
            with self.subTest(failure=failure), self.provisioned() as (directory, digest, clock, _policy):
                if failure == "boot":
                    clock.boot = OTHER_BOOT
                elif failure == "clock":
                    clock.now -= 1
                elif failure == "state":
                    (directory / "state.json").write_bytes(b'{"schema_version":')
                elif failure == "intent":
                    (directory / "intent.json").write_bytes(b"interrupted")
                elif failure == "policy":
                    with (directory / "policy.json").open("ab") as stream:
                        stream.write(b" ")
                else:
                    (directory / "state.json").unlink()
                with self.assertRaises((governor.GovernorError, OSError)):
                    with self.instance(directory, digest, clock).operation("inference", 32):
                        self.fail("unprovisioned or changed state executed")
                if failure == "missing":
                    self.assertFalse((directory / "state.json").exists())

    def test_symlinks_fifos_and_hardlinks_are_rejected_without_native_entry(self):
        for name in ("policy.json", "operation.lock", "state.json", "intent.json"):
            for kind in ("symlink", "fifo", "hardlink"):
                with self.subTest(name=name, kind=kind), self.provisioned() as (directory, digest, clock, _policy):
                    path = directory / name
                    target = directory / "saved-file"
                    path.rename(target)
                    if kind == "symlink":
                        path.symlink_to(target)
                    elif kind == "fifo":
                        os.mkfifo(path, 0o660)
                    else:
                        os.link(target, path)
                    if kind == "fifo":
                        result = subprocess.run(
                            [sys.executable, "-c", (
                                "import sys\nfrom unittest.mock import patch\n"
                                "from packages.openvino import inference_governor as g\n"
                                "with patch.object(g, '_root_owner'), patch.object(g, '_directory_metadata'):\n"
                                " with g.InferenceGovernor(sys.argv[1], sys.argv[2]).operation('inference', 32):\n"
                                "  raise AssertionError('unsafe native entry')\n"
                            ), str(directory), digest],
                            cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=5,
                        )
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("files must be regular", result.stderr)
                    else:
                        with self.assertRaises((governor.GovernorError, OSError)):
                            with self.instance(directory, digest, clock).operation("inference", 32):
                                self.fail("unsafe state file allowed native entry")

    def test_failed_completion_fsync_cannot_clear_pending_intent(self):
        with self.provisioned() as (directory, digest, clock, _policy):
            original_fsync = os.fsync
            state_inode = (directory / "state.json").stat().st_ino

            def fsync(descriptor):
                if os.fstat(descriptor).st_ino == state_inode and self.state(directory)["last_completed"] is not None:
                    raise OSError("completion fsync failed")
                original_fsync(descriptor)

            with patch.object(governor.os, "fsync", side_effect=fsync):
                with self.assertRaisesRegex(OSError, "completion fsync failed"):
                    with self.instance(directory, digest, clock).operation("inference", 128):
                        clock.now += 1_000_000
            self.assertIsNotNone(self.state(directory)["last_completed"])
            self.assertTrue((directory / "intent.json").read_bytes())
            with self.assertRaisesRegex(governor.GovernorError, "pending native intent"):
                with self.instance(directory, digest, clock).operation("inference", 128):
                    self.fail("uncommitted completion was treated as durable")

    def child(self, code, *arguments):
        process = subprocess.Popen(
            [sys.executable, "-c", code, *map(str, arguments)],
            cwd=Path(__file__).resolve().parents[3], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        def cleanup():
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)

        self.addCleanup(cleanup)
        ready, _, _ = select.select([process.stdout], [], [], 5)
        self.assertTrue(ready, "child did not become ready within its bound")
        self.assertEqual(process.stdout.readline(), b"ready\n")
        return process

    def test_other_process_lock_wait_is_bounded_and_crashed_intent_survives(self):
        change = lambda p: p.update(lock_wait_ns=10_000_000)
        with self.provisioned(change) as (directory, digest, clock, _policy):
            holder = self.child(
                "import fcntl, os, sys\n"
                "with open(sys.argv[1], 'r+') as lock:\n"
                " fcntl.flock(lock, fcntl.LOCK_EX)\n"
                " print('ready', flush=True)\n"
                " os.read(0, 1)\n", directory / "operation.lock",
            )
            before = clock.now
            with self.assertRaisesRegex(governor.GovernorError, "lock wait exhausted"):
                with self.instance(directory, digest, clock).operation("inference", 32):
                    self.fail("second process bypassed the shared lock")
            self.assertEqual(clock.now - before, 10_000_000)
            self.assertEqual((directory / "intent.json").read_bytes(), b"")
            holder.stdin.write(b"x")
            holder.stdin.flush()
            holder.wait(timeout=5)
            worker = self.child(
                "import os, sys, time, types\n"
                "from unittest.mock import patch\n"
                "from packages.openvino import inference_governor as g\n"
                f"clock = types.SimpleNamespace(boot_id=lambda: '{BOOT}', monotonic_ns=lambda: 2_000_000_000, sleep=time.sleep)\n"
                "with patch.object(g, '_root_owner'), patch.object(g, '_directory_metadata'):\n"
                " with g.InferenceGovernor(sys.argv[1], sys.argv[2], clock=clock).operation('inference', 64):\n"
                "  print('ready', flush=True)\n"
                "  os.read(0, 1)\n", directory, digest,
            )
            worker.kill()
            worker.wait(timeout=5)
            with self.assertRaisesRegex(governor.GovernorError, "pending native intent"):
                with self.instance(directory, digest, clock).operation("inference", 64):
                    self.fail("process death reset the active intent")


if __name__ == "__main__":
    unittest.main()
