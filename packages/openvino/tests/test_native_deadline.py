from __future__ import annotations

from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
import unittest


REPOSITORY = Path(__file__).resolve().parents[3]
PRELUDE = "from packages.openvino.native_deadline import NativeDeadline\nimport signal, time\n"


@unittest.skipUnless(sys.platform == "linux", "kernel interval-timer contract is Linux-only")
class NativeDeadlineTests(unittest.TestCase):
    def child(self, source):
        return subprocess.run(
            [sys.executable, "-c", PRELUDE + textwrap.dedent(source)],
            cwd=REPOSITORY, capture_output=True, text=True, timeout=5,
        )

    def assert_success(self, source):
        result = self.child(source)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_kernel_alarm_terminates_gil_holding_call_and_unrelated_child_survives(self):
        # The unrelated process waits for this test's explicit completion
        # message, so its lifetime cannot race the deadline assertion.
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import sys; assert sys.stdin.readline() == 'finish\\n'; print('alive')"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            start = time.monotonic()
            result = self.child("""
                import ctypes
                # PyDLL deliberately retains the GIL throughout this libc
                # call. A Python signal callback could not dispatch here.
                libc = ctypes.PyDLL(None)
                libc.sleep.argtypes = [ctypes.c_uint]
                libc.sleep.restype = ctypes.c_uint
                with NativeDeadline(time.monotonic_ns() + 200_000_000):
                    libc.sleep(30)
                raise AssertionError('native call survived its deadline')
            """)
            self.assertEqual(result.returncode, -signal.SIGALRM, result.stderr)
            self.assertLess(time.monotonic() - start, 5)
            self.assertIsNone(unrelated.poll())
            stdout, stderr = unrelated.communicate("finish\n", timeout=3)
            self.assertEqual(unrelated.returncode, 0, stderr)
            self.assertEqual(stdout.strip(), "alive")
        finally:
            if unrelated.poll() is None:
                unrelated.kill()
                unrelated.communicate(timeout=3)

    def test_normal_exit_cancels_alarm_and_retains_default_disposition(self):
        self.assert_success("""
            with NativeDeadline(time.monotonic_ns() + 200_000_000):
                remaining, interval = signal.getitimer(signal.ITIMER_REAL)
                assert remaining > 0 and interval == 0
                assert signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
            assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
            time.sleep(0.3)
            assert signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
        """)

    def test_exception_exit_cancels_alarm_without_swallowing_body_error(self):
        self.assert_success("""
            try:
                with NativeDeadline(time.monotonic_ns() + 200_000_000):
                    raise ValueError('body error')
            except ValueError as error:
                assert str(error) == 'body error'
            else:
                raise AssertionError('body error was swallowed')
            assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
            time.sleep(0.3)
        """)

    def test_nested_guard_refuses_without_cancelling_outer_timer(self):
        self.assert_success("""
            with NativeDeadline(time.monotonic_ns() + 10_000_000_000):
                try:
                    with NativeDeadline(time.monotonic_ns() + 20_000_000_000):
                        raise AssertionError('nested body entered')
                except RuntimeError as error:
                    assert 'reserved' in str(error)
                assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
            assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        """)

    def test_reserved_timer_is_rejected_and_not_replaced(self):
        self.assert_success("""
            signal.setitimer(signal.ITIMER_REAL, 10.0, 2.0)
            try:
                try:
                    with NativeDeadline(time.monotonic_ns() + 20_000_000_000):
                        raise AssertionError('reserved-timer body entered')
                except RuntimeError as error:
                    assert 'reserved' in str(error)
                remaining, interval = signal.getitimer(signal.ITIMER_REAL)
                assert 0 < remaining <= 10 and interval == 2
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
        """)

    def test_nondefault_handlers_and_blocked_alarm_are_rejected(self):
        self.assert_success("""
            for handler in (signal.SIG_IGN, lambda signum, frame: None):
                signal.signal(signal.SIGALRM, handler)
                try:
                    with NativeDeadline(time.monotonic_ns() + 1_000_000_000):
                        raise AssertionError('handler-reserved body entered')
                except RuntimeError as error:
                    assert 'handler' in str(error)
                assert signal.getsignal(signal.SIGALRM) == handler
                assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGALRM])
            try:
                try:
                    with NativeDeadline(time.monotonic_ns() + 1_000_000_000):
                        raise AssertionError('blocked-alarm body entered')
                except RuntimeError as error:
                    assert 'blocked' in str(error)
                assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        """)

    def test_worker_thread_is_rejected_before_any_timer_is_armed(self):
        self.assert_success("""
            import threading
            errors = []
            def work():
                try:
                    with NativeDeadline(time.monotonic_ns() + 1_000_000_000):
                        errors.append('entered')
                except RuntimeError as error:
                    errors.append(str(error))
            thread = threading.Thread(target=work)
            thread.start()
            thread.join(timeout=1)
            assert not thread.is_alive()
            assert len(errors) == 1 and 'main thread' in errors[0]
            assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        """)

    def test_invalid_expired_unrepresentable_and_unsupported_deadlines_fail_closed(self):
        self.assert_success("""
            from unittest.mock import patch
            for deadline in (None, True, 1.5, '123', 0, -1, time.monotonic_ns() - 1, 10**1000):
                try:
                    with NativeDeadline(deadline):
                        raise AssertionError('invalid deadline body entered')
                except RuntimeError:
                    pass
                assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
            with patch('packages.openvino.native_deadline.sys.platform', 'unsupported'):
                try:
                    with NativeDeadline(time.monotonic_ns() + 1_000_000_000):
                        raise AssertionError('unsupported-platform body entered')
                except RuntimeError as error:
                    assert 'Linux' in str(error)
            assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        """)

    def test_deadline_expiring_during_arming_prevents_entry_and_cancels_timer(self):
        self.assert_success("""
            from unittest.mock import patch
            entered = False
            # Remaining-time calculation sees a one-second budget, but the
            # first clock read after the real timer is armed sees expiry.
            with patch('packages.openvino.native_deadline.time.monotonic_ns', side_effect=[0, 1_000_000_000]):
                try:
                    with NativeDeadline(1_000_000_000):
                        entered = True
                except RuntimeError as error:
                    assert 'before native entry' in str(error)
                else:
                    raise AssertionError('expired deadline allowed entry')
            assert not entered
            assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        """)

    def test_monotonic_overrun_is_rejected_even_before_kernel_delivery(self):
        self.assert_success("""
            from unittest.mock import patch
            # Arm a real one-second timer, but make the post-call monotonic
            # check land exactly on the deadline before the kernel alarm.
            with patch('packages.openvino.native_deadline.time.monotonic_ns', side_effect=[0, 0, 1_000_000_000]):
                try:
                    with NativeDeadline(1_000_000_000):
                        pass
                except RuntimeError as error:
                    assert 'exceeded' in str(error)
                else:
                    raise AssertionError('deadline equality accepted')
            assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        """)


if __name__ == "__main__":
    unittest.main()
