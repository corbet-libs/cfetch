"""Owned-process deadline for supervised native calls on Linux.

SIGALRM retains its kernel default termination action: a native call holding
the GIL cannot delay expiry through Python handler dispatch. The caller's
durable governor intent must surround this guard. This cannot recover a
wedged device or SoC and does not signal any other process.

ITIMER_REAL is a relative timer. Arming/setup time, timer resolution, and
scheduling can delay termination beyond the requested monotonic deadline;
this is not a nanosecond-precise absolute timer. Entry and return checks reject
observed overruns, while the armed kernel alarm bounds a stalled native call.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import signal
import sys
import threading
import time
from typing import Iterator


@contextmanager
def NativeDeadline(deadline_ns: int) -> Iterator[None]:
    """Bound native work with a relative kernel timer and monotonic checks."""
    required = ("SIGALRM", "ITIMER_REAL", "getitimer", "setitimer", "pthread_sigmask")
    if sys.platform != "linux" or not all(hasattr(signal, name) for name in required):
        raise RuntimeError("native deadline requires Linux interval timers")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("native deadline requires the main thread")
    if type(deadline_ns) is not int or deadline_ns <= 0:
        raise RuntimeError("native deadline must be a positive monotonic nanosecond integer")
    try:
        if signal.getsignal(signal.SIGALRM) != signal.SIG_DFL:
            raise RuntimeError("SIGALRM is reserved by an existing handler")
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError("ITIMER_REAL is already reserved")
        if signal.SIGALRM in signal.pthread_sigmask(signal.SIG_BLOCK, []):
            raise RuntimeError("SIGALRM is blocked in the main thread")
    except (OSError, ValueError) as error:
        raise RuntimeError("cannot establish native deadline signal prerequisites") from error

    remaining_ns = deadline_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        raise RuntimeError("native deadline has already expired")
    try:
        seconds = remaining_ns / 1_000_000_000
        if not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError("native deadline duration is not representable")
        previous = signal.setitimer(signal.ITIMER_REAL, seconds, 0.0)
    except (OSError, ValueError, OverflowError) as error:
        raise RuntimeError("cannot arm native deadline timer") from error
    entered = False
    try:
        # setitimer returns the displaced timer atomically. Detect another
        # timer owner appearing between the prerequisite check and arming;
        # never silently adopt or restore that competing reservation.
        if previous != (0.0, 0.0):
            raise RuntimeError("ITIMER_REAL became reserved while arming the deadline")
        if time.monotonic_ns() >= deadline_ns:
            raise RuntimeError("native deadline expired before native entry")
        entered = True
        yield
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
        except (OSError, ValueError) as error:
            raise RuntimeError("cannot cancel native deadline timer") from error
        # The kernel may round a sub-microsecond duration upwards. Even if
        # native code returns in that interval, never accept an overrun.
        if entered and time.monotonic_ns() >= deadline_ns:
            raise RuntimeError("native call exceeded its monotonic deadline")
