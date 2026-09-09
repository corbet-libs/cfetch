"""Linux governor for one native compile or inference per operation context.

Provisioning is external and privileged: expose the SAME host-local directory
to every participating adapter/user/namespace. Its complete directory ancestry
must be root-owned and not group/world writable. Precreate root-owned policy.json
(not group/world writable), operation.lock, state.json, and empty intent.json.
The last three may be group writable for a dedicated, trusted adapter group;
their group must match the directory. The directory itself stays non-writable,
so participants cannot replace the lock inode. Cooperating group members can
write state; this primitive is not a sandbox against malicious authorized users.

Pin the exact policy bytes. An explicit epoch has finite compile/inference call
and summed-bucket budgets; nothing resets them on restart, midnight, or reboot.
initial_state returns provisioning data only: this module NEVER creates files,
initializes missing state, clears failed intents, or recovers a changed boot.

Wrap ONLY the actual native call, after final padding. Compile independently,
outside an inference context. Nested contexts are refused. A supervising parent
must enforce Lease.deadline_ns by terminating and confirming its owned worker's
exit: synchronous Python cannot interrupt a stuck native call. The durable intent
then survives flock release on process death and blocks further operations.
Successful completion/cooldown is fsynced before intent is cleared. Torn writes,
native exceptions, overruns, boot changes, and backward clocks fail closed.
Each policy must reserve at least as much idle time as active time, including
at its maximum permitted call duration. This limits the active/idle duty cycle
to 50%; it does not cap instantaneous device utilization or other programs.
No supplied policy is implied safe for an NPU or sufficient to lift quarantine.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import uuid

MAX_BYTES = 16 * 1024
MAX_INT = (1 << 63) - 1
BUCKETS = (32, 64, 128, 257, 512, 1024, 2048)
KINDS = ("compile", "inference")
INSTALLATION_DIRECTORY = Path("/var/lib/cfetch/inference")


class GovernorError(RuntimeError):
    """Work was refused; no automatic reset or alternative namespace is allowed."""


def _require(condition, message):
    if not condition:
        raise GovernorError(message)


def _integer(value, minimum=0):
    _require(type(value) is int and minimum <= value <= MAX_INT, "invalid bounded integer")
    return value


def _keys(value, names):
    _require(type(value) is dict and set(value) == set(names), "invalid governor schema")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate governor JSON key")
        result[key] = value
    return result


def _encode(value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    _require(len(raw) <= MAX_BYTES, "governor record exceeds its bound")
    return raw


def _root_owner(metadata):
    _require(metadata.st_uid == 0, "governor namespace must be installation-owned")


def _directory_metadata(metadata):
    _root_owner(metadata)
    _require(stat.S_ISDIR(metadata.st_mode) and metadata.st_mode & 0o022 == 0,
             "governor directory ancestry must not be group/world writable")


def _open_directory(path):
    _require(path.is_absolute() and ".." not in path.parts, "governor directory must be absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        _directory_metadata(os.fstat(descriptor))
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            _directory_metadata(os.fstat(descriptor))
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_file(directory, name, writable):
    flags = (os.O_RDWR if writable else os.O_RDONLY) | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = os.open(name, flags, dir_fd=directory)
    try:
        metadata = os.fstat(descriptor)
        _root_owner(metadata)
        _require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
                 "governor files must be regular and have exactly one link")
        _require(metadata.st_gid == os.fstat(directory).st_gid and metadata.st_mode & 0o002 == 0,
                 "governor files must use the installation group and reject world writes")
        _require(writable or metadata.st_mode & 0o022 == 0, "policy must be installation-writable only")
        _require(metadata.st_size <= MAX_BYTES, "governor file exceeds its size bound")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read(descriptor):
    os.lseek(descriptor, 0, os.SEEK_SET)
    result = bytearray()
    while chunk := os.read(descriptor, MAX_BYTES + 1 - len(result)):
        result.extend(chunk)
        _require(len(result) <= MAX_BYTES, "governor file exceeds its size bound")
    return bytes(result)


def _parse(raw):
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda value: _require(False, "nonfinite governor JSON"))
    except (ValueError, UnicodeError) as error:
        raise GovernorError("invalid governor JSON; external recovery required") from error


def _json(descriptor):
    return _parse(_read(descriptor))


def _write(descriptor, raw):
    _require(len(raw) <= MAX_BYTES, "governor record exceeds its bound")
    os.lseek(descriptor, 0, os.SEEK_SET)
    position = 0
    while position < len(raw):
        written = os.write(descriptor, raw[position:])
        _require(written > 0, "governor write made no progress")
        position += written
    os.ftruncate(descriptor, len(raw))
    os.fsync(descriptor)


def _boot(value):
    try:
        _require(type(value) is str and str(uuid.UUID(value)) == value, "invalid boot identity")
    except (ValueError, AttributeError) as error:
        raise GovernorError("invalid boot identity") from error
    return value


def initial_state(policy_sha256, epoch_id, boot_id, monotonic_ns):
    """Return data for an explicit external provisioning/recovery operation."""
    _require(type(policy_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", policy_sha256), "invalid policy digest")
    _require(type(epoch_id) is str and re.fullmatch(r"[a-zA-Z0-9._-]{1,128}", epoch_id), "invalid epoch")
    return {"schema_version": 1, "policy_sha256": policy_sha256, "epoch_id": epoch_id,
            "boot_id": _boot(boot_id), "last_observed_ns": _integer(monotonic_ns),
            "not_before_ns": monotonic_ns, "last_completed": None,
            "usage": {kind: {"operations": 0, "charged_buckets": 0} for kind in KINDS}}


class LinuxClock:
    def boot_id(self):
        descriptor = os.open("/proc/sys/kernel/random/boot_id", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            _require(stat.S_ISREG(os.fstat(descriptor).st_mode), "invalid boot identity source")
            return _boot(os.read(descriptor, 128).decode("ascii").strip())
        finally:
            os.close(descriptor)

    def monotonic_ns(self):
        return time.monotonic_ns()

    def sleep(self, seconds):
        time.sleep(seconds)


@dataclass(frozen=True)
class Lease:
    kind: str
    bucket: int
    operation_id: int
    deadline_ns: int
    timeout_seconds: float


class InferenceGovernor:
    def __init__(self, state_directory, pinned_policy_sha256, *, clock=None):
        _require(sys.platform == "linux", "host inference governor supports Linux only")
        _require(type(pinned_policy_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", pinned_policy_sha256),
                 "invalid pinned policy digest")
        self.directory = Path(state_directory)
        self.policy_sha256 = pinned_policy_sha256
        self.clock = clock or LinuxClock()
        self._last_clock = None
        self._entered = False

    def _now(self):
        now = _integer(self.clock.monotonic_ns())
        _require(self._last_clock is None or now >= self._last_clock, "monotonic clock moved backwards")
        self._last_clock = now
        return now

    def _policy(self, descriptor):
        raw = _read(descriptor)
        _require(hashlib.sha256(raw).hexdigest() == self.policy_sha256, "pinned governor policy changed")
        policy = _parse(raw)
        _keys(policy, ("schema_version", "namespace", "epoch_id", "state_directory", "lock_wait_ns", "operations"))
        _require(type(policy["schema_version"]) is int and policy["schema_version"] == 1
                 and policy["namespace"] == "cfetch-host-inference-v1", "unsupported governor policy")
        _require(policy["state_directory"] == str(self.directory), "policy belongs to another namespace directory")
        initial_state(self.policy_sha256, policy["epoch_id"], self.clock.boot_id(), 0)
        _integer(policy["lock_wait_ns"], 1)
        _keys(policy["operations"], KINDS)
        for limits in policy["operations"].values():
            _keys(limits, ("max_operations", "max_charged_buckets", "max_duration_ns",
                           "minimum_cooldown_ns", "cooldown_numerator", "cooldown_denominator"))
            for key, value in limits.items():
                _integer(value, 0 if key == "cooldown_numerator" else 1)
            # For cooldown = minimum + duration * numerator / denominator,
            # the maximum duration is the worst case when the ratio is < 1.
            # Fixed cooling may satisfy the bound without a proportional part.
            _require(
                limits["minimum_cooldown_ns"] * limits["cooldown_denominator"]
                + limits["max_duration_ns"] * limits["cooldown_numerator"]
                >= limits["max_duration_ns"] * limits["cooldown_denominator"],
                "governor policy exceeds the 50% background duty-cycle limit",
            )
            _integer(limits["max_duration_ns"] + limits["minimum_cooldown_ns"] + (
                limits["max_duration_ns"] * limits["cooldown_numerator"] + limits["cooldown_denominator"] - 1
            ) // limits["cooldown_denominator"])
        return policy

    def _state(self, descriptor, policy, now):
        state = _json(descriptor)
        _keys(state, initial_state(self.policy_sha256, policy["epoch_id"], self.clock.boot_id(), now))
        _require(type(state["schema_version"]) is int and state["schema_version"] == 1
                 and state["policy_sha256"] == self.policy_sha256 and state["epoch_id"] == policy["epoch_id"],
                 "governor state belongs to another policy or epoch")
        _require(_boot(state["boot_id"]) == self.clock.boot_id(), "boot changed; external recovery required")
        _require(_integer(state["last_observed_ns"]) <= now, "state observes a later monotonic clock")
        _require(_integer(state["not_before_ns"]) >= state["last_observed_ns"], "invalid persisted cooldown")
        _keys(state["usage"], KINDS)
        total = 0
        for kind, usage in state["usage"].items():
            _keys(usage, ("operations", "charged_buckets"))
            count, charged = _integer(usage["operations"]), _integer(usage["charged_buckets"])
            _require(32 * count <= charged <= 2048 * count, "inconsistent charged bucket counter")
            _require(count <= policy["operations"][kind]["max_operations"]
                     and charged <= policy["operations"][kind]["max_charged_buckets"], "persisted budget exceeds policy")
            total += count
        completed = state["last_completed"]
        if completed is None:
            _require(total == 0, "charged operations lack durable completion")
        else:
            _keys(completed, ("kind", "bucket", "total_operations", "started_ns", "finished_ns", "cooldown_until_ns"))
            _require(completed["kind"] in KINDS and type(completed["bucket"]) is int
                     and completed["bucket"] in BUCKETS, "invalid completed native operation")
            _require(_integer(completed["total_operations"], 1) == total, "completion counter mismatch")
            _require(_integer(completed["started_ns"]) <= _integer(completed["finished_ns"]) == state["last_observed_ns"]
                     and _integer(completed["cooldown_until_ns"]) == state["not_before_ns"], "completion clock mismatch")
            limits = policy["operations"][completed["kind"]]
            elapsed = completed["finished_ns"] - completed["started_ns"]
            expected_cooldown = limits["minimum_cooldown_ns"] + (
                elapsed * limits["cooldown_numerator"] + limits["cooldown_denominator"] - 1
            ) // limits["cooldown_denominator"]
            _require(state["usage"][completed["kind"]]["operations"] > 0
                     and elapsed <= limits["max_duration_ns"]
                     and state["not_before_ns"] == completed["finished_ns"] + expected_cooldown,
                     "completed duration or cooldown violates pinned policy")
        return state

    @contextmanager
    def _locked(self):
        import fcntl

        with ExitStack() as stack:
            directory = _open_directory(self.directory)
            stack.callback(os.close, directory)
            handles = {}
            for name in ("policy.json", "operation.lock", "state.json", "intent.json"):
                handles[name] = _open_file(directory, name, name != "policy.json")
                stack.callback(os.close, handles[name])
            policy = self._policy(handles["policy.json"])
            wait_deadline = _integer(self._now() + policy["lock_wait_ns"])
            while True:
                try:
                    fcntl.flock(handles["operation.lock"], fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = wait_deadline - self._now()
                    _require(remaining > 0, "governor lock wait exhausted")
                    self.clock.sleep(min(remaining / 1_000_000_000, 0.05))
            _require(not _read(handles["intent.json"]), "pending native intent; external recovery required")
            policy = self._policy(handles["policy.json"])
            state = self._state(handles["state.json"], policy, self._now())
            _require(self._now() <= wait_deadline, "governor lock wait exhausted")
            yield policy, handles, state, wait_deadline

    @contextmanager
    def operation(self, kind, bucket):
        _require(kind in KINDS and type(bucket) is int and bucket in BUCKETS, "invalid native operation or final bucket")
        _require(not self._entered, "nested native operation contexts are forbidden")
        self._entered = True
        try:
            with self._locked() as (policy, handles, state, wait_deadline):
                while True:
                    now = self._now()
                    if now >= state["not_before_ns"]:
                        break
                    _require(state["not_before_ns"] <= wait_deadline, "cooldown exceeds bounded governor wait")
                    self.clock.sleep(min((state["not_before_ns"] - now) / 1_000_000_000, 0.05))
                limits, usage = policy["operations"][kind], state["usage"][kind]
                _require(usage["operations"] < limits["max_operations"]
                         and usage["charged_buckets"] + bucket <= limits["max_charged_buckets"], "provisioned epoch budget exhausted")
                intent_ns = self._now()
                _require(intent_ns <= wait_deadline, "governor lock wait exhausted")
                deadline = _integer(intent_ns + limits["max_duration_ns"])
                usage["operations"] += 1
                usage["charged_buckets"] += bucket
                ordinal = _integer(sum(item["operations"] for item in state["usage"].values()), 1)
                intent = {"schema_version": 1, "policy_sha256": self.policy_sha256,
                          "epoch_id": policy["epoch_id"], "boot_id": state["boot_id"],
                          "kind": kind, "bucket": bucket, "operation_id": ordinal,
                          "intent_ns": intent_ns, "deadline_ns": deadline,
                          "pid": os.getpid(), "uid": os.geteuid()}
                _write(handles["intent.json"], _encode(intent))
                state["last_observed_ns"] = intent_ns
                _write(handles["state.json"], _encode(state))
                started = self._now()
                _require(started < deadline, "native deadline expired before entry")
                yield Lease(kind, bucket, ordinal, deadline, (deadline - started) / 1_000_000_000)
                finished = self._now()
                _require(finished <= deadline, "native operation exceeded its deadline")
                _require(self.clock.boot_id() == state["boot_id"], "boot changed during native operation")
                ratio = ((finished - started) * limits["cooldown_numerator"] + limits["cooldown_denominator"] - 1) // limits["cooldown_denominator"]
                cooldown = _integer(finished + limits["minimum_cooldown_ns"] + ratio)
                state["last_observed_ns"], state["not_before_ns"] = finished, cooldown
                state["last_completed"] = {"kind": kind, "bucket": bucket, "total_operations": ordinal,
                                           "started_ns": started, "finished_ns": finished, "cooldown_until_ns": cooldown}
                _write(handles["state.json"], _encode(state))
                _write(handles["intent.json"], b"")
        finally:
            self._entered = False


def from_installation():
    """Inspect the fixed, privileged Linux namespace; never provision or reset it."""
    _require(sys.platform == "linux", "host inference governor supports Linux only")
    with ExitStack() as stack:
        directory = _open_directory(INSTALLATION_DIRECTORY)
        stack.callback(os.close, directory)
        policy = _open_file(directory, "policy.json", False)
        stack.callback(os.close, policy)
        instance = InferenceGovernor(INSTALLATION_DIRECTORY, hashlib.sha256(_read(policy)).hexdigest())
    with instance._locked():
        pass
    return instance
