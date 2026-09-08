"""Bounded, immutable working journal for physical evidence, never an admission.

The exclusive Linux flock covers an entire collection. Each request nonce is
durably reserved before HTTP, and each verified transaction is durably published
before returning to the collector. Only complete attempts can be reused; this
module deliberately has no per-request resume API. Interrupted attempts and
publication temporaries are retained. The collector must revalidate signatures,
input order, live evidence, and regenerated summaries before reusing anything.
The directory is trusted local working storage, not a signed timing authority.
"""

from __future__ import annotations

import hashlib
import ctypes
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any


MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_FILES = 100_000
HEX = re.compile(r"[0-9a-f]{64}")
KEY = re.compile(r"(?:bucket|wire)-[0-9]{1,4}")


class CheckpointError(ValueError):
    pass


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8") + b"\n"


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError("duplicate checkpoint JSON key")
        result[key] = value
    return result


def rename_new(source: str | Path, destination: str | Path, *, directory_fd: int = -100):
    """Linux atomic rename which refuses even an existing empty destination."""
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if function is None:
        raise CheckpointError("atomic no-overwrite publication requires Linux renameat2")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                         ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(directory_fd, os.fsencode(source), directory_fd,
                os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


class Attempt:
    def __init__(self, journal, identifier: str, key: str):
        self.journal, self.identifier, self.key = journal, identifier, key
        self.reservations: list[dict[str, Any]] = []
        self.transactions: list[dict[str, Any]] = []
        self.transaction_digests: list[str] = []
        self.completion: dict[str, Any] | None = None

    def reserve(self, nonce: bytes, request_body: bytes) -> None:
        if self.completion is not None or len(self.reservations) != len(self.transactions):
            raise CheckpointError("cannot continue a complete or interrupted attempt")
        if len(nonce) != 32 or nonce in self.journal.nonces:
            raise CheckpointError("checkpoint nonce repeated")
        record = {"attempt": self.identifier, "index": len(self.reservations),
                  "nonce_hex": nonce.hex(),
                  "request_body_sha256": hashlib.sha256(request_body).hexdigest()}
        self.journal._publish(f"nonce-{nonce.hex()}.json", record)
        self.journal.nonces.add(nonce)
        self.reservations.append(record)

    def save(self, transaction) -> None:
        raw = transaction.raw_document()
        if self.completion is not None or len(self.reservations) != len(self.transactions) + 1:
            raise CheckpointError("transaction has no outstanding nonce reservation")
        reservation = self.reservations[-1]
        if any(raw.get(key) != reservation[key] for key in
               ("nonce_hex", "request_body_sha256")):
            raise CheckpointError("transaction differs from its reserved request")
        record = {"attempt": self.identifier, "index": len(self.transactions),
                  "transaction": raw}
        digest = self.journal._publish(f"transaction-{raw['nonce_hex']}.json", record)
        self.transactions.append(raw)
        self.transaction_digests.append(digest)

    def complete(self, metadata: dict, summary: object) -> None:
        if self.completion is not None or not self.transactions or (
                len(self.reservations) != len(self.transactions)):
            raise CheckpointError("cannot complete an empty or interrupted attempt")
        record = {"attempt": self.identifier,
                  "transaction_sha256": self.transaction_digests,
                  "metadata": metadata, "summary": summary}
        self.journal._publish(f"complete-{self.identifier}.json", record)
        self.completion = record


class Checkpoint:
    def __init__(self, directory: Path, identity: dict):
        self.directory = Path(os.path.abspath(directory))
        self.identity = identity
        self.identity_sha256 = hashlib.sha256(canonical(identity)).hexdigest()
        self.fd = self.lock_fd = -1
        self.nonces: set[bytes] = set()
        self.attempts: dict[str, Attempt] = {}
        self._files = self._bytes = 0

    def __enter__(self):
        if sys.platform != "linux":
            raise CheckpointError("physical checkpoints require Linux flock")
        import fcntl
        # Walk without following any symlink, including ancestors. The caller
        # must create the parent; only the final working directory is created.
        parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in self.directory.parts[1:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=parent)
                os.close(parent)
                parent = child
            try:
                os.mkdir(self.directory.name, 0o700, dir_fd=parent)
                os.fsync(parent)
            except FileExistsError:
                pass
            self.fd = os.open(self.directory.name, os.O_RDONLY | os.O_DIRECTORY |
                              os.O_NOFOLLOW, dir_fd=parent)
        finally:
            os.close(parent)
        try:
            self.lock_fd = os.open("writer.lock", os.O_RDWR | os.O_CREAT |
                                   os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=self.fd)
            self._regular(self.lock_fd)
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise CheckpointError("physical checkpoint has another writer") from error
            self._load()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_args):
        for descriptor in (self.lock_fd, self.fd):
            if descriptor >= 0:
                os.close(descriptor)
        self.lock_fd = self.fd = -1

    @staticmethod
    def _regular(fd: int):
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not (
                0 <= info.st_size <= MAX_FILE_BYTES):
            raise CheckpointError("checkpoint requires bounded regular, unlinked-alias files")
        return info

    def _read(self, name: str) -> bytes:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
        try:
            info = self._regular(fd)
            chunks, size = [], 0
            while True:
                chunk = os.read(fd, min(1024 * 1024, MAX_FILE_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise CheckpointError("checkpoint file exceeds size bound")
            if size != info.st_size:
                raise CheckpointError("checkpoint file changed during read")
            return b"".join(chunks)
        finally:
            os.close(fd)

    def _publish(self, name: str, record: object) -> str:
        raw = canonical({"schema_version": 1, "identity_sha256": self.identity_sha256,
                         "record": record})
        if len(raw) > MAX_FILE_BYTES or self._files + 2 > MAX_FILES or (
                self._bytes + len(raw) > MAX_TOTAL_BYTES):
            raise CheckpointError("physical checkpoint size bound exceeded")
        temporary = f".pending-{secrets.token_hex(16)}"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=self.fd)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            rename_new(temporary, name, directory_fd=self.fd)
            os.fsync(self.fd)
        except BaseException:
            # Keep interrupted publication work. Never overwrite an old record.
            raise
        self._files += 1
        self._bytes += len(raw)
        return hashlib.sha256(raw).hexdigest()

    def _load(self):
        records = {}
        with os.scandir(self.fd) as entries:
            for entry in entries:
                self._files += 1
                if self._files > MAX_FILES:
                    raise CheckpointError("too many checkpoint files")
                raw = self._read(entry.name)
                self._bytes += len(raw)
                if self._bytes > MAX_TOTAL_BYTES:
                    raise CheckpointError("physical checkpoint size bound exceeded")
                if entry.name == "writer.lock" or re.fullmatch(r"\.pending-[0-9a-f]{32}", entry.name):
                    continue
                try:
                    value = json.loads(raw, object_pairs_hook=_pairs,
                                       parse_constant=lambda _: (_ for _ in ()).throw(
                                           CheckpointError("nonfinite checkpoint JSON")))
                except (ValueError, UnicodeError) as error:
                    raise CheckpointError("invalid checkpoint JSON") from error
                if not isinstance(value, dict) or set(value) != {
                        "schema_version", "identity_sha256", "record"} or (
                        type(value["schema_version"]) is not int or value["schema_version"] != 1
                        or value["identity_sha256"] != self.identity_sha256):
                    raise CheckpointError("checkpoint experiment identity mismatch")
                if canonical(value) != raw:
                    raise CheckpointError("checkpoint JSON must be canonical")
                records[entry.name] = (value["record"], hashlib.sha256(raw).hexdigest())
        identity = records.pop("identity.json", None)
        if identity is None:
            if records or self._files > 1:
                raise CheckpointError("populated checkpoint has no experiment identity")
            self._publish("identity.json", self.identity)
        elif canonical(identity[0]) != canonical(self.identity):
            raise CheckpointError("checkpoint experiment identity mismatch")
        for name, (record, _) in records.items():
            match = re.fullmatch(r"attempt-([0-9a-f]{32})\.json", name)
            if match:
                if not isinstance(record, dict) or set(record) != {"attempt", "key"} or (
                        record["attempt"] != match[1] or not isinstance(record["key"], str)
                        or not KEY.fullmatch(record["key"])):
                    raise CheckpointError("invalid checkpoint attempt")
                self.attempts[match[1]] = Attempt(self, match[1], record["key"])
        pending = {identifier: ({}, {}) for identifier in self.attempts}
        for name, (record, digest) in records.items():
            if not isinstance(record, dict) or not isinstance(record.get("attempt"), str) or (
                    record["attempt"] not in self.attempts):
                raise CheckpointError("checkpoint record has no attempt")
            attempt = self.attempts[record["attempt"]]
            if name == f"attempt-{attempt.identifier}.json":
                continue
            if name == f"complete-{attempt.identifier}.json":
                if set(record) != {"attempt", "transaction_sha256", "metadata", "summary"}:
                    raise CheckpointError("invalid checkpoint completion")
                attempt.completion = record
                continue
            match = re.fullmatch(r"(nonce|transaction)-([0-9a-f]{64})\.json", name)
            index = record.get("index")
            if not match or type(index) is not int or not 0 <= index < MAX_FILES:
                raise CheckpointError("invalid checkpoint transaction index")
            target = pending[attempt.identifier][match[1] == "transaction"]
            if index in target:
                raise CheckpointError("duplicate checkpoint transaction index")
            if match[1] == "nonce":
                if set(record) != {"attempt", "index", "nonce_hex", "request_body_sha256"} or (
                        record["nonce_hex"] != match[2] or not isinstance(record["request_body_sha256"], str)
                        or not HEX.fullmatch(record["request_body_sha256"])):
                    raise CheckpointError("invalid checkpoint nonce reservation")
                nonce = bytes.fromhex(match[2])
                if nonce in self.nonces:
                    raise CheckpointError("duplicate checkpoint nonce")
                self.nonces.add(nonce)
            elif set(record) != {"attempt", "index", "transaction"} or (
                    not isinstance(record["transaction"], dict)
                    or record["transaction"].get("nonce_hex") != match[2]):
                raise CheckpointError("invalid checkpoint transaction")
            target[index] = (record, digest)
        completed_keys = set()
        for identifier, (reserved, verified) in pending.items():
            attempt = self.attempts[identifier]
            if sorted(reserved) != list(range(len(reserved))) or sorted(verified) != list(range(len(verified))) or (
                    len(reserved) - len(verified) not in (0, 1)):
                raise CheckpointError("checkpoint transaction sequence has gaps")
            attempt.reservations = [reserved[i][0] for i in range(len(reserved))]
            attempt.transactions = [verified[i][0]["transaction"] for i in range(len(verified))]
            attempt.transaction_digests = [verified[i][1] for i in range(len(verified))]
            for reservation, transaction in zip(attempt.reservations, attempt.transactions):
                if any(reservation[k] != transaction.get(k) for k in ("nonce_hex", "request_body_sha256")):
                    raise CheckpointError("checkpoint transaction differs from reservation")
            if attempt.completion is not None:
                if not verified or len(reserved) != len(verified) or (
                        attempt.completion["transaction_sha256"] != attempt.transaction_digests
                        or attempt.key in completed_keys):
                    raise CheckpointError("checkpoint completion does not bind one whole attempt")
                completed_keys.add(attempt.key)

    def completed(self, key: str) -> Attempt | None:
        return next((a for a in self.attempts.values()
                     if a.key == key and a.completion is not None), None)

    def start(self, key: str) -> Attempt:
        if not KEY.fullmatch(key) or self.completed(key) is not None:
            raise CheckpointError("invalid or already completed checkpoint attempt")
        identifier = secrets.token_hex(16)
        self._publish(f"attempt-{identifier}.json", {"attempt": identifier, "key": key})
        attempt = Attempt(self, identifier, key)
        self.attempts[identifier] = attempt
        return attempt
