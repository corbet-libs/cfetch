"""Checkpoint tests use signed synthetic replies and mock all dispatcher work."""

from dataclasses import replace
import base64
from email.message import Message
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import physical_evidence as evidence
from physical_checkpoint import Checkpoint, CheckpointError, rename_new
from test_physical_evidence import live_evidence, scope_contract


class PhysicalCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "checkpoint"
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.identity = {"dispatcher": "a" * 64, "package": "b" * 64,
                         "runtime": "c" * 64, "implementation": "d" * 64,
                         "inputs": "e" * 64, "host_policy": "f" * 64,
                         "warmups": 1, "samples": 20, "timeout": 10.0}
        self.key = Ed25519PrivateKey.generate()
        public = self.key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
        self.scope = replace(scope_contract(), public_key_hex=public)
        self.package = SimpleNamespace(scope=self.scope, manifest_sha256="b" * 64,
                                       runtime_manifest_sha256="c" * 64)
        self.counter = 0

    def transaction(self, texts, *, measure_rss=True, nonce=None):
        self.counter += 1
        nonce = nonce or self.counter.to_bytes(32, "big")
        probes = evidence.sequence_semantic_probe_inputs(32)
        vectors = [[1.0, 0.2] if text == probes[1] else
                   [-1.0, 0.0] if text == probes[2] else [1.0, 0.0] for text in texts]
        body = evidence._request_body(self.scope, texts)
        runtime = live_evidence()
        payload = {
            "model": evidence.MODEL, "cfetch_profile": evidence.PROFILE_ID,
            "cfetch_profile_manifest_sha256": evidence.PROFILE_MANIFEST_SHA256,
            "cfetch_admission_policy_sha256": evidence.ADMISSION_POLICY_SHA256,
            "cfetch_model_revision": evidence.MODEL_REVISION,
            "cfetch_execution": self.scope.expected_execution(),
            "cfetch_runtime_evidence": runtime,
            "data": [{"index": i, "cfetch_scope_id": self.scope.scope_id,
                      "token_count": 32, "sequence_bucket": 32, "truncated": False,
                      "embedding": vector + [0.0] * 766}
                     for i, vector in enumerate(vectors)],
        }
        response = evidence._canonical_json(payload)
        signature = self.key.sign(evidence.ATTESTATION_DOMAIN + nonce +
                                  hashlib.sha256(body).digest() + hashlib.sha256(response).digest())
        rows = tuple(evidence.ResponseRow(32, 32, evidence.canonical_i8_bytes(v + [0.0] * 766))
                     for v in vectors)
        return evidence.SignedTransaction(nonce.hex(), signature.hex(), body, response,
                                          10000 + self.counter, 1000000 if measure_rss else None,
                                          3 if measure_rss else 0, rows, runtime)

    def mocked_dispatcher(self, *, fail_at=None):
        owner = self
        state = SimpleNamespace(calls=0, sessions=0, fail_at=fail_at)

        class Session:
            startup_peak_rss_bytes, startup_rss_sample_count = 2000000, 4
            def __init__(self, *_args):
                state.sessions += 1
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                pass

        class Client:
            def __init__(self, _session, _timeout, nonces, checkpoint_attempt=None):
                self.nonces, self.attempt = nonces, checkpoint_attempt
            def request(self, texts, measure_rss=True):
                state.calls += 1
                transaction = owner.transaction(texts, measure_rss=measure_rss)
                nonce = bytes.fromhex(transaction.nonce_hex)
                owner.assertNotIn(nonce, self.nonces)
                self.nonces.add(nonce)
                self.attempt.reserve(nonce, transaction.request_body)
                if state.calls == state.fail_at:
                    raise evidence.EvidenceError("mock interrupted request")
                self.attempt.save(transaction)
                return transaction

        return state, Session, Client

    def run_bucket(self, checkpoint):
        return evidence._run_bucket(Path("dispatcher"), "a" * 64, self.package,
                                    10.0, 10.0, 1, 20, "synthetic test", 32, self.raw,
                                    set(checkpoint.nonces), checkpoint)

    def test_identity_aliases_and_orphan_history_are_rejected(self):
        with Checkpoint(self.directory, self.identity) as journal:
            attempt = journal.start("wire-1")
            transaction = self.transaction(["a"], measure_rss=False)
            attempt.reserve(bytes.fromhex(transaction.nonce_hex), transaction.request_body)
            attempt.save(transaction)
        for field in self.identity:
            with self.subTest(field=field):
                changed = {**self.identity, field: "changed"}
                with self.assertRaisesRegex(CheckpointError, "identity"):
                    with Checkpoint(self.directory, changed):
                        pass
        (self.directory / "identity.json").unlink()
        with self.assertRaisesRegex(CheckpointError, "no experiment identity"):
            with Checkpoint(self.directory, self.identity):
                pass

    def test_interrupted_publication_preserves_nonce_and_never_overwrites(self):
        with Checkpoint(self.directory, self.identity) as journal:
            attempt = journal.start("wire-1")
            transaction = self.transaction(["a"], measure_rss=False)
            attempt.reserve(bytes.fromhex(transaction.nonce_hex), transaction.request_body)
            with patch("physical_checkpoint.rename_new", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    attempt.save(transaction)
        self.assertEqual(len(list(self.directory.glob(".pending-*"))), 1)
        with Checkpoint(self.directory, self.identity) as journal:
            self.assertIsNone(journal.completed("wire-1"))
            fresh = journal.start("wire-1")
            with self.assertRaisesRegex(CheckpointError, "nonce repeated"):
                fresh.reserve(bytes.fromhex(transaction.nonce_hex), transaction.request_body)
            original = (self.directory / "identity.json").read_bytes()
            with self.assertRaises(FileExistsError):
                journal._publish("identity.json", {"changed": True})
            self.assertEqual((self.directory / "identity.json").read_bytes(), original)
        source, destination = self.root / "source", self.root / "destination"
        source.mkdir()
        destination.mkdir()
        with self.assertRaises(FileExistsError):
            rename_new(source, destination)
        self.assertTrue(source.is_dir())

    def test_lock_survives_writer_contention_and_killed_writer_preserves_nonce(self):
        script = '''import sys
from pathlib import Path
from physical_checkpoint import Checkpoint
with Checkpoint(Path(sys.argv[1]), {"test": 1}) as journal:
    journal.start("wire-1").reserve(bytes(32), b"request")
    print("ready", flush=True)
    sys.stdin.read()
'''
        process = subprocess.Popen([sys.executable, "-c", script, str(self.directory)],
                                   cwd=Path(__file__).parent, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, text=True)
        try:
            import select
            self.assertTrue(select.select([process.stdout], [], [], 5)[0])
            self.assertEqual(process.stdout.readline().strip(), "ready")
            with self.assertRaisesRegex(CheckpointError, "another writer"):
                with Checkpoint(self.directory, {"test": 1}):
                    pass
        finally:
            process.kill()
            process.communicate(timeout=5)
        with Checkpoint(self.directory, {"test": 1}) as journal:
            self.assertIn(bytes(32), journal.nonces)
            self.assertIsNone(journal.completed("wire-1"))

    def test_fifo_symlinks_and_hardlinks_are_rejected_without_blocking(self):
        self.directory.mkdir()
        os.mkfifo(self.directory / "writer.lock")
        script = '''from pathlib import Path
import sys
from physical_checkpoint import Checkpoint, CheckpointError
try:
    with Checkpoint(Path(sys.argv[1]), {}): pass
except (CheckpointError, OSError): sys.exit(0)
sys.exit(1)
'''
        result = subprocess.run([sys.executable, "-c", script, str(self.directory)],
                                cwd=Path(__file__).parent, timeout=5)
        self.assertEqual(result.returncode, 0)
        (self.directory / "writer.lock").unlink()
        outside = self.root / "outside"
        outside.write_bytes(b"")
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                lock = self.directory / "writer.lock"
                if kind == "symlink":
                    lock.symlink_to(outside)
                else:
                    os.link(outside, lock)
                with self.assertRaises((CheckpointError, OSError)):
                    with Checkpoint(self.directory, {}):
                        pass
                lock.unlink()
        alias = self.root / "alias"
        alias.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaises(OSError):
            with Checkpoint(alias / "nested", {}):
                pass

    def test_bucket_restart_uses_full_fresh_trial_then_complete_replay_needs_no_dispatcher(self):
        state, Session, Client = self.mocked_dispatcher(fail_at=5)
        with patch.object(evidence, "DispatcherSession", Session), patch.object(evidence, "SignedAdapterClient", Client):
            with Checkpoint(self.directory, self.identity) as journal:
                with self.assertRaisesRegex(evidence.EvidenceError, "interrupted"):
                    self.run_bucket(journal)
        with Checkpoint(self.directory, self.identity) as journal:
            self.assertEqual(len(journal.nonces), 5)
            abandoned = next(iter(journal.attempts.values()))
            self.assertEqual(len(abandoned.transactions), 4)
            state.fail_at = None
            with patch.object(evidence, "DispatcherSession", Session), patch.object(evidence, "SignedAdapterClient", Client):
                result = self.run_bucket(journal)
            self.assertEqual(state.calls, 5 + 23)
            self.assertEqual(state.sessions, 2)
            self.assertEqual(len(journal.attempts), 2)
        with Checkpoint(self.directory, self.identity) as journal, patch.object(
                evidence, "DispatcherSession", side_effect=AssertionError("must not launch")):
            self.assertEqual(self.run_bucket(journal), result)
            journal.completed("bucket-32").completion["summary"][2]["sample_count"] = 19
            with self.assertRaisesRegex(evidence.EvidenceError, "summary differs"):
                self.run_bucket(journal)

    def test_wire_grouping_resume_restarts_partial_group_and_revalidates_signed_order(self):
        inputs = ["a", "b", "c"]
        state, Session, Client = self.mocked_dispatcher(fail_at=5)
        def run(journal):
            return evidence._wire_grouping_results(Path("dispatcher"), "a" * 64,
                self.package, 10.0, 10.0, inputs, self.raw, set(journal.nonces), journal)
        with patch.object(evidence, "SUPPORTED_MAX_BATCH_SIZE", 3):
            with Checkpoint(self.directory, self.identity) as journal:
                with patch.object(evidence, "DispatcherSession", Session), patch.object(evidence, "SignedAdapterClient", Client):
                    with self.assertRaisesRegex(evidence.EvidenceError, "interrupted"):
                        run(journal)
            state.fail_at = None
            # Only the first group replays while the remaining groups are newly collected.
            original = evidence.SignedAdapterClient
            class ReplayClient(Client):
                _verify_signature = original._verify_signature
                _validate_response = original._validate_response
            with Checkpoint(self.directory, self.identity) as journal, patch.object(
                    evidence, "DispatcherSession", Session), patch.object(evidence, "SignedAdapterClient", ReplayClient):
                results = run(journal)
                self.assertEqual(state.calls, 8)
                self.assertEqual(len(results), 3)
            with Checkpoint(self.directory, self.identity) as journal, patch.object(
                    evidence, "DispatcherSession", side_effect=AssertionError("must not launch")):
                self.assertEqual(run(journal), results)
                attempt = journal.completed("wire-2")
                with self.assertRaisesRegex(evidence.EvidenceError, "request order changed"):
                    evidence._restore_attempt(attempt, self.scope, [["b", "a"], ["c"]], measure_rss=False)
                raw = dict(attempt.transactions[0], signature_hex="00" * 64)
                with self.assertRaisesRegex(evidence.EvidenceError, "Ed25519 signature"):
                    evidence._restore_transaction(self.scope, raw, ["a", "b"], measure_rss=False)
                raw = dict(attempt.transactions[0])
                payload = json.loads(base64.b64decode(raw["response_body_base64"]))
                payload["cfetch_runtime_evidence"]["host"]["kernel_release"] = "other-kernel"
                response = evidence._canonical_json(payload)
                raw["response_body_base64"] = base64.b64encode(response).decode("ascii")
                raw["response_body_sha256"] = hashlib.sha256(response).hexdigest()
                raw["signature_hex"] = self.key.sign(evidence.ATTESTATION_DOMAIN +
                    bytes.fromhex(raw["nonce_hex"]) + bytes.fromhex(raw["request_body_sha256"]) +
                    hashlib.sha256(response).digest()).hex()
                with self.assertRaisesRegex(evidence.EvidenceError, "host/kernel/file evidence"):
                    evidence._restore_transaction(self.scope, raw, ["a", "b"], measure_rss=False)

    def test_live_client_publishes_verified_transaction_before_return(self):
        owner = self
        class Connection:
            def __init__(self, *_args, **_kwargs):
                pass
            def request(self, _method, _path, *, body, headers):
                nonce = bytes.fromhex(headers[evidence.ATTESTATION_NONCE_HEADER])
                owner.assertTrue((owner.directory / f"nonce-{nonce.hex()}.json").exists())
                self.transaction = owner.transaction(json.loads(body)["input"], measure_rss=False, nonce=nonce)
            def getresponse(self):
                transaction = self.transaction
                headers = Message()
                headers[evidence.ATTESTATION_SIGNATURE_HEADER] = transaction.signature_hex
                return SimpleNamespace(status=200, headers=headers,
                    getheader=lambda name, default=None: {"Content-Length": str(len(transaction.response_body)),
                        "Content-Type": "application/json"}.get(name, default),
                    read=lambda _maximum: transaction.response_body)
            def close(self):
                pass
        session = SimpleNamespace(process=SimpleNamespace(pid=os.getpid()),
            endpoint="http://127.0.0.1:1/v1/embeddings", bearer="synthetic", package=self.package)
        with Checkpoint(self.directory, self.identity) as journal, patch.object(
                evidence.http.client, "HTTPConnection", Connection):
            attempt = journal.start("wire-1")
            client = evidence.SignedAdapterClient(session, 1.0, set(), attempt)
            transaction = client.request(["a"], measure_rss=False)
            self.assertTrue((self.directory / f"transaction-{transaction.nonce_hex}.json").exists())
            with patch.object(attempt, "save", side_effect=OSError("fsync failed")):
                with self.assertRaisesRegex(OSError, "fsync failed"):
                    client.request(["b"], measure_rss=False)
            self.assertEqual(len(attempt.reservations), 2)
            self.assertEqual(len(attempt.transactions), 1)


if __name__ == "__main__":
    unittest.main()
