"""Pinned CPU-only candidate measurements, never shared-vector admission.

Consumes the exact manifest exported by `cfetch retrieval-eval`. Uses existing
local model bytes only. Each actual inference is one input in its smallest
profile bucket; interrupted runs resume their own verified working records.
The community artifact's lineage to cfetch's canonical checkpoint is unproven.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import tempfile
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

REVISION = "5090578d9565bb06545b4552f76e6bc2c93e4a66"
FILES = {
    "onnx/model.onnx": "ea91fd315a7c152d427d231746f0f811a1ac93beaba656abfdf2b24e091265e4",
    "onnx/model.onnx_data": "ef835ae565d8695236652475903078e8ed794c7c35faf1164d78ec3238e8a88d",
    "onnx/model_q4.onnx": "ad1dfee81a70f7944b9b9d1cc6e48075b832881cf33fab2f2b248be78f3f0043",
    "onnx/model_q4.onnx_data": "599962c3143b040de2dd05e5975be3e9091dd067cacc6a8f7186e3203bab9e02",
    "tokenizer.json": "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47",
    "config.json": "6e1f06404b7163e0325ed2ea3e6781cde50f4a50b31780a95ad0d30e8404d77b",
}
BUCKETS = [32, 64, 128, 257, 512, 1024, 2048]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        maximum = 64 * 1024 * 1024
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
            raise ValueError("input must be a bounded regular file")
        raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise ValueError("input grew beyond the size limit")
    return json.loads(raw, object_pairs_hook=strict_object)


def publish(path: Path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".part-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes atomically without replacing any existing entry.
        # A crash may leave an unused temporary, which never prevents resume.
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def run(args):
    manifest = read_json(args.manifest)
    if manifest["schema_version"] != 1 or manifest["dimensions"] != 768 or manifest["sequence_buckets"] != BUCKETS:
        raise ValueError("unsupported input/profile manifest")
    if not 1 <= len(manifest["inputs"]) <= 4096:
        raise ValueError("expected 1..4096 explicitly bounded inputs")
    seen = set()
    for item in manifest["inputs"]:
        expected = hashlib.sha256(b"cfetch-evaluation-input-v1\0" + item["text"].encode()).hexdigest()
        if item["id"] != expected or expected in seen or item["kind"] not in {"query", "document"}:
            raise ValueError("duplicate or invalid input identity")
        seen.add(expected)
    filename = "model.onnx" if args.artifact == "fp32" else "model_q4.onnx"
    used = ["onnx/" + filename, "onnx/" + filename + "_data", "tokenizer.json", "config.json"]
    hashes = {name: digest(args.model_root / name) for name in used}
    if any(hashes[name] != FILES[name] for name in used):
        raise ValueError("pinned model/tokenizer digest mismatch")
    provenance = {
        "purpose": "CPU candidate memory retrieval; not admission or canonical-source lineage proof",
        "model_repository": "onnx-community/embeddinggemma-300m-ONNX",
        "revision": REVISION,
        "artifact": args.artifact,
        "files": hashes,
        "runner_sha256": digest(Path(__file__)),
        "ort_build": ort.get_build_info(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "onnxruntime", "tokenizers")},
        "platform": platform.platform(),
        "provider": "CPUExecutionProvider",
        "execution_batch_size": 1,
        "bucket_policy": "smallest per input; no truncation",
        "graph_optimization": "ORT_ENABLE_ALL",
        "intra_op_threads": 4,
        "inter_op_threads": 1,
        "pause_seconds_after_each_inference": 0.15,
    }
    identity = {"manifest_sha256": digest(args.manifest), "provenance": provenance}
    experiment_sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    if args.checkpoints.is_symlink():
        raise ValueError("checkpoint directory cannot be a symlink")
    descriptor = os.open(args.checkpoints / "writer.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("checkpoint lock must be a regular file")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = args.checkpoints / "identity.json"
        if identity_path.exists():
            if read_json(identity_path) != identity:
                raise ValueError("checkpoint belongs to a different exact experiment")
        else:
            if any(path.suffix == ".json" for path in args.checkpoints.iterdir()):
                raise ValueError("completed checkpoints lack their experiment identity")
            publish(identity_path, identity)
        tokenizer = Tokenizer.from_file(str(args.model_root / "tokenizer.json"))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        pad = read_json(args.model_root / "config.json")["pad_token_id"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.use_deterministic_compute = True
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = ort.InferenceSession(str(args.model_root / "onnx" / filename), sess_options=options, providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("unexpected execution provider")
        outputs = []
        for ordinal, item in enumerate(manifest["inputs"]):
            ids = tokenizer.encode(item["text"], add_special_tokens=True).ids
            bucket = next((width for width in BUCKETS if len(ids) <= width), None)
            record_path = args.checkpoints / (item["id"] + ".json")
            if record_path.exists():
                checkpoint = read_json(record_path)
                if checkpoint["experiment_sha256"] != experiment_sha256:
                    raise ValueError("record belongs to a different exact experiment")
                record = checkpoint["output"]
                if record["id"] != item["id"] or record["token_count"] != len(ids) or record["bucket"] != bucket:
                    raise ValueError("checkpoint no longer matches tokenized input")
            else:
                record = {"id": item["id"], "token_count": len(ids), "bucket": bucket, "vector": None, "error": None}
                if bucket is None:
                    record["error"] = "prefixed input exceeds 2048 tokens; truncation forbidden"
                else:
                    inputs = {
                        "input_ids": np.asarray([ids + [pad] * (bucket - len(ids))], dtype=np.int64),
                        "attention_mask": np.asarray([[1] * len(ids) + [0] * (bucket - len(ids))], dtype=np.int64),
                    }
                    raw = session.run(["sentence_embedding"], inputs)[0]
                    if raw.shape != (1, 768) or not np.isfinite(raw).all() or not 0.99 <= float(np.sum(raw.astype(np.float64) ** 2)) <= 1.01:
                        raise ValueError(f"invalid embedding for {item['id']}")
                    record["vector"] = raw[0].tolist()
                    time.sleep(0.15)
                publish(record_path, {"experiment_sha256": experiment_sha256, "output": record})
            # Candidate output validation is also performed independently by
            # the Rust importer before any retrieval scores are computed.
            outputs.append(record)
            print(f"{args.artifact}: {ordinal + 1}/{len(manifest['inputs'])} inputs", flush=True)
        publish(args.output, {"schema_version": 1, **identity, "outputs": outputs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--artifact", choices=("fp32", "q4"), required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
