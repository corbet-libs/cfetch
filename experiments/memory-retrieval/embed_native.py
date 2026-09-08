#!/usr/bin/env python3
"""Resumable canonical-source OpenVINO CPU retrieval diagnostic; never admission.

Consumes existing local source/artifact bytes and an exact retrieval-eval
manifest. Each actual inference uses one canonical BOS/text/EOS input in its
smallest frozen bucket. Completed checkpoints are validated without model
initialization; output order always follows the manifest.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import os
from pathlib import Path
import platform
import stat
import sys
import time

from audit_native import CPU_CONFIG, REPOSITORY, digest, load_inputs, publish, read_json, value_digest


REFUSAL = "prefixed input exceeds 2048 tokens; truncation forbidden"
PAUSE_SECONDS = 0.15


def validate_record(record, item, ids, bucket):
    import numpy as np

    if (not isinstance(record, dict)
            or set(record) != {"id", "token_count", "bucket", "vector", "error"}
            or record["id"] != item["id"] or type(record["token_count"]) is not int
            or record["token_count"] != len(ids)
            or (record["bucket"] is not None and type(record["bucket"]) is not int)
            or record["bucket"] != bucket):
        raise ValueError("checkpoint no longer matches the exact tokenized input")
    if bucket is None:
        if record["vector"] is not None or record["error"] != REFUSAL:
            raise ValueError("overlength checkpoint must contain the exact refusal and no vector")
        return
    vector = record["vector"]
    if (record["error"] is not None or not isinstance(vector, list) or len(vector) != 768
            or any(type(value) not in (int, float) for value in vector)):
        raise ValueError("runnable checkpoint requires exactly 768 numeric components and no error")
    array = np.asarray(vector, dtype=np.float32)
    if not np.isfinite(array).all() or not 0.99 <= float(np.sum(array.astype(np.float64) ** 2)) <= 1.01:
        raise ValueError("checkpoint vector is nonfinite or not unit normalized")


def run(args):
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise ValueError("requires isolated Linux CPython 3.12 with the OpenVINO build lock")
    if os.path.lexists(args.output) or not args.output.parent.is_dir():
        raise ValueError("output must be absent and its parent directory must exist")
    if args.checkpoints.is_symlink() or (args.checkpoints.exists() and not args.checkpoints.is_dir()):
        raise ValueError("checkpoint path must be a real directory or absent")
    if args.output.resolve().is_relative_to(args.checkpoints.resolve()):
        raise ValueError("final output must be outside the checkpoint directory")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(REPOSITORY))
    import numpy as np
    import openvino as ov
    from tokenizers import Tokenizer
    from packages.openvino import convert, manifest as native

    items, manifest_hash = load_inputs(args.manifest, native.SEQUENCE_BUCKETS)
    convert.verify_source_files(args.source_dir)
    convert.validate_semantic_source(args.source_dir)
    artifact_document, artifact_hash = read_json(args.artifact_dir / "artifact-manifest.json")
    artifact = native.load_artifact(args.artifact_dir, "artifact-manifest.json", artifact_hash)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "openvino", "tokenizers", "torch", "transformers", "safetensors")}
    if any(versions[name] != version for name, version in artifact.conversion_versions.items()):
        raise ValueError("installed conversion libraries differ from artifact provenance")
    helper_paths = {
        "embed_native.py": Path(__file__), "audit_native.py": Path(__file__).with_name("audit_native.py"),
        "convert.py": REPOSITORY / "packages/openvino/convert.py",
        "manifest.py": REPOSITORY / "packages/openvino/manifest.py",
        "legal.py": REPOSITORY / "packages/openvino/legal.py",
        "requirements-build.lock": REPOSITORY / "packages/openvino/requirements-build.lock",
    }
    helper_hashes = {name: digest(path) for name, path in helper_paths.items()}
    # CPU identity omits volatile clocks and hostnames so a legitimate resume
    # remains stable, while another CPU family/features cannot adopt the work.
    cpu_fields = {"vendor_id", "cpu family", "model", "model name", "stepping", "flags"}
    cpu_identity = [line.strip() for line in Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0].splitlines()
                    if line.partition(":")[0].strip() in cpu_fields]
    provenance = {
        "purpose": "canonical-source native CPU retrieval diagnostic; not shared-store admission",
        "model_repository": native.MODEL, "revision": native.MODEL_REVISION,
        "source_files": dict(native.PINNED_SOURCE_FILE_SHA256),
        "artifact_manifest_sha256": artifact_hash, "files": artifact_document["files"],
        "conversion": artifact_document["conversion"], "packages": versions, "openvino_build": ov.get_version(),
        "implementation_sha256": helper_hashes, "cpu_identity_sha256": value_digest(cpu_identity),
        "system": platform.system(), "machine": platform.machine(), "kernel": platform.release(),
        "libc": list(platform.libc_ver()), "provider": "OpenVINO CPU", "compile_config": dict(CPU_CONFIG),
        "execution_batch_size": 1, "tokenization": "[2] + encode(add_special_tokens=False).ids + [1]",
        "bucket_policy": "smallest per input; right PAD=0; no truncation", "pause_seconds_after_each_inference": PAUSE_SECONDS,
    }
    identity = {"manifest_sha256": manifest_hash, "provenance": provenance}
    experiment_hash = value_digest(identity)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.checkpoints / "writer.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("checkpoint writer lock must be a regular file")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = args.checkpoints / "identity.json"
        allowed = {"writer.lock", "identity.json", *(item["id"] + ".json" for item in items)}
        if os.path.lexists(identity_path):
            if value_digest(read_json(identity_path)[0]) != experiment_hash:
                raise ValueError("checkpoint belongs to a different exact experiment")
            for path in args.checkpoints.iterdir():
                if path.is_symlink() or not path.is_file() or (
                        path.name not in allowed and not path.name.startswith(".native-audit-")):
                    raise ValueError("unexpected checkpoint entry")
        else:
            if any(path.name != "writer.lock" for path in args.checkpoints.iterdir()):
                raise ValueError("checkpoint entries lack their exact experiment identity")
            publish(identity_path, identity)
        tokenizer = Tokenizer.from_file(str(artifact.tokenizer_json))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        if [tokenizer.token_to_id(token) for token in ("<pad>", "<bos>", "<eos>")] != [0, 2, 1]:
            raise ValueError("canonical special-token IDs differ from the frozen contract")
        outputs, pending = {}, []
        # Validate all completed rows before any model is initialized, including
        # rows later in manifest order than the first missing checkpoint.
        for ordinal, item in enumerate(items):
            ids = [2, *tokenizer.encode(item["text"], add_special_tokens=False).ids, 1]
            bucket = next((width for width in native.SEQUENCE_BUCKETS if len(ids) <= width), None)
            path = args.checkpoints / (item["id"] + ".json")
            if os.path.lexists(path):
                checkpoint = read_json(path)[0]
                if (not isinstance(checkpoint, dict) or set(checkpoint) != {"experiment_sha256", "output"}
                        or checkpoint["experiment_sha256"] != experiment_hash):
                    raise ValueError("record belongs to a different exact experiment")
                validate_record(checkpoint["output"], item, ids, bucket)
                outputs[item["id"]] = checkpoint["output"]
            else:
                pending.append((bucket, ordinal, item, ids, path))
        core, graph, compiled, compiled_bucket = None, None, None, None
        for bucket, ordinal, item, ids, path in sorted(pending, key=lambda row: (row[0] or 4096, row[1])):
            record = {"id": item["id"], "token_count": len(ids), "bucket": bucket,
                      "vector": None, "error": REFUSAL if bucket is None else None}
            if bucket is not None:
                if core is None:
                    core = ov.Core()
                    graph = core.read_model(str(artifact.graph_xml), str(artifact.graph_bin))
                if compiled_bucket != bucket:
                    compiled = None
                    static = graph.clone()
                    static.reshape({artifact.input_ids_name: [1, bucket], artifact.attention_mask_name: [1, bucket]})
                    compiled = core.compile_model(static, "CPU", CPU_CONFIG)
                    compiled_bucket = bucket
                    if list(compiled.get_property("EXECUTION_DEVICES")) != ["CPU"]:
                        raise ValueError("native placement is not exclusively CPU")
                    if compiled.get_property("INFERENCE_PRECISION_HINT") != ov.Type.f32:
                        raise ValueError("native CPU precision hint is not f32")
                arrays = {
                    artifact.input_ids_name: np.asarray([ids + [0] * (bucket - len(ids))], dtype=np.int64),
                    artifact.attention_mask_name: np.asarray([[1] * len(ids) + [0] * (bucket - len(ids))], dtype=np.int64),
                }
                try:
                    result = compiled(arrays)
                    vector = np.asarray(result[compiled.output(artifact.output_name)], dtype=np.float32)
                    if vector.shape != (1, 768):
                        raise ValueError("native output shape differs from [1,768]")
                    record["vector"] = vector[0].tolist()
                    del result, vector
                finally:
                    time.sleep(PAUSE_SECONDS)
            validate_record(record, item, ids, bucket)
            publish(path, {"experiment_sha256": experiment_hash, "output": record})
            outputs[item["id"]] = record
            print(f"native CPU: {len(outputs)}/{len(items)} inputs", flush=True)
        convert.verify_source_files(args.source_dir)
        convert.validate_semantic_source(args.source_dir)
        native.load_artifact(args.artifact_dir, "artifact-manifest.json", artifact_hash)
        if (digest(args.manifest) != manifest_hash
                or any(digest(path) != helper_hashes[name] for name, path in helper_paths.items())):
            raise ValueError("manifest or implementation changed during execution")
        publish(args.output, {"schema_version": 1, **identity, "outputs": [outputs[item["id"]] for item in items]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source-dir", "artifact-dir", "checkpoints", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
