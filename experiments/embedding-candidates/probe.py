#!/usr/bin/env python3
"""Three-call direct OpenVINO comparison, requiring the installed host governor.

Experiment only. Does not configure devices, grant approval, reset budgets,
admit a backend, or write cfetch's vector index.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from candidates import canonical, load_candidate, require, verify_files, write_new

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packages.openvino.inference_governor import from_installation
from packages.openvino.native_deadline import NativeDeadline

BUCKET = 64
TEXTS = ("cat animal", "A cat is a small domesticated feline animal.",
         "A violin is a musical instrument with four strings.")
PIPELINE = "candidate-right-pad64-masked-mean-or-cls-l2-v1"


def pool(output, mask, candidate):
    import numpy as np
    data = np.asarray(output, dtype=np.float32)
    require(data.shape == (1, BUCKET, candidate["dimensions"]), "unexpected token embedding shape")
    require(np.isfinite(data).all(), "nonfinite token embeddings")
    if candidate["pooling"] == "cls":
        vector = data[0, 0]
    else:
        weights = np.asarray(mask, dtype=np.float32)
        require(weights.shape == (BUCKET,) and weights.sum() > 0, "invalid pooling mask")
        vector = (data[0] * weights[:, None]).sum(axis=0) / weights.sum()
    norm = float(np.linalg.norm(vector))
    require(np.isfinite(norm) and norm > 0, "degenerate embedding")
    return (vector / norm).tolist()


def inputs(directory, candidate):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    pad = tokenizer.token_to_id("[PAD]")
    if pad is None:
        pad = tokenizer.token_to_id("<|padding|>")
    require(pad is not None, "candidate padding token is unknown")
    rows = []
    for index, text in enumerate(TEXTS):
        prefix = candidate["query_prefix"] if index == 0 else candidate["document_prefix"]
        encoding = tokenizer.encode(prefix + text, add_special_tokens=True)
        length = len(encoding.ids)
        require(0 < length <= min(BUCKET, candidate["max_tokens"]), "input would require truncation")
        rows.append({"input_ids": encoding.ids + [pad] * (BUCKET - length),
                     "attention_mask": [1] * length + [0] * (BUCKET - length),
                     "token_type_ids": encoding.type_ids + [0] * (BUCKET - length)})
    return rows


def reference_vectors(path, identity, dimensions):
    import numpy as np
    reference = json.loads(path.read_bytes())
    require(reference["identity"] == identity and reference["device"] == "CPU",
            "reference belongs to another model, pipeline or input set")
    require(reference["execution_devices"] == ["CPU"], "reference did not execute on CPU")
    values = np.asarray(reference["vectors"], dtype=np.float64)
    require(values.shape == (len(TEXTS), dimensions) and np.isfinite(values).all(), "invalid reference vectors")
    require(np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5), "reference is not normalized")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("candidate")
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("NPU", "GPU", "CPU"), required=True)
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    require(args.device == "CPU" or args.reference is not None, "accelerator comparison requires CPU reference")
    governor = from_installation()
    require(governor.policy_sha256 == args.policy_sha256, "installed policy differs from reviewed policy")
    candidate = load_candidate(args.candidate)
    directory = args.root / args.candidate
    files = verify_files(directory, candidate)
    rows = inputs(directory, candidate)
    identity = {"pipeline": PIPELINE, "candidate": candidate,
                "files": files, "inputs_sha256": hashlib.sha256(canonical(rows)).hexdigest()}
    reference = reference_vectors(args.reference, identity, candidate["dimensions"]) if args.reference else None
    import numpy as np
    import openvino as ov
    import tokenizers
    version = ov.get_version()
    require(version.split("-")[0] == args.runtime_version, "OpenVINO version differs from reviewed runtime")
    core = ov.Core()
    model = core.read_model(str(directory / "model.onnx"))
    names = {p.get_any_name() for p in model.inputs}
    require({"input_ids", "attention_mask"} <= names <= set(rows[0]), "unsupported graph inputs")
    require(len(model.outputs) == 1, "ambiguous graph output")
    model.reshape({name: [1, BUCKET] for name in names})
    config = {"PERFORMANCE_HINT": "LATENCY"}
    if args.device == "CPU":
        config.update({"INFERENCE_NUM_THREADS": 1, "INFERENCE_PRECISION_HINT": "f32"})
    with governor.operation("compile", BUCKET) as lease, NativeDeadline(lease.deadline_ns):
        compiled = core.compile_model(model, args.device, config)
    devices = list(compiled.get_property("EXECUTION_DEVICES"))
    require(len(devices) == 1 and devices[0].split(".")[0] == args.device,
            "runtime execution device differs from explicit selection")
    vectors, durations = [], []
    for row in rows:
        tensors = {name: np.asarray([row[name]], dtype=np.int64) for name in names}
        with governor.operation("inference", BUCKET) as lease, NativeDeadline(lease.deadline_ns):
            started = time.monotonic()
            output = compiled(tensors)
            durations.append(time.monotonic() - started)
        vectors.append(pool(output[compiled.output(0)], row["attention_mask"], candidate))
    require(verify_files(directory, candidate) == files, "model changed during inference")
    cosines = None if reference is None else (np.asarray(vectors) * reference).sum(axis=1).tolist()
    scores = np.asarray(vectors) @ np.asarray(vectors).T
    write_new(args.output, {"schema": 1, "admission": "candidate-only", "identity": identity,
        "device": args.device, "execution_devices": devices, "openvino": version,
        "numpy": np.__version__, "tokenizers": tokenizers.__version__, "compile_config": config,
        "policy_sha256": governor.policy_sha256, "inference_seconds": durations, "vectors": vectors,
        "reference_cosines": cosines, "same_runtime_parity_passed":
            None if cosines is None else all(v >= 0.9999 for v in cosines),
        "semantic_ordering_passed": bool(scores[0, 1] > scores[0, 2]),
        "tool_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (Path(__file__), Path(__file__).with_name("candidates.py"),
                                     ROOT / "packages/openvino/inference_governor.py",
                                     ROOT / "packages/openvino/native_deadline.py")}})
    require(scores[0, 1] > scores[0, 2], "semantic smoke failed; failure result retained")
    require(cosines is None or all(v >= 0.9999 for v in cosines), "device parity failed; failure result retained")


if __name__ == "__main__":
    main()
