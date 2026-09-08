#!/usr/bin/env python3
"""Bounded CPU audit of tokenizer inputs and independent native semantics.

Requires the existing OpenVINO build lock in an isolated hub environment. No
downloads are permitted. Canonical native inputs explicitly prepend BOS=2 and
append EOS=1 to encode(add_special_tokens=False); the community runner instead
uses encode(add_special_tokens=True). Both behaviors are audited before using
the canonical arrays for all three numerical paths. This is not admission.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time


REPOSITORY = Path(__file__).resolve().parents[2]
COMMUNITY_REVISION = "5090578d9565bb06545b4552f76e6bc2c93e4a66"
COMMUNITY_TOKENIZER_SHA256 = "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47"
MAX_SAMPLES = 32
COOLDOWN_SECONDS = 0.15
TOLERANCES = {
    "upstream_vs_patched": {"minimum_cosine": 0.999999, "maximum_absolute_difference": 0.0001},
    "patched_vs_openvino": {"minimum_cosine": 0.999, "maximum_absolute_difference": 0.005},
    "upstream_vs_openvino": {"minimum_cosine": 0.999, "maximum_absolute_difference": 0.005},
}
MAXIMUM_NORM_ERROR = 0.005
CPU_CONFIG = {
    "INFERENCE_NUM_THREADS": 4,
    "NUM_STREAMS": 1,
    "PERFORMANCE_HINT": "LATENCY",
    "INFERENCE_PRECISION_HINT": "f32",
}


class AuditError(ValueError):
    """A bounded diagnostic failure whose message contains no private paths."""


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def value_digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AuditError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(_value):
    raise AuditError("nonfinite JSON constant")


def read_json(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        maximum = 64 * 1024 * 1024
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum:
            raise AuditError("JSON input must be a bounded nonempty regular file")
        raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise AuditError("JSON input grew beyond its bound")
    return json.loads(raw, object_pairs_hook=strict_object, parse_constant=reject_constant), hashlib.sha256(raw).hexdigest()


def publish(path, report):
    descriptor, temporary = tempfile.mkstemp(prefix=".native-audit-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded(report) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def load_inputs(path, buckets):
    manifest, manifest_hash = read_json(path)
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("production_admission") is not False
            or manifest.get("dimensions") != 768
            or manifest.get("sequence_buckets") != list(buckets)):
        raise AuditError("unsupported evaluation manifest/profile")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 4096:
        raise AuditError("expected 1..4096 manifest inputs")
    seen = set()
    for item in inputs:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise AuditError("invalid manifest input")
        expected = hashlib.sha256(b"cfetch-evaluation-input-v1\0" + item["text"].encode()).hexdigest()
        if (item.get("id") != expected or expected in seen
                or item.get("kind") not in ("query", "document")):
            raise AuditError("duplicate or invalid manifest input identity")
        seen.add(expected)
    return inputs, manifest_hash


def audit_tokenizers(inputs, canonical, community, buckets):
    rows, prepared = [], []
    canonical_vocab = canonical.get_vocab(with_added_tokens=True)
    community_vocab = community.get_vocab(with_added_tokens=True)
    for tokenizer in (canonical, community):
        tokenizer.no_padding()
        tokenizer.no_truncation()
        if [tokenizer.token_to_id(token) for token in ("<pad>", "<bos>", "<eos>")] != [0, 2, 1]:
            raise AuditError("tokenizer special-token IDs differ from PAD=0/BOS=2/EOS=1")
    for ordinal, item in enumerate(inputs):
        plain = canonical.encode(item["text"], add_special_tokens=False).ids
        canonical_auto = canonical.encode(item["text"], add_special_tokens=True).ids
        community_plain = community.encode(item["text"], add_special_tokens=False).ids
        community_auto = community.encode(item["text"], add_special_tokens=True).ids
        native = [2, *plain, 1]
        bucket = next((candidate for candidate in buckets if len(native) <= candidate), None)
        rows.append({
            "id": item["id"], "kind": item["kind"], "ordinal": ordinal,
            "canonical_token_count": len(native), "community_token_count": len(community_auto),
            "bucket": bucket, "plain_ids_equal": plain == community_plain,
            "prepared_ids_equal": native == community_auto,
            "canonical_automatic_matches_explicit_bos_eos": canonical_auto == native,
            "canonical_automatic_token_count": len(canonical_auto),
            "canonical_ids_sha256": value_digest(native),
            "community_ids_sha256": value_digest(community_auto),
            "refusal": "prefixed input exceeds 2048 tokens; truncation forbidden" if bucket is None else None,
        })
        if bucket is not None:
            prepared.append({"id": item["id"], "ordinal": ordinal, "ids": native, "bucket": bucket})
    return {
        "canonical_policy": "[2] + encode(add_special_tokens=False).ids + [1]",
        "community_policy": "encode(add_special_tokens=True).ids",
        "padding": "right PAD=0 to smallest frozen bucket; real tokens including BOS/EOS have mask=1",
        "vocabulary_equal": canonical_vocab == community_vocab,
        "canonical_vocabulary_sha256": value_digest(canonical_vocab),
        "community_vocabulary_sha256": value_digest(community_vocab),
        "scope": "prepared-input equivalence on this manifest, not every possible string",
        "passed": all(row["prepared_ids_equal"] and row["plain_ids_equal"] for row in rows),
        "inputs": rows,
    }, prepared


def select_samples(prepared, count):
    if type(count) is not int or not 1 <= count <= MAX_SAMPLES or count > len(prepared):
        raise AuditError("sample count must be 1..32 and no larger than runnable manifest inputs")
    short = [item for item in prepared if len(item["ids"]) <= 512]
    long = [item for item in prepared if len(item["ids"]) > 512]
    anchors = []
    if short:
        anchors.append(min(short, key=lambda item: (len(item["ids"]), item["ordinal"])))
    if long:
        anchors.append(max(long, key=lambda item: (len(item["ids"]), -item["ordinal"])))
    if count < len(anchors):
        raise AuditError("sample count cannot cover both available short and >512-token inputs")
    chosen = {item["id"] for item in anchors}
    for item in prepared:
        if len(anchors) == count:
            break
        if item["id"] not in chosen:
            anchors.append(item)
            chosen.add(item["id"])
    return sorted(anchors, key=lambda item: (item["bucket"], item["ordinal"]))


def build_upstream(source_dir):
    """Independent reference: ordinary upstream mask and pooling, no converter helpers."""
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        str(source_dir), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="sdpa",
    ).cpu().eval()
    weights = []
    for directory, shape in (("2_Dense", (3072, 768)), ("3_Dense", (768, 3072))):
        tensors = load_file(str(source_dir / directory / "model.safetensors"), device="cpu")
        if set(tensors) != {"linear.weight"} or tuple(tensors["linear.weight"].shape) != shape:
            raise AuditError("independent Dense tensor keys/shape mismatch")
        weights.append(tensors["linear.weight"].to(dtype=torch.float32))
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    def forward(ids, mask):
        hidden = backbone(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
        expanded = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.dtype)
        # Ordinary sentence-transformers mean semantics, deliberately not the
        # converter's torch.where mask or patched attention-mask dictionary.
        pooled = (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1e-9)
        projected = functional.linear(pooled.to(torch.float32), weights[0])
        projected = functional.linear(projected, weights[1])
        return functional.normalize(projected, p=2.0, dim=1)

    return forward


def audit_attention_masks(source_dir, buckets, report):
    """Compare real-query edges with the locked Gemma3 factories, without weights."""
    import inspect
    import torch
    from transformers import AutoConfig
    from transformers.models.gemma3 import modeling_gemma3
    from packages.openvino import convert

    report.update({"passed": False, "cases": [], "model_initialization": False})
    source_config, source_hash = read_json(source_dir / "config.json")
    config = AutoConfig.from_pretrained(str(source_dir), local_files_only=True, trust_remote_code=False)
    config._attn_implementation = "sdpa"
    report["configuration"] = {
        "source_config_sha256": source_hash,
        "source_sliding_window": source_config.get("sliding_window"),
        "source_use_bidirectional_attention": source_config.get("use_bidirectional_attention"),
        "effective_sliding_window": config.sliding_window,
        "effective_use_bidirectional_attention": config.use_bidirectional_attention,
        "attention_implementation": config._attn_implementation,
        "transformers_version": importlib.metadata.version("transformers"),
    }
    report["upstream_implementation_sha256"] = {
        "modeling_gemma3.py": digest(Path(modeling_gemma3.__file__)),
        "masking_utils.py": digest(Path(inspect.getfile(modeling_gemma3.create_causal_mask))),
        "configuration_gemma3.py": digest(Path(inspect.getfile(type(config)))),
    }
    if (type(source_config.get("sliding_window")) is not int or source_config["sliding_window"] != 512
            or source_config.get("use_bidirectional_attention") is not True
            or type(config.sliding_window) is not int or config.sliding_window != 257
            or config.use_bidirectional_attention is not True
            or report["configuration"]["transformers_version"] != "5.10.1"):
        raise AuditError("expected pinned Gemma3 source window 512 and loaded bidirectional window 257")
    report["contract"] = (
        "exact upstream equality on real-query rows, including exclusion of padding keys; "
        "only diagonal additions to otherwise empty padded-query rows are permitted"
    )
    for bucket in buckets:
        real_lengths = sorted({length for length in (1, min(bucket, 256), 257, 258, 512, bucket)
                               if 1 <= length <= bucket})
        for real_length in real_lengths:
            mask = torch.zeros((1, bucket), dtype=torch.int64)
            mask[:, :real_length] = 1
            # Use the actual model hidden width; these are only zero-valued
            # factory inputs, never token embeddings or a model forward call.
            kwargs = {
                "config": config, "inputs_embeds": torch.zeros((1, bucket, 768), dtype=torch.float32),
                "attention_mask": mask, "past_key_values": None,
                "position_ids": torch.arange(bucket, dtype=torch.int64).unsqueeze(0),
                "or_mask_function": lambda *args: torch.tensor(True, dtype=torch.bool),
            }
            sliding_kwargs = dict(kwargs)
            sliding_kwargs["or_mask_function"] = modeling_gemma3._bidirectional_window_overlay(config.sliding_window)
            with torch.no_grad():
                upstream = {
                    "full_attention": modeling_gemma3.create_causal_mask(**kwargs),
                    "sliding_attention": modeling_gemma3.create_sliding_window_causal_mask(**sliding_kwargs),
                }
                patched = convert.safe_bidirectional_attention_masks(mask)
            case = {"bucket": bucket, "real_tokens": real_length, "masks": {}, "passed": False}
            report["cases"].append(case)
            diagonal = torch.eye(bucket, dtype=torch.bool).reshape(1, 1, bucket, bucket)
            padded_queries = (torch.arange(bucket) >= real_length).reshape(1, 1, bucket, 1)
            for name in ("full_attention", "sliding_attention"):
                reference, candidate = upstream[name], patched[name]
                for value in (reference, candidate):
                    if (not isinstance(value, torch.Tensor) or value.dtype != torch.bool
                            or list(value.shape) != [1, 1, bucket, bucket] or value.device.type != "cpu"):
                        raise AuditError("mask factory must return a materialized CPU bool [1,1,bucket,bucket] matrix")
                differences = reference != candidate
                real_differences = int(differences[:, :, :real_length, :].sum().item())
                reference_padding = int(reference[:, :, :real_length, real_length:].sum().item())
                candidate_padding = int(candidate[:, :, :real_length, real_length:].sum().item())
                permitted_repairs = (~reference.any(dim=-1, keepdim=True)) & padded_queries & diagonal
                unexpected_changes = int((candidate != (reference | permitted_repairs)).sum().item())
                case["masks"][name] = {
                    "shape": list(reference.shape), "real_query_edge_mismatches": real_differences,
                    "padded_query_edge_mismatches": int(differences[:, :, real_length:, :].sum().item()),
                    "upstream_real_query_padding_edges": reference_padding,
                    "patched_real_query_padding_edges": candidate_padding,
                    "permitted_empty_padded_diagonal_repairs": int(permitted_repairs.sum().item()),
                    "unexpected_edge_changes": unexpected_changes,
                    "upstream_sha256": hashlib.sha256(reference.contiguous().numpy().tobytes()).hexdigest(),
                    "patched_sha256": hashlib.sha256(candidate.contiguous().numpy().tobytes()).hexdigest(),
                    "upstream_real_query_sha256": hashlib.sha256(reference[:, :, :real_length, :].contiguous().numpy().tobytes()).hexdigest(),
                    "patched_real_query_sha256": hashlib.sha256(candidate[:, :, :real_length, :].contiguous().numpy().tobytes()).hexdigest(),
                    "passed": not any((real_differences, reference_padding, candidate_padding, unexpected_changes)),
                }
            case["passed"] = all(value["passed"] for value in case["masks"].values())
    report["passed"] = bool(report["cases"]) and all(case["passed"] for case in report["cases"])
    if not report["passed"]:
        raise AuditError("independent upstream attention-mask comparison failed")


def vector_summary(vector):
    import numpy as np

    result = {"shape": list(vector.shape), "finite": bool(np.isfinite(vector).all())}
    result["nonfinite_components"] = int(np.count_nonzero(~np.isfinite(vector)))
    result["norm"] = float(np.linalg.norm(vector.astype(np.float64))) if result["finite"] else None
    result["valid"] = (result["shape"] == [1, 768] and result["finite"]
                       and abs(result["norm"] - 1.0) <= MAXIMUM_NORM_ERROR)
    result["f32_sha256"] = hashlib.sha256(vector.astype("<f4", copy=False).tobytes()).hexdigest()
    return result


def compare_vectors(first, second, tolerance):
    import numpy as np

    if not vector_summary(first)["valid"] or not vector_summary(second)["valid"]:
        return {"passed": False, "error": "invalid shape, nonfinite components or nonunit norm"}
    left, right = first.astype(np.float64).ravel(), second.astype(np.float64).ravel()
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    maximum = float(np.max(np.abs(left - right)))
    return {
        "cosine": cosine, "maximum_absolute_difference": maximum,
        "passed": cosine >= tolerance["minimum_cosine"] and maximum <= tolerance["maximum_absolute_difference"],
    }


def failure(error):
    # Library errors may embed private filenames or input text. Numeric details
    # belong in the explicit metrics; only our controlled errors expose prose.
    return {"type": type(error).__name__, "detail": str(error) if isinstance(error, AuditError) else "operation failed"}


def run(args, report):
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise AuditError("requires isolated Linux CPython 3.12 with the OpenVINO build lock")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.path.insert(0, str(REPOSITORY))
    import numpy as np
    import openvino as ov
    import torch
    from tokenizers import Tokenizer
    from packages.openvino import convert, manifest as native

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    inputs, manifest_hash = load_inputs(args.manifest, native.SEQUENCE_BUCKETS)
    report["manifest_sha256"] = manifest_hash
    report["stage"] = "source_and_artifact_validation"
    native_source = args.source_dir.resolve()
    convert.verify_source_files(native_source)
    convert.validate_semantic_source(native_source)
    community_path = args.community_model_root / "tokenizer.json"
    if digest(community_path) != COMMUNITY_TOKENIZER_SHA256:
        raise AuditError("community tokenizer digest differs from the pinned candidate")
    artifact_document, artifact_hash = read_json(args.artifact_dir / "artifact-manifest.json")
    artifact = native.load_artifact(args.artifact_dir, "artifact-manifest.json", artifact_hash)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "openvino", "torch", "transformers", "safetensors", "tokenizers")}
    if any(versions[name] != version for name, version in artifact.conversion_versions.items()):
        raise AuditError("installed conversion libraries differ from artifact provenance")
    report["provenance"] = {
        "source": {"model": native.MODEL, "revision": native.MODEL_REVISION,
                   "verified_files": dict(native.PINNED_SOURCE_FILE_SHA256)},
        "community": {"model": "onnx-community/embeddinggemma-300m-ONNX", "revision": COMMUNITY_REVISION,
                      "tokenizer_sha256": COMMUNITY_TOKENIZER_SHA256, "onnx_execution": False},
        "artifact": {"manifest_sha256": artifact_hash, "files": artifact_document["files"],
                     "conversion": artifact_document["conversion"]},
        "versions": versions, "openvino_build": ov.get_version(),
        "implementation_sha256": {name: digest(REPOSITORY / relative) for name, relative in {
            "audit_native.py": "experiments/memory-retrieval/audit_native.py",
            "convert.py": "packages/openvino/convert.py", "manifest.py": "packages/openvino/manifest.py",
            "legal.py": "packages/openvino/legal.py", "requirements-build.lock": "packages/openvino/requirements-build.lock",
        }.items()},
    }
    report["stage"] = "tokenizer_audit"
    tokenization, prepared = audit_tokenizers(
        inputs, Tokenizer.from_file(str(artifact.tokenizer_json)),
        Tokenizer.from_file(str(community_path)), native.SEQUENCE_BUCKETS,
    )
    report["tokenizers"] = tokenization
    selected = select_samples(prepared, args.sample_count)
    report["selection"] = {
        "requested": args.sample_count,
        "policy": "shortest <=512 and longest >512 when present, then manifest order; execute sorted by bucket/ordinal",
        "selected_ids": [item["id"] for item in selected],
        "short_inputs_available": sum(len(item["ids"]) <= 512 for item in prepared),
        "long_inputs_available": sum(len(item["ids"]) > 512 for item in prepared),
        "overlength_refusals": len(inputs) - len(prepared),
    }
    report["stage"] = "independent_attention_mask_audit"
    report["attention_masks"] = {}
    audit_attention_masks(native_source, native.SEQUENCE_BUCKETS, report["attention_masks"])
    report["stage"] = "cpu_model_initialization"
    upstream = build_upstream(native_source)
    patched = convert.build_torch_pipeline(native_source).cpu().eval()
    core = ov.Core()
    graph = core.read_model(str(artifact.graph_xml), str(artifact.graph_bin))
    report["samples"] = []
    compiled, compiled_bucket = None, None
    for item in selected:
        report["stage"] = "cpu_semantic_comparison"
        count, bucket = len(item["ids"]), item["bucket"]
        arrays = {
            "input_ids": np.asarray([item["ids"] + [0] * (bucket - count)], dtype=np.int64),
            "attention_mask": np.asarray([[1] * count + [0] * (bucket - count)], dtype=np.int64),
        }
        sample = {"id": item["id"], "token_count": count, "bucket": bucket,
                  "array_sha256": {name: hashlib.sha256(value.astype("<i8").tobytes()).hexdigest()
                                   for name, value in arrays.items()}, "outputs": {}, "comparisons": {}}
        report["samples"].append(sample)
        if compiled_bucket != bucket:
            compiled = None  # Retain at most one compiled bucket, never seven copies.
            static_graph = graph.clone()
            static_graph.reshape({artifact.input_ids_name: [1, bucket], artifact.attention_mask_name: [1, bucket]})
            compiled = core.compile_model(static_graph, "CPU", CPU_CONFIG)
            compiled_bucket = bucket
        devices = list(compiled.get_property("EXECUTION_DEVICES"))
        if devices != ["CPU"]:
            raise AuditError("OpenVINO execution placement is not exclusively CPU")
        sample["execution_devices"] = devices
        sample["observed_precision_hint"] = str(compiled.get_property("INFERENCE_PRECISION_HINT"))
        vectors = {}
        for name in ("upstream", "patched", "openvino"):
            start = time.monotonic()
            try:
                if name == "openvino":
                    result = compiled({artifact.input_ids_name: arrays["input_ids"],
                                       artifact.attention_mask_name: arrays["attention_mask"]})
                    vector = np.asarray(result[compiled.output(artifact.output_name)], dtype=np.float32).copy()
                else:
                    with torch.no_grad():
                        result = (upstream if name == "upstream" else patched)(
                            torch.from_numpy(arrays["input_ids"]), torch.from_numpy(arrays["attention_mask"]))
                        vector = result.detach().cpu().to(torch.float32).numpy().copy()
                vectors[name] = vector
                sample["outputs"][name] = vector_summary(vector)
            except Exception as error:
                sample["outputs"][name] = {"valid": False, "error": failure(error)}
            finally:
                sample["outputs"][name]["seconds"] = time.monotonic() - start
                time.sleep(COOLDOWN_SECONDS)
        for pair, tolerance in TOLERANCES.items():
            first, second = pair.split("_vs_")
            sample["comparisons"][pair] = (
                compare_vectors(vectors[first], vectors[second], tolerance)
                if first in vectors and second in vectors else {"passed": False, "error": "execution failed"})
        sample["passed"] = all(value["passed"] for value in sample["comparisons"].values())
        print(f"native audit: {len(report['samples'])}/{len(selected)} samples", flush=True)
    # Recheck the bound input bytes after model use; no output should vouch for
    # source/artifact changes that occurred while this audit was running.
    convert.verify_source_files(native_source)
    native.load_artifact(args.artifact_dir, "artifact-manifest.json", artifact_hash)
    if digest(args.manifest) != manifest_hash or digest(community_path) != COMMUNITY_TOKENIZER_SHA256:
        raise AuditError("audit inputs changed during execution")
    report["passed"] = (report["attention_masks"]["passed"] and tokenization["passed"]
                        and all(item["passed"] for item in report["samples"]))
    report["stage"] = "complete"


def sample_count(value):
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("sample count must be an integer in 1..32") from error
    if not 1 <= count <= MAX_SAMPLES:
        raise argparse.ArgumentTypeError("sample count must be in 1..32")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source-dir", "community-model-root", "artifact-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sample-count", type=sample_count, required=True)
    args = parser.parse_args()
    if os.path.lexists(args.output) or not args.output.parent.is_dir():
        parser.error("output must be absent and its parent directory must exist")
    report = {
        "schema_version": 1, "production_admission": False, "passed": False, "stage": "preflight",
        "purpose": "independent upstream semantics, patched export parity and manifest-tokenizer equivalence",
        "reference": "fresh AutoModel(sdpa) with ordinary 2D mask; independent mean + Dense2 + Dense3 + L2",
        "export_parity": "converter patched safe-mask pipeline versus native OpenVINO; not the independent reference",
        "execution": {"device": "CPU", "batch_size": 1, "torch_intra_threads": 4,
                      "torch_inter_threads": 1, "openvino_config": CPU_CONFIG,
                      "cooldown_seconds_after_each_call": COOLDOWN_SECONDS},
        "tolerances": TOLERANCES, "maximum_norm_error": MAXIMUM_NORM_ERROR,
    }
    try:
        run(args, report)
    except Exception as error:
        report["passed"] = False
        report["error"] = failure(error)
    try:
        publish(args.output, report)
    except OSError:
        print("native audit: could not publish report without replacement", file=sys.stderr)
        return 1
    print("native audit: " + ("passed" if report["passed"] else "failed"), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
