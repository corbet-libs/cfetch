#!/usr/bin/env python3
"""Build-only repair of pinned EmbeddingGemma attention masks.

Requires onnx 1.19.1 and numpy in the build environment. No Python runtime ships
with cfetch. Source weights/tokenizer stay unchanged; the output is a separate
model pack, never an edit to an upstream cache snapshot.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator

PIPELINE = "embeddinggemma-local-cpu-v1-fastembed6.0.2-ort2.0.0rc13"
PREFIX = "cfetch_local_attention/"
PADDING = "/model/attn_mask_reformat/Expand/output_0"


def mask_nodes():
    initializers = []
    nodes = []
    def constant(name, array):
        name = PREFIX + name
        initializers.append(nh.from_array(np.asarray(array), name))
        return name
    def op(kind, name, inputs, **attributes):
        name = PREFIX + name
        nodes.append(h.make_node(kind, inputs, [name], name=name, **attributes))
        return name
    zero = constant("zero", np.int64(0))
    one = constant("one", np.int64(1))
    radius = constant("radius", np.int64(257))
    axis0 = constant("axis0", np.array([0], dtype=np.int64))
    axis1 = constant("axis1", np.array([1], dtype=np.int64))
    axis3 = constant("axis3", np.array([3], dtype=np.int64))
    float_zero = constant("float_zero", np.float32(0))
    negative = constant("negative", np.float32(-3.4028234663852886e38))
    shape = op("Shape", "shape", ["attention_mask"])
    length = op("Gather", "length", [shape, one], axis=0)
    positions = op("Range", "positions", [zero, length, one])
    row = op("Unsqueeze", "row", [positions, axis0])
    col = op("Unsqueeze", "col", [positions, axis1])
    distance = op("Abs", "distance", [op("Sub", "difference", [row, col])])
    local = op("Less", "local", [distance, radius])
    diagonal = op("Equal", "diagonal", [distance, zero])
    real_keys = op("Equal", "real_keys", [PADDING, float_zero])
    permitted = op("And", "permitted", [real_keys, local])
    integer = op("Cast", "integer", [permitted], to=T.INT64)
    count = op("ReduceSum", "count", [integer, axis3], keepdims=1)
    empty = op("Equal", "empty", [count, zero])
    rescue = op("And", "rescue", [empty, diagonal])
    safe = op("Or", "safe", [permitted, rescue])
    output = op("Where", "mask", [safe, float_zero, negative])
    return nodes, initializers, output


def check_mask(nodes, initializers, output):
    graph = h.make_graph(nodes, "local-mask", [
        h.make_tensor_value_info("attention_mask", T.INT64, [None, None]),
        h.make_tensor_value_info(PADDING, T.FLOAT, [None, 3, None, None]),
    ], [h.make_tensor_value_info(output, T.FLOAT, [None, 3, None, None])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 21)], ir_version=10)
    onnx.checker.check_model(model)
    evaluator = ReferenceEvaluator(model)
    for size in (256, 257, 258):
        attention = np.ones((2, size), dtype=np.int64)
        attention[1, 1:] = 0
        padding = np.broadcast_to((1-attention[:, None, None, :]).astype(np.float32) * np.finfo(np.float32).min,
                                  (2, 3, size, size)).copy()
        actual = evaluator.run(None, {"attention_mask": attention, PADDING: padding})[0]
        distance = abs(np.arange(size)[:, None] - np.arange(size)[None, :])
        expected = (distance < 257)[None, None, :, :] & (attention[:, None, None, :] != 0)
        expected |= (~expected.any(axis=-1, keepdims=True)) & np.eye(size, dtype=bool)[None, None, :, :]
        np.testing.assert_array_equal(actual == 0, np.broadcast_to(expected, actual.shape))
        assert np.isfinite(actual).all()
    print("mask boundaries 256/257/258 and fully masked padded rows passed")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def build(source, output):
    source = source.resolve(strict=True)
    pinned = json.loads((Path(__file__).parents[1] / "experiments/npu-ort-foundation/embeddinggemma-cache.json").read_text())
    for name, expected in pinned["files"].items():
        path = source / name
        assert path.is_file() and path.stat().st_size == expected["bytes"], name
        if "sha256" in expected: assert sha(path) == expected["sha256"], name
        else:
            content = path.read_bytes()
            assert hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest() == expected["git_blob_sha1"], name
    graph = onnx.load(source / "onnx/model.onnx", load_external_data=False)
    assert all(not node.name.startswith(PREFIX) for node in graph.graph.node)
    nodes, initializers, mask = mask_nodes()
    check_mask(nodes, initializers, mask)
    changed = []
    original = list(graph.graph.node)
    insertion = next(i+1 for i, node in enumerate(original) if PADDING in node.output)
    for layer in range(24):
        matches = [node for node in original if node.name == f"/model/layers.{layer}/attn/MultiHeadAttention"]
        assert len(matches) == 1 and matches[0].input[5] == PADDING, layer
        if layer % 6 != 5:
            matches[0].input[5] = mask
            changed.append(layer)
    assert len(changed) == 20
    del graph.graph.node[:]
    graph.graph.node.extend(original[:insertion] + nodes + original[insertion:])
    graph.graph.initializer.extend(initializers)
    output.mkdir()  # Refuse replacing any prior/partly built pack.
    for name in pinned["files"]:
        if name == "onnx/model.onnx": continue
        os.link(source / name, output / Path(name).name)
    onnx.save_model(graph, output / "model.onnx")
    onnx.checker.check_model(str(output / "model.onnx"))
    manifest = {"pipeline": PIPELINE, "model": "google/embeddinggemma-300m", "files": {}}
    for name in ("model.onnx", "model.onnx_data", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = output / name
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"changed_layers": changed, "graph_sha256": sha(output / "model.onnx")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="complete pinned Hugging Face snapshot")
    parser.add_argument("output", type=Path, help="new directory on the same filesystem")
    args = parser.parse_args()
    build(args.source, args.output)
