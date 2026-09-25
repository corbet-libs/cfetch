#!/usr/bin/env python3
"""Build the standard-operator EmbeddingGemma model pack with int8 weights.

Stages, each usable alone:
  export   pinned sentence-transformers snapshot -> fp32 ONNX (opset 17, eager
           attention: plain MatMul/Softmax, no vendor contrib operators). Needs
           torch, transformers and sentence-transformers in the build environment.
  quantize fp32 ONNX -> symmetric per-output-channel int8 weights for every
           constant MatMul operand (DequantizeLinear -> MatMul) and a per-row int8
           embedding table (Gather int8 rows, Cast, Mul by the row scale).
           Activations stay fp32. Needs onnx and numpy only.
  pack     graph + tokenizer files -> a content-addressed pack with manifest.json.

No Python runtime ships with cfetch; the pack is loaded by the Rust binary and
must pass `cfetch qualify-model` against canonical reference rows before use.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PIPELINE = "embeddinggemma-plain-int8w-v1-fastembed6-ort2.0.0rc13"
MODEL = "google/embeddinggemma-300m"
OPSET = 17
PACK_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def export(source: Path, output: Path) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    class Wrapped(torch.nn.Module):
        def __init__(self, st):
            super().__init__()
            self.st = st

        def forward(self, input_ids, attention_mask):
            # Precomputed masks with the library's bidirectional semantics: full
            # layers see every valid key, sliding layers see |q - kv| < window.
            # Passing the mapping avoids the untraceable vmap mask builder.
            backbone = self.st[0].auto_model
            window = backbone.config.sliding_window
            length = input_ids.shape[1]
            position = torch.arange(length, device=input_ids.device)
            near = (position[:, None] - position[None, :]).abs() < window
            valid = attention_mask[:, None, None, :].bool()
            floor = torch.finfo(torch.float32).min
            zero = torch.zeros((), dtype=torch.float32)
            full = torch.where(valid, zero, floor).expand(-1, 1, length, length)
            sliding = torch.where(valid & near[None, None], zero, floor)
            hidden = backbone(
                input_ids=input_ids,
                attention_mask={"full_attention": full, "sliding_attention": sliding},
                use_cache=False,
            ).last_hidden_state
            features = {"input_ids": input_ids, "attention_mask": attention_mask, "token_embeddings": hidden}
            for module in list(self.st)[1:]:
                features = module(features)
            return hidden, features["sentence_embedding"]

    st = SentenceTransformer(
        str(source), device="cpu", model_kwargs={"attn_implementation": "eager", "dtype": torch.float32}
    )
    model = Wrapped(st.eval()).eval()
    ids = torch.tensor([[2] + [100] * 40 + [1]], dtype=torch.int64)
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (ids, torch.ones_like(ids)),
            str(output / "model.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state", "sentence_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
                "sentence_embedding": {0: "batch_size"},
            },
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    save(onnx.load(str(output / "model.onnx")), output)


def symmetric_int8(weights: np.ndarray, axis: int):
    """Round-half-to-even int8 in [-127, 127] with one scale per slice along `axis`."""
    reduce_axes = tuple(i for i in range(weights.ndim) if i != axis)
    scale = np.abs(weights).max(axis=reduce_axes) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    shape = [1] * weights.ndim
    shape[axis] = -1
    quantized = np.clip(np.rint(weights / scale.reshape(shape)), -127, 127).astype(np.int8)
    return quantized, scale


def quantize(model: onnx.ModelProto, min_matmul: int = 4096, min_table: int = 1 << 20) -> onnx.ModelProto:
    graph = model.graph
    initializers = {i.name: i for i in graph.initializer}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    nodes, drop = [], set()
    for node in graph.node:
        if node.op_type == "MatMul" and node.input[1] in initializers:
            weights = numpy_helper.to_array(initializers[node.input[1]])
            if weights.ndim == 2 and weights.size >= min_matmul:
                base = node.input[1]
                q, scale = symmetric_int8(weights, axis=1)
                graph.initializer.extend([
                    numpy_helper.from_array(q, base + "_q8"),
                    numpy_helper.from_array(scale, base + "_q8_scale"),
                    numpy_helper.from_array(np.zeros(scale.shape, np.int8), base + "_q8_zp"),
                ])
                dequantized = base + "_q8_dq"
                nodes.append(helper.make_node(
                    "DequantizeLinear", [base + "_q8", base + "_q8_scale", base + "_q8_zp"],
                    [dequantized], axis=1, name=dequantized))
                node.input[1] = dequantized
                if all(c.op_type == "MatMul" for c in consumers[base]):
                    drop.add(base)
        elif node.op_type == "Gather" and node.input[0] in initializers:
            table = numpy_helper.to_array(initializers[node.input[0]])
            if table.ndim == 2 and table.size >= min_table and len(consumers[node.input[0]]) == 1:
                base = node.input[0]
                q, scale = symmetric_int8(table, axis=0)
                graph.initializer.extend([
                    numpy_helper.from_array(q, base + "_q8"),
                    numpy_helper.from_array(scale.reshape(-1, 1), base + "_q8_scale"),
                ])
                rows, scales, cast = base + "_q8_rows", base + "_q8_row_scale", base + "_q8_cast"
                nodes += [
                    helper.make_node("Gather", [base + "_q8", node.input[1]], [rows], axis=0, name=rows),
                    helper.make_node("Gather", [base + "_q8_scale", node.input[1]], [scales], axis=0, name=scales),
                    helper.make_node("Cast", [rows], [cast], to=TensorProto.FLOAT, name=cast),
                    helper.make_node("Mul", [cast, scales], [node.output[0]], name=base + "_q8_mul"),
                ]
                drop.add(base)
                continue
        nodes.append(node)
    # Replacement nodes precede their consumers, so the order stays topological.
    del graph.node[:]
    graph.node.extend(nodes)
    kept = [i for i in graph.initializer if i.name not in drop]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    return model


def save(model: onnx.ModelProto, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for stale in ("model.onnx", "model.onnx_data"):
        (output / stale).unlink(missing_ok=True)
    onnx.save_model(model, str(output / "model.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="model.onnx_data", size_threshold=1024)
    onnx.checker.check_model(str(output / "model.onnx"))
    domains = {n.domain for n in onnx.load(str(output / "model.onnx"), load_external_data=False).graph.node}
    if domains - {"", "ai.onnx"}:
        raise SystemExit(f"non-standard operator domains in export: {sorted(domains)}")


def pack(graph: Path, tokenizer: Path, output: Path) -> str:
    output.mkdir(parents=True, exist_ok=False)
    for name in ("model.onnx", "model.onnx_data"):
        shutil.copyfile(graph / name, output / name)
    for name in PACK_FILES:
        shutil.copyfile(tokenizer / name, output / name)
    files = {}
    for name in sorted(("model.onnx", "model.onnx_data") + PACK_FILES):
        data = (output / name).read_bytes()
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {"files": files, "model": MODEL, "pipeline": PIPELINE}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    e = stages.add_parser("export")
    e.add_argument("source", type=Path)
    e.add_argument("output", type=Path)
    q = stages.add_parser("quantize")
    q.add_argument("source", type=Path)
    q.add_argument("output", type=Path)
    p = stages.add_parser("pack")
    p.add_argument("graph", type=Path)
    p.add_argument("tokenizer", type=Path, help="directory holding the verified tokenizer files")
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.stage == "export":
        export(args.source, args.output)
    elif args.stage == "quantize":
        save(quantize(onnx.load(str(args.source / "model.onnx"))), args.output)
    else:
        print(pack(args.graph, args.tokenizer, args.output))


if __name__ == "__main__":
    main()
