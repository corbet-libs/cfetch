#!/usr/bin/env python3
"""Convert an already-fetched exact EmbeddingGemma revision to OpenVINO IR.

This command never downloads a model.  It first verifies every semantic source
file against the cfetch profile, then builds one dynamic-sequence IR containing
the transformer, masked mean, both identity-activation dense projections, and
L2 normalization.  It rewrites only the two frozen K=1 rotary outer products
to algebraically identical broadcast multiplication.  The adapter reshapes
that IR into all seven static buckets before compiling it for one exact
manifest-selected device.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

if __package__:
    from .legal import LEGAL_FILES, LegalError, copy_legal_payload
    from .manifest import (
        DIMENSIONS,
        MAX_TOKENS,
        MODEL,
        MODEL_REVISION,
        PINNED_SOURCE_FILE_SHA256,
        SEQUENCE_BUCKETS,
        SOURCE_MIRROR,
        SOURCE_MIRROR_REVISION,
    )
else:
    from legal import (  # type: ignore[no-redef]
        LEGAL_FILES,
        LegalError,
        copy_legal_payload,
    )
    from manifest import (  # type: ignore[no-redef]
        DIMENSIONS,
        MAX_TOKENS,
        MODEL,
        MODEL_REVISION,
        PINNED_SOURCE_FILE_SHA256,
        SEQUENCE_BUCKETS,
        SOURCE_MIRROR,
        SOURCE_MIRROR_REVISION,
    )


EXPECTED_MODULES = [
    {
        "idx": 0,
        "name": "0",
        "path": "",
        "type": "sentence_transformers.models.Transformer",
    },
    {
        "idx": 1,
        "name": "1",
        "path": "1_Pooling",
        "type": "sentence_transformers.models.Pooling",
    },
    {
        "idx": 2,
        "name": "2",
        "path": "2_Dense",
        "type": "sentence_transformers.models.Dense",
    },
    {
        "idx": 3,
        "name": "3",
        "path": "3_Dense",
        "type": "sentence_transformers.models.Dense",
    },
    {
        "idx": 4,
        "name": "4",
        "path": "4_Normalize",
        "type": "sentence_transformers.models.Normalize",
    },
]
EXPECTED_POOLING = {
    "word_embedding_dimension": 768,
    "pooling_mode_cls_token": False,
    "pooling_mode_mean_tokens": True,
    "pooling_mode_max_tokens": False,
    "pooling_mode_mean_sqrt_len_tokens": False,
    "pooling_mode_weightedmean_tokens": False,
    "pooling_mode_lasttoken": False,
    "include_prompt": True,
}
EXPECTED_DENSE_2 = {
    "in_features": 768,
    "out_features": 3072,
    "bias": False,
    "activation_function": "torch.nn.modules.linear.Identity",
}
EXPECTED_DENSE_3 = {
    "in_features": 3072,
    "out_features": 768,
    "bias": False,
    "activation_function": "torch.nn.modules.linear.Identity",
}
EXPECTED_SENTENCE_BERT = {"max_seq_length": 2048, "do_lower_case": False}
EXPECTED_SLIDING_WINDOW = 512
# Gemma3TextConfig converts the serialized full bidirectional window into an
# exclusive radius: (512 // 2) + 1. Passing the serialized value to SDPA would
# double the neighborhood and change long-input embeddings.
EXPECTED_EFFECTIVE_SLIDING_WINDOW = EXPECTED_SLIDING_WINDOW // 2 + 1


class ConversionError(ValueError):
    """The supplied source or converted artifact violated the recipe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_files(
    source_dir: Path,
    expected: Mapping[str, str] = PINNED_SOURCE_FILE_SHA256,
) -> None:
    source_dir = source_dir.resolve()
    for relative, expected_digest in expected.items():
        path = source_dir / relative
        if not path.is_file():
            raise ConversionError(f"pinned source file is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected_digest:
            raise ConversionError(
                f"pinned source digest mismatch for {relative}: expected "
                f"{expected_digest}, found {actual}"
            )


def _read_json(path: Path) -> Any:
    try:
        metadata = path.stat()
    except OSError as error:
        raise ConversionError(f"cannot inspect {path}: {error}") from error
    if path.is_symlink() or not path.is_file():
        raise ConversionError(f"{path} must be a regular non-symlink file")
    if metadata.st_size < 1 or metadata.st_size > 1024 * 1024:
        raise ConversionError(f"{path} must contain bounded nonempty JSON")
    try:
        with path.open("rb") as source:
            raw = source.read(1024 * 1024 + 1)
    except OSError as error:
        raise ConversionError(f"cannot read {path}: {error}") from error
    if not raw or len(raw) > 1024 * 1024:
        raise ConversionError(f"{path} changed size while it was read")
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConversionError(f"{path} is not valid UTF-8 JSON: {error}") from error


def validate_semantic_source(source_dir: Path) -> None:
    checks = {
        "modules.json": EXPECTED_MODULES,
        "1_Pooling/config.json": EXPECTED_POOLING,
        "2_Dense/config.json": EXPECTED_DENSE_2,
        "3_Dense/config.json": EXPECTED_DENSE_3,
        "sentence_bert_config.json": EXPECTED_SENTENCE_BERT,
    }
    for relative, expected in checks.items():
        if _read_json(source_dir / relative) != expected:
            raise ConversionError(
                f"{relative} does not describe the frozen mean+dense2+dense3+L2 pipeline"
            )
    config = _read_json(source_dir / "config.json")
    if not isinstance(config, dict):
        raise ConversionError("config.json must contain an object")
    expected_model_fields = {
        "architectures": ["Gemma3TextModel"],
        "dtype": "float32",
        "hidden_size": 768,
        "max_position_embeddings": 2048,
        "model_type": "gemma3_text",
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "sliding_window": EXPECTED_SLIDING_WINDOW,
        "use_bidirectional_attention": True,
    }
    for field, expected in expected_model_fields.items():
        if config.get(field) != expected:
            raise ConversionError(
                f"config.json {field}={config.get(field)!r}, expected {expected!r}"
            )


def _load_dense_weight(path: Path, expected_shape: tuple[int, int]):
    from safetensors.torch import load_file

    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {"linear.weight"}:
        raise ConversionError(
            f"{path} must contain exactly the bias-free linear.weight tensor"
        )
    weight = tensors["linear.weight"]
    if tuple(weight.shape) != expected_shape:
        raise ConversionError(
            f"{path} linear.weight has shape {tuple(weight.shape)}, expected {expected_shape}"
        )
    return weight


def masked_mean(token_embeddings: Any, attention_mask: Any) -> Any:
    """Reduce only real token rows, even if a backend poisons padded rows.

    Some accelerator decompositions of scaled dot-product attention return
    NaN for a fully masked padded-query row.  Multiplication is not a mask for
    that value because IEEE-754 defines ``NaN * 0`` as NaN.  Elementwise
    selection makes the sentence-transformers pooling contract explicit:
    padded rows do not participate in the reduction at all.
    """
    import torch

    real_tokens = attention_mask.unsqueeze(-1).bool()
    selected = torch.where(
        real_tokens, token_embeddings, torch.zeros_like(token_embeddings)
    )
    denominator = (
        attention_mask.sum(dim=1, keepdim=True)
        .to(token_embeddings.dtype)
        .clamp_min(1.0)
    )
    return selected.sum(dim=1) / denominator


def _ensure_nonempty_attention_rows(allowed: Any, diagonal: Any) -> Any:
    """Add only the diagonal bit of an otherwise fully masked query row."""
    empty_rows = ~allowed.any(dim=-1, keepdim=True)
    return allowed | (empty_rows & diagonal)


def safe_bidirectional_attention_masks(
    attention_mask: Any, sliding_window: int = EXPECTED_EFFECTIVE_SLIDING_WINDOW
) -> dict[str, Any]:
    """Build Gemma3's exact full/local masks without empty query rows.

    ``True`` means that the key participates in PyTorch SDPA.  For every row
    that already has an admitted real key, these masks are identical to the
    pinned bidirectional Gemma3 mask construction.  A padded query farther
    than the local window from every real key would otherwise have no admitted
    key at all; only such a row receives its own diagonal bit.  Padding keys
    remain invisible to every real-token query.
    """
    import torch

    sequence_length = attention_mask.shape[1]
    positions = torch.arange(sequence_length, device=attention_mask.device)
    offsets = positions.unsqueeze(1) - positions.unsqueeze(0)
    diagonal = (offsets == 0).unsqueeze(0).unsqueeze(0)
    local_window = (offsets.abs() < sliding_window).unsqueeze(0).unsqueeze(0)
    real_keys = attention_mask.bool().unsqueeze(1).unsqueeze(1)

    # Materialize the query axis while retaining broadcast over the frozen
    # batch dimension.  The pinned full-attention layers admit every real key;
    # the sliding layers admit real keys at exclusive distance < 257.
    full_attention = torch.ones_like(local_window) & real_keys
    sliding_attention = local_window & real_keys
    return {
        "full_attention": _ensure_nonempty_attention_rows(
            full_attention, diagonal
        ),
        "sliding_attention": _ensure_nonempty_attention_rows(
            sliding_attention, diagonal
        ),
    }


def build_torch_pipeline(source_dir: Path):
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        str(source_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        # Transformers 5.2 defaults to SDPA when available. Freeze it instead
        # of allowing dependency or host capability to select another graph.
        attn_implementation="sdpa",
    )
    if (
        backbone.config.use_bidirectional_attention is not True
        or backbone.config.sliding_window != EXPECTED_EFFECTIVE_SLIDING_WINDOW
    ):
        raise ConversionError(
            "loaded Gemma3 config does not have the frozen bidirectional radius 257"
        )
    dense_2 = _load_dense_weight(
        source_dir / "2_Dense/model.safetensors", (3072, 768)
    ).to(dtype=torch.float32)
    dense_3 = _load_dense_weight(
        source_dir / "3_Dense/model.safetensors", (768, 3072)
    ).to(dtype=torch.float32)

    class FrozenEmbeddingGemmaPipeline(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.register_buffer("dense_2_weight", dense_2)
            self.register_buffer("dense_3_weight", dense_3)

        def forward(self, input_ids, attention_mask):
            backbone_attention = safe_bidirectional_attention_masks(
                attention_mask
            )
            token_embeddings = self.backbone(
                input_ids=input_ids,
                attention_mask=backbone_attention,
                use_cache=False,
                return_dict=False,
            )[0]
            pooled = masked_mean(token_embeddings, attention_mask)
            # Both upstream Dense modules use Identity activation and no bias.
            projected = functional.linear(
                pooled.to(torch.float32), self.dense_2_weight
            )
            projected = functional.linear(projected, self.dense_3_weight)
            # The upstream final sentence-transformers module is Normalize.
            return functional.normalize(projected, p=2.0, dim=1)

    pipeline = FrozenEmbeddingGemmaPipeline().eval()
    for parameter in pipeline.parameters():
        parameter.requires_grad_(False)
    return pipeline


def _set_tensor_name(port: Any, name: str) -> None:
    port.get_tensor().set_names({name})


def export_torch_pipeline(pipeline: Any, example_ids: Any, example_mask: Any) -> Any:
    """Export both inputs with one shared, bounded sequence dimension.

    OpenVINO's convenience ``dynamo=True`` path currently derives a distinct
    ``torch.export.Dim`` for each input shape. EmbeddingGemma requires the
    sequence axes of ``input_ids`` and ``attention_mask`` to be equal, so the
    shared symbol must be expressed explicitly before OpenVINO conversion.
    """
    import torch

    sequence = torch.export.Dim("sequence", min=1, max=MAX_TOKENS)
    return torch.export.export(
        pipeline,
        (example_ids, example_mask),
        dynamic_shapes=({1: sequence}, {1: sequence}),
        # Match the pinned OpenVINO release's supported non-strict
        # ExportedProgram path;
        # the explicit shared Dim still carries the equality and bounds.
        strict=False,
    )


# Exact rotary nodes from requirements-build.lock: torch 2.13.0+cpu,
# transformers 5.10.1, and OpenVINO 2026.2.1. Names and structural checks
# jointly select the two inspected outer products; shape alone is insufficient.
DEGENERATE_OUTER_PRODUCT_NODES = frozenset(
    {
        "bmm/aten.bmm.default/MatMul",
        "bmm_1/aten.bmm.default/MatMul",
    }
)


def rewrite_unit_reduction_matmuls(model: Any) -> None:
    """Replace the two K=1 rotary outer products with exact multiplication.

    A matrix product whose reduction dimension is exactly one performs one
    multiplication and no addition.  Broadcasting ``Multiply`` is therefore
    algebraically identical here, while avoiding an Intel GPU plugin shape
    lowering bug for ``[1,128,1] x [1,1,S]``.  Refuse graph drift rather than
    rewriting any additional MatMul selected only by shape.  The converted
    OpenVINO graph does not retain the input sequence bounds on this internal
    position-id edge, so only the dimensions relevant to the K=1 identity are
    frozen here.  ``convert_graph`` separately verifies the bounded model
    inputs before calling this rewrite.
    """

    import openvino.opset13 as ops

    rewritten: set[str] = set()
    for node in list(model.get_ordered_ops()):
        if node.get_type_name() != "MatMul":
            continue
        name = node.get_friendly_name()
        if name not in DEGENERATE_OUTER_PRODUCT_NODES:
            continue
        if node.get_attributes() != {
            "transpose_a": False,
            "transpose_b": False,
        }:
            raise ConversionError(f"{name} is not the frozen non-transposed MatMul")
        left = node.input_value(0)
        right = node.input_value(1)
        left_shape = left.get_partial_shape()
        right_shape = right.get_partial_shape()
        expected_shape = node.output(0).get_partial_shape()
        if (
            not left_shape.rank.is_static
            or left_shape.rank.get_length() != 3
            or not right_shape.rank.is_static
            or right_shape.rank.get_length() != 3
            or not left_shape[0].is_static
            or left_shape[0].get_length() != 1
            or not left_shape[1].is_static
            or left_shape[1].get_length() != 128
            or not left_shape[2].is_static
            or left_shape[2].get_length() != 1
            or not right_shape[0].is_static
            or right_shape[0].get_length() != 1
            or not right_shape[1].is_static
            or right_shape[1].get_length() != 1
        ):
            raise ConversionError(
                f"{name} does not have the frozen [1,128,1] x "
                "[1,1,S] unit-reduction contract"
            )
        replacement = ops.multiply(left, right)
        replacement.set_friendly_name(name)
        if (
            replacement.output(0).get_partial_shape() != expected_shape
            or replacement.output(0).get_element_type()
            != node.output(0).get_element_type()
        ):
            raise ConversionError(f"{name} multiplication rewrite changed its contract")
        node.output(0).replace(replacement.output(0))
        rewritten.add(name)
    if rewritten != DEGENERATE_OUTER_PRODUCT_NODES:
        missing = sorted(DEGENERATE_OUTER_PRODUCT_NODES - rewritten)
        raise ConversionError(
            "converted graph lacks the two frozen unit-reduction MatMul nodes: "
            + ", ".join(missing)
        )
    model.validate_nodes_and_infer_types()


def convert_graph(source_dir: Path, output_dir: Path, weight_storage: str) -> None:
    import openvino as ov
    import torch

    pipeline = build_torch_pipeline(source_dir)
    example_ids = torch.zeros((1, SEQUENCE_BUCKETS[0]), dtype=torch.int64)
    example_mask = torch.ones((1, SEQUENCE_BUCKETS[0]), dtype=torch.int64)
    dynamic_ids = ov.PartialShape([1, ov.Dimension(1, MAX_TOKENS)])
    dynamic_mask = ov.PartialShape([1, ov.Dimension(1, MAX_TOKENS)])
    with torch.no_grad():
        # A static torch.jit trace at length 32 cannot prove that the
        # 2,048-token graph is the same semantic pipeline. Export the bounded
        # shared dimension ourselves because OpenVINO's dynamo convenience
        # path otherwise assigns independent symbols to the two inputs.
        exported = export_torch_pipeline(pipeline, example_ids, example_mask)
        # The OpenVINO ExportedProgram decoder intentionally generalizes
        # symbolic inputs to fully dynamic shapes. Reapply the frozen batch
        # and sequence bounds during IR conversion; this does not re-export
        # the PyTorch program or split its shared sequence constraint.
        model = ov.convert_model(
            exported,
            input=[
                ("input_ids", dynamic_ids, ov.Type.i64),
                ("attention_mask", dynamic_mask, ov.Type.i64),
            ],
        )
    if len(model.inputs) != 2 or len(model.outputs) != 1:
        raise ConversionError(
            f"converted graph has {len(model.inputs)} inputs and "
            f"{len(model.outputs)} outputs; expected 2 and 1"
        )
    _set_tensor_name(model.inputs[0], "input_ids")
    _set_tensor_name(model.inputs[1], "attention_mask")
    _set_tensor_name(model.outputs[0], "embedding")
    for port, expected_shape, name in (
        (model.input("input_ids"), dynamic_ids, "input_ids"),
        (model.input("attention_mask"), dynamic_mask, "attention_mask"),
    ):
        shape = port.partial_shape
        bounded_shape = (
            shape.rank.is_static
            and shape.rank.get_length() == 2
            and shape[0].is_static
            and shape[0].get_length() == 1
            and shape[1].get_min_length() == 1
            and shape[1].get_max_length() == MAX_TOKENS
        )
        if not bounded_shape or port.element_type != ov.Type.i64:
            raise ConversionError(
                f"converted {name} contract is {port.element_type} "
                f"{port.partial_shape}, expected i64 {expected_shape}"
            )
    rewrite_unit_reduction_matmuls(model)
    # Before serialization, prove that every required static compilation shape
    # can be derived from this exact graph.  Physical device compilation and
    # placement are deliberately later evidence, not a conversion claim.
    for bucket in SEQUENCE_BUCKETS:
        candidate = model.clone()
        candidate.reshape(
            {"input_ids": [1, bucket], "attention_mask": [1, bucket]}
        )
        output_shape = candidate.output("embedding").partial_shape
        if not output_shape.is_static or list(output_shape.to_shape()) != [1, DIMENSIONS]:
            raise ConversionError(
                f"bucket {bucket} produces {output_shape}, expected [1,{DIMENSIONS}]"
            )
    ov.save_model(
        model,
        output_dir / "embeddinggemma.xml",
        compress_to_fp16=weight_storage == "f16",
    )


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise ConversionError(
            f"conversion dependency {distribution} is not installed"
        ) from error


def write_artifact_manifest(
    source_dir: Path,
    output_dir: Path,
    weight_storage: str,
) -> tuple[Path, str]:
    graph_xml = output_dir / "embeddinggemma.xml"
    graph_bin = output_dir / "embeddinggemma.bin"
    tokenizer_json = output_dir / "tokenizer.json"
    artifact_files = (graph_xml, graph_bin, tokenizer_json) + tuple(
        output_dir / name for name in LEGAL_FILES
    )
    for path in artifact_files:
        if not path.is_file() or path.stat().st_size < 1:
            raise ConversionError(f"conversion did not produce {path.name}")
    files = []
    for path in artifact_files:
        files.append(
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": 1,
        "artifact_format": "openvino-ir-dynamic-sequence-static-buckets-v1",
        "source": {
            "model": MODEL,
            "revision": MODEL_REVISION,
            "acquisition": {
                "repository": SOURCE_MIRROR,
                "revision": SOURCE_MIRROR_REVISION,
                "mode": "public-byte-identical-mirror",
            },
            "files": dict(PINNED_SOURCE_FILE_SHA256),
        },
        "semantic_pipeline": {
            "dimensions": DIMENSIONS,
            "pooling": "attention-mask-weighted-mean-include-prompt",
            "dense_2": "linear-768x3072-identity",
            "dense_3": "linear-3072x768-identity",
            "normalization": "l2",
            "truncation": "disabled",
            "padding": "right-attention-mask-excludes-padding",
        },
        "sequence_buckets": list(SEQUENCE_BUCKETS),
        "graph": {
            "xml": graph_xml.name,
            "bin": graph_bin.name,
            "input_ids": "input_ids",
            "attention_mask": "attention_mask",
            "output": "embedding",
        },
        "tokenizer": {
            "json": tokenizer_json.name,
            "sha256": PINNED_SOURCE_FILE_SHA256["tokenizer.json"],
            "pad_token": "<pad>",
            "pad_token_id": 0,
            "bos_token": "<bos>",
            "bos_token_id": 2,
            "eos_token": "<eos>",
            "eos_token_id": 1,
            "add_bos_token": True,
            "add_eos_token": True,
        },
        "legal": {
            "terms_url": "https://ai.google.dev/gemma/terms",
            "prohibited_use_policy_url": "https://ai.google.dev/gemma/prohibited_use_policy",
            "terms_file": "GEMMA_TERMS.txt",
            "prohibited_use_policy_file": "GEMMA_PROHIBITED_USE_POLICY.txt",
            "use_restrictions_file": "MODEL_USE_RESTRICTIONS.txt",
            "modifications_file": "MODEL_MODIFICATIONS.txt",
            "notice_file": "NOTICE",
        },
        "files": files,
        "conversion": {
            "recipe": "packages/openvino/convert.py",
            "export": (
                "torch-export-bounded-dynamic-sequence-1-to-2048-"
                "unit-reduction-matmul-rewrite-v1"
            ),
            "weight_storage": weight_storage,
            "openvino": _version("openvino"),
            "safetensors": _version("safetensors"),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
        },
    }
    raw = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    path = output_dir / "artifact-manifest.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def convert(
    source_dir: Path,
    legal_dir: Path,
    output_dir: Path,
    weight_storage: str,
) -> tuple[Path, str]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.is_dir():
        raise ConversionError(f"source directory does not exist: {source_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ConversionError(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_source_files(source_dir)
    validate_semantic_source(source_dir)
    convert_graph(source_dir, output_dir, weight_storage)
    shutil.copyfile(source_dir / "tokenizer.json", output_dir / "tokenizer.json")
    copy_legal_payload(legal_dir.resolve(), output_dir)
    return write_artifact_manifest(source_dir, output_dir, weight_storage)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-dir", required=True, type=Path)
    result.add_argument("--legal-dir", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument(
        "--weight-storage",
        choices=("f16", "f32"),
        default="f16",
        help="IR constant storage; execution precision remains target-native and must be admitted",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest_path, digest = convert(
            args.source_dir, args.legal_dir, args.output_dir, args.weight_storage
        )
    except (ConversionError, LegalError, OSError, RuntimeError) as error:
        print(f"OpenVINO conversion refused: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_manifest": str(manifest_path),
                "artifact_sha256": digest,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
