#!/usr/bin/env python3
"""Smoke-test converted IR against an independent, unmodified upstream model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

PROFILE_TOOLS = (
    Path(__file__).resolve().parents[2] / "experiments" / "embedding-profile"
)
if str(PROFILE_TOOLS) not in sys.path:
    sys.path.insert(0, str(PROFILE_TOOLS))

from admission_evidence import (  # noqa: E402
    SEQUENCE_BUCKETS as SEMANTIC_SEQUENCE_BUCKETS,
    sequence_semantic_probe_inputs,
)

if __package__:
    from . import reference as upstream_reference
    from .convert import ConversionError
    from .manifest import DIMENSIONS, MODEL, MODEL_REVISION, PINNED_SOURCE_FILE_SHA256, load_artifact
else:
    import reference as upstream_reference  # type: ignore[no-redef]
    from convert import ConversionError  # type: ignore[no-redef]
    from manifest import (  # type: ignore[no-redef]
        DIMENSIONS, MODEL, MODEL_REVISION, PINNED_SOURCE_FILE_SHA256, load_artifact,
    )


CASES = (
    (
        "short-query",
        32,
        "task: search result | query: portable local semantic search",
    ),
    (
        "paragraph-document",
        128,
        "title: Shared vector space\ntext: Devices exchange signed INT8 embeddings "
        "while each admitted accelerator executes its own exact packaged runtime.",
    ),
    (
        "maximum-static-shape",
        2048,
        "task: search result | query: exercise the maximum static sequence bucket",
    ),
)
LONG_TOKEN_CASES = (
    ("window-boundary-258", 512, 258),
    ("long-document-1946", 2048, 1946),
)
LONG_TOKEN_SEEDS = (
    "Production changes require current approval and a tested recovery plan.",
    "A violin melody follows the rhythm while musicians listen to the harmony.",
    "Distributed databases retain snapshots and reconcile independently written records.",
    "Mountain paths cross forests and streams before reaching the summit.",
)
MINIMUM_COSINE = 0.999
MAXIMUM_NORM_ERROR = 0.005


class ParityError(ValueError):
    """The converted graph failed a deterministic CPU semantic smoke test."""


def validate_pair(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    label: str = "parity case",
) -> tuple[float, float, float]:
    if len(reference) != DIMENSIONS or len(candidate) != DIMENSIONS:
        raise ParityError(
            f"{label}: parity vectors must both contain {DIMENSIONS} components"
        )
    reference_nonfinite = [
        index for index, value in enumerate(reference) if not math.isfinite(value)
    ]
    candidate_nonfinite = [
        index for index, value in enumerate(candidate) if not math.isfinite(value)
    ]
    if reference_nonfinite or candidate_nonfinite:
        raise ParityError(
            f"{label}: parity vectors contain non-finite components: "
            f"PyTorch={len(reference_nonfinite)} "
            f"(first indices {reference_nonfinite[:8]}), "
            f"OpenVINO={len(candidate_nonfinite)} "
            f"(first indices {candidate_nonfinite[:8]})"
        )
    reference_norm = math.sqrt(sum(value * value for value in reference))
    candidate_norm = math.sqrt(sum(value * value for value in candidate))
    if (
        abs(reference_norm - 1.0) > MAXIMUM_NORM_ERROR
        or abs(candidate_norm - 1.0) > MAXIMUM_NORM_ERROR
    ):
        raise ParityError(
            f"{label}: PyTorch and OpenVINO outputs must both be L2 normalized "
            f"(PyTorch={reference_norm:.9f}, OpenVINO={candidate_norm:.9f})"
        )
    cosine = sum(left * right for left, right in zip(reference, candidate)) / (
        reference_norm * candidate_norm
    )
    if not math.isfinite(cosine) or cosine < MINIMUM_COSINE:
        raise ParityError(
            f"{label}: OpenVINO/PyTorch cosine {cosine:.9f} is below "
            f"{MINIMUM_COSINE}"
        )
    return reference_norm, candidate_norm, cosine


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    pieces = tokenizer.encode(text, add_special_tokens=False).ids
    # Frozen tokenizer contract: BOS=2, EOS=1, PAD=0.
    return [2, *pieces, 1]


def long_token_fixtures(tokenizer: Any) -> list[tuple[str, int, list[int]]]:
    """Build exact-length prepared-token probes; never truncate input text.

    Topic changes every 97 tokens exercise local attention across different
    neighborhoods. All interior IDs come from the pinned tokenizer's prose,
    with explicit BOS=2/EOS=1; these are structural parity probes, not qrels.
    """
    banks = [tokenizer.encode(text, add_special_tokens=False).ids for text in LONG_TOKEN_SEEDS]
    if any(
        not bank or any(type(token) is not int or token <= 2 or tokenizer.id_to_token(token) is None for token in bank)
        for bank in banks
    ):
        raise ParityError("long parity fixture requires valid non-special prose token IDs")
    results = []
    for label, bucket, token_count in LONG_TOKEN_CASES:
        interior = []
        for position in range(token_count - 2):
            bank = banks[(position // 97) % len(banks)]
            interior.append(bank[position % len(bank)])
        results.append((label, bucket, [2, *interior, 1]))
    return results


def validate_sequence_semantic_fixture(tokenizer: Any) -> list[dict[str, Any]]:
    """Prove the policy's three texts reach every exact static bucket."""

    results: list[dict[str, Any]] = []
    for bucket in SEMANTIC_SEQUENCE_BUCKETS:
        counts = [
            len(_token_ids(tokenizer, text))
            for text in sequence_semantic_probe_inputs(bucket)
        ]
        if counts != [bucket, bucket, bucket]:
            raise ParityError(
                f"sequence semantic fixture bucket {bucket} tokenized to {counts}, "
                f"expected three exact {bucket}-token inputs"
            )
        results.append({"bucket": bucket, "token_counts": counts})
    return results


def run(source_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    import numpy as np
    import openvino as ov
    import torch
    import transformers
    from tokenizers import Tokenizer

    source_dir = source_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "artifact-manifest.json"
    manifest_raw = manifest_path.read_bytes()
    artifact = load_artifact(
        artifact_dir,
        "artifact-manifest.json",
        hashlib.sha256(manifest_raw).hexdigest(),
    )
    tokenizer = Tokenizer.from_file(str(artifact.tokenizer_json))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    semantic_fixture = validate_sequence_semantic_fixture(tokenizer)
    pipeline = upstream_reference.build_upstream(source_dir)
    core = ov.Core()
    if "CPU" not in core.available_devices:
        raise ParityError("OpenVINO CPU plugin is unavailable for conversion smoke parity")
    graph = core.read_model(str(artifact.graph_xml), str(artifact.graph_bin))
    results: list[dict[str, Any]] = []
    cases = [(label, bucket, _token_ids(tokenizer, text), "frozen-text") for label, bucket, text in CASES]
    cases.extend((label, bucket, ids, "prepared-token-probe") for label, bucket, ids in long_token_fixtures(tokenizer))
    for label, bucket, ids, input_kind in cases:
        if not 1 <= len(ids) <= bucket:
            raise ParityError(
                f"frozen smoke text {label} produced {len(ids)} tokens for bucket {bucket}"
            )
        input_ids_sha256 = hashlib.sha256(
            np.asarray(ids, dtype="<i8").tobytes(order="C")
        ).hexdigest()
        mask = [1] * len(ids) + [0] * (bucket - len(ids))
        ids += [0] * (bucket - len(ids))
        ids_array = np.asarray([ids], dtype=np.int64)
        mask_array = np.asarray([mask], dtype=np.int64)
        with torch.no_grad():
            reference_array = pipeline(
                torch.from_numpy(ids_array), torch.from_numpy(mask_array)
            ).detach().cpu().to(torch.float32).numpy()
        static_graph = graph.clone()
        static_graph.reshape(
            {
                artifact.input_ids_name: [1, bucket],
                artifact.attention_mask_name: [1, bucket],
            }
        )
        compiled = core.compile_model(static_graph, "CPU")
        output = compiled(
            {
                artifact.input_ids_name: ids_array,
                artifact.attention_mask_name: mask_array,
            }
        )
        candidate_array = np.asarray(
            output[compiled.output(artifact.output_name)], dtype=np.float32
        )
        if reference_array.shape != (1, DIMENSIONS) or candidate_array.shape != (
            1,
            DIMENSIONS,
        ):
            raise ParityError(
                f"smoke case {label} returned {reference_array.shape} and "
                f"{candidate_array.shape}, expected (1,{DIMENSIONS})"
            )
        reference = reference_array[0].tolist()
        candidate = candidate_array[0].tolist()
        reference_norm, candidate_norm, cosine = validate_pair(
            reference, candidate, label=label
        )
        results.append(
            {
                "label": label,
                "bucket": bucket,
                "token_count": sum(mask),
                "input_kind": input_kind,
                "input_ids_sha256": input_ids_sha256,
                "reference_l2": reference_norm,
                "openvino_l2": candidate_norm,
                "cosine": cosine,
                "reference_f32_sha256": hashlib.sha256(
                    reference_array.astype("<f4", copy=False).tobytes(order="C")
                ).hexdigest(),
                "openvino_f32_sha256": hashlib.sha256(
                    candidate_array.astype("<f4", copy=False).tobytes(order="C")
                ).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "purpose": "conversion-smoke-not-admission-evidence",
        "device": "CPU",
        "minimum_cosine": MINIMUM_COSINE,
        "artifact_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "smoke_recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference": {
            "implementation": upstream_reference.DESCRIPTION,
            "implementation_sha256": hashlib.sha256(Path(upstream_reference.__file__).read_bytes()).hexdigest(),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "source_files_sha256": dict(PINNED_SOURCE_FILE_SHA256),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "sequence_semantic_fixture": semantic_fixture,
        "cases": results,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-dir", required=True, type=Path)
    result.add_argument("--artifact-dir", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = run(args.source_dir, args.artifact_dir)
        raw = (
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if args.output.exists():
            raise ParityError(f"refusing to overwrite parity report {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
    except (ConversionError, OSError, ParityError, RuntimeError) as error:
        print(f"OpenVINO conversion parity smoke refused: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": 1,
                "report": str(args.output),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
