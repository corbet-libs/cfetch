"""Independent upstream oracle for conversion parity, never an exported graph."""

from __future__ import annotations

from pathlib import Path

if __package__:
    from .convert import ConversionError, validate_semantic_source, verify_source_files
else:
    from convert import (  # type: ignore[no-redef]
        ConversionError,
        validate_semantic_source,
        verify_source_files,
    )


DESCRIPTION = (
    "fresh AutoModel(sdpa) with ordinary 2D attention mask; "
    "independent masked mean + Dense2 + Dense3 + L2"
)


def build_upstream(source_dir: Path):
    """Load verified source without converter attention or pooling helpers."""
    source_dir = source_dir.resolve()
    verify_source_files(source_dir)
    validate_semantic_source(source_dir)

    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        str(source_dir), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="sdpa",
    ).cpu().eval()
    # Assert the upstream interpretation independently of the converter's
    # effective-window constant or mask implementation. Raw source pins 512;
    # Gemma3TextConfig turns bidirectional width into exclusive radius 257.
    if (
        backbone.config.use_bidirectional_attention is not True
        or backbone.config.sliding_window != 257
    ):
        raise ConversionError("upstream reference requires bidirectional radius 257")
    weights = []
    for directory, shape in (("2_Dense", (3072, 768)), ("3_Dense", (768, 3072))):
        tensors = load_file(str(source_dir / directory / "model.safetensors"), device="cpu")
        if set(tensors) != {"linear.weight"} or tuple(tensors["linear.weight"].shape) != shape:
            raise ConversionError("independent Dense tensor keys/shape mismatch")
        weights.append(tensors["linear.weight"].to(dtype=torch.float32))
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    def forward(ids, mask):
        hidden = backbone(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
        expanded = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.dtype)
        # Ordinary sentence-transformers mean semantics. In particular, do
        # not hide padded NaNs using the converter's torch.where helper.
        pooled = (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1e-9)
        projected = functional.linear(pooled.to(torch.float32), weights[0])
        projected = functional.linear(projected, weights[1])
        return functional.normalize(projected, p=2.0, dim=1)

    return forward
