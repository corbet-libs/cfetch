#!/usr/bin/env python3
"""Convert verified canonical sources to a candidate F32 GGUF, without inference.

Run with the conversion environment's Python. Source and fresh output directories
must be outside this repository. The output contains a public, path-free lineage
document and a private conversion log (upstream logs include local paths). Failed
conversion work is retained; no lineage document is published on failure. This
establishes conversion provenance, not tokenizer/numerical parity or admission.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import stat
import subprocess
import sys
from typing import Any


REVISION = "b95502ba9aa0eb73a2f4fc8878d7fbe6a847a0b9"
REPOSITORY = Path(__file__).resolve().parents[3]
ARTIFACT = "embeddinggemma-300m-f32.gguf"
REQUIREMENTS = "requirements/requirements-convert_hf_to_gguf.txt"
MAX_SOURCE_BYTES = 4 * 1024**3
MAX_CODE_BYTES = 16 * 1024**2
EXPECTED_FIELDS = {
    "general.architecture": "gemma-embedding",
    "general.file_type": 0,  # ALL_F32
    "gemma-embedding.embedding_length": 768,
    "gemma-embedding.block_count": 24,
    "gemma-embedding.context_length": 2048,
    "gemma-embedding.attention.sliding_window": 512,
    "gemma-embedding.pooling_type": 1,  # MEAN
    "gemma-embedding.dense_2_feat_in": 768,
    "gemma-embedding.dense_2_feat_out": 3072,
    "gemma-embedding.dense_3_feat_in": 3072,
    "gemma-embedding.dense_3_feat_out": 768,
    "tokenizer.ggml.model": "llama",
    "tokenizer.ggml.bos_token_id": 2,
    "tokenizer.ggml.eos_token_id": 1,
    "tokenizer.ggml.padding_token_id": 0,
    "tokenizer.ggml.add_bos_token": True,
    "tokenizer.ggml.add_eos_token": True,
    "tokenizer.ggml.add_space_prefix": False,
}


class ConversionError(ValueError):
    pass


def hash_file(path: Path, maximum: int, *, git_blob: bool = False) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
            raise ConversionError(f"{path.name} must be a bounded regular file")
        digest = hashlib.sha1() if git_blob else hashlib.sha256()
        if git_blob:
            digest.update(f"blob {before.st_size}\0".encode())
        size = 0
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > maximum:
                raise ConversionError(f"{path.name} grew beyond its size bound")
            digest.update(chunk)
        after = os.fstat(stream.fileno())
        current = path.lstat()
        if size != before.st_size or (before.st_size, before.st_mtime_ns) != (
                after.st_size, after.st_mtime_ns) or (before.st_dev, before.st_ino) != (
                current.st_dev, current.st_ino):
            raise ConversionError(f"{path.name} changed during hashing")
        return digest.hexdigest()


def git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True,
                            timeout=30, check=True)
    if len(result.stdout) > 8 * 1024**2:
        raise ConversionError("git inventory exceeds size bound")
    return result.stdout


def verify_converter(root: Path) -> dict[str, str]:
    if git(root, "rev-parse", "--show-toplevel").decode().strip() != str(root):
        raise ConversionError("llama-source must be the checkout root")
    if git(root, "rev-parse", "HEAD").decode().strip() != REVISION:
        raise ConversionError(f"llama.cpp must be exactly {REVISION}")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ConversionError("llama.cpp checkout must have no tracked or untracked changes")
    entries = git(root, "ls-tree", "-rz", "HEAD", "--", "convert_hf_to_gguf.py",
                  "conversion", "gguf-py", "requirements").split(b"\0")
    inventory = {}
    for entry in filter(None, entries):
        metadata, encoded_name = entry.split(b"\t", 1)
        mode, kind, blob = metadata.decode().split()
        name = encoded_name.decode("utf-8")
        if mode not in {"100644", "100755"} or kind != "blob":
            raise ConversionError("converter inventory contains a link or submodule")
        path = root / name
        # Compare actual working bytes to Git blobs, including assume-unchanged
        # files that ordinary git status can omit.
        if hash_file(path, MAX_CODE_BYTES, git_blob=True) != blob:
            raise ConversionError(f"pinned converter bytes differ: {name}")
        inventory[name] = hash_file(path, MAX_CODE_BYTES)
    required = {"convert_hf_to_gguf.py", "conversion/base.py", "conversion/gemma.py",
                "gguf-py/gguf/gguf_reader.py", REQUIREMENTS}
    if not required <= inventory.keys():
        raise ConversionError("pinned converter checkout is incomplete")
    return inventory


def check_dependencies(root: Path) -> dict[str, dict[str, str]]:
    try:
        from packaging.requirements import Requirement
    except ImportError as error:
        raise ConversionError("packaging is required to validate converter dependencies") from error
    pending, seen, records, errors = [root / REQUIREMENTS], set(), {}, []
    while pending:
        path = pending.pop().resolve()
        if not path.is_relative_to(root / "requirements") or path in seen:
            raise ConversionError("unexpected recursive converter requirement include")
        seen.add(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("-r "):
                pending.append(path.parent / line[3:].strip())
                continue
            if line == "--extra-index-url https://download.pytorch.org/whl/cpu":
                continue
            requirement = Requirement(line)
            if requirement.url or requirement.marker or requirement.extras:
                raise ConversionError("unexpected converter requirement form")
            try:
                installed = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                installed = "missing"
            records[requirement.name] = {"required": str(requirement.specifier), "installed": installed}
            if installed == "missing" or not requirement.specifier.contains(installed):
                errors.append(f"{requirement.name}{requirement.specifier}: installed {installed}")
    if errors:
        raise ConversionError("converter dependency mismatch before conversion: " + "; ".join(errors))
    for name in ("packaging", "safetensors", "huggingface-hub", "tokenizers"):
        try:
            records.setdefault(name, {"required": "transitive", "installed": importlib.metadata.version(name)})
        except importlib.metadata.PackageNotFoundError as error:
            raise ConversionError(f"converter dependency missing before conversion: {name}") from error
    return records


def validate_reader(reader: Any, f32_type: Any) -> dict[str, Any]:
    diagnostics, errors = {}, []
    for name, expected in EXPECTED_FIELDS.items():
        field = reader.get_field(name)
        actual = None if field is None else field.contents()
        diagnostics[name] = actual
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"{name}: expected {expected!r}, found {actual!r}")
    for name, expected in (("gemma-embedding.embedding_length_out", 768),
                           ("gemma-embedding.attention.causal", False)):
        field = reader.get_field(name)
        if field is not None:
            actual = field.contents()
            diagnostics[name] = actual
            if type(actual) is not type(expected) or actual != expected:
                errors.append(f"{name}: unexpected override {actual!r}")
    tokens = reader.get_field("tokenizer.ggml.tokens")
    for index, expected in ((0, "<pad>"), (1, "<eos>"), (2, "<bos>")):
        actual = tokens.contents(index) if tokens is not None else None
        diagnostics[f"tokenizer.ggml.tokens[{index}]"] = actual
        if actual != expected:
            errors.append(f"token {index}: expected {expected!r}, found {actual!r}")
    tensors = {}
    for tensor in reader.tensors:
        if tensor.name in tensors:
            errors.append(f"duplicate tensor {tensor.name}")
        tensors[tensor.name] = {"shape": [int(value) for value in tensor.shape],
                                "type": tensor.tensor_type.name}
        if tensor.tensor_type != f32_type:
            errors.append(f"non-F32 tensor {tensor.name}")
    for name, shape in (("dense_2.weight", [768, 3072]), ("dense_3.weight", [3072, 768])):
        if name not in tensors or tensors[name]["shape"] != shape:
            errors.append(f"{name}: expected GGUF dimensions {shape}, found {tensors.get(name)!r}")
    result = {"fields": diagnostics, "tensors": tensors, "errors": errors}
    return result


def inspect_gguf(root: Path, artifact: Path) -> dict[str, Any]:
    local = root / "gguf-py"
    sys.path.insert(0, str(local))
    gguf = importlib.import_module("gguf")
    if not Path(gguf.__file__).resolve().is_relative_to(local):
        raise ConversionError("GGUF reader was imported from outside the pinned checkout")
    reader = gguf.GGUFReader(artifact, "r")
    return validate_reader(reader, gguf.GGMLQuantizationType.F32)


def write_json(path: Path, document: object):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def convert(source: Path, llama_source: Path, output: Path, timeout_seconds: float) -> dict[str, Any]:
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 14400:
        raise ConversionError("timeout must be finite and in (0, 14400] seconds")
    if os.path.lexists(output):
        raise ConversionError("output directory must be fresh")
    source, llama_source = source.resolve(strict=True), llama_source.resolve(strict=True)
    output = output.resolve()
    if source.is_relative_to(REPOSITORY) or output.is_relative_to(REPOSITORY):
        raise ConversionError("source and outputs must be outside the cfetch repository")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ConversionError("source and output directories must be separate")
    if output.is_relative_to(llama_source):
        raise ConversionError("output must be outside the clean llama.cpp checkout")
    if os.path.lexists(output):
        raise ConversionError("output directory must be fresh")
    sys.path.insert(0, str(REPOSITORY))
    from packages.openvino.convert import verify_source_files, validate_semantic_source
    from packages.openvino.manifest import MODEL, MODEL_REVISION, PINNED_SOURCE_FILE_SHA256
    def verify_inputs():
        actual = set()
        for directory, folders, files in os.walk(source, followlinks=False):
            for folder in folders:
                relative = (Path(directory) / folder).relative_to(source).as_posix()
                if relative not in {"1_Pooling", "2_Dense", "3_Dense"}:
                    raise ConversionError("canonical source tree contains an unexpected directory")
            for name in folders + files:
                if (Path(directory) / name).is_symlink():
                    raise ConversionError("canonical source tree must not contain symlinks")
            actual.update((Path(directory) / name).relative_to(source).as_posix() for name in files)
            if len(actual) > len(PINNED_SOURCE_FILE_SHA256):
                raise ConversionError("canonical source tree contains extra files")
        if actual != set(PINNED_SOURCE_FILE_SHA256):
            raise ConversionError("source tree must contain exactly the 13 pinned source files")
        # Check types before the existing verifier opens files (in particular FIFOs).
        for name in actual:
            info = (source / name).lstat()
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_SOURCE_BYTES:
                raise ConversionError(f"invalid canonical source file: {name}")
        verify_source_files(source)
        validate_semantic_source(source)
    verify_inputs()
    code = verify_converter(llama_source)
    dependencies = check_dependencies(llama_source)
    wrapper = hash_file(Path(__file__).resolve(), MAX_CODE_BYTES)
    validators = {name: hash_file(REPOSITORY / name, MAX_CODE_BYTES) for name in (
        "packages/openvino/convert.py", "packages/openvino/manifest.py", "packages/openvino/legal.py")}
    output.mkdir(mode=0o700)  # Atomic fresh-directory claim; preserve it on failure.
    artifact = output / ARTIFACT
    suffix = ["--outtype", "f32", "--sentence-transformers-dense-modules",
              "--model-name", "embeddinggemma-300m", "--outfile"]
    command = [sys.executable, "-E", "-s", "-B", str(llama_source / "convert_hf_to_gguf.py"),
               str(source), *suffix, str(artifact)]
    environment = dict(os.environ)
    environment.pop("NO_LOCAL_GGUF", None)
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                        "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""})
    with (output / "conversion.private.log").open("xb") as log:
        child = subprocess.Popen(command, cwd=llama_source, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = child.wait(timeout=timeout_seconds)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
            raise
        log.flush()
        os.fsync(log.fileno())
    if status != 0:
        raise ConversionError(f"converter exited {status}; private log and partial output retained")
    verify_inputs()
    if code != verify_converter(llama_source) or dependencies != check_dependencies(llama_source):
        raise ConversionError("converter sources or dependencies changed during conversion")
    if wrapper != hash_file(Path(__file__).resolve(), MAX_CODE_BYTES) or any(
            digest != hash_file(REPOSITORY / name, MAX_CODE_BYTES) for name, digest in validators.items()):
        raise ConversionError("wrapper or source validators changed during conversion")
    digest = hash_file(artifact, MAX_SOURCE_BYTES)
    diagnostics = inspect_gguf(llama_source, artifact)
    write_json(output / "gguf-diagnostics.json", diagnostics)
    if diagnostics["errors"]:
        raise ConversionError("GGUF validation failed: " + "; ".join(diagnostics["errors"]))
    if digest != hash_file(artifact, MAX_SOURCE_BYTES):
        raise ConversionError("GGUF artifact changed during validation")
    result = {
        "schema_version": 1, "kind": "cfetch-canonical-gguf-candidate-lineage-v1",
        "admission_status": "not_evaluated", "model": MODEL, "model_revision": MODEL_REVISION,
        "source_files_sha256": dict(PINNED_SOURCE_FILE_SHA256),
        "artifact": {"file": ARTIFACT, "sha256": digest, "bytes": artifact.stat().st_size, "format": "F32 GGUF"},
        "converter": {"repository": "https://github.com/ggml-org/llama.cpp", "revision": REVISION,
                      "files_sha256": code, "wrapper_sha256": wrapper, "validators_sha256": validators},
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version(),
                   "executable_sha256": hash_file(Path(sys.executable).resolve(), MAX_SOURCE_BYTES)},
        "dependencies": dependencies,
        "invocation": ["<python>", "-E", "-s", "-B", "<llama-source>/convert_hf_to_gguf.py",
                       "<canonical-source>", *suffix, f"<output>/{ARTIFACT}"],
        "environment": {name: environment[name] for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                        "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "CUDA_VISIBLE_DEVICES",
                        "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
        "validation": diagnostics,
        "remaining_validation": ["tokenizer ID parity", "long-input numeric parity", "physical runtime evidence"],
    }
    with artifact.open("rb") as stream:
        os.fsync(stream.fileno())
    write_json(output / "lineage.json", result)
    for directory in (output, output.parent):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    args = parser.parse_args()
    try:
        result = convert(args.source_dir, args.llama_source, args.output_directory, args.timeout_seconds)
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as error:
        print(f"canonical GGUF conversion refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"artifact_sha256": result["artifact"]["sha256"], "admission_status": "not_evaluated"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
