"""A contained, content-verified view of a pinned Hugging Face snapshot.

ORT 1.24.1 rejects external weights symlinked outside the model directory.
Hard links preserve the existing cached bytes without disabling that check or
duplicating large weights. The native worker receives a complete cache view.
"""

import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile


def verify(path, expected):
    size = path.stat().st_size
    if size != expected["bytes"]:
        raise RuntimeError(f"cached file size mismatch: {path.name}")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            sha256.update(chunk)
            git_blob.update(chunk)
    actual = {"sha256": sha256.hexdigest(), "git_blob_sha1": git_blob.hexdigest()}
    if not any(key in expected for key in actual):
        raise RuntimeError("cached file has no expected content identity")
    if any(actual[key] != expected[key] for key in actual if key in expected):
        raise RuntimeError(f"cached file hash mismatch: {path.name}")
    return {"bytes": size, "sha256": actual["sha256"]}


def prepare(source_cache, destination, manifest):
    source_cache = source_cache.resolve(strict=True)
    repository, revision = manifest["repository"], manifest["revision"]
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("invalid pinned repository/revision")
    repo_dir = "models--" + repository.replace("/", "--")
    source_repo = source_cache / repo_dir
    source_blobs = source_repo / "blobs"
    source_blobs.mkdir(parents=True, exist_ok=True)
    if not source_blobs.resolve(strict=True).is_relative_to(source_cache):
        raise RuntimeError("source blob directory escapes the cache")
    destination.mkdir()  # An interrupted or completed view is never overwritten.
    snapshot = destination / repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    observed = {}
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise RuntimeError("unsafe snapshot file path")
        identity = expected.get("sha256", expected.get("git_blob_sha1", ""))
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", identity):
            raise RuntimeError("invalid cached content identity")
        blob = source_blobs / identity
        if not blob.exists():
            # A cache-layout repair must never redownload large model weights.
            if expected["bytes"] > 25 * 1024 * 1024:
                raise RuntimeError(f"required retained model file is absent: {name}")
            fd, temporary_name = tempfile.mkstemp(prefix="pinned-download-", dir=source_blobs)
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                subprocess.run([
                    "curl", "--fail", "--silent", "--show-error", "--location",
                    "--max-time", "60", "--max-filesize", str(expected["bytes"]),
                    "--output", str(temporary),
                    f"https://huggingface.co/{repository}/resolve/{revision}/{name}",
                ], check=True, timeout=65)
                verify(temporary, expected)
                try:
                    os.link(temporary, blob)
                except FileExistsError:
                    pass  # A competing writer's bytes are verified below.
            finally:
                temporary.unlink(missing_ok=True)
        resolved = blob.resolve(strict=True)
        if not resolved.is_relative_to(source_cache):
            raise RuntimeError("cached file escapes the source cache")
        observed[name] = verify(resolved, expected)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(resolved, target)  # Fail on EXDEV; do not silently duplicate weights.
    refs = destination / repo_dir / "refs"
    refs.mkdir()
    # hf-hub 0.5.0 uses the bytes verbatim, including any trailing newline.
    (refs / "main").write_text(revision)
    return {"repository": repository, "revision": revision, "files": observed}
