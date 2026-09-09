"""One explicit CPU probe using the already measured, retained native bundle.

This command never selects an accelerator or initializes the host governor.
It is intended for the existing CI worker, with bounded CPU/memory resources.
"""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from pinned_cache import prepare, verify


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def check_bundle(bundle, expected_manifest):
    if digest(bundle / "MANIFEST.json") != expected_manifest:
        raise RuntimeError("bundle does not match the measured manifest")
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = bundle / name
        if not path.resolve(strict=True).is_relative_to(bundle):
            raise RuntimeError(f"bundle path escapes its root: {name}")
        if "symlink" in expected:
            if not path.is_symlink() or os.readlink(path) != expected["symlink"]:
                raise RuntimeError(f"bundle symlink changed: {name}")
        elif path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise RuntimeError(f"bundle file changed: {name}")


def main():
    bundle = Path(os.environ["CFETCH_FOUNDATION_BUNDLE"]).resolve(strict=True)
    output = Path(os.environ["CFETCH_FOUNDATION_OUTPUT"])
    if not output.is_absolute():
        raise RuntimeError("CFETCH_FOUNDATION_OUTPUT must be an absolute fresh directory")
    model = os.environ.get("CFETCH_FOUNDATION_MODEL", "")
    if model not in ("", "AllMiniLML6V2", "EmbeddingGemma300M"):
        raise RuntimeError("choose loader-only (empty), AllMiniLML6V2 or EmbeddingGemma300M")
    evidence = json.loads(Path(__file__).with_name("evidence.json").read_text())
    check_bundle(bundle, evidence["bundle_manifest_sha256"])
    probe = bundle / "cfetch-npu-ort-foundation"
    probe_directory = os.environ.get("CFETCH_FOUNDATION_PROBE_DIR", "")
    diagnostic = os.environ.get("CFETCH_FOUNDATION_CPU_DIAGNOSTIC", "")
    if diagnostic not in ("", "1") or (diagnostic and (not probe_directory or not model)):
        raise RuntimeError("CPU diagnostic requires an explicit newly built probe and model")
    build_identity = None
    if probe_directory:
        probe_directory = Path(probe_directory).resolve(strict=True)
        build_identity = json.loads((probe_directory / "build-identity.json").read_text())
        if build_identity["bundle_manifest_sha256"] != evidence["bundle_manifest_sha256"]:
            raise RuntimeError("new probe was built for a different retained runtime")
        source = Path(__file__).parent
        for name in ("Cargo.toml", "Cargo.lock", "fastembed-session.patch", "src/main.rs"):
            if build_identity["source_sha256"][name] != digest(source / name):
                raise RuntimeError(f"probe source differs from this checkout: {name}")
        probe = probe_directory / "cfetch-npu-ort-foundation"
        if digest(probe) != build_identity["binary_sha256"]:
            raise RuntimeError("new probe binary differs from its build receipt")
    output.mkdir()
    if build_identity is not None:
        save(output / "build-identity.json", build_identity)
    cache = bundle / "model-cache"
    if model == "EmbeddingGemma300M":
        cache_manifest = json.loads(Path(__file__).with_name("embeddinggemma-cache.json").read_text())
        cache = output / "cache"
        observed = prepare(bundle / "model-cache", cache, cache_manifest)
        save(output / "model-identity.json", observed)
    # The independent timer also survives abrupt loss of this Python parent.
    command = ["timeout", "--signal=KILL", "300", str(probe), "--ort",
               str(bundle / "lib/libonnxruntime.so.1.24.1")]
    if model:
        command += ["--run-model", model, "--device", "CPU", "--cache",
                    str(cache), "--output", str(output / "model")]
    if diagnostic:
        command += ["--cpu-fallback", "diagnostic"]
    probe_sha256 = digest(probe)
    # Match the explicit cache selection; HF_HOME otherwise overrides FastEmbed.
    environment = dict(os.environ)
    environment.pop("HF_HOME", None)
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["RAYON_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    # Preserve partition diagnostics even when initialization fails before a profile.
    environment["ORT_LOG"] = "verbose"
    auxiliary_library_directories = []
    if probe_directory and environment.get("NIX_LD_LIBRARY_PATH"):
        # Nix-linked binaries bypass nix-ld, which supplied these dependencies
        # for the original portable probe. Resolve its existing library view
        # once so a system-profile update cannot change this call's selection.
        auxiliary_library_directories = [
            str(Path(directory).resolve(strict=True))
            for directory in environment["NIX_LD_LIBRARY_PATH"].split(os.pathsep) if directory
        ]
        environment["LD_LIBRARY_PATH"] = os.pathsep.join([
            str(bundle / "lib"), *auxiliary_library_directories,
            *([environment["LD_LIBRARY_PATH"]] if environment.get("LD_LIBRARY_PATH") else []),
        ])
    if model == "EmbeddingGemma300M":
        # Cache misses must fail locally rather than fetching an unpinned main.
        environment["HF_ENDPOINT"] = "http://[cfetch-offline"
    save(output / "intent.json", {"model": model or None, "device": "CPU",
         "bundle_manifest_sha256": evidence["bundle_manifest_sha256"],
         "cpu_fallback_diagnostic": bool(diagnostic), "binary_sha256": probe_sha256,
         "auxiliary_library_directories": auxiliary_library_directories,
         "deadline_seconds": 300, "maximum_embedding_calls": int(bool(model))})
    result = {"model": model or None, "requested_device": "CPU", "passed": False,
              "cpu_fallback_diagnostic": bool(diagnostic), "accelerator_qualified": False}
    started = time.monotonic()
    def interrupted(signum, _frame):
        raise RuntimeError(f"supervisor interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    with (output / "stdout").open("xb") as stdout, (output / "stderr").open("xb") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment,
                                 start_new_session=True)
        try:
            result["returncode"] = child.wait(timeout=300)
            result["passed"] = result["returncode"] == 0
        except BaseException as error:
            result["error"] = repr(error)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                try:
                    result["returncode"] = child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    save(output / "unreaped.json", result)
                    os._exit(1)
        finally:
            result["worker_seconds_including_downloads"] = time.monotonic() - started
            try:
                check_bundle(bundle, evidence["bundle_manifest_sha256"])
                if digest(probe) != probe_sha256:
                    raise RuntimeError("probe binary changed during execution")
                if build_identity is not None:
                    for name, expected in build_identity["source_sha256"].items():
                        if digest(Path(__file__).parent / name) != expected:
                            raise RuntimeError("probe source changed during execution")
                result["runtime_identity_unchanged"] = True
            except Exception as error:
                result["passed"] = False
                result["runtime_identity_error"] = repr(error)
            if model == "EmbeddingGemma300M":
                try:
                    repo = cache / ("models--" + observed["repository"].replace("/", "--"))
                    if (repo / "refs/main").read_text() != observed["revision"]:
                        raise RuntimeError("model cache revision changed during execution")
                    snapshot = repo / "snapshots" / observed["revision"]
                    for name, expected in observed["files"].items():
                        path = snapshot / name
                        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(snapshot):
                            raise RuntimeError("model cache containment changed during execution")
                        verify(path, expected)
                    result["model_identity_unchanged"] = True
                except Exception as error:
                    result["passed"] = False
                    result["identity_error"] = repr(error)
            save(output / "result.json", result)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
