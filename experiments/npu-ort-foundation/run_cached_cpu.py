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


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    bundle = Path(os.environ["CFETCH_FOUNDATION_BUNDLE"]).resolve(strict=True)
    output = Path(os.environ["CFETCH_FOUNDATION_OUTPUT"])
    if not output.is_absolute():
        raise RuntimeError("CFETCH_FOUNDATION_OUTPUT must be an absolute fresh directory")
    model = os.environ.get("CFETCH_FOUNDATION_MODEL", "")
    if model not in ("", "AllMiniLML6V2", "EmbeddingGemma300M"):
        raise RuntimeError("choose loader-only (empty), AllMiniLML6V2 or EmbeddingGemma300M")
    evidence = json.loads(Path(__file__).with_name("evidence.json").read_text())
    if digest(bundle / "MANIFEST.json") != evidence["bundle_manifest_sha256"]:
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
    output.mkdir()
    # The independent timer also survives abrupt loss of this Python parent.
    command = ["timeout", "--signal=KILL", "300", str(bundle / "cfetch-npu-ort-foundation"), "--ort",
               str(bundle / "lib/libonnxruntime.so.1.24.1")]
    if model:
        command += ["--run-model", model, "--device", "CPU", "--cache",
                    str(bundle / "model-cache"), "--output", str(output / "model")]
    # Match the explicit cache selection; HF_HOME otherwise overrides FastEmbed.
    environment = dict(os.environ)
    environment.pop("HF_HOME", None)
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["RAYON_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    save(output / "intent.json", {"model": model or None, "device": "CPU",
         "bundle_manifest_sha256": evidence["bundle_manifest_sha256"],
         "deadline_seconds": 300, "maximum_embedding_calls": int(bool(model))})
    result = {"model": model or None, "requested_device": "CPU", "passed": False}
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
            save(output / "result.json", result)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
