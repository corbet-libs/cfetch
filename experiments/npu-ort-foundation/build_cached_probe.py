"""Build a separate diagnostic probe from retained, verified offline inputs."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import tarfile


ARCHIVE_SHA256 = "b5e6a29ef0fc303e63ce89f2b71d98bee9df51d79cbbedb78dd0d6ba9f3b172f"
PATCH_SHA256 = "50114597b4e382bb76057f56b5e059b3c20f97e70e6a29462101f1ce5df6ea20"
# The measured patch changes only this file; all other bytes must match the crate.
PATCHED_IMPL_SHA256 = "076cabc5fb7b659f675fc929fb1fe7976e145b5c7686d5c43b0fd7ed3ae2c407"
SOURCE_FILES = ("Cargo.toml", "Cargo.lock", "fastembed-session.patch", "src/main.rs")
BINARY = "cfetch-npu-ort-foundation"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def tree(path):
    result = {}
    for item in sorted(path.rglob("*")):
        require(not item.is_symlink(), "vendor symlinks are forbidden")
        if item.is_dir():
            continue
        require(stat.S_ISREG(item.stat().st_mode), "vendor contains a special file")
        result[str(item.relative_to(path))] = {
            "bytes": item.stat().st_size, "sha256": digest(item)}
    return result


def tree_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_vendor(retained):
    archive = retained / "fastembed-6.0.2.crate"
    require(digest(archive) == ARCHIVE_SHA256, "published FastEmbed archive changed")
    expected = {}
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            require(member.isdir() or member.isfile(), "crate contains a special entry")
            if member.isdir():
                continue
            name = str(Path(member.name).relative_to("fastembed-6.0.2"))
            require(".." not in Path(name).parts and name not in expected, "invalid crate member")
            content = stream.extractfile(member).read()
            expected[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    actual = tree(retained / "build/vendor/fastembed")
    changed = "src/text_embedding/impl.rs"
    require(actual.get(changed, {}).get("sha256") == PATCHED_IMPL_SHA256,
            "retained FastEmbed does not contain the measured patch")
    expected[changed] = actual[changed]
    require(actual == expected, "retained vendor differs from the verified patched crate")
    return actual


def verify_bundle(bundle, expected_sha256):
    require(digest(bundle / "MANIFEST.json") == expected_sha256, "retained bundle manifest changed")
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = bundle / name
        require(path.resolve(strict=True).is_relative_to(bundle), "bundle path escapes its root")
        if "symlink" in expected:
            require(path.is_symlink() and os.readlink(path) == expected["symlink"], "bundle symlink changed")
        else:
            require(path.stat().st_size == expected["bytes"] and digest(path) == expected["sha256"],
                    "retained bundle file changed: " + name)
    return manifest


def command(output, label, args, environment, cwd, seconds=15):
    """One owned process group, with an independent timer and bounded reap."""
    save(output / (label + "-intent.json"), {"argv": args, "deadline_seconds": seconds})
    with (output / (label + ".stdout")).open("xb") as stdout, \
            (output / (label + ".stderr")).open("xb") as stderr:
        child = subprocess.Popen(["timeout", "--signal=KILL", str(seconds), *args],
                                 cwd=cwd, env=environment, stdout=stdout, stderr=stderr,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=seconds + 1)
        except BaseException:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    save(output / (label + "-unreaped.json"), {"pid": child.pid})
                    os._exit(1)
            raise
    save(output / (label + "-result.json"), {"returncode": code})
    require(code == 0, label + " failed; see retained stdout/stderr")
    return (output / (label + ".stdout")).read_text()


def main():
    require(os.uname().sysname == "Linux" and os.uname().machine == "x86_64", "Linux x86-64 worker required")
    retained = Path(os.environ["CFETCH_FOUNDATION_RETAINED_ROOT"]).resolve(strict=True)
    output = Path(os.environ["CFETCH_FOUNDATION_PROBE_DIR"])
    require(output.is_absolute() and not output.exists(), "probe output must be an absolute fresh directory")
    source = Path(__file__).resolve().parent
    evidence = json.loads((source / "evidence.json").read_text())
    require(digest(source / "Cargo.lock") == evidence["lockfile_sha256"], "measured Cargo.lock changed")
    require(digest(source / "fastembed-session.patch") == PATCH_SHA256, "measured patch changed")
    require(digest(source / "Cargo.toml") == evidence["source_sha256"]["Cargo.toml"], "dependency configuration changed")
    for name in ("cargo-home/registry/src", "target/debug", "build/vendor/fastembed"):
        require((retained / name).is_dir(), "missing retained build input: " + name)
    environment = dict(os.environ)
    pkg_config = environment.get("CFETCH_FOUNDATION_PKG_CONFIG", "")
    openssl_dev = environment.get("CFETCH_FOUNDATION_OPENSSL_DEV", "")
    if pkg_config or openssl_dev:
        require(pkg_config and openssl_dev, "supply both existing pkg-config and OpenSSL development paths")
        pkg_config = Path(pkg_config)
        openssl_dev = Path(openssl_dev)
        require(pkg_config.is_absolute() and pkg_config.is_file() and os.access(pkg_config, os.X_OK),
                "explicit pkg-config must be an existing absolute executable")
        require(openssl_dev.is_absolute() and (openssl_dev / "lib/pkgconfig/openssl.pc").is_file(),
                "explicit OpenSSL development path lacks package metadata")
        environment["PATH"] = str(pkg_config.parent) + os.pathsep + environment["PATH"]
        environment["PKG_CONFIG"] = str(pkg_config)
        environment["PKG_CONFIG_PATH"] = str(openssl_dev / "lib/pkgconfig")
    tools = {}
    for name in ("cargo", "rustc", "cc", "c++", "pkg-config", "timeout"):
        executable = shutil.which(name, path=environment["PATH"])
        require(executable is not None, "worker lacks required tool: " + name)
        path = Path(executable).resolve(strict=True)
        tools[name] = {"path": str(path), "sha256": digest(path)}
    vendor_files = verify_vendor(retained)
    bundle = (retained / "runtime-bundle").resolve(strict=True)
    manifest_sha256 = evidence["bundle_manifest_sha256"]
    runtime = verify_bundle(bundle, manifest_sha256)
    output.mkdir(mode=0o700)
    output = output.resolve(strict=True)
    save(output / "intent.json", {"kind": "offline-probe-build", "model_execution": False,
         "bundle_manifest_sha256": manifest_sha256, "maximum_build_seconds": 600})
    environment.update(CARGO_HOME=str(retained / "cargo-home"), CARGO_TARGET_DIR=str(retained / "target"),
                       CARGO_BUILD_JOBS="1", CARGO_NET_OFFLINE="true", RUSTUP_AUTO_INSTALL="0",
                       CC=shutil.which("cc"), CXX=shutil.which("c++"), RUSTC=shutil.which("rustc"))
    require(not environment.get("RUSTC_WRAPPER") and not environment.get("RUSTC_WORKSPACE_WRAPPER"),
            "compiler wrappers are not supported by this retained build")
    require(not environment.get("CARGO_BUILD_TARGET"), "this probe requires the existing native debug target")
    require(not any((retained / "cargo-home" / name).exists() for name in ("config", "config.toml")),
            "unexpected Cargo cache configuration")
    for ancestor in (output, *output.parents):
        require(not any((ancestor / ".cargo" / name).exists() for name in ("config", "config.toml")),
                "build output inherits an unexpected Cargo configuration")
    versions = {}
    for name, args in (("cargo", ["--version"]), ("rustc", ["-vV"]), ("cc", ["--version"]),
                       ("c++", ["--version"]), ("pkg-config", ["--version"]), ("timeout", ["--version"])):
        versions[name] = command(output, name + "-version", [name, *args], environment, output).strip()
    sysroot = command(output, "rustc-sysroot", ["rustc", "--print", "sysroot"], environment, output).strip()
    compiler = (Path(sysroot) / "bin/rustc").resolve(strict=True)
    tools["selected_rustc"] = {"path": str(compiler), "sha256": digest(compiler)}
    versions["openssl"] = command(output, "openssl-version", ["pkg-config", "--modversion", "openssl"],
                                  environment, output).strip()
    include = command(output, "openssl-includes", ["pkg-config", "--variable=includedir", "openssl"],
                      environment, output).strip()
    require(include and all((Path(include) / "openssl" / name).is_file() for name in ("ssl.h", "crypto.h")),
            "worker lacks OpenSSL development headers")
    versions["openssl_libraries"] = command(output, "openssl-libraries", ["pkg-config", "--libs", "openssl"],
                                            environment, output).strip()
    build = output / "build-source"
    (build / "src").mkdir(parents=True)
    for name in SOURCE_FILES:
        shutil.copyfile(source / name, build / name)
    shutil.copytree(retained / "build/vendor/fastembed", build / "vendor/fastembed")
    require(tree(build / "vendor/fastembed") == vendor_files, "vendor changed while staging")
    source_hashes = {name: digest(build / name) for name in SOURCE_FILES}
    command(output, "cargo-build", ["cargo", "build", "--offline", "--locked", "--jobs", "1", "--bin", BINARY],
            environment, build, seconds=600)
    require({name: digest(build / name) for name in SOURCE_FILES} == source_hashes, "build sources changed")
    require(tree(build / "vendor/fastembed") == vendor_files, "build modified vendor sources")
    verify_bundle(bundle, manifest_sha256)
    produced = retained / "target/debug" / BINARY
    require(produced.is_file() and not produced.is_symlink(), "build did not produce a regular binary")
    with produced.open("rb") as src, (output / BINARY).open("xb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    (output / BINARY).chmod(0o755)
    save(output / "build-identity.json", {"schema_version": 1, "kind": "cfetch-ort-foundation-build-v1",
         "model_execution": False, "binary_sha256": digest(output / BINARY), "source_sha256": source_hashes,
         "vendor_tree_sha256": tree_digest(vendor_files), "fastembed_archive_sha256": ARCHIVE_SHA256,
         "bundle_manifest_sha256": manifest_sha256, "runtime_files": runtime["files"],
         "compiler_tools": tools, "versions": versions, "builder_sha256": digest(Path(__file__)),
         "build_environment": {key: value for key, value in environment.items() if key in (
             "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTUP_TOOLCHAIN", "CC", "CXX", "RUSTC",
             "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "OPENSSL_DIR", "OPENSSL_LIB_DIR",
             "OPENSSL_INCLUDE_DIR", "OPENSSL_STATIC", "PKG_CONFIG_PATH")},
         "build_command": ["cargo", "build", "--offline", "--locked", "--jobs", "1", "--bin", BINARY]})


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise RuntimeError(f"build helper interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    main()
