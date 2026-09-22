#!/usr/bin/env python3
"""Prepare cfetch's native release once; verify complete handoffs before publishing."""
import argparse
import contextlib
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from types import SimpleNamespace
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "corbet-libs/cfetch"
PUBLISHER_REVISION = "33600b394910c643c5a6af6dff479fb16418428a"
TOKENS = ("GH_TOKEN", "GITHUB_TOKEN", "CARGO_REGISTRY_TOKEN", "NPM_TOKEN", "JSR_TOKEN", "PYPI_TOKEN", "PACKAGES_TOKEN")
METADATA = {"onnxruntime-LICENSE.txt": "release/onnxruntime-LICENSE.txt",
            "onnxruntime-NOTICES.txt": "release/onnxruntime-NOTICES.txt", "LICENSE.md": "LICENSE.md", "THIRD-PARTY-LICENSES.txt": "THIRD-PARTY-LICENSES.txt",
            "variants.json": "release/variants.json", "inference-backends.json": "release/inference-backends.json"}
MAX_FILE = 2 * 1024**3
MAX_TOTAL = 12 * 1024**3
CHECKS = {"catalog", "licenses", "profile", "rust"}


class Failure(Exception):
    pass


def require(condition, message):
    if not condition:
        raise Failure(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    require(Path(path).stat().st_size <= 16 * 1024**2, "Oversized release receipt")
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def durable(path, value):
    path = Path(path)
    payload = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == payload, "Existing immutable receipt differs")
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def command(args, *, env=None, capture=False):
    return subprocess.run(args, cwd=ROOT, env=env, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None).stdout


def no_credentials():
    require(not any(os.environ.get(name) for name in TOKENS), "Credentials must be absent from preparation and checks")


def output_directory():
    raw = os.environ.get("RELEASE_ARTIFACT_DIR")
    if not raw and os.environ.get("CARGO_TARGET_DIR"):
        namespace = os.environ.get("RELEASE_BUNDLE_SHA256") or os.environ.get("SOURCE_SHA256") or os.environ.get("CI_COMMIT_SHA")
        if not namespace:
            namespace = command(["git", "rev-parse", "HEAD"], capture=True).strip()
        require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", namespace), "Release namespace requires an exact source/bundle identity")
        raw = str(Path(os.environ["CARGO_TARGET_DIR"]) / "cfetch-release" / namespace)
    require(raw and Path(raw).is_absolute(), "Supply an absolute retained RELEASE_ARTIFACT_DIR")
    path = Path(raw)
    require(not path.is_symlink(), "Release artifact directory must not be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value):
    path = PurePosixPath(value)
    require(value and not path.is_absolute() and str(path) == value and not set(path.parts) & {".", ".."}
            and "\\" not in value, "Unsafe archive member")
    return path


def source_files(path):
    result = {}
    total = 0
    with tarfile.open(path, "r:*") as archive:
        commit = archive.pax_headers.get("comment", "").strip()
        require(re.fullmatch(r"[0-9a-f]{40}", commit), "Source archive lacks its exact Git commit")
        for member in archive:
            name = str(safe_name(member.name.rstrip("/")))
            if member.isdir():
                continue
            require(member.isfile() and name not in result, "Source members must be unique regular files")
            total += member.size
            require(len(result) < 50000 and member.size <= 128 * 1024**2 and total <= 512 * 1024**2, "Source exceeds release bounds")
            result[name] = archive.extractfile(member).read()
    return commit, result


def identity(source):
    commit, files = source_files(source)
    package = tomllib.loads(files["Cargo.toml"].decode())["package"]
    version = package["version"]
    require(package["name"] == "cfetch" and package["repository"].removesuffix(".git") == "https://github.com/" + REPOSITORY,
            "Source package/repository differs")
    require(re.fullmatch(r"0\.[0-9]+\.[0-9]+", version), "cfetch 1.0+ and prereleases require a separate release decision")
    tag = os.environ.get("RELEASE_TAG") or "v" + version
    require(tag == "v" + version, "Release tag differs from source version")
    return {"schema": 1, "repository": REPOSITORY, "source_commit": commit, "source_sha256": sha(source),
            "version": version, "tag": tag, "cargo_lock_sha256": hashlib.sha256(files["Cargo.lock"]).hexdigest()}, files


def current_source(directory):
    target = directory / "source.tar"
    supplied = os.environ.get("RELEASE_SOURCE_ARCHIVE") or os.environ.get("SOURCE_ARCHIVE")
    if supplied:
        expected_hash = os.environ.get("RELEASE_SOURCE_SHA256") if os.environ.get("RELEASE_SOURCE_ARCHIVE") else os.environ.get("SOURCE_SHA256")
        require(sha(supplied) == expected_hash, "Worker source archive checksum mismatch")
        expected = os.environ.get("CI_COMMIT_SHA") or os.environ.get("GITHUB_SHA")
        if not target.exists():
            shutil.copyfile(supplied, target)
        require(sha(target) == sha(supplied), "Retained source archive differs")
    else:
        expected = command(["git", "rev-parse", "HEAD"], capture=True).strip()
        command(["git", "diff", "--quiet", "HEAD", "--"])
        payload = subprocess.check_output(["git", "archive", "--format=tar", expected], cwd=ROOT)
        if target.exists():
            require(target.read_bytes() == payload, "Retained source archive differs")
        else:
            target.write_bytes(payload)
    result, files = identity(target)
    require(result["source_commit"] == expected, "Dispatched source commit mismatch")
    bind_source(target)
    return result, files


def bind_source(source):
    """Execution must use the committed source whose identity the receipt records."""
    expected = set()
    with tarfile.open(source) as archive:
        for member in archive:
            if member.isdir():
                continue
            path = ROOT.joinpath(*safe_name(member.name).parts)
            expected.add(member.name)
            require(member.isfile() and path.is_file() and not path.is_symlink(), "Execution source inventory differs")
            require(path.read_bytes() == archive.extractfile(member).read(), "Execution source bytes differ: " + member.name)
            if os.name != "nt":
                require(bool(path.stat().st_mode & 0o111) == bool(member.mode & 0o111), "Execution source mode differs: " + member.name)
    excluded = {ROOT / ".git", ROOT / "target", ROOT / ".ci-tool-source"}
    excluded.update(Path(value) for name in ("CARGO_TARGET_DIR", "RELEASE_ARTIFACT_DIR") if (value := os.environ.get(name)))
    actual = set()
    for parent, directories, filenames in os.walk(ROOT):
        base = Path(parent)
        require(not any((base / name).is_symlink() for name in directories if base / name not in excluded),
                "Uncommitted execution directory symlink")
        directories[:] = [name for name in directories if name != "__pycache__" and base / name not in excluded]
        for name in filenames:
            path = base / name
            if path in excluded or name.endswith(".pyc"):
                continue
            actual.add(path.relative_to(ROOT).as_posix())
    require(actual == expected, "Uncommitted execution inputs differ: " + repr(sorted(actual ^ expected)[:8]))


def matrix(files):
    for path in ("scripts/variant-matrix.sh", "scripts/stage_local_inference.py"):
        require(files.get(path) == (ROOT / path).read_bytes(), "Producing release evaluator differs; use its reviewed driver or explicitly review an import")
    with tempfile.TemporaryDirectory(prefix="cfetch-release-plan-") as temporary:
        directory = Path(temporary)
        (directory / "variants.json").write_bytes(files["release/variants.json"])
        (directory / "registry.json").write_bytes(files["release/inference-backends.json"])
        return json.loads(command(["bash", "scripts/variant-matrix.sh", "--release", str(directory / "variants.json"), str(directory / "registry.json")],
                                  env={key: value for key, value in os.environ.items() if key not in TOKENS}, capture=True))["include"]


def host():
    operating_system = {"Linux": "linux", "Darwin": "mac", "Windows": "win"}.get(platform.system())
    architecture = {"x86_64": "x86_64", "AMD64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(platform.machine())
    require(operating_system and architecture, "Unsupported native release host")
    return {"os": operating_system, "arch": architecture, "platform": platform.platform()}


def tools(rust=False):
    os.environ.setdefault("CARGO_BUILD_JOBS", "1")
    os.environ.setdefault("RUST_TEST_THREADS", "1")
    require(os.environ["CARGO_BUILD_JOBS"] == "1" and os.environ["RUST_TEST_THREADS"] == "1", "Release work requires one job and one test thread")
    result = {"python": sys.version, "build_jobs": os.environ.get("CARGO_BUILD_JOBS", "1"),
              "test_threads": os.environ.get("RUST_TEST_THREADS", "1")}
    if rust:
        result.update(rustc=command(["rustc", "--version", "--verbose"], capture=True).strip(),
                      cargo=command(["cargo", "--version"], capture=True).strip())
    return result


def plan(directory):
    no_credentials()
    source, files = current_source(directory)
    result = {**source, "variants": matrix(files), "checks": sorted(CHECKS), "rust_platforms": ["linux", "mac", "win"]}
    durable(directory / "plan.json", result)
    return result


def check(directory, selected):
    require(selected in CHECKS, "Unknown release check")
    result = plan(directory)
    native = host()
    key = selected + "-" + native["os"] + "-" + native["arch"]
    target = directory / "checks" / ("check-" + key + ".json")
    require(not target.exists(), "This check already has retained evidence; verify/reuse it instead of rerunning")
    provenance = tools(selected in {"rust", "licenses"})
    command(["bash", "scripts/ci-check.sh", selected])
    bind_source(directory / "source.tar")
    durable(target, {**{name: result[name] for name in identity(directory / "source.tar")[0]},
                     "kind": "check", "check": selected, "status": "success", "host": native,
                     "tools": provenance, "command": ["bash", "scripts/ci-check.sh", selected]})


def inventory(directory, *, modes=False):
    result = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "Package inventory must not contain symlinks")
        if path.is_file():
            require(path.stat().st_size <= MAX_FILE, "Package file exceeds the release limit")
            result[path.relative_to(directory).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
            if modes:
                result[path.relative_to(directory).as_posix()]["executable"] = bool(path.stat().st_mode & 0o111)
    return result


def pack(package, destination, extension):
    if extension == "zip":
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    info = zipfile.ZipInfo(package.name + "/" + path.relative_to(package).as_posix(), (1980, 1, 1, 0, 0, 0))
                    info.external_attr = (path.stat().st_mode & 0o777) << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    with path.open("rb") as stream, archive.open(info, "w", force_zip64=True) as output:
                        shutil.copyfileobj(stream, output)
    else:
        with destination.open("xb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted(package.rglob("*")):
                    if path.is_file():
                        info = archive.gettarinfo(str(path), arcname=package.name + "/" + path.relative_to(package).as_posix())
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ""
                        with path.open("rb") as member:
                            archive.addfile(info, member)


def prepare(directory, selected):
    result = plan(directory)
    require(selected == "cargo" or selected in {row["id"] for row in result["variants"]}, "Unknown preparation selection")
    source, files = identity(directory / "source.tar")
    native = host()
    target = directory / selected
    target.mkdir(exist_ok=True)
    receipt = target / ("cargo.json" if selected == "cargo" else "variant-" + selected + ".json")
    require(not receipt.exists(), "Prepared artifact already exists; use verify/status without rebuilding")
    require(not any(target.iterdir()), "Unreceipted preparation output exists; inspect it before running more compilation")
    provenance = tools(True)
    if selected == "cargo":
        cargo_target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
        filename = "cfetch-" + source["version"] + ".crate"
        candidates = [cargo_target / "package" / filename, cargo_target / "package/tmp-crate" / filename,
                      cargo_target / "package/tmp-registry" / filename]
        artifact = target / filename
        require(not artifact.exists(), "Unreceipted Cargo output exists; inspect it before rebuilding")
        require(not any(path.exists() for path in candidates), "Prior Cargo upload archives exist; preserve them and select a fresh target directory")
        command(["cargo", "publish", "--dry-run", "--locked"])
        produced = [path for path in candidates if path.is_file()]
        require(produced and len({sha(path) for path in produced}) == 1, "Cargo dry run did not retain one unambiguous upload archive")
        shutil.copyfile(produced[0], artifact)
        evidence = {"kind": "cargo", "dry_run": True}
    else:
        rows = [row for row in result["variants"] if row["id"] == selected]
        require(len(rows) == 1, "Variant is not admitted to this release")
        row = rows[0]
        require((row["os"], row["arch"]) == (native["os"], native["arch"]), "Native host does not match the selected variant; cross-builds are not native proof")
        package = target / (selected + "-" + source["tag"])
        require(not package.exists(), "Unreceipted native output exists; inspect it before rebuilding")
        args = ["cargo", "build", "--release", "--locked"]
        if row["target"]:
            args += ["--target", row["target"]]
        args += ["--features", row["cargo_features"]]
        command(args, env={**os.environ, "CFETCH_VARIANT": selected})
        cargo_target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
        binary = cargo_target / row["target"] / "release" / row["binary"] if row["target"] else cargo_target / "release" / row["binary"]
        package.mkdir()
        shutil.copy2(binary, package / row["binary"])
        for original in METADATA.values():
            shutil.copyfile(ROOT / original, package / Path(original).name)
        command([sys.executable, "scripts/stage_local_inference.py", "--registry", "release/inference-backends.json", "--catalog", "release/variants.json",
                 "--variant", selected, "--destination", str(package)])
        version = command([str(package / row["binary"]), "--version"], capture=True).strip()
        require(version == "cfetch " + source["version"], "Prepared executable reports another version")
        reported = json.loads(command([str(package / row["binary"]), "variants", "--json"], capture=True))
        require(reported.get("build_variant") == selected, "Executable embeds a different variant")
        filename = package.name + "." + row["archive"]
        artifact = target / filename
        contents = inventory(package, modes=True)
        pack(package, artifact, row["archive"])
        evidence = {"kind": "variant", "variant": selected, "row": row, "inventory": contents,
                    "native_smoke": {"version": version, "build_variant": reported["build_variant"]}, "build_command": args}
    bind_source(directory / "source.tar")
    durable(receipt, {**source, **evidence, "host": native, "tools": provenance,
                      "artifact": {"name": filename, "sha256": sha(artifact), "bytes": artifact.stat().st_size}})

def publisher():
    """Use the existing audited Cargo/immutable-intent protocol, without compiling it."""
    path = os.environ.get("CFETCH_PUBLISHER_ARCHIVE") or os.environ.get("CI_TOOL_ARCHIVE", "")
    digest = os.environ.get("CFETCH_PUBLISHER_SHA256") if os.environ.get("CFETCH_PUBLISHER_ARCHIVE") else os.environ.get("CI_TOOL_SHA256")
    require(path and sha(path) == digest, "Supply the verified ccid publisher archive and checksum")
    with tarfile.open(path) as archive:
        require(archive.pax_headers.get("comment", "").strip() == PUBLISHER_REVISION, "Publisher source revision mismatch")
        member = archive.getmember("adapters/registry_publish.py")
        require(member.isfile() and member.size < 256 * 1024, "Invalid publisher resource")
        payload = archive.extractfile(member).read()
    module = SimpleNamespace(__name__="cfetch_registry_publish")
    exec(compile(payload, "verified-ccid/adapters/registry_publish.py", "exec"), module.__dict__)
    return module


def archive_inventory(path, prefix):
    """Read package bytes, rejecting ambiguous paths, links and archive bombs."""
    result, metadata = {}, {}
    total = 0
    def add(name, size, mode, stream):
        nonlocal total
        require(name.startswith(prefix + "/"), "Native package root differs")
        relative = str(safe_name(name[len(prefix) + 1:]))
        require(relative not in result and size <= MAX_FILE, "Duplicate/oversized native member")
        total += size
        require(total <= MAX_TOTAL and len(result) < 50000, "Native package exceeds release bounds")
        digest = hashlib.sha256()
        received = 0
        payload = bytearray()
        while chunk := stream.read(1024 * 1024):
            received += len(chunk)
            require(received <= size, "Native member exceeds declared size")
            digest.update(chunk)
            if relative in METADATA:
                require(received <= 16 * 1024**2, "Oversized package metadata")
                payload.extend(chunk)
        require(received == size, "Truncated native package member")
        result[relative] = {"sha256": digest.hexdigest(), "bytes": size, "executable": bool(mode & 0o111)}
        if relative in METADATA:
            metadata[relative] = bytes(payload)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                safe_name(member.filename.rstrip("/"))
                require(not member.is_dir() and ((member.external_attr >> 16) & 0o170000) in {0, 0o100000}, "Nonregular native ZIP member")
                with archive.open(member) as stream:
                    add(member.filename, member.file_size, member.external_attr >> 16, stream)
    else:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                safe_name(member.name)
                require(member.isfile(), "Nonregular native TAR member")
                add(member.name, member.size, member.mode, archive.extractfile(member))
    return result, metadata


def inspect_tree(directory, core):
    source, files = identity(directory / "source.tar")
    planned = read_json(directory / "plan.json")
    require(all(planned.get(key) == value for key, value in source.items()), "Plan/source identity mismatch")
    rows = matrix(files)
    require(planned.get("variants") == rows and planned.get("checks") == sorted(CHECKS)
            and planned.get("rust_platforms") == ["linux", "mac", "win"], "Release plan changes the required gate inventory")
    observed, variants, crates = set(), {}, []
    for path in sorted((directory / "receipts").glob("*.json")):
        receipt = read_json(path)
        require(all(receipt.get(key) == value for key, value in source.items()), "Receipt belongs to another producing source")
        runtime = receipt.get("tools", {})
        require(runtime.get("python") and runtime.get("build_jobs") == "1" and runtime.get("test_threads") == "1", "Receipt lacks its actual constrained runtime")
        native = receipt.get("host", {})
        require(native.get("os") in {"linux", "mac", "win"} and native.get("arch") in {"x86_64", "aarch64"}, "Invalid native receipt host")
        if receipt.get("kind") == "check":
            selected = receipt.get("check")
            require(selected in CHECKS and receipt.get("status") == "success"
                    and receipt.get("command") == ["bash", "scripts/ci-check.sh", selected], "Receipt does not prove the common check")
            if selected in {"rust", "licenses"}:
                require(runtime.get("rustc") and runtime.get("cargo"), "Check receipt lacks actual compiler evidence")
            key = (selected, native["os"] if selected == "rust" else "any")
            require(key not in observed, "Duplicate release check receipt")
            observed.add(key)
            continue
        item = receipt["artifact"]
        require(str(safe_name(item["name"])) == item["name"] and "/" not in item["name"], "Unsafe release artifact name")
        artifact = directory / "artifacts" / item["name"]
        require(artifact.is_file() and not artifact.is_symlink() and artifact.stat().st_size == item["bytes"]
                and sha(artifact) == item["sha256"], "Prepared artifact checksum/size differs")
        require(runtime.get("rustc") and runtime.get("cargo"), "Prepared artifact lacks actual compiler evidence")
        if receipt.get("kind") == "cargo":
            require(receipt.get("dry_run") is True and item["bytes"] <= 128 * 1024**2, "Missing bounded Cargo dry-run evidence")
            crates.append((receipt, artifact))
        else:
            selected = receipt.get("variant")
            matches = [row for row in rows if row["id"] == selected]
            require(receipt.get("kind") == "variant" and len(matches) == 1 and selected not in variants, "Unexpected/duplicate release variant")
            row = matches[0]
            require(receipt.get("row") == row and (native["os"], native["arch"]) == (row["os"], row["arch"]), "Native variant identity mismatch")
            require(receipt.get("native_smoke") == {"version": "cfetch " + source["version"], "build_variant": selected}, "Native executable smoke proof is missing")
            args = ["cargo", "build", "--release", "--locked"] + (["--target", row["target"]] if row["target"] else []) + ["--features", row["cargo_features"]]
            require(receipt.get("build_command") == args, "Variant build command differs")
            prefix = selected + "-" + source["tag"]
            require(item["name"] == prefix + "." + row["archive"], "Native archive filename differs")
            contents, metadata = archive_inventory(artifact, prefix)
            require(contents == receipt.get("inventory") and row["binary"] in contents, "Native archive inventory differs")
            require(row["os"] == "win" or contents[row["binary"]]["executable"], "Native executable lost its execution mode")
            require(metadata == {name: files[original] for name, original in METADATA.items()}, "Native license/catalog/registry bytes differ from source")
            if row["backend"] == "endpoint":
                require(set(contents) == {*METADATA, row["binary"]}, "Endpoint package contains unexplained payload")
            variants[selected] = artifact
    required = {("rust", name) for name in ("linux", "mac", "win")} | {(name, "any") for name in CHECKS - {"rust"}}
    require(observed == required, "Incomplete catalog/license/profile/native Rust gates: " + repr(sorted(required - observed)))
    require(set(variants) == {row["id"] for row in rows}, "Incomplete eligible native artifact inventory")
    require(len(crates) == 1, "Exactly one prepared Cargo archive is required")
    receipt, crate = crates[0]
    proof = {path.name: sha(path) for path in sorted((directory / "receipts").glob("*.json"))}
    publication_identity = {"repository": REPOSITORY, "package": "cfetch", "version": source["version"],
                            "producing_commit": source["source_commit"], "source_sha256": source["source_sha256"],
                            "tag_commit": source["source_commit"], "gate_receipts": proof}
    cargo = {"identity": {**publication_identity, "registry": "cargo", "artifacts": {crate.name: sha(crate)}},
             "source": files, "artifacts": {crate.name: crate.read_bytes()}}
    core.Bundle.inspect_cargo(SimpleNamespace(name="cfetch", version=source["version"]), cargo)
    expected = {path.name for path in variants.values()} | {crate.name}
    require({path.name for path in (directory / "artifacts").iterdir()} == expected, "Unexpected prepared release artifact")
    return source, files, variants, cargo, publication_identity


def collect(directory, staged, core):
    no_credentials()
    target = directory / "bundle"
    require(not target.exists(), "Verified bundle already exists; inspect it instead of rebuilding")
    target.mkdir()
    (target / "receipts").mkdir()
    (target / "artifacts").mkdir()
    for name in ("source.tar", "plan.json"):
        matches = sorted(staged.rglob(name))
        require(matches and len({sha(path) for path in matches}) == 1, "Missing/conflicting staged " + name)
        shutil.copyfile(matches[0], target / name)
    for pattern in ("check-*.json", "variant-*.json", "cargo.json"):
        for receipt in sorted(staged.rglob(pattern)):
            destination = target / "receipts" / receipt.name
            require(not destination.exists(), "Duplicate staged receipt")
            shutil.copyfile(receipt, destination)
            item = read_json(receipt).get("artifact")
            if item:
                safe_name(item["name"])
                require("/" not in item["name"], "Artifact must be a plain filename")
                shutil.copyfile(receipt.parent / item["name"], target / "artifacts" / item["name"])
    source, files, variants, cargo, proof = inspect_tree(target, core)
    for name, original in METADATA.items():
        (target / "artifacts" / name).write_bytes(files[original])
    checksums = "".join(sha(path) + "  " + path.name + "\n" for path in sorted((target / "artifacts").iterdir()))
    (target / "artifacts/checksums_sha256.txt").write_text(checksums)
    durable(target / "bundle.json", {**source, "publication_identity": proof, "files": inventory(target)})
    archive_path = directory / ("cfetch-" + source["tag"] + "-bundle.tar")
    with tarfile.open(archive_path, "x", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                info = archive.gettarinfo(str(path), arcname=path.relative_to(target).as_posix())
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
    print(json.dumps({"bundle": str(archive_path), "sha256": sha(archive_path), "source_commit": source["source_commit"]}))


@contextlib.contextmanager
def imported(core):
    path = Path(os.environ.get("RELEASE_BUNDLE", ""))
    expected = os.environ.get("RELEASE_BUNDLE_SHA256", "")
    require(path.is_file() and re.fullmatch(r"[0-9a-f]{64}", expected) and sha(path) == expected, "Supply the retained release bundle with its independently reviewed SHA-256")
    with tempfile.TemporaryDirectory(prefix="cfetch-release-inspect-") as temporary:
        directory = Path(temporary)
        total = 0
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                name = safe_name(member.name)
                total += member.size
                destination = directory.joinpath(*name.parts)
                require(member.isfile() and not destination.exists() and member.size <= MAX_FILE and total <= MAX_TOTAL, "Unsafe/oversized release bundle")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    shutil.copyfileobj(archive.extractfile(member), stream)
        manifest = read_json(directory / "bundle.json")
        actual = inventory(directory)
        actual.pop("bundle.json")
        require(actual == manifest.get("files"), "Bundle inventory differs from its retained manifest")
        # Derived public metadata is checked independently and omitted only while
        # inspecting the exact prepared artifact set.
        _, files = identity(directory / "source.tar")
        metadata = [*METADATA, "checksums_sha256.txt"]
        derived = {name: (directory / "artifacts" / name).read_bytes() for name in metadata}
        require(all(derived[name] == files[original] for name, original in METADATA.items()), "Bundle metadata differs from source")
        for name in metadata:
            (directory / "artifacts" / name).unlink()
        result = inspect_tree(directory, core)
        require(manifest.get("publication_identity") == result[4] and all(manifest.get(key) == value for key, value in result[0].items()), "Bundle publication identity differs")
        for name, payload in derived.items():
            (directory / "artifacts" / name).write_bytes(payload)
        expected_checksums = "".join(sha(item) + "  " + item.name + "\n" for item in sorted((directory / "artifacts").iterdir()) if item.name != "checksums_sha256.txt")
        require(derived["checksums_sha256.txt"] == expected_checksums.encode(), "Published checksum inventory differs")
        yield directory, result

def remote_for(core, bundle):
    class CfetchRemote(core.Remote):
        def verify_tag(self):
            selected = self.release["id"] if self.release else None
            observed = release_object(core, self, self.bundle.channels["github"]["identity"], None)
            require(observed is not None and (selected is None or observed["id"] == selected),
                    "Selected release object changed; reconcile without publication")

        def asset_items(self):
            items = []
            for page in range(1, 101):
                batch = self.http.json("GET", self.api + f"/releases/{self.release['id']}/assets?per_page=100&page={page}", headers=self.github_headers())
                items.extend(batch)
                if len(batch) < 100:
                    require(len({item["name"] for item in items}) == len(items), "Duplicate release asset names")
                    return {item["name"]: item for item in items}
            raise Failure("Release asset inventory exceeds bounds")

        def verify_download(self, item, path):
            require(item.get("size") == path.stat().st_size and item.get("digest") == "sha256:" + sha(path), "Existing GitHub asset metadata conflicts")
            url = self.api + f"/releases/assets/{item['id']}"
            headers = {**self.github_headers(), "Accept": "application/octet-stream", "User-Agent": "cfetch-release"}
            try:
                request = urllib.request.Request(url, headers=headers)
                try:
                    response = urllib.request.build_opener(core.NoRedirect()).open(request, timeout=60)
                except urllib.error.HTTPError as error:
                    require(error.code in {301, 302, 303, 307, 308}, "GitHub asset download failed")
                    target = urllib.parse.urlsplit(error.headers.get("Location", ""))
                    require(target.scheme == "https" and target.hostname in {"release-assets.githubusercontent.com", "objects.githubusercontent.com"}
                            and not target.username and not target.password, "Unsafe GitHub asset redirect")
                    response = urllib.request.build_opener(core.NoRedirect()).open(urllib.request.Request(target.geturl()), timeout=60)
                digest, size = hashlib.sha256(), 0
                with response:
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        require(size <= path.stat().st_size, "Oversized GitHub asset")
                        digest.update(chunk)
                require(size == path.stat().st_size and digest.hexdigest() == sha(path), "Existing GitHub download differs from prepared bytes")
            except (urllib.error.URLError, TimeoutError, OSError):
                raise Failure("GitHub download unavailable; no publication was retried") from None

        def present(self, registry, data):
            if registry != "github":
                return super().present(registry, data)
            items = self.asset_items()
            for name in set(items) - set(data["artifacts"]):
                require(re.fullmatch(r"publication-(github|cargo)-[A-Za-z0-9_.-]+-(intent|response|complete)\.json", name), "Unexpected GitHub release asset: " + name)
                receipt = self.asset(name)
                identity_value = receipt.get("identity", {})
                channel = identity_value.get("registry")
                require(channel in self.bundle.channels and all(identity_value.get(key) == value for key, value in self.bundle.channels[channel]["identity"].items()), "Unexpected/conflicting publication journal asset")
                require(identity_value.get("unit") in self.bundle.channels[channel]["artifacts"] or name == "publication-github-finalize-intent.json", "Publication journal refers to an unknown artifact")
            present = set()
            for name, path in data["artifacts"].items():
                if name in items:
                    self.verify_download(items[name], path)
                    present.add(name)
            return present

        def credentials(self, registry):
            if registry != "github":
                return super().credentials(registry)
            require(bool(self.environment.get("GH_TOKEN")), "GH_TOKEN is missing for GitHub publication")
            require(self.release.get("draft") is True, "Adding release assets requires the verified draft; existing public releases are reconciliation-only")
            return self.environment["GH_TOKEN"]

        def upload(self, registry, unit, data, token):
            if registry != "github":
                return super().upload(registry, unit, data, token)
            path = data["artifacts"][unit]
            url = f"https://uploads.github.com/repos/{REPOSITORY}/releases/{self.release['id']}/assets?name=" + urllib.parse.quote(unit, safe="")
            with path.open("rb") as stream:
                self.http.request("POST", url, data=stream, headers={**self.github_headers(), "Content-Type": "application/octet-stream", "Content-Length": str(path.stat().st_size)})

    return CfetchRemote(bundle)


def release_object(core, remote, identity_value, journal, *, create=False):
    tag = remote.bundle.manifest["tag"]
    encoded = urllib.parse.quote(tag, safe="")
    reference = remote.http.json("GET", remote.api + "/git/ref/tags/" + encoded, headers=remote.github_headers())["object"]
    for _ in range(5):
        if reference.get("type") == "commit":
            break
        require(reference.get("type") == "tag" and re.fullmatch(r"[0-9a-f]{40}", reference.get("sha", "")), "Invalid live tag object")
        reference = remote.http.json("GET", remote.api + "/git/tags/" + reference["sha"], headers=remote.github_headers())["object"]
    require(reference.get("type") == "commit" and reference.get("sha") == identity_value["producing_commit"], "Live release tag differs from producing source")
    def observed():
        matches = []
        for page in range(1, 101):
            batch = remote.http.json("GET", remote.api + f"/releases?per_page=100&page={page}", headers=remote.github_headers())
            matches.extend(item for item in batch if item.get("tag_name") == tag)
            if len(batch) < 100:
                break
        else:
            raise Failure("Release inventory exceeds bounds")
        require(len(matches) <= 1, "Multiple release objects match the tag; reconcile without mutation")
        result = matches[0] if matches else None
        if result:
            require(result.get("tag_name") == tag and isinstance(result.get("id"), int), "GitHub release object identity differs")
        return result
    result = observed()
    if result is None and create:
        require(os.environ.get("GH_TOKEN"), "GH_TOKEN is missing for draft release creation")
        intent_path = journal / "github-create-intent.json"
        require(not intent_path.exists(), "Previous release creation intent exists; reconcile it without another POST")
        require(int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")) == 1 or os.environ.get("RELEASE_RECOVERY_JOURNAL") == "retained",
                "Hosted rerun requires the retained creation journal; use status before recovery")
        durable(intent_path, {"identity": identity_value, "owner": secrets.token_hex(24), "operation": "create-draft"})
        failure = None
        try:
            remote.http.json("POST", remote.api + "/releases", data=json_bytes({"tag_name": tag, "target_commitish": identity_value["producing_commit"], "name": tag, "draft": True, "generate_release_notes": True}), headers={**remote.github_headers(), "Content-Type": "application/json"})
        except core.Failure as error:
            failure = error
        result = observed()
        require(result is not None, str(failure) if failure else "Draft creation remains unobserved; do not retry")
        durable(journal / "github-create-complete.json", {"identity": identity_value, "release_id": result["id"]})
    remote.release = result
    return result


def publish(directory, core, operation, destination):
    with imported(core) as (tree, inspected):
        source, files, variants, cargo, proof = inspected
        if operation == "inspect":
            print(json.dumps({**source, "variants": sorted(variants), "gates": "complete", "cargo": sorted(cargo["artifacts"])}))
            return
        journal_raw = os.environ.get("RELEASE_JOURNAL_ROOT") or str(directory / "publication")
        journal = Path(journal_raw) / source["version"]
        require(journal.is_absolute() and not journal.is_symlink(), "Use a retained absolute publication journal")
        journal.mkdir(parents=True, exist_ok=True)
        assets = {path.name: path for path in (tree / "artifacts").iterdir()}
        github = {"identity": {**proof, "registry": "github", "artifacts": {name: sha(path) for name, path in assets.items()}}, "artifacts": assets}
        bundle = SimpleNamespace(repository=REPOSITORY, name="cfetch", version=source["version"],
                                 manifest={"tag": source["tag"], "tag_commit": source["source_commit"]},
                                 channels={"cargo": cargo, "github": github})
        remote = remote_for(core, bundle)
        # flock is already a required part of the shared Linux publication driver.
        with (journal / "cfetch-release.lock").open("a") as lock:
            core.fcntl.flock(lock, core.fcntl.LOCK_EX)
            release = release_object(core, remote, proof, journal, create=operation == "create")
            if release is None:
                require(operation == "status", "Create the verified draft release before publishing")
                print(json.dumps({"release": "absent", "cargo": sorted(remote.present("cargo", cargo))}))
                return
            if operation == "create":
                print(json.dumps({"release_id": release["id"], "draft": release["draft"]}))
                return
            destinations = ["github", "cargo"] if destination == "all" else [destination]
            require(operation != "publish" or destination != "all", "Select one publication destination per credentialed step")
            if operation == "publish" and destination == "cargo":
                require_cargo_release(remote, github)
            result = core.Publisher(bundle, remote, journal).execute(destinations, operation == "publish")
            if operation == "finish":
                require(destination == "github" and not result["github"]["missing"], "Complete exact native assets are required before making the draft public")
                if remote.release.get("draft"):
                    name = "publication-github-finalize-intent.json"
                    local = journal / name
                    require(not local.exists() and remote.asset(name) is None, "Prior finalize intent exists; reconcile without another PATCH")
                    intent = {"identity": github["identity"], "owner": secrets.token_hex(24), "operation": "publish-draft"}
                    require(os.environ.get("GH_TOKEN"), "GH_TOKEN is missing for finalizing the release")
                    durable(local, intent)
                    remote.persist_asset(name, intent, claim=True)
                    try:
                        remote.http.json("PATCH", remote.api + f"/releases/{remote.release['id']}", data=json_bytes({"draft": False}), headers={**remote.github_headers(), "Content-Type": "application/json"})
                    except core.Failure:
                        pass  # Reconcile the observed object; never send a second PATCH.
                    remote.verify_tag()
                    require(remote.release.get("draft") is False, "Finalization remains pending; inspect the saved intent without retrying")
            print(json.dumps({"release_id": remote.release["id"], "draft": remote.release["draft"], "destinations": result}))


def require_cargo_release(remote, github):
    require(remote.release.get("draft") is False and remote.present("github", github) == set(github["artifacts"]),
            "Cargo publication requires the complete exact public GitHub release")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["guards", "plan", "check", "prepare", "verify", "inspect", "status", "create", "publish", "finish"])
    parser.add_argument("selection", nargs="?", default="all")
    parser.add_argument("--input", type=Path, help="Directory containing retained provider preparation/check artifacts")
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        arguments = [os.environ.get("RELEASE_OPERATION", "inspect"), os.environ.get("RELEASE_SELECTION", "all")]
        if os.environ.get("RELEASE_INPUT"):
            arguments += ["--input", os.environ["RELEASE_INPUT"]]
    args = parser.parse_args(arguments)
    directory = output_directory()
    if args.operation == "guards":
        no_credentials()
        command([sys.executable, "-m", "unittest", "-v", "scripts.test_release"])
    elif args.operation == "plan":
        print(json.dumps({"include": plan(directory)["variants"]}, separators=(",", ":")))
    elif args.operation == "check":
        check(directory, args.selection)
    elif args.operation == "prepare":
        prepare(directory, args.selection)
    elif args.operation == "verify":
        require(args.input and args.input.is_dir(), "verify requires the retained staged artifact directory")
        collect(directory, args.input, publisher())
    else:
        require(args.selection in {"all", "github", "cargo"}, "Unknown publication destination")
        publish(directory, publisher(), args.operation, args.selection)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # HTTP helpers never include tokens/response bodies in their errors.
        print("release: " + str(error), file=sys.stderr)
        raise SystemExit(1) from None
