#!/usr/bin/env python3
"""Stage the exact content-addressed local inference payload for one variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import tempfile
from typing import Any, BinaryIO
import urllib.request
import zipfile


MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
MAX_FILES = 4096
SHA256_RE = re.compile(r"[0-9a-f]{64}")
URL_RE = re.compile(
    r"https://github\.com/corbet-libs/cfetch/releases/download/"
    r"([A-Za-z0-9][A-Za-z0-9._-]{0,127})/([0-9a-f]{64})\.(zip|tar\.gz)"
)


class StagingError(ValueError):
    """A local inference payload cannot be staged without weakening identity."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StagingError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw or len(raw) > 1024 * 1024:
        raise StagingError(f"{label} must contain bounded nonempty JSON")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StagingError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise StagingError(f"{label} must contain one JSON object")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise StagingError(f"{label} must be a lowercase SHA-256")
    return value


def _plain_basename(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 255
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).name != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise StagingError(f"{label} must be a plain basename")
    return value


def _member_name(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != value
    ):
        raise StagingError(f"archive contains unsafe path {value!r}")
    return value


def _copy_bounded(source: BinaryIO, target: Path, declared_size: int) -> None:
    written = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        while chunk := source.read(1024 * 1024):
            written += len(chunk)
            if written > declared_size:
                raise StagingError(f"archive member {target.name!r} exceeded its size")
            output.write(chunk)
    if written != declared_size:
        raise StagingError(f"archive member {target.name!r} was truncated")


def _extract_zip(archive_path: Path, root: Path) -> set[str]:
    names: set[str] = set()
    expanded = 0
    try:
        archive = zipfile.ZipFile(archive_path, "r")
    except zipfile.BadZipFile as error:
        raise StagingError("local inference payload is not a valid ZIP") from error
    with archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_FILES:
            raise StagingError(f"ZIP must contain 1..{MAX_FILES} regular files")
        for info in infos:
            name = _member_name(info.filename)
            if name in names:
                raise StagingError(f"ZIP repeats member {name!r}")
            names.add(name)
            mode = info.external_attr >> 16
            if info.is_dir() or not stat.S_ISREG(mode) or info.file_size < 1:
                raise StagingError(f"ZIP member {name!r} is not a nonempty regular file")
            expanded += info.file_size
            if expanded > MAX_EXPANDED_BYTES:
                raise StagingError("ZIP exceeds the expanded payload bound")
            target = root.joinpath(*PurePosixPath(name).parts)
            with archive.open(info, "r") as source:
                _copy_bounded(source, target, info.file_size)
            os.chmod(target, stat.S_IMODE(mode))
    return names


def _extract_tar_gz(archive_path: Path, root: Path) -> set[str]:
    names: set[str] = set()
    expanded = 0
    try:
        archive = tarfile.open(archive_path, "r:gz")
    except tarfile.TarError as error:
        raise StagingError("local inference payload is not a valid tar.gz") from error
    with archive:
        members = archive.getmembers()
        regular = [member for member in members if member.isfile()]
        if len(regular) != len(members) or not regular or len(regular) > MAX_FILES:
            raise StagingError(f"tar.gz must contain 1..{MAX_FILES} regular files only")
        for member in regular:
            name = _member_name(member.name)
            if name in names:
                raise StagingError(f"tar.gz repeats member {name!r}")
            names.add(name)
            if member.size < 1:
                raise StagingError(f"tar.gz member {name!r} is empty")
            expanded += member.size
            if expanded > MAX_EXPANDED_BYTES:
                raise StagingError("tar.gz exceeds the expanded payload bound")
            source = archive.extractfile(member)
            if source is None:
                raise StagingError(f"tar.gz member {name!r} could not be read")
            target = root.joinpath(*PurePosixPath(name).parts)
            with source:
                _copy_bounded(source, target, member.size)
            os.chmod(target, stat.S_IMODE(member.mode))
    return names


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    request = urllib.request.Request(
        url,
        headers={"Accept-Encoding": "identity", "User-Agent": "cfetch-release-packager/1"},
    )
    digest = hashlib.sha256()
    size = 0
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except OSError as error:
        raise StagingError(f"could not download local inference payload: {error}") from error
    with response, destination.open("xb") as output:
        if response.geturl().split(":", 1)[0].lower() != "https":
            raise StagingError("local inference payload redirected away from HTTPS")
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise StagingError("local inference archive exceeds its byte bound")
            digest.update(chunk)
            output.write(chunk)
    if size < 1 or digest.hexdigest() != expected_sha256:
        raise StagingError("downloaded local inference archive failed its SHA-256")


def _validate_plan(
    registry: dict[str, Any], catalog: dict[str, Any], variant_id: str
) -> dict[str, Any] | None:
    variants = catalog.get("variants")
    packages = registry.get("local_packages")
    if not isinstance(variants, list) or not isinstance(packages, list):
        raise StagingError("release catalog or local package registry is malformed")
    variant_rows = [row for row in variants if isinstance(row, dict) and row.get("id") == variant_id]
    if len(variant_rows) != 1:
        raise StagingError(f"release variant {variant_id!r} is not unique")
    variant = variant_rows[0]
    package_rows = [
        row
        for row in packages
        if isinstance(row, dict) and row.get("release_variant_id") == variant_id
    ]
    if variant.get("backend") in ("endpoint", "cpu", "lexical"):
        if package_rows:
            raise StagingError("endpoint release variant unexpectedly has a local payload")
        return None
    if variant.get("backend") != "local" or len(package_rows) != 1:
        raise StagingError("local release variant must have exactly one target payload")
    plan = package_rows[0]
    expected = _digest(plan.get("package_sha256"), "package_sha256")
    package_format = plan.get("package_format")
    match = URL_RE.fullmatch(plan.get("package_url", ""))
    if (
        match is None
        or match.group(2) != expected
        or match.group(3) != package_format
    ):
        raise StagingError("local package URL is not exact content-addressed cfetch release data")
    dispatcher = plan.get("dispatcher")
    if not isinstance(dispatcher, dict) or set(dispatcher) != {"binary", "sha256"}:
        raise StagingError("local package plan has no exact dispatcher")
    _plain_basename(dispatcher["binary"], "dispatcher.binary")
    _digest(dispatcher["sha256"], "dispatcher.sha256")
    _digest(plan.get("package_manifest_sha256"), "package_manifest_sha256")
    return plan


def stage_archive(
    archive_path: Path,
    package_format: str,
    plan: dict[str, Any],
    destination: Path,
) -> None:
    """Publish one complete payload beneath a privately owned release staging root.

    The release builder exclusively owns destination while this runs. This is
    not a concurrent installer: refuse an existing payload and recheck before
    the single directory rename, without merging or replacing package files.
    """
    if not destination.is_dir() or destination.is_symlink():
        raise StagingError("release staging destination must be a real existing directory")
    payload = destination / "inference"
    if os.path.lexists(payload):
        raise StagingError("local payload destination inference already exists")
    temporary = Path(tempfile.mkdtemp(prefix=".cfetch-local-", dir=destination))
    try:
        if package_format == "zip":
            names = _extract_zip(archive_path, temporary)
        elif package_format == "tar.gz":
            names = _extract_tar_gz(archive_path, temporary)
        else:
            raise StagingError(f"unsupported local package format {package_format!r}")
        dispatcher = plan["dispatcher"]
        binary = _plain_basename(dispatcher["binary"], "dispatcher.binary")
        if binary not in names:
            raise StagingError("local payload omitted its declared dispatcher")
        dispatcher_path = temporary / binary
        if not os.access(dispatcher_path, os.X_OK):
            raise StagingError("local payload dispatcher is not executable")
        if file_sha256(dispatcher_path) != dispatcher["sha256"]:
            raise StagingError("local payload dispatcher failed its SHA-256")
        manifest = _load_json(temporary / "package-manifest.json", "package manifest")
        if (
            file_sha256(temporary / "package-manifest.json")
            != plan.get("package_manifest_sha256")
        ):
            raise StagingError("local payload package manifest failed its externally pinned SHA-256")
        if manifest.get("package_state") != "release":
            raise StagingError("local payload package manifest must be in release state")
        scopes = manifest.get("scopes")
        scope_ids = (
            [scope.get("scope_id") for scope in scopes]
            if isinstance(scopes, list) and all(isinstance(scope, dict) for scope in scopes)
            else None
        )
        if scope_ids != plan.get("ordered_scope_ids"):
            raise StagingError("local payload scope order differs from its release plan")
        if os.path.lexists(payload):
            raise StagingError("local payload destination inference appeared during staging")
        os.rename(temporary, payload)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def stage_variant(
    registry_path: Path,
    catalog_path: Path,
    variant_id: str,
    destination: Path,
) -> bool:
    plan = _validate_plan(
        _load_json(registry_path, "local inference registry"),
        _load_json(catalog_path, "release variant catalog"),
        variant_id,
    )
    if plan is None:
        return False
    with tempfile.TemporaryDirectory(prefix="cfetch-local-download-") as directory:
        archive = Path(directory) / f"payload.{plan['package_format']}"
        _download(plan["package_url"], archive, plan["package_sha256"])
        stage_archive(archive, plan["package_format"], plan, destination)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    try:
        staged = stage_variant(
            args.registry, args.catalog, args.variant, args.destination
        )
    except (OSError, StagingError, ValueError) as error:
        print(f"local inference staging refused: {error}")
        return 1
    print("local inference payload staged" if staged else "endpoint-only variant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
