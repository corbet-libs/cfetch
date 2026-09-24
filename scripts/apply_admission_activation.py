#!/usr/bin/env python3
"""Apply one fully verified cfetch admission activation bundle to a checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence
import urllib.parse
import urllib.request


MAX_ACTIVATION_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_REGISTRY_BYTES = 1024 * 1024
MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_FILES = 1024
MAX_ACTIVATION_BYTES = 4 * 1024 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SLUG_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
RELEASE_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
PROFILE_SOURCE_PATH = "src/embedding_profile.rs"
PROFILE_STATUS_CANDIDATE_TEXT = 'pub const PROFILE_STATUS: &str = "candidate";'
PROFILE_STATUS_ACTIVE_TEXT = 'pub const PROFILE_STATUS: &str = "active";'
REGISTRY_PATH = "release/inference-backends.json"
VARIANTS_PATH = "release/variants.json"
RELEASE_REPOSITORY = "corbet-libs/cfetch"

ACTIVATION_FIELDS = {
    "schema_version",
    "stage_id",
    "release_tag",
    "base_registry_sha256",
    "base_variants_sha256",
    "admission_implementation_bundle_sha256",
    "profile_source_promotion",
    "compatibility_report",
    "compatibility_report_sha256",
    "compatibility_report_bytes",
    "registry",
    "registry_sha256",
    "registry_bytes",
    "assets",
    "receipts",
    "status",
}
PROMOTION_FIELDS = {
    "path",
    "base_sha256",
    "active_sha256",
    "candidate_text",
    "active_text",
}
ASSET_FIELDS = {"kind", "owner_id", "filename", "path", "sha256", "bytes", "format"}
RECEIPT_FIELDS = {
    "package_id",
    "scope_id",
    "receipt",
    "receipt_sha256",
    "receipt_bytes",
}


class ActivationError(ValueError):
    """An activation bundle or target checkout failed closed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ActivationError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing or unknown:
        raise ActivationError(
            f"{label} schema mismatch; missing={missing}, unknown={unknown}"
        )


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ActivationError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_size(value: object, label: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        bound = f"1..{maximum}" if maximum is not None else "positive"
        raise ActivationError(f"{label} bytes must be {bound}")
    return value


def _slug(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 128
        or SLUG_RE.fullmatch(value) is None
    ):
        raise ActivationError(f"{label} must be a canonical lowercase slug")
    return value


def _relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ActivationError(f"{label} must be a canonical relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != value
    ):
        raise ActivationError(f"{label} must be a canonical relative POSIX path")
    return value


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ActivationError(f"{label} must be a regular non-symlink file")
    with path.open("rb") as source:
        size = os.fstat(source.fileno()).st_size
        if size < 1 or size > maximum:
            raise ActivationError(f"{label} must contain 1..{maximum} bytes")
        data = source.read(maximum + 1)
    if len(data) != size:
        raise ActivationError(f"{label} changed while it was read")
    return data


def _parse_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActivationError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise ActivationError(f"{label} must contain one JSON object")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_download_url(url: str) -> None:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise ActivationError(
            "published admission asset redirected to an invalid URL"
        ) from error
    hostname = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or not (
            hostname == "github.com"
            or hostname.endswith(".githubusercontent.com")
        )
    ):
        raise ActivationError(
            "published admission assets must remain on credential-free GitHub HTTPS"
        )


class _HttpsGithubRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> urllib.request.Request | None:
        _validate_download_url(new_url)
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


def _release_asset_url(release_tag: str, filename: str) -> str:
    return (
        f"https://github.com/{RELEASE_REPOSITORY}/releases/download/"
        f"{release_tag}/{filename}"
    )


def _verify_published_assets(
    activation: Mapping[str, object], opener: object | None = None
) -> dict[str, int]:
    """Download every bound release asset before any checkout mutation."""
    assets = activation["assets"]
    if not isinstance(assets, list) or not assets:
        raise ActivationError("activation assets must be a nonempty array")
    active_opener = opener or urllib.request.build_opener(
        _HttpsGithubRedirectHandler()
    )
    verified_bytes = 0
    for index, asset in enumerate(assets):
        label = f"published assets[{index}]"
        if not isinstance(asset, dict):
            raise ActivationError(f"{label} must be an object")
        digest = _digest(asset.get("sha256"), f"{label}.sha256")
        size = _positive_size(asset.get("bytes"), label)
        filename = asset.get("filename")
        if not isinstance(filename, str) or filename != (
            f"{digest}.{asset.get('format')}"
        ):
            raise ActivationError(f"{label} is not content-addressed")
        url = _release_asset_url(str(activation["release_tag"]), filename)
        _validate_download_url(url)
        request = urllib.request.Request(
            url, headers={"User-Agent": "cfetch-admission-activation/1"}
        )
        observed_digest = hashlib.sha256()
        observed_size = 0
        try:
            response_context = active_opener.open(request, timeout=60)
            with response_context as response:
                _validate_download_url(response.geturl())
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_size = int(content_length)
                    except ValueError as error:
                        raise ActivationError(
                            f"{label} has an invalid Content-Length"
                        ) from error
                    if declared_size != size:
                        raise ActivationError(
                            f"{label} Content-Length does not match the activation"
                        )
                while chunk := response.read(1024 * 1024):
                    observed_size += len(chunk)
                    if observed_size > size:
                        raise ActivationError(
                            f"{label} exceeds its activation byte count"
                        )
                    observed_digest.update(chunk)
        except ActivationError:
            raise
        except OSError as error:
            raise ActivationError(f"{label} could not be downloaded: {error}") from error
        if observed_size != size or observed_digest.hexdigest() != digest:
            raise ActivationError(
                f"{label} downloaded bytes do not match the activation"
            )
        verified_bytes += observed_size
        if verified_bytes > MAX_ACTIVATION_BYTES:
            raise ActivationError(
                "published admission assets exceed the activation byte bound"
            )
    return {"assets": len(assets), "bytes": verified_bytes}


def _verified_bundle_file(
    root: Path,
    relative: str,
    expected_digest: str,
    expected_size: int,
    label: str,
) -> Path:
    relative = _relative_path(relative, label)
    path = root.joinpath(*PurePosixPath(relative).parts)
    if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
        raise ActivationError(f"{label} must be a regular in-bundle file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        declared = os.fstat(source.fileno()).st_size
        if declared != expected_size:
            raise ActivationError(f"{label} size does not match its activation claim")
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    if size != declared or digest.hexdigest() != expected_digest:
        raise ActivationError(f"{label} bytes do not match their activation hash")
    return path


def _validate_promotion(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ActivationError("profile_source_promotion must be an object")
    _exact_keys(value, PROMOTION_FIELDS, "profile_source_promotion")
    if value["path"] != PROFILE_SOURCE_PATH:
        raise ActivationError(
            f"profile_source_promotion.path must be {PROFILE_SOURCE_PATH!r}"
        )
    _digest(value["base_sha256"], "profile_source_promotion.base_sha256")
    _digest(value["active_sha256"], "profile_source_promotion.active_sha256")
    if (
        value["candidate_text"] != PROFILE_STATUS_CANDIDATE_TEXT
        or value["active_text"] != PROFILE_STATUS_ACTIVE_TEXT
    ):
        raise ActivationError(
            "profile_source_promotion does not bind the exact candidate-to-active constant"
        )
    return value


def _load_activation(path: Path) -> tuple[dict[str, Any], Path, str]:
    if path.is_symlink() or not path.is_file():
        raise ActivationError("activation manifest must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    raw = _read_bounded(
        resolved, MAX_ACTIVATION_MANIFEST_BYTES, "activation manifest"
    )
    digest = _sha256(raw)
    if resolved.name != f"{digest}.activation.json":
        raise ActivationError(
            "activation manifest filename must equal the SHA-256 of its exact bytes"
        )
    activation = _parse_json(raw, "activation manifest")
    _exact_keys(activation, ACTIVATION_FIELDS, "activation manifest")
    if activation["schema_version"] != 1:
        raise ActivationError("activation schema_version must be 1")
    if activation["status"] != "release-ready-not-published":
        raise ActivationError("activation is not release-ready-not-published")
    for field in (
        "stage_id",
        "base_registry_sha256",
        "base_variants_sha256",
        "admission_implementation_bundle_sha256",
        "compatibility_report_sha256",
        "registry_sha256",
    ):
        _digest(activation[field], field)
    if (
        not isinstance(activation["release_tag"], str)
        or RELEASE_TAG_RE.fullmatch(activation["release_tag"]) is None
        or activation["release_tag"].lower() in {"latest", "current", "draft"}
    ):
        raise ActivationError("activation release_tag is not canonical")
    _validate_promotion(activation["profile_source_promotion"])
    if activation["registry"] != REGISTRY_PATH:
        raise ActivationError(f"activation registry must be {REGISTRY_PATH!r}")
    report_digest = activation["compatibility_report_sha256"]
    expected_report = f"release/admission/{report_digest}.json"
    if activation["compatibility_report"] != expected_report:
        raise ActivationError(
            "activation compatibility report path must be its digest-named release path"
        )
    _positive_size(activation["registry_bytes"], "registry", MAX_REGISTRY_BYTES)
    _positive_size(
        activation["compatibility_report_bytes"],
        "compatibility report",
        MAX_REPORT_BYTES,
    )
    return activation, resolved.parent, digest


def _validate_inventory(
    activation: dict[str, Any], bundle_root: Path, manifest_name: str
) -> tuple[Path, Path]:
    expected_paths = {
        manifest_name,
        activation["registry"],
        activation["compatibility_report"],
    }
    total_bytes = (
        activation["registry_bytes"] + activation["compatibility_report_bytes"]
    )
    assets = activation["assets"]
    if not isinstance(assets, list) or not assets:
        raise ActivationError("activation assets must be a nonempty array")
    for index, asset in enumerate(assets):
        label = f"assets[{index}]"
        if not isinstance(asset, dict):
            raise ActivationError(f"{label} must be an object")
        _exact_keys(asset, ASSET_FIELDS, label)
        digest = _digest(asset["sha256"], f"{label}.sha256")
        _slug(asset["owner_id"], f"{label}.owner_id")
        if asset["kind"] not in {
            "admission-cache",
            "measurement-evidence",
            "target-package",
        }:
            raise ActivationError(f"{label}.kind is invalid")
        if asset["format"] not in {"npz", "zip"}:
            raise ActivationError(f"{label}.format is invalid")
        filename = f"{digest}.{asset['format']}"
        if asset["filename"] != filename or asset["path"] != f"assets/{filename}":
            raise ActivationError(f"{label} is not canonically content-addressed")
        size = _positive_size(asset["bytes"], label)
        total_bytes += size
        if total_bytes > MAX_ACTIVATION_BYTES:
            raise ActivationError("activation bundle exceeds its total byte bound")
        if asset["path"] in expected_paths:
            raise ActivationError("activation inventory repeats a path")
        expected_paths.add(asset["path"])
        _verified_bundle_file(bundle_root, asset["path"], digest, size, label)

    receipts = activation["receipts"]
    if not isinstance(receipts, list) or not receipts:
        raise ActivationError("activation receipts must be a nonempty array")
    receipt_pairs: set[tuple[str, str]] = set()
    for index, receipt in enumerate(receipts):
        label = f"receipts[{index}]"
        if not isinstance(receipt, dict):
            raise ActivationError(f"{label} must be an object")
        _exact_keys(receipt, RECEIPT_FIELDS, label)
        package_id = _slug(receipt["package_id"], f"{label}.package_id")
        scope_id = _slug(receipt["scope_id"], f"{label}.scope_id")
        pair = (package_id, scope_id)
        if pair in receipt_pairs:
            raise ActivationError("activation repeats a package/scope receipt")
        receipt_pairs.add(pair)
        digest = _digest(receipt["receipt_sha256"], f"{label}.receipt_sha256")
        expected = f"receipts/{digest}.json"
        if receipt["receipt"] != expected:
            raise ActivationError(f"{label} path is not content-addressed")
        size = _positive_size(receipt["receipt_bytes"], label)
        total_bytes += size
        if total_bytes > MAX_ACTIVATION_BYTES:
            raise ActivationError("activation bundle exceeds its total byte bound")
        if expected in expected_paths:
            raise ActivationError("activation inventory repeats a path")
        expected_paths.add(expected)
        _verified_bundle_file(bundle_root, expected, digest, size, label)

    actual_paths: set[str] = set()
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise ActivationError("activation bundle must not contain symlinks")
        if path.is_file():
            relative = path.relative_to(bundle_root).as_posix()
            actual_paths.add(relative)
            if len(actual_paths) > MAX_BUNDLE_FILES:
                raise ActivationError("activation bundle contains too many files")
        elif not path.is_dir():
            raise ActivationError("activation bundle contains a special filesystem entry")
    if actual_paths != expected_paths:
        raise ActivationError(
            "activation bundle inventory mismatch; "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"unexpected={sorted(actual_paths - expected_paths)}"
        )

    registry = _verified_bundle_file(
        bundle_root,
        activation["registry"],
        activation["registry_sha256"],
        activation["registry_bytes"],
        "active registry",
    )
    report = _verified_bundle_file(
        bundle_root,
        activation["compatibility_report"],
        activation["compatibility_report_sha256"],
        activation["compatibility_report_bytes"],
        "compatibility report",
    )
    return registry, report


def _repository_file(repository: Path, relative: str, label: str) -> Path:
    path = repository.joinpath(*PurePosixPath(relative).parts)
    if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
        raise ActivationError(f"{label} must be a regular checkout file")
    return path


def _validate_registry_transition(
    base_bytes: bytes,
    active_bytes: bytes,
    report_path: str,
    report_digest: str,
) -> None:
    base = _parse_json(base_bytes, "base registry")
    active = _parse_json(active_bytes, "active registry")
    if (
        base.get("profile_status") != "candidate"
        or base.get("admitted_backends") != []
        or base.get("local_packages") != []
    ):
        raise ActivationError("base registry is not the empty candidate boundary")
    if (
        active.get("profile_status") != "active"
        or active.get("profile_id") != "cfetch-embedding-v1"
        or not isinstance(active.get("admitted_backends"), list)
        or not active["admitted_backends"]
        or not isinstance(active.get("local_packages"), list)
        or not active["local_packages"]
    ):
        raise ActivationError("proposed registry is not a nonempty active cohort")
    if set(active) != set(base):
        raise ActivationError("active registry changed the top-level registry schema")
    mutable = {"profile_status", "admitted_backends", "local_packages"}
    if any(active[field] != base[field] for field in set(base).difference(mutable)):
        raise ActivationError("active registry changed fields outside the admission cohort")
    for index, entry in enumerate(active["admitted_backends"]):
        if (
            not isinstance(entry, dict)
            or entry.get("compatibility_report") != report_path
            or entry.get("compatibility_report_sha256") != report_digest
        ):
            raise ActivationError(
                f"active registry scope {index} does not bind the activation report"
            )


def _prepare_replacement(target: Path, data: bytes, mode: int) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, stat.S_IMODE(mode))
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _restore_file(target: Path, data: bytes, mode: int) -> None:
    temporary = _prepare_replacement(target, data, mode)
    os.replace(temporary, target)


def apply_activation(
    activation_manifest: Path,
    repository: Path,
    publication_opener: object | None = None,
) -> dict[str, object]:
    activation, bundle_root, activation_digest = _load_activation(activation_manifest)
    registry_source, report_source = _validate_inventory(
        activation, bundle_root, activation_manifest.resolve(strict=True).name
    )

    if repository.is_symlink() or not repository.is_dir():
        raise ActivationError("repository must be a real directory, not a symlink")
    repository = repository.resolve(strict=True)
    registry_target = _repository_file(repository, REGISTRY_PATH, "base registry")
    variants_target = _repository_file(repository, VARIANTS_PATH, "release variants")
    source_target = _repository_file(repository, PROFILE_SOURCE_PATH, "profile source")

    base_registry = _read_bounded(registry_target, MAX_REGISTRY_BYTES, "base registry")
    variants = _read_bounded(
        variants_target, MAX_ACTIVATION_MANIFEST_BYTES, "release variants"
    )
    source = _read_bounded(source_target, MAX_SOURCE_BYTES, "profile source")
    if _sha256(base_registry) != activation["base_registry_sha256"]:
        raise ActivationError("current base registry digest does not match activation")
    if _sha256(variants) != activation["base_variants_sha256"]:
        raise ActivationError("current release variants digest does not match activation")
    promotion = _validate_promotion(activation["profile_source_promotion"])
    if _sha256(source) != promotion["base_sha256"]:
        raise ActivationError("current profile source digest does not match activation")
    candidate = promotion["candidate_text"].encode("utf-8")
    active = promotion["active_text"].encode("utf-8")
    if source.count(candidate) != 1 or source.count(active) != 0:
        raise ActivationError(
            "profile source does not contain exactly the bound candidate status text"
        )
    active_source = source.replace(candidate, active, 1)
    if _sha256(active_source) != promotion["active_sha256"]:
        raise ActivationError("bound profile source transformation has drifted")

    active_registry = _read_bounded(
        registry_source, MAX_REGISTRY_BYTES, "active registry"
    )
    report = _read_bounded(report_source, MAX_REPORT_BYTES, "compatibility report")
    report_document = _parse_json(report, "compatibility report")
    if not isinstance(report_document.get("admission_gate"), dict) or (
        report_document["admission_gate"].get("passed") is not True
    ):
        raise ActivationError("compatibility report does not record a passed admission gate")
    _validate_registry_transition(
        base_registry,
        active_registry,
        activation["compatibility_report"],
        activation["compatibility_report_sha256"],
    )
    publication_verification = _verify_published_assets(
        activation, publication_opener
    )

    report_target = repository.joinpath(
        *PurePosixPath(activation["compatibility_report"]).parts
    )
    report_parent = report_target.parent
    created_report_parent = False
    if report_parent.exists():
        if report_parent.is_symlink() or not report_parent.is_dir():
            raise ActivationError("release admission destination is not a real directory")
    else:
        release_directory = repository / "release"
        if release_directory.is_symlink() or not release_directory.is_dir():
            raise ActivationError("release destination is not a real directory")
        report_parent.mkdir(mode=0o755)
        created_report_parent = True
    if report_target.exists() or report_target.is_symlink():
        if created_report_parent:
            try:
                report_parent.rmdir()
            except OSError:
                pass
        raise ActivationError("refusing to overwrite an existing compatibility report")

    registry_mode = registry_target.stat().st_mode
    source_mode = source_target.stat().st_mode
    registry_temp: Path | None = None
    source_temp: Path | None = None
    report_temp: Path | None = None
    report_installed = False
    registry_installed = False
    source_installed = False
    try:
        registry_temp = _prepare_replacement(
            registry_target, active_registry, registry_mode
        )
        source_temp = _prepare_replacement(source_target, active_source, source_mode)
        report_temp = _prepare_replacement(report_target, report, 0o644)
        os.link(report_temp, report_target)
        report_installed = True
        report_temp.unlink()
        report_temp = None
        os.replace(registry_temp, registry_target)
        registry_temp = None
        registry_installed = True
        os.replace(source_temp, source_target)
        source_temp = None
        source_installed = True

        if (
            _sha256(_read_bounded(registry_target, MAX_REGISTRY_BYTES, "installed registry"))
            != activation["registry_sha256"]
            or _sha256(_read_bounded(report_target, MAX_REPORT_BYTES, "installed report"))
            != activation["compatibility_report_sha256"]
            or _sha256(_read_bounded(source_target, MAX_SOURCE_BYTES, "promoted source"))
            != promotion["active_sha256"]
            or _sha256(
                _read_bounded(
                    variants_target,
                    MAX_ACTIVATION_MANIFEST_BYTES,
                    "release variants",
                )
            )
            != activation["base_variants_sha256"]
        ):
            raise ActivationError("post-apply verification did not match activation")
    except Exception:
        if source_installed:
            _restore_file(source_target, source, source_mode)
        if registry_installed:
            _restore_file(registry_target, base_registry, registry_mode)
        if report_installed:
            report_target.unlink(missing_ok=True)
        for temporary in (registry_temp, source_temp, report_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        registry_temp = source_temp = report_temp = None
        if created_report_parent and report_parent.exists():
            try:
                report_parent.rmdir()
            except OSError:
                pass
        raise
    finally:
        for temporary in (registry_temp, source_temp, report_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    return {
        "status": "applied",
        "activation_sha256": activation_digest,
        "stage_id": activation["stage_id"],
        "registry": REGISTRY_PATH,
        "registry_sha256": activation["registry_sha256"],
        "compatibility_report": activation["compatibility_report"],
        "compatibility_report_sha256": activation[
            "compatibility_report_sha256"
        ],
        "profile_source": PROFILE_SOURCE_PATH,
        "profile_source_sha256": promotion["active_sha256"],
        "variants_sha256": activation["base_variants_sha256"],
        "published_assets_verified": publication_verification["assets"],
        "published_asset_bytes_verified": publication_verification["bytes"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-manifest", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = apply_activation(args.activation_manifest, args.repository)
    except (ActivationError, OSError) as error:
        print(f"admission activation refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
