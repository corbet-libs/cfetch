#!/usr/bin/env bash
set -euo pipefail

release=false
if [[ ${1:-} == --release ]]; then
  release=true
  shift
fi
catalog=${1:-release/variants.json}
registry=${2:-release/inference-backends.json}
if [[ $# -gt 2 || ( $release == false && $# -gt 1 ) ]]; then
  echo 'usage: variant-matrix.sh [catalog] | --release [catalog [registry]]' >&2
  exit 1
fi

# This is the no-compile gate used by both CI and the tag workflow. The Rust
# parser enforces the same product rules inside the executable; this shell
# boundary exists so a malformed catalog cannot even create a build matrix.
jq -e '
  .schema_version == 1 and
  (.variants | length > 0) and
  ([.variants[].id] | length == (unique | length)) and
  ([.variants[] | select(.backend == "endpoint") | [.os, .arch]] |
    length == (unique | length)) and
  all(.variants[];
    (.id | test("^[a-z0-9_-]+$")) and
    (.os == "linux" or .os == "mac" or .os == "win") and
    (.arch == "x86_64" or .arch == "aarch64") and
    (.runner | length > 0) and
    (.binary | length > 0) and
    (.archive == "tar.gz" or .archive == "zip") and
    ((.backend == "endpoint" and (.id | contains("-cfetch-remote-"))) or
     (.backend == "local" and (.id | contains("-cfetch-local-")))) and
    .cargo_features == "")
' "$catalog" >/dev/null

if [[ $release == false ]]; then
  # Build CI continues checking every candidate target before its admission.
  jq -c '{include: .variants}' "$catalog"
  exit 0
fi

# Selecting release targets must not hide an inconsistent active registry.
# Reuse the dependency-light payload validator; staging still downloads and
# verifies every selected local package before the release can be published.
script_directory=$(cd "$(dirname "$0")" && pwd)
python - "$catalog" "$registry" "$script_directory" <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[3])
from stage_local_inference import StagingError, _load_json, _validate_plan


def require(condition, message):
    if not condition:
        raise StagingError(message)


def release_matrix(catalog, registry):
    require(registry.get("schema_version") == 1, "unsupported inference registry schema")
    packages = registry.get("local_packages")
    scopes = registry.get("admitted_backends")
    require(isinstance(packages, list) and isinstance(scopes, list), "invalid inference registry arrays")
    require(bool(packages) == bool(scopes), "local packages and admitted backends must activate together")
    require(
        registry.get("profile_status") == ("active" if packages else "candidate"),
        "profile status does not match local package activation",
    )
    by_scope = {}
    for scope in scopes:
        require(isinstance(scope, dict), "admitted backend must be an object")
        scope_id = scope.get("scope_id")
        require(isinstance(scope_id, str) and bool(scope_id), "admitted backend has no scope id")
        require(scope_id not in by_scope, "duplicate admitted scope")
        require(scope.get("transport") in ("supervised-local", "remote-attested"), "invalid admitted transport")
        by_scope[scope_id] = scope

    variants = {row["id"]: row for row in catalog["variants"]}
    package_ids = set()
    active_variants = set()
    packaged_scopes = set()
    for package in packages:
        require(isinstance(package, dict), "local package must be an object")
        package_id = package.get("package_id")
        require(isinstance(package_id, str) and bool(package_id), "local package has no id")
        require(package_id not in package_ids, "duplicate local package id")
        package_ids.add(package_id)
        variant_id = package.get("release_variant_id")
        require(isinstance(variant_id, str) and variant_id in variants, "local package references an unknown variant")
        variant = variants[variant_id]
        require(variant["backend"] == "local", "local package references an endpoint variant")
        require(variant_id not in active_variants, "local variant has duplicate packages")
        require(
            (package.get("os"), package.get("arch")) == (variant["os"], variant["arch"]),
            "local package target does not match its variant",
        )
        _validate_plan(registry, catalog, variant_id)
        scope_ids = package.get("ordered_scope_ids")
        require(isinstance(scope_ids, list) and bool(scope_ids), "local package has no ordered scopes")
        require(all(isinstance(scope_id, str) for scope_id in scope_ids), "invalid package scope id")
        require(len(scope_ids) == len(set(scope_ids)), "duplicate package scope")
        classes = []
        for scope_id in scope_ids:
            require(scope_id in by_scope, "local package references an unadmitted scope")
            scope = by_scope[scope_id]
            require(scope.get("transport") == "supervised-local", "local package references a remote scope")
            require(scope.get("accelerated_placement") is True, "local package scope is not accelerated")
            device_class = scope.get("device_class")
            require(device_class in ("npu", "gpu", "cpu"), "invalid local device class")
            classes.append(("npu", "gpu", "cpu").index(device_class))
        require(classes == sorted(classes) and set(classes) == {0, 1, 2}, "local package lacks ordered NPU/GPU/CPU fallbacks")
        packaged_scopes.update(scope_ids)
        active_variants.add(variant_id)
    require(
        all(scope_id in packaged_scopes or scope["transport"] == "remote-attested" for scope_id, scope in by_scope.items()),
        "admitted local scope has no release package",
    )
    return {"include": [
        variant for variant in catalog["variants"]
        if variant["backend"] == "endpoint" or variant["id"] in active_variants
    ]}


try:
    matrix = release_matrix(
        _load_json(Path(sys.argv[1]), "release variant catalog"),
        _load_json(Path(sys.argv[2]), "inference registry"),
    )
except (OSError, ValueError, TypeError, KeyError) as error:
    raise SystemExit(f"release matrix refused: {error}") from error
print(json.dumps(matrix, separators=(",", ":")))
PY
