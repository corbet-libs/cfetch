#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Evaluate source-change reuse and feature invalidation without compiling a
# second package. Only the final package in ci-nix.sh performs the full build.
trial=$(mktemp -d)
trap 'rm -rf -- "$trial"' EXIT
cp -R . "$trial/source"
flake="path:$trial/source"
eval_path() {
  nix eval --no-write-lock-file --raw "$flake#$1.drvPath"
}
before_deps=$(eval_path cfetch.cargoArtifacts)
before_package=$(eval_path cfetch)
printf '\n// Nix dependency-cache source-change probe.\n' >> "$trial/source/src/main.rs"
after_deps=$(eval_path cfetch.cargoArtifacts)
after_package=$(eval_path cfetch)
[[ "$before_deps" == "$after_deps" ]]
[[ "$before_package" != "$after_package" ]]
python3 - "$trial/source/nix/package.nix" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
source = path.read_text()
needle = 'cargoExtraArgs = "--locked"'
assert source.count(needle) == 1
path.write_text(source.replace(needle, 'cargoExtraArgs = "--locked --features native-openvino"'))
PY
feature_deps=$(eval_path cfetch.cargoArtifacts)
[[ "$before_deps" != "$feature_deps" ]]
printf 'Dependency cache reused after source edit: %s\n' "$before_deps"
printf 'Feature change invalidated dependency cache: %s\n' "$feature_deps"
