#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CFETCH_NIX_OUTPUT:?persistent Nix output directory required}"
mkdir -p "$CFETCH_NIX_OUTPUT"
nix --version
nix build --no-write-lock-file --max-jobs "${CI_NIX_JOBS:-1}" --cores "${CI_JOBS:-2}" \
  --out-link "$CFETCH_NIX_OUTPUT/result" .#cfetch
binary="$CFETCH_NIX_OUTPUT/result/bin/cfetch"
"$binary" --version
"$binary" variants --json
"$binary" embed-model status --json
nix path-info --json "$CFETCH_NIX_OUTPUT/result" > "$CFETCH_NIX_OUTPUT/closure.json"
printf '%s\n' "$CI_COMMIT_SHA" > "$CFETCH_NIX_OUTPUT/source-commit"
