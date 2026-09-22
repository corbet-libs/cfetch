#!/usr/bin/env bash
# Resolve release inputs on the build worker; never edit the operator's checkout.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CFETCH_RELEASE_INPUTS_OUTPUT:?new retained output directory required}"
mkdir "$CFETCH_RELEASE_INPUTS_OUTPUT"
cargo update
cargo deny check advisories licenses
bash scripts/generate-third-party-licenses.sh "$CFETCH_RELEASE_INPUTS_OUTPUT/THIRD-PARTY-LICENSES.txt"
cp Cargo.lock "$CFETCH_RELEASE_INPUTS_OUTPUT/Cargo.lock"
date -u +%FT%TZ > "$CFETCH_RELEASE_INPUTS_OUTPUT/resolved-at"
rustc -vV > "$CFETCH_RELEASE_INPUTS_OUTPUT/compiler"
printf '%s\n' "$CI_COMMIT_SHA" > "$CFETCH_RELEASE_INPUTS_OUTPUT/source-commit"
