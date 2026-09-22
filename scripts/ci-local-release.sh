#!/usr/bin/env bash
set -euo pipefail
: "${CFETCH_LOCAL_RELEASE_OUTPUT:?new artifact directory required}"
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo build --release --locked --features embedded-embeddings
if [[ -n "${CFETCH_TEST_LOCAL_MODEL:-}" ]]; then
  cargo test --release --locked --features embedded-embeddings --test local_memory -- --ignored --nocapture
fi
mkdir "$CFETCH_LOCAL_RELEASE_OUTPUT"
install -m755 "${CARGO_TARGET_DIR:-target}/release/cfetch" "$CFETCH_LOCAL_RELEASE_OUTPUT/cfetch"
cp LICENSE.md THIRD-PARTY-LICENSES.txt "$CFETCH_LOCAL_RELEASE_OUTPUT/"
(
  cd "$CFETCH_LOCAL_RELEASE_OUTPUT"
  sha256sum cfetch > SHA256SUMS
  printf '%s\n' "$CI_COMMIT_SHA" > source-commit
  ./cfetch --version
  ldd ./cfetch
)
