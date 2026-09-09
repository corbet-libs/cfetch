#!/usr/bin/env bash
# Provider-independent checks; run on a build worker with existing tools.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export RUST_TEST_THREADS="${RUST_TEST_THREADS:-2}"
python_bin="${CFETCH_POLICY_PYTHON:-python3}"

if (($# == 0)); then
  set -- catalog rust
fi
for check in "$@"; do
  case "$check" in
    catalog)
      bash scripts/check-packaging-variants.sh
      bash scripts/variant-matrix.sh >/dev/null
      "$python_bin" -m unittest -v scripts.test_stage_local_inference scripts.test_variant_matrix
      ;;
    rust)
      version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -1)
      case "$version" in
        0.*) ;;
        *) echo "cfetch 1.0+ is operator-blocked (Cargo.toml has $version)" >&2; exit 1 ;;
      esac
      rustc --version
      cargo test --locked
      cargo check --all-targets --all-features --locked
      cargo test --locked --features embedded-embeddings --test embed_model_cli
      cargo test --locked --features embedded-embeddings embedded_embed::tests
      cargo clippy --all-targets --all-features --locked -- -D warnings
      ;;
    variants)
      # Native Linux compilation only. This does not qualify any accelerator.
      [[ $(uname -s) == Linux ]] || { echo 'variants requires Linux' >&2; exit 2; }
      architecture=$(uname -m)
      matrix=$(bash scripts/variant-matrix.sh)
      selected=$(jq -r --arg arch "$architecture" '.include[] | select(.os == "linux" and .arch == $arch) | .id' <<<"$matrix")
      [[ -n $selected ]] || { echo 'No native Linux variant for this worker' >&2; exit 2; }
      while IFS= read -r variant; do
        CFETCH_VARIANT="$variant" cargo check --release --locked
      done <<<"$selected"
      ;;
    licenses)
      cargo deny check advisories licenses
      generated=$(mktemp)
      trap 'rm -f -- "$generated"' EXIT
      bash scripts/generate-third-party-licenses.sh "$generated"
      diff -u THIRD-PARTY-LICENSES.txt "$generated"
      ;;
    ort-foundation-cpu)
      "$python_bin" experiments/npu-ort-foundation/run_cached_cpu.py
      ;;
    governor)
      # Native deadline and persisted load policy; no model/runtime dependency.
      "$python_bin" -m unittest -v \
        packages.openvino.tests.test_inference_governor \
        packages.openvino.tests.test_native_deadline
      ;;
    profile)
      # The caller supplies the environment from requirements-lock.txt.
      # No dependency installation or physical-device probe occurs here.
      (
        cd experiments/embedding-profile
        "$python_bin" -m unittest -v \
          test_cross_backend_eval.py test_export_adapter_cache.py \
          test_kat_host_runner.py test_measurement_bundle.py \
          test_physical_evidence.py test_physical_checkpoint.py \
          test_openvino_scope_keys.py test_scifact_contract.py \
          test_final_package_conformance.py test_admission_transaction.py
        HF_HUB_DISABLE_TELEMETRY=1 "$python_bin" scifact_contract.py
        "$python_bin" cross_backend_eval.py --verify-implementation-bundle
        HF_HUB_DISABLE_TELEMETRY=1 "$python_bin" cross_backend_eval.py --verify-release-registry
      )
      "$python_bin" -m unittest discover -s packages/openvino/tests -v
      "$python_bin" -m unittest discover -s experiments/memory-retrieval -v
      "$python_bin" -m unittest scripts/test_apply_admission_activation.py -v
      for command in \
        experiments/embedding-profile/export_adapter_cache.py \
        experiments/embedding-profile/cross_backend_eval.py \
        experiments/embedding-profile/final_package_conformance.py \
        experiments/embedding-profile/admission_transaction.py \
        experiments/embedding-profile/physical_evidence.py \
        scripts/apply_admission_activation.py; do
        "$python_bin" "$command" --help >/dev/null
      done
      ;;
    *) echo "Unknown check: $check (catalog rust variants licenses governor profile ort-foundation-cpu)" >&2; exit 2 ;;
  esac
done
