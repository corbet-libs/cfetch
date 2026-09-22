#!/usr/bin/env bash
# Build-worker-only qualification. Model and reference paths are explicit inputs.
set -euo pipefail
: "${CFETCH_LOCAL_MODEL_SOURCE:?complete pinned snapshot required}"
: "${CFETCH_LOCAL_MODEL_OUTPUT:?new model pack directory required}"
: "${CFETCH_LOCAL_MODEL_REFERENCE:?canonical reference JSON required}"
python_bin="${CFETCH_POLICY_PYTHON:-python3}"
model_tools="${CFETCH_LOCAL_MODEL_OUTPUT}.build-tools"
if [[ ! -e "$model_tools/ready" ]]; then
  mkdir -p "$model_tools"
  PYTHONPATH="${CFETCH_PIP_BOOTSTRAP:-}" "$python_bin" -m pip install --disable-pip-version-check --only-binary=:all: --no-deps --target "$model_tools" \
    onnx==1.19.1 numpy==2.3.3 protobuf==6.32.1 typing_extensions==4.15.0 ml_dtypes==0.5.3
  touch "$model_tools/ready"
fi
if [[ ! -f "$CFETCH_LOCAL_MODEL_OUTPUT/manifest.json" ]]; then
  PYTHONPATH="$model_tools" "$python_bin" scripts/prepare-local-model.py "$CFETCH_LOCAL_MODEL_SOURCE" "$CFETCH_LOCAL_MODEL_OUTPUT"
fi
cargo build --locked --features embedded-embeddings
"${CARGO_TARGET_DIR:-target}/debug/cfetch" qualify-model "$CFETCH_LOCAL_MODEL_OUTPUT" "$CFETCH_LOCAL_MODEL_REFERENCE"
