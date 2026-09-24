# First Intel OpenVINO cohort: operator runbook

This is the exact post-build chain for the first Lunar Lake NPU/GPU/CPU
cohort. It never treats conversion smoke output, host diagnostics, or a copied
JSON claim as physical admission evidence. Commands are run from the repository
root unless stated otherwise.

## 1. Accept one successful build artifact

Use only a successful `OpenVINO pinned package inputs` run whose `headSha` is
the exact reviewed checkout. The GitHub artifact expires after seven days.

```bash
set -euo pipefail
export CFETCH_RUN_ID="<successful-run-id>"
export CFETCH_COMMIT="$(git rev-parse HEAD)"
export CFETCH_OPS_ROOT="$PWD/results/openvino-$CFETCH_RUN_ID"
test ! -e "$CFETCH_OPS_ROOT"
mkdir -p "$CFETCH_OPS_ROOT/download" "$CFETCH_OPS_ROOT/artifact" \
  "$CFETCH_OPS_ROOT/runtime"

gh run view "$CFETCH_RUN_ID" --repo corbet-libs/cfetch \
  --json conclusion,headSha,workflowName > "$CFETCH_OPS_ROOT/run.json"
jq -e --arg commit "$CFETCH_COMMIT" '
  .conclusion == "success" and
  .headSha == $commit and
  .workflowName == "OpenVINO pinned package inputs"
' "$CFETCH_OPS_ROOT/run.json" >/dev/null

gh run download "$CFETCH_RUN_ID" --repo corbet-libs/cfetch \
  --name "openvino-pinned-inputs-$CFETCH_RUN_ID" \
  --dir "$CFETCH_OPS_ROOT/download"
(cd "$CFETCH_OPS_ROOT/download" && sha256sum --check SHA256SUMS)
jq -e --arg commit "$CFETCH_COMMIT" '.repository_commit == $commit' \
  "$CFETCH_OPS_ROOT/download/openvino-build-metadata.json" >/dev/null

artifact_archive="$(jq -er '.path | split("/")[-1]' \
  "$CFETCH_OPS_ROOT/download/artifact-archive-result.json")"
runtime_archive="$(jq -er '.path | split("/")[-1]' \
  "$CFETCH_OPS_ROOT/download/runtime-archive-result.json")"
test "$(sha256sum "$CFETCH_OPS_ROOT/download/$artifact_archive" | cut -d' ' -f1)" = \
  "$(jq -er .sha256 "$CFETCH_OPS_ROOT/download/artifact-archive-result.json")"
test "$(sha256sum "$CFETCH_OPS_ROOT/download/$runtime_archive" | cut -d' ' -f1)" = \
  "$(jq -er .sha256 "$CFETCH_OPS_ROOT/download/runtime-archive-result.json")"

tar --extract --gzip --file "$CFETCH_OPS_ROOT/download/$artifact_archive" \
  --directory "$CFETCH_OPS_ROOT/artifact" --no-same-owner
tar --extract --gzip --file "$CFETCH_OPS_ROOT/download/$runtime_archive" \
  --directory "$CFETCH_OPS_ROOT/runtime" --no-same-owner
export RUNTIME_MANIFEST_SHA256="$(jq -er .runtime_manifest_sha256 \
  "$CFETCH_OPS_ROOT/download/runtime-result.json")"
export ARTIFACT_MANIFEST_SHA256="$(jq -er .artifact_sha256 \
  "$CFETCH_OPS_ROOT/download/conversion-result.json")"
test "$(sha256sum "$CFETCH_OPS_ROOT/runtime/runtime-manifest.json" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
test "$(sha256sum "$CFETCH_OPS_ROOT/artifact/artifact-manifest.json" | cut -d' ' -f1)" = \
  "$ARTIFACT_MANIFEST_SHA256"
"$CFETCH_OPS_ROOT/runtime/cfetch-openvino-adapter-runtime" runtime-check \
  > "$CFETCH_OPS_ROOT/raw-runtime-check.json"
jq -e --arg digest "$RUNTIME_MANIFEST_SHA256" \
  '.runtime_manifest_sha256 == $digest' \
  "$CFETCH_OPS_ROOT/raw-runtime-check.json" >/dev/null
```

A run that fails before `Upload content-addressed build inputs` has no usable
partial artifact.

## 2. Create frozen inputs that are independent of hardware evidence

Install the hash-locked policy environment under CPython 3.12. Do not use an
arbitrary host Python or dependency version.

```bash
python3.12 -m venv "$CFETCH_OPS_ROOT/venv"
"$CFETCH_OPS_ROOT/venv/bin/python" -m pip install \
  --disable-pip-version-check --require-hashes --only-binary=:all: \
  -r experiments/embedding-profile/requirements-lock.txt

"$CFETCH_OPS_ROOT/venv/bin/python" \
  experiments/embedding-profile/scifact_contract.py \
  --wire-inputs-output "$CFETCH_OPS_ROOT/wire-probe-inputs.json" \
  > "$CFETCH_OPS_ROOT/wire-inputs-result.json"
test "$(sha256sum "$CFETCH_OPS_ROOT/wire-probe-inputs.json" | cut -d' ' -f1)" = \
  "$(jq -er .wire_inputs_sha256 "$CFETCH_OPS_ROOT/wire-inputs-result.json")"

"$CFETCH_OPS_ROOT/venv/bin/python" \
  experiments/embedding-profile/openvino_scope_keys.py \
  --scope-id intel-lnl-npu \
  --scope-id intel-lnl-gpu \
  --scope-id intel-lnl-cpu \
  --output-directory "$CFETCH_OPS_ROOT/operator" \
  > "$CFETCH_OPS_ROOT/scope-key-result.json"
```

Keep `scope-config.physical-probe.json` beside the generated key files. Insert
the public keys and relative private-key filenames from `scope-keys.json` and
use the exact schema in `packages/openvino/README.md`.

## 3. Run the frozen-runtime host preflight

Assembly requires exact values for all of these fields before the package can
start:

- Linux system, `x86_64` machine, exact kernel release, and one through sixteen
  explicitly selected regular driver/library paths plus their SHA-256 values;
- the exact normalized, regular, non-symlink resolutions of the target's
  `libstdc++.so.6` and `libgcc_s.so.1` in every NPU, GPU, and CPU host binding;
- NPU `FULL_DEVICE_NAME`, `DEVICE_ARCHITECTURE`, `NPU_DRIVER_VERSION`, and
  `NPU_COMPILER_VERSION` with exact JSON types;
- GPU `FULL_DEVICE_NAME`, `DEVICE_ARCHITECTURE`, `GPU_UARCH_VERSION`, and
  `GPU_DEVICE_ID` with exact JSON types;
- CPU `FULL_DEVICE_NAME` and `DEVICE_ARCHITECTURE` with exact JSON types;
- the exact compile configuration and compiled-model `EXECUTION_DEVICES` for
  every one of the seven static buckets on NPU, GPU, and CPU.

The unassembled native launcher intentionally has no package-inventory binding.
Invoke the raw PyInstaller dispatcher shown below; before reading the artifact
it verifies its complete `runtime-manifest.json`, CPython ABI, dependency
versions, glibc floor, and CPU plugin. It verifies the runtime and artifact
again after all compiles. The externally verified runtime-manifest digest is a
required input, so the dispatcher cannot authenticate a substituted runtime by
reference to its own adjacent manifest. System Python is not involved.

Select one through sixteen exact regular driver/library files for each scope.
Paths must already be normalized, absolute, non-symlink paths below an
allowlisted system library prefix. Selection is an explicit operator input; the
command hashes files but does not guess which libraries define the device. The
frozen runtime intentionally excludes `libstdc++.so.6` and `libgcc_s.so.1` so
it cannot shadow newer target drivers. Resolve both sonames to their real files
and include those two files in every scope; add the device-specific driver and
loader files for that scope.

```bash
HOST_LIBSTDCXX="<exact-normalized-regular-resolution-of-libstdc++.so.6>"
HOST_LIBGCC="<exact-normalized-regular-resolution-of-libgcc_s.so.1>"
declare -a CXX_RUNTIME_HOST_FILES=("$HOST_LIBSTDCXX" "$HOST_LIBGCC")
declare -a NPU_HOST_FILES=(
  "${CXX_RUNTIME_HOST_FILES[@]}"
  "<exact-npu-driver-or-library-path>"
)
declare -a GPU_HOST_FILES=(
  "${CXX_RUNTIME_HOST_FILES[@]}"
  "<exact-gpu-driver-or-library-path>"
)
declare -a CPU_HOST_FILES=(
  "${CXX_RUNTIME_HOST_FILES[@]}"
  "<exact-cpu-runtime-library-path>"
)

run_preflight() {
  local device_class="$1" device="$2" compile_config="$3" output="$4"
  shift 4
  local host_args=() path
  for path in "$@"; do host_args+=(--host-file "$path"); done
  "$CFETCH_OPS_ROOT/runtime/cfetch-openvino-adapter-runtime" host-preflight \
    --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
    --artifact-dir "$CFETCH_OPS_ROOT/artifact" \
    --artifact-manifest-sha256 "$ARTIFACT_MANIFEST_SHA256" \
    --device-class "$device_class" \
    --device "$device" \
    --compile-config-json "$compile_config" \
    "${host_args[@]}" > "$output"
}

run_preflight npu NPU '{}' "$CFETCH_OPS_ROOT/preflight-npu.json" \
  "${NPU_HOST_FILES[@]}"
run_preflight gpu GPU '{}' "$CFETCH_OPS_ROOT/preflight-gpu.json" \
  "${GPU_HOST_FILES[@]}"
run_preflight cpu CPU '{}' "$CFETCH_OPS_ROOT/preflight-cpu.json" \
  "${CPU_HOST_FILES[@]}"

for result in "$CFETCH_OPS_ROOT"/preflight-{npu,gpu,cpu}.json; do
  jq -e --arg runtime "$RUNTIME_MANIFEST_SHA256" \
    --arg artifact "$ARTIFACT_MANIFEST_SHA256" '
    .purpose == "physical-probe-scope-config-input-not-admission-evidence" and
    .runtime_manifest_sha256 == $runtime and
    .artifact_manifest_sha256 == $artifact and
    .openvino_property_device == .required_execution_devices[0] and
    .host_binding_source ==
      "operator-selected-paths-sha256-before-and-after-compilation" and
    (.bucket_results | map(.bucket) == [32,64,128,257,512,1024,2048])
  ' "$result" >/dev/null
done
```

Each stdout file is one bounded canonical JSON document. Copy its
`openvino_compile_config`, `required_openvino_properties`,
`required_execution_devices`, and `required_host` fields verbatim into the
matching scope. Its runtime/artifact digests and dependency versions must equal
the already verified build inputs. `bucket_results` proves that every static
compile reported that same singleton `EXECUTION_DEVICES`; it is configuration
provenance, not admission evidence. Do not guess values, use another OpenVINO
installation, or label preflight output as physical evidence.

## 4. Assemble and collect the physical probe

```bash
"$CFETCH_OPS_ROOT/venv/bin/python" packages/openvino/assemble.py \
  --artifact-dir "$CFETCH_OPS_ROOT/artifact" \
  --runtime-dir "$CFETCH_OPS_ROOT/runtime" \
  --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --scope-config "$CFETCH_OPS_ROOT/operator/scope-config.physical-probe.json" \
  --output-dir "$CFETCH_OPS_ROOT/physical-probe" \
  > "$CFETCH_OPS_ROOT/physical-probe-result.json"

export PROBE_MANIFEST_SHA256="$(jq -er .package_manifest_sha256 \
  "$CFETCH_OPS_ROOT/physical-probe-result.json")"
export DISPATCHER_SHA256="$(jq -er .dispatcher_sha256 \
  "$CFETCH_OPS_ROOT/physical-probe-result.json")"
```

Run the following on the physical Lunar Lake host from the same repository
commit and hash-locked environment. Transfer the complete probe directory and
wire manifest without changing bytes, then recheck the two digests above. Never
record a private host alias or address in evidence output.

```bash
mkdir -p "$CFETCH_OPS_ROOT/evidence"
for scope_id in intel-lnl-npu intel-lnl-gpu intel-lnl-cpu; do
  output="$CFETCH_OPS_ROOT/evidence/$scope_id"
  "$CFETCH_OPS_ROOT/venv/bin/python" \
    experiments/embedding-profile/physical_evidence.py \
    --dispatcher "$CFETCH_OPS_ROOT/physical-probe/cfetch-openvino-adapter" \
    --dispatcher-sha256 "$DISPATCHER_SHA256" \
    --package-manifest "$CFETCH_OPS_ROOT/physical-probe/package-manifest.json" \
    --package-manifest-sha256 "$PROBE_MANIFEST_SHA256" \
    --scope-id "$scope_id" \
    --wire-inputs "$CFETCH_OPS_ROOT/wire-probe-inputs.json" \
    --energy-not-measured-reason \
      "No device-scoped physical meter was available" \
    --output-directory "$output" \
    > "$CFETCH_OPS_ROOT/$scope_id-collector-result.json"
done
```

Each run owns the exact dispatcher process, verifies every signed response, and
retains one distinct content-addressed raw signed-transaction record for each
wire grouping plus the per-bucket profiler and benchmark records. A `503`,
fallback, property mismatch, host drift, signature failure, or missing RSS
measurement is a failure, not partial success.

## 5. Reassemble the evidence-bound candidate and export caches

Create the candidate config by changing only `package_state` and the three
per-scope evidence bindings. Keep every report binding `null`.

```bash
jq -s 'map({key:.scope_id,value:.}) | from_entries' \
  "$CFETCH_OPS_ROOT"/*-collector-result.json \
  > "$CFETCH_OPS_ROOT/evidence-bindings.json"
jq --slurpfile bindings "$CFETCH_OPS_ROOT/evidence-bindings.json" '
  .package_state = "candidate" |
  .scopes |= map(
    . as $scope | $bindings[0][$scope.scope_id] as $e |
    .placement_evidence_sha256 = $e.placement_evidence_sha256 |
    .sequence_capability_evidence_sha256 = $e.sequence_capability_evidence_sha256 |
    .performance_evidence_sha256 = $e.performance_evidence_sha256 |
    .compatibility_report_sha256 = null
  )
' "$CFETCH_OPS_ROOT/operator/scope-config.physical-probe.json" \
  > "$CFETCH_OPS_ROOT/operator/scope-config.candidate.json"

"$CFETCH_OPS_ROOT/venv/bin/python" packages/openvino/assemble.py \
  --artifact-dir "$CFETCH_OPS_ROOT/artifact" \
  --runtime-dir "$CFETCH_OPS_ROOT/runtime" \
  --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --scope-config "$CFETCH_OPS_ROOT/operator/scope-config.candidate.json" \
  --output-dir "$CFETCH_OPS_ROOT/candidate" \
  > "$CFETCH_OPS_ROOT/candidate-result.json"
```

The exporter needs the candidate dispatcher alive with its stdin pipe held
open. This Bash supervisor keeps the bearer only in memory and derives every
CLI identity from the candidate manifest:

```bash
candidate_manifest="$CFETCH_OPS_ROOT/candidate/package-manifest.json"
bearer="$("$CFETCH_OPS_ROOT/venv/bin/python" -c \
  'import secrets; print(secrets.token_hex(32))')"
coproc ADAPTER {
  cd "$CFETCH_OPS_ROOT/candidate"
  exec ./cfetch-openvino-adapter serve --host 127.0.0.1 --port 0 --auth-stdin
}
adapter_stdout_fd="${ADAPTER[0]}"
adapter_stdin_fd="${ADAPTER[1]}"
printf '{"bearer":"%s"}\n' "$bearer" >&"$adapter_stdin_fd"
IFS= read -r readiness <&"$adapter_stdout_fd"
expected_scope_ids="$(jq -c '[.scopes[].scope_id]' "$candidate_manifest")"
endpoint="$(jq -er --argjson expected "$expected_scope_ids" '
  select(.schema_version == 1 and .scope_ids == $expected) |
  .url + "/embeddings"
' <<<"$readiness")"
export CFETCH_ADAPTER_BEARER="$bearer"
mkdir -p "$CFETCH_OPS_ROOT/caches"

for scope_id in intel-lnl-npu intel-lnl-gpu intel-lnl-cpu; do
  scope="$(jq -ec --arg id "$scope_id" '.scopes[] | select(.scope_id == $id)' \
    "$candidate_manifest")"
  result="$CFETCH_OPS_ROOT/$scope_id-collector-result.json"
  args=()
  while IFS= read -r bucket; do
    args+=(--supported-sequence-bucket "$bucket")
  done < <(jq -r '.supported_sequence_buckets[]' <<<"$scope")
  "$CFETCH_OPS_ROOT/venv/bin/python" \
    experiments/embedding-profile/export_adapter_cache.py \
    --endpoint "$endpoint" \
    --output "$CFETCH_OPS_ROOT/caches/$scope_id.npz" \
    --scope-id "$scope_id" \
    --transport supervised-local \
    --backend "$(jq -er .backend <<<"$scope")" \
    --runtime "$(jq -er .runtime <<<"$scope")" \
    --compiler "$(jq -er .compiler <<<"$scope")" \
    --package-target "$(jq -er .package_target <<<"$scope")" \
    --artifact-source "$(jq -er .artifact_source <<<"$scope")" \
    --artifact-sha256 "$(jq -er .artifact_sha256 <<<"$scope")" \
    --attestation-public-key "$(jq -er .attestation_public_key <<<"$scope")" \
    --internal-precision "$(jq -er .internal_precision <<<"$scope")" \
    --supported-max-tokens "$(jq -er .supported_max_tokens <<<"$scope")" \
    "${args[@]}" \
    --sequence-capability-evidence "$(jq -er .sequence_capability_evidence "$result")" \
    --sequence-capability-evidence-sha256 \
      "$(jq -er .sequence_capability_evidence_sha256 "$result")" \
    --device "$(jq -er .device <<<"$scope")" \
    --device-class "$(jq -er .device_class <<<"$scope")" \
    --placement-evidence "$(jq -er .placement_evidence "$result")" \
    --placement-evidence-sha256 "$(jq -er .placement_evidence_sha256 "$result")" \
    --performance-evidence "$(jq -er .performance_evidence "$result")" \
    --performance-evidence-sha256 "$(jq -er .performance_evidence_sha256 "$result")" \
    --accelerated-placement --batch-size 64 \
    --scifact-snapshot "$CFETCH_OPS_ROOT/scifact" \
    --bearer-token-env CFETCH_ADAPTER_BEARER
done

unset CFETCH_ADAPTER_BEARER bearer
exec {adapter_stdin_fd}>&-
exec {adapter_stdout_fd}<&-
wait "$ADAPTER_PID"
```

## 6. Stage candidate, report, and release package

Generate the separate raw-binary receipt key; do not reuse package keys:

```bash
"$CFETCH_OPS_ROOT/venv/bin/python" \
  experiments/embedding-profile/admission_transaction.py keygen \
  --output "$CFETCH_OPS_ROOT/receipt-attestation.key" \
  > "$CFETCH_OPS_ROOT/receipt-public-key.txt"
```

Build `admission-transaction.json` using the exact schema in `README.md`. Use:

- release variant `linux-cfetch-local-intel-lunar-lake-x86_64`;
- dispatcher basename `cfetch-openvino-adapter` and the candidate assembly's
  `dispatcher_sha256`;
- package format `zip`;
- one scope row per cache and its matching physical raw directory;
- the candidate package directory and manifest;
- the exact current `release/inference-backends.json` and
  `release/variants.json` digests;
- `scifact_snapshot` as the normalized manifest-relative path to the exact
  pinned raw SciFact directory used by every cache export;
- the public key printed by `keygen` and a fixed, not previously mutated,
  release tag.

Run the complete local boundary once on the physical Lunar Lake host:

```bash
"$CFETCH_OPS_ROOT/venv/bin/python" \
  experiments/embedding-profile/admission_transaction.py run \
  --manifest "$CFETCH_OPS_ROOT/admission-transaction.json" \
  --receipt-attestation-private-key \
    "$CFETCH_OPS_ROOT/receipt-attestation.key" \
  --output "$CFETCH_OPS_ROOT/complete-admission"
```

The command replays the current admitted registry, reconstructs the measured
probe manifest, verifies the manifest-bound SciFact raw snapshot against its
three frozen file digests, runs the full all-pairs gate, injects the report
binding, builds the exact target ZIP, launches it for every NPU/GPU/CPU scope
against that same snapshot, retains all signed final-conformance receipts, and
creates the activation bundle. It then writes one digest-named
`*.publication.json` plan binding every upload, remote URL, repository byte,
and ordering constraint. A failure removes the partial output. The command
performs no upload or repository mutation and persists no local snapshot path.

## 7. Immutable publication and activation

The one-command transaction ends with a `ready-not-published` publication
plan; its activation manifest remains `release-ready-not-published`. Upload
every release asset named by the publication plan to its exact fixed
`release_tag`. There is no
repository-owned publisher for this external mutation, so it remains an
explicit operator boundary. Do not edit the plan, rename an asset, or replace
an asset at the tag.

Only after every immutable asset is remotely present may the checkout be
mutated:

```bash
"$CFETCH_OPS_ROOT/venv/bin/python" scripts/apply_admission_activation.py \
  --activation-manifest \
  "$CFETCH_OPS_ROOT/complete-admission/activation/<sha256>.activation.json" \
  --repository .
"$CFETCH_OPS_ROOT/venv/bin/python" \
  experiments/embedding-profile/cross_backend_eval.py \
  --verify-release-registry
```

The activation command itself downloads every planned asset through
credential-free GitHub HTTPS, constrains all redirects to GitHub, and verifies
the exact size and SHA-256 before writing the report, registry, or active
profile status. The following registry replay repeats the same bounded remote
verification and recomputes the global gate from the published evidence.

Any absent command, placeholder, failed scope, missing receipt, or unpublished
asset means stop. It is not permission to synthesize a value or weaken a gate.
