# Native package schema 2

This is the Rust serving contract for the still-unadmitted Gemma profile. It
changes packaging and internal worker/startup protocols, not vector semantics.
There are no old-layout, old-owner, or protocol compatibility fallbacks.

## Build and trust order

1. Build the native sibling with `native-openvino`, without the final client's
   `LocalPackagePlan`. Its hidden `native-serve` and `native-worker` commands own
   transport and governed native execution.
2. Assemble the sibling, canonical converted Gemma artifact, tokenizer/legal
   files, OpenVINO libraries/plugins, and distinct scope keys. Hash the complete
   immutable payload into `runtime-manifest.json`, then bind that manifest and
   the artifact/scopes in `package-manifest.json`.
3. Qualify the complete physical cohort and compile the final client with the
   exact package archive, root-manifest, dispatcher, scope, key and report
   digests. The existing admission gates remain mandatory.
4. Stage the exact archive as `inference/` beside the final `cfetch` executable.
   The archive itself contains payload-relative entries. Existing `inference`
   files, directories, partial payloads, and symlinks are rejected.

The final client is structurally outside the payload inventory. The independently
built sibling must not inventory or require byte identity with the final client;
that would introduce a manifest/binary hash cycle.

The final client validates the compiled plan, full file closure and admitted
scope attestations before creating its cached supervisor. Child startup receives
only private stdin JSON with exact fields `schema_version: 2`, `bearer`,
`package_manifest_sha256`, and `ordered_scope_ids`. The frame is at most 8192
bytes and the cohort at most 16 unique canonical scope slugs. The sibling checks
that permit against the same full closure, without consulting its own compiled
package plan. Direct invocation cannot admit responses into the final client's
compiled registry.

## Runtime manifest

The exact root fields are:

- `schema_version: 2`, `format: "cfetch-native-openvino-v2"`, and
  `target: "x86_64-unknown-linux-gnu"`;
- `source_revision` (40 lowercase hex), `cargo_lock_sha256`, `rustc`,
  `openvino_version`, and `openvino_build`;
- `dispatcher` (payload-root basename), `openvino_library` (relative path),
  `plugin_configuration: "single-device-absolute-library-v1"`, and `plugins`
  containing exactly `NPU`, `GPU`, and `CPU` relative library paths;
- `files`: a sorted unique array of exact `{path, sha256, bytes, executable}`
  records, covering every regular payload file except the two root manifests.

The root package manifest binds the runtime manifest's digest; the compiled plan
binds the root package manifest. Extra/missing files, duplicates, symlinks,
noncanonical paths, executable-bit changes and altered bytes are rejected. The
native loader caps the payload at 4096 files/4 GiB; the release archive stager's
existing tighter 2 GiB ceiling still applies.

## Package manifest and scope

The exact root fields are `schema_version: 2`, `package_state: "release"`,
`profile_id`, `profile_manifest_sha256`, `admission_policy_sha256`, `model`,
`model_revision`, `artifact_manifest`, `artifact_manifest_sha256`,
`runtime_manifest_sha256`, and `scopes`. All profile/source/policy fields must
match the frozen compiled Gemma contract. The canonical converted artifact's
schema and legal/source/tokenizer/pooling/dense/L2 checks remain unchanged.

Each scope retains its exact existing execution identity, evidence digests,
compatibility report, public/private attestation key, required host and physical
OpenVINO properties/device. `native_compile` replaces the unrestricted Python
compile map with exactly `precision` and `threads`. Gemma compute accepts F32 or
BF16; F16 is permitted only as artifact weight storage. Full 2048-token,
seven-bucket, 64-row cohort support remains required, with NPU/GPU/CPU scopes in
that order. No null evidence or candidate/probe scope enters serving.

Host bindings pin Linux/x86_64/kernel release and exact regular dependencies
under declared library directories (including Nix store paths), plus the
installed `/var/lib/cfetch/inference/policy.json`. Resolved `libstdc++.so.6` and
`libgcc_s.so.1` files are mandatory. Every executable library actually loaded by
the worker, including its loader/libc/driver dependencies, must also be bound.
All scopes use the same installed policy; no namespace or budget is provisioned
or reset by serving.

## Governed native boundary

Internal worker protocol 3 requires exact expected properties/execution device,
the complete file closure, and the chosen plugin library/hash. Under the compile
lease, the worker rehashes that closure and derives a private, single-device XML
configuration from the pinned absolute plugin path. `Core::new_with_config`
prevents ambient plugin XML selection. Loader override variables are rejected,
not silently removed or accepted through aliases.

On Linux, bounded `/proc/self/maps` checks bind executable mappings to the
expected file identity before/after native loading, compilation, and inference.
Unknown, deleted, replaced, or changed mapped dependencies are hard failures.
These checks detect an unexpected load; they are not a sandbox preventing an
ELF constructor from running. This Linux implementation is not a claim of
Windows, macOS, ARM, or physical device qualification.

The finite request deadline is derived from the pinned policy: seven compile
wait/budget/stop allowances, 64 inference wait/budget allowances, and bounded
serialization overhead. The parent adds only bounded transport overhead. Receipt
files live under the existing user's cfetch state owner; persistent governor
intent and its explicit recovery rules remain authoritative. A timeout, crash,
malformed response, invalid signature, or uncertain cleanup never causes CPU
fallback or an automatic native restart.
