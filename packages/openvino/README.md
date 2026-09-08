# Intel OpenVINO target package

Status: build and physical-probe tooling, not an admitted backend.

This directory produces the first cfetch target package for one exactly
identified Intel NPU, GPU, and accelerated CPU cohort. The three scopes share
one converted EmbeddingGemma graph but are compiled, executed, and evidenced
independently. This is not a generic-Intel package and it does not activate a
scope in the shared vector space.

The package keeps these boundaries explicit:

- cfetch's portable boundary is signed `INT8x768`. OpenVINO executes the
  target-native internal precision recorded by each scope; internal INT8 is
  not assumed. The IR stores weights as F16, but that is an archive property,
  not a claim that every device executes one common F16 numeric path.
- the IR contains the pinned transformer, attention-mask-weighted mean pooling
  including the supplied prompt, the bias-free 768 -> 3072 and 3072 -> 768
  identity projections, and L2 normalization.
- each OpenVINO model is reshaped and compiled statically at 32, 64, 128, 257,
  512, 1,024, or 2,048 tokens. Input over 2,048 tokens is rejected; it is never
  truncated. The otherwise conventional 256 boundary is deliberately 257:
  the first Lunar Lake NPU cohort produced a deterministic semantic failure
  only for the exact 256-token static graph while adjacent 255, 257, 258, and
  all other profile shapes passed. The public shape is changed honestly rather
  than executing a hidden padded shape.
- a request names a package-admitted scope ID. The manifest alone maps that ID
  to exactly `NPU`, `GPU`, or `CPU`; `AUTO`, `MULTI`, `HETERO`, and implicit
  cross-device fallback are forbidden.
- NPU initialization remains lazy. A generic HTTP 503
  `scope_unavailable` response lets cfetch try the next independently admitted
  GPU or CPU scope without the adapter claiming where a failed request ran.

## Package states

The top-level `package_state` prevents the evidence bootstrap cycle:

| State | Three per-scope physical evidence digests | Compatibility report |
| --- | --- | --- |
| `physical-probe` | all explicitly `null` | `null` |
| `candidate` | all required | `null` |
| `release` | all required | required |

Only `physical-probe` may be used to collect the initial live device evidence.
It is deliberately unreleasable. The collector retains the probe package
manifest digest. Candidate construction may fill only the three evidence
bindings and state; release construction additionally fills the global report
binding. Admission tooling must reject pending bindings and prove that graph,
runtime, device, host, key, and all other immutable fields did not change.

## Reproducible build inputs

The manual `OpenVINO pinned package inputs` GitHub Actions workflow is the
supported build entry point. It runs on Ubuntu 22.04 x86_64 with CPython 3.12
and an exact glibc 2.35 build floor. It:

1. installs only binary wheels from `requirements-build.lock` with
   `--require-hashes`, using PyPI plus the official PyTorch CPU wheel index;
2. fetches the public immutable mirror `unsloth/embeddinggemma-300m` commit
   `bfa3c846ac738e62aa61806ef9112d34acb1dc5a`, whose 13 required files are
   byte-identical to the frozen `google/embeddinggemma-300m` commit
   `57c266a740f537b4dc058e1b0cda161fd15afa75`, and verifies every file against
   the canonical SHA-256 allowlist without sending credentials;
3. fetches, extracts, and hash-checks the pinned official Gemma Terms and
   Prohibited Use Policy;
4. converts the exact graph and runs CPU parity against an independent
   upstream PyTorch reference: short inputs, the maximum static shape, and
   prepared-token probes with 258 and 1,946 real tokens; conversion replaces
   exactly the two rotary-position MatMuls whose reduction dimension is one
   with the algebraically identical broadcast multiplication, refusing graph
   drift;
5. freezes the runtime, executes its integrity launcher and CPU plugin
   self-check, and emits regular-file-only content-addressed archives.

The fetch report and artifact manifest record both the canonical Google source
identity and the exact public acquisition commit. The mirror is transport,
not a new model identity: a single differing byte aborts the build. The parity
report is a broken-export smoke check, not compatibility admission evidence.

The equivalent local conversion commands, after obtaining the exact source
under the Gemma terms, are:

```console
python packages/openvino/legal.py fetch --output-dir "$LEGAL_DIR"
python packages/openvino/fetch_source.py \
  --output-dir "$PINNED_SOURCE_DIR" \
  --cache-dir "$HF_CACHE_DIR"
python packages/openvino/convert.py \
  --source-dir "$PINNED_SOURCE_DIR" \
  --legal-dir "$LEGAL_DIR" \
  --output-dir "$ARTIFACT_DIR" \
  --weight-storage f16
python packages/openvino/smoke_parity.py \
  --source-dir "$PINNED_SOURCE_DIR" \
  --artifact-dir "$ARTIFACT_DIR" \
  --output "$PARITY_REPORT"
```

`f16` compresses constants stored in the IR; it is not a cross-device
arithmetic claim. The converter verifies all source and semantic configuration
hashes before loading weights and proves that all seven required static shapes
can be formed before serialization. The unit-reduction rewrite avoids a GPU
shape-lowering defect without changing the model function: one multiplication
and no accumulation is exactly the original K=1 matrix product.

### Attention semantics and independent parity

The pinned source `config.json` contains `sliding_window: 512` and
`use_bidirectional_attention: true`. Locked Transformers 5.10.1 interprets
that serialized width in `Gemma3TextConfig` as `(512 // 2) + 1 = 257`. The
effective local mask therefore admits keys at exclusive distance `< 257`.
The converter verifies the raw value 512 and the loaded value 257 separately.
Using the serialized width directly as the effective radius changes long-input
embeddings. This attention parameter is independent of the profile's separate
257-token execution bucket.

`reference.py` loads a fresh upstream `AutoModel` with SDPA and an ordinary 2D
attention mask, then independently implements mean pooling, both verified
Dense projections and L2 normalization. It does not reuse the converter's
attention or pooling helpers. `smoke_parity.py` uses that reference and requires
finite 768-dimensional outputs, L2 norm error at most 0.005, and cosine at least
0.999. Its long probes contain 258 real tokens in bucket 512 and 1,946 in bucket
2,048, including BOS/EOS; their prepared-token and output digests are
recorded. A short input padded to a large shape does not exercise these long
attention neighborhoods. These structural probes are separate from retrieval
relevance labels and hardware admission.

The optional [native audit](../../experiments/memory-retrieval/audit_native.py)
also compares the actual pinned upstream mask factories with the converter
across every frozen bucket and boundary token count, before model
initialization. Real query rows must agree exactly and exclude padding keys;
only diagonal repairs to otherwise empty padded query rows may differ. It
then compares independent upstream, patched PyTorch and native OpenVINO CPU
outputs on selected evaluation-manifest inputs, with separate semantic and
export-parity results. Tokenizer equivalence is checked on the manifest's exact
prefixed inputs, including explicit canonical BOS/EOS handling.

## Gemma redistribution boundary

The converted IR is a Gemma Model Derivative. A distributable artifact must
contain all five exact files below, and `archive.py --require-gemma-legal`
fails closed if any byte differs:

- `GEMMA_TERMS.txt`
- `GEMMA_PROHIBITED_USE_POLICY.txt`
- `MODEL_USE_RESTRICTIONS.txt`
- `MODEL_MODIFICATIONS.txt`
- `NOTICE`

The payload includes the full pinned Agreement and Prohibited Use Policy, an
enforceable Section 3.2 pass-through restriction, prominent conversion
modification notice, and Google's mandated NOTICE sentence. Do not publish
weights or converted IR without this payload and the required downstream use
agreement.

## Frozen runtime and integrity launcher

Build the target runtime with:

```console
python packages/openvino/build_runtime.py \
  --output-dir "$RUNTIME_DIR" \
  --minimum-glibc 2.35 \
  --cc /usr/bin/cc
```

The result contains a native root executable `cfetch-openvino-adapter`, a
PyInstaller-frozen `cfetch-openvino-adapter-runtime`, CPython, and the exact
packaged `cryptography`, `numpy`, `openvino`, and `tokenizers` dependencies. It
does not depend on a target host's system Python. The bundle deliberately
excludes `libstdc++.so.6` and `libgcc_s.so.1`: those two libraries define the
ABI used by the installed accelerator drivers and must come from the target
system. Manifest creation and verification reject either soname at any bundle
depth so a build-host copy cannot shadow a newer host driver dependency.

Final assembly inventories every runtime, interpreter, native-library,
adapter, manifest, graph, tokenizer, legal, and key file. The inventory digest
is patched into the native root launcher; the launcher is the sibling binary
whose SHA-256 cfetch binds. Before Python or OpenVINO starts, that launcher
rejects missing, extra, symlinked, mode-changed, size-changed, or digest-changed
files. The frozen adapter verifies the same inventory again.

This relocation claim is intentionally narrow: Linux x86_64 with glibc 2.35 or
newer. Kernel drivers, firmware, Level Zero/OpenCL user-mode drivers, and
admitted CPU instruction support remain host prerequisites and are not
vendored. Compatible target-system `libstdc++.so.6` and `libgcc_s.so.1` are an
explicit prerequisite for every device class. Each scope therefore binds exact
OpenVINO properties, the observed `EXECUTION_DEVICES`, kernel release, and
hashes of the resolved regular files behind those two C++ runtime sonames plus
the relevant regular driver libraries. A driver, C++ runtime, or kernel change
requires recertification.

OpenVINO IR is shipped instead of a compiled blob because compiled blobs are
not stable across OpenVINO/device versions. The exact runtime compiles the IR
on the admitted host; any generated cache is disposable.

## Host governor provisioning and deadlines

Serving and physical preflight require an externally provisioned Linux governor
at the fixed path `/var/lib/cfetch/inference`. Every participating adapter
process, user and container namespace must reach the same host-local directory
and lock inode there, with the same host boot identity and monotonic clock.
There is no per-user, per-package or per-device fallback
namespace. `serve` and `host-preflight` inspect this state before native imports;
the build-only `runtime-check` does not require a host governor installation.

Provision all four fixed files before starting adapters:

| File | Contract |
| --- | --- |
| `policy.json` | Exact pinned policy bytes; writable only by the installation owner. |
| `operation.lock` | Stable inode used for exclusive cross-process `flock`. |
| `state.json` | Valid state bound to the exact policy digest, explicit epoch and current Linux boot ID. |
| `intent.json` | Empty at initial provisioning; carries durable pending native work. |

The directory and every ancestor up to `/` must be root-owned, must not be
symlinks, and must not be group/world writable. All four files must be
root-owned regular files with exactly one link, the directory's group, no
world-write permission, and size at most 16 KiB. The policy must also reject
group writes. The other three files may grant group read/write access to a
dedicated trusted adapter group; that group also needs directory read/search
access. Participants cannot replace the lock inode through directory writes.
Authorized group members can modify mutable state, so this is cooperation
between trusted adapters, not isolation from a malicious member of that group.

No policy values are supplied or installed by these tools. The exact policy
schema requires `schema_version: 1`, `namespace: "cfetch-host-inference-v1"`,
`state_directory: "/var/lib/cfetch/inference"`, an explicit `epoch_id`, positive
`lock_wait_ns`, and `operations` containing exactly `compile` and `inference`.
The epoch is 1–128 ASCII letters, digits, dots, underscores or hyphens. Each
operation kind requires all six fields:

| Field | Meaning |
| --- | --- |
| `max_operations` | Finite number of actual calls permitted in this epoch. |
| `max_charged_buckets` | Finite sum of final padded bucket lengths charged in this epoch. |
| `max_duration_ns` | Lease duration from the durable intent's monotonic start time. |
| `minimum_cooldown_ns` | Minimum delay after a successfully completed call. |
| `cooldown_numerator` | Nonnegative elapsed-time cooldown multiplier numerator. |
| `cooldown_denominator` | Positive denominator for that multiplier. |

Numeric fields are integers bounded by `2^63 - 1`; operation limits are
positive except that the numerator may be zero. Derived duration/cooldown sums
must also fit. The next eligible start is completion time plus
`minimum_cooldown_ns + ceil(elapsed_ns * cooldown_numerator / cooldown_denominator)`.
Lock acquisition and any remaining cooldown share the bounded `lock_wait_ns`
wait. Compile and inference have separate counters, with one shared lock and
persisted cooldown across participating scopes. These schema constraints do
not establish safe numerical limits for a device.

`initial_state(policy_sha256, epoch_id, boot_id, monotonic_ns)` returns data for
privileged provisioning or explicit recovery. It does not create files or
authorize a new budget. Before each actual native call the governor fsyncs a
pending intent and charged state while holding the lock. After a successful
call it fsyncs completion and cooldown before clearing the intent. A native
exception, deadline failure or process death leaves the intent pending even
after the kernel releases `flock`. Subsequent operations refuse that state.
Malformed/torn records, changed boots, backward clocks and exhausted epoch
budgets also fail closed. Restart, midnight and reboot never renew a budget or
clear pending work. Recovery is an external privileged operation after the
failed worker and device state have been assessed; adapters never reset state,
remove intent, or create an alternative directory to resume.

Each scope must include `/var/lib/cfetch/inference/policy.json` and its exact
SHA-256 in `required_host.files`, alongside the driver and runtime libraries.
`host-preflight` automatically includes this file, and engine initialization
requires its digest to equal the installed governor policy. The existing
host-file evidence therefore binds the policy without another wire field.
Changing the policy requires new host bindings and qualification; changing only
an epoch also changes the policy bytes and their digest.

The adapter obtains a separate lease for every actual `compile_model` and
every `compiled_model(inputs)` call after final padding. Compiling a missing
bucket finishes its own lease before inference starts. A wire batch pays for
each native input call; a cached compiled bucket incurs no new compile call.
Physical preflight's bucket compilations use the same governor.

Inside each lease, `NativeDeadline(lease.deadline_ns)` arms `ITIMER_REAL` with
the kernel's default `SIGALRM` termination action. It requires the main thread,
an unblocked default-disposition alarm and no existing timer; conflicting
handlers, nested timers and expired deadlines are refused. It checks the
monotonic deadline after arming and again after return, cancelling the timer
on ordinary exit. A native call holding the GIL cannot defer default kernel
termination through Python handler dispatch. The timer is relative: setup,
timer resolution and scheduling can delay termination, so it provides no
nanosecond-precise absolute guarantee. Its scope is the actual compile or
inference call; other runtime phases require their own qualification.

The governor's pending intent survives alarm termination for the supervising
parent and external recovery process to assess. Killing an adapter cannot
recover a wedged device or SoC. Governor provisioning and call deadlines alone
do not lift device quarantine, admit a backend, or establish safe fallback and
endurance behavior. Existing supervised physical-test prerequisites remain in
force.

## Probe package assembly

A scope configuration contains the top-level state, exact frozen runtime
versions, and three ordered NPU/GPU/CPU entries. A physical-probe NPU entry has
this shape; angle-bracket values are intentionally not usable evidence:

```json
{
  "schema_version": 1,
  "package_state": "physical-probe",
  "dependency_versions": {
    "cryptography": "<exact>",
    "numpy": "<exact>",
    "openvino": "<exact>",
    "tokenizers": "<exact>"
  },
  "scopes": [
    {
      "scope_id": "<exact-intel-npu-scope>",
      "backend": "openvino",
      "transport": "supervised-local",
      "runtime": "<exact-runtime-identity>",
      "compiler": "<exact-compiler-and-settings-identity>",
      "package_target": "linux-x86_64-glibc2.35",
      "artifact_source": "google/embeddinggemma-300m@57c266a740f537b4dc058e1b0cda161fd15afa75",
      "artifact_sha256": "<filled-by-assemble.py>",
      "internal_precision": "f16-weight-storage-target-native-compute",
      "device_class": "npu",
      "device": "<exact-device-family>",
      "openvino_device": "NPU",
      "openvino_compile_config": {},
      "required_openvino_properties": {
        "FULL_DEVICE_NAME": "<observed>",
        "DEVICE_ARCHITECTURE": "<observed>",
        "NPU_DRIVER_VERSION": 0,
        "NPU_COMPILER_VERSION": 0
      },
      "required_execution_devices": ["NPU"],
      "required_host": {
        "system": "Linux",
        "machine": "x86_64",
        "kernel_release": "<observed>",
        "files": [
          {"path": "/usr/lib/<exact-resolved-libstdc++-file>", "sha256": "<sha256>"},
          {"path": "/usr/lib/<exact-resolved-libgcc_s-file>", "sha256": "<sha256>"},
          {"path": "/usr/lib/<exact-driver-library>", "sha256": "<sha256>"},
          {"path": "/var/lib/cfetch/inference/policy.json", "sha256": "<exact-policy-sha256>"}
        ]
      },
      "placement_evidence_sha256": null,
      "supported_max_tokens": 2048,
      "supported_sequence_buckets": [32, 64, 128, 257, 512, 1024, 2048],
      "supported_max_batch_size": 64,
      "sequence_capability_evidence_sha256": null,
      "performance_evidence_sha256": null,
      "compatibility_report_sha256": null,
      "attestation_public_key": "<64-lowercase-hex>",
      "attestation_private_key_file": "<input-key-file>",
      "accelerated_placement": true
    }
  ]
}
```

GPU requires exact `FULL_DEVICE_NAME`, `DEVICE_ARCHITECTURE`,
`GPU_UARCH_VERSION`, and `GPU_DEVICE_ID`; CPU requires exact
`FULL_DEVICE_NAME` and `DEVICE_ARCHITECTURE`. The host-file bindings cover
the exact normalized, regular, non-symlink resolutions of `libstdc++.so.6` and
`libgcc_s.so.1` for every scope, plus operator-selected libraries that OpenVINO
does not expose as supported device properties. They do not prove that the
selected files were driver-loaded. All three classes and a distinct Ed25519
key per scope are mandatory. The governor policy binding described above is
also required for each scope, including physical probes.

Create correctly encoded, distinct package keys without using the unrelated
raw-binary admission receipt key command:

```console
python experiments/embedding-profile/openvino_scope_keys.py \
  --scope-id intel-lnl-npu \
  --scope-id intel-lnl-gpu \
  --scope-id intel-lnl-cpu \
  --output-directory results/openvino-operator
```

The output manifest contains only the public keys and relative key filenames;
the sibling `.key` files contain the required 64 lowercase hexadecimal private
bytes and are mode `0600`. Keep the scope configuration in that directory (or
adjust its relative key paths explicitly).

Obtain required properties, host-file hashes, and exact compile-time
`EXECUTION_DEVICES` from the verified raw frozen runtime before assembly:

```console
./cfetch-openvino-adapter-runtime host-preflight \
  --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --artifact-dir "$ARTIFACT_DIR" \
  --artifact-manifest-sha256 "$ARTIFACT_MANIFEST_SHA256" \
  --device-class npu \
  --device NPU \
  --compile-config-json '{}' \
  --host-file /usr/lib/<exact-resolved-regular-libstdc++-file> \
  --host-file /usr/lib/<exact-resolved-regular-libgcc_s-file> \
  --host-file /usr/lib/<exact-regular-non-symlink-driver-library>
```

Repeat separately for GPU and CPU. The externally obtained runtime digest is
required; the runtime cannot vouch for its own manifest identity. The command
verifies that pinned raw runtime and the artifact before and after use, queries
the exact typed allowlisted properties on the stable physical device returned
by `EXECUTION_DEVICES`, hashes the selected normalized host files plus the
automatically included governor policy before and after compilation, compiles
all seven static buckets, and requires one stable physical `EXECUTION_DEVICES`
value. It emits
one bounded canonical JSON line containing the runtime/artifact digests,
dependency versions, compile config, properties, host binding, and bucket
results. Copy only the named scope-configuration fields into the matching
physical-probe entry. This output is configuration provenance, not admission
evidence; it does not discover relevant driver files. The total host-file limit
is sixteen, including the governor policy; leave room for its automatic addition
when selecting driver/library paths. Do not substitute system
OpenVINO, guessed paths, or marketing names.

Assemble and self-check the final directory with:

```console
python packages/openvino/assemble.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --runtime-dir "$RUNTIME_DIR" \
  --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --scope-config "$SCOPE_CONFIG" \
  --output-dir "$PACKAGE_DIR"
```

`assemble.py` never creates keys, invents evidence, or silently chooses a
device. Its JSON result reports the final launcher, launcher digest, runtime
manifest digest, and package inventory digest.

## Supervisor and signed HTTP contract

The cfetch parent invokes the packaged sibling directly:

```console
./cfetch-openvino-adapter serve --host 127.0.0.1 --port 0 --auth-stdin
```

It writes exactly one JSON line containing a fresh 32-byte lowercase-hex bearer
and keeps the pipe open:

```json
{"bearer":"<64-lowercase-hex>"}
```

After package/runtime integrity checks, the child emits exactly one bounded
readiness line to stdout:

```json
{"schema_version":1,"url":"http://127.0.0.1:<ephemeral>/v1","scope_ids":["<npu>","<gpu>","<cpu>"]}
```

EOF on stdin shuts it down. Diagnostics go to stderr. HTTP accepts only
authenticated `POST /v1/embeddings` with a fresh
`X-Cfetch-Attestation-Nonce` and this exact body shape:

```json
{
  "model": "google/embeddinggemma-300m",
  "dimensions": 768,
  "input": ["already-prefixed text"],
  "cfetch_requested_scope_id": "<selected-scope>"
}
```

The signed response carries `cfetch_execution`, including `package_state`,
`transport: "supervised-local"`, and all four explicit evidence/report
bindings. It also carries live `cfetch_runtime_evidence` with exact host
identity, host-file hashes, required OpenVINO properties, and one record per
executed bucket. Placement comes from
`compiled_model.get_property(EXECUTION_DEVICES)` and device properties come
from `core.get_property`; echoed request labels are not placement evidence.

The packaged Ed25519 private keys are distributed bytes. Their signatures bind
the nonce, exact request body, and exact response body inside the
supervisor-controlled local process; they are not proof of remote identity or
secret package possession. Remote service scopes require a separate
`remote-attested` transport and a non-distributed operator key.

## Remaining physical work

The manual build must first reproduce the canonical bytes from the pinned
public mirror. Then an exact `physical-probe` package must run on the target
Intel cohort. The external
physical collector—not this build recipe—must retain signed raw transactions
covering all seven buckets, live placement/properties/host identity, latency,
RSS, 1-through-64 wire grouping, output digests, and repeatability. Energy may
be explicitly unmeasured; it must not be synthesized. Only the global
all-pairs compatibility evaluation and final published-package replay can
produce candidate/release bindings.

Run dependency-light checks with:

```console
python -m unittest discover -s packages/openvino/tests -v
```
