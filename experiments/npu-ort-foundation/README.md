# NPU execution through FastEmbed and ORT

This standalone probe starts with runtime loading, then runs one built-in
FastEmbed model through an explicitly selected OpenVINO device. It does not
change cfetch's model, vector store or admitted backends.

The [CPU diagnostic evidence](cpu-diagnostic-evidence.json) and
[actual vector](cpu-diagnostic-vector.json) record one successful built-in
EmbeddingGemma call: 13 tokens, 768 finite components, 6.208 seconds to
initialize and 0.119 seconds for the first embedding. These are single-call
observations, not a warm-latency benchmark or semantic admission.
OpenVINO executed 99 subgraphs, while ORT's CPU provider executed 24
`MultiHeadAttention` and 48 `RotaryEmbedding` operations. Those are model
computation, not shape bookkeeping. On this short input their summed profile
time was 3.418 ms, versus 112.528 ms for the OpenVINO kernels. Long-input cost
and device handoff overhead remain unmeasured; this does not prove that an
explicitly qualified mixed route is impractical. This exact graph/runtime pair
therefore
does not establish exclusive accelerator execution; the strict mode correctly
refuses it. NPU/GPU execution and canonical long-input compatibility remain
unqualified.

For this CPU-only investigation, the new probe accepts
`--cpu-fallback diagnostic`. The CPU wrapper requires an explicit
`CFETCH_FOUNDATION_PROBE_DIR` build receipt and
`CFETCH_FOUNDATION_CPU_DIAGNOSTIC=1`. NPU mode rejects that flag and retains
strict fallback and provider checks. Vector, timing and placement files are
synced before success; the supervisor rechecks model, probe and runtime bytes.

The measured stack is FastEmbed 6.0.2, `ort` 2.0.0-rc.13 with **API 24 only**,
and Intel's ONNX Runtime OpenVINO 1.24.1 package containing OpenVINO 2025.4.1.
Enabling `ort`'s default features would raise the required API version.

Build in an isolated workspace. Verify the FastEmbed crate against the checksum
in `Cargo.toml`, extract it to `vendor/fastembed`, and apply
`fastembed-session.patch` with `patch --fuzz=0 -p1`. The patch exposes session
configuration and profile completion while preserving built-in model retrieval,
tokenization and pooling. Build with `cargo build --locked` using the retained
`Cargo.lock`; do not resolve a new dependency set for the measured experiment.

The default command loads the explicit runtime and reports its version and
OpenVINO provider availability without loading a model:

```sh
cfetch-npu-ort-foundation --ort /absolute/path/libonnxruntime.so.1.24.1
```

Model execution requires explicit `--run-model AllMiniLML6V2` or
`--run-model EmbeddingGemma300M`, `--device CPU|NPU`, `--cache DIR`, and a fresh
`--output DIR`. It uses one fixed short input, one inference call, one thread,
disabled ORT graph optimizations and rejected CPU fallback. The resulting
profile must contain OpenVINO execution events and no other provider events.
NPU execution requests one tile with turbo disabled.

Provider profiling proves OpenVINO EP execution. Independently retain native
device diagnostics to establish NPU placement and effective tile configuration;
the provider name alone does not establish either.

The caller must provide the physical-test supervision and deadline. Serialize
NPU work and require cooling time greater than or equal to active time to keep
the duty cycle at or below 50%. Tile count and duty cycle are separate controls;
neither is a guaranteed instantaneous utilization percentage. This executable
is not an unattended workload runner and does not implement the production
host-wide governor. It cannot lift an existing device quarantine.

For the initial supervised canary, allow only the built-in MiniLM's single
seven-token input. Bound the entire worker, including initialization and
cleanup, with a 45-second hard process deadline, followed by at least 60
seconds without NPU work. Retain a permanent attempt marker before starting;
an error or interrupted attempt does not permit an automatic retry. Record
heartbeat observations externally and stop on the first missing heartbeat,
display artifact or driver warning. These are proposed test bounds, not
qualified safe operating limits.

Verified so far: runtime loading and one built-in MiniLM call through the
OpenVINO CPU provider, returning a finite nonzero 384-dimensional embedding.
The same bundle loads on the target Intel NPU host without executing a model.
Its model-free native property query reports six maximum tiles, configurable
tile selection and turbo off. Physical NPU model execution remains unmeasured.

The pinned native runtime is the x86-64 Linux `onnxruntime-openvino` 1.24.1
CPython 3.12 wheel, SHA-256
`d617fac2f59a6ab5ea59a788c3e1592240a129642519aaeaa774761dfe35150e`.
Rust loads its standalone shared libraries; it does not import the wheel's
Python module or the host's Python OpenVINO installation. Runtime loading
reports ORT 1.24.1, API 24 and upstream commit `b5963e82c8`. The wheel's
OpenVINO C API reports build `2025.4.1-0-test`; it also loads the host's NPU
compiler loader, which remains part of the physical test tuple.

The explicit Crow/ccid selector `ort-foundation-cpu` can reuse this measured
bundle without building anything. Supply `CFETCH_FOUNDATION_BUNDLE` and a fresh
absolute `CFETCH_FOUNDATION_OUTPUT`; set `CFETCH_FOUNDATION_MODEL` to
`EmbeddingGemma300M` or `AllMiniLML6V2` for one CPU embedding call. Leaving the
model empty performs only runtime loading. The selector verifies the retained
manifest and every bundled file, fixes CPU selection, limits tokenization and
OpenMP threads to one, and retains output/error/intent/result files. A separate
300-second process timer bounds cold loading and downloads as well as the
single model call. It never selects NPU or GPU. The recorded total duration
includes downloads and is not an inference-latency measurement.

EmbeddingGemma uses the six files at the exact Hugging Face revision recorded
in `embeddinggemma-cache.json`. ORT 1.24.1 rejects the usual snapshot symlinks
for external weights ([upstream report](https://github.com/qdrant/fastembed/issues/603)).
The runner verifies existing content and creates a fresh hard-linked snapshot
beside the result, so the external weights resolve inside the model directory.
This retains the same bytes without another model download or a runtime change.
Missing tokenizer/configuration files may be fetched at that exact revision;
missing large weights and corrupted cached files stop the check. All six files
are checked before starting the native worker. Its Hugging Face endpoint is
invalid, so a cache miss cannot fetch a moving `main`.
The result retains and rechecks the model/tokenizer identities. This candidate's
long-input compatibility with the canonical source still requires measurement.

If the worker's default environment omits existing native build tools, supply
`CFETCH_FOUNDATION_PKG_CONFIG` (absolute executable) and
`CFETCH_FOUNDATION_OPENSSL_DEV` (existing headers and `lib/pkgconfig` directory)
together. The build records the selected tools and OpenSSL version. These
inputs select provisioned files; they never install packages. A worker-native
diagnostic binary is not evidence of portability to another machine.
For a Nix-linked probe, the CPU runner resolves the worker's existing
`NIX_LD_LIBRARY_PATH` into its own library search path and records those
immutable directories. Nix-linked executables bypass the `nix-ld` shim that
supplied dependencies for the original portable executable.

For a diagnostic source change, `build_cached_probe.py` prepares a separate
binary on the CPU build worker. Set `CFETCH_FOUNDATION_RETAINED_ROOT` to the
existing experiment directory containing `build/vendor/fastembed`,
`fastembed-6.0.2.crate`, `cargo-home`, `target`, and `runtime-bundle`. Set
`CFETCH_FOUNDATION_PROBE_DIR` to a fresh absolute output directory, then run:

```sh
python3 experiments/npu-ort-foundation/build_cached_probe.py
```

The helper verifies the published crate, the exact retained vendor patch, the
locked dependencies, and the original runtime manifest. It copies source and
vendor files into the fresh output's `build-source`, then reuses the existing
Cargo cache and target directory with `cargo build --offline --locked --jobs 1`.
The worker must already provide Cargo, Rust, C/C++ compilers, pkg-config,
OpenSSL development headers/libraries, and GNU timeout; missing prerequisites
stop the build. No dependencies are downloaded or installed. Crow must supply
the CPU and memory resource limits and serialize use of this retained target.

The build has an independent 600-second process deadline and bounded reap.
Successful output contains `cfetch-npu-ort-foundation` and
`build-identity.json`, recording `binary_sha256`, `source_sha256` for
`Cargo.toml`, `Cargo.lock`, `fastembed-session.patch`, and `src/main.rs`,
`vendor_tree_sha256`, `bundle_manifest_sha256`, runtime file identities, and
compiler identities. The original build source and runtime bundle are retained;
the Cargo target remains a mutable build cache. Building does not load the
runtime or execute a model. Model execution is a separate explicit CPU check.
