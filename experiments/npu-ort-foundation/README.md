# NPU execution through FastEmbed and ORT

This standalone probe starts with runtime loading, then runs one built-in
FastEmbed model through an explicitly selected OpenVINO device. It does not
change cfetch's model, vector store or admitted backends.

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
