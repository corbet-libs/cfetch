# CI checks

Run `bash scripts/ci-check.sh <check>...` on a provisioned build worker. With
no arguments it runs `catalog rust`. Each named check is independent so a
failure can be investigated and rerun without repeating successful work.

| Check | Coverage | Existing tools |
|---|---|---|
| `catalog` | Packaging catalog, exact payload staging, admitted release matrix | Bash, jq, Python 3 |
| `rust` | Locked tests, feature combinations, isolated FastEmbed diagnostics, Clippy | Rust, Clippy, native build dependencies |
| `variants` | Each native Linux architecture's candidate release build | Rust, jq |
| `licenses` | Dependency policy and generated notices | cargo-deny, cargo-about |
| `profile` | Policy arithmetic and retained admission evidence replay | Python environment matching `experiments/embedding-profile/requirements-lock.txt` |

`CFETCH_POLICY_PYTHON` may select an existing absolute Python executable. The
check command never installs tools. The shared Crow adapter supplies a
memory-bounded Cargo job and test-thread budget. Direct calls without those
environment settings retain conservative script defaults.

Crow's manual `ccid` workflow uses the selectors in `.ci/ccid.toml`.
The operator submission helper stages the exact committed source closure and
supplies the pinned shared Rust tool archive and binary with SHA-256 digests.
The adapter verifies tool identity, archive integrity and source commit before
execution. Missing locally available source objects fail closed.
Choose `CHECKS=catalog`, `rust`, `variants`, `licenses`, or `profile`; use
`CHECKS=catalog,rust` for both portable selectors. The inexpensive catalog
check is the manual default. The superseded single-core `verify` workflow
has been removed.

Compiled targets use persistent dedicated Cargo storage and canonical repository
namespaces. An explicit `CARGO_TARGET_DIR` remains authoritative. Shared `ccid`
locks the actual target, preserves unchanged source freshness and cleans owned
source scratch. Existing package-cache settings are preserved. `CI_JOBS`,
`CI_TEST_THREADS`, `CI_MEMORY_MB` and `CI_TIMEOUT` allow bounded overrides;
memory admission still applies before checks execute.

This path removes the source checkout's dependency on GitHub availability.
Crow may still obtain workflow configuration through the configured forge;
the dispatcher must report a configuration-fetch failure rather than claim a
run started. Crow registration and worker provisioning are operator concerns.

These Linux results do not replace required native macOS/Windows results,
physical accelerator placement/admission evidence, or release publication.
The existing platform and publication gates remain required. Retained bytes
and cohort replay do not establish new physical-device evidence.
