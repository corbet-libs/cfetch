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
check command never installs tools. It defaults to one Cargo job and two Rust
test threads; the worker's resource limits remain the enclosing bound.

Crow's manual `verify` workflow requires `SOURCE_ARCHIVE` and `SOURCE_SHA256`.
Create the archive from the exact committed revision with `git archive`; stage
it on the worker and submit that revision as `CI_COMMIT_SHA`. The source step
checks both the archive SHA-256 and Git's embedded commit ID before extraction.
Choose `CHECKS=catalog`, `rust`, `variants`, `licenses`, `profile`, or `portable`
(`catalog rust`). The inexpensive catalog check is the manual default.
Crow defaults its compiled cache to `ci-targets/cfetch` under `CARGO_HOME`
(or `$HOME/.cargo`). An optional `CARGO_TARGET_DIR` overrides that directory.
Crow serializes users of that cache with a one-minute lock wait;
checks have a 45-minute deadline and a 30-second forced-termination grace.
The worker must provide a persistent writable Cargo home for reuse across runs.

This path removes the source checkout's dependency on GitHub availability.
Crow may still obtain workflow configuration through the configured forge;
the dispatcher must report a configuration-fetch failure rather than claim a
run started. Crow registration and worker provisioning are operator concerns.

These Linux results do not replace required native macOS/Windows results,
physical accelerator placement/admission evidence, or release publication.
The existing platform and publication gates remain required. Retained bytes
and cohort replay do not establish new physical-device evidence.
