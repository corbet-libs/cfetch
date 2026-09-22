# CI checks and provider coverage

Prefer eligible, free GitHub Actions for public checks. Use the existing Crow
route while hosted execution is unavailable and for private or physical-device
inputs. Both routes execute `scripts/ci-check.sh`; select the checks whose inputs
changed. The default is one Cargo job and one test thread. Hosted Rust selects
current `stable`; Crow uses its provisioned compatible compiler. `rust-version`
in Cargo.toml remains a minimum, and Cargo.lock remains the dependency snapshot.

| Selector | GitHub Actions | Crow |
| --- | --- | --- |
| `ci-config` | Selected workflow; provisions Actionlint | Existing Actionlint/ShellCheck; no build |
| `release-guards` | Shared fixture source | `release` workflow, operation `guards`; uses the exact publisher resource |
| `catalog` | Automatic CI or selected workflow | Catalog, staging and variant-command fixtures |
| `governor` | Selected workflow | Model-free deadline and load-policy checks |
| `staging` | Existing native CI suite | Focused staging, migration, initialization, import and index checks |
| `rust` | Linux on PR/main; Windows on main; all three OS families on release/manual CI | Provisioned native host, with the same locked commands |
| `variants` | Selected Linux checks or automatic platform matrix | Only the worker's native Linux architecture |
| `licenses` | Provisions the declared license tools | Requires existing cargo-deny and cargo-about |
| `profile` | Installs the fully hashed public policy requirements | Requires the same provisioned policy environment |
| `ort-foundation-build`, `ort-foundation-cpu` | Excluded | Explicit retained bundle/runtime inputs; independent private/device policy |

`.github/workflows/selected.yml` accepts a comma-separated selection, defaulting
to `catalog,governor`. It rejects private probes, unknown selectors, duplicates,
and commands that differ from `.ci/ccid.toml` before starting work. It provisions
only the selected public prerequisites. Its artifact records source/archive,
dependency and workflow hashes, check outcome, actual tools/platform, and the
resource budget. A failed or skipped check remains failed or skipped. These
receipts do not claim native accelerator execution or shared ccid execution.

For Crow, use the existing source-staging dispatcher:

```sh
crow-ci plan --repo . --workflow ccid --provider crow --var CHECKS=ci-config,catalog
crow-ci run --repo . --workflow ccid --provider crow --var CHECKS=ci-config,catalog
```

Only committed source is staged. The pinned ccid verifies the source/archive,
tool binary and canonical repository identity before running commands. Persistent
dependency and target caches remain reusable; changed workspace source paths do
not reuse binaries embedding obsolete temporary paths. The worker installs no
tools. Raise a check timeout only for an explicit justified run.

Inspect existing work before dispatch. Do not submit equivalent hosted and Crow
checks simultaneously; source alone is insufficient for result reuse when the
dependency graph, toolchain, configuration or relevant environment differs.
Native variant caches are shared between hosted check and build jobs, while each
variant retains its own identity. Release work is serialized and never cancelled
by superseded validation.

## Automatic selection and current proof

`.ci/providers.toml` maps only `catalog,governor` to eligible free GHA with one
job, one test thread, an 8192 MiB reserve and a 900-second timeout. The same
selected workflow retains its broader direct-manual public selectors. Use
`crow-ci run --repo . --workflow ccid --provider auto --var CHECKS=catalog,governor`:
the dispatcher reconciles prior requests first and chooses Crow when GHA or its
verified portable bootstrap is unavailable. Explicit resource overrides use
Crow. A real hosted test failure remains a failure.

At this implementation's review, the portable ccid hosted bootstrap had no run
and its release asset was absent. The hosted route is configured, not proven
operational. GHA is assumed unavailable; Crow remains the execution route.
Push/PR CI is still a GHA workflow; automatic provider selection occurs through
the shared dispatcher, not a new always-running push service. No worker,
infrastructure, credential or registry policy is installed by these files.

## Release commands and retained files

Both providers call `scripts/release.py`. The hosted tag workflow calls the
existing CI workflow once and waits for its catalog, license, profile, native
Rust and unadmitted candidate-build jobs before preparation. The former separate
tag CI trigger is removed. Release calls omit admitted rows from CI's variant
check matrix because native preparation builds and smokes those exact rows;
unadmitted candidates retain their CI check. Routine CI keeps its full selected
matrix. Every existing candidate build remains covered without compiling the
six admitted rows twice. Stable
Rust is selected on hosted workers; Cargo.toml's minimum compiler is preserved.
The x86_64 macOS catalog row uses the standard `macos-26-intel` native runner;
the driver independently rejects a mismatched host architecture.

| Command | Required inputs and resulting evidence | Credentials |
| --- | --- | --- |
| `plan` | Clean exact source; admitted matrix and required gates | None |
| `check catalog`, `licenses`, `profile`, `rust` | Existing shared check command; actual source, host and tool receipt | None |
| `prepare <variant>` | Matching native host; build, admitted payload staging, executable version/variant smoke, immutable archive | None |
| `prepare cargo` | Locked credential-free Cargo dry run; its exact retained crate | None |
| `verify --input <staged-directory>` | All gate and artifact receipts; emits `cfetch-v<version>-bundle.tar` plus its SHA-256 | None |
| `inspect` | Retained bundle and independently supplied SHA-256; full offline byte/gate validation | None |
| `status` | Same bundle; read-only exact GitHub/Cargo reconciliation | GitHub read access for drafts |
| `create` | Complete verified bundle; create/reconcile the draft release object for the existing exact tag | GitHub release write |
| `publish github`, `finish github` | Exact immutable uploads, then make the fully checked draft public | GitHub release write |
| `publish cargo` | Complete public GitHub release, then upload the already verified crate | Scoped Cargo token/OIDC and GitHub intent write |

Preparation requires absolute `RELEASE_ARTIFACT_DIR`, or uses
`CARGO_TARGET_DIR/cfetch-release/<source-or-bundle-identity>`. It retains:

```text
<retained directory>/
  source.tar, plan.json
  checks/check-<check>-<native-host>.json
  <variant>/variant-<variant>.json, <native archive>, <expanded staging directory>/
  cargo/cargo.json, cfetch-<version>.crate
  bundle/                         # verified source, receipts and exact artifacts
  cfetch-v<version>-bundle.tar     # transport this with its independent SHA-256
  publication/                    # durable local mutation intent and receipts
```

Collect retained preparation outputs from the same producing source; pass their
common directory to `verify`. Checks require Linux/macOS/Windows Rust receipts,
catalog/license/profile success, and every admitted native artifact with its
actual version/embedded-variant smoke. Cargo metadata and payload are verified
against producing source by the existing pinned ccid publisher resource. The
binary/tool archive used by release has its own exact revision; ordinary check
pins remain independent. Set `CI_TOOL_ARCHIVE` and `CI_TOOL_SHA256` to that
verified public tool source for verification/publication; nothing compiles it.

For cross-provider publication, stage the complete bundle through the existing
shared Crow transport, set `RELEASE_BUNDLE` and `RELEASE_BUNDLE_SHA256`, and use
workflow `release` with `RELEASE_OPERATION`/`RELEASE_SELECTION`. Driver revision
and producing source remain distinct. Bundle import checks the old exact
source/tag identity; it never relabels an artifact as a later CI commit. If the
release matrix evaluator itself changed, import stops for an explicit review.

Hosted gate/native/crate jobs download the plan artifact and consume its exact
source.tar and separately passed checksum. Hosted checkouts explicitly disable
Git autocrlf before checkout; strict file/mode comparison remains enabled.

For mixed-provider preparation, carry the **same original source.tar bytes** to
every worker using `RELEASE_SOURCE_ARCHIVE` and `RELEASE_SOURCE_SHA256`; its
commit and complete file/mode inventory must match the executing source.
Stage that file with the existing shared `transport.stage` call, using the
configured repository ID, SSH transport and source roots:

```python
worker_path = transport.stage(source_path, repo_id, source_sha256,
                              ssh, host_sources, worker_sources, extension="tar")
```

Pass the returned worker path as `--var RELEASE_SOURCE_ARCHIVE=...` and the same
external digest as `--var RELEASE_SOURCE_SHA256=...` to `crow-ci run --repo .
--workflow release --provider crow --var RELEASE_OPERATION=prepare --var
RELEASE_SELECTION=<native-variant>`. Whole bundles use the same transport with
`extension="bundle"`, then `RELEASE_BUNDLE` / `RELEASE_BUNDLE_SHA256` and operation
`inspect`. The shared transport verifies and atomically stages the bytes; merely
passing a local path does not transfer it. Connection details remain operator
configuration, outside the public repository.

Independently serialized Git archives are not interchangeable receipts merely
because they name the same commit. An entire verified bundle can be moved
without this preparation step. Existing outputs are inspected/reused; stale or
unreceipted crate candidates are never silently selected after another dry run.

Hosted artifacts retain preparation bundles for 30 days and publication journals
for 90 days. Preserve the bundle, external SHA-256 and creation journal in the
existing retained artifact store before expiry or provider migration. Crow uses
persistent artifact/journal directories. GitHub per-artifact intent assets
carry unique ownership; a second actor cannot upload after observing an existing
intent. Existing registry/assets must match downloaded bytes exactly. There is
no overwrite, registry-policy change or blind upload retry. A creation/finalize
response that remains unknown stops; `status` reconciles without another
mutation. A hosted rerun with an absent release requires the retained creation
journal before attempting recovery. Do not discard uncertain intent files.

The Crow wrapper resolves secrets only for the selected credentialed operation.
`cfetch_release_github_token` serves GitHub release/intent writes;
`cfetch_release_cargo_token` is needed only by Cargo publication. Hosted Cargo
retains its registered `release.yml` / `crates-io` OIDC boundary; the built-in
GitHub token writes independent publication intents. Compilation and staging
receive neither token.

## Maintenance preparation and remaining publication routes

Homebrew's existing GHA updater now runs after the exact GitHub release becomes
public, or through manual version dispatch. It validates the version, peeled tag,
public release, catalog checksum and four endpoint archive references/checksums
before tap access.
The formula renderer is shared with Crow; historical release metadata does not
claim fresh native build evidence. The updater preserves its existing tap commit
and normal fast-forward push. `PACKAGES_TOKEN` needs Contents write on that tap;
the renderer receives no token. No asset polling or duplicate tag-triggered job
is needed.

The separate Crow `maintenance` workflow contains no secrets and permits only
`MAINTENANCE_OPERATION=guards`, `patch-plan`, or `homebrew-render`. It uses the
existing shared publisher source revision for complete bundle inspection.
`homebrew-render` requires `RELEASE_BUNDLE` and its independent
`RELEASE_BUNDLE_SHA256`; it verifies all retained release gates and prepares a
formula. This offline result does not establish that the release is public.

`patch-plan` calls the existing patch-preparation script inside a disposable
checkout from an explicitly staged full Git bundle. Set
`MAINTENANCE_HISTORY_BUNDLE` and `MAINTENANCE_HISTORY_SHA256`; use the existing
shared transport with `extension="bundle"`. Its source commit and files must
match the exact dispatched archive, and its latest release tag must be an
ancestor. Create the bundle with `git bundle create <path> --all` after fetching
the intended public history; the worker never fetches it. No `.ci/archives.toml`
is added, so automatic ordinary check routing remains available. The matching
GHA `Inspect patch preparation` workflow uses the same command.

The retained `maintenance/<operation>/` contains a formula or patch plus a
source/history/bundle-bound receipt. Patch planning leaves source, main and tags
unchanged and preserves resumption of an already prepared untagged version.
License regeneration and full checks remain explicit pending steps. The existing
credentialed GHA preparation transaction still checks CI is enabled before a
version/main write and waits for success at the exact candidate SHA before
tagging. Running that workflow establishes current GHA execution; enabled state
alone does not promise future capacity. Its bounded wait remains necessary.

Crow **tap writes and candidate/main/tag publication are still unserved**. The
minimal shared mutation design is an explicit prepare/finish transaction:

1. Verify the prepared receipt, unchanged base and complete required evidence;
   for a tap update reuse the existing release driver's exact public-release
   verification. Regenerate licenses before freezing one candidate commit.
2. Retain the expected base SHA, candidate commit/tree, tag name, output digest
   and operation intent before any write. Share this exact identity across
   providers; never recompute the next version during reconciliation.
3. Push the pre-created commit by normal fast-forward push. For a new immutable
   tag, use a create-only ref operation; never force or rewrite an existing tag.
   Remote main/tag must still match the expected source. A competing advance
   invalidates the transaction for review; no automatic rebase of checked bytes.
4. Read back exact refs after success or an uncertain response. A matching ref
   completes the existing intent; a different ref fails; an unresolved outcome
   stops without another write. Require complete source-bound CI evidence before
   the tag, and retain the existing release driver's all-native publication gates.

The current preparation helper implements no publication or credential storage.
Scoped source/tap credentials and complete native evidence remain prerequisites
to implementing and executing those shared mutation commands.

## Native and downstream coverage limits

Crow's provisioned Linux worker can execute its native checks/preparation and
inspect/publish a complete imported bundle. It cannot manufacture missing macOS,
Windows or other-architecture evidence. Those compatible native workers and
retained receipts are prerequisites for a fresh full release while GHA is
unavailable. No model or physical accelerator is exercised by the release smoke;
local-package admission still uses the existing explicit policy and staged
payload verifier. Keep the one-thread and approximately 50% hardware-budget
policy for any separate device work.

Candidate/main/tag writes and downstream Homebrew tap writes keep their existing
GHA publication routes and do not yet have Crow equivalents. Read-only patch
planning and formula rendering now have the separate shared maintenance routes
described above. A configured route is not proof of a completed native release
or downstream package update.

Run the lightweight configuration/catalog checks through workflow `ccid`, and
run release fixtures through workflow `release` with `RELEASE_OPERATION=guards`.
The latter deliberately uses release's pinned publisher resource and exercises
the real shared Publisher against a fake network, including selected draft
preservation. Running that fixture with the older check-tool archive is rejected;
it is not a substitute for validating the release tool actually in use.
