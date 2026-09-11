<h1 align="center">cfetch</h1>

<p align="center">
  <strong>Local-first memory and context control for AI coding agents.</strong>
</p>

<p align="center">
  Give Claude Code, Codex, Gemini, Cursor, and MCP clients one durable memory:<br />
  cited Markdown recall, exact code navigation, autonomous Obsidian maintenance,<br />
  knowledge graphs, vector search, and cross-machine access from one Rust binary.
</p>

<p align="center">
  <a href="https://github.com/corbet-labs/cfetch/actions/workflows/ci.yml"><img src="https://github.com/corbet-labs/cfetch/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status" /></a>
  <a href="https://github.com/corbet-labs/cfetch/releases/latest"><img src="https://img.shields.io/github/v/release/corbet-labs/cfetch?display_name=tag" alt="Latest release" /></a>
  <a href="https://crates.io/crates/cfetch"><img src="https://img.shields.io/crates/v/cfetch.svg" alt="crates.io version" /></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-FSL--1.1--ALv2-blue.svg" alt="License: FSL-1.1-ALv2" /></a>
</p>

<p align="center">
  <a href="#terminal-dashboard">Dashboard</a> ·
  <a href="#system-diagnostics-and-hardware-visibility">Diagnostics</a> ·
  <a href="#feature-map">Feature map</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#openwolf-openwolf-enhanced-and-cfetch">OpenWolf comparison</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#configuration">Configuration</a>
</p>

## What is cfetch?

cfetch is a source-available AI agent memory system, Obsidian second brain, and
MCP server for software development. It indexes ordinary Markdown, retrieves the
most relevant statements with stable citations, maps source code to exact
symbols and line ranges, and connects that context to coding agents through
native hooks, instructions, MCP, or the command line.

Your Markdown remains the source of truth. Search databases, vector caches,
and daemon state are derived or disposable; capture and usage streams remain
inspectable files. cfetch does not require a hosted memory service, a
proprietary database, or a change to how you edit your notes.

| Without cfetch | With cfetch |
|---|---|
| A new agent session starts cold | A small, scoped memory digest arrives at session start |
| Finding one decision means searching and opening several files | `cfetch recall` returns ranked statements with file, line, and trust-aware citations |
| Finding one function can pull an entire file into context | `cfetch find` returns the exact symbol range and an estimated token cost |
| Large command output occupies every later model turn | Safe output families are condensed; the complete original remains available |
| Context compaction drops in-flight knowledge | cfetch restores the session's modified-file record and relevant reminders |
| Hooks, hardware, or remote routes can fail silently | Heartbeats, transcript verification, `status`, `doctor`, and `selfcheck` expose the exact gap |
| Every machine builds a private, potentially stale copy | A storage machine can serve fresh, generation-stamped answers to thin clients |
| Captured fixes accumulate as an inbox someone must clean | Evidence-driven maintenance continuously updates Markdown, while direct Obsidian edits always win |

## Terminal dashboard

![cfetch terminal dashboard showing hook health, indexed Markdown and code, context usage, staging, and cited recall](docs/assets/cfetch-dashboard.png)

`cfetch dashboard` makes the memory layer visible in one terminal UI. Recall
searches cited knowledge. Activity shows captured evidence, automatic changes,
independent reviews, exact checks, exceptions, and reversible transactions.
Graph exposes the Obsidian wikilink network derived from Markdown. System shows
daemon and hook health, local or remote routes, catalog generation, vector
coverage, model identity, detected hardware, actual backend selection, peer
reachability, grants, and actionable findings.

The screenshot shows the shipped v0.9.9 dashboard; `main` adds the Activity,
Graph, and expanded System views described above. The TUI is an observability
and debugging surface. Normal maintenance runs in the daemon and does not wait
for this screen or for routine approval.

## Feature map

The latest tagged release is v0.9.9. Features added since that tag are clearly
marked **Main — next release**; install from source to try them before the next
release. The semantic transport is implemented, but main currently admits no
embedding execution scope: semantic production fails closed until an exact
local or remote scope passes the compatibility and package-evidence gates.
Optional reranking uses an
OpenAI-compatible endpoint and is outside the shared-vector ABI. Lexical
recall, code navigation, hooks, capture, and measurement need neither.

| Area | Availability | What you experience | Main surfaces |
|---|---|---|---|
| Persistent agent memory | v0.9.9 | Decisions, rules, notes, and working state stay in plain Markdown and git | `cfetch init`, configurable knowledge tree |
| Selective learning cards | **Main — next release** | A huge nixcards catalogue can contribute only the locally selected Markdown sets; the nixcards TUI and cfetch share Git's native sparse selection without a shadow config | `cards init`, `cards list`, `cards select`, `cards sync`, `cards tui` |
| Cited retrieval | v0.9.9 | BM25, semantic, and hybrid search return statement-level citations that survive file reordering | `recall`, `recall --semantic`, `recall --hybrid`, `recall --id` |
| Retrieval quality | v0.9.9 | Trust-aware ranking, optional cross-encoder reranking, lexical precision gates, duplicate suppression, and wikilink expansion | `recall --expand`, `rerank.*`, `recall.gate` |
| Code intelligence | **Main — next release** | Tree-sitter symbols, exact line ranges, import-graph importance, token-budgeted repository maps, explainable dependency paths, bounded bidirectional context, conservative symbol relationships, and reverse-impact analysis | `find`, `map`, `code-graph path`, `code-graph context`, `code-graph impact`, `code-graph symbol`, `.cfetchignore` |
| Session continuity | v0.9.9 | Scoped startup memory, periodic rule refresh, modified-file recovery after compaction, and known-failure lookup | hooks, `failures` |
| Context control | v0.9.9 | Repeat-read guidance, large-file slice hints, token-capped answers, and structural command-output condensation | native tool hooks, `--budget-tokens` |
| Learning capture | v0.9.9 | Redacted session activity is captured and deterministic signals flag evidence worth maintaining | hooks, ring-6 exhaust, ring-5 staging |
| Autonomous AI maintenance | **Main — next release** | A change-driven daemon proposes, independently reviews, verifies, and applies evidence-grounded Markdown updates without routine approval; external edits win and every outcome is journaled | `maintain run`, `maintain history`, Activity pane |
| Obsidian knowledge graph | **Main — next release** | Human-authored wikilinks become a rebuildable, ambiguity-safe graph with focused neighborhoods, degree counts, rings, and slice-scoped remote access | `graph`, Graph pane, `recall --expand` |
| Honest measurement | v0.9.9 | Transcript-derived usage, cfetch's own injection cost, rewrite-point savings, cache-rebuild attribution, and paired A/B analysis | `audit`, `bench`, dashboard |
| Reliability | v0.9.9 | A warm daemon, incremental scanning, hook heartbeats, delivery verification, and fail-open hook behavior | `daemon`, `status`, `selfcheck` |
| Runtime visibility and diagnostics | **Main — next release** | Compact agent-safe status plus an evidence-rich doctor distinguish configured, detected, selected, last-used, local, remote, paused, failed, and unmeasured state | Activity/System panes, `doctor`, `doctor --json`, `status`, `cfetch_runtime_status`, Claude status line, Codex hooks |
| Agent integrations | v0.9.9 | Capability-detected native hooks, MCP registration, and recall-first instruction blocks across common coding agents | `install`, `install --agent`, `mcp` |
| Multi-machine access | v0.9.9 | Storage hosts serve bounded, freshness-labeled queries; clients can hold no local index | serving daemon, drain barrier |
| Selective sharing | v0.9.9 | Nested slices, authenticated host identities, one-time invites, and per-slice grants | `slices`, `identity`, `invite`, `join`, `grants` |
| NPU-first local shared-vector profile | **Main — candidate** | One frozen semantic pipeline and signed INT8x768 output/index codec permit target-native internal precision; concrete NPU, GPU, and accelerated CPU packages enter only through absolute all-pairs and adversarial mixed-store retrieval tests | `embedding-profile`, semantic and hybrid recall |
| Peer vector artifacts | **Main — next release** | A second storage group fetches matching canonical vectors from authorized origins before considering its own endpoint, avoiding duplicate embedding work | `embed-index`, iroh-blobs, `doctor` |
| Continuous index and vector upkeep | **Main — next release** | Direct Obsidian edits and automatic maintenance advance the local catalog; generation changes hydrate shared/peer vectors and derive only missing content hashes | daemon watcher, vector worker, `status`, `doctor` |
| Privacy and safety | v0.9.9 | Secret-shaped files are excluded, `<private>` regions are blanked before indexing, and hooks never approve tools | built-in boundaries, local state |
| Cross-platform delivery | v0.9.9 | Linux, macOS, and Windows block releases; packages are available through Cargo, Homebrew, Nix, AUR, and release archives | `variants`, `hardware` |

## Benchmarks

cfetch measures output reduction where the rewrite happens, retaining both the
original and model-facing sizes. With the v0.9.9 release binary, 13 real command
outputs from the public cfetch and dotkeeper repositories produced these results:

| Measurement | Result |
|---|---:|
| Oversized outputs rewritten | 8 |
| Original size | 86,469 estimated tokens |
| Model-facing size, including preservation pointers | 5,702 estimated tokens |
| Context avoided | **80,767 estimated tokens (93.4%)** |
| Median reduction per rewritten output | **91.2%** |
| Short-output controls passed through unchanged | 4 of 4 |
| 2,150-line CI test-log control passed through unchanged | yes |

The benchmark deliberately selects oversized repository listings, search
results, and version-control history—the output families cfetch is designed to
condense. Token counts use cfetch's labeled characters/3.5 estimate; the byte
measurements themselves are taken from the real hook input and replacement.
Test and build output is never rewritten, because the useful failure can be in
the middle. Full inputs, commands, pinned commits, and per-case results are in
[the benchmark study](studies/context-reduction-v0.9.9.md).

This is an output-reduction result, not a whole-session savings claim. For that,
`cfetch bench` compares transcript-grounded cfetch and bare runs with identical
first prompts and refuses to report a paired delta below three task pairs.

## OpenWolf, OpenWolf Enhanced, and cfetch

cfetch was informed by the public behavior and lessons of
[OpenWolf 2.x](https://github.com/cytostack/openwolf#readme) and the separate
[OpenWolf Enhanced 1.x](https://github.com/bassprofressor-lab/openwolf-enhanced#readme)
lineage. It is an independent Rust implementation with a different storage,
deployment, trust, and sharing model.

This map answers the practical question: which ideas exist in each project,
and how are they experienced?

This comparison follows cfetch `main`. Rows marked as next-release features in
the feature map are not part of the v0.9.9 binaries.

| Capability | OpenWolf 2.x | OpenWolf Enhanced 1.x | cfetch (`main`) |
|---|---|---|---|
| Memory scope | One `.wolf/` memory per project | One bounded `.wolf/` memory per project | Any Markdown tree, composed from nested slices |
| Session startup | Budgeted project digest and handoff | Smart resume digest and structured summaries | Budgeted, trust-aware, host/repository-scoped injection |
| Compaction survival | Restores state, rules, and scoped instructions | Carries the 1.x session state forward | Restores modified-file evidence and re-arms relevant guidance |
| Knowledge retrieval | Project index, symbol search, and bug search | BM25, semantic/hybrid recall, citations, and MCP | BM25, semantic/hybrid recall, reranking, citations, wikilinks, and slice filters |
| Code navigation | Symbol ranges and import-aware project map | `find`, optional tree-sitter ranges, and PageRank | Tree-sitter `find`, exact ranges, import graph, and token-fitted `map` |
| Context reduction | Repeat-read awareness, file-size hints, and Bash output governor | Same 1.x family plus bounded storage controls | Repeat-read guidance, slice hints, answer budgets, precision gate, and output condensation |
| Memory capture | Corrections, bug fixes, action log, and handoff files | Optional activity capture, bug memory, lint, and distillation | Redacted exhaust → deterministic flags → autonomous evidence-grounded maintenance |
| Measurement | Real transcript usage, verified delivery, cache attribution, and A/B bench | Estimated ledger including its own injection cost | Transcript usage, rewrite deltas, injection cost, cache attribution, audit, and paired bench |
| Health and UI | Heartbeats, selfcheck, and token-authenticated web dashboard | Project health/size doctor, cleanup checks, and expanded web dashboard | Terminal-only Activity, Graph, Recall, and System views plus `doctor`, `status`, and `selfcheck` |
| Agent reach | Full integration for Claude Code, Codex, and OpenCode; context for several others | Hooks for four agents plus MCP clients | Confirmed configuration surfaces across 25 harnesses; native hooks only where the payload contract is understood |
| Sharing | Git carries useful project state | Optional explicit push to a linked workspace | Authenticated serving, per-slice grants, and content-verified vector delivery over iroh |
| Maintenance extras | Project update/restore, cron, and skills | Doctor, lint, distill, export, Design QC, and optional AI tasks | Constant change-driven AI maintenance, independent review, exact-byte gates, immutable history, pause/interject controls, and safe revert |
| Runtime | Node.js 20+, per-project hook files | Node.js 20+, per-project hook files | One Rust binary plus an optional per-host daemon |
| License | AGPL-3.0 | AGPL-3.0 | FSL-1.1-ALv2, converting to Apache-2.0 after two years |

cfetch keeps Markdown—not a hidden graph or vector database—as the record.
Captured evidence crosses the staging boundary only after an independent
semantic review and deterministic checks, but the healthy path is autonomous.
A person interjects by editing Markdown, pausing the worker, inspecting history,
or running the same transaction commands manually for debugging.

## How cfetch works

### Plain Markdown is the record

The knowledge tree is the only fact store. The local SQLite catalog, symbol
index, fingerprints, and vector cache can be deleted and rebuilt without
losing knowledge. This also means ordinary editors, git history, Obsidian,
shell tools, and code review continue to work.

### Selective nixcards knowledge

`cfetch cards init` creates a `blob:none` partial clone of the canonical nixcards `cards` branch at
`<brain_root>/knowledge/cards`. The checkout's Git sparse-checkout list is the only local selection
record: cfetch and the nixcards Ratatui interface read and update the same state, so neither tool can
overwrite a shadow configuration owned by the other.

```console
$ cfetch cards init
$ cfetch cards list
$ cfetch cards select cloud.bearingpoint.interview
$ cfetch cards status
$ cfetch cards tui
```

Dotted selectors may name one set or a whole hierarchy branch. The root `catalog.json` remains
local so either interface can show the complete taxonomy; only the selected set directories and
their Markdown blobs materialize. `cards sync` fast-forwards the catalogue without changing that
selection. Store mutations share one bounded cross-platform lock, and the daemon watches selected
files even when the outer brain repository ignores the nested checkout.

This managed checkout is also the contribution source. A correction made while reading under
`knowledge/cards` is an ordinary Git change that can be committed on a topic branch and proposed
directly against nixcards' `cards` branch. `cards sync` refuses to overwrite local changes.

The cards carry no cfetch trust setting. Their physical `knowledge/cards/...` paths pass through
the brain's normal ordered `ring_rules`, exactly like manually written knowledge. The nixcards
application remains independent: cfetch is not required for its TUI, web application, catalogue,
or progress store.

### Trust is visible in every citation

cfetch uses configurable rings to distinguish critical constraints from raw
session capture. A lower ring number means higher trust; the ring prefix is
part of every citation.

| Ring | Typical content | Default context behavior |
|---|---|---|
| 0 | Critical invariants and safety constraints | Eligible for explicit resident injection |
| 1 | Durable policy and settled decisions | Eligible for explicit resident injection |
| 2 | Distilled behavior and scoped guidance | Injectable only with an explicit scope |
| 3 | Curated knowledge and documentation | Retrieved on demand |
| 4 | Current tasks and working state | Retrieved on demand |
| 5 | Review candidates | Never recalled or injected implicitly |
| 6 | Redacted raw capture | Never recalled or injected implicitly |

Ring assignment defaults by path and can be overridden with `ring: N` in
frontmatter. Rings express trust, not authorization; slice grants control who
can access which content.

### AI maintenance is continuous, inspectable, and reversible

> Available on `main` for the next release; not included in v0.9.9 binaries.

The daemon reacts to new ring-5 evidence after a quiet period. It builds a
bounded packet, asks one model pass for a typed proposal, asks a fresh isolated
pass (optionally a different model) for semantic review, re-runs deterministic
checks, and applies the exact complete Markdown bytes. Healthy maintenance does
not wait for a person, a browser, a git commit, or an approval click.

```json
{
  "maintenance": {
    "endpoint": "http://127.0.0.1:8080/v1",
    "model": "memory-maintainer",
    "review_model": "memory-reviewer",
    "api_key_env": "MAINTENANCE_API_KEY"
  }
}
```

```console
$ cfetch daemon start
$ cfetch maintain history            # immutable outcomes and exceptions
$ cfetch maintain run                # request one bounded cycle now
$ cfetch maintain pause debugging    # interject without losing evidence
$ cfetch maintain resume
```

Obsidian and direct Markdown edits are authoritative. Every proposal captures
the exact before bytes; if the file changes before apply, cfetch records an
exception and leaves the new bytes untouched. Applied events retain before and
after hashes, candidate ids, review ids, checks, and rationale. A transaction
can be reverted only while its exact applied bytes still match, so revert also
refuses to overwrite a later human edit.

Ring 0 and ring 1 remain protected. Automatic ring-2 changes require direct
operator instruction in the evidence itself; a model cannot manufacture that
authority. Attested evidence may update rings 3 and 4, unendorsed claims cannot
cross into trusted memory, secret-shaped content is refused, and symlinks or
stale targets fail closed.

The Activity pane and `cfetch doctor` show whether maintenance is local or
remote, which proposal and review models are configured, what is staged, what
changed, and why anything stopped. See [Autonomous AI memory maintenance](docs/ai-maintenance.md)
for the full transaction contract and debugging workflow.

### Retrieval is layered, bounded, and explicit

Lexical BM25 recall works immediately. Semantic and hybrid search are optional,
use content-addressed vector artifacts, and report partial coverage instead of
quietly pretending to be semantic. An optional cross-encoder reranks the
retrieved shortlist. Every text answer has a token budget, so asking for ten
hits cannot unexpectedly fill the context window.

```console
$ cfetch recall --hybrid "request retry policy"
$ cfetch recall --slice engineering "deployment rollback"
$ cfetch recall --expand "database migration"
$ cfetch retrieval-fixture
$ cfetch doctor --deep
$ cfetch doctor --deep --require production
```

`cfetch retrieval-fixture` builds a small temporary Markdown tree and compares
BM25, vector search, hybrid fusion, optional reranking, and graph expansion on
the same fixed query. It says `ACTIVE` only after the configured embedding
route successfully returns both query and document vectors. The normal view
shows vector dimensions, encoding, checksums, a short component preview, and
cosine scores; `--show-vectors` prints every component and `--json` provides a
stable machine-readable report. It never reads or changes the configured brain,
index, or shared vector store.

`cfetch doctor --deep` includes the same test in the wider system report. It
also explains the job of each configured model and distinguishes “configured”
from “actually answered.” Deep mode may start a packaged local model or contact
the configured model endpoints. The ordinary `cfetch doctor` command does not.

Both commands now publish explicit pass, fail, and not-run gates. Add a
repeatable `--require` for BM25, profile admission, vectors, hybrid fusion,
reranking, graph expansion, local acceleration, or the complete production
set to turn the report into a nonzero-exit CI or release check. The production
gate requires an active canonical profile and evidence that the call used a
real local NPU, GPU, or accelerated CPU package; a working test endpoint is
not enough. See [retrieval readiness gates](docs/retrieval-readiness.md).

### Graphs and vectors remain derived from readable files

Source imports form a separate, disposable code graph. `code-graph path`
explains one deterministic shortest chain of project-internal imports;
`code-graph impact` walks the same edges backwards to expose a bounded blast
radius. `code-graph context` traverses both incoming and outgoing edges around
one file, retaining one deterministic shortest explanation edge per related
file instead of returning an unbounded induced subgraph. `code-graph symbol`
adds `contains`, direct `calls`, and type `references` relationships around an
exact symbol. A call or reference resolves only through an explicit import to
exactly one file-level definition; dynamic, local-only, or ambiguous names stay
unresolved instead of becoming guessed edges. Every relationship carries its
exact source range, result limits count what they omit, ambiguous file suffixes
fail instead of guessing, and paths stay relative when served by another
machine. Direct calls cover Rust, TypeScript/JavaScript, Python, and Go; safe
grammar-distinguished type references currently cover Rust, TypeScript, and Go.

```console
$ cfetch code-graph path src/main.rs src/runtime/worker.rs
$ cfetch code-graph context src/runtime/worker.rs --depth 2 --limit 50
$ cfetch code-graph impact src/config.rs --depth 4 --limit 50
$ cfetch code-graph context src/config.rs --json
$ cfetch code-graph symbol resolve_request --limit 50 --json
```

`[[Obsidian wikilinks]]` are the knowledge graph. cfetch resolves them only
when a target is unambiguous, exposes incoming and outgoing relationships, and
can center a bounded neighborhood on one note. It does not require a separate
graph database or make hidden relationships authoritative.

```console
$ cfetch graph
$ cfetch graph --focus deployment --limit 30
$ cfetch graph --focus knowledge/runbooks/deployment.md --json
$ cfetch graph --slice engineering --focus deployment
```

Vectors follow the same rule. They are keyed by statement content hashes and
stored as reusable artifacts; the daemon watches Markdown generations,
hydrates vectors already produced by the storage group or authorized peers,
and sends only missing hashes to the configured endpoint. Direct Obsidian edits
therefore update lexical search, graph edges, and semantic coverage without an
export step.

### Freshness is part of the answer

A machine holding the Markdown can serve recall, citation expansion, knowledge
graphs, code search, repository maps, and dependency explanations to other
machines. Every response carries an origin, catalog generation, and `fresh`
flag. Queries pass a bounded drain
barrier: cfetch either proves that visible writes have reached the catalog or
labels the answer stale and explains why.

```console
# Storage machine
$ cfetch daemon start

# Client configured with client.serving
$ cfetch recall "release checklist"
...
served by docs-host (generation 42, fresh)
```

## Agent integrations

`cfetch install` detects initialized agent configurations and adds only the
surfaces each agent actually supports. Entries are ownership-ledgered,
idempotent, backed up on first touch, and removed symmetrically with
`cfetch install --remove`.

```console
$ cfetch install                         # detected agents
$ cfetch install --agent claude --agent codex
$ cfetch install --project . --all       # confirmed project-local surfaces
```

Native hooks are capability-gated. Claude Code, Codex, and CodeBuddy receive
the full lifecycle where available; Gemini, iFlow, and Tabnine receive only
the verified tool-event subset. Other supported agents receive MCP and/or
recall-first instructions according to their confirmed configuration format.

### Runtime visibility in Claude Code, Codex, and MCP

cfetch exposes one cached `RuntimeStatusV1` contract everywhere instead of
letting each integration guess. It distinguishes three different facts:

- **configured** is the requested inference mode;
- **selected** means a backend initialized successfully;
- **last used** records an actual local or remote attempt and its outcome.

Hardware detection alone never claims that an NPU or GPU is in use. Cached
freshness is likewise written as history (`last fresh`), not as a promise that
the tree is fresh now.

```console
$ cfetch status --line   # one terminal-width line; no network or inference
$ cfetch status --json   # stable, machine-readable RuntimeStatusV1
```

`cfetch install --agent claude` registers the cached line as
[Claude Code's native status line](https://code.claude.com/docs/en/statusline)
with a five-second refresh. An existing non-cfetch status
line is preserved; compose `cfetch status --line` into its command or use the
explicit `--replace-status-line` option. Removal deletes only the exact
cfetch-owned value.

Codex currently documents
[built-in footer items](https://learn.chatgpt.com/docs/config-file/config-reference)
rather than an arbitrary command-backed footer. cfetch therefore uses
[native hooks](https://learn.chatgpt.com/docs/hooks): one short visible
notice at session start, then notices only when the memory route, selected
backend, or failure severity changes or recovers. Healthy state spends no
model-context tokens; a short adaptation note is added only when degraded
runtime state should change the agent's behavior.

MCP clients can call the read-only `cfetch_runtime_status` tool when runtime
health affects a task. Its cached response is bounded to 2 KiB and its tool
description explicitly says not to poll it. RuntimeStatusV1 surfaces never include
endpoint URLs, raw addresses, token paths, credentials, response bodies, or
hardware evidence paths.

The same line and JSON contract report autonomous-maintenance state: disabled,
not configured, local or remote and idle, paused, exception, or model failure.
They also report the proposal and review model labels, candidate count, history
count, last outcome, and last inference attempt without exposing endpoint
details. Agent integrations can therefore say that maintenance is active—or
why it is not—without injecting a recurring explanation into model context.

<details>
<summary>Show the current adapter registry</summary>

```text
claude cursor gemini openclaw hermes codex copilot opencode cline roo
windsurf kilocode antigravity antigravitycli amp codebuddy crush forge
iflow junie pi qodercli qwen tabnine trae
```

</details>

For any MCP client, the manual registration is the standard stdio shape:

```json
{
  "mcpServers": {
    "cfetch": { "command": "cfetch", "args": ["mcp"] }
  }
}
```

The v0.9.9 server exposes read-only `cfetch_recall`, `cfetch_expand`, and
`cfetch_find`. On `main` for the next release, it also exposes read-only
`cfetch_code_path`, `cfetch_code_impact`, `cfetch_code_context`,
`cfetch_code_symbol`,
`cfetch_runtime_status`, `cfetch_maintenance_packet`, and
`cfetch_maintenance_show` tools.
`cfetch_maintenance_propose` and `cfetch_maintenance_review` can write only
idempotent ring-5 records for debugging and intervention. Apply, revert, reject,
and finalize remain CLI-only; MCP cannot promote or overwrite trusted memory.

## System diagnostics and hardware visibility

`cfetch doctor` is the evidence-rich debugging surface behind the dashboard's
System pane. It answers the practical questions that a compact agent status
line should not try to fit:

- which CPU, GPU, and NPU devices the operating system exposed, including the
  evidence used to identify them;
- whether each device is architecturally usable, supported by this binary,
  selected by the runtime, or merely present;
- which embedding profile, model revision, artifact, vector encoding, endpoint
  route, and optional reranker are configured, plus what each model does;
- which backend was successfully selected and which inference route was last
  attempted;
- whether the daemon and its authenticated network endpoint are running, and
  whether joined origins currently answer through the granted serving path;
- vector coverage, shared artifact count, hook liveness, outbound grants, and
  concrete repair actions;
- whether peer artifact delivery is ready, how many authorized routes exist,
  and the actual resolution order: shared store, authorized peers, then the
  configured embedding endpoint;
- whether autonomous maintenance is configured and running, its local or
  remote route, proposal and review models, pending candidates, immutable
  outcomes, tampered or unreadable history, exceptions, and last model attempt.

```console
$ cfetch doctor               # one read-only diagnostic report with bounded peer probes
$ cfetch doctor --deep        # call retrieval models on temporary data and compare every stage
$ cfetch doctor --deep --show-vectors # include every vector component
$ cfetch doctor --deep --json # include the retrieval test in DoctorReportV1
$ cfetch doctor --deep --require production # fail unless production retrieval is ready
$ cfetch doctor --json        # stable, machine-readable DoctorReportV1
$ cfetch doctor --no-network  # inspect local state without contacting peers
$ cfetch doctor --tui         # open the live, scrollable System pane
```

The wording is deliberately strict. Hardware **detected** is not hardware
**selected**. A remembered membership is not called reachable until an
authenticated, slice-authorized request answers. A completed inference attempt
is not presented as current utilization. Until a backend exposes a real device
counter, utilization is shown as `not_reported` instead of an invented
percentage.

Doctor does not compact files, repair configuration, or apply maintenance. Its
normal mode does not call an inference model. The explicit `--deep` mode calls
only the embedding and optional reranking routes, using temporary fixture data;
it does not call the maintenance models or alter the configured brain, index, or
shared vector store. `cfetch selfcheck` remains the installation verifier with a
nonzero exit on hard failures; doctor is the wider evidence report for debugging
and support.

## Installation

### Package managers

```console
# Homebrew on macOS or Linux
brew tap corbet-labs/cfetch
brew install cfetch

# crates.io on any platform with Rust 1.95+
cargo install cfetch --locked

# Arch Linux (AUR package name avoids an unrelated cfetch package)
paru -S cfetch-agent

# Nix flake
nix profile install github:corbet-labs/cfetch
```

Prebuilt archives for Linux, macOS, and Windows are attached to
[GitHub releases](https://github.com/corbet-labs/cfetch/releases/latest).
Every archive includes the binary, cfetch license, generated third-party
notices, and embedded variant metadata. `cfetch variants` prints the exact
catalog used to build the release.

### Local inference status

The current public archives are endpoint builds. The former local ORT package
and exact-byte certification route were a pre-activation experiment, not
released NPU/GPU/CPU support, and have been retired.

Local packages will return as an NPU-first family: an admitted NPU is selected
first, then an admitted GPU, then an admitted accelerated CPU implementation. Each may
use its runtime-native artifact and internal precision, then cfetch encodes its
output into the shared signed INT8x768 index format. NPU-first is the execution
order, not a numerical anchor. Remote inference remains an explicit
configuration choice, not a substitute for local capability or an automatic
fallback.

The immediate packaging boundary is an attested local adapter over loopback:
Core ML, LiteRT, OpenVINO, or another native runner remains on the same device
while cfetch owns the semantic-profile check and canonical output encoding.
See the [candidate embedding profile](docs/embedding-profile-v1.md),
[local inference plan](docs/local-inference.md), and
[local adapter contract](docs/local-embedding-adapter.md). Existing Core ML
and LiteRT artifacts are candidates to adapt and test, not certified support.

### Build the development branch

```console
$ git clone https://github.com/corbet-labs/cfetch.git
$ cd cfetch
$ cargo build --release
```

Linux, macOS, and Windows all gate releases in CI. Platform-specific local
control transport is hidden behind the same CLI: Unix sockets on Linux/macOS,
and authenticated loopback TCP on Windows.

See [CI checks and provider coverage](.ci/README.md) for selected public checks,
Crow execution, exact-input evidence, and the remaining native/release routes.

## Configuration

cfetch searches for one JSON configuration in this order:

- `CFETCH_CONFIG`, when set;
- `<brain_root>/.cfetch/config.json` for tree-wide policy; then
- `~/.config/cfetch/config.json` on Linux/macOS or
  `%APPDATA%\cfetch\config.json` on Windows for host-specific settings.

The first existing file wins; configuration files are not layered. Missing
fields use built-in defaults.

Environment variables `CFETCH_BRAIN` and `CFETCH_STATE_DIR` override the
knowledge and local-state paths.

A compact starting point:

```json
{
  "brain_root": "/home/you/agent-memory",
  "resident": [
    { "path": "rules/critical.md", "ring": 0 },
    {
      "path": "guidance/rust.md",
      "ring": 2,
      "scope": { "repos": ["api-service"] }
    }
  ],
  "code_roots": ["projects"],
  "budget_chars": 6000,
  "exclude_prefixes": ["vendor/", "archive/", "scratch/"]
}
```

Ring rules are ordered; the first matching path prefix wins:

```json
{
  "ring_rules": [
    { "prefix": "rules/critical.md", "ring": 0 },
    { "prefix": "rules/", "ring": 1 },
    { "prefix": "guidance/", "ring": 2 },
    { "prefix": "tasks/", "ring": 4 },
    { "prefix": "staging/", "ring": 5 },
    { "prefix": "", "ring": 3 }
  ]
}
```

Semantic recall is off by default. The OpenAI-compatible transport supports
both packaged loopback adapters and explicit remote deployments. A producer's
profile strings are not sufficient: it must attest an exact execution scope,
artifact digest, device class, and accelerated placement that this release's
registry admits, then sign the nonce-bound request and exact response with that
package scope's pinned Ed25519 key. Main currently admits none, so semantic
production fails closed while the candidate matrix is built. Keep any
credentials in an environment variable, never in JSON.

Network major 1 fixes the EmbeddingGemma source revision, tokenizer, full 768
dimensions, query/document prompts, pooling, normalization, and the signed
INT8x768 output/index record. It does not freeze internal weight or activation
precision, one vendor graph, or one backend's numerical output. Target-native
NPU, GPU, and accelerated CPU packages must attest the semantic profile and
pass the global ordered query-backend x document-backend retrieval matrix
and the adversarial mixed-document-store test against the same absolute quality
floors before they ship.

```json
{
  "embeddings": {
    "enabled": true,
    "endpoint": "https://embeddings.example/v1",
    "api_key_env": "CFETCH_EMBEDDINGS_KEY"
  }
}
```

Run `cfetch embed-index` for an explicit backfill or debugging pass. With the
daemon running, catalog generation changes trigger the same shared-first
resolution automatically. Vectors are keyed by statement content hash, so
editing one file processes only changed statements. Missing or partial vector
coverage is always reported.

After the profile and registry are active, `embed-index` checks the shared tree
first, then requests only its missing content hashes from each authorized
slice. Matching canonical records stream over iroh-blobs with BLAKE3
verification; only the remainder reaches the configured embedding endpoint.
Artifact capabilities are salted and isolated by authenticated peer, and the
receiver verifies the exact profile, content hash, record width, and canonical
codec before appending them. The compact record carries no per-vector producer
receipt, so the authenticated authorized storage group is the peer trust
boundary. A host whose peers cover every block can then complete `embed-index`
with its local embedding endpoint disabled and zero endpoint calls. While the
profile or registry is inactive, hydration and peer transfer fail closed.

Run `cfetch embedding-profile --json` for the executable manifest. Changing
a semantic pipeline field is a new incompatible network major and requires
re-embedding. Runtime-native artifact hashes and internal precision remain
backend-scoped. There is no reference backend: exact bytes across device
families are not required, while same-scope repeatability and the full ordered
pair plus adversarial mixed-store absolute retrieval gates are mandatory. See
[embedding profile v1](docs/embedding-profile-v1.md) and
[local accelerated inference](docs/local-inference.md).

Named slices limit recall and sharing by path:

```json
{
  "slices": [
    { "name": "engineering", "prefixes": ["knowledge/engineering"] },
    { "name": "backend", "prefixes": ["knowledge/engineering/backend"] }
  ]
}
```

Every document belongs to its innermost slice. Unknown slice names are refused
instead of widening to the whole tree.

Hard exclusions cannot be disabled: secret directories, logs, git internals,
and secret-shaped filenames never enter recall, and their paths are withheld
from capture. Configured `exclude_prefixes` and `.cfetchignore` add
project-specific boundaries.

## Frequently asked questions

### Does cfetch replace built-in agent memory?

No. It can index supported native memory read-only and adds a shared,
agent-independent retrieval and governance layer. You can run both without
duplicating resident injection.

### Does cfetch require a vector database or cloud service?

No. Lexical recall and code search are local and work without embeddings.
Semantic search can use a supervised local package or an explicitly configured
remote-attested service with the OpenAI embeddings request shape. A generic
OpenAI-compatible endpoint lacks the admitted transport and scope metadata.
Local identity comes from cfetch's hash-pinned package and supervised-child
boundary; remote identity additionally uses an operator-held response key.
The per-host catalog is disposable SQLite; Markdown remains the record.

### Is cfetch only for Claude Code?

No. Claude Code and Codex have the deepest native integration, while other
agents connect through verified hook subsets, MCP, and instruction files.

### Is captured session text trusted automatically?

No. Capture is redacted and placed in ring 6. Deterministic signals may create
a ring-5 candidate, but neither ring is implicitly recalled or injected.
Promotion into curated memory requires an independent semantic review plus all
deterministic evidence, authority, trust, path, secret, and exact-byte gates.
The healthy path is automatic; it is not blind trust in captured text.

### Does cfetch use AI to maintain memory?

On `main` for the next release, yes. The daemon reacts to changed evidence,
runs a bounded proposal pass and a fresh isolated review pass, rechecks every
deterministic gate under a write lock, and applies exact Markdown bytes without
routine approval. Direct Obsidian edits remain authoritative, maintenance can
be paused globally or per file, every outcome is journaled, and safe revert
refuses to overwrite newer work. MCP cannot promote a proposal into trusted
memory.

### How is cfetch different from OpenWolf?

OpenWolf centers a project-local `.wolf/` directory and Node.js lifecycle
hooks. cfetch centers an arbitrary Markdown tree, trust-aware citations,
token-bounded retrieval, a single Rust daemon, and authenticated slices that
can be served across machines. See the
[feature comparison](#openwolf-openwolf-enhanced-and-cfetch).

## Security

- Hooks never emit an automatic permission approval.
- Internal hook failures exit successfully and degrade to silence, so memory
  infrastructure cannot trap or break an agent session.
- Secret-shaped paths and private regions are filtered at collection time.
- Serving and Windows local control channels use bearer tokens and
  constant-time comparison.
- Configured URLs require HTTPS or loopback unless a host is explicitly
  allowed; redirects cannot carry credentials across the boundary.

Report suspected vulnerabilities through the private process in
[SECURITY.md](SECURITY.md).

## License and contributing

[FSL-1.1-ALv2](LICENSE.md): use, modify, and redistribute cfetch for any
purpose except offering a competing commercial product or service. Each
release converts automatically to Apache-2.0 two years after publication.

Contributions are welcome. Fork cfetch, send a focused pull request, and agree to
the organization-wide [Individual Contributor License Agreement](https://github.com/corbet-labs/.github/blob/cla-v1.0/CLA.md).
Contributors retain copyright and grant the project the rights needed to license
accepted contributions coherently. Read
[CONTRIBUTING.md](CONTRIBUTING.md) and follow the clean-room rule: do not copy or
closely paraphrase code from OpenWolf, OpenWolf Enhanced, or other incompatibly
licensed projects.

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
