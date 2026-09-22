# Changelog

Notable changes per release. cfetch is young and moving fast; anything not
listed here is a fix or an internal change with no effect on behavior.

## Unreleased

## 0.10.2

## 0.10.1

- Ship the local-memory rewrite with an explicit manual release trigger. The 0.10.0 tag produced no published artifacts.
- Retain ONNX Runtime notices in Homebrew installations.

## 0.10.0 (unpublished)

- Replace peer sharing and remote computation with ordinary Git repositories and local retrieval.
- Add qualified offline CPU embeddings, hybrid search, Obsidian graph paths and independent mind selection.
- Build CPU retrieval into Linux, Apple Silicon and Windows release packages; Intel Mac retains lexical and graph retrieval while a native ONNX runtime is unavailable.
- Package Nix native dependencies as pinned, sandbox-compatible inputs.
- Preserve direct Markdown edits and report incomplete semantic coverage explicitly.

- **Resumable physical qualification.** Optional journals reserve nonces before
  requests and persist authenticated responses. Resume revalidates complete
  trials and preserves interrupted evidence; partial trials restart with fresh
  warmups. Final snapshots are durable and refuse overwrite. The admission
  implementation and policy identities include the new checkpoint code.
- **Canonical long-input OpenVINO attention.** Conversion now uses the pinned
  model's effective bidirectional window. Independent upstream parity includes
  actual 258- and 1,946-token probes, detecting defects hidden by padded short
  inputs. The corrected CPU candidate is measured; backend admission is open.
- **Host-wide native operation budgets.** Operational OpenVINO startup requires
  an explicitly provisioned, evidence-bound governor policy. Every compilation
  and inference has a durable lease, finite budget, cooldown and kernel alarm.
  Interrupted operations refuse further work until external recovery; no
  device safety limits or activation are supplied automatically.
- **Exact statement identity and vector invalidation.** Case, spacing, and
  line-break changes produce distinct body hashes. Schema 9 separates those
  citation hashes from vector payload hashes: documents include enclosing
  headings and full table headers under the unchanged encoder prefix.
  Context-only edits invalidate dependent vectors and preserve citations.
  Legacy vectors remain stored but cannot hydrate the new keys; identical
  table rows under different headers remain distinct recall hits.
- **Bounded derivation failures.** Only an explicit overlength-input refusal
  triggers row isolation. Transport, authentication, scope, malformed output,
  and invalid-vector failures stop immediately. Refused rows are not retried
  during the same run; completed work remains resumable. An unresponsive
  adapter is stopped, and failed scopes remain disabled across its restart.
- **Hybrid recall keeps lexical trust reservation.** The lexical input to
  fusion now applies the same single top-trust slot as standalone recall,
  preventing fusion from silently using a different lexical ordering.
- **Replayable memory-retrieval measurements.** `retrieval-eval` exports exact
  inputs and evaluates labeled synthetic Markdown with the real rankers and
  INT8 codec in an isolated catalog. Experimental heading context and imported
  candidate vectors do not activate or change production embedding semantics.
- **Resumable signed adapter exports and bounded child cleanup.** Opt-in
  export checkpoints authenticate completed transactions on resume while
  keeping independent repeatability trials separate. Package-local adapter
  cleanup has a deadline and refuses further launch after unconfirmed cleanup
  or exhaustion of the single crash restart.
- **Explicit, read-only FastEmbed model diagnostics.** `embed-model --model`
  passes the selected catalogue variant to the loader, including distinct
  quantized variants. `list` uses the linked library's actual catalogue.
  `status` observes the selected cache directory, including `HF_HOME`, without
  downloading or loading anything. Removed `check-compat` and
  `switch-to-shared`: model-name matching cannot certify the shared semantic
  pipeline, and the old switch discarded its selected model. Diagnostic
  output remains isolated from the canonical vector store; query-local
  reranking is unchanged.
- **`--settings <file>` stops double-installing Claude's hooks.** The
  explicit settings path redirected `apply_claude`'s hooks and status
  line, but agent-config's Claude hook surface still wrote the DEFAULT
  `~/.claude/settings.json` — a double registration that created the
  very files the operator pointed elsewhere to avoid (four other files,
  three directories, in the reporting tree). With an explicit `--settings`
  file, the default-path hook surface is deselected: hooks and status
  line land only in the named file. MCP and instruction surfaces keep
  their own locations.
- **The hard-rule cap counts characters, not bytes.** A 120-character
  German rule is 208 UTF-8 bytes and silently dropped while its English
  twin passed — the cap is a readability limit and now measures what the
  operator reads. The English-only opener set (`never`, `no`, `don't`,
  `do not`) is unchanged and recorded in #69 as a design question.
- **Commands stop reporting success they did not achieve.** Three from the
  sweep. (1) `daemon stop` printed `daemon stopped` on a mere
  acknowledgement — the acknowledged process outlived the claim by minutes
  on the reporting host. Stop now polls until the daemon stops answering,
  and the daemon bounds its own shutdown with a five-second watchdog: the
  graceful tail (iroh endpoint close, listener cleanup) can block on
  network drains, and a daemon that acknowledged shutdown must not become
  a zombie. A stop that cannot confirm exits nonzero instead of lying.
  (2) `daemon start` sent the actual startup failure to /dev/null and
  reported only "did not answer". The daemon's stderr now lands in
  `state/daemon-start.stderr.log` (truncated per start), and a failed
  start includes its last lines in the error. (3) `cards sync` completed
  fetch and merge, then deadlocked on its own store lock: it held the lock
  and called `status()`, which acquires it again. The report now runs
  under the lock `sync` already holds.
- **selfcheck stops claiming what it never tested.** (1) The ring-6 line
  said `ok … (this host writes)` without a single write: on a read-only
  brain that was a lie a live hook then paid for, silently dropping every
  exhaust record while the diagnostic nodded. When the exhaust directory
  exists, selfcheck now writes and removes a probe file — `ok … writable`
  is a measurement; a probe failure is a FAIL naming that hooks silently
  drop records. When the directory does not exist, selfcheck does not
  create it (a diagnostic must not be the first writer into the brain)
  and says so: `writability untested; the first hook write decides` —
  no more untested claim. (2) A resident entry pointing at a missing
  file counted as a healthy digest; the ~154 characters reported were
  placeholder text, injected into every session. The placeholder is still
  injected (a session should learn its contract broke), but
  `ResidentDigest.missing` now carries the labels and selfcheck warns:
  `resident file missing — the entry injects only a placeholder: …`.
- **doctor sees the catalogue.** A never-scanned, stale, unopenable, or
  randomly overwritten index — and a missing brain root — all reported
  `0 critical, exit 0` while selfcheck or status detected each one, and
  the `--json` contract had no field for catalogue state at all. doctor
  now takes the same liveness measurement selfcheck prints (the two
  diagnostics cannot disagree about the same index): a new `catalog`
  field (`current` / `stale` / `never_scanned` / `unavailable` /
  `remote` with detail), findings for `index_never_scanned` and
  `index_stale` (warnings, action: run `cfetch scan`), `index_unopenable`
  (critical), and `brain_root_missing` (critical). A none-tier client
  reports `remote` by design instead of building a second, silently
  stale local catalog.
- **Index losses are reported, not silent.** Three drops the sweep found
  in a real tree are now lines in `cfetch scan` output. (1) An unclosed
  `<private>` — often a mere mention of the tag in prose — blanks to end
  of file fail-closed (that stays the design); the file is now named:
  `<private> never closed — content blanked to end of file: …` (the
  reporting tree lost 84% of one file and 33% of another while scan said
  `0 skipped`). (2) Files that could not be read (invalid UTF-8, or a
  Windows sharing violation) were reported under the ring-5+ skip reason,
  sending the operator to look for a frontmatter that was not the
  problem; they now carry their own line: `unreadable (invalid UTF-8 or
  locked): …`. (3) Statements dropped as generated blobs (a line over
  8 KiB) are counted: `N statement(s) dropped as generated blobs`.
  `ScanReport` gained `unreadable`, `private_swallowed`, and
  `blob_dropped` for the same visibility in the daemon path.
- **The secrets bar holds for the code index, and a rescan heals it.**
  Pointing `code_roots` at — or beneath — the brain's hard-excluded
  prefixes used to make `find` return paths, symbol names, and exact line
  ranges from `mind/secrets/`. The code walker now refuses those prefixes
  at any configuration (component-wise path comparison, so a root outside
  the brain never matches by string accident), a bare file root inside a
  refused prefix is skipped whole, and the next scan purges whatever an
  earlier configuration let in — refused rows are unseen rows, and unseen
  rows are stale. `scan_code` gained a brain-root parameter internally
  (`scan_code_in`) so the bar is testable without mutating the process
  environment.
- **The secrets bar holds at the resident layer, and buglog ids can no
  longer traverse.** Two fixes from the six-area sweep. (1) The strongest
  guarantee the tool makes — "never indexed, recalled or injected, at ANY
  configuration" — broke at the one mechanism that injects without recall
  being asked: a resident entry pointing under `mind/secrets/` (or
  `logs/`) injected the secret into every session, whatever layer the
  entry arrived from, including the agent-written, machine-cloned tree
  config, and selfcheck approved it. Config load now refuses such entries
  loudly; resident entries under the operator's own `exclude_prefixes`
  are refused for the same reason (a file cannot be excluded from the
  index and injected into every session at once). (2) openwolf buglog ids
  were used unchecked as note filenames: `../../x` and `/abs/path/y`
  wrote outside the brain with exit 0 and no report entry. Ids are now
  validated as filenames (`[A-Za-z0-9._-]+`, no leading dot, no `..`
  hops, no doubled extension); an id that cannot become one is a reported
  error with the original id named, never a write.
- **`import openwolf`: the old protocol no longer speaks with ring-1
  authority.** The imported `OPENWOLF.md` used to become `AGENT.md` — the
  top-trust resident file — so the highest-trust instruction a migrated
  brain injected was, verbatim, the operating protocol of the tool being
  replaced: read `.wolf/STATUS.md`, append to `.wolf/memory.md`, log bugs
  to `.wolf/buglog.json`, prefer `openwolf recall`. None of those paths
  exist in a cfetch brain, and the measured consequence was a session
  that "recorded a memory" while the brain stayed byte-identical — the
  write went to the store the old instruction named, outside the tree.
  Now `OPENWOLF.md` migrates verbatim to `knowledge/openwolf-protocol.md`
  (ring 3, recall-only: findable, never injected), and when no operator
  `AGENT.md` exists the import authors one — a migration bridge naming
  where knowledge, bugs, memories, and staging live now, and stating that
  the old protocol's paths do not exist here. The bridge is cfetch's own
  text; an operator's `AGENT.md` is never overwritten, and the same-report
  guarantee covers the authored entry.
- **Hyphenated compounds rank as compounds.** `fts_query` now emits a
  hyphenated query term BOTH as an exact-adjacency phrase and as its split
  prefix terms. The recall contract is unchanged (`state-machine` still
  finds "the machine's state" through the split terms); what is new is that
  an id lookup like `bug-400` finds its note first. Before, the compound
  dissolved into `bug` (near-zero signal once a tree holds hundreds of bug
  notes) and `400` (prefix-matching 400002, 17.400, 86_400_000 elsewhere),
  and the note sank below the default limit — roughly one id in eight from
  a real migrated tree did not come back at all. The phrase restores the
  compound's identity as high signal: measured on the reporter's fixture
  shape, the id note moves from invisible to rank 1 in raw lexical order.
- **`import openwolf`: a migrated tree reaches the model from the first
  scan.** Follow-up to the silent-import fixes. A fresh import used to
  produce a brain that injected nothing at all — selfcheck reported an
  empty resident digest and both session hooks emitted zero bytes — while
  the import output read like success. The real run now writes a starter
  tree config (`<brain>/.cfetch/config.json`, never overwriting an
  existing one) making `AGENT.md` resident at ring 1 and the migrated
  handoff at ring 0, so the hooks inject immediately after `cfetch scan`.
  `STATUS.md` migrates as `knowledge/handoff.md` (ring 0): for a one-way
  import it is a snapshot of project state, not live runtime state, and
  it is the natural ring-0 nomination. An existing config with an empty
  resident list is named in the output instead of leaving the "nothing
  will be injected" state silent. The resident note is identical in dry
  run and real run — the same-report guarantee now covers it.
- **`import openwolf`: honest reporting and the curated bug log.** Three
  fixes from a report filed against a real, grown `.wolf/` tree. (1) The
  dry run and the real run now share one code path — `plan` builds the
  report, the real run executes it — so the preview can no longer hide
  skips it never consulted. (2) Top-level files that match no table are
  reported in a new `found, not recognized — left in place` section
  instead of being dropped in silence (in the reporting tree: live
  instruction files, pre-registrations with their hash evidence,
  hand-written proposals). (3) `buglog.json` is no longer skipped: each
  curated entry becomes one ring-3 note under `knowledge/bugs/<id>.md`
  with the `bug-NNN` id kept in filename and title, so the references
  inside imported files keep resolving and recall returns the specific
  bug instead of nothing. Ring-6 exhaust remains the forward-looking
  error stream; the bug log is imported history, and it does not
  regenerate. A broken or empty bug log is a reported error or skip,
  never an abort.
- **Retrieval readiness gates.** The temporary retrieval fixture and
  `doctor --deep` now publish stable pass, fail, and not-run checks for BM25,
  canonical profile admission, vector output, semantic ordering, hybrid RRF,
  optional reranking, graph expansion, and actual local execution evidence.
  Repeatable `--require` options turn those reports into nonzero-exit CI or
  release gates. The strict `production` requirement cannot be satisfied by a
  custom endpoint or hardware discovery alone.
- **Dependency and supply-chain hardening.** Rust dependencies and GitHub
  Actions are updated as a tested set, SwiftNIO is raised past three published
  security advisories, and the OpenVINO build environment now uses a fixed
  Transformers release. Dependabot coverage now includes the maintained
  Python packaging and Swift manifests. SHA-256 identifiers retain their
  exact lowercase representation through a safe, dependency-free encoder.
- **Host KAT reproduction harness (#56).** Added
  `experiments/embedding-profile/kat_host_runner.py`: replays the bundle's
  recorded session contract on any host and compares canonical `INT8x768`
  outputs against the schema-2 known answers, with flags that refuse the
  two classic measurement traps — silent CPU fallback after a GPU factory
  failure (`--require-provider`) and hybrid CPU remainder (`--strict`).
  A `--baseline-schema1` diagnostic detects hosts that deterministically
  reproduce the superseded saturating-kernel bytes (observed on physical
  Windows Zen 3 with official ORT 1.28.0; see the hardware report on the
  certification issue). Unit-tested in CI; dependency-light at import.
- **Local cross-encoder reranking without an endpoint (#54).** With the
  `embedded-embeddings` feature, an EMPTY `rerank.endpoint` in the config
  now selects the in-process reranker (Jina reranker v2 multilingual via
  fastembed) instead of erroring. A set endpoint keeps the HTTP path
  unchanged. Recall reorders its shortlist locally — zero network, zero
  external server.
- **Model compatibility check for shared vector stores (#53).** When
  multiple hosts share a brain tree via iroh, all hosts must use the
  exact same embedding model — vectors from different models live in
  different spaces and cosine similarity between them is meaningless.
  `cfetch embed-model check-compat` reads the shared store's model
  metadata proactively (instead of waiting for a hydration error) and
  reports Compatible or Incompatible with a fix suggestion.
  `cfetch embed-model switch-to-shared` downloads the correct model
  via fastembed and guides the re-embedding. `cfetch embed-model list`
  shows all available embedding and reranker models.
- **fastembed: 30+ embedding models, local reranking, automatic caching
  (#49, #50).** The embedded backend now uses the `fastembed` crate
  (built on ONNX Runtime + HuggingFace tokenizers), replacing ~300 lines
  of manual session, tokenizer, download, and pooling code with ~100
  lines. Models download automatically on first use and cache in the
  state directory. The default embedding model is multilingual-e5-base
  (278M params, 768 dims, 100+ languages including German); the default
  reranker is Jina v2 multilingual (local cross-encoder, no HTTP
  endpoint needed). `cfetch embed-model test` verifies the installation
  end-to-end. Available models include multilingual-e5 (small/base/large),
  embeddinggemma-300m, BGE-M3 (dense+sparse+ColBERT), Qwen3 embeddings,
  and quantized variants of many models. Verified: German and English
  embeddings at 768 dims, ~15ms per query on CPU.
- **Embedded embedding backend: ONNX Runtime + BPE tokenizer (#43, #44,
  #46).** A new `embedded-embeddings` cargo feature adds ONNX Runtime
  (~20 MB) and the HuggingFace `tokenizers` crate to the binary.
  `cfetch embed-model download` fetches the quantized nomic-embed-text
  ONNX model (~131 MB) and tokenizer (~700 KB); `cfetch embed-model
  status` and `test` verify the installation. The `EmbedClient` gains an
  `Embedded` backend variant that runs inference in-process — no Ollama,
  LM Studio, or any HTTP endpoint. 6-8 ms per embedding on a modern CPU,
  512-token truncation for long documents, masked mean pooling. The
  default build is completely unaffected. (The safe-Rust entry above
  additionally isolates the experimental Nomic path until the admission
  framework certifies it.)
- **Safe-Rust and embedding-boundary hardening.** The cfetch package now
  forbids `unsafe` code on every target and feature combination. Atomic file
  replacement uses a safe cross-platform library boundary, and auth tests no
  longer mutate the process environment. The optional Nomic ONNX experiment
  is isolated to download/status/test commands, pins and verifies exact model
  artifacts, and cannot produce local or shared cfetch vectors. Canonical
  EmbeddingGemma transports remain fail-closed until admitted and attested.
- **`cfetch import openwolf` (#41).** A one-way importer that migrates an
  openwolf-enhanced `.wolf/` directory into the brain tree. Content files
  are placed at their ring-appropriate destinations (`OPENWOLF.md` →
  `AGENT.md` at ring 1, `cerebrum.md` → `knowledge/` at ring 3, `memory.md`
  → `mind/memories/` at ring 2), dated archives and backups go to
  `knowledge/archive/` (excluded from the index), and staging candidates
  keep their ring-5 quarantine. Ring frontmatter is prepended only when
  the file has none. `--dry-run` shows the mapping without writing.
  Everything cfetch regenerates on its own is skipped with a printed
  reason (code index, token ledger, buglog, cron, embeddings, config).
- **Semantic recall on explicitly configured endpoints (#39).** A new
  `embeddings.endpoint_model` config field carries the wire-level name a
  serving layer expects (LM Studio prefixes `text-embedding-`, Ollama uses
  short names) while the canonical `model` field remains the vector-space
  identity. Explicitly named noncanonical endpoints remain separate vector
  spaces; endpoints claiming the canonical model must pass the same
  admission and profile-attestation checks as a package-local producer. The
  first semantic recall
  benchmark is recorded alongside (`experiments/membench/results/`):
  cfetch at 71 ms vs openwolf-enhanced at 155 ms, with hardware load
  (~140 MB total) and weaker-system extrapolation. Three root causes are
  documented: IPv6 localhost adds a 2-second connection-refused penalty on
  Windows (use `127.0.0.1`), LM Studio unloads embedding models on every
  connection close (Ollama with `KEEP_ALIVE=-1` does not), and serving
  layers rename models (the `endpoint_model` field resolves this).
- **Round 4: the last three findings (#35), closing the 2026-08-31 bug
  hunt at 22/22.** The model's output budget (16384 tokens) now gates at
  submit time — a proposal whose `after` bytes exceed what the model can
  echo is refused with a clear split-or-apply-manually message, instead of
  truncating, failing the JSON parse, and retrying forever. Evidence
  coverage holds across late arrivals: when raw events rotated out at
  submit time (candidate-record citation was sufficient) and new matching
  events arrive during the review round-trip, the fallback stays
  sufficient at apply time — the proposer cannot cite events it has never
  seen. And stream appends hold a per-stream advisory lock across
  rotate+open+write, closing the race where a concurrent writer's
  rotation between our rotate check and our open landed our line in the
  wrong generation. Issue #27 is closed.
- **Round 3 of the 2026-08-31 bug hunt (#32, #33): boundaries and
  consistency.** `ensure_fresh` refuses to serve a never-scanned empty
  catalog instead of silently answering every query with zero hits;
  same-line symbol containers resolve to the innermost scope; the recall
  candidate pool expands on demand so a pathological block with many copies
  cannot shrink the result below the limit; Windows brain-root matching is
  case-insensitive (8.3 names, subst drives, symlinked homes no longer
  blind the hot-file trap); `cards list`/`status` hold the same store lock
  as the mutators; doctor reuses the pre-computed daemon probe instead of
  issuing its own; doctor and the statusline agree on maintenance priority
  (exception before model-unavailable); and `daemon status`'s durable
  observation after a SIGKILL is documented as the designed repair path,
  not an invariant violation.
- **Round 2 of the 2026-08-31 bug hunt (#29, #30): accuracy and robustness.**
  Nested `ring:` frontmatter keys no longer self-promote (only unindented
  top-level declarations count); ``` `bash` ``` inside a fence no longer
  closes it (CommonMark trailing-text rule); self-wikilinks stop inflating
  the unresolved-references metric; hyphenated query terms split into
  independent prefix terms (`state-machine` finds "the machine's state");
  rerank refuses an omitted `index` field (matching the embeddings client);
  the `query_for` 12-term cap now actually caps across payload keys; adapter
  launch failures retry instead of poisoning the process cache; torn vector
  index tails (crash between `writeln!`'s two writes) are repaired on the
  next open instead of corrupting two records; session-state GC covers
  crash debris (`.tmp.<pid>` and `.lock` files); `failure_history` redacts
  its query so a pasted failing line with a real secret can actually find
  its stored (redacted) signature; and `hardware.rs` detects MOVBE for the
  v3 verdict as its own comment specifies.
- **Ring-6 traps no longer record healthy commands as failures.** Any
  non-empty `stderr` used to mark a bash event `failed` — but `git push`,
  `npm install` and `pip` stream progress to stderr while exiting 0, and
  every trap (fix-discovered, recurring-failure, failure history) keys on
  that field. Stderr now signals failure only when the response carries no
  explicit success marker (`is_error:false`, `exit_code:0`), keeping the
  legacy-harness path the branch existed for. Found by the 2026-08-31 bug
  hunt, along with everything below in this block.
- **The embed queue cannot be frozen by one bad block.** The pending set is
  ordered and a row-level refusal (degenerate or wrong-width vector) aborted
  the batch before any write, re-selecting the identical head forever. A
  failed batch now retries its rows individually — refused rows are skipped
  with a note and retried next run — and a fully-refused batch bails with a
  resumable error instead of looping.
- **A corrupt vector record no longer bricks group semantic recall.** One
  flipped byte in the shared store hard-failed every hydrate; the read side
  now skips invalid records (they stay "missing" and re-embed) while
  write-side validation stays strict.
- **Maintenance reject is idempotent and submit no longer resurrects.** The
  autonomous loop re-derives deterministic proposal ids; a rejected twin
  used to wedge its PENDING copy in place — two model calls and one
  exception event every cycle, forever. An identical terminal copy now
  means the decision was already made.
- **A UTF-8 BOM no longer hides the ring frontmatter.** Windows editors emit
  BOMs by default; the BOM made the `---` comparison fail, so a BOM'd
  `ring: 5` quarantine marker was invisible and the file indexed — the
  documented fail-closed contract inverted by encoding accident.
- **Unreadable files are visible, and incremental rescans keep their last
  good rows.** A read failure (invalid UTF-8, an editor's sharing violation)
  was invisible: the file stayed in the fingerprint claiming fresh, and the
  rescan path deleted existing rows without reinserting. Full scans report
  skips; a one-second editor lock must not delete a document.
- **Semantic and hybrid recall dedup native mirrors**, like lexical recall
  always did: a mirrored block has the same hash and the same vector and
  ranked adjacent to itself, doubling one statement and displacing a
  distinct block.
- **Redaction fixes: glued short-flag secrets redact themselves** (`mysql
  -psecret` used to keep the secret and redact the next argument), **and
  sessionless events no longer merge**: two harness windows without a
  session id shared one `unknown-session` key, cross-contaminating
  repeat-read state and disabling the cross-session traps.
- **Consume is permanent, dotted hosts cannot collide, and prefix flags find
  the program.** A consumed candidate moves to `dismissed/` (the permanent
  do-not-restage marker) instead of being deleted and resurrecting when its
  consume record left the trap window; host ids containing dots can no
  longer collide with the rotation suffix grammar (one host's live stream
  is another's rotated generation); `sudo -u nobody pytest` now finds
  `pytest` instead of eliding the failing assertion from the condensed log.
- **`src/bin` files resolve `use crate::` against their own crate**, not the
  library beside them — false edges on module-name collisions and missing
  edges otherwise; **`cards status` survives a store without an `origin`
  remote** (`git remote get-url` exits 2 on absence, which was treated as a
  hard error instead of the designed degraded report). Remaining findings
  from the hunt are tracked in #27.
- **The tree config is content-only.** `.cfetch/config.json` inside the brain
  is agent-written and cloned across machines, so it may now carry only
  content keys (rings, slices, resident entries, budgets): `embeddings`,
  `rerank`, `maintenance`, `serve`, `client`, and `brain_root` found there are
  a hard error naming the key, the machine-local file overlays the tree for
  everything it sets, and the root itself never comes from any file — the
  documented invariant, now enforced. Tree-layer resident and code-root paths
  must stay tree-relative; absolute paths remain a machine-local decision.
- **Quarantine locations ignore self-promoting frontmatter.** A file in a
  ring-5+ directory (staging, logs) that declares a lower ring is skipped
  exactly like one whose frontmatter was stripped: the location decides, and
  the "cannot be edited away file by file" promise now holds against a
  well-formed lie. Promotion below the location ring is unchanged.
- **The egress guard resolves hostnames before trusting them.** `check_endpoint`
  refuses any address a configured host resolves to inside the already-refused
  ranges (plus loopback), closing DNS-rebinding names and resolver spellings
  like `0x7f000001`; rerank responses gained the 2 MiB cap the other HTTP
  clients already had.
- **Edge-supplied ids and local surfaces hardened.** `cfetch staging`
  consume/dismiss validate their id as one filename segment, a grant can no
  longer be named `root` (config reserves the whole-tree slice), the unix IPC
  socket is pinned to mode 0600 after bind, hook stdin reads are bounded at
  64 MiB, and maintenance review reads gate their proposal id.
- **The TCP serving listener is bounded.** At most 64 unauthenticated
  connections are held before the token is known, mirroring the iroh accept
  loop's permit cap, and responses past the 16 MiB serving limit are refused
  instead of serialized whole — a tokened `limit: usize::MAX` no longer ships
  the entire catalog in one line.
- **Dependency advisories gate CI.** `deny.toml` gained an `[advisories]`
  section (`yanked = "deny"`, reasoned ignores for the two unmaintained
  transitive advisories), CI runs `cargo deny check advisories licenses`, and
  the yanked `chacha20 0.10.1` left the lockfile for 0.10.2.
- **A maximum-performance self-build profile.** `[profile.release-max]` adds
  fat LTO and single-codegen on top of `release`, for binaries that never
  leave the machine they were built on; `docs/self-build-performance.md`
  records when that pays and why shipped binaries stay portable.
- **One-line duplicate symbol uses no longer abort the scan.** Two calls to
  the same function written on a single line inside one container produce an
  identical `symbol_uses` tuple; the duplicate is dropped at the insert. Found
  by the new membench harness on lodash, where 7 of 48 JS files — `lodash.js`
  included — aborted every code scan.
- **membench: a reproducible memory-tool benchmark.** `experiments/membench`
  carries a four-arm agent battery (continuity, locate, bugfix, stale-memory
  poisoning) and a speed harness, with its first recorded run stamped and
  committed: cfetch vs openwolf-enhanced 1.28.1 on lodash, fastify, tokio, and
  a synthetic 300-module tree.

- **Atomic local admission release boundary.** One hash-locked command replays
  the current admitted cohort, stages bounded physical evidence, runs every
  exact target-package/scope conformance challenge, creates the activation
  bundle, and emits a content-addressed immutable publication plan. It performs
  no external mutation. After an operator publishes the planned assets, source
  activation downloads each one through credential-free GitHub HTTPS, confines
  redirects to GitHub, and verifies its exact size and SHA-256 before writing.
- **Explainable source dependency graph.** `cfetch code-graph path` returns one
  deterministic shortest import chain, while `cfetch code-graph impact` walks
  reverse imports for a bounded blast radius and `cfetch code-graph context`
  traverses both directions around one file. Context retains one deterministic
  shortest explanation edge per related file and counts limit omissions. Typed
  steps now retain exact source ranges. `cfetch code-graph symbol` adds
  parser-proven `contains`, direct `calls`, and type `references` edges,
  resolving them only through explicit imports to one unambiguous file-level
  definition. Ambiguous names fail closed, relative paths are stable across
  serving hosts, and equivalent read-only MCP tools use the same graph query
  pipeline.
- **Selective nixcards knowledge.** `cfetch cards` initializes and manages the
  public nixcards catalogue as a blobless sparse checkout at
  `knowledge/cards`, with dotted branch selectors, JSON status, explicit
  fast-forward sync, and an optional handoff to the nixcards TUI. Git's native
  sparse state is the sole local selection record shared by both tools;
  materialized cards inherit trust only from their canonical knowledge path.
- **Autonomous AI memory maintenance.** A configured daemon reacts to changed
  ring-5 evidence, runs bounded proposal and isolated review passes, rechecks
  deterministic authority, citation, path, secret, expiry, and exact-byte
  gates under a transaction lock, and applies Markdown without routine
  approval. Direct Obsidian edits win every race. Immutable activity and
  exception history, global and per-file pause controls, exact safe revert,
  manual debugging commands, and the terminal Activity pane keep the process
  inspectable without turning maintenance into a human queue.
- **Obsidian knowledge graph.** `cfetch graph` and the terminal Graph pane expose
  the rebuildable wikilink graph with focused neighborhoods, ambiguity-safe
  note resolution, ring and degree metadata, bounded JSON, and local, serving,
  or authenticated slice-scoped remote access. Markdown remains the only
  authoritative relationship store.
- **Continuous derived-state upkeep.** Every storage daemon watches direct
  Markdown edits regardless of whether it serves clients. Catalog generations
  rebuild lexical and graph views, while a resident vector worker hydrates
  compatible shared or authorized-peer artifacts before deriving only missing
  content hashes. Joined peers and newly appearing shared artifacts take effect
  without restarting the daemon.
- **Maintenance-aware diagnostics.** RuntimeStatusV1, Claude's status line,
  Codex transition notices, MCP status, `cfetch doctor`, and the terminal System
  pane distinguish maintenance configuration, local or remote route, proposal
  and review models, last attempt, paused/degraded state, candidates, outcomes,
  and exceptions without exposing endpoint or credential details.
- **One semantic and output ABI for v1.** The candidate v1 profile freezes the
  pinned EmbeddingGemma-300M source revision, tokenizer, retrieval prompts,
  full 768 dimensions, pooling/normalization, and the canonical 768-byte signed
  INT8 output/index codec. Core ML, LiteRT, OpenVINO, and other packages may
  use target-native artifacts and internal precision. No NPU or runtime is a
  numerical anchor; model-pipeline or codec changes require a new network
  major and coordinated re-embedding.
- **Absolute cross-device admission.** Every concrete artifact/runtime/device
  scope must be repeatable and meet the same fixed retrieval floors in every
  ordered query-backend x document-backend pairing, including self-pairs.
  Each query backend must also pass a conservative adversarial mix of document
  producers, covering the derive-once store's real per-content first-writer
  behavior. Cross-backend exact bytes and cosine are diagnostics. Semantic
  vector identity and versioned admission policy have separate digests, so a
  policy change recertifies backends without re-embedding unchanged vectors.
  The admitted registry remains empty until real placement and global evidence
  exists.
- **Local native-adapter boundary.** Target packages can expose Core ML,
  LiteRT, OpenVINO, or another native runner through the attested loopback
  embedding protocol that cfetch already consumes. The runner stays local and
  cfetch owns canonical INT8x768 output encoding; this is not remote inference.
- **Cross-major networking fails closed.** TCP, iroh ALPN, invites, grants, and
  remote memberships carry network major 1. Missing or different majors are
  rejected before slice data is served. `cfetch embedding-profile [--json]`
  prints the executable contract.
- **Public feature and benchmark guide.** The README now maps the product's
  agent-memory, retrieval, code-navigation, capture, measurement, serving, and
  sharing surfaces; includes the real v0.9.9 terminal dashboard; and compares
  cfetch with OpenWolf and OpenWolf Enhanced. A reproducible v0.9.9 study
  records 93.4% aggregate model-facing reduction across eight selected
  oversized command outputs, explicitly without claiming whole-session savings.
- **Organization-wide contributor terms.** External contributors retain
  copyright and grant the project broad copyright and patent rights through
  Corbet Labs' versioned Individual Contributor License Agreement. This keeps
  the whole project under one coherent FSL-to-Apache release promise.

## 0.9.9

## 0.9.8

## 0.9.7

## 0.9.6

## 0.9.5

- **crates.io distribution.** The verified v0.9.4 source release is available
  as `cargo install cfetch --locked`; future tags publish through short-lived
  GitHub OIDC credentials after the complete platform release succeeds, with
  no registry token stored in GitHub.
- **Platform-stable resident budgets.** Crowded resident indexes stop repeating
  long absolute brain-root prefixes on every line, so macOS and Windows keep
  the same hard digest cap as Linux without dropping configured files.
- **Windows self-read accounting.** PowerShell drive paths retain their
  backslash separators when cfetch recognizes safe whole-file shell reads,
  keeping those reads in the self-read ledger instead of silently losing them.

## 0.9.4

- **Ring-aware resident budgets.** Resident digest space is water-filled by
  ring-derived or explicit entry weight. Short entries return unused space to
  longer ones, ring-0 invariants outrank less load-bearing material, and the
  configured digest budget remains a hard cap.
- **Truthful Arch package identity.** The AUR build and check phases now use
  the same catalog variant, so `cargo test` cannot overwrite the packaged
  binary with an unidentified developer build. A CI contract ties both Arch
  architectures and both package phases back to the executable release
  catalog.
- **Unix-friendly output pipes.** Normal consumers such as `head` may close a
  pipe early without making cfetch print a Rust panic or fail the command.
- **Release plumbing maintenance.** Homebrew reads the published release
  manifest alongside its checksums instead of assuming a source checkout, and
  GitHub artifact transfer uses the current Node 24 actions.

## 0.9.3

- **Executable release catalog.** `release/variants.json` is now the single
  source for runtime advice, CI, release archives, and Homebrew. Artifact names
  explicitly say `remote` while inference remains endpoint-only; the empty
  backend Cargo features and nonexistent silicon recommendations are gone.
  `cfetch variants` reports the embedded catalog, and `cfetch hardware` can no
  longer recommend an artifact that does not exist.
- **ARM release coverage.** Linux and Windows ARM64 join Linux, macOS, and
  Windows x86_64 in the generated release matrix. Every catalog entry gets a
  blocking compile job before a tag can become a complete release.
- **Transactional patch releases.** A manual release action now computes the
  next pre-1.0 patch, updates every version and license-notice surface, pushes
  the preparation commit, waits for the complete blocking CI matrix, and tags
  only that verified commit. The tag then drives archives and Homebrew; Crow
  independently reconciles the AUR package.

## 0.9.2

- **Homebrew install.** `brew tap corbet-labs/cfetch && brew install cfetch`
  on macOS (Apple silicon and Intel) and Linux x86_64. Releases now publish
  `checksums_sha256.txt`, and the tap's formula is regenerated from those
  published checksums on every tag rather than hand-written. There is no
  linux-arm64 build, and the formula says so instead of offering an x86_64
  tarball that cannot execute.

- **Multi-harness adapters.** The official `rmcp` SDK now owns MCP framing and
  negotiation, `agent-session` normalizes Claude/Codex/Gemini/Cursor transcript
  discovery, and `agent-config` installs cfetch's confirmed MCP, instruction,
  and native-hook surfaces across its 25-harness registry. The libraries are
  exact-pinned behind cfetch-owned compatibility facades; cfetch 0.9 marker
  formats are converted once without leaving parallel legacy registrations.
  `cfetch install --project <path>` exposes confirmed project-only surfaces.

- **Dependency-license compliance.** CI now rejects unreviewed licenses and
  verifies a generated third-party license/NOTICE bundle. Release archives and
  the Nix and Arch packages all install that bundle beside cfetch's own license;
  copied third-party material requires an explicit provenance record.

- **Authenticated iroh transport.** Invites now carry the origin's real iroh
  endpoint address, remote redemption binds a grant to the QUIC-authenticated
  joining endpoint, and `recall --slice` plus citation expansion route through
  the joining daemon without persisting the one-time secret. Citation results
  are filtered against the granted slice before they cross the network.
- **Pre-1.0 release lock.** The crate build, blocking CI, and release preflight
  all reject a 1.x package version. A release tag must also match Cargo.toml.
  The lock stays in place until the operator explicitly authorizes 1.0.
- **Codex output condensation.** Oversized Bash listings now use Codex's
  PostToolUse model-feedback replacement path instead of adding a second copy
  as context. The complete original is preserved in private local state and
  linked from the condensed result under a 64 MiB retention cap; test and build
  output remains untouched.
- **Cross-platform state locking.** Identity creation now reads the key only
  while holding its creation lock, so Windows cannot observe a winner's
  zero-byte in-progress file. Lock retries honor their full deadline, and
  concurrent session updates have enough bounded time on macOS and Windows.

- **Engine selection (historical 0.9.2 behavior; superseded on main by the
  candidate shared-output architecture above).** A variant is now a Cargo feature selection rather
  than a source fork: which engines are compiled in is a build-time fact,
  which device to use is a run-time one, and `cfetch hardware` reports both
  plus what it will actually do. A device no compiled-in engine can drive is
  skipped rather than selected and then failed on. At that release, everything
  fell back to the remote endpoint for a host that held nothing; main no longer
  permits automatic remote fallback.
- **Hardware detection.** `cfetch hardware` reports the accelerators it can
  see, what proved each one, and the variant this machine should run, under
  the `<os>-cfetch-<silicon>[-<level>]` scheme. The policy is NPU > GPU > CPU
  and it is encoded as an ordering rather than as a comment — an NPU is
  preferred even where the GPU beside it is faster, because it draws less
  power and is the one processor nothing else on the machine is competing
  for.

- **Document-side embedding prefixes, as part of the artifact identity.**
  `embeddings.document_prefix` carries the instruction models like E5 and
  EmbeddingGemma expect on documents. Because it changes every stored vector,
  it is keyed into the shared artifact's filename and recorded exactly in its
  header — two hosts configured differently write separate files rather than
  appending incompatible vectors to one. Stores written before this exist
  unchanged and keep working: no prefix line means raw documents, which is
  what they hold.
- **Invites and grants.** `cfetch invite <slice>` mints a one-time ticket,
  `cfetch join <ticket>` redeems it, `cfetch grants` shows who holds what.
  A ticket is an address plus a secret, NOT an authorization: the origin looks
  the secret up in its own records, so a forged ticket buys nothing and a real
  one can be pasted through any channel. The origin stores only the secret's
  hash. An invite binds to the first host that redeems it, a retry from that
  same host is not an error, and a second host is refused. Inside one storage
  group — hosts sharing the tree — redemption needs no network at all.
- **Host identity.** Each host now has a persistent iroh keypair, created on
  first use, whose public half is the endpoint id peers will grant slices to.
  `cfetch identity` prints it and `cfetch status` names it. The key lives in
  the per-host state directory and never in the brain tree — a shared tree
  would hand every host the same identity, making a grant to one machine a
  grant to all of them. A key that cannot be read is an error rather than a
  fresh identity, because regenerating would orphan every grant naming the old
  one.

## 0.8.0

- **Slices.** Named prefix sets over the tree, nestable, with `cfetch slices`
  and `cfetch recall --slice`. A document belongs to its innermost slice; a
  query for a slice reaches everything inside it; an unknown name is refused
  rather than widened to the whole tree. Membership is derived from the path
  rather than stored, so it cannot fall out of step with the configuration.
  A brain with no slices behaves exactly as before.

- **The drain barrier is sound on every platform, and says which path it is
  on.** Proving coverage with a sentinel works only where the watcher delivers
  events in order, which is inotify and nothing else; macOS FSEvents coalesces
  per directory and does not order across them, and a concurrent-writer run
  there could answer `fresh: true` while missing a committed statement. The
  watcher backend now declares its ordering capability at runtime: Linux keeps
  the sentinel path unchanged, and every other backend proves coverage by
  comparing a stat fingerprint of the tree, taken at query entry, against what
  the committed catalog has seen. `serve-status` and `cfetch status` name the
  mode in force. Where one such walk would cost more than the barrier's own
  budget, the answer comes back immediately as stale-and-labeled rather than
  blowing the bound.

- **Rings are configuration, not code.** `ring_rules` (ordered prefix → ring)
  and `exclude_prefixes` replace a hardcoded taxonomy. The shipped defaults
  reproduce the previous behavior exactly. Secrets, logs and `.git` remain
  excluded by a boundary no config can lift.
- **Injection is a policy.** A resident entry may declare
  `scope: { hosts, repos }`; only matching sessions receive it, and skipped
  entries cost no budget and are reported rather than silently dropped.
- **Windows and macOS are supported platforms.** A local-channel abstraction
  (unix socket / loopback TCP with a per-daemon token), platform-native paths
  and quoting, and a lock implementation per platform. Linux, macOS and
  Windows all gate the build — a failure on any of them blocks. Served paths
  are canonicalized in the process, so a repository answered by a Windows host
  and by a Linux host returns identical lines rather than the same file under
  two spellings.
- **Semantic recall works on a living brain.** Vectors are keyed by content
  hash instead of a volatile row id, so editing one file no longer discards
  every embedding; they are stored as shared f16 artifacts (dimensions
  configurable) that any host can read and only a capable host computes.
  Partial or missing coverage is reported, never silently answered lexically.
- **Rings 5 and 6 live in the tree.** Session exhaust and the cost ledger are
  versioned append-only JSONL streams per host; staging candidates are
  markdown files with content-addressed ids, so a candidate flagged on one
  machine is visible — and drainable — from any of them.
- **Queries are embedded the way retrieval models expect.** `embeddings.query_prefix`
  carries the instruction modern embedders are trained to see on the query
  side, and only there — documents stay raw, so stored vectors are unchanged.
  Without it cfetch was embedding queries and documents identically, which
  these models are explicitly not trained for.
- **A host that holds nothing can still rank semantically.** `--semantic` and
  `--hybrid` used to refuse over remote serving, because the querying host had
  no vectors and no endpoint to embed its own query against. The serving host
  now ranks on the client's behalf — it is the one holding both — so the
  none-tier hosts the whole design is for are no longer second-class. Ranking
  itself moved into one shared pipeline, so the same query against the same
  tree ranks identically whether it was answered locally or over the wire, and
  any degradation travels back with the answer.
- **Cross-encoder reranking.** An optional second stage reorders a retrieved
  shortlist by reading query and document together, which no retrieval scorer
  does. Recall widens to `rerank.candidates` and the answer is cut back
  afterwards, so the reranker can only promote what retrieval proposed. Off by
  default; every failure answers in retrieval order with the reason attached.
- A serving daemon now indexes its own code roots on startup instead of
  waiting to be asked, `map` is answerable remotely, and the dashboard routes
  to its serving host rather than opening a local index it does not have.

## 0.7.0

- **Serving mode.** The machine holding the markdown runs a daemon that
  watches its own tree and answers queries behind a drain barrier
  (serve-fresh-or-wait, bounded); every answer carries its origin, catalog
  generation and a `fresh` flag. Other machines can hold nothing at all and
  query it. Read-your-writes across hosts, measured at zero misses over 300
  cross-host checks.
- Watch scope derived from the indexer's own walk: no symlink following, no
  watching what is never indexed (a real tree went from ~524k required
  watches to 90), and the listener binds before registration completes so a
  restart is not an outage.
- The 26 findings of an adversarial design review, including a torn-ledger
  race, a hook that read whole files to count lines, and a staging queue that
  its own retention could delete.

## 0.6.0

- Reminder queue delivered on the next prompt instead of at stop (a stop-time
  injection costs an entire extra model turn), cadence re-injection of the
  top rules, and `cfetch audit` — the always-on context bill priced honestly,
  including cfetch's own injections.
- Recall deduplicates content mirrored between sources, keeping the
  highest-trust copy and naming the others.

## 0.5.0

- Automatic capture (ring 6), promotion traps and the ring-5 staging queue.
- Measured token accounting from the session transcript, booked as per-turn
  deltas.
- Import-graph importance, `cfetch map`, background code scanning.
- Packaging: nix flake, Arch package, CI and release automation.

## 0.4.0 and earlier

Recall over the ring model with content-addressed citations, the tree-sitter
code index, hook-based injection, the MCP server, and the terminal dashboard.
