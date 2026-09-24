// Rust's standard print macros panic when a downstream pipe closes normally
// (for example `cfetch status | head`). Keep test-harness capture unchanged,
// but make every production CLI write treat BrokenPipe as a clean early exit.
#[cfg(not(test))]
macro_rules! print {
    ($($arg:tt)*) => {
        $crate::output::stdout(format_args!($($arg)*))
    };
}

#[cfg(not(test))]
macro_rules! println {
    () => {
        $crate::output::stdout_line(format_args!(""))
    };
    ($($arg:tt)*) => {
        $crate::output::stdout_line(format_args!($($arg)*))
    };
}

#[cfg(not(test))]
macro_rules! eprintln {
    () => {
        $crate::output::stderr_line(format_args!(""))
    };
    ($($arg:tt)*) => {
        $crate::output::stderr_line(format_args!($($arg)*))
    };
}

mod audit;
mod bench;
mod cards;
mod code;
mod condense;
mod config;
mod daemon;
mod dashboard;
mod doctor;
mod embed;
#[cfg(feature = "embedded-embeddings")]
mod embedded_embed;
mod embedding_input;
mod embedding_profile;
mod exhaust;
mod fsutil;
mod govern;
mod graph;
mod hardware;
mod hashing;
mod heartbeat;
mod hook_io;
mod hooks;
mod import;
mod index;
#[cfg(target_os = "linux")]
pub mod inference_governor;
mod init;
mod install;
mod ipc;
mod jsonl;
mod knowledge_graph;
mod ledger;
mod local_adapter;
mod local_embedding;
mod local_inference;
mod lockfile;
mod maintenance;
mod maintenance_inbox;
mod maintenance_model;
mod maintenance_worker;
mod markers;
mod mcp;
mod memory_eval;
#[cfg(all(target_os = "linux", feature = "native-openvino"))]
mod native_adapter;
#[cfg(all(target_os = "linux", feature = "native-openvino"))]
mod native_worker;
#[cfg(not(test))]
mod output;
mod paths;
mod pipeline;
mod repositories;
mod rerank;
mod resident;
mod retrieval_fixture;
mod runtime_status;
mod serve;
mod session_state;
mod staging;
#[cfg(test)]
mod testhttp;
mod transcript;
mod variant;
mod vector_worker;
mod vectors;

use anyhow::Context as _;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "cfetch", version, about = "A second brain for coding agents")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[cfg(all(target_os = "linux", feature = "native-openvino"))]
    #[command(hide = true)]
    NativeWorker {
        #[arg(long)]
        parent_pid: u32,
    },
    #[cfg(all(target_os = "linux", feature = "native-openvino"))]
    #[command(hide = true)]
    NativeProbe {
        plan: std::path::PathBuf,
        evidence: std::path::PathBuf,
        #[arg(long)]
        policy_sha256: String,
    },
    /// Inspect independent Git repositories, or synchronize their committed work
    Repos {
        /// Explicit checkout/discovery roots; defaults to the configured git.roots
        roots: Vec<std::path::PathBuf>,
        /// Fetch, merge clean changes, and push; leave working edits untouched
        #[arg(long)]
        sync: bool,
        #[arg(long)]
        json: bool,
    },
    /// Run a hook entrypoint (invoked by the agent harness, reads stdin)
    Hook {
        /// session-start | user-prompt | pre-tool | post-tool | stop | precompact
        event: String,
        /// Coding-agent adapter invoking the hook
        #[arg(long, value_name = "ID")]
        agent: Option<String>,
    },
    /// Import data from another memory tool
    Import {
        #[command(subcommand)]
        source: ImportSource,
    },
    /// Qualify a pinned offline CPU model against canonical short and long vectors
    #[cfg(feature = "embedded-embeddings")]
    QualifyModel {
        model_dir: std::path::PathBuf,
        reference: std::path::PathBuf,
    },
    /// Inspect or run diagnostic FastEmbed models (never shared vectors)
    #[cfg(feature = "embedded-embeddings")]
    EmbedModel {
        /// Diagnostic catalogue variant; does not select a shared-store model
        #[arg(long, global = true, default_value = "MultilingualE5Base", value_parser = embedded_embed::parse_model)]
        model: fastembed::EmbeddingModel,
        #[command(subcommand)]
        action: EmbedModelAction,
    },
    /// Manage the per-host daemon
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },
    /// Register (or remove) cfetch in detected coding-agent harnesses
    Install {
        /// Explicit Claude settings.json path (otherwise Claude is feature-detected)
        #[arg(long)]
        settings: Option<std::path::PathBuf>,
        /// Target one agent ID; repeat to select several (defaults to detection)
        #[arg(long = "agent", value_name = "ID")]
        agents: Vec<String>,
        /// Configure every harness supported by the pinned adapter library
        #[arg(long, conflicts_with = "agents")]
        all: bool,
        /// Install project-local surfaces under this existing project root
        #[arg(long, value_name = "PATH", conflicts_with = "settings")]
        project: Option<std::path::PathBuf>,
        /// Remove cfetch's managed entries instead of adding them
        #[arg(long)]
        remove: bool,
        /// Replace an existing foreign Claude status line instead of preserving it
        #[arg(long, conflicts_with = "remove")]
        replace_status_line: bool,
    },
    /// Rebuild the recall index and sync the code index
    Scan {
        /// Hand the (slow) code scan to the daemon's background thread;
        /// falls back inline with a warning when the daemon is unreachable
        #[arg(long = "async")]
        background: bool,
    },
    /// Locate a symbol or file in the code index, with exact line ranges
    Find {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Token budget the answer must fit; 0 lifts the cap
        #[arg(long, default_value_t = answer::FIND_BUDGET_TOKENS)]
        budget_tokens: u64,
        #[arg(long)]
        json: bool,
    },
    /// Print a repo map fitted to a token budget, ordered by import-graph importance
    Map {
        /// Personalize the ranking toward files whose path or symbols match this term
        #[arg(long)]
        focus: Option<String>,
        /// Token budget the rendered map must fit
        #[arg(long, default_value_t = graph::DEFAULT_MAP_BUDGET_TOKENS)]
        budget_tokens: u64,
    },
    /// Explain paths and blast radius in the indexed source import graph
    CodeGraph {
        #[command(subcommand)]
        action: CodeGraphAction,
    },
    /// Inspect the Obsidian knowledge graph derived from curated Markdown links
    Graph {
        /// Center the view on a Markdown path or note name
        #[arg(long)]
        focus: Option<String>,
        /// Restrict to one local or joined slice and everything nested inside it
        #[arg(long)]
        slice: Option<String>,
        /// Maximum documents to return (1-200)
        #[arg(long, default_value_t = 40)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Explain how two notes connect through curated links and backlinks
    GraphPath {
        from: String,
        to: String,
        #[arg(long, default_value_t = 6)]
        depth: usize,
        #[arg(long)]
        json: bool,
    },
    /// Search the brain (rings 0-4), BM25-ranked, with ring-prefixed citations
    Recall {
        /// Search terms (word-prefix matched, OR-combined)
        query: Vec<String>,
        /// Expand a citation id instead of searching
        #[arg(long)]
        id: Option<String>,
        /// Also list docs wikilinked to the top hits (1-hop curated graph)
        #[arg(long)]
        expand: bool,
        /// Require a background refresh started after this request (5-second wait);
        /// otherwise return an explicit freshness error. Default: cached snapshot.
        #[arg(long)]
        fresh: bool,
        /// Rank purely by embedding cosine similarity (requires embeddings config)
        #[arg(long, conflicts_with = "hybrid")]
        semantic: bool,
        /// Fuse BM25 and semantic rankings via reciprocal rank fusion
        #[arg(long)]
        hybrid: bool,
        /// Restrict to one slice and everything nested inside it
        #[arg(long)]
        slice: Option<String>,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        /// Token budget the answer must fit — hits or an expanded block;
        /// 0 lifts the cap
        #[arg(long, default_value_t = answer::RECALL_BUDGET_TOKENS)]
        budget_tokens: u64,
        #[arg(long)]
        json: bool,
    },
    /// Detect accelerators and name the variant this machine should run
    Hardware {
        #[arg(long)]
        json: bool,
    },
    /// List release artifacts that actually exist
    Variants {
        #[arg(long)]
        json: bool,
    },
    /// Print the immutable embedding and network compatibility contract
    EmbeddingProfile {
        #[arg(long)]
        json: bool,
    },
    /// Compare BM25, vectors, hybrid fusion, reranking, and graph expansion on safe temporary data
    RetrievalFixture {
        #[arg(long)]
        json: bool,
        /// Print every vector component instead of a short preview
        #[arg(long)]
        show_vectors: bool,
        /// Exit nonzero unless this capability passes (repeatable)
        #[arg(long, value_enum, value_delimiter = ',')]
        require: Vec<retrieval_fixture::Requirement>,
    },
    /// Evaluate labeled Markdown and imported candidate vectors in isolated temporary storage
    RetrievalEval {
        #[arg(long)]
        corpus: std::path::PathBuf,
        /// Experimental input policy; does not change the production profile
        #[arg(long, value_enum, default_value = "body")]
        representation: memory_eval::Representation,
        /// Create an exact input manifest (refuses to overwrite an existing file)
        #[arg(long)]
        export: Option<std::path::PathBuf>,
        /// Evaluate a candidate vector bundle bound to that exact manifest
        #[arg(long)]
        vectors: Option<std::path::PathBuf>,
    },
    /// List the configured slices with what each one holds
    Slices {
        #[arg(long)]
        json: bool,
    },
    /// Review ring-5 memories: auto-flagged exhaust in the tree, shared across
    /// hosts, never injected
    Memories {
        #[command(subcommand)]
        action: MemoriesAction,
    },
    /// Continuous second-brain maintenance, history, exceptions, and debugging
    Maintain {
        #[command(subcommand)]
        action: MaintainAction,
    },
    /// Ask the ring-6 exhaust whether a command has failed here before, how
    /// often, on which hosts, and whether it ever recovered
    Failures {
        /// A command line or terms; empty ranks the whole failure history
        query: Vec<String>,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Embed index blocks lacking vectors (resumable; requires embeddings config)
    EmbedIndex {
        /// Blocks per embeddings request
        #[arg(long, default_value_t = 64)]
        batch: usize,
    },
    /// A/B bench: paired cfetch-on / cfetch-off sessions, read from
    /// transcript ground truth — cache dimensions plus the bash re-run rate
    Bench {
        /// Only consider sessions the harness touched within this many days
        #[arg(long, default_value_t = bench::DEFAULT_WINDOW_DAYS)]
        since_days: u64,
        #[arg(long)]
        json: bool,
    },
    /// Create the standard brain tree: the top-level directories cfetch's
    /// rules, slices and grants all name. Additive and idempotent
    Init {
        /// Where to create it. Defaults to the configured brain root
        path: Option<std::path::PathBuf>,
    },
    /// Manage the nixcards catalogue stored under knowledge/cards
    Cards {
        #[command(subcommand)]
        action: CardsAction,
    },
    /// Open the terminal dashboard: health, system diagnostics, recall, and maintenance
    Dashboard,
    /// Explain hardware, inference, peers, artifacts, hooks, and daemon health
    Doctor {
        /// Print the stable DoctorReportV1 diagnostic contract
        #[arg(long, conflicts_with = "tui")]
        json: bool,
        /// Open the live System pane in the terminal dashboard
        #[arg(long, conflicts_with = "json")]
        tui: bool,
        /// Do not contact joined remote origins
        #[arg(long)]
        no_network: bool,
        /// Call the configured retrieval models on safe temporary data
        #[arg(long, conflicts_with = "tui")]
        deep: bool,
        /// Include every vector component in the deep report
        #[arg(long, requires = "deep", conflicts_with = "tui")]
        show_vectors: bool,
        /// Exit nonzero unless this deep retrieval capability passes (repeatable)
        #[arg(
            long,
            value_enum,
            value_delimiter = ',',
            requires = "deep",
            conflicts_with = "tui"
        )]
        require: Vec<retrieval_fixture::Requirement>,
    },
    /// Serve recall/find/expand over MCP (stdio) for any MCP client
    Mcp,
    /// Verify the installation end to end; nonzero exit on hard failures
    Selfcheck,
    /// Price the always-on context bill: CLAUDE.md + imports, MCP servers,
    /// booked injection vs budget, position-weighted cost, measurement gaps
    Audit {
        #[arg(long)]
        json: bool,
    },
    /// Show runtime routing, inference, maintenance, and detailed diagnostics
    Status {
        /// Print the stable RuntimeStatusV1 contract
        #[arg(long, conflicts_with = "line")]
        json: bool,
        /// Print one cached, terminal-width-aware line (no network or inference)
        #[arg(long, conflicts_with = "json")]
        line: bool,
    },
}

#[derive(Subcommand)]
enum CodeGraphAction {
    /// Find one deterministic shortest import path between two files
    Path {
        from: String,
        to: String,
        /// Maximum import hops to traverse (1-32)
        #[arg(long, default_value_t = graph::DEFAULT_PATH_DEPTH)]
        depth: usize,
        #[arg(long)]
        json: bool,
    },
    /// Show files that directly or transitively import the target
    Impact {
        target: String,
        /// Maximum reverse-import hops to traverse (1-32)
        #[arg(long, default_value_t = graph::DEFAULT_IMPACT_DEPTH)]
        depth: usize,
        /// Maximum affected files to return (1-200)
        #[arg(long, default_value_t = graph::DEFAULT_IMPACT_LIMIT)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Show bounded incoming and outgoing import context around one file
    Context {
        target: String,
        /// Maximum import hops to traverse in either direction (1-32)
        #[arg(long, default_value_t = graph::DEFAULT_CONTEXT_DEPTH)]
        depth: usize,
        /// Maximum related files to return (1-200)
        #[arg(long, default_value_t = graph::DEFAULT_CONTEXT_LIMIT)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Show parser-proven calls and type references around an exact symbol
    Symbol {
        query: String,
        /// Maximum matching symbols and relationships to return (1-200)
        #[arg(long, default_value_t = graph::DEFAULT_SYMBOL_LIMIT)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum CardsAction {
    /// Create the blobless sparse catalogue checkout
    Init {
        #[arg(long, default_value = cards::OFFICIAL_REPOSITORY)]
        repository: String,
    },
    /// Show every published set and whether it is local
    List {
        #[arg(long)]
        json: bool,
    },
    /// Replace the local selection with dotted set IDs or category prefixes
    Select {
        selectors: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// Fast-forward the catalogue without changing its sparse selection
    Sync {
        #[arg(long)]
        json: bool,
    },
    /// Explain the local checkout, revision, filter, and selected sets
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Open the nixcards TUI against this brain's catalogue checkout
    Tui,
}

#[derive(Subcommand)]
enum MemoriesAction {
    /// List flagged, unconsumed candidates, newest first
    List {
        #[arg(long)]
        json: bool,
    },
    /// Mark a candidate consumed (a distillation session has taken it)
    Consume { id: String },
    /// Move a candidate out of staging without promoting it
    Dismiss { id: String },
}

#[derive(Subcommand)]
enum MaintainAction {
    /// Run one bounded autonomous propose → review → verify → apply cycle
    Run {
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        json: bool,
    },
    /// Pause background and manual autonomous cycles with a visible reason
    Pause { reason: Vec<String> },
    /// Resume autonomous maintenance after debugging or an intervention
    Resume,
    /// Show immutable automatic activity and exception history
    History {
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Build a bounded evidence packet for an agent to analyze
    Packet {
        candidate_id: String,
        #[arg(long)]
        json: bool,
    },
    /// Submit a typed proposal into ring-5 quarantine (reads stdin by default)
    Submit {
        #[arg(long, value_name = "JSON")]
        file: Option<std::path::PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Record one independent semantic review (reads stdin by default)
    Review {
        id: String,
        #[arg(long, value_name = "JSON")]
        file: Option<std::path::PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// List maintenance proposals and their lifecycle state
    List {
        #[arg(long)]
        json: bool,
    },
    /// Show one proposal exactly as cfetch recorded it
    Show {
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// Revalidate evidence, authority, target bytes, and trust boundaries
    Verify {
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// Apply the exact verified bytes; remains reversible until finalization
    Apply {
        id: String,
        #[arg(long, value_name = "TOKEN")]
        approval_token: String,
    },
    /// Apply a passing independent review through the autonomous policy gates
    AutoApply { id: String },
    /// Restore captured before-bytes while the exact applied bytes still match
    Revert { id: String },
    /// Reject a proposal without dismissing its source candidate
    Reject { id: String },
    /// Finish a legacy manual apply after git HEAD contains its exact bytes
    Finalize { id: String },
}

#[derive(Subcommand)]
#[cfg(feature = "embedded-embeddings")]
enum EmbedModelAction {
    /// Download and load the selected diagnostic model
    Download,
    /// Inspect the selected model's cache location without loading or downloading
    Status,
    /// Embed a test string locally; the result is never stored or shared
    Test {
        /// Text to embed
        text: String,
    },
    /// List diagnostic embedding models and query-local rerankers
    List,
}

#[derive(Subcommand)]
enum ImportSource {
    /// Migrate an openwolf-enhanced .wolf/ directory into the brain tree
    Openwolf {
        /// Path to the .wolf/ directory to import
        path: std::path::PathBuf,
        /// Show what would be imported without writing anything
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum DaemonAction {
    /// Run in the foreground (what systemd should call)
    Run,
    /// Start detached
    Start,
    /// Stop a running daemon
    Stop,
    /// Show whether the daemon answers
    Status,
}

fn selfcheck() -> anyhow::Result<()> {
    let mut hard_failures = 0;

    let cfg = match config::Config::load() {
        Ok(c) => {
            println!("ok    config loads ({} resident entries)", c.resident.len());
            Some(c)
        }
        Err(e) => {
            println!("FAIL  config: {e}");
            hard_failures += 1;
            None
        }
    };

    if let Some(cfg) = &cfg {
        if cfg.brain_root.is_dir() {
            println!("ok    brain root {}", cfg.brain_root.display());
        } else {
            println!("FAIL  brain root missing: {}", cfg.brain_root.display());
            hard_failures += 1;
        }
        let scope = resident::SessionScope::current();
        let digest = resident::build(cfg, &scope);
        if digest.text.is_empty() {
            println!("warn  resident digest is empty — nothing will be injected");
        } else {
            println!(
                "ok    resident digest builds ({} chars, ~{} tokens estimated)",
                digest.text.len(),
                hook_io::estimate_tokens(digest.text.len())
            );
            for (label, chars) in &digest.sources {
                println!("        {label}: {chars} chars");
            }
        }
        // A placeholder is injected so the SESSION learns its contract broke,
        // but it must not pass as health in the operator's diagnostic.
        for label in &digest.missing {
            println!("warn  resident file missing — the entry injects only a placeholder: {label}");
        }
        if !digest.skipped_by_scope.is_empty() {
            println!(
                "ok    {} resident entr(ies) out of scope for host {}{}",
                digest.skipped_by_scope.len(),
                scope.host,
                scope
                    .repo
                    .as_deref()
                    .map(|r| format!(" / repo {r}"))
                    .unwrap_or_default(),
            );
            for label in &digest.skipped_by_scope {
                println!("        {label}: not injected here");
            }
        }
    }

    let state = paths::state_dir();
    match std::fs::create_dir_all(&state) {
        Ok(()) => println!("ok    state dir writable: {}", state.display()),
        Err(e) => {
            println!("FAIL  state dir {}: {e}", state.display());
            hard_failures += 1;
        }
    }

    if let Some(cfg) = &cfg {
        // Rings 5 and 6 are records in the TREE, not per-host state. A
        // diagnostic must not be the thing that first writes into the
        // brain — so the exhaust dir is never CREATED here, which also
        // means writability cannot be honestly claimed before it exists.
        // Test what can be tested: when the dir is there, write and remove
        // a probe (leaves no record); when it is not, say "untested"
        // instead of "this host writes", because on a read-only brain that
        // claim was a lie a live hook then paid for.
        let logs = paths::logs_dir(&cfg.brain_root);
        let staging = paths::staging_dir(&cfg.brain_root);
        let host = paths::host_id();
        if logs.is_dir() {
            let probe = logs.join(".selfcheck-write-probe");
            match std::fs::write(&probe, b"").and_then(|()| std::fs::remove_file(&probe)) {
                Ok(()) => println!(
                    "ok    ring-6 exhaust writable: {} (this host writes as {host})",
                    logs.display()
                ),
                Err(e) => {
                    println!(
                        "FAIL  ring-6 exhaust not writable: {} ({e}) — hooks silently drop every record",
                        logs.display()
                    );
                    hard_failures += 1;
                }
            }
        } else {
            println!(
                "warn  ring-6 exhaust dir not created yet: {} — writability untested; the first hook write decides",
                logs.display()
            );
        }
        println!(
            "ok    ring-5 memories: {} ({} candidate(s) pending, selected mind)",
            staging.display(),
            staging::pending_count(&staging)
        );
        for note in ledger::read(&logs).unreadable {
            println!("warn  ledger stream unreadable: {note}");
        }
    }
    // The derived catalog is a measurement too: an installation that has
    // never scanned answers every recall from nothing, and one lagging the
    // tree answers from a version of it that no longer exists.
    if let Some(cfg) = &cfg {
        let (severity, line) = index_liveness_line(cfg, &state, None);
        match severity {
            heartbeat::Severity::Healthy => println!("ok    {line}"),
            _ => println!("warn  {line}"),
        }
    }

    match daemon::call("ping", std::time::Duration::from_millis(300)) {
        Some(r) => println!("ok    daemon answers (v{})", r.version.unwrap_or_default()),
        None => println!("warn  daemon not running — hooks fall back to direct reads"),
    }

    if let Some(issues) = install::codex_registration_issues() {
        if issues.is_empty() {
            println!("ok    Codex integration current (AGENTS.md + native hooks + MCP)");
            println!("note  Codex requires one-time hook approval in /hooks after changes");
        } else {
            for issue in issues {
                println!("FAIL  Codex integration: {issue}");
                hard_failures += 1;
            }
        }
    }

    // "No degraded hooks" was the answer whether the hooks were fine or had
    // never run at all — the one check meant to prove the installation works
    // could not tell a clean result from a dead one.
    let liveness = heartbeat::liveness();
    match liveness.severity() {
        heartbeat::Severity::Healthy => println!("ok    {}", liveness.summary()),
        heartbeat::Severity::Unobserved => {
            println!("warn  {}", liveness.summary());
            println!(
                "        a hook that has never fired reports nothing to fail on; start a session, then re-run"
            );
        }
        // Still a warning, not a hard failure: a repeatedly failing hook is
        // loud but recoverable, and selfcheck's nonzero exit is reserved for
        // an installation that cannot work at all.
        heartbeat::Severity::Failing => println!("warn  {}", liveness.summary()),
    }
    for h in liveness.degraded() {
        if let heartbeat::HookState::Failing {
            consecutive,
            last_error,
            ..
        } = &h.state
        {
            println!(
                "warn  hook {} failing ({consecutive} consecutive; last: {})",
                h.name,
                last_error.as_deref().unwrap_or("no message recorded")
            );
        }
    }

    if hard_failures > 0 {
        anyhow::bail!("{hard_failures} hard failure(s)");
    }
    Ok(())
}

fn scan(background: bool) -> anyhow::Result<()> {
    let cfg = config::Config::load()?;

    {
        let mut conn = index::open(&paths::state_dir())?;
        let report = index::scan(&mut conn, &cfg.brain_root, None, &cfg.rings())?;
        println!(
            "indexed {} docs, {} blocks (generation {}, {} file(s) skipped as ring 5+)",
            report.docs, report.blocks, report.generation, report.skipped_high_ring
        );
        // Name the skipped files (a malformed ring declaration also lands
        // here, fail-closed) so a quarantined-by-accident file is visible.
        if !report.skipped.is_empty() {
            let shown: Vec<&str> = report.skipped.iter().take(10).map(String::as_str).collect();
            let more = report.skipped.len() - shown.len();
            println!(
                "  skipped (ring 5+ or unparseable ring frontmatter): {}{}",
                shown.join(", "),
                if more > 0 {
                    format!(" . and {more} more")
                } else {
                    String::new()
                }
            );
        }
        // Unreadable files are a different failure with a different fix
        // (encoding, or an editor/antivirus lock on Windows) — naming them
        // under the ring-5+ reason sends the operator to the wrong place.
        if !report.unreadable.is_empty() {
            let shown: Vec<&str> = report
                .unreadable
                .iter()
                .take(10)
                .map(String::as_str)
                .collect();
            let more = report.unreadable.len() - shown.len();
            println!(
                "  unreadable (invalid UTF-8 or locked): {}{}",
                shown.join(", "),
                if more > 0 {
                    format!(" . and {more} more")
                } else {
                    String::new()
                }
            );
        }
        // An unclosed <private> blanks to end of file by design, fail-closed
        // — often a mere mention of the tag in prose. The design keeps the
        // hiding; the silence was the bug.
        if !report.private_swallowed.is_empty() {
            let shown: Vec<&str> = report
                .private_swallowed
                .iter()
                .take(10)
                .map(String::as_str)
                .collect();
            let more = report.private_swallowed.len() - shown.len();
            println!(
                "  <private> never closed — content blanked to end of file: {}{}",
                shown.join(", "),
                if more > 0 {
                    format!(" . and {more} more")
                } else {
                    String::new()
                }
            );
        }
        if report.blob_dropped > 0 {
            println!(
                "  {} statement(s) dropped as generated blobs (a line over 8 KiB)",
                report.blob_dropped
            );
        }
        // Connection dropped here: the daemon's scan thread is the next writer.
    }
    if background {
        match daemon::call("scan-code", std::time::Duration::from_secs(1)) {
            Some(r) if r.ok => {
                println!("code scan running in the daemon (watch: cfetch status)");
                return Ok(());
            }
            Some(r) => {
                println!(
                    "daemon refused: {}",
                    r.error.unwrap_or_else(|| "unknown error".into())
                );
                return Ok(());
            }
            None => println!("warning: daemon unreachable — running the code scan inline"),
        }
    }
    let mut conn = index::open(&paths::state_dir())?;
    let code = code::scan_code(&mut conn, &cfg.effective_code_roots())?;
    println!(
        "code: {} files, {} symbols (re)parsed, {} import edges",
        code.files, code.symbols, code.edges
    );
    Ok(())
}

/// What a query answer is allowed to COST, in estimated tokens, instead of
/// how many rows it may contain.
///
/// A result count prices eight one-line file matches the same as eight
/// thousand-line blocks, so the same `--limit 8` bought answers two orders of
/// magnitude apart and the Ledger only learned about it afterwards. The
/// budget is applied where the answer is RENDERED, by whoever renders it —
/// local index or remote serving host — so a peer's idea of a reasonable
/// answer cannot spend this host's context window.
///
/// Truncation is never silent. Every cut is followed by a line naming how
/// much went and the exact way to get it back: a narrower query, a lifted cap
/// (`--budget-tokens 0`), or the file range that still holds the text. A short
/// answer is cheap; an answer that quietly lost the hit you needed is wrong.
mod answer {
    use crate::hook_io::estimate_tokens;

    /// Default budget for a `find` answer. One hit is one line, so this binds
    /// only on deep paths or a raised `--limit`; it exists so that no answer
    /// path is unpriced, not because find is the expensive one.
    pub const FIND_BUDGET_TOKENS: u64 = 1200;

    /// Default budget for a `recall` answer, hits or expanded blocks. Blocks
    /// are why this mechanism exists: a hit snippet is capped at 160
    /// characters, a block is whatever the operator wrote.
    pub const RECALL_BUDGET_TOKENS: u64 = 1500;

    /// What the CLI tells a caller to do about a truncated answer. Naming the
    /// escape hatch is the whole point — "results omitted" alone is the
    /// silent drop this mechanism exists to prevent.
    pub const CLI_RECOVERY: &str =
        "narrow the query, or re-run with --budget-tokens 0 for the whole answer";

    /// The MCP surface has no flags, so the pointers already in the answer are
    /// the way back: every hit carries its file and line range.
    pub const MCP_RECOVERY: &str = "narrow the query, or read the file ranges above";

    /// Zero lifts the cap. It is the documented recovery from a truncated
    /// answer, so it can never also mean "nothing fits".
    fn uncapped(budget: u64) -> bool {
        budget == 0
    }

    /// Largest prefix of a `max`-byte string whose estimate still fits.
    /// Binary search over [`estimate_tokens`] rather than an inverse formula,
    /// so the two cannot drift apart when the chars/3.5 heuristic is replaced
    /// by the measured tokenizer.
    fn fitting_bytes(budget: u64, max: usize) -> usize {
        let (mut lo, mut hi) = (0usize, max);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if estimate_tokens(mid) <= budget {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }

    /// A ranked list fitted to a budget.
    struct Fitted {
        entries: Vec<String>,
        dropped: usize,
    }

    /// Keeps the longest RANKED PREFIX of `entries` that fits. Entries arrive
    /// best-first and a prefix is the only truncation that preserves that
    /// order: skipping one fat entry to fit two thin ones hands back a
    /// differently-ranked answer than the one the pipeline computed. At least
    /// one entry always survives — an answer holding nothing answers nothing,
    /// and the drop line still says what that cost.
    fn fit(mut entries: Vec<String>, budget: u64) -> Fitted {
        if uncapped(budget) {
            return Fitted {
                entries,
                dropped: 0,
            };
        }
        let mut chars = 0usize;
        let mut kept = 0usize;
        for entry in &entries {
            // +1 for the newline that joins this entry to the previous one.
            let next = chars + entry.len() + usize::from(kept > 0);
            if kept > 0 && estimate_tokens(next) > budget {
                break;
            }
            chars = next;
            kept += 1;
        }
        let dropped = entries.len() - kept;
        entries.truncate(kept);
        Fitted { entries, dropped }
    }

    /// The same fit over serialized JSON entries. An agent parsing stdout pays
    /// for the bytes it reads, so `--json` is priced on the JSON it emits, not
    /// on a text rendering it never sees.
    pub fn fit_json(
        entries: Vec<serde_json::Value>,
        budget: u64,
    ) -> (Vec<serde_json::Value>, usize) {
        let fitted = fit(entries.iter().map(ToString::to_string).collect(), budget);
        let kept = fitted.entries.len();
        let mut entries = entries;
        entries.truncate(kept);
        (entries, fitted.dropped)
    }

    /// The line that makes a dropped tail honest. Charged ON TOP of the budget
    /// on purpose: a truncation notice that could itself be truncated would
    /// defeat the mechanism.
    fn dropped_note(dropped: usize, budget: u64, recovery: &str) -> String {
        format!("… {dropped} more hit(s) dropped by the {budget}-token answer budget — {recovery}")
    }

    fn clipped_note(omitted: u64, locator: &str) -> String {
        format!(
            "… ~{omitted} more token(s) of this block not shown (answer budget) — read {locator} for the rest"
        )
    }

    /// One block body, clipped to what the budget had left.
    pub struct Clipped {
        pub text: String,
        /// Estimated tokens NOT shown; zero means the body arrived whole.
        pub omitted_tokens: u64,
    }

    /// Clips a block body, preferring the last line boundary that fits: half a
    /// markdown line reads as corrupted content rather than as a truncation.
    /// A body with no newline inside the budget is cut on a character
    /// boundary instead — some text beats none, and the caller always prints
    /// the file range holding the rest.
    fn clip(text: &str, budget: u64) -> Clipped {
        if uncapped(budget) || estimate_tokens(text.len()) <= budget {
            return Clipped {
                text: text.to_string(),
                omitted_tokens: 0,
            };
        }
        let mut end = fitting_bytes(budget, text.len());
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        if let Some(newline) = text[..end].rfind('\n') {
            end = newline;
        }
        Clipped {
            text: text[..end].to_string(),
            omitted_tokens: estimate_tokens(text.len() - end),
        }
    }

    /// One budget spanning several blocks. `expand` hands back every mirrored
    /// copy of a statement at once, so the copies share one allowance instead
    /// of each being granted the whole of it.
    pub struct BlockBudget {
        left: u64,
        uncapped: bool,
    }

    impl BlockBudget {
        pub fn new(budget: u64) -> BlockBudget {
            BlockBudget {
                left: budget,
                uncapped: uncapped(budget),
            }
        }

        pub fn take(&mut self, body: &str) -> Clipped {
            if self.uncapped {
                return Clipped {
                    text: body.to_string(),
                    omitted_tokens: 0,
                };
            }
            if self.left == 0 {
                // Exhausted, which `clip` would read as "no cap" — the exact
                // opposite. The block still gets named by its caller; only its
                // text is withheld.
                return Clipped {
                    text: String::new(),
                    omitted_tokens: estimate_tokens(body.len()),
                };
            }
            let clipped = clip(body, self.left);
            self.left = self
                .left
                .saturating_sub(estimate_tokens(clipped.text.len()));
            clipped
        }
    }

    /// One `find` hit, as the CLI and the MCP server both render it.
    pub fn find_entry(
        path: &str,
        name: Option<&str>,
        kind: Option<&str>,
        start_line: usize,
        end_line: usize,
        token_estimate: u64,
    ) -> String {
        match (name, kind) {
            (Some(name), Some(kind)) => {
                format!("{path}:{start_line}-{end_line}  {kind} {name}  (~{token_estimate} tok)")
            }
            _ => format!("{path}  (file match, ~{token_estimate} tok)"),
        }
    }

    /// One `recall` hit, snippet and mirrors included.
    pub fn hit_entry(
        cite: &str,
        path: &str,
        ring: u8,
        start_line: usize,
        end_line: usize,
        snippet: &str,
        mirrors: &[String],
    ) -> String {
        let mut entry =
            format!("{cite} {path}:{start_line}-{end_line} (ring {ring})\n    {snippet}");
        if !mirrors.is_empty() {
            entry.push_str(&format!("\n    (also at: {})", mirrors.join(", ")));
        }
        entry
    }

    /// A whole ranked listing: the prefix that fits, then — only when
    /// something was cut — the line saying so.
    pub fn listing(entries: Vec<String>, budget: u64, recovery: &str) -> String {
        let fitted = fit(entries, budget);
        let mut out = fitted.entries.join("\n");
        if fitted.dropped > 0 {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&dropped_note(fitted.dropped, budget, recovery));
        }
        out
    }

    /// One expanded block on its way to being rendered.
    pub struct BlockIn {
        pub cite: String,
        pub path: String,
        pub ring: u8,
        pub start_line: usize,
        pub end_line: usize,
        pub text: String,
    }

    /// Expanded blocks under ONE budget. Every block keeps its citation and
    /// file range whatever the budget did to its text: a statement that is
    /// merely unaffordable must still be findable, so nothing ever disappears
    /// without a pointer to where it lives.
    pub fn blocks(blocks: &[BlockIn], budget: u64) -> String {
        let mut running = BlockBudget::new(budget);
        blocks
            .iter()
            .map(|block| {
                let locator = format!("{}:{}-{}", block.path, block.start_line, block.end_line);
                let clipped = running.take(&block.text);
                let mut out = format!(
                    "{} {locator} (ring {})\n{}",
                    block.cite, block.ring, clipped.text
                );
                if clipped.omitted_tokens > 0 {
                    out.push('\n');
                    out.push_str(&clipped_note(clipped.omitted_tokens, &locator));
                }
                out
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

fn find(query: &str, limit: usize, budget_tokens: u64, json: bool) -> anyhow::Result<()> {
    // None-tier: the serving host's code index answers; no local index opens.

    // Serves the existing snapshot ONLY: a full code scan over NFS roots
    // takes minutes and must never ride on an interactive lookup. `cfetch
    // scan` (or the daemon's background scan) owns index freshness.
    let conn = index::open(&paths::state_dir())?;
    let hits = code::find(&conn, query, limit)?;
    if let Some(r) = daemon::call("scan-status", std::time::Duration::from_millis(250))
        && let Some(s) = r.scan
        && s.running
    {
        // stderr so --json stdout stays parseable.
        eprintln!("note: a code scan is running — results may be stale");
    }
    if json {
        // One shape whoever answered: the none-tier branch above has always
        // emitted an object, so a caller parsing `find --json` no longer has
        // to know which host holds the code index.
        let (arr, dropped) = answer::fit_json(
            hits.iter()
                .map(|h| {
                    serde_json::json!({
                        "path": h.path, "name": h.name, "kind": h.kind,
                        "lines": [h.start_line, h.end_line],
                        "tokens_estimated": h.token_estimate,
                        "rank_pct": h.rank_pct,
                    })
                })
                .collect(),
            budget_tokens,
        );
        println!(
            "{}",
            serde_json::json!({
                "hits": arr, "dropped": dropped, "budget_tokens": budget_tokens,
            })
        );
        return Ok(());
    }
    if hits.is_empty() {
        if code::file_count(&conn)? == 0 {
            println!("code index is empty — run `cfetch scan` (or `cfetch scan --async`)");
        } else {
            println!("no hits for \"{query}\"");
        }
        return Ok(());
    }
    let entries = hits
        .iter()
        .map(|h| {
            answer::find_entry(
                &h.path,
                h.name.as_deref(),
                h.kind.as_deref(),
                h.start_line,
                h.end_line,
                h.token_estimate,
            )
        })
        .collect();
    println!(
        "{}",
        answer::listing(entries, budget_tokens, answer::CLI_RECOVERY)
    );
    Ok(())
}

/// Renders one repo map, whoever computed it. The serving host and the
/// none-tier client print the SAME lines — only the coherence footer differs.
/// `served_by` is the serving host's address when the map came over the wire:
/// an empty remote index is not something the local host can fix by scanning.
fn print_map(m: &serve::WireMap, focus: Option<&str>, served_by: Option<&str>) {
    if m.lines.is_empty() {
        match served_by {
            Some(addr) => println!(
                "the code index on serving host {addr} is empty — scanning happens there, on the storage host"
            ),
            None => println!("code index is empty — run `cfetch scan` first"),
        }
        return;
    }
    if focus.is_some() && !m.focus_matched {
        eprintln!("note: --focus matched no path or symbol — showing the unpersonalized map");
    }
    for line in &m.lines {
        println!("{line}");
    }
    if m.lines.len() < m.total_files {
        println!(
            "… {} more file(s) beyond the token budget",
            m.total_files - m.lines.len()
        );
    }
}

fn map_cmd(focus: Option<&str>, budget_tokens: u64) -> anyhow::Result<()> {
    let cfg = config::Config::load()?;
    // None-tier: the serving host's code index answers, exactly as for find.
    // (scan and embed-index stay local — they write.)

    let conn = index::open(&paths::state_dir())?;
    let m = graph::map(&conn, &cfg.effective_code_roots(), focus, budget_tokens)?;
    print_map(&m.into(), focus, None);
    Ok(())
}

fn code_graph_cmd(action: CodeGraphAction) -> anyhow::Result<()> {
    let cfg = config::Config::load()?;
    match action {
        CodeGraphAction::Path {
            from,
            to,
            depth,
            json,
        } => {
            let conn = index::open(&paths::state_dir())?;
            let path =
                graph::dependency_path(&conn, &cfg.effective_code_roots(), &from, &to, depth)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&path)?);
            } else {
                println!("{}", graph::render_dependency_path(&path));
            }
        }
        CodeGraphAction::Impact {
            target,
            depth,
            limit,
            json,
        } => {
            let conn = index::open(&paths::state_dir())?;
            let impact = graph::dependency_impact(
                &conn,
                &cfg.effective_code_roots(),
                &target,
                depth,
                limit,
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&impact)?);
            } else {
                println!("{}", graph::render_dependency_impact(&impact));
            }
        }
        CodeGraphAction::Context {
            target,
            depth,
            limit,
            json,
        } => {
            let conn = index::open(&paths::state_dir())?;
            let context = graph::dependency_context(
                &conn,
                &cfg.effective_code_roots(),
                &target,
                depth,
                limit,
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&context)?);
            } else {
                println!("{}", graph::render_dependency_context(&context));
            }
        }
        CodeGraphAction::Symbol { query, limit, json } => {
            let conn = index::open(&paths::state_dir())?;
            let context = graph::symbol_context(&conn, &cfg.effective_code_roots(), &query, limit)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&context)?);
            } else {
                println!("{}", graph::render_symbol_context(&context));
            }
        }
    }
    Ok(())
}

fn print_knowledge_graph(graph: &knowledge_graph::KnowledgeGraph) {
    println!(
        "knowledge graph: {} of {} document(s), {} curated link(s), {} unresolved reference(s), generation {}",
        graph.nodes.len(),
        graph.total_nodes,
        graph.total_edges,
        graph.unresolved_references,
        graph.generation,
    );
    if let Some(requested) = &graph.requested_focus {
        match &graph.resolved_focus {
            Some(path) => println!("focus: {requested:?} → {path}"),
            None if !graph.ambiguous_focus.is_empty() => println!(
                "focus: {requested:?} is ambiguous across {} — showing the connected overview",
                graph.ambiguous_focus.join(", ")
            ),
            None => println!(
                "focus: {requested:?} matched no document — showing the connected overview"
            ),
        }
    }
    if graph.nodes.is_empty() {
        println!("no indexed Markdown documents — run `cfetch scan`");
        return;
    }
    println!("\ndocuments:");
    for node in &graph.nodes {
        println!(
            "{} r{}  ←{:<3} →{:<3} {:<12} {}",
            if node.focused { "●" } else { " " },
            node.ring,
            node.inbound,
            node.outbound,
            node.kind,
            node.path,
        );
    }
    if !graph.edges.is_empty() {
        println!("\ncurated links:");
        for edge in &graph.edges {
            println!("{} → {}", edge.from, edge.to);
        }
        if graph.omitted_edges > 0 {
            println!(
                "… {} more link(s) omitted from this bounded view",
                graph.omitted_edges
            );
        }
    }
}

fn graph_cmd(
    focus: Option<&str>,
    slice: Option<&str>,
    limit: usize,
    json: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        (1..=200).contains(&limit),
        "--limit must be between 1 and 200"
    );
    let cfg = config::Config::load()?;

    let conn = index::ensure_fresh(&paths::state_dir(), &cfg.brain_root, None, &cfg.rings())?;
    let graph = if let Some(slice) = slice {
        let slices = cfg.slice_model()?;
        anyhow::ensure!(
            slice == config::ROOT_SLICE || slices.names().any(|name| name == slice),
            "unknown slice {slice:?}"
        );
        knowledge_graph::build_matching(&conn, focus, limit, |path| slices.contains(slice, path))?
    } else {
        knowledge_graph::build(&conn, focus, limit)?
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&graph)?);
    } else {
        print_knowledge_graph(&graph);
    }
    Ok(())
}

/// Coherence footer for answers that went through a serving daemon.
fn print_served_by(resp: &daemon::Response) {
    println!("\n{}", daemon::freshness_line(resp));
}

/// The rendered entries of a served hit list, best first.
fn wire_hit_entries(hits: &[serve::WireHit]) -> Vec<String> {
    hits.iter()
        .map(|h| {
            answer::hit_entry(
                &h.cite,
                &h.path,
                h.ring,
                h.start_line,
                h.end_line,
                &h.snippet,
                &h.mirrors,
            )
        })
        .collect()
}

/// recall/expand answered by a serving daemon (remote TCP or the local unix
/// socket) — shared rendering for both.
fn recall_served(
    resp: &daemon::Response,
    id: Option<&str>,
    budget_tokens: u64,
    json: bool,
) -> anyhow::Result<()> {
    if let Some(cite) = id {
        let blocks = resp.blocks.clone().unwrap_or_default();
        if blocks.is_empty() {
            if json {
                println!(
                    "{}",
                    serde_json::json!({"blocks": [], "cite": cite,
                    "origin": resp.origin, "generation": resp.generation, "fresh": resp.fresh,
                    "stale_note": resp.stale_note})
                );
            } else {
                println!(
                    "no block with citation {cite} (content-addressed ids change when the entry changes)"
                );
                print_served_by(resp);
            }
            return Ok(());
        }
        if json {
            let mut budget = answer::BlockBudget::new(budget_tokens);
            for b in &blocks {
                let clipped = budget.take(&b.text);
                println!(
                    "{}",
                    serde_json::json!({
                        "cite": b.cite, "path": b.path, "ring": b.ring,
                        "lines": [b.start_line, b.end_line], "text": clipped.text,
                        "omitted_tokens": clipped.omitted_tokens,
                        "budget_tokens": budget_tokens,
                        "origin": resp.origin, "generation": resp.generation, "fresh": resp.fresh,
                        "stale_note": resp.stale_note,
                    })
                );
            }
            return Ok(());
        }
        let ins: Vec<answer::BlockIn> = blocks
            .iter()
            .map(|b| answer::BlockIn {
                cite: b.cite.clone(),
                path: b.path.clone(),
                ring: b.ring,
                start_line: b.start_line,
                end_line: b.end_line,
                text: b.text.clone(),
            })
            .collect();
        println!("{}", answer::blocks(&ins, budget_tokens));
        print_served_by(resp);
        return Ok(());
    }
    let hits = resp.hits.clone().unwrap_or_default();
    // The serving host's degradation is the caller's to see: it ranked on our
    // behalf against endpoints we cannot reach, so if it fell back we would
    // otherwise never know.
    if let Some(note) = &resp.note {
        eprintln!("cfetch recall: {note}");
    }
    if json {
        let (arr, dropped) = answer::fit_json(
            hits.iter()
                .map(|h| {
                    serde_json::json!({
                        "cite": h.cite, "path": h.path, "ring": h.ring,
                        "lines": [h.start_line, h.end_line], "snippet": h.snippet,
                        "mirrors": h.mirrors,
                    })
                })
                .collect(),
            budget_tokens,
        );
        println!(
            "{}",
            serde_json::json!({
                "hits": arr, "linked": resp.linked.as_deref().unwrap_or_default().iter()
                    .map(|(path, ring)| serde_json::json!({"path": path, "ring": ring})).collect::<Vec<_>>(),
                "dropped": dropped, "budget_tokens": budget_tokens,
                "origin": resp.origin, "generation": resp.generation,
                "fresh": resp.fresh, "stale_note": resp.stale_note, "note": resp.note,
            })
        );
    } else if hits.is_empty() {
        println!("no hits");
        print_served_by(resp);
    } else {
        println!(
            "{}",
            answer::listing(wire_hit_entries(&hits), budget_tokens, answer::CLI_RECOVERY)
        );
        if let Some(linked) = &resp.linked
            && !linked.is_empty()
        {
            println!("\nlinked (curated wikilinks, 1 hop from top hits):");
            for (path, ring) in linked {
                println!("    {path} (ring {ring})");
            }
        }
        print_served_by(resp);
        println!("expand a hit: cfetch recall --id <citation>");
    }
    Ok(())
}

/// All memory queries go to the daemon before reading tree-owned config.
/// An unavailable daemon is a bounded error, never a synchronous NFS scan.
#[allow(clippy::too_many_arguments)]
fn recall(
    query: &str,
    id: Option<&str>,
    expand: bool,
    fresh: bool,
    semantic: bool,
    hybrid: bool,
    slice: Option<&str>,
    limit: usize,
    budget_tokens: u64,
    json: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        id.is_some() || !query.trim().is_empty(),
        "empty query (pass search terms or --id <citation>)"
    );
    anyhow::ensure!(
        (1..=100).contains(&limit),
        "--limit must be between 1 and 100"
    );
    let body = match id {
        Some(cite) => serde_json::json!({"op": "expand", "cite": cite, "slice": slice}),
        None => serde_json::json!({"op": "recall", "query": query, "limit": limit,
            "slice": slice, "semantic": semantic, "hybrid": hybrid, "expand": expand}),
    };
    let mut body = body;
    body["freshness"] = serde_json::json!(if fresh { "strict" } else { "cached" });
    let response = daemon::memory_query(
        &body,
        std::time::Duration::from_secs(if semantic || hybrid { 60 } else { 8 }),
    )?;
    recall_served(&response, id, budget_tokens, json)
}

/// Manual/debug view over ring-5 evidence in the shared tree. The autonomous
/// worker normally settles these candidates; this surface remains available
/// for intervention and inspection across every host.
fn staging_cmd(action: MemoriesAction) -> anyhow::Result<()> {
    let cfg = config::Config::load()?;
    let ex = exhaust::Exhaust::from_config(&cfg);
    let dir = ex.staging_dir.clone();
    match action {
        MemoriesAction::List { json } => {
            let rows = staging::list(&dir);
            if json {
                let arr: Vec<_> = rows
                    .iter()
                    .map(|c| {
                        serde_json::json!({
                            "id": c.id, "reason": c.reason, "kind": c.kind,
                            "session_id": c.session, "host": c.host, "ts": c.ts,
                            "payload": c.payload,
                        })
                    })
                    .collect();
                println!("{}", serde_json::json!(arr));
            } else if rows.is_empty() {
                println!("memories are empty — no captured evidence awaiting maintenance");
                println!("  ({})", dir.display());
            } else {
                for c in &rows {
                    let session = c.session.get(..8).unwrap_or(&c.session);
                    println!(
                        "{}  {}  {}  {}  session {}  {}",
                        c.id, c.reason, c.kind, c.host, session, c.payload
                    );
                }
                println!(
                    "\ndebug manually: cfetch maintain packet <id> | cfetch memories dismiss <id>"
                );
            }
        }
        MemoriesAction::Consume { id } => {
            if staging::consume(&dir, &id)? {
                // The file is gone, so the stream is what remembers the
                // decision and keeps the trap from re-staging it.
                let _ = ex.record_decision(&id, "consume");
                println!("consumed {id}");
            } else {
                anyhow::bail!(
                    "no staged candidate {id} (never flagged, or already consumed/dismissed)"
                );
            }
        }
        MemoriesAction::Dismiss { id } => {
            if staging::dismiss(&dir, &id)? {
                let _ = ex.record_decision(&id, "dismiss");
                println!("dismissed {id} (kept in {}/dismissed)", dir.display());
            } else {
                anyhow::bail!(
                    "no staged candidate {id} (never flagged, or already consumed/dismissed)"
                );
            }
        }
    }
    Ok(())
}

fn maintain_cmd(action: MaintainAction) -> anyhow::Result<()> {
    let cfg = config::Config::load()?;
    match action {
        MaintainAction::Run { limit, json } => {
            if let Some(limit) = limit {
                anyhow::ensure!(
                    (1..=config::MAX_MAINTENANCE_CANDIDATES).contains(&limit),
                    "--limit must be between 1 and {}",
                    config::MAX_MAINTENANCE_CANDIDATES
                );
            }
            let mut model = maintenance_model::MaintenanceClient::new(&cfg.maintenance)?;
            let report = maintenance::run_once_with(
                &cfg,
                &mut model,
                limit.unwrap_or(cfg.maintenance.max_candidates),
            )?;
            if json {
                println!("{}", serde_json::to_string(&report)?);
            } else if report.paused {
                println!(
                    "maintenance paused: {}",
                    maintenance::pause_reason(&cfg)
                        .unwrap_or_else(|| "pause marker present".into())
                );
            } else {
                println!(
                    "maintenance cycle: {} examined, {} applied, {} dismissed, {} noop, {} exception(s)",
                    report.examined,
                    report.applied,
                    report.dismissed,
                    report.noops,
                    report.exceptions
                );
            }
        }
        MaintainAction::Pause { reason } => {
            let reason = reason.join(" ");
            maintenance::pause(&cfg, &reason)?;
            println!("maintenance paused: {reason}");
        }
        MaintainAction::Resume => {
            maintenance::resume(&cfg)?;
            println!("maintenance resumed");
        }
        MaintainAction::History { limit, json } => {
            let events: Vec<_> = maintenance::history(&cfg).into_iter().take(limit).collect();
            if json {
                println!("{}", serde_json::to_string(&events)?);
            } else if events.is_empty() {
                println!("no automatic maintenance activity recorded");
            } else {
                for event in events {
                    println!(
                        "{}  {:<10}  {}  {}",
                        event.id,
                        format!("{:?}", event.outcome).to_ascii_lowercase(),
                        event.target.as_deref().unwrap_or("no memory write"),
                        event.detail
                    );
                }
            }
        }
        MaintainAction::Packet { candidate_id, json } => {
            let packet = maintenance::packet(&cfg, &candidate_id)?;
            if json {
                println!("{}", serde_json::to_string(&packet)?);
            } else {
                println!(
                    "maintenance evidence packet for {}",
                    packet.candidate.candidate.id
                );
                println!(
                    "  candidate evidence: {}  raw events: {}{}",
                    packet.candidate.evidence_id,
                    packet.events.len(),
                    if packet.events_truncated {
                        " (bounded; more matched)"
                    } else {
                        ""
                    }
                );
                if !packet.unreadable_streams.is_empty() {
                    println!(
                        "  WARNING: {} exhaust stream(s) unreadable",
                        packet.unreadable_streams.len()
                    );
                }
                if let Some(target) = &packet.target_snapshot {
                    println!(
                        "  current target: {} (ring {}, {})",
                        target.path,
                        target.ring,
                        if target.exists { "present" } else { "absent" }
                    );
                }
                println!(
                    "  relevant statements: {}",
                    packet.relevant_statements.len()
                );
                for statement in &packet.relevant_statements {
                    println!(
                        "    {}  {}:{}-{}  ring {}",
                        statement.cite,
                        statement.path,
                        statement.start_line,
                        statement.end_line,
                        statement.ring
                    );
                }
                if let Some(note) = &packet.context_note {
                    println!("  note: {note}");
                }
                println!("\nAgent input (complete, machine-readable):");
                println!("{}", serde_json::to_string_pretty(&packet)?);
                println!(
                    "\nSubmit a proposal: cfetch maintain submit --file proposal.json\n\
                     The submission stays in ring 5; only `maintain apply` can cross inward."
                );
            }
        }
        MaintainAction::Submit { file, json } => {
            use std::io::Read as _;
            let mut text = String::new();
            match file.as_deref() {
                Some(path) if path != std::path::Path::new("-") => {
                    text = std::fs::read_to_string(path)
                        .with_context(|| format!("read proposal input {}", path.display()))?;
                }
                _ => {
                    std::io::stdin()
                        .read_to_string(&mut text)
                        .context("read proposal JSON from stdin")?;
                }
            }
            let input: maintenance::ProposalInput =
                serde_json::from_str(&text).context("decode proposal JSON")?;
            let result = maintenance::submit(&cfg, input)?;
            if json {
                println!("{}", serde_json::to_string(&result)?);
            } else {
                println!(
                    "{} {} in ring-5 quarantine",
                    if result.created {
                        "submitted"
                    } else {
                        "already recorded"
                    },
                    result.proposal.id
                );
                println!(
                    "review it independently: cfetch maintain review {} --file review.json",
                    result.proposal.id
                );
            }
        }
        MaintainAction::Review { id, file, json } => {
            use std::io::Read as _;
            let mut text = String::new();
            match file.as_deref() {
                Some(path) if path != std::path::Path::new("-") => {
                    text = std::fs::read_to_string(path)
                        .with_context(|| format!("read review input {}", path.display()))?;
                }
                _ => {
                    std::io::stdin()
                        .read_to_string(&mut text)
                        .context("read review JSON from stdin")?;
                }
            }
            let input: maintenance::ReviewInput =
                serde_json::from_str(&text).context("decode review JSON")?;
            let (review, created) = maintenance::submit_review(&cfg, &id, input)?;
            if json {
                println!(
                    "{}",
                    serde_json::json!({"created": created, "review": review})
                );
            } else {
                println!(
                    "{} {} for {} ({:?})",
                    if created {
                        "recorded"
                    } else {
                        "already recorded"
                    },
                    review.id,
                    review.proposal_id,
                    review.verdict
                );
                println!("verify it: cfetch maintain verify {}", review.proposal_id);
            }
        }
        MaintainAction::List { json } => {
            let proposals = maintenance::list(&cfg);
            if json {
                println!("{}", serde_json::to_string(&proposals)?);
            } else if proposals.is_empty() {
                println!("no maintenance proposals");
            } else {
                for proposal in proposals {
                    println!(
                        "{}  {:<9}  {:<10}  {}  [{}]",
                        proposal.id,
                        proposal.state,
                        format!("{:?}", proposal.transition).to_ascii_lowercase(),
                        proposal.target.as_deref().unwrap_or("no memory write"),
                        proposal.candidates.join(", ")
                    );
                }
            }
        }
        MaintainAction::Show { id, json } => {
            let (state, proposal) = maintenance::get(&cfg, &id)?;
            let review = maintenance::get_review(&cfg, &id)?;
            if json {
                println!(
                    "{}",
                    serde_json::json!({"state": state, "proposal": proposal, "review": review})
                );
            } else {
                println!("proposal {} ({state})", proposal.id);
                println!("{}", serde_json::to_string_pretty(&proposal)?);
                match review {
                    Some(review) => {
                        println!("\nsemantic review {}:", review.id);
                        println!("{}", serde_json::to_string_pretty(&review)?);
                    }
                    None => println!("\nsemantic review: not recorded"),
                }
            }
        }
        MaintainAction::Verify { id, json } => {
            let report = maintenance::verify(&cfg, &id)?;
            if json {
                println!("{}", serde_json::to_string(&report)?);
            } else {
                for check in &report.checks {
                    println!(
                        "{}  {:<20} {}",
                        if check.ok { "ok  " } else { "FAIL" },
                        check.name,
                        check.detail
                    );
                }
                if let Some(diff) = &report.diff {
                    println!("\nexact proposed diff:\n{diff}");
                }
                if report.valid {
                    println!("verified {}", report.proposal_id);
                    println!(
                        "approval token: {}",
                        report.approval_token.as_deref().unwrap_or_default()
                    );
                    println!(
                        "apply exactly this revision: cfetch maintain apply {} --approval-token {}",
                        report.proposal_id,
                        report.approval_token.as_deref().unwrap_or_default()
                    );
                } else {
                    anyhow::bail!("proposal failed verification");
                }
            }
        }
        MaintainAction::Apply { id, approval_token } => {
            let proposal = maintenance::apply(&cfg, &id, &approval_token)?;
            if proposal.transition.changes_memory() {
                println!(
                    "applied {} to {}",
                    proposal.id,
                    proposal.target.as_deref().unwrap_or_default()
                );
                println!("reversible now: cfetch maintain revert {}", proposal.id);
                println!(
                    "after committing the brain tree: cfetch maintain finalize {}",
                    proposal.id
                );
            } else {
                println!("applied decision {}", proposal.id);
                println!("finalize it: cfetch maintain finalize {}", proposal.id);
            }
        }
        MaintainAction::AutoApply { id } => {
            let proposal = maintenance::automatic_apply(&cfg, &id)?;
            println!(
                "automatically finalized {} ({})",
                proposal.id,
                proposal.target.as_deref().unwrap_or("no memory write")
            );
            println!(
                "reversible while exact bytes remain: cfetch maintain revert {}",
                proposal.id
            );
        }
        MaintainAction::Revert { id } => {
            let proposal = maintenance::revert(&cfg, &id)?;
            println!(
                "reverted {} (source candidates remain pending)",
                proposal.id
            );
        }
        MaintainAction::Reject { id } => {
            maintenance::reject(&cfg, &id)?;
            println!("rejected {id} (source candidates remain pending)");
        }
        MaintainAction::Finalize { id } => {
            let result = maintenance::finalize(&cfg, &id)?;
            println!(
                "{} {} and settled {} source candidate(s)",
                if result.already_finalized {
                    "reconciled"
                } else {
                    "finalized"
                },
                result.proposal_id,
                result.candidate_ids.len()
            );
        }
    }
    Ok(())
}

/// `init` deliberately does not load the config: it must work before there is
/// anything to configure, which is the only moment it is useful.
fn init_cmd(path: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    let root = path.unwrap_or_else(paths::default_brain_root);
    let out = init::run(&root)?;
    println!("brain tree at {}", out.root.display());
    for (name, fresh) in &out.dirs {
        println!("  {} {name}/", if *fresh { "created" } else { "present" });
    }
    for (name, written) in &out.files {
        println!("  {} {name}", if *written { "wrote  " } else { "kept   " });
    }
    println!("\nreserved, not created — a rule keys on each of these if it ever exists:");
    for (name, why) in init::reserved() {
        println!("  {name}  — {why}");
    }
    Ok(())
}

/// The bench reads only what the harness recorded, so it needs neither the
/// config nor a local index: a none-tier host can still answer whether cfetch
/// paid for itself on the sessions it ran.
fn bench_cmd(since_days: u64, json: bool) -> anyhow::Result<()> {
    let report = bench::build(
        &bench::BenchPaths::defaults(),
        since_days,
        std::time::SystemTime::now(),
    );
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", bench::render(&report));
    }
    Ok(())
}

/// Coarse age of a unix timestamp. The failure history answers "just now or
/// long ago"; a wall-clock date would need a calendar dependency to say less.
fn age(now: i64, ts: i64) -> String {
    let secs = now.saturating_sub(ts).max(0);
    match secs {
        s if s < 90 => format!("{s}s ago"),
        s if s < 5400 => format!("{}m ago", s / 60),
        s if s < 172_800 => format!("{}h ago", s / 3600),
        s => format!("{}d ago", s / 86_400),
    }
}

/// The ring-6 query surface: the same normalized signature the traps key on,
/// asked instead of acted on. Reads the shared tree directly, exactly like
/// `staging` — there is no index to be stale about, and none-tier hosts hold
/// no exhaust of their own to answer from.
fn failures_cmd(query: &str, limit: usize, json: bool) -> anyhow::Result<()> {
    let cfg = config::Config::load()?;
    let ex = exhaust::Exhaust::from_config(&cfg);
    let history = ex.failure_history(query, limit);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    if json {
        let arr: Vec<_> = history
            .matches
            .iter()
            .map(|m| {
                serde_json::json!({
                    "norm": m.norm, "failures": m.failures, "sessions": m.sessions,
                    "recovered_sessions": m.recovered_sessions, "hosts": m.hosts,
                    "first_ts": m.first_ts, "last_ts": m.last_ts,
                    "last_command": m.last_command, "staged": m.staged,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "signatures": arr, "signatures_total": history.signatures,
                "failures_total": history.failures, "unreadable": history.unreadable,
            })
        );
        return Ok(());
    }

    // A partly-read tree must never look like a clean "never happened".
    for note in &history.unreadable {
        eprintln!("cfetch failures: skipped {note}");
    }
    if history.matches.is_empty() {
        if history.signatures == 0 {
            println!(
                "no failing command captured yet in {}",
                ex.logs_dir.display()
            );
        } else if query.trim().is_empty() {
            // Everything matches an empty query, so only --limit 0 lands here.
            println!(
                "{} failing signature(s) captured, none asked for",
                history.signatures
            );
        } else {
            println!(
                "no match for \"{query}\" among {} failing signature(s)",
                history.signatures
            );
        }
        return Ok(());
    }
    for m in &history.matches {
        println!("{}", m.norm);
        // A hand-edited line can carry no host; say so rather than render a
        // gap where the fleet answer belongs.
        let hosts = if m.hosts.is_empty() {
            "an unnamed host".to_string()
        } else {
            m.hosts.join(", ")
        };
        println!(
            "  {} failure(s) in {} session(s) on {}; first {}, last {}",
            m.failures,
            m.sessions,
            hosts,
            age(now, m.first_ts),
            age(now, m.last_ts),
        );
        if m.recovered_sessions > 0 {
            println!(
                "  recovered in {} session(s) — the same signature later succeeded",
                m.recovered_sessions
            );
        }
        if !m.last_command.is_empty() {
            println!("  last: {}", m.last_command);
        }
        if m.staged {
            println!("  already a ring-5 candidate (cfetch memories list)");
        }
        println!();
    }
    println!(
        "{} of {} failing signature(s), {} failure(s) captured",
        history.matches.len(),
        history.signatures,
        history.failures
    );
    Ok(())
}

fn audit_cmd(json: bool) -> anyhow::Result<()> {
    let cfg = config::Config::load()?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let ledger = ledger::load_from(&paths::logs_dir(&cfg.brain_root));
    let report = audit::build(
        &audit::AuditPaths::defaults(),
        &ledger,
        &heartbeat::liveness(),
        cfg.budget_chars,
        now,
    );
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", audit::render(&report));
    }
    Ok(())
}

/// What this host can honestly say about its derived catalog.
///
/// A boolean staleness flag ages into meaninglessness: an index a minute
/// behind the tree and one nine days behind print the same word, and the
/// catalog carries no build timestamp to recover the difference from. The
/// heartbeat records when the disagreement started, so the age here is
/// measured rather than guessed.
fn index_liveness_line(
    cfg: &config::Config,
    state: &std::path::Path,
    native: Option<&std::path::Path>,
) -> (heartbeat::Severity, String) {
    let conn = match index::open(state) {
        Ok(c) => c,
        Err(e) => {
            return (
                heartbeat::Severity::Failing,
                format!("index: unavailable ({e})"),
            );
        }
    };
    let tree = index::tree_fingerprint(&cfg.brain_root, native, &cfg.rings());
    let verdict =
        heartbeat::observe_index_in(state, index::stored_fingerprint(&conn).as_deref(), &tree);
    let line = match verdict {
        heartbeat::IndexLiveness::NeverScanned => verdict.describe(),
        _ => format!(
            "{} (generation {})",
            verdict.describe(),
            index::generation(&conn)
        ),
    };
    (verdict.severity(), line)
}

fn delivery_status_line(agent: &str, verification: Option<(u64, u64)>) -> String {
    let agent = match agent {
        agent_session::AGENT_CLAUDE => "Claude",
        agent_session::AGENT_CODEX => "Codex",
        agent_session::AGENT_GEMINI => "Gemini",
        agent_session::AGENT_CURSOR => "Cursor",
        other => other,
    };
    match verification {
        Some((fired, delivered)) => format!(
            "delivery: {fired} hook firing(s) observed, {delivered} injection(s) verified ({agent} transcript)"
        ),
        None => format!(
            "delivery: not measurable from newest {agent} transcript (no recognizable cfetch hook-delivery records)"
        ),
    }
}

fn status() -> anyhow::Result<()> {
    let mut runtime = runtime_status::refresh_static()?;
    let daemon_running = daemon::call("ping", std::time::Duration::from_millis(300)).is_some();
    runtime_status::apply_daemon_observation(&mut runtime, daemon_running);
    println!("{}", runtime_status::render_line(&runtime));
    daemon::status()?;
    // Diagnostics must survive a broken config: the paths below fall back to
    // the defaults rather than making `status` the second casualty.
    let cfg = match config::Config::load() {
        Ok(cfg) => cfg,
        Err(e) => {
            println!("config: {e} — showing the default locations below");
            config::Config::default()
        }
    };
    let logs = paths::logs_dir(&cfg.brain_root);
    let loaded = ledger::read(&logs);
    for note in &loaded.unreadable {
        println!("ledger: UNREADABLE stream {note}");
    }
    let hosts: Vec<String> = loaded.hosts.iter().cloned().collect();
    let ledger = loaded.ledger;
    let sessions = ledger.sessions.len();
    let injected: u64 = ledger
        .sessions
        .values()
        .flat_map(|s| s.by_source.values())
        .map(|t| t.tokens_estimated)
        .sum();
    // A ledger nobody has written to is not a ledger of zero cost. Printing
    // "0 session(s), ~0 tokens" for it would answer a question that was never
    // asked of the disk.
    if sessions == 0 && hosts.is_empty() {
        println!(
            "ledger: NEVER WRITTEN in {} — no host has booked an injection here, so there is nothing to count (unmeasured, not zero)",
            logs.display()
        );
    } else {
        println!(
            "ledger: {sessions} session(s) from {} host(s){} in {}",
            hosts.len(),
            if hosts.is_empty() {
                String::new()
            } else {
                format!(" ({})", hosts.join(", "))
            },
            logs.display()
        );
        println!("  estimated: ~{injected} tokens injected by cfetch (chars/3.5 heuristic)");
    }
    // Measured truth from transcripts, side by side with the estimate — and
    // clearly labeled absent when the transcript could not be parsed.
    let mut measured = ledger::MeasuredUsage::default();
    let mut measured_sessions = 0usize;
    for s in ledger.sessions.values() {
        if !s.measured.is_zero() {
            measured_sessions += 1;
            measured.accumulate(&s.measured);
        }
    }
    if measured_sessions > 0 {
        println!(
            "  measured:  {} api call(s) over {measured_sessions} session(s): {} input / {} output / {} cache-read / {} cache-created tokens (transcript ground truth)",
            measured.api_calls,
            measured.input_tokens,
            measured.output_tokens,
            measured.cache_read_input_tokens,
            measured.cache_creation_input_tokens
        );
    } else {
        println!("  measured:  none — no transcript usage booked yet, numbers above are estimates");
    }
    // Transcript-VERIFIED delivery: did our hook output actually enter the
    // conversation? Read from the newest transcript, never assumed — and a
    // drifted format is reported as unverifiable, never as zero.
    let transcript_roots = [
        (agent_session::AGENT_CLAUDE, paths::native_projects_root()),
        (agent_session::AGENT_CODEX, paths::codex_sessions_root()),
        (agent_session::AGENT_GEMINI, paths::gemini_sessions_root()),
        (agent_session::AGENT_CURSOR, paths::cursor_sessions_root()),
    ];
    match transcript::newest_transcript_among(&transcript_roots) {
        None => println!(
            "delivery: no supported transcripts found (measurement gap; checked Claude, Codex, Gemini and Cursor)"
        ),
        Some(t) => {
            let agent = agent_session::agent_source_for_path(&t).unwrap_or("unknown-agent");
            println!(
                "{}",
                delivery_status_line(agent, transcript::verified_injections(&t))
            );
        }
    }
    // Ring 5/6 live in the tree, so their figures are the fleet's, not this
    // machine's.
    let ex = exhaust::Exhaust::from_config(&cfg);
    let ring56 = ex.stats();
    if ring56.staged_total > 0 {
        let reasons = ring56
            .staged_by_reason
            .iter()
            .map(|(r, n)| format!("{r}: {n}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "memories: {} ring-5 candidate(s) awaiting autonomous maintenance [{reasons}] in {}",
            ring56.staged_total,
            ex.staging_dir.display()
        );
    } else if ring56.bytes == 0 {
        // Staging is fed by the ring-6 capture traps. With no exhaust stream
        // anywhere in the tree, no turn has ever been examined, so an empty
        // queue says nothing about whether anything was worth flagging.
        println!(
            "memories: 0 ring-5 candidates — UNOBSERVED: no ring-6 exhaust has ever been written, so no turn has been examined"
        );
    } else {
        println!("memories: no ring-5 candidates awaiting maintenance (measured)");
    }
    println!(
        "exhaust: {} of ring-6 stream in {}",
        jsonl::human_bytes(ring56.bytes),
        logs.display()
    );
    // Semantic coverage belongs in status, not only in a query's warning:
    // an operator must be able to SEE that the vectors are missing before a
    // ranking quietly falls back to lexical.
    if cfg.embeddings.enabled {
        match semantic_status(&cfg) {
            Ok(line) => println!("{line}"),
            Err(e) => println!("semantic: unavailable ({e})"),
        }
    }
    // Same reasoning for reranking: an operator must see that the second
    // stage is configured and reachable BEFORE a query quietly answers in
    // retrieval order.
    if cfg.rerank.enabled {
        match rerank::RerankClient::new(&cfg.rerank) {
            Ok(c) => println!(
                "rerank: {} over the top {} hit(s)",
                c.model(),
                c.candidates()
            ),
            Err(e) => println!("rerank: unavailable ({e})"),
        }
    }
    let state = paths::state_dir();
    println!("{}", index_liveness_line(&cfg, &state, None).1);
    let bytes: u64 = std::fs::read_dir(&state)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0);
    println!("state:  {} ({} KiB)", state.display(), bytes / 1024);
    Ok(())
}

/// Lists the slices and what each one holds.
///
/// Counts are derived from the indexed paths at call time, so what this
/// prints is what a `--slice` query would actually match — a stored count
/// could disagree with the configuration.
fn slices(json: bool) -> anyhow::Result<()> {
    let cfg = config::Config::load()?;
    let model = cfg.slice_model()?;
    let conn = index::open_ro(&paths::state_dir())?;
    let docs = index::doc_block_counts(&conn)?;

    // Every configured slice appears even when empty: a slice that claims
    // nothing is a configuration mistake worth seeing, not an absence.
    let mut order: Vec<String> = model.names().map(str::to_string).collect();
    order.push(config::ROOT_SLICE.to_string());
    let mut counts: std::collections::HashMap<&str, (usize, usize)> =
        order.iter().map(|n| (n.as_str(), (0, 0))).collect();
    for (path, blocks) in &docs {
        let name = model.slice_for(path);
        let e = counts.entry(name).or_insert((0, 0));
        e.0 += 1;
        e.1 += blocks;
    }

    if json {
        let arr: Vec<_> = order
            .iter()
            .map(|n| {
                let (d, b) = counts.get(n.as_str()).copied().unwrap_or((0, 0));
                serde_json::json!({"slice": n, "docs": d, "blocks": b})
            })
            .collect();
        println!("{}", serde_json::json!({"slices": arr}));
        return Ok(());
    }
    if model.is_empty() {
        println!("no slices configured — the whole tree is one implicit slice");
    }
    for n in &order {
        let (d, b) = counts.get(n.as_str()).copied().unwrap_or((0, 0));
        let prefixes = match model.prefixes_of(n) {
            None => "the whole tree, minus anything a slice claims".to_string(),
            Some(p) => p.join(", "),
        };
        println!("{n}: {d} doc(s), {b} block(s) — {prefixes}");
    }
    Ok(())
}

fn semantic_status(cfg: &config::Config) -> anyhow::Result<String> {
    let spec = cfg.embeddings.spec();
    let shared = vectors::VectorStore::open(&cfg.brain_root, &spec)?.len();

    let conn = index::open_ro(&paths::state_dir())?;
    let (embedded, total) = index::vector_coverage(&conn, &spec)?;
    Ok(embed::coverage_status_line(&spec, embedded, total, shared))
}

fn main() {
    // Hook invocations bypass clap entirely: a parse failure would exit 2,
    // which is the harness's BLOCKING code (a Stop hook exiting 2 traps the
    // session in a loop). Hooks must reach hooks::run no matter what.
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(String::as_str) == Some("hook") {
        let event = argv.get(2).cloned().unwrap_or_default();
        let agent = argv
            .windows(2)
            .find(|pair| pair[0] == "--agent")
            .map(|pair| pair[1].as_str())
            .or_else(|| argv.iter().find_map(|arg| arg.strip_prefix("--agent=")));
        hooks::run(&event, agent);
        return;
    }
    let cli = Cli::parse();
    match cli.command {
        #[cfg(all(target_os = "linux", feature = "native-openvino"))]
        Command::NativeWorker { parent_pid } => {
            if let Err(error) = native_worker::run_stdio(parent_pid) {
                eprintln!("cfetch native worker: {error:#}");
                std::process::exit(1);
            }
        }
        #[cfg(all(target_os = "linux", feature = "native-openvino"))]
        Command::NativeProbe {
            plan,
            evidence,
            policy_sha256,
        } => {
            if let Err(error) = native_adapter::probe(&plan, &evidence, policy_sha256) {
                eprintln!("cfetch native probe: {error:#}");
                std::process::exit(1);
            }
        }
        #[cfg(feature = "embedded-embeddings")]
        Command::QualifyModel {
            model_dir,
            reference,
        } => {
            if let Err(error) = local_embedding::qualify(&model_dir, &reference) {
                eprintln!("cfetch qualify-model: {error:#}");
                std::process::exit(1);
            }
        }
        Command::Import { source } => match source {
            ImportSource::Openwolf { path, dry_run } => {
                let brain = crate::paths::default_brain_root();
                if dry_run {
                    println!("dry run — nothing will be written");
                }
                // One code path builds the report; the dry run simply
                // does not execute it. Preview and act cannot disagree.
                let outcome = if dry_run {
                    import::plan_openwolf(&path, &brain)
                } else {
                    import::import_openwolf(&path, &brain)
                };
                let report = match outcome {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("cfetch import openwolf: {e:#}");
                        std::process::exit(1);
                    }
                };
                println!();
                if !report.imported.is_empty() {
                    println!("imported:");
                    for (src, dest) in &report.imported {
                        println!("  {} -> {}", src, dest);
                    }
                }
                if !report.skipped.is_empty() {
                    println!("skipped:");
                    for (name, reason) in &report.skipped {
                        println!("  {} ({})", name, reason);
                    }
                }
                if !report.unrecognized.is_empty() {
                    println!("found, not recognized — left in place:");
                    for name in &report.unrecognized {
                        println!("  {}", name);
                    }
                }
                if !report.errors.is_empty() {
                    println!("errors:");
                    for (name, error) in &report.errors {
                        println!("  {}: {}", name, error);
                    }
                }
                if let Some(note) = &report.resident_note {
                    println!();
                    println!("{note}");
                }
                if report.imported.is_empty()
                    && report.skipped.is_empty()
                    && report.unrecognized.is_empty()
                {
                    println!("nothing to import from {}", path.display());
                } else {
                    println!();
                    println!("run `cfetch scan` to index the imported content");
                }
            }
        },
        Command::Hook { event, agent } => {
            // Unreachable in practice (pre-dispatch above), kept for --help.
            hooks::run(&event, agent.as_deref());
        }
        #[cfg(feature = "embedded-embeddings")]
        Command::EmbedModel { model, action } => {
            println!("FastEmbed embeddings: diagnostic only; output is never stored or shared");
            match action {
                EmbedModelAction::Download => {
                    println!("model: {model:?}");
                    match embedded_embed::EmbeddedEmbedder::load(model) {
                        Ok(_) => println!("selected diagnostic model downloaded and loaded"),
                        Err(e) => {
                            eprintln!("cfetch embed-model download: {e:#}");
                            std::process::exit(1);
                        }
                    }
                }
                EmbedModelAction::Status => {
                    let cache = embedded_embed::cache_dir();
                    let info = fastembed::TextEmbedding::get_model_info(&model)
                        .expect("the CLI accepts only listed diagnostic models");
                    println!("model: {model:?} ({} dimensions)", info.dim);
                    println!("repository: {}", info.model_code);
                    println!("cache dir: {}", cache.display());
                    let repository =
                        cache.join(format!("models--{}", info.model_code.replace('/', "--")));
                    if repository.is_dir() {
                        println!("cache entry: present (file completeness unverified)");
                    } else {
                        println!("cache entry: not cached");
                    }
                    println!(
                        "loadability: not checked; use `download` or `test` to load explicitly"
                    );
                }
                EmbedModelAction::List => {
                    println!("Embedding models (select with --model):");
                    for info in embedded_embed::available_models() {
                        println!(
                            "  {:<32} {:>4}d  {}",
                            format!("{:?}", info.model),
                            info.dim,
                            info.model_code
                        );
                    }
                    println!();
                    println!("Reranker models:");
                    for (name, desc) in embedded_embed::available_rerankers() {
                        println!("  {:<28} {}", name, desc);
                    }
                }
                EmbedModelAction::Test { text } => {
                    println!("model: {model:?}");
                    let mut embedder = match embedded_embed::EmbeddedEmbedder::load(model) {
                        Ok(e) => e,
                        Err(e) => {
                            eprintln!("cfetch embed-model test: {e:#}");
                            std::process::exit(1);
                        }
                    };
                    let start = std::time::Instant::now();
                    match embedder.embed(vec![text.clone()]) {
                        Ok(embeddings) => {
                            let elapsed = start.elapsed();
                            if let Some(vec) = embeddings.first() {
                                let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
                                println!(
                                    "dims: {}, norm: {:.4}, time: {:?}",
                                    vec.len(),
                                    norm,
                                    elapsed
                                );
                                println!("first 5: {:?}", &vec[..5.min(vec.len())]);
                                println!("last 5:  {:?}", &vec[vec.len().saturating_sub(5)..]);
                            } else {
                                println!("no embedding returned");
                            }
                        }
                        Err(e) => {
                            eprintln!("embed failed: {e:#}");
                            std::process::exit(1);
                        }
                    }
                }
            }
        }
        Command::Repos { roots, sync, json } => {
            let result = (|| -> anyhow::Result<bool> {
                let cfg = config::Config::load()?;
                let roots = if roots.is_empty() {
                    cfg.git.roots.iter().map(|path| cfg.resolve(path)).collect()
                } else {
                    roots
                };
                anyhow::ensure!(
                    !roots.is_empty(),
                    "provide repository roots or configure git.roots"
                );
                let statuses = repositories::run(
                    &roots,
                    sync,
                    std::time::Duration::from_secs(cfg.git.timeout_secs),
                )?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&statuses)?);
                } else {
                    for status in &statuses {
                        println!(
                            "{}: {}{}",
                            status.path.display(),
                            status.state,
                            status
                                .detail
                                .as_ref()
                                .map(|text| format!(" ({text})"))
                                .unwrap_or_default()
                        );
                    }
                }
                Ok(!sync || statuses.iter().all(|status| status.state == "synchronized"))
            })();
            match result {
                Ok(true) => {}
                Ok(false) => std::process::exit(1),
                Err(error) => {
                    eprintln!("cfetch repos: {error:#}");
                    std::process::exit(1);
                }
            }
        }
        Command::Daemon { action } => {
            let result = match action {
                DaemonAction::Run => daemon::run(),
                DaemonAction::Start => daemon::start(),
                DaemonAction::Stop => daemon::stop(),
                DaemonAction::Status => daemon::status(),
            };
            if let Err(e) = result {
                eprintln!("cfetch daemon: {e}");
                std::process::exit(1);
            }
        }
        Command::Install {
            settings,
            agents,
            all,
            project,
            remove,
            replace_status_line,
        } => {
            if let Err(e) = install::configure(
                settings.as_deref(),
                &agents,
                all,
                remove,
                project.as_deref(),
                replace_status_line,
            ) {
                eprintln!("cfetch install: {e}");
                std::process::exit(1);
            }
        }
        Command::Hardware { json } => {
            let found = hardware::detect();
            let release = variant::recommended_release();
            let local_plan = match local_inference::selected_local_package_plan() {
                Ok(plan) => plan,
                Err(error) => {
                    eprintln!("cfetch hardware: invalid embedded local inference plan: {error:#}");
                    std::process::exit(1);
                }
            };
            let local_inference = local_plan.is_some();
            let packaged_backend = if local_inference {
                "package-local"
            } else {
                "endpoint"
            };
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "build_variant": variant::build_id(),
                        "recommended_release_variant": release,
                        "os": variant::os_token(),
                        "arch": variant::arch_token(),
                        "x86_64_level": hardware::x86_64_level(),
                        "local_inference": local_inference,
                        "backend": packaged_backend,
                        "ordered_scope_ids": local_plan.as_ref().map(|plan| &plan.ordered_scope_ids),
                        "devices": found.iter().map(|f| serde_json::json!({
                            "device": f.device.describe(),
                            "token": f.device.token(),
                            "class": format!("{:?}", f.device.class()).to_lowercase(),
                            "evidence": f.evidence,
                            "evidence_level": "discovery",
                            "caveat": f.caveat(),
                        })).collect::<Vec<_>>(),
                    })
                );
            } else {
                println!("detected, best first (policy: NPU > GPU > CPU):");
                for f in &found {
                    println!("  {:<26} {}", f.device.describe(), f.evidence);
                    if let Some(note) = f.caveat() {
                        println!("    note: {note}");
                    }
                }
                println!("  discovery is not execution or backend admission");
                println!(
                    "\nlocal inference:     {}",
                    if local_inference {
                        "included"
                    } else {
                        "not included"
                    }
                );
                println!("embeddings backend:   {packaged_backend}");
                println!(
                    "build variant:        {}",
                    variant::build_id().unwrap_or("unidentified source build")
                );
                match release {
                    Some(v) => println!("available release:    {}", v.id),
                    None => println!(
                        "available release:    none for {} / {}",
                        variant::os_token(),
                        variant::arch_token()
                    ),
                }
            }
        }
        Command::Variants { json } => {
            let catalog = variant::catalog();
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "schema_version": catalog.schema_version,
                        "build_variant": variant::build_id(),
                        "variants": catalog.variants,
                    })
                );
            } else {
                println!("shipping release variants:");
                for v in &catalog.variants {
                    println!("  {:<36} {:<8} {:<8} {}", v.id, v.os, v.arch, v.backend);
                }
                println!(
                    "build variant: {}",
                    variant::build_id().unwrap_or("unidentified source build")
                );
            }
        }
        Command::EmbeddingProfile { json } => {
            let profile = embedding_profile::manifest();
            let policy = embedding_profile::admission_policy();
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&embedding_profile::manifest_document())
                        .expect("profile serializes")
                );
            } else {
                println!(
                    "{}: network major {}, {} @ {}, {} dims / {} bytes",
                    profile.profile_id,
                    profile.network_major,
                    profile.model,
                    profile.model_revision,
                    profile.dimensions,
                    profile.vector_bytes,
                );
                println!("profile status: {}", embedding_profile::PROFILE_STATUS);
                match embedding_profile::production_availability() {
                    Ok(()) => println!("production availability: active"),
                    Err(error) => println!("production availability: unavailable ({error})"),
                }
                println!("query prefix: {:?}", profile.query_prefix);
                println!("document prefix: {:?}", profile.document_prefix);
                println!(
                    "pooling: {}; artifact policy: {}",
                    profile.pooling, policy.artifact_policy
                );
                println!(
                    "backend internal precision: {}",
                    policy.backend_internal_precision
                );
                println!("normalization: {}", profile.normalization);
                println!(
                    "profile manifest sha256: {}",
                    embedding_profile::manifest_sha256()
                );
                println!(
                    "admission policy sha256: {}",
                    embedding_profile::admission_policy_sha256()
                );
                println!(
                    "execution: {} (numerical anchor: {}, exact cross-backend bytes: {})",
                    policy.execution_policy,
                    policy.numerical_anchor.unwrap_or("none"),
                    policy.cross_backend_exact_bytes,
                );
            }
        }
        Command::RetrievalFixture {
            json,
            show_vectors,
            require,
        } => {
            if let Err(error) = retrieval_fixture::run(json, show_vectors, &require) {
                eprintln!("cfetch retrieval-fixture: {error:#}");
                std::process::exit(1);
            }
        }
        Command::RetrievalEval {
            corpus,
            representation,
            export,
            vectors,
        } => {
            if let Err(error) = memory_eval::run(
                &corpus,
                representation,
                export.as_deref(),
                vectors.as_deref(),
            ) {
                eprintln!("cfetch retrieval-eval: {error:#}");
                std::process::exit(1);
            }
        }
        Command::Slices { json } => {
            if let Err(e) = slices(json) {
                eprintln!("cfetch slices: {e:#}");
                std::process::exit(1);
            }
        }
        Command::Memories { action } => {
            if let Err(e) = staging_cmd(action) {
                eprintln!("cfetch memories: {e}");
                std::process::exit(1);
            }
        }
        Command::Maintain { action } => {
            if let Err(e) = maintain_cmd(action).and_then(|()| {
                let _ = runtime_status::refresh_static()?;
                Ok(())
            }) {
                eprintln!("cfetch maintain: {e:#}");
                std::process::exit(1);
            }
        }
        Command::Bench { since_days, json } => {
            if let Err(e) = bench_cmd(since_days, json) {
                eprintln!("cfetch bench: {e}");
                std::process::exit(1);
            }
        }
        Command::Init { path } => {
            if let Err(e) = init_cmd(path) {
                eprintln!("cfetch init: {e}");
                std::process::exit(1);
            }
        }
        Command::Cards { action } => {
            if let Err(e) = cards::run(action) {
                eprintln!("cfetch cards: {e:#}");
                std::process::exit(1);
            }
        }
        Command::Dashboard => {
            // No none-tier guard: the dashboard routes to the serving host
            // like the rest of the read path.
            if let Err(e) = dashboard::run() {
                eprintln!("cfetch dashboard: {e}");
                std::process::exit(1);
            }
        }
        Command::Doctor {
            json,
            tui,
            no_network,
            deep,
            show_vectors,
            require,
        } => {
            let result = if tui {
                dashboard::run_system(!no_network)
            } else {
                let report = if deep {
                    doctor::gather_deep(!no_network, show_vectors)
                } else {
                    doctor::gather(!no_network)
                };
                let rendered = if json {
                    serde_json::to_string_pretty(&report)
                        .map(|body| println!("{body}"))
                        .map_err(Into::into)
                } else {
                    println!("{}", doctor::render_text(&report));
                    Ok(())
                };
                rendered.and_then(|()| {
                    if require.is_empty() {
                        return Ok(());
                    }
                    let probe = report.retrieval_probe.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "required retrieval gates could not run: {}",
                            report
                                .retrieval_probe_error
                                .as_deref()
                                .unwrap_or("deep retrieval report is unavailable")
                        )
                    })?;
                    retrieval_fixture::enforce_requirements(probe, &require)
                })
            };
            if let Err(e) = result {
                eprintln!("cfetch doctor: {e:#}");
                std::process::exit(1);
            }
        }
        Command::Mcp => {
            if let Err(e) = mcp::serve() {
                eprintln!("cfetch mcp: {e}");
                std::process::exit(1);
            }
        }
        Command::Scan { background } => {
            if let Err(e) = scan(background).and_then(|()| {
                let _ = runtime_status::refresh_static()?;
                Ok(())
            }) {
                eprintln!("cfetch scan: {e}");
                std::process::exit(1);
            }
        }
        Command::Find {
            query,
            limit,
            budget_tokens,
            json,
        } => {
            if let Err(e) = find(&query, limit, budget_tokens, json) {
                eprintln!("cfetch find: {e}");
                std::process::exit(1);
            }
        }
        Command::Map {
            focus,
            budget_tokens,
        } => {
            if let Err(e) = map_cmd(focus.as_deref(), budget_tokens) {
                eprintln!("cfetch map: {e}");
                std::process::exit(1);
            }
        }
        Command::CodeGraph { action } => {
            if let Err(e) = code_graph_cmd(action) {
                eprintln!("cfetch code-graph: {e}");
                std::process::exit(1);
            }
        }
        Command::GraphPath {
            from,
            to,
            depth,
            json,
        } => {
            let result = (|| -> anyhow::Result<()> {
                let cfg = config::Config::load()?;
                let conn =
                    index::ensure_fresh(&paths::state_dir(), &cfg.brain_root, None, &cfg.rings())?;
                let route = knowledge_graph::trace(&conn, &from, &to, depth)?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&route)?);
                } else if route.found {
                    println!("{}", route.paths.join(" ↔ "));
                } else if route.from_matches.len() != 1 || route.to_matches.len() != 1 {
                    println!(
                        "endpoints must each identify one note; from: {:?}; to: {:?}",
                        route.from_matches, route.to_matches
                    );
                } else {
                    println!("no curated path within {depth} hops");
                }
                Ok(())
            })();
            if let Err(error) = result {
                eprintln!("cfetch graph-path: {error:#}");
                std::process::exit(1);
            }
        }
        Command::Graph {
            focus,
            slice,
            limit,
            json,
        } => {
            if let Err(e) = graph_cmd(focus.as_deref(), slice.as_deref(), limit, json) {
                eprintln!("cfetch graph: {e}");
                std::process::exit(1);
            }
        }
        Command::Recall {
            query,
            id,
            expand,
            fresh,
            semantic,
            hybrid,
            slice,
            limit,
            budget_tokens,
            json,
        } => {
            if let Err(e) = recall(
                &query.join(" "),
                id.as_deref(),
                expand,
                fresh,
                semantic,
                hybrid,
                slice.as_deref(),
                limit,
                budget_tokens,
                json,
            ) {
                eprintln!("cfetch recall: {e}");
                std::process::exit(1);
            }
        }
        Command::Failures { query, limit, json } => {
            if let Err(e) = failures_cmd(&query.join(" "), limit, json) {
                eprintln!("cfetch failures: {e}");
                std::process::exit(1);
            }
        }
        Command::EmbedIndex { batch } => {
            if let Err(e) = embed::embed_index_cmd(batch).and_then(|()| {
                let _ = runtime_status::refresh_static()?;
                Ok(())
            }) {
                eprintln!("cfetch embed-index: {e}");
                std::process::exit(1);
            }
        }
        Command::Selfcheck => {
            if let Err(e) = selfcheck() {
                eprintln!("cfetch selfcheck: {e}");
                std::process::exit(1);
            }
        }
        Command::Audit { json } => {
            if let Err(e) = audit_cmd(json) {
                eprintln!("cfetch audit: {e}");
                std::process::exit(1);
            }
        }
        Command::Status { json, line } => {
            let result = if line {
                println!(
                    "{}",
                    runtime_status::render_line(&runtime_status::load_cached())
                );
                Ok(())
            } else if json {
                runtime_status::refresh_static().and_then(|snapshot| {
                    println!("{}", serde_json::to_string_pretty(&snapshot)?);
                    Ok(())
                })
            } else {
                status()
            };
            if let Err(e) = result {
                eprintln!("cfetch status: {e}");
                std::process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_delivery_records_are_a_named_measurement_gap() {
        let gap = delivery_status_line(agent_session::AGENT_CODEX, None);
        assert!(gap.contains("not measurable"), "{gap}");
        assert!(gap.contains("Codex"), "{gap}");
        assert!(
            !gap.contains("format drift"),
            "absence alone does not prove schema drift: {gap}"
        );

        let observed = delivery_status_line(agent_session::AGENT_CLAUDE, Some((6, 4)));
        assert!(observed.contains("6 hook firing(s) observed"), "{observed}");
        assert!(observed.contains("4 injection(s) verified"), "{observed}");
        assert!(observed.contains("Claude"), "{observed}");
    }

    #[test]
    fn an_unbuilt_catalog_reports_absence_and_a_lagging_one_reports_its_age() {
        // The status line used to be a boolean at best: an index nobody had
        // built and one describing the tree exactly both printed counts, and
        // a stale one never said for how long.
        let dir = tempfile::tempdir().unwrap();
        let brain = dir.path().join("brain");
        std::fs::create_dir_all(&brain).unwrap();
        std::fs::write(brain.join("note.md"), "# note\n\nbody\n").unwrap();
        let state = dir.path().join("state");
        let cfg = config::Config {
            brain_root: brain.clone(),
            ..config::Config::default()
        };

        let (severity, line) = index_liveness_line(&cfg, &state, None);
        assert_eq!(severity, heartbeat::Severity::Unobserved);
        assert!(line.contains("NEVER SCANNED"), "{line}");
        assert!(
            !line.contains("generation"),
            "an unbuilt catalog has no generation to quote: {line}"
        );

        {
            let mut conn = index::open(&state).unwrap();
            index::scan(&mut conn, &brain, None, &cfg.rings()).unwrap();
        }
        let (severity, line) = index_liveness_line(&cfg, &state, None);
        assert_eq!(severity, heartbeat::Severity::Healthy);
        assert!(line.contains("index: current"), "{line}");
        assert!(line.contains("generation 1"), "{line}");

        // Change the tree and the same call must report the lag, with an age.
        std::fs::write(brain.join("second.md"), "# second\n\nmore\n").unwrap();
        let (_, line) = index_liveness_line(&cfg, &state, None);
        assert!(line.contains("stale for"), "{line}");
        // And the onset is now on record, which is the only way the age can
        // later grow past the warning threshold.
        assert!(
            heartbeat::load_from(&state)
                .index
                .unwrap()
                .stale_since
                .is_some()
        );
    }

    use super::answer::{self, BlockIn};
    use crate::hook_io::estimate_tokens;

    /// Splits a rendered listing into (results, truncation note). The note is
    /// deliberately charged outside the budget, so the assertion has to price
    /// the two halves separately.
    fn split_note(out: &str) -> (&str, &str) {
        out.rsplit_once('\n')
            .expect("a truncated listing ends with its note")
    }

    #[test]
    fn a_listing_is_capped_in_tokens_not_in_result_count() {
        // The defect, stated as data: the same `--limit 8` buys these two
        // answers, and one of them costs an order of magnitude more.
        let wide: Vec<String> = (0..8)
            .map(|i| {
                answer::find_entry(
                    &format!("{}/file{i}.rs", "/deeply/nested/workspace/crate".repeat(4)),
                    Some(&format!("resolve_cascade_tensor_variant_{i}")),
                    Some("function_item"),
                    1,
                    420,
                    5040,
                )
            })
            .collect();
        let narrow: Vec<String> = (0..8)
            .map(|i| {
                answer::find_entry(
                    &format!("a{i}.rs"),
                    Some("f"),
                    Some("function_item"),
                    1,
                    2,
                    24,
                )
            })
            .collect();
        let ratio = wide.join("\n").len() as f64 / narrow.join("\n").len() as f64;
        assert!(
            ratio > 4.0,
            "same count, {ratio:.1}x the cost — that is what a budget fixes"
        );

        let capped = answer::listing(wide.clone(), 200, answer::CLI_RECOVERY);
        let (results, note) = split_note(&capped);
        assert!(
            estimate_tokens(results.len()) <= 200,
            "results over budget: {} tok",
            estimate_tokens(results.len())
        );
        assert!(
            note.contains("dropped by the 200-token answer budget"),
            "{note}"
        );
        assert!(
            note.contains("--budget-tokens 0"),
            "the note must name the way back: {note}"
        );
        // Kept hits are the ranked prefix, in order, unmodified.
        assert!(capped.starts_with(&wide[0]), "the best hit must survive");
        assert!(!capped.contains(&wide[7]), "the tail is what gets dropped");
        // The cheap answer is not touched by the same budget.
        assert_eq!(
            answer::listing(narrow.clone(), 200, answer::CLI_RECOVERY),
            narrow.join("\n")
        );
    }

    #[test]
    fn one_entry_always_survives_and_zero_lifts_the_cap() {
        let one = vec!["x".repeat(4000)];
        let starved = answer::listing(one.clone(), 1, answer::CLI_RECOVERY);
        assert!(
            starved.starts_with(&one[0]),
            "an answer holding nothing answers nothing"
        );

        let many: Vec<String> = (0..40)
            .map(|i| format!("entry {i} {}", "y".repeat(200)))
            .collect();
        assert_eq!(
            answer::listing(many.clone(), 0, answer::CLI_RECOVERY),
            many.join("\n"),
            "--budget-tokens 0 is the documented escape hatch and must drop nothing"
        );
    }

    #[test]
    fn expanded_blocks_share_one_budget_and_name_what_they_cut() {
        // Mirrored copies of one statement arrive together. Before the budget
        // each copy was printed whole, so `recall --id` on a mirrored heading
        // chain cost N times a block nobody sized.
        let body = "a line of block text that keeps going and going\n".repeat(300);
        let blocks: Vec<BlockIn> = (0..3)
            .map(|i| BlockIn {
                cite: format!("r2-abcdef{i}"),
                path: format!("knowledge/big{i}.md"),
                ring: 2,
                start_line: 1,
                end_line: 300,
                text: body.clone(),
            })
            .collect();

        let whole = answer::blocks(&blocks, 0);
        assert!(
            estimate_tokens(whole.len()) > 3000,
            "uncapped, this is the bill"
        );

        let capped = answer::blocks(&blocks, 300);
        // 300 for the bodies plus one truncation note per block; nowhere near
        // the uncapped cost.
        assert!(
            estimate_tokens(capped.len()) < 600,
            "over budget: {} tok",
            estimate_tokens(capped.len())
        );
        for i in 0..3 {
            // Nothing vanishes: an unaffordable block is still named, with the
            // exact range that holds its text.
            assert!(
                capped.contains(&format!("r2-abcdef{i}")),
                "block {i} lost its citation"
            );
            assert!(
                capped.contains(&format!("knowledge/big{i}.md:1-300")),
                "block {i} lost its range"
            );
        }
        assert_eq!(capped.matches("not shown (answer budget)").count(), 3);
        // The first block gets the allowance, the last gets none of it.
        let parts: Vec<&str> = capped.split("\n\n").collect();
        assert!(
            parts[0].len() > parts[2].len() * 4,
            "the budget is spent in rank order"
        );
    }

    #[test]
    fn clipping_cuts_on_a_line_boundary_and_never_mid_character() {
        let text = "ærste linje\nzweite Zeile mit Umlauten äöü\nthird line\n";
        let clipped = answer::blocks(
            &[BlockIn {
                cite: "r3-deadbeef".into(),
                path: "notes/x.md".into(),
                ring: 3,
                start_line: 4,
                end_line: 6,
                text: text.into(),
            }],
            4,
        );
        let body = clipped
            .strip_prefix("r3-deadbeef notes/x.md:4-6 (ring 3)\n")
            .unwrap();
        let (shown, note) = body.split_once("\n… ").expect("a clipped block says so");
        assert!(
            text.starts_with(shown),
            "the clip must be a prefix of the block"
        );
        assert!(
            !shown.contains('\n'),
            "cut at the first line boundary that fits: {shown:?}"
        );
        assert!(note.contains("read notes/x.md:4-6 for the rest"), "{note}");

        // A block that fits arrives whole, with no note at all.
        let short = answer::blocks(
            &[BlockIn {
                cite: "r3-deadbeef".into(),
                path: "notes/x.md".into(),
                ring: 3,
                start_line: 4,
                end_line: 6,
                text: text.into(),
            }],
            9999,
        );
        assert!(short.ends_with(text));
        assert!(!short.contains("answer budget"));
    }

    #[test]
    fn json_answers_are_priced_on_the_json_they_emit() {
        let entries: Vec<serde_json::Value> = (0..30)
            .map(|i| serde_json::json!({"cite": format!("r2-{i:08x}"), "snippet": "s".repeat(300)}))
            .collect();
        let (kept, dropped) = answer::fit_json(entries.clone(), 400);
        assert_eq!(
            kept.len() + dropped,
            30,
            "every entry is either kept or counted"
        );
        assert!(
            dropped > 0,
            "300-char snippets cannot all fit in 400 tokens"
        );
        let cost: usize = kept.iter().map(|v| v.to_string().len() + 1).sum();
        assert!(
            estimate_tokens(cost) <= 400,
            "priced on serialized bytes, not on prose"
        );
        assert_eq!(
            answer::fit_json(entries, 0).1,
            0,
            "zero lifts the cap here too"
        );
    }
}
