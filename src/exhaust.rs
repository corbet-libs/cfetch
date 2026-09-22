//! Ring-6 exhaust: raw capture of what the agent actually did, the 6->5
//! flagging traps, the hand-off into ring-5 staging, and the failure query
//! that ASKS the stream what the traps only ever decided.
//!
//! Exhaust is DATA OF RECORD, so it lives in the tree — one append-only JSONL
//! stream per host at `<brain_root>/logs/cfetch/exhaust-<host>.jsonl` (see
//! [`crate::jsonl`] for the format, its version envelope and its byte cap).
//! It used to be a SQLite database in the per-host state dir, which made a
//! candidate flagged on one machine invisible to maintenance running on
//! another; the tree is the only storage of record, and this is part of it.
//! No database survives here: the traps stream the stream.
//!
//! Secrets are redacted at FIRST capture (commands) or withheld entirely
//! (secret-shaped paths) so every downstream consumer inherits the guard.
//!
//! The 6->5 crossing runs from the Stop hook over a BOUNDED window of the
//! local host's stream — the traps are heuristics over recent behavior, and a
//! hook whose cost grows with history is a hook that eventually stalls:
//! - fix-discovered: a command failed, then the same normalized command later
//!   succeeded in the same session — the candidate carries both.
//! - recurring-failure: the same normalized command failed in >=2 sessions.
//! - hot-file: a brain file resolving to rings 0-3 written in >=2 distinct
//!   sessions, or >=10 times in one session. Code files and ring-4 working
//!   files are churn by contract and never fire.
//!
//! Flagging is idempotent WITHOUT any writer-side bookkeeping: a candidate's
//! id is a hash of the trap's key, so the existence of `<id>.md` under staging
//! (pending or dismissed) is the whole "already flagged" test — and it holds
//! across hosts, because the staging directory is shared. A candidate consumed
//! on THIS host is remembered through its `consume` record in the stream; one
//! consumed on another host may legitimately stage again here, which is a
//! second look at a live pattern rather than a lost one.
//!
//! Bash payloads store the buglog-style normalized command (`norm`) at capture
//! time, and write payloads store the file's resolved ring, so the traps stay
//! cheap field lookups rather than re-derivations.
//!
//! That same signature also answers questions: [`Exhaust::failure_history`]
//! aggregates every host's failing bash events by `norm`, so "has this failed
//! here before, and did it ever recover" has an answer. It SCANS rather than
//! indexes. A derived index would be cold exactly when it is asked — the
//! stream grows on every turn — and a second copy of ring-6 data on one
//! machine is what moving exhaust into the tree got rid of. The scan is a
//! CLI-only cost, bounded by the writer's byte cap; no hook may pay it.

use std::path::{Path, PathBuf};

use crate::config::{Config, RingRules};
use crate::hook_io::HookEvent;
use crate::jsonl::{self, Record};
use crate::staging::{self, Candidate};

/// Stream name: files are `exhaust-<host>.jsonl`.
pub const STREAM: &str = "exhaust";

/// How much of the local stream's tail the Stop-hook traps read: ~7k recent
/// events at typical line sizes. Bounded on purpose — this is the one read on
/// the hook path, it happens once per TURN, and the tree it reads from may be
/// a network mount. Traps are heuristics over recent behavior; a pattern
/// older than this window is history, not a live signal.
pub const TRAP_WINDOW_BYTES: u64 = 1024 * 1024;

/// Upper bound on PENDING ring-5 candidates. Reaching it stages one explicit
/// `staging-full` candidate and stops adding more: unlike the old row cap,
/// no pending evidence is ever deleted merely to make room.
pub const MAX_STAGED: usize = 2_000;

/// Captured commands are clamped so one line stays small enough to be written
/// (and appended) as a single unit.
const MAX_COMMAND_CHARS: usize = 2_000;

const REDACTED: &str = "<redacted>";
/// Stored instead of a secret-shaped file path — even the path leaks.
pub const WITHHELD: &str = "<secret path withheld>";

/// The ring-6/5 surfaces bound to one brain tree and one host identity.
#[derive(Debug, Clone)]
pub struct Exhaust {
    /// Where the JSONL streams live (`<brain_root>/logs/cfetch`).
    pub logs_dir: PathBuf,
    /// Where ring-5 candidates live (`<brain_root>/scratch/cfetch-staging`).
    pub staging_dir: PathBuf,
    /// This host's identity, stamped into file names and candidates.
    pub host: String,
    /// Writer-side byte cap for the exhaust stream.
    pub max_bytes: u64,
}

impl Exhaust {
    pub fn new(logs_dir: PathBuf, staging_dir: PathBuf, host: String, max_bytes: u64) -> Exhaust {
        Exhaust {
            logs_dir,
            staging_dir,
            host,
            max_bytes,
        }
    }

    pub fn from_config(cfg: &Config) -> Exhaust {
        Exhaust::new(
            crate::paths::logs_dir(&cfg.brain_root),
            crate::paths::staging_dir(&cfg.brain_root),
            crate::paths::host_id(),
            cfg.exhaust_max_bytes,
        )
    }

    /// Appends one ring-6 event. This is the whole hook write path: one
    /// `O_APPEND` line, no fsync, no read, no scan.
    pub fn record(
        &self,
        session: &str,
        kind: &str,
        payload: &serde_json::Value,
    ) -> anyhow::Result<()> {
        jsonl::append(
            &self.logs_dir,
            STREAM,
            &self.host,
            self.max_bytes,
            serde_json::json!({"kind": kind, "session": session, "payload": payload}),
        )
    }

    /// [`Exhaust::record`] with an explicit timestamp. Only the legacy import
    /// uses it: history carried into the tree must keep the moment it
    /// happened, not the moment it was moved.
    pub fn record_at(
        &self,
        ts: i64,
        session: &str,
        kind: &str,
        payload: &serde_json::Value,
    ) -> anyhow::Result<()> {
        jsonl::append(
            &self.logs_dir,
            STREAM,
            &self.host,
            self.max_bytes,
            serde_json::json!({"ts": ts, "kind": kind, "session": session, "payload": payload}),
        )
    }

    /// PostToolUse capture: Bash -> 'bash' (redacted command + norm + error
    /// hint), Write|Edit|MultiEdit|apply_patch -> 'write' (path + resolved ring
    /// for brain files), Read -> 'read'. Anything else, or missing fields,
    /// records nothing. `brain_root` locates the brain for ring resolution,
    /// `rules` the configured taxonomy that resolves it.
    pub fn capture_post_tool(
        &self,
        event: &HookEvent,
        brain_root: &Path,
        rules: &RingRules,
    ) -> anyhow::Result<()> {
        let session = event.session();
        let str_field = |key: &str| {
            event
                .tool_input
                .as_ref()
                .and_then(|i| i.get(key))
                .and_then(serde_json::Value::as_str)
        };
        match event.tool_name.as_deref() {
            Some("Bash") => {
                let Some(cmd) = str_field("command") else {
                    return Ok(());
                };
                let redacted = clamp(&redact_secrets(cmd), MAX_COMMAND_CHARS);
                let norm = clamp(&normalize_command(&redacted), MAX_COMMAND_CHARS);
                self.record(
                    session,
                    "bash",
                    &serde_json::json!({
                        "command": redacted, "norm": norm, "failed": tool_failed(event),
                    }),
                )
            }
            Some("Write" | "Edit" | "MultiEdit" | "apply_patch") => {
                for path in written_paths(event) {
                    self.record(session, "write", &write_payload(&path, brain_root, rules))?;
                }
                Ok(())
            }
            Some("Read") => {
                let Some(path) = str_field("file_path") else {
                    return Ok(());
                };
                self.record(session, "read", &file_payload(path))
            }
            _ => Ok(()),
        }
    }

    /// Stop: writes one 'turn' summary event (counts by kind this session),
    /// then runs the 6->5 traps over the bounded window and stages what fires.
    pub fn record_stop(&self, session: &str) -> anyhow::Result<()> {
        let window = jsonl::read_tail(&self.logs_dir, STREAM, &self.host, TRAP_WINDOW_BYTES);

        let mut counts = serde_json::Map::new();
        for r in &window.records {
            if r.str("session") != session {
                continue;
            }
            if matches!(r.kind(), "bash" | "write" | "read") {
                let n = counts
                    .entry(r.kind().to_string())
                    .or_insert(serde_json::Value::from(0));
                *n = serde_json::Value::from(n.as_i64().unwrap_or(0) + 1);
            }
        }
        self.record(session, "turn", &serde_json::Value::Object(counts))?;

        let decided = consumed_ids(&window.records);
        let mut pending: Option<usize> = None;
        for candidate in traps(session, &self.host, &window.records) {
            if decided.contains(&candidate.id) || staging::exists(&self.staging_dir, &candidate.id)
            {
                continue;
            }
            let count = match pending {
                Some(n) => n,
                None => *pending.insert(staging::pending_count(&self.staging_dir)),
            };
            if count >= MAX_STAGED {
                self.stage_full_notice(session)?;
                break;
            }
            if staging::write(&self.staging_dir, &candidate)? {
                pending = Some(count + 1);
                self.record(
                    session,
                    "stage",
                    &serde_json::json!({"id": candidate.id, "reason": candidate.reason}),
                )?;
            }
        }
        Ok(())
    }

    /// Records a staging decision so the traps do not re-stage what this host
    /// has already dealt with. Dismissals need no record (the moved file is
    /// its own marker); consumption deletes the file, so this IS the marker.
    pub fn record_decision(&self, id: &str, decision: &str) -> anyhow::Result<()> {
        self.record("cli", decision, &serde_json::json!({"id": id}))
    }

    /// One explicit candidate saying the queue is full. Idempotent by id, so
    /// an overflowing store converges instead of churning out a warning per
    /// Stop.
    fn stage_full_notice(&self, session: &str) -> anyhow::Result<()> {
        let id = staging::id_for("staging-full", "");
        if staging::exists(&self.staging_dir, &id) {
            return Ok(());
        }
        let candidate = Candidate {
            id: id.clone(),
            reason: "staging-full".into(),
            session: session.to_string(),
            host: self.host.clone(),
            ts: now(),
            kind: "warning".into(),
            payload: serde_json::json!({
                "cap": MAX_STAGED,
                "note": "ring-5 staging is at its cap; no further candidates are staged until \
                         the queue is drained (cfetch staging list)",
            }),
        };
        if staging::write(&self.staging_dir, &candidate)? {
            self.record(
                session,
                "stage",
                &serde_json::json!({"id": id, "reason": "staging-full"}),
            )?;
        }
        Ok(())
    }

    /// Ring-5/6 counts for read-only reporting surfaces. FAIL SILENT by
    /// design: an absent tree yields zeros — a reporting pane must never
    /// create ring-6 state.
    pub fn stats(&self) -> ExhaustStats {
        let s = staging::stats(&self.staging_dir);
        ExhaustStats {
            staged_total: s.total as i64,
            staged_by_reason: s
                .by_reason
                .into_iter()
                .map(|(r, n)| (r, n as i64))
                .collect(),
            bytes: jsonl::footprint(&self.logs_dir, STREAM),
        }
    }

    /// Answers "has this failed here before" out of the streams themselves.
    ///
    /// The traps already key on [`normalize_command`], but only to DECIDE
    /// something; nothing could ASK. This aggregates the failing bash events
    /// of the whole shared tree — every host, every rotation — by signature,
    /// so the query surface exists without a second store of ring-6 data: the
    /// stream stays the only copy, exactly as the module header requires. The
    /// cost is one linear pass over files the writer already caps by bytes,
    /// and an explicit CLI query pays it — never a hook, which is why this
    /// takes no window bound while the traps do.
    ///
    /// An empty query ranks the whole failure history, most recurrent first.
    pub fn failure_history(&self, query: &str, limit: usize) -> FailureHistory {
        let scan = scan_failures(&self.logs_dir);
        // The query is REDACTED then normalized, exactly like capture: a
        // pasted failing command carrying a real secret (`deploy --token=x`)
        // used to normalize against a different alphabet than the stored
        // (redacted) signatures, so the failure was invisible to the exact
        // query surface that would look for it.
        let wanted = normalize_command(&redact_secrets(query));
        let terms: Vec<&str> = wanted.split_whitespace().collect();
        let mut matches: Vec<FailureSignature> = scan
            .by_norm
            .iter()
            .filter(|(norm, _)| terms.iter().all(|t| norm.contains(t)))
            .map(|(norm, agg)| FailureSignature {
                norm: norm.clone(),
                failures: agg.failures,
                sessions: agg.sessions.len(),
                recovered_sessions: agg.recovered_sessions,
                hosts: agg.hosts.iter().cloned().collect(),
                first_ts: agg.first_ts,
                last_ts: agg.last_ts,
                last_command: agg.last_command.clone(),
                staged: false,
            })
            .collect();
        matches.sort_by(|a, b| {
            // An exact signature hit answers the question that was asked;
            // everything else is context, ranked by how much of it there is
            // and how recent it is.
            (a.norm != wanted)
                .cmp(&(b.norm != wanted))
                .then(b.failures.cmp(&a.failures))
                .then(b.last_ts.cmp(&a.last_ts))
                .then(a.norm.cmp(&b.norm))
        });
        matches.truncate(limit);
        for m in &mut matches {
            // Only for what is returned: one stat per shown row, never one
            // per signature in the tree.
            m.staged = staging::exists(
                &self.staging_dir,
                &staging::id_for("recurring-failure", &m.norm),
            );
        }
        FailureHistory {
            matches,
            signatures: scan.by_norm.len(),
            failures: scan.failures,
            unreadable: scan.unreadable,
        }
    }
}

/// Paths changed by the two hook dialects cfetch supports. Claude supplies a
/// single `file_path`; Codex supplies its native apply_patch text in
/// `tool_input.command`. Codex patches are rooted at the hook event's cwd, so
/// relative paths are resolved before ring classification and capture.
pub(crate) fn written_paths(event: &HookEvent) -> Vec<String> {
    let Some(input) = event.tool_input.as_ref() else {
        return Vec::new();
    };
    match event.tool_name.as_deref() {
        Some("Write" | "Edit" | "MultiEdit") => input
            .get("file_path")
            .and_then(serde_json::Value::as_str)
            .map(|path| vec![path.to_string()])
            .unwrap_or_default(),
        Some("apply_patch") => {
            let Some(patch) = input.get("command").and_then(serde_json::Value::as_str) else {
                return Vec::new();
            };
            let mut paths = Vec::new();
            for line in patch.lines() {
                let path = [
                    "*** Add File: ",
                    "*** Update File: ",
                    "*** Delete File: ",
                    "*** Move to: ",
                ]
                .iter()
                .find_map(|prefix| line.strip_prefix(prefix));
                let Some(path) = path.filter(|path| !path.is_empty()) else {
                    continue;
                };
                let resolved = if Path::new(path).is_absolute() {
                    PathBuf::from(path)
                } else if let Some(cwd) = event.cwd.as_deref() {
                    Path::new(cwd).join(path)
                } else {
                    PathBuf::from(path)
                };
                let resolved = resolved.to_string_lossy().to_string();
                if !paths.contains(&resolved) {
                    paths.push(resolved);
                }
            }
            paths
        }
        _ => Vec::new(),
    }
}

/// Ring-5/6 figures for the dashboard.
#[derive(Default)]
pub struct ExhaustStats {
    /// Ring-5 candidates awaiting review, across every host.
    pub staged_total: i64,
    /// Per-reason breakdown of `staged_total`, in trap order.
    pub staged_by_reason: Vec<(String, i64)>,
    /// Bytes of exhaust stream on disk — the store's footprint, without
    /// reading a line of it.
    pub bytes: u64,
}

/// One normalized command signature with everything the streams know about
/// its failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailureSignature {
    /// The signature: [`normalize_command`] of the failing command lines.
    pub norm: String,
    /// Failing events carrying it.
    pub failures: usize,
    /// Distinct sessions that saw it fail.
    pub sessions: usize,
    /// Sessions where the same signature later SUCCEEDED. A signature that
    /// always recovers is a flake; one that never does is a standing blocker,
    /// and telling those apart is most of why the question gets asked.
    pub recovered_sessions: usize,
    /// Hosts whose exhaust holds one of the failures, sorted.
    pub hosts: Vec<String>,
    pub first_ts: i64,
    pub last_ts: i64,
    /// Redacted command text of the most recent failure. Signatures erase
    /// paths, digits and case, so the concrete line is worth carrying back.
    pub last_command: String,
    /// A ring-5 candidate for this signature is already queued or was
    /// dismissed — the recurring-failure trap saw the same pattern.
    pub staged: bool,
}

/// The answer to one failure query, plus the corpus it was drawn from.
#[derive(Debug, Default)]
pub struct FailureHistory {
    /// Matching signatures, best first, at most the requested limit.
    pub matches: Vec<FailureSignature>,
    /// Distinct failing signatures in the whole stream, matched or not — the
    /// denominator that tells "nothing matched" apart from "nothing captured".
    pub signatures: usize,
    /// Failing events behind them.
    pub failures: usize,
    /// Stream files that could not be read, `"<path>: <reason>"`. Surfaced,
    /// never swallowed: a pass that skipped half the tree must not read as a
    /// clean "this never happened".
    pub unreadable: Vec<String>,
}

/// Serialized value of the only record kind that carries a signature. Raw
/// lines are pre-filtered on it and only what passes is parsed, which keeps a
/// full-history pass bound by I/O instead of by JSON. The filter matches the
/// VALUE rather than `"kind":"bash"` so it survives any change in the
/// writer's key order — a pre-filter that silently stopped matching would
/// report "never happened before" over a stream full of failures, and a query
/// surface that lies that way is worse than none.
const BASH_VALUE: &str = "\"bash\"";

/// Per-signature accumulator of the scan.
struct Agg {
    failures: usize,
    sessions: std::collections::BTreeSet<String>,
    hosts: std::collections::BTreeSet<String>,
    recovered_sessions: usize,
    first_ts: i64,
    last_ts: i64,
    last_command: String,
}

impl Agg {
    fn new(ts: i64) -> Agg {
        Agg {
            failures: 0,
            sessions: std::collections::BTreeSet::new(),
            hosts: std::collections::BTreeSet::new(),
            recovered_sessions: 0,
            first_ts: ts,
            last_ts: ts,
            last_command: String::new(),
        }
    }
}

#[derive(Default)]
struct FailureScan {
    by_norm: std::collections::BTreeMap<String, Agg>,
    failures: usize,
    unreadable: Vec<String>,
}

/// Stream files OLDEST first: rotation `.2` predates `.1`, which predates the
/// live file. Recovery is a question about order ("it failed, then the same
/// signature succeeded"), and plain name order reads a host's generations
/// backwards, which would silently under-report it.
fn chronological_streams(dir: &Path) -> Vec<PathBuf> {
    let mut paths = jsonl::stream_paths(dir, STREAM);
    paths.sort_by_key(|p| {
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let stem = name.strip_suffix(".jsonl").unwrap_or(&name).to_string();
        // `<stem>-<host>.<n>.jsonl` is generation n; anything else is a live
        // file, which is the newest of its host by construction. A host id
        // containing dots parses as live, which is what it is.
        match stem
            .rsplit_once('.')
            .map(|(base, suffix)| (base.to_string(), suffix.parse::<u64>()))
        {
            Some((base, Ok(generation))) => (base, u64::MAX - generation),
            _ => (stem, u64::MAX),
        }
    });
    paths
}

/// One pass over the whole exhaust history, aggregating failures by
/// signature. Line by line and pre-filtered: the streams are byte-capped per
/// host, but nothing bounds how many hosts share a tree, so the pass must not
/// hold the history in memory to answer a question about a fraction of it.
fn scan_failures(logs_dir: &Path) -> FailureScan {
    use std::io::BufRead as _;

    let mut out = FailureScan::default();
    // Pairs, not signatures: a command failing in one session and succeeding
    // in another is not a recovery. Only sessions that FAILED are remembered,
    // so the scan's memory tracks failures, not command volume.
    let mut failed: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    let mut recovered: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();

    for path in chronological_streams(logs_dir) {
        let file = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                out.unreadable.push(format!("{}: {e}", path.display()));
                continue;
            }
        };
        for line in std::io::BufReader::new(file).lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    out.unreadable.push(format!("{}: {e}", path.display()));
                    break;
                }
            };
            if !line.contains(BASH_VALUE) {
                continue;
            }
            let Some(rec) = jsonl::decode_line(&line) else {
                continue;
            };
            if rec.kind() != "bash" {
                continue;
            }
            let Some(payload) = rec.value("payload").and_then(|v| v.as_object()) else {
                continue;
            };
            let norm = payload
                .get("norm")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            if norm.is_empty() {
                continue;
            }
            let session = rec.str("session").to_string();
            let failed_now = payload
                .get("failed")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            if !failed_now {
                let key = (session, norm.to_string());
                if failed.contains(&key)
                    && recovered.insert(key)
                    && let Some(agg) = out.by_norm.get_mut(norm)
                {
                    agg.recovered_sessions += 1;
                }
                continue;
            }
            let agg = out
                .by_norm
                .entry(norm.to_string())
                .or_insert_with(|| Agg::new(rec.ts));
            agg.failures += 1;
            out.failures += 1;
            agg.sessions.insert(session.clone());
            if !rec.host.is_empty() {
                agg.hosts.insert(rec.host.clone());
            }
            agg.first_ts = agg.first_ts.min(rec.ts);
            // Ties go to the later-scanned record: within a host that is the
            // later append, and the newest concrete command is the useful one.
            if rec.ts >= agg.last_ts {
                agg.last_ts = rec.ts;
                agg.last_command = payload
                    .get("command")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string();
            }
            failed.insert((session, norm.to_string()));
        }
    }
    out
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn clamp(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    s.chars().take(max_chars).collect::<String>() + "…"
}

/// Ids this host has already consumed — their staging files are gone, so the
/// stream is what remembers them.
fn consumed_ids(records: &[Record]) -> std::collections::HashSet<String> {
    records
        .iter()
        .filter(|r| r.kind() == "consume")
        .filter_map(|r| r.value("payload")?.get("id")?.as_str().map(str::to_string))
        .collect()
}

/// One captured bash event, flattened out of its record.
struct BashRow<'a> {
    session: &'a str,
    norm: &'a str,
    command: &'a str,
    failed: bool,
    ts: i64,
}

/// One captured write event with a resolved brain ring.
struct WriteRow<'a> {
    session: &'a str,
    path: &'a str,
    ring: i64,
    ts: i64,
}

fn bash_rows<'a>(records: &'a [Record]) -> Vec<BashRow<'a>> {
    records
        .iter()
        .filter(|r| r.kind() == "bash")
        .filter_map(|r| {
            let p = r.value("payload")?.as_object()?;
            Some(BashRow {
                session: r.str("session"),
                norm: p.get("norm")?.as_str()?,
                command: p
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default(),
                failed: p
                    .get("failed")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
                ts: r.ts,
            })
        })
        .collect()
}

fn write_rows<'a>(records: &'a [Record]) -> Vec<WriteRow<'a>> {
    records
        .iter()
        .filter(|r| r.kind() == "write")
        .filter_map(|r| {
            let p = r.value("payload")?.as_object()?;
            Some(WriteRow {
                session: r.str("session"),
                path: p.get("file_path")?.as_str()?,
                // Withheld secret paths carry NO ring, so they can never
                // aggregate into a fake hot file.
                ring: p.get("ring")?.as_i64()?,
                ts: r.ts,
            })
        })
        .collect()
}

/// The 6->5 crossing, in most-specific-first order. Pure over the window: the
/// caller decides what is new by asking the staging directory.
fn traps(session: &str, host: &str, records: &[Record]) -> Vec<Candidate> {
    let mut out = Vec::new();
    let bash = bash_rows(records);
    let writes = write_rows(records);
    fix_discovered(session, host, &bash, &mut out);
    recurring_failure(host, &bash, &mut out);
    hot_files(session, host, &writes, &mut out);
    out
}

/// (a) fix-discovered: a normalized command failed, then the same normalized
/// command succeeded LATER in the stopping session. One candidate per norm,
/// carrying both halves of the story.
fn fix_discovered(session: &str, host: &str, bash: &[BashRow<'_>], out: &mut Vec<Candidate>) {
    let mut seen: Vec<&str> = Vec::new();
    for (i, failure) in bash.iter().enumerate() {
        if failure.session != session || !failure.failed || seen.contains(&failure.norm) {
            continue;
        }
        let Some(fix) = bash[i + 1..]
            .iter()
            .find(|s| s.session == session && !s.failed && s.norm == failure.norm)
        else {
            continue;
        };
        seen.push(failure.norm);
        out.push(Candidate {
            id: staging::id_for("fix-discovered", &format!("{session}\u{0}{}", failure.norm)),
            reason: "fix-discovered".into(),
            session: session.to_string(),
            host: host.to_string(),
            ts: fix.ts,
            kind: "bash".into(),
            payload: serde_json::json!({
                "norm": failure.norm,
                "failed_command": failure.command,
                "fixed_command": fix.command,
            }),
        });
    }
}

/// (b) recurring-failure: the same normalized command failed in >= 2 distinct
/// sessions. Cross-session by definition, so it looks at the whole window.
fn recurring_failure(host: &str, bash: &[BashRow<'_>], out: &mut Vec<Candidate>) {
    let mut by_norm: std::collections::BTreeMap<&str, (Vec<&str>, &str, i64)> =
        std::collections::BTreeMap::new();
    for row in bash.iter().filter(|b| b.failed) {
        let entry = by_norm
            .entry(row.norm)
            .or_insert_with(|| (Vec::new(), row.command, row.ts));
        if !entry.0.contains(&row.session) {
            entry.0.push(row.session);
        }
        entry.2 = row.ts;
    }
    for (norm, (sessions, command, ts)) in by_norm {
        if sessions.len() < 2 {
            continue;
        }
        out.push(Candidate {
            id: staging::id_for("recurring-failure", norm),
            reason: "recurring-failure".into(),
            session: sessions.last().copied().unwrap_or_default().to_string(),
            host: host.to_string(),
            ts,
            kind: "bash".into(),
            payload: serde_json::json!({
                "norm": norm, "command": command, "sessions": sessions.len(),
            }),
        });
    }
}

/// (c) hot-file: a brain file resolving to rings 0-3 written in >= 2 distinct
/// sessions OR >= 10 times in the stopping session. Code files carry no ring
/// and ring-4 working files are churn by contract, so neither ever fires.
fn hot_files(session: &str, host: &str, writes: &[WriteRow<'_>], out: &mut Vec<Candidate>) {
    let mut by_path: std::collections::BTreeMap<&str, (Vec<&str>, usize, i64, i64)> =
        std::collections::BTreeMap::new();
    for w in writes.iter().filter(|w| (0..=3).contains(&w.ring)) {
        let entry = by_path
            .entry(w.path)
            .or_insert_with(|| (Vec::new(), 0, w.ring, w.ts));
        if !entry.0.contains(&w.session) {
            entry.0.push(w.session);
        }
        if w.session == session {
            entry.1 += 1;
        }
        entry.3 = w.ts;
    }
    for (path, (sessions, in_session, ring, ts)) in by_path {
        if sessions.len() < 2 && in_session < 10 {
            continue;
        }
        out.push(Candidate {
            id: staging::id_for("hot-file", path),
            reason: "hot-file".into(),
            session: session.to_string(),
            host: host.to_string(),
            ts,
            kind: "write".into(),
            payload: serde_json::json!({
                "file_path": path, "ring": ring,
                "sessions": sessions.len(), "writes_this_session": in_session,
            }),
        });
    }
}

/// Redacts secret-shaped material from a shell command: values of
/// secret-named KEY=value pairs, arguments of secret-named flags, URL
/// userinfo, JWTs, and long hex/base64 runs.
pub fn redact_secrets(cmd: &str) -> String {
    let mut out = String::with_capacity(cmd.len() + 16);
    let mut redact_next = false;
    let mut pending_assignment = false;
    let mut seeking_setter_value = 0u8;
    let mut after_setter = false;
    let mut cursor = 0;
    for (start, end) in shell_token_spans(cmd) {
        out.push_str(&cmd[cursor..start]);
        let word = &cmd[start..end];
        let plain = unquote_shell_word(word);
        if redact_next {
            out.push_str(REDACTED);
            redact_next = false;
            seeking_setter_value = 0;
            after_setter = false;
            pending_assignment = false;
        } else if seeking_setter_value > 0 {
            if plain.starts_with('-') {
                seeking_setter_value -= 1;
                out.push_str(word);
            } else {
                out.push_str(REDACTED);
                seeking_setter_value = 0;
                after_setter = false;
            }
            pending_assignment = false;
        } else if pending_assignment && plain == "=" {
            out.push_str(word);
            redact_next = true;
        } else {
            pending_assignment = false;
            if after_setter && !plain.contains('=') && keyish(&plain) {
                out.push_str(word);
                seeking_setter_value = 3;
                cursor = end;
                continue;
            }
            if is_setter(&plain) {
                after_setter = true;
                out.push_str(word);
                cursor = end;
                continue;
            }
            if after_setter && plain.starts_with('-') {
                out.push_str(word);
                cursor = end;
                continue;
            }
            after_setter = false;
            pending_assignment = !plain.contains('=') && keyish(&plain);
            out.push_str(&redact_word(word, &mut redact_next));
        }
        cursor = end;
    }
    out.push_str(&cmd[cursor..]);
    out
}

/// Shell-like token boundaries, including whitespace inside single/double
/// quotes. This is intentionally not a command evaluator; it only prevents
/// capture and read-hygiene code from splitting quoted values incorrectly.
pub(crate) fn shell_token_spans(command: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    let mut quote = None;
    let mut escaped = false;
    for (idx, ch) in command.char_indices() {
        if start.is_none() {
            if ch.is_whitespace() {
                continue;
            }
            start = Some(idx);
        }
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && quote != Some('\'') {
            escaped = true;
            continue;
        }
        match quote {
            Some(q) if ch == q => quote = None,
            Some(_) => {}
            None if matches!(ch, '\'' | '"') => quote = Some(ch),
            None if ch.is_whitespace() => {
                spans.push((start.take().expect("token started"), idx));
            }
            None => {}
        }
    }
    if let Some(start) = start {
        spans.push((start, command.len()));
    }
    spans
}

pub(crate) fn shell_words(command: &str) -> Vec<String> {
    shell_token_spans(command)
        .into_iter()
        .map(|(start, end)| unquote_shell_word(&command[start..end]))
        .collect()
}

fn unquote_shell_word(word: &str) -> String {
    // PowerShell accepts unquoted drive paths such as `C:\Users\me\file`.
    // Treating their separators as POSIX escapes turns that real path into
    // `C:Usersmefile`, so Windows hook reads can never be attributed. Strip a
    // matching outer quote but otherwise preserve a drive-absolute word.
    let inner = if word.len() >= 2
        && ((word.starts_with('\'') && word.ends_with('\''))
            || (word.starts_with('"') && word.ends_with('"')))
    {
        &word[1..word.len() - 1]
    } else {
        word
    };
    let bytes = inner.as_bytes();
    if bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'\\' | b'/')
    {
        return inner.to_string();
    }
    // A UNC share is all separators and no escapes. POSIX unescaping reads the
    // leading pair as one escaped backslash and silently demotes
    // \\\\server\\share to \\server\\share, which resolves to a different
    // place or to nothing.
    if inner.starts_with("\\\\") {
        return inner.to_string();
    }

    let mut out = String::with_capacity(word.len());
    let mut quote = None;
    let mut escaped = false;
    let mut chars = word.chars().peekable();
    while let Some(ch) = chars.next() {
        if escaped {
            out.push(ch);
            escaped = false;
            continue;
        }
        // Only a shell-meaningful escape consumes its backslash. A relative
        // Windows path has no drive letter to recognise it by, so `src\main.rs`
        // is otherwise read as an escaped `m` and recorded as `srcmain.rs` — a
        // read that can never be attributed to the file it touched. No shell
        // escapes an ordinary letter, so keeping the pair costs nothing.
        if ch == '\\' && quote != Some('\'') {
            let next = chars.peek().copied();
            if next.is_none_or(|c| matches!(c, ' ' | '\'' | '"' | '\\' | '$' | '`' | '\n')) {
                escaped = true;
                continue;
            }
            out.push('\\');
            continue;
        }
        match quote {
            Some(q) if ch == q => quote = None,
            Some(_) => out.push(ch),
            None if matches!(ch, '\'' | '"') => quote = Some(ch),
            None => out.push(ch),
        }
    }
    if escaped {
        out.push('\\');
    }
    out
}

/// Commands that introduce `NAME value` rather than `NAME=value`. The
/// secret-shaped name remains the gate, so ordinary assignments stay visible.
fn is_setter(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "set"
            | "setenv"
            | "export"
            | "declare"
            | "typeset"
            | "env"
            | "local"
            | "set-item"
            | "set-variable"
            | "new-variable"
            | "si"
            | "sv"
    )
}

/// Does this look like the NAME of a secret? Deliberately over-approximate:
/// redacting a harmless value is free, missing a secret is not.
fn keyish(word: &str) -> bool {
    let w = word.trim_start_matches('-').to_ascii_lowercase();
    [
        "token",
        "secret",
        "key",
        "pass",
        "pwd",
        "auth",
        "credential",
    ]
    .iter()
    .any(|k| w.contains(k))
}

fn redact_word(word: &str, redact_next: &mut bool) -> String {
    // KEY=value with a secret-shaped key: the whole value goes.
    if let Some((key, value)) = word.split_once('=')
        && !key.is_empty()
        && !value.is_empty()
        && keyish(key)
    {
        return format!("{key}={REDACTED}");
    }
    // Header syntax, including `Authorization: Bearer ...`. A header name
    // without an inline value redacts the following token.
    if let Some((key, value)) = word.split_once(':')
        && keyish(key.trim_matches(['\'', '"']))
    {
        if value.is_empty() {
            *redact_next = true;
            return word.to_string();
        }
        return format!("{key}:{REDACTED}");
    }
    // --password / --api-key style flags redact their next argument. A SHORT
    // flag with the value glued on (`mysql -psecret`, `docker -ukey`)
    // carries the secret inside the token itself: the old behavior set
    // redact_next for it, which kept the secret verbatim and destroyed the
    // innocent argument after it. Redact the token itself instead.
    if word.starts_with('-') && !word.contains('=') && keyish(word) {
        let rest = word.trim_start_matches('-');
        let short_attached = !word.starts_with("--") && rest.chars().count() > 2;
        if short_attached {
            return REDACTED.to_string();
        }
        *redact_next = true;
        return word.to_string();
    }
    // scheme://user:pass@host — the userinfo goes as one unit.
    if let Some(scheme_end) = word.find("://") {
        let tail = &word[scheme_end + 3..];
        if let Some(at) = tail.find('@') {
            let userinfo = &tail[..at];
            if userinfo.contains(':') && !userinfo.contains('/') {
                return format!("{}{REDACTED}@{}", &word[..scheme_end + 3], &tail[at + 1..]);
            }
        }
    }
    // Token-shaped runs inside the word: JWTs, long hex, long base64. The
    // base64 charset deliberately excludes '/' (it would swallow paths), and
    // demands mixed case plus a digit so long kebab-case words survive.
    let w = redact_jwt_runs(word);
    let w = redact_char_runs(&w, 32, REDACTED, is_hex, any_run);
    redact_char_runs(&w, 32, REDACTED, is_base64ish, mixed_case_with_digit)
}

fn is_hex(c: char) -> bool {
    c.is_ascii_hexdigit()
}

fn is_base64ish(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '=' | '-' | '_')
}

fn is_digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn any_run(_run: &str) -> bool {
    true
}

fn mixed_case_with_digit(run: &str) -> bool {
    run.chars().any(|c| c.is_ascii_digit())
        && run.chars().any(|c| c.is_ascii_uppercase())
        && run.chars().any(|c| c.is_ascii_lowercase())
}

/// Replaces maximal runs of `member` chars that are at least `min_len` bytes
/// long AND pass `qualifies`. Runs are ASCII by construction, so byte length
/// equals char count.
fn redact_char_runs(
    s: &str,
    min_len: usize,
    replacement: &str,
    member: fn(char) -> bool,
    qualifies: fn(&str) -> bool,
) -> String {
    let mut out = String::with_capacity(s.len());
    let mut run = String::new();
    for c in s.chars() {
        if member(c) {
            run.push(c);
        } else {
            flush_run(&mut out, &mut run, min_len, replacement, qualifies);
            out.push(c);
        }
    }
    flush_run(&mut out, &mut run, min_len, replacement, qualifies);
    out
}

fn flush_run(
    out: &mut String,
    run: &mut String,
    min_len: usize,
    replacement: &str,
    qualifies: fn(&str) -> bool,
) {
    if run.len() >= min_len && qualifies(run) {
        out.push_str(replacement);
    } else {
        out.push_str(run);
    }
    run.clear();
}

fn is_jwtish(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')
}

/// Redacts `eyJ…`-led base64url runs of at least 20 chars (a bare JWT header
/// is exactly 20) anywhere in the word, shorter lookalikes stay.
fn redact_jwt_runs(s: &str) -> String {
    let Some(start) = s.find("eyJ") else {
        return s.to_string();
    };
    let tail = &s[start..];
    let run_len = tail.find(|c: char| !is_jwtish(c)).unwrap_or(tail.len());
    if run_len >= 20 {
        format!(
            "{}{REDACTED}{}",
            &s[..start],
            redact_jwt_runs(&tail[run_len..])
        )
    } else {
        format!("{}{}", &s[..start + 3], redact_jwt_runs(&s[start + 3..]))
    }
}

/// Buglog-style command signature: lowercase, collapse whitespace, paths ->
/// `<path>`, long hex runs -> `<hex>`, digit runs -> `n`. Makes "the same
/// command" match across retries, sessions, and machines.
pub fn normalize_command(cmd: &str) -> String {
    let mut tokens = Vec::new();
    for word in cmd.split_whitespace() {
        let lower = word.to_ascii_lowercase();
        if lower.contains('/') || lower.contains('\\') {
            tokens.push("<path>".to_string());
            continue;
        }
        let no_hex = redact_char_runs(&lower, 8, "<hex>", is_hex, any_run);
        tokens.push(redact_char_runs(&no_hex, 1, "n", is_digit, any_run));
    }
    tokens.join(" ")
}

/// Secret-shaped path predicate (the recall index's idea, extended with
/// secret-store path components): such paths are withheld at capture.
pub fn secret_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    if lower
        .split(['/', '\\'])
        .any(|c| matches!(c, "secrets" | ".ssh" | ".gnupg" | ".aws"))
    {
        return true;
    }
    let base = lower.rsplit(['/', '\\']).next().unwrap_or(&lower);
    base.contains("secret")
        || base.contains("credential")
        || base.contains("password")
        || base.starts_with(".env")
        || base.ends_with(".pem")
        || base.ends_with(".key")
        || base.ends_with(".keystore")
        || base.ends_with(".p12")
        || base.ends_with(".pfx")
        || base.ends_with(".tfstate")
        || base == "id_rsa"
        || base == "id_ed25519"
}

fn file_payload(path: &str) -> serde_json::Value {
    let stored = if secret_path(path) { WITHHELD } else { path };
    serde_json::json!({"file_path": stored})
}

/// Write payload: the path (or the withheld placeholder) plus, for files
/// under `brain_root`, the resolved ring — computed at capture time so the
/// hot-file trap stays a field lookup (the same pattern as bash `norm`).
/// Withheld paths carry NO ring, which keeps them out of the trap by
/// construction. `rules` is the configured taxonomy the ring resolves through.
fn write_payload(path: &str, brain_root: &Path, rules: &RingRules) -> serde_json::Value {
    if secret_path(path) {
        return serde_json::json!({"file_path": WITHHELD});
    }
    match brain_ring(brain_root, path, rules) {
        Some(ring) => serde_json::json!({"file_path": path, "ring": ring}),
        None => serde_json::json!({"file_path": path}),
    }
}

/// Ring of a brain file: the location default for its brain-root-relative
/// path, overridden by a `ring:` frontmatter key read from a BOUNDED prefix
/// (the hook path never reads whole files). `None` for paths outside the
/// brain — code files carry no ring and are churn by contract.
fn brain_ring(brain_root: &Path, path: &str, rules: &RingRules) -> Option<u8> {
    // `strip_prefix` is case-SENSITIVE on Windows except for the drive
    // letter: `C:\Users\me\agents` vs `C:\Users\Me\agents\...` (8.3 short
    // names, subst drives, symlinked homes) all fail, classifying brain
    // writes as ringless code and blinding the hot-file trap for the whole
    // tree. Compare case-insensitively on Windows, as the filesystem does.
    #[cfg(windows)]
    fn strip_brain(root: &Path, p: &Path) -> Option<std::path::PathBuf> {
        let root_s = root
            .to_string_lossy()
            .trim_end_matches('\\')
            .to_ascii_lowercase();
        let p_s = p.to_string_lossy();
        if p_s.to_ascii_lowercase().starts_with(&root_s) {
            let rest = p_s[root_s.len()..].trim_start_matches(['\\', '/']);
            Some(std::path::PathBuf::from(rest))
        } else {
            None
        }
    }
    #[cfg(not(windows))]
    fn strip_brain(root: &Path, p: &Path) -> Option<std::path::PathBuf> {
        p.strip_prefix(root).ok().map(std::path::PathBuf::from)
    }
    let rel = strip_brain(brain_root, Path::new(path))?;
    let rel = crate::index::rel_doc_path(&rel);
    // An EXCLUDED path is not brain content, whatever ring its prefix would
    // otherwise imply. Without this the unmatched ring makes every excluded
    // path look like knowledge: `projects/` is code by contract, and a churn
    // trap fired on it stages candidates about throwaway build trees. Six of
    // one operator's sixteen staged candidates pointed inside ephemeral agent
    // worktrees, and five of those at directories already deleted.
    if rules.excluded(&rel) {
        return None;
    }
    let by_location = crate::index::default_ring(&rel, rules);
    Some(frontmatter_ring_bounded(Path::new(path)).unwrap_or(by_location))
}

/// `ring:` frontmatter override, parsed from the file's first 4 KiB only.
/// Unreadable file or absent key -> `None` (the location default applies).
/// A malformed value fails CLOSED to 255 inside the parser, exactly like the
/// index scan — quarantined-by-accident must never look recallable.
fn frontmatter_ring_bounded(path: &Path) -> Option<u8> {
    use std::io::Read as _;
    let f = std::fs::File::open(path).ok()?;
    let mut buf = Vec::with_capacity(4096);
    f.take(4096).read_to_end(&mut buf).ok()?;
    crate::index::frontmatter_ring(&String::from_utf8_lossy(&buf)).0
}

/// Error hint, parsed leniently: harness versions disagree on where failure
/// lives (a `tool_error` field, response `stderr`, error/is_error keys).
fn tool_failed(event: &HookEvent) -> bool {
    if event.tool_error.as_ref().is_some_and(|v| !v.is_null()) {
        return true;
    }
    let Some(obj) = event
        .tool_response
        .as_ref()
        .and_then(serde_json::Value::as_object)
    else {
        return false;
    };
    // Non-empty stderr alone is NOT failure: `git push`, `npm install` and
    // pip stream progress and warnings to stderr while exiting 0, and every
    // trap keys on this one field. Keep stderr as a failure hint only when
    // the response carries no explicit success marker - harnesses that
    // report failure ONLY through stderr (the reason this branch exists)
    // never set one.
    if obj
        .get("stderr")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| !s.trim().is_empty())
    {
        let explicit_success = ["is_error", "isError"]
            .iter()
            .any(|k| obj.get(*k).and_then(serde_json::Value::as_bool) == Some(false))
            || ["exit_code", "code"]
                .iter()
                .any(|k| obj.get(*k).and_then(serde_json::Value::as_i64) == Some(0));
        if !explicit_success {
            return true;
        }
    }
    if ["tool_error", "error"]
        .iter()
        .any(|k| obj.get(*k).is_some_and(|v| !v.is_null()))
    {
        return true;
    }
    ["is_error", "isError"]
        .iter()
        .any(|k| obj.get(*k).and_then(serde_json::Value::as_bool) == Some(true))
}

#[cfg(test)]
mod tests {
    #[test]
    fn stderr_noise_on_an_explicit_success_is_not_failure() {
        // git push / npm install / pip stream progress to stderr while
        // exiting 0; with an explicit success marker present (Claude Code's
        // Bash response always carries is_error), that must not record
        // `failed: true` - every trap keys on this one field.
        let ev = |resp: &str| crate::hook_io::HookEvent {
            tool_response: serde_json::from_str(resp).unwrap(),
            ..Default::default()
        };
        assert!(!tool_failed(&ev(
            r#"{"stdout":"done","stderr":"warning: 42 packets outdated","is_error":false}"#
        )));
        assert!(!tool_failed(&ev(
            r#"{"stderr":"progress...","exit_code":0}"#
        )));
        // Legacy harnesses that signal failure ONLY through stderr keep the
        // old behavior: no explicit marker, non-empty stderr -> failed.
        assert!(tool_failed(&ev(r#"{"stderr":"command not found"}"#)));
        assert!(tool_failed(&ev(r#"{"stderr":"x","is_error":true}"#)));
    }

    #[test]
    fn a_glued_short_flag_secret_redacts_itself_not_the_next_word() {
        // `mysql -psecretpw -h db` used to keep the password verbatim and
        // redact `-h`; the secret must go and the innocent host survive.
        let out = redact_secrets("mysql -psecretpw -h db.corp.example");
        assert!(!out.contains("secretpw"), "secret must be gone: {out}");
        assert!(out.contains("db.corp.example"), "host must survive: {out}");
        // Separated form unchanged: flag redacts its next argument.
        let out = redact_secrets("tool --password hunter2 -v");
        assert!(!out.contains("hunter2"), "separated value must go: {out}");
        assert!(out.contains("--password"), "flag name stays: {out}");
    }

    /// Windows is a first-class platform, and a corrupted path is a read that
    /// can never be attributed to the file it touched.
    #[test]
    fn windows_paths_survive_unquoting_in_every_shape() {
        // Drive-absolute, quoted and bare.
        assert_eq!(
            unquote_shell_word(r"C:\Users\me\notes.md"),
            r"C:\Users\me\notes.md"
        );
        assert_eq!(
            unquote_shell_word("\"C:\\Users\\me\\a b.md\""),
            r"C:\Users\me\a b.md"
        );
        // Relative: no drive letter to recognise it by.
        assert_eq!(unquote_shell_word(r"src\main.rs"), r"src\main.rs");
        // UNC: all separators, no escapes.
        assert_eq!(
            unquote_shell_word(r"\\server\share\file.md"),
            r"\\server\share\file.md"
        );
        // POSIX escaping still works, or every unix path breaks instead.
        assert_eq!(unquote_shell_word(r"my\ file.md"), "my file.md");
        assert_eq!(unquote_shell_word(r"a\'b"), "a'b");
        assert_eq!(unquote_shell_word("'/tmp/x y.md'"), "/tmp/x y.md");
    }

    /// The failure this prevents was real: an operator's staging queue held
    /// candidates about files inside throwaway agent worktrees, five of them
    /// pointing at directories that had already been deleted.
    #[test]
    fn an_excluded_path_carries_no_ring_and_so_never_reaches_a_trap() {
        let brain = tempfile::tempdir().unwrap();
        let rules = RingRules::default();
        let ring = |rel: &str| {
            let p = brain.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, "x").unwrap();
            brain_ring(brain.path(), &p.to_string_lossy(), &rules)
        };
        // Excluded by default: code, retired knowledge, disposable material.
        assert_eq!(
            ring("projects/github/x/.claude/worktrees/w1/src/main.rs"),
            None
        );
        assert_eq!(ring("knowledge/archive/old.md"), None);
        assert_eq!(ring("todo/scratch/dump.md"), None);
        // Real brain content still resolves, or the trap would never fire.
        assert_eq!(ring("knowledge/hosts/server/zfs.md"), Some(3));
        assert_eq!(ring("knowledge/behaviours/feedback_x.md"), Some(2));
    }

    use super::*;
    use serde_json::json;

    /// Test brain root: capture resolves write rings against this prefix.
    /// The files need not exist — ring resolution then falls back to the
    /// location default, exactly like production on a deleted file.
    const BRAIN: &str = "/b/agents";

    struct Fixture {
        _dir: tempfile::TempDir,
        ex: Exhaust,
    }

    fn fixture(host: &str) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let ex = Exhaust::new(
            dir.path().join("logs/cfetch"),
            dir.path().join("staging/cfetch"),
            host.to_string(),
            1 << 20,
        );
        Fixture { _dir: dir, ex }
    }

    /// A second host writing into the SAME tree.
    fn peer(f: &Fixture, host: &str) -> Exhaust {
        Exhaust::new(
            f.ex.logs_dir.clone(),
            f.ex.staging_dir.clone(),
            host.into(),
            1 << 20,
        )
    }

    fn bash_event(session: &str, cmd: &str, failed: bool) -> HookEvent {
        HookEvent {
            session_id: Some(session.into()),
            tool_name: Some("Bash".into()),
            tool_input: Some(json!({"command": cmd})),
            tool_response: Some(json!({"stderr": if failed { "boom" } else { "" }})),
            ..Default::default()
        }
    }

    fn file_event(session: &str, tool: &str, path: &str) -> HookEvent {
        HookEvent {
            session_id: Some(session.into()),
            tool_name: Some(tool.into()),
            tool_input: Some(json!({"file_path": path})),
            ..Default::default()
        }
    }

    impl Fixture {
        fn cap(&self, event: &HookEvent) {
            self.ex
                .capture_post_tool(event, Path::new(BRAIN), &RingRules::default())
                .unwrap();
        }

        fn stop(&self, session: &str) {
            self.ex.record_stop(session).unwrap();
        }

        fn records(&self) -> Vec<Record> {
            jsonl::read_all(&self.ex.logs_dir, STREAM).records
        }

        fn staged(&self) -> Vec<Candidate> {
            staging::list(&self.ex.staging_dir)
        }
    }

    #[test]
    fn capture_appends_versioned_lines_to_this_hosts_stream() {
        let f = fixture("host-alpha");
        f.cap(&bash_event("s1", "cargo build", false));
        let path = jsonl::stream_path(&f.ex.logs_dir, STREAM, "host-alpha");
        assert!(path.is_file(), "the stream file carries the host name");
        let raw = std::fs::read_to_string(&path).unwrap();
        assert_eq!(raw.lines().count(), 1, "one event, one line");
        let line: serde_json::Value = serde_json::from_str(raw.lines().next().unwrap()).unwrap();
        assert_eq!(line["v"], 1);
        assert_eq!(line["host"], "host-alpha");
        assert_eq!(line["kind"], "bash");
        assert_eq!(line["session"], "s1");
        assert_eq!(line["payload"]["command"], "cargo build");
    }

    #[test]
    fn bash_capture_is_redacted_session_keyed_and_error_hinted() {
        let f = fixture("h1");
        f.cap(&bash_event("s1", "export API_TOKEN=abc123", true));
        f.cap(&bash_event("s2", "ls", false));
        let all = f.records();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].str("session"), "s1");
        assert_eq!(all[0].kind(), "bash");
        let p = all[0].value("payload").unwrap();
        assert_eq!(p["command"], "export API_TOKEN=<redacted>");
        assert_eq!(p["failed"], true);
        assert_eq!(p["norm"], "export api_token=<redacted>");
        assert_eq!(all[1].str("session"), "s2");
        assert_eq!(all[1].value("payload").unwrap()["failed"], false);
    }

    #[test]
    fn secret_paths_are_withheld_at_capture() {
        let f = fixture("h1");
        f.cap(&file_event("s1", "Write", "/home/x/.env.production"));
        f.cap(&file_event("s1", "Edit", "/home/x/mind/secrets/koyeb.yml"));
        f.cap(&file_event("s1", "Read", "/home/x/server.pem"));
        f.cap(&file_event("s1", "Write", "/home/x/src/main.rs"));
        let all = f.records();
        assert_eq!(all.len(), 4);
        for r in &all[..3] {
            assert_eq!(r.value("payload").unwrap()["file_path"], WITHHELD);
        }
        assert_eq!(
            all[3].value("payload").unwrap()["file_path"],
            "/home/x/src/main.rs"
        );
    }

    #[test]
    fn codex_apply_patch_captures_every_changed_path_from_the_event_cwd() {
        let f = fixture("h1");
        let event = HookEvent {
            session_id: Some("codex-session".into()),
            cwd: Some(BRAIN.into()),
            tool_name: Some("apply_patch".into()),
            tool_input: Some(
                json!({"command": "*** Begin Patch\n*** Update File: knowledge/a.md\n*** Move to: knowledge/b.md\n*** Add File: /tmp/new.rs\n*** Delete File: mind/secrets/token.md\n*** End Patch"}),
            ),
            ..Default::default()
        };
        f.cap(&event);

        let all = f.records();
        assert_eq!(all.len(), 4, "one write record per distinct patch target");
        let actual = all[0].value("payload").unwrap()["file_path"]
            .as_str()
            .unwrap()
            .to_string();
        let expected = Path::new(BRAIN).join("knowledge/a.md");
        assert_eq!(
            Path::new(&actual).components().collect::<Vec<_>>(),
            expected.components().collect::<Vec<_>>()
        );
        assert_eq!(all[0].value("payload").unwrap()["ring"], 3);
        let actual = all[1].value("payload").unwrap()["file_path"]
            .as_str()
            .unwrap()
            .to_string();
        let expected = Path::new(BRAIN).join("knowledge/b.md");
        assert_eq!(
            Path::new(&actual).components().collect::<Vec<_>>(),
            expected.components().collect::<Vec<_>>()
        );
        let actual = all[2].value("payload").unwrap()["file_path"]
            .as_str()
            .unwrap();
        assert_eq!(
            Path::new(actual).components().collect::<Vec<_>>(),
            Path::new("/tmp/new.rs").components().collect::<Vec<_>>()
        );
        assert_eq!(all[3].value("payload").unwrap()["file_path"], WITHHELD);
    }

    #[test]
    fn codex_apply_patch_path_extraction_is_deduplicated_and_lenient() {
        let event = HookEvent {
            cwd: Some("/work".into()),
            tool_name: Some("apply_patch".into()),
            tool_input: Some(
                json!({"command": "*** Update File: src/lib.rs\n*** Update File: src/lib.rs\nnot a header"}),
            ),
            ..Default::default()
        };
        let paths = written_paths(&event);
        assert_eq!(paths.len(), 1);
        assert_eq!(
            Path::new(&paths[0]).components().collect::<Vec<_>>(),
            Path::new("/work")
                .join("src/lib.rs")
                .components()
                .collect::<Vec<_>>()
        );

        let missing = HookEvent {
            tool_name: Some("apply_patch".into()),
            tool_input: Some(json!({"patch": "different dialect"})),
            ..Default::default()
        };
        assert!(written_paths(&missing).is_empty());
    }

    #[test]
    fn tool_error_field_counts_as_failure() {
        let f = fixture("h1");
        let mut ev = bash_event("s1", "ls", false);
        ev.tool_error = Some(json!("exit status 1"));
        f.cap(&ev);
        assert_eq!(f.records()[0].value("payload").unwrap()["failed"], true);
    }

    #[test]
    fn non_capture_tools_and_missing_fields_are_ignored() {
        let f = fixture("h1");
        let mut glob = bash_event("s1", "x", false);
        glob.tool_name = Some("Glob".into());
        f.cap(&glob);
        let mut no_cmd = bash_event("s1", "x", false);
        no_cmd.tool_input = Some(json!({"description": "no command field"}));
        f.cap(&no_cmd);
        let mut no_path = file_event("s1", "Write", "x");
        no_path.tool_input = Some(json!({}));
        f.cap(&no_path);
        assert!(f.records().is_empty());
        f.cap(&file_event("s1", "MultiEdit", "/a/b.rs"));
        f.cap(&file_event("s1", "Read", "/a/b.rs"));
        let all = f.records();
        assert_eq!(all[0].kind(), "write");
        assert_eq!(all[1].kind(), "read");
    }

    #[test]
    fn long_commands_are_clamped_so_one_event_stays_one_short_line() {
        let f = fixture("h1");
        f.cap(&bash_event("s1", &"x".repeat(MAX_COMMAND_CHARS * 3), false));
        let raw =
            std::fs::read_to_string(jsonl::stream_path(&f.ex.logs_dir, STREAM, "h1")).unwrap();
        assert_eq!(raw.lines().count(), 1);
        assert!(
            raw.len() < 3 * MAX_COMMAND_CHARS,
            "line stayed small: {}",
            raw.len()
        );
    }

    #[test]
    fn the_stream_rotates_at_its_byte_cap() {
        let dir = tempfile::tempdir().unwrap();
        let ex = Exhaust::new(
            dir.path().join("logs/cfetch"),
            dir.path().join("staging/cfetch"),
            "h1".into(),
            1000,
        );
        for i in 0..200 {
            ex.record("s1", "bash", &json!({"i": i})).unwrap();
        }
        let files = jsonl::stream_paths(&ex.logs_dir, STREAM);
        assert!(files.len() > 1, "the cap rotated the live file");
        assert!(
            files.len() <= jsonl::MAX_ROTATIONS + 1,
            "at most 2 rotations are kept"
        );
        for p in &files {
            assert!(
                std::fs::metadata(p).unwrap().len() <= 1000 + 128,
                "no generation exceeds the cap: {}",
                p.display()
            );
        }
    }

    #[test]
    fn stop_records_turn_summary_counts() {
        let f = fixture("h1");
        f.cap(&bash_event("s1", "cargo test", true));
        f.cap(&bash_event("s1", "ls", false));
        f.cap(&file_event("s1", "Write", "/a/b.rs"));
        f.cap(&file_event("s1", "Read", "/a/b.rs"));
        f.cap(&bash_event("s2", "other session", false));
        f.stop("s1");
        let all = f.records();
        let turn = all.iter().rev().find(|r| r.kind() == "turn").unwrap();
        assert_eq!(turn.str("session"), "s1");
        let p = turn.value("payload").unwrap();
        assert_eq!(p["bash"], 2, "s2 activity must not leak into s1's summary");
        assert_eq!(p["write"], 1);
        assert_eq!(p["read"], 1);
    }

    #[test]
    fn fix_discovered_trap_stages_one_candidate_carrying_both_halves() {
        let f = fixture("h1");
        f.cap(&bash_event("s1", "cargo test --lib", true));
        f.cap(&bash_event("s1", "echo unrelated", false));
        // Same normalized command (case + whitespace noise), now succeeding.
        f.cap(&bash_event("s1", "CARGO   TEST --lib", false));
        f.stop("s1");
        let staged = f.staged();
        assert_eq!(staged.len(), 1);
        assert_eq!(staged[0].reason, "fix-discovered");
        assert_eq!(staged[0].payload["failed_command"], "cargo test --lib");
        assert_eq!(staged[0].payload["fixed_command"], "CARGO   TEST --lib");
        f.stop("s1");
        assert_eq!(f.staged(), staged, "the trap must be idempotent");
    }

    #[test]
    fn no_fix_flag_when_success_precedes_failure() {
        let f = fixture("h1");
        f.cap(&bash_event("s1", "cargo test", false));
        f.cap(&bash_event("s1", "cargo test", true));
        f.stop("s1");
        assert!(
            f.staged().is_empty(),
            "success BEFORE failure is not a discovered fix"
        );
    }

    #[test]
    fn recurring_failure_trap_fires_once() {
        let f = fixture("h1");
        // Same normalized command (paths normalize away) failing in 2 sessions.
        f.cap(&bash_event("s1", "bash /tmp/deploy.sh", true));
        f.cap(&bash_event("s2", "bash /opt/deploy.sh", true));
        f.stop("s2");
        let staged = f.staged();
        assert_eq!(staged.len(), 1);
        assert_eq!(staged[0].reason, "recurring-failure");
        assert_eq!(staged[0].payload["sessions"], 2);
        f.stop("s2");
        f.stop("s1");
        assert_eq!(f.staged(), staged, "the trap must be idempotent");
    }

    #[test]
    fn hot_file_counts_only_brain_rings_0_to_3() {
        let f = fixture("h1");
        for _ in 0..10 {
            f.cap(&file_event("s1", "Write", "/elsewhere/churn.rs"));
            f.cap(&file_event(
                "s1",
                "Write",
                "/b/agents/todo/active/x/STATUS.md",
            ));
            f.cap(&file_event(
                "s1",
                "Write",
                "/b/agents/knowledge/hosts/server.md",
            ));
        }
        f.stop("s1");
        let staged = f.staged();
        assert_eq!(staged.len(), 1, "code and ring-4 working files never fire");
        assert_eq!(staged[0].reason, "hot-file");
        assert_eq!(
            staged[0].payload["file_path"],
            "/b/agents/knowledge/hosts/server.md"
        );
        assert_eq!(
            staged[0].payload["ring"], 3,
            "the candidate carries the resolved ring"
        );
        f.stop("s1");
        assert_eq!(f.staged().len(), 1, "the trap must be idempotent");
    }

    #[test]
    fn hot_file_needs_ten_writes_within_one_session() {
        let f = fixture("h1");
        for _ in 0..9 {
            f.cap(&file_event("s1", "Write", "/b/agents/knowledge/one.md"));
        }
        f.stop("s1");
        assert!(
            f.staged().is_empty(),
            "9 same-session writes are churn, not heat"
        );
        f.cap(&file_event("s1", "Write", "/b/agents/knowledge/one.md"));
        f.stop("s1");
        assert_eq!(f.staged().len(), 1, "the 10th write crosses the threshold");
    }

    #[test]
    fn hot_file_fires_across_two_sessions_and_only_once() {
        let f = fixture("h1");
        f.cap(&file_event("s1", "Write", "/b/agents/knowledge/two.md"));
        f.stop("s1");
        assert!(
            f.staged().is_empty(),
            "one session alone is no cross-session pattern"
        );
        f.cap(&file_event("s2", "Write", "/b/agents/knowledge/two.md"));
        f.stop("s2");
        assert_eq!(f.staged().len(), 1);
        f.stop("s2");
        f.stop("s1");
        assert_eq!(
            f.staged().len(),
            1,
            "the trap must be idempotent across sessions"
        );
    }

    #[test]
    fn hot_file_respects_frontmatter_ring_override() {
        // A knowledge file declaring itself ring 5 is quarantined content and
        // never a candidate; a todo file declaring ring 3 counts.
        let brain = tempfile::tempdir().unwrap();
        let quarantined = brain.path().join("knowledge/q.md");
        std::fs::create_dir_all(quarantined.parent().unwrap()).unwrap();
        std::fs::write(&quarantined, "---\nring: 5\n---\nbody\n").unwrap();
        let promoted = brain.path().join("todo/notes.md");
        std::fs::create_dir_all(promoted.parent().unwrap()).unwrap();
        std::fs::write(&promoted, "---\nring: 3\n---\nbody\n").unwrap();

        let f = fixture("h1");
        for s in ["s1", "s2"] {
            for p in [&quarantined, &promoted] {
                f.ex.capture_post_tool(
                    &file_event(s, "Write", &p.to_string_lossy()),
                    brain.path(),
                    &RingRules::default(),
                )
                .unwrap();
            }
            f.stop(s);
        }
        let staged = f.staged();
        assert_eq!(
            staged.len(),
            1,
            "the ring-5 override quarantines, the ring-3 override admits"
        );
        assert_eq!(
            staged[0].payload["file_path"],
            promoted.to_string_lossy().as_ref()
        );
        assert_eq!(staged[0].payload["ring"], 3);
    }

    #[test]
    fn hot_file_never_aggregates_withheld_paths() {
        // Two DIFFERENT secret files under the brain share one stored
        // placeholder across two sessions: they must not merge into a fake hot
        // file (withheld payloads carry no ring).
        let f = fixture("h1");
        f.cap(&file_event(
            "s1",
            "Write",
            "/b/agents/knowledge/api-password.md",
        ));
        f.cap(&file_event(
            "s2",
            "Write",
            "/b/agents/knowledge/db-password.md",
        ));
        f.stop("s2");
        assert!(f.staged().is_empty());
    }

    #[test]
    fn consumed_candidates_do_not_re_stage_on_this_host() {
        let f = fixture("h1");
        for s in ["s1", "s2"] {
            f.cap(&file_event(s, "Write", "/b/agents/knowledge/one.md"));
        }
        f.stop("s2");
        let id = f.staged()[0].id.clone();
        assert!(staging::consume(&f.ex.staging_dir, &id).unwrap());
        f.ex.record_decision(&id, "consume").unwrap();
        f.stop("s2");
        assert!(
            f.staged().is_empty(),
            "a consumed candidate must not come back"
        );
    }

    #[test]
    fn dismissed_candidates_do_not_re_stage_on_any_host() {
        let f = fixture("host-alpha");
        for s in ["s1", "s2"] {
            f.cap(&file_event(s, "Write", "/b/agents/knowledge/one.md"));
        }
        f.stop("s2");
        let id = f.staged()[0].id.clone();
        assert!(staging::dismiss(&f.ex.staging_dir, &id).unwrap());
        f.stop("s2");
        assert!(
            f.staged().is_empty(),
            "the dismissed file is the tree-wide marker"
        );
        // …including for a different host writing the same pattern.
        let other = peer(&f, "host-beta");
        for s in ["s3", "s4"] {
            other
                .capture_post_tool(
                    &file_event(s, "Write", "/b/agents/knowledge/one.md"),
                    Path::new(BRAIN),
                    &RingRules::default(),
                )
                .unwrap();
        }
        other.record_stop("s4").unwrap();
        assert!(
            f.staged().is_empty(),
            "candidate ids are content-addressed, not per host"
        );
    }

    #[test]
    fn two_hosts_share_one_staging_queue_and_one_log_directory() {
        // The defect this change fixes: what host alpha flags, host beta sees.
        let f = fixture("host-alpha");
        let beta = peer(&f, "host-beta");
        for s in ["s1", "s2"] {
            f.cap(&file_event(s, "Write", "/b/agents/knowledge/alpha.md"));
            beta.capture_post_tool(
                &file_event(s, "Write", "/b/agents/knowledge/beta.md"),
                Path::new(BRAIN),
                &RingRules::default(),
            )
            .unwrap();
        }
        f.stop("s2");
        beta.record_stop("s2").unwrap();

        let staged = f.staged();
        assert_eq!(staged.len(), 2, "both hosts' candidates are in one queue");
        let hosts: Vec<&str> = staged.iter().map(|c| c.host.as_str()).collect();
        assert!(hosts.contains(&"host-alpha") && hosts.contains(&"host-beta"));
        assert_eq!(
            staged,
            staging::list(&beta.staging_dir),
            "and every host lists the same queue"
        );

        // Each host's exhaust went to its OWN file; reads see both.
        assert!(jsonl::stream_path(&f.ex.logs_dir, STREAM, "host-alpha").is_file());
        assert!(jsonl::stream_path(&f.ex.logs_dir, STREAM, "host-beta").is_file());
        let all = jsonl::read_all(&f.ex.logs_dir, STREAM);
        assert!(all.records.iter().any(|r| r.host == "host-alpha"));
        assert!(all.records.iter().any(|r| r.host == "host-beta"));
        assert!(all.unreadable.is_empty());
    }

    #[test]
    fn staging_cap_stages_one_explicit_notice_and_deletes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let ex = Exhaust::new(
            dir.path().join("logs/cfetch"),
            dir.path().join("staging/cfetch"),
            "h1".into(),
            1 << 20,
        );
        // Fill staging to the cap with hand-made candidates, then make the
        // traps fire.
        for i in 0..MAX_STAGED {
            staging::write(
                &ex.staging_dir,
                &Candidate {
                    id: format!("filler-{i:05}"),
                    reason: "hot-file".into(),
                    session: "s0".into(),
                    host: "h1".into(),
                    ts: 1,
                    kind: "write".into(),
                    payload: serde_json::json!({}),
                },
            )
            .unwrap();
        }
        for s in ["s1", "s2"] {
            ex.capture_post_tool(
                &file_event(s, "Write", "/b/agents/knowledge/hot.md"),
                Path::new(BRAIN),
                &RingRules::default(),
            )
            .unwrap();
        }
        ex.record_stop("s2").unwrap();
        let staged = staging::list(&ex.staging_dir);
        assert_eq!(
            staged.len(),
            MAX_STAGED + 1,
            "the queue grew by exactly the notice"
        );
        assert_eq!(
            staged.iter().filter(|c| c.reason == "staging-full").count(),
            1
        );
        assert_eq!(
            staged
                .iter()
                .filter(|c| c.reason == "hot-file" && c.session == "s2")
                .count(),
            0,
            "no new candidate is admitted past the cap"
        );
        ex.record_stop("s2").unwrap();
        assert_eq!(
            staging::list(&ex.staging_dir).len(),
            MAX_STAGED + 1,
            "an overflowed queue converges instead of churning notices"
        );
    }

    #[test]
    fn stats_report_staging_by_reason_and_the_stream_footprint() {
        let f = fixture("h1");
        let empty = f.ex.stats();
        assert_eq!(empty.staged_total, 0);
        assert_eq!(empty.bytes, 0);
        assert!(
            !f.ex.logs_dir.exists(),
            "a reporting surface must never create ring-6 state"
        );

        for s in ["s1", "s2"] {
            f.cap(&file_event(s, "Write", "/b/agents/knowledge/one.md"));
            f.cap(&bash_event(s, "bash /tmp/deploy.sh", true));
        }
        f.stop("s2");
        let s = f.ex.stats();
        assert_eq!(s.staged_total, 2);
        assert_eq!(
            s.staged_by_reason,
            vec![
                ("recurring-failure".to_string(), 1),
                ("hot-file".to_string(), 1)
            ],
            "reasons in trap order"
        );
        assert!(s.bytes > 0);
    }

    #[test]
    fn redaction_covers_each_secret_shape() {
        let cases = [
            // KEY=value with secret-shaped key
            ("export API_TOKEN=abc123", "export API_TOKEN=<redacted>"),
            ("PASSWORD=x id", "PASSWORD=<redacted> id"),
            ("AUTH_HEADER=Basic123", "AUTH_HEADER=<redacted>"),
            // --flag=value and --flag value style
            ("run --password=hunter2", "run --password=<redacted>"),
            (
                "mysql --password hunter2 -h db",
                "mysql --password <redacted> -h db",
            ),
            ("curl --api-key sk-live-abc", "curl --api-key <redacted>"),
            // Quoted values and PowerShell assignment syntax.
            (
                "export API_TOKEN=\"value with spaces\"",
                "export API_TOKEN=<redacted>",
            ),
            (
                "$env:API_TOKEN = \"totally-fake-password-12345\"",
                "$env:API_TOKEN = <redacted>",
            ),
            (
                "set -x api_token sk-live-abc",
                "set -x api_token <redacted>",
            ),
            (
                "set -gx OPENAI_API_KEY sk-live-abc",
                "set -gx OPENAI_API_KEY <redacted>",
            ),
            (
                "Set-Item Env:\\API_TOKEN -Value sk-live-abc",
                "Set-Item Env:\\API_TOKEN -Value <redacted>",
            ),
            (
                "Set-Variable -Name api_secret -Value sk-live-abc",
                "Set-Variable -Name api_secret -Value <redacted>",
            ),
            (
                "declare -x DB_PASSWORD hunter2",
                "declare -x DB_PASSWORD <redacted>",
            ),
            (
                "curl -H 'Authorization: Bearer totally-fake-token' https://example.com",
                "curl -H 'Authorization:<redacted> https://example.com",
            ),
            // URL userinfo
            (
                "git clone https://user:hunter2@github.com/x/y.git",
                "git clone https://<redacted>@github.com/x/y.git",
            ),
            // JWT
            (
                "echo eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.abcDEF123",
                "echo <redacted>",
            ),
            // 32+ hex run
            (
                "verify deadbeefdeadbeefdeadbeefdeadbeef",
                "verify <redacted>",
            ),
            // 32+ mixed-case base64 run
            (
                "echo VGhpc0lzQVNlY3JldFZhbHVlMTIzNDU2Nzg5MA==",
                "echo <redacted>",
            ),
        ];
        for (input, want) in cases {
            assert_eq!(redact_secrets(input), want, "input: {input}");
        }
    }

    #[test]
    fn redaction_leaves_ordinary_commands_alone() {
        let cases = [
            "cargo build --release --target-dir /very/long/path/to/the/target/directory",
            "git commit -m updating-the-frobnicator-subsystem-for-release",
            "rg pattern src --type rust",
            "echo eyJshort",
            "curl https://example.com/deep/path/segment",
        ];
        for input in cases {
            assert_eq!(redact_secrets(input), input, "must stay untouched: {input}");
        }
    }

    #[test]
    fn incomplete_setter_does_not_redact_a_later_command() {
        let got = redact_secrets("set -x api_token -a -b -c -d -e echo hello");
        assert!(
            got.contains("hello"),
            "setter state wandered too far: {got:?}"
        );
    }

    #[test]
    fn normalization_matches_across_noise() {
        assert_eq!(
            normalize_command("CARGO  TEST --lib"),
            normalize_command("cargo test --lib")
        );
        assert_eq!(
            normalize_command("bash /tmp/a/b.sh 42"),
            normalize_command("bash /opt/c/d.sh 7")
        );
        assert_eq!(
            normalize_command("git checkout deadbeefcafe1234"),
            "git checkout <hex>"
        );
        assert_eq!(normalize_command("retry attempt 12"), "retry attempt n");
    }

    #[test]
    fn secret_path_predicate() {
        for yes in [
            "/home/x/.env.production",
            "/home/x/agents/mind/secrets/koyeb.yml",
            "/etc/ssl/private/server.key",
            "/home/x/credentials.json",
            "/home/x/my-password-list.md",
            "cert.pem",
            "/home/x/.ssh/config",
            r"C:\Users\alice\.ssh\config",
            r"C:\repo\secrets\token.txt",
            "/home/x/id_ed25519",
            "/srv/deploy/terraform.tfstate",
        ] {
            assert!(secret_path(yes), "must be withheld: {yes}");
        }
        for no in [
            "/home/x/src/main.rs",
            "/home/x/keyboard.md",
            "notes/envelope-budget.md",
        ] {
            assert!(!secret_path(no), "must pass through: {no}");
        }
    }

    /// A captured bash event at an explicit moment, in the shape
    /// `capture_post_tool` writes. The query surface is a question about
    /// history, so its fixtures have to place events in time.
    fn bash_at(ex: &Exhaust, ts: i64, session: &str, cmd: &str, failed: bool) {
        ex.record_at(
            ts,
            session,
            "bash",
            &json!({"command": cmd, "norm": normalize_command(cmd), "failed": failed}),
        )
        .unwrap();
    }

    #[test]
    fn failure_history_aggregates_one_signature_across_hosts_and_sessions() {
        let f = fixture("host-alpha");
        let beta = peer(&f, "host-beta");
        bash_at(&f.ex, 100, "s1", "cargo test --workspace", true);
        bash_at(&f.ex, 200, "s2", "cargo test --workspace", true);
        // Same signature from another machine's stream, spelled differently.
        bash_at(&beta, 300, "s3", "CARGO test  --workspace", true);
        bash_at(&f.ex, 400, "s2", "cargo build", true);
        bash_at(&f.ex, 500, "s2", "ls", false);
        f.ex.record_at(
            600,
            "s2",
            "write",
            &json!({"file_path": "/b/x.md", "ring": 3}),
        )
        .unwrap();

        let h = f.ex.failure_history("cargo test", 8);
        assert_eq!(h.signatures, 2, "successes and writes are not failures");
        assert_eq!(h.failures, 4);
        assert_eq!(h.matches.len(), 1);
        let m = &h.matches[0];
        assert_eq!(m.norm, "cargo test --workspace");
        assert_eq!(m.failures, 3);
        assert_eq!(m.sessions, 3);
        assert_eq!(
            m.hosts,
            vec!["host-alpha", "host-beta"],
            "the tree is shared"
        );
        assert_eq!(m.first_ts, 100);
        assert_eq!(m.last_ts, 300);
        assert_eq!(
            m.last_command, "CARGO test  --workspace",
            "the concrete newest line, not the signature"
        );
        assert_eq!(m.recovered_sessions, 0);
        assert!(!m.staged);

        // A limit truncates the answer, never the denominator that says how
        // much history it was drawn from.
        let h = f.ex.failure_history("", 1);
        assert_eq!(h.matches.len(), 1);
        assert_eq!((h.signatures, h.failures), (2, 4));
    }

    #[test]
    fn recovery_needs_a_success_after_the_failure_in_the_same_session() {
        let f = fixture("h1");
        bash_at(&f.ex, 100, "s1", "cargo test", true);
        bash_at(&f.ex, 110, "s1", "cargo test", false);
        bash_at(&f.ex, 120, "s1", "cargo test", false);
        // Succeeded BEFORE it first failed here, then failed.
        bash_at(&f.ex, 200, "s2", "make lint", false);
        bash_at(&f.ex, 210, "s2", "make lint", true);
        // Succeeded in a session that never saw it fail.
        bash_at(&f.ex, 300, "s3", "make lint", false);

        let h = f.ex.failure_history("", 8);
        let recovered = |norm: &str| {
            h.matches
                .iter()
                .find(|m| m.norm == norm)
                .unwrap_or_else(|| panic!("{norm} missing"))
                .recovered_sessions
        };
        assert_eq!(recovered("cargo test"), 1, "one session, counted once");
        assert_eq!(
            recovered("make lint"),
            0,
            "a success that did not follow a failure of the same signature in the \
             same session is not a recovery"
        );
    }

    #[test]
    fn rotated_generations_are_read_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        // A one-byte cap rotates on every append, so the three events land in
        // three generations: the oldest in `.2`, the newest in the live file.
        let ex = Exhaust::new(
            dir.path().join("logs/cfetch"),
            dir.path().join("staging/cfetch"),
            "h1".into(),
            1,
        );
        bash_at(&ex, 100, "s1", "cargo test", true);
        bash_at(&ex, 200, "s1", "cargo test", false);
        bash_at(&ex, 300, "s1", "ls", false);
        assert!(
            ex.logs_dir.join("exhaust-h1.2.jsonl").is_file(),
            "the fixture must actually rotate"
        );

        let h = ex.failure_history("cargo test", 8);
        assert_eq!(
            h.matches[0].recovered_sessions, 1,
            "name order reads a host's rotations newest-first and would see the \
             success before the failure it followed"
        );
    }

    #[test]
    fn the_exact_signature_wins_and_the_query_is_normalized_like_a_command() {
        let f = fixture("h1");
        for (i, session) in ["s1", "s2", "s3", "s4", "s5"].iter().enumerate() {
            bash_at(&f.ex, 100 + i as i64, session, "cargo test --all", true);
        }
        bash_at(&f.ex, 200, "s6", "cargo test", true);
        bash_at(&f.ex, 300, "s7", "cargo test --jobs 8", true);

        let h = f.ex.failure_history("cargo test", 8);
        assert_eq!(h.matches.len(), 3);
        assert_eq!(
            h.matches[0].norm, "cargo test",
            "the signature that was asked about outranks a more frequent one"
        );
        assert_eq!(
            h.matches[1].norm, "cargo test --all",
            "then by how often it failed"
        );

        // Case and digits are erased from a stored signature, so the query
        // goes through the same normalizer before it is matched.
        let h = f.ex.failure_history("CARGO TEST --jobs 4", 8);
        assert_eq!(h.matches.len(), 1);
        assert_eq!(h.matches[0].norm, "cargo test --jobs n");
    }

    #[test]
    fn a_signature_the_trap_already_staged_says_so() {
        let f = fixture("h1");
        // The same command failing in two sessions is what the
        // recurring-failure trap stages.
        f.cap(&bash_event("s1", "cargo test", true));
        f.cap(&bash_event("s2", "cargo test", true));
        f.cap(&bash_event("s2", "cargo build", true));
        f.stop("s2");
        assert!(f.staged().iter().any(|c| c.reason == "recurring-failure"));

        let h = f.ex.failure_history("cargo", 8);
        let flags: Vec<(&str, bool)> = h
            .matches
            .iter()
            .map(|m| (m.norm.as_str(), m.staged))
            .collect();
        assert!(flags.contains(&("cargo test", true)), "got {flags:?}");
        assert!(
            flags.contains(&("cargo build", false)),
            "one failure in one session is not a staged pattern: {flags:?}"
        );
    }
}
