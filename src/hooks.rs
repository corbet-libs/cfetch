//! Hook entrypoints. Contract: read stdin, act within the latency budget, emit
//! at most one JSON object, ALWAYS exit 0 — a hook failure must degrade to
//! silence, never to a broken session.

use std::path::Path;
use std::time::Duration;

use crate::config::Config;
use crate::hook_io::{Emit, HookEvent};
use crate::resident::SessionScope;
use crate::{
    condense, daemon, exhaust, fsutil, govern, heartbeat, ledger, paths, resident, session_state,
    transcript,
};

const DAEMON_BUDGET: Duration = Duration::from_millis(250);
/// Private full-output artifacts are a recovery path, not an unbounded log.
const CONDENSED_OUTPUT_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// Where this host books its ledger lines: the tree's log directory, this
/// host's identity, and the writer-side byte cap. Derived from the config,
/// with working defaults when the config is the thing that broke — booking is
/// a fact of record and must not stop because a setting is unreadable.
struct LedgerSink {
    dir: std::path::PathBuf,
    host: String,
    cap: u64,
}

impl LedgerSink {
    fn of(cfg: Option<&Config>) -> LedgerSink {
        let defaults = Config::default();
        LedgerSink {
            dir: paths::logs_dir(cfg.map_or(&defaults.brain_root, |c| &c.brain_root)),
            host: paths::host_id(),
            cap: cfg.map_or(defaults.ledger_max_bytes, |c| c.ledger_max_bytes),
        }
    }

    fn book(&self, session: &str, source: &str, chars: usize) {
        ledger::book_injection(&self.dir, &self.host, self.cap, session, source, chars);
    }

    fn book_measured(&self, session: &str, usage: &ledger::MeasuredUsage) {
        ledger::book_measured(&self.dir, &self.host, self.cap, session, usage);
    }

    fn book_self_read(&self, session: &str, chars: usize) {
        ledger::book_self_read(&self.dir, &self.host, self.cap, session, chars);
    }

    fn book_condensation(&self, session: &str, original_chars: usize, entered_chars: usize) {
        ledger::book_condensation(
            &self.dir,
            &self.host,
            self.cap,
            session,
            original_chars,
            entered_chars,
        );
    }
}

/// Files above this byte size get symbol-slice hints at pre-read. A
/// metadata-only gate: the hook path must never read file bodies (a full
/// read to count lines measured 292ms on a big file and is unbounded on
/// NFS). Exact line counts, if ever wanted, come from the index at scan
/// time — never from the hook path.
const LARGE_FILE_BYTES: u64 = 16 * 1024;
/// At most this many slice hints per advisory — an index dump is not a hint.
const MAX_SLICE_HINTS: usize = 5;
/// At most this many modified paths are named in the post-compaction recap.
/// The recap only earns its tokens while it stays cheaper than re-deriving the
/// same facts, so a session that touched a hundred files gets a count instead.
const COMPACT_RECAP_MAX_FILES: usize = 12;
/// Once-per-session key for the redirect away from whole-file brain reads. It
/// shares the reminder queue's shown-key set on purpose, so the same advice
/// cannot arrive twice by two different routes.
const SELF_READ_REDIRECT_KEY: &str = "self-read-redirect";

/// Dispatches a hook event by name. Never returns an error to the harness.
pub fn run(event_name: &str, agent_hint: Option<&str>) {
    let event = HookEvent::from_stdin();
    let result = match event_name {
        "session-start" => session_start(&event, agent_hint),
        "user-prompt" => user_prompt(&event, agent_hint),
        "pre-tool" => pre_tool(&event),
        "post-tool" => post_tool(&event),
        "stop" => stop(&event),
        "precompact" => precompact(&event),
        other => Err(anyhow::anyhow!("unknown hook: {other}")),
    };
    match result {
        Ok(()) => heartbeat::record_ok(event_name),
        Err(e) => heartbeat::record_error(event_name, &e.to_string()),
    }
    // Exit 0 unconditionally — see module doc.
}

/// Direct-read fallback with a hard deadline: the tree may be NFS, and a hung
/// mount must not eat the whole hook timeout. The worker thread is detached on
/// overrun.
fn resident_with_deadline(cfg: &Config, scope: &SessionScope) -> String {
    let cfg = cfg.clone();
    let scope = scope.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(resident::build(&cfg, &scope).text);
    });
    rx.recv_timeout(Duration::from_secs(2)).unwrap_or_default()
}

/// The state half of SessionStart: decides from the start reason whether the
/// arriving conversation inherits this session's records or starts without
/// them, and clears the file when it does not. Split out from the injection
/// half so the decision can be exercised without a config or a daemon.
fn session_start_state(state_dir: &Path, event: &HookEvent) -> session_state::StartKind {
    let kind = session_state::StartKind::parse(event.start_reason());
    // The session id cannot make this call: the harness keeps one id across
    // compaction and resume, so the file routinely outlives the conversation
    // that wrote it. Only the reason distinguishes the two.
    if kind.resets_state() {
        let _ = session_state::update(state_dir, event.session(), |st| st.reset());
    }
    kind
}

fn session_start(event: &HookEvent, agent_hint: Option<&str>) -> anyhow::Result<()> {
    // Subagents inherit fresh context on purpose; the resident set is for the
    // primary session (a fork re-injecting rings would double-pay the budget).
    if event.is_subagent() {
        return Ok(());
    }

    // Before anything is injected: a session start that begins a new
    // conversation must not leave the previous one's read history and spent
    // reminder caps standing for the rest of this session.
    let state_dir = paths::state_dir();
    let start = session_start_state(&state_dir, event);

    let mut emit = Emit::new("SessionStart");
    let runtime_chars = add_codex_runtime_notice(&mut emit, &state_dir, event, agent_hint, true);

    // Everything below the digest is CONFIG-INDEPENDENT and must reach the
    // model even when the config is the thing that broke — otherwise the one
    // surface built to announce breakage is suppressed by the breakage.
    let cfg = Config::load();
    // Which entries this session is entitled to is decided from the session
    // itself — the machine and the directory the agent was started in.
    let scope = SessionScope::from_event(event);
    let digest = match &cfg {
        Ok(cfg) => {
            // Prefer the warm daemon; fall back to a bounded direct read —
            // session start works with no daemon at all. The daemon shares
            // the host but not the cwd, so the scope travels with the call.
            let req = serde_json::json!({ "op": "resident", "cwd": event.cwd });
            match daemon::call_req(&req, DAEMON_BUDGET) {
                Some(r) if r.ok => r.digest.unwrap_or_default(),
                _ => resident_with_deadline(cfg, &scope),
            }
        }
        Err(_) => String::new(),
    };

    if !digest.is_empty() {
        match start.continuation_label() {
            // Memory arriving in the middle of a conversation needs its cause
            // named, or the model reads the repeat as new instruction.
            Some(reason) => emit.add_context(format!(
                "[cfetch resident memory (rings 0-1), re-injected after {reason}]\n{digest}"
            )),
            None => emit.add_context(format!("[cfetch resident memory (rings 0-1)]\n{digest}")),
        }
    }

    // After the digest, never before it: rings 0-1 outrank anything a single
    // session did. Its length is kept so the ledger can book it as itself —
    // a block this size hidden inside the digest's line would make the
    // digest look like it grew.
    let recap = compact_recap_for(&state_dir, event, start);
    let recap_chars = recap.as_ref().map_or(0, String::len);
    if let Some(recap) = recap {
        emit.add_context(recap);
    }

    if let Err(e) = &cfg {
        emit.add_context(format!(
            "[cfetch degraded: config unusable ({e}) — memory injection disabled; run `cfetch selfcheck`]"
        ));
    }
    let degraded = heartbeat::degraded();
    if !degraded.is_empty() {
        let names: Vec<String> = degraded.iter().map(|(n, _)| n.clone()).collect();
        emit.add_context(format!(
            "[cfetch degraded: hook(s) {} failing repeatedly — memory capture may be incomplete; run `cfetch status`]",
            names.join(", ")
        ));
    }

    let emitted = emit.finish();
    let sink = LedgerSink::of(cfg.as_ref().ok());
    sink.book(event.session(), "compact-recap", recap_chars);
    sink.book(event.session(), "runtime-status", runtime_chars);
    sink.book(
        event.session(),
        "resident-digest",
        emitted
            .saturating_sub(recap_chars)
            .saturating_sub(runtime_chars),
    );
    // The config failure still counts as a hook failure for the heartbeat.
    cfg.map(|_| ())
}

/// UserPromptSubmit: drains the reminder queue onto the prompt — the
/// zero-extra-turn delivery channel for everything queued at Stop and by the
/// cadence counter.
fn user_prompt(event: &HookEvent, agent_hint: Option<&str>) -> anyhow::Result<()> {
    let cfg = Config::load().ok();
    user_prompt_drain(
        &paths::state_dir(),
        &LedgerSink::of(cfg.as_ref()),
        event,
        agent_hint,
    )
}

fn user_prompt_drain(
    state_dir: &Path,
    sink: &LedgerSink,
    event: &HookEvent,
    agent_hint: Option<&str>,
) -> anyhow::Result<()> {
    // Reminders describe the primary session's own activity; a subagent
    // prompt must never receive them — nor consume them out of the queue.
    if event.is_subagent() {
        return Ok(());
    }
    let reminders = session_state::update(state_dir, event.session(), |st| st.drain_reminders())
        .unwrap_or_default();
    let mut emit = Emit::new("UserPromptSubmit");
    let runtime_chars = add_codex_runtime_notice(&mut emit, state_dir, event, agent_hint, false);
    for r in reminders {
        emit.add_context(r);
    }
    // ONE JSON object regardless of how many reminders were queued.
    let emitted = emit.finish();
    sink.book(event.session(), "runtime-status", runtime_chars);
    sink.book(
        event.session(),
        "reminders",
        emitted.saturating_sub(runtime_chars),
    );
    Ok(())
}

fn is_codex_event(event: &HookEvent, agent_hint: Option<&str>) -> bool {
    if let Some(agent) = agent_hint {
        return agent.eq_ignore_ascii_case(agent_session::AGENT_CODEX);
    }
    event
        .transcript_path
        .as_deref()
        .and_then(|path| agent_session::agent_source_for_path(Path::new(path)))
        == Some(agent_session::AGENT_CODEX)
}

fn add_codex_runtime_notice(
    emit: &mut Emit,
    state_dir: &Path,
    event: &HookEvent,
    agent_hint: Option<&str>,
    session_start: bool,
) -> usize {
    if !is_codex_event(event, agent_hint) {
        return 0;
    }
    let status = crate::runtime_status::load_cached();
    let notice = session_state::update(state_dir, event.session(), |state| {
        crate::runtime_status::codex_hook_notice(
            state,
            &status,
            session_start,
            crate::runtime_status::now(),
        )
    })
    .unwrap_or_default();
    if let Some(message) = notice.system_message {
        emit.system_message(message);
    }
    let context_chars = notice.additional_context.as_ref().map_or(0, String::len);
    if let Some(context) = notice.additional_context {
        emit.add_context(context);
    }
    context_chars
}

/// Stop-side reminder producers (wt/govern): QUEUE only, never emit — a
/// Stop-level injection forces a whole extra model turn. Runs after the
/// capture half so candidates the traps just flagged are already countable.
fn stop_govern(state_dir: &Path, cfg: &Config, event: &HookEvent) -> anyhow::Result<()> {
    if !cfg.governance.enabled || event.is_subagent() {
        return Ok(());
    }
    let _ = session_state::update(state_dir, event.session(), |st| {
        govern::queue_status_nudge(st, &cfg.brain_root);
        govern::queue_staging_visibility(st, &paths::staging_dir(&cfg.brain_root));
    });
    Ok(())
}

/// Cadence re-injection (wt/govern): every `reinject_every`-th post-tool
/// event queues the top ring-0 rules for the next user prompt. Queued here,
/// delivered at UserPromptSubmit — never at Stop.
fn post_tool_cadence(state_dir: &Path, cfg: &Config, event: &HookEvent) -> anyhow::Result<()> {
    if !cfg.governance.enabled || event.is_subagent() || event.tool_name.is_none() {
        return Ok(());
    }
    let rules = govern::top_ring0_rules(cfg, &SessionScope::from_event(event));
    // The refresh re-injects ring-0 rules, so it obeys the same per-entry
    // scope SessionStart did — a file this session never received must not
    // arrive by the back door 25 tool calls later.
    let _ = session_state::update(state_dir, event.session(), |st| {
        if let Some(n) = st.count_tool_event(cfg.governance.reinject_every)
            && let Some(rules) = rules
        {
            st.queue_reminder(&format!("rules-{n}"), &rules);
        }
    });
    Ok(())
}

/// Ring-6 exhaust capture (from wt/capture). One `O_APPEND` line into this
/// host's stream in the tree: no daemon, no fsync, no read — the whole write
/// path is a single short append, so it stays inside the hook latency budget
/// even when the tree is a network mount. Emits nothing.
fn post_tool_capture(cfg: &Config, event: &HookEvent) -> anyhow::Result<()> {
    if !cfg.capture.enabled {
        return Ok(());
    }
    exhaust::Exhaust::from_config(cfg).capture_post_tool(event, &cfg.brain_root, &cfg.rings())
}

fn prune_condensed_outputs(dir: &Path, keep: &Path, max_bytes: u64) -> anyhow::Result<()> {
    let mut files = Vec::new();
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry.path().extension().and_then(|ext| ext.to_str()) != Some("txt")
            || !entry.file_type()?.is_file()
        {
            continue;
        }
        let metadata = entry.metadata()?;
        total = total.saturating_add(metadata.len());
        files.push((
            metadata
                .modified()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
            entry.path(),
            metadata.len(),
        ));
    }
    files.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    for (_, path, len) in files {
        if total <= max_bytes {
            break;
        }
        if path == keep {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => total = total.saturating_sub(len),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

/// A rewritten tool result, carried together with the size of the output it
/// replaced. The pair is taken HERE because here is the only place both sides
/// exist at once: once the replacement is emitted, the original is gone and
/// any savings figure would have to be reconstructed from a ratio.
struct Rewrite {
    /// What enters the conversation instead of the tool result.
    text: String,
    /// Characters the untouched result would have entered with.
    original_chars: usize,
}

/// Condenses a completed Bash result for a harness whose PostToolUse carries
/// the tool response as an OBJECT — Claude Code and every client that copied
/// its hook envelope (Gemini, Qwen). Codex is the exception: it sends a bare
/// string and takes its replacement through `continue:false` instead.
///
/// The returned value MIRRORS the received response and changes only `stdout`.
/// A fresh object of the apparently-right shape is silently discarded, and so
/// is a value that drops a field the tool emitted — see
/// [`crate::hook_io::HookSpecificOutput::updated_tool_output`]. stderr is
/// never touched: an error must reach the model verbatim.
fn object_condensed_output(
    state_dir: &Path,
    event: &HookEvent,
) -> anyhow::Result<Option<(serde_json::Value, usize)>> {
    // Codex is identified by turn_id plus a string response; this path is for
    // everyone else, so both must be absent.
    if event.turn_id.is_some() || event.tool_name.as_deref() != Some("Bash") {
        return Ok(None);
    }
    let Some(response) = event
        .tool_response
        .as_ref()
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(None);
    };
    let Some(stdout) = response.get("stdout").and_then(serde_json::Value::as_str) else {
        return Ok(None);
    };
    let Some(command) = event
        .tool_input
        .as_ref()
        .and_then(|input| input.get("command"))
        .and_then(serde_json::Value::as_str)
    else {
        return Ok(None);
    };
    if u64::try_from(stdout.len()).unwrap_or(u64::MAX) > CONDENSED_OUTPUT_MAX_BYTES {
        return Ok(None);
    }
    let Some(rewrite) = preserve_full_output(state_dir, event, command, stdout)? else {
        return Ok(None);
    };
    let mut replacement = response.clone();
    replacement.insert(
        "stdout".to_string(),
        serde_json::Value::String(rewrite.text),
    );
    // The original size travels with the replacement: once emitted, the
    // original is gone, and a savings figure reconstructed later is a ratio
    // rather than a measurement.
    Ok(Some((
        serde_json::Value::Object(replacement),
        rewrite.original_chars,
    )))
}

/// Condenses a completed Codex Bash result and preserves the full original in
/// private local state. Current Codex PostToolUse input is identifiable by the
/// required `turn_id` plus a string `tool_response`; Claude does not get the
/// `continue:false` replacement output because it interprets that universal
/// field as an instruction to stop the agent.
fn codex_condensed_output(state_dir: &Path, event: &HookEvent) -> anyhow::Result<Option<Rewrite>> {
    if event.turn_id.is_none() || event.tool_name.as_deref() != Some("Bash") {
        return Ok(None);
    }
    let Some(command) = event
        .tool_input
        .as_ref()
        .and_then(|input| input.get("command"))
        .and_then(serde_json::Value::as_str)
    else {
        return Ok(None);
    };
    let Some(output) = event
        .tool_response
        .as_ref()
        .and_then(serde_json::Value::as_str)
    else {
        return Ok(None);
    };
    if u64::try_from(output.len()).unwrap_or(u64::MAX) > CONDENSED_OUTPUT_MAX_BYTES {
        return Ok(None);
    }
    preserve_full_output(state_dir, event, command, output)
}

/// Condense, write the original to private local state, and return the
/// condensed text with a pointer to it. `None` when condensation was not
/// worthwhile — the caller then leaves the output alone.
///
/// Shared by every harness: the decision of WHAT to condense and what to keep
/// must not vary by client, or the same command yields different context
/// depending on who asked.
fn preserve_full_output(
    state_dir: &Path,
    event: &HookEvent,
    command: &str,
    output: &str,
) -> anyhow::Result<Option<Rewrite>> {
    use sha2::Digest as _;

    let condensed = condense::condense(command, output);
    if !condensed.was_condensed() {
        return Ok(None);
    }
    let artifact_identity = format!(
        "{}\0{}\0{}\0{}",
        event.session(),
        event.turn_id.as_deref().unwrap_or_default(),
        event.tool_use_id.as_deref().unwrap_or_default(),
        output
    );
    let id = crate::hashing::hex_lower(sha2::Sha256::digest(artifact_identity.as_bytes()));
    let path = state_dir.join("condensed-output").join(format!("{id}.txt"));
    let preserved = format!(
        "command: {}\nsession: {}\ntool_use_id: {}\n\n{}",
        exhaust::redact_secrets(command),
        event.session(),
        event.tool_use_id.as_deref().unwrap_or("unknown"),
        output
    );
    fsutil::atomic_write(&path, preserved)?;
    prune_condensed_outputs(
        path.parent().expect("condensed output has a parent"),
        &path,
        CONDENSED_OUTPUT_MAX_BYTES,
    )?;
    Ok(Some(Rewrite {
        text: format!(
            "{}\n\n[cfetch: full uncondensed output preserved at {}]",
            condensed.text,
            path.display()
        ),
        original_chars: output.len(),
    }))
}

/// Books one rewrite: the replacement's own cost as an injection, and — in the
/// same call, from the same two measurements — what the rewrite avoided.
/// Deliberately one function: a savings line that can outlive its cost line is
/// how a memory system starts believing its own brochure.
fn book_rewrite(sink: &LedgerSink, session: &str, original_chars: usize, entered_chars: usize) {
    sink.book(session, "output-condensation", entered_chars);
    sink.book_condensation(session, original_chars, entered_chars);
}

/// Turn summary + the 6->5 flagging traps (from wt/capture). Emits nothing —
/// Stop-level additionalContext would force an extra model turn, and ring
/// 5/6 content is never injected anyway.
fn stop_capture(cfg: &Config, event: &HookEvent) -> anyhow::Result<()> {
    if !cfg.capture.enabled {
        return Ok(());
    }
    // Upgrade work is forbidden here. A migration can legitimately take
    // longer than the harness deadline; if the harness kills it before its
    // completion marker, every later Stop retries from zero and the hook
    // livelocks. `cfetch install` owns one-time state conversion instead.
    exhaust::Exhaust::from_config(cfg).record_stop(event.session())
}

/// The Read tool's target file, if this is a Read invocation we track.
fn read_target(input: &serde_json::Value) -> Option<&str> {
    input.get("file_path")?.as_str()
}

/// A ranged read (offset/limit) is already narrow: it triggers no advice and
/// is never recorded — a ranged read must not mark a later full read as
/// duplicate.
fn is_ranged(input: &serde_json::Value) -> bool {
    input.get("offset").is_some() || input.get("limit").is_some()
}

/// A file read visible through either Claude's structured Read tool or
/// Codex's shell tool. The shell subset is deliberately strict: only a single
/// `cat`, `head`, `tail`, or `sed` invocation with one final file operand is
/// Does this shell command write something? Deliberately the MIRROR of the
/// read parser's caution: that one refuses anything ambiguous because it names
/// a path, while this one only ever increments a counter, so a false negative
/// (an unrecognized write) costs a missed tally and a false positive would
/// invent activity that never happened. Recognized: an unquoted `>`/`>>`
/// redirect, and a leading program whose whole purpose is to modify the
/// filesystem. `sed`/`perl` count only with an in-place flag.
fn is_shell_write(command: &str) -> bool {
    if has_unquoted_redirect(command) {
        return true;
    }
    // Any segment of a pipeline or list may be the writer.
    // Segment separators shared by bash, zsh and fish. fish spells its
    // conjunctions `and`/`or` as well as `&&`/`||`, but both reduce to
    // whitespace-separated segments once split on these.
    command.split(['|', ';', '\n', '&']).any(|segment| {
        let Some((prog, mut words)) = condense::leading_program(segment) else {
            return false;
        };
        match prog {
            "tee" | "cp" | "mv" | "rm" | "rmdir" | "mkdir" | "touch" | "install" | "dd"
            | "truncate" | "ln" | "chmod" | "chown" | "patch" | "rsync" | "shred" | "unlink" => {
                true
            }
            // In-place only: a plain `sed`/`perl` filters to stdout.
            "sed" | "perl" => words.any(|w| w == "-i" || w.starts_with("-i.") || w == "--in-place"),
            _ => false,
        }
    })
}

/// `>` or `>>` outside single or double quotes. A quoted `>` is data — most
/// commonly a here-doc body or an echoed string — not a redirect.
fn has_unquoted_redirect(command: &str) -> bool {
    let (mut single, mut double, mut escaped) = (false, false, false);
    for c in command.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if !single => escaped = true,
            '\'' if !double => single = !single,
            '"' if !single => double = !double,
            '>' if !single && !double => return true,
            _ => {}
        }
    }
    false
}

/// recognized, never a pipeline, redirect, substitution, or compound command.
fn read_invocation(event: &HookEvent) -> Option<(String, bool)> {
    let tool = event.tool_name.as_deref()?;
    let input = event.tool_input.as_ref()?;
    if tool == "Read" {
        return Some((
            resolve_event_path(event, read_target(input)?),
            is_ranged(input),
        ));
    }
    if tool != "Bash" {
        return None;
    }
    let command = input.get("command")?.as_str()?;
    let words = exhaust::shell_words(command);
    if words.len() < 2
        || words
            .iter()
            .any(|w| w.contains([';', '|', '>', '<', '`']) || w.contains("&&") || w.contains("$("))
    {
        return None;
    }
    let program = Path::new(&words[0]).file_name()?.to_str()?;
    let target = words.last()?;
    if target == "-" || target.starts_with('-') {
        return None;
    }
    let ranged = match program {
        "cat" => {
            if !words[1..words.len() - 1].iter().all(|w| w.starts_with('-')) {
                return None;
            }
            false
        }
        "head" | "tail" => {
            let middle = &words[1..words.len() - 1];
            let mut i = 0;
            while i < middle.len() {
                let option = &middle[i];
                if matches!(option.as_str(), "-n" | "-c" | "--lines" | "--bytes") {
                    i += 1;
                    if i >= middle.len() || !middle[i].chars().all(|c| c.is_ascii_digit()) {
                        return None;
                    }
                } else if !option.starts_with('-') {
                    return None;
                }
                i += 1;
            }
            true
        }
        "sed"
            if words.len() == 3
                || (words.len() == 4 && matches!(words[1].as_str(), "-n" | "-e")) =>
        {
            true
        }
        _ => return None,
    };
    Some((resolve_event_path(event, target), ranged))
}

fn resolve_event_path(event: &HookEvent, path: &str) -> String {
    let path = Path::new(path);
    if path.is_absolute() {
        return path.to_string_lossy().into_owned();
    }
    event
        .cwd
        .as_deref()
        .map(|cwd| Path::new(cwd).join(path).to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

/// Metadata-only size gate for the slice-hint advisory. `false` when the
/// file is unstattable — a missed hint beats a broken hook.
fn is_large_file(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.len() > LARGE_FILE_BYTES)
}

/// Read-only code-index query: named slices of `file_path`, largest first,
/// formatted `name start-end`. ANY failure (no DB yet, WAL sidecars
/// unreadable, schema drift) yields no hints — a missed optimization, never a
/// broken hook. The read-only open also guarantees the hook cannot contend
/// with the indexer for the write lock. Also serves the daemon's `slices` op.
pub(crate) fn symbol_slices(db: &Path, file_path: &str, limit: usize) -> Vec<String> {
    use rusqlite::OpenFlags;
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let Ok(conn) = rusqlite::Connection::open_with_flags(db, flags) else {
        return Vec::new();
    };
    let Ok(mut stmt) = conn.prepare(
        "SELECT s.name, s.start_line, s.end_line
         FROM symbols s JOIN code_files f ON f.id = s.file_id
         WHERE f.path = ?1
         ORDER BY (s.end_line - s.start_line) DESC, s.start_line ASC
         LIMIT ?2",
    ) else {
        return Vec::new();
    };
    stmt.query_map(rusqlite::params![file_path, limit as i64], |r| {
        Ok(format!(
            "{} {}-{}",
            r.get::<_, String>(0)?,
            r.get::<_, i64>(1)?,
            r.get::<_, i64>(2)?
        ))
    })
    .map(|rows| rows.filter_map(Result::ok).collect())
    .unwrap_or_default()
}

/// Meets a whole-file read of an oversized brain file with the cheap path,
/// once per session. The threshold is the SAME budget the write side enforces
/// (see [`post_tool_budget`]): a file too big to sit in every session is too
/// big to slurp whole here, and one number governing both ends means an
/// operator who tunes it tunes the whole policy. Once per session and not per
/// file — this is a habit worth naming once; a hook that repeats itself gets
/// switched off.
fn brain_read_redirect(
    cfg: Option<&Config>,
    state_dir: &Path,
    event: &HookEvent,
    path: &str,
) -> Option<String> {
    let cfg = cfg?;
    let budget = cfg.governance.state_file_budget_tokens;
    if budget == 0 {
        return None;
    }
    let name = Path::new(path)
        .strip_prefix(&cfg.brain_root)
        .ok()?
        .display()
        .to_string();
    let len = std::fs::metadata(path).ok()?.len();
    let tokens = crate::hook_io::estimate_tokens(usize::try_from(len).unwrap_or(usize::MAX));
    if tokens <= budget {
        return None;
    }
    // Claim the key only once the read is genuinely worth redirecting, so a
    // session's one shot is not spent on a file that was fine to read.
    let claimed = session_state::update(state_dir, event.session(), |st| {
        st.shown_keys.insert(SELF_READ_REDIRECT_KEY.to_string())
    })?;
    if !claimed {
        return None;
    }
    Some(format!(
        "[cfetch: {name} is a brain file of ~{tokens} tokens, over the {budget}-token budget for one file — `cfetch recall <terms>` answers out of the whole brain for a fraction of that and `cfetch recall --id <cite>` expands a single statement; read it whole only when you need all of it]"
    ))
}

/// PreToolUse: read hygiene advice at the moment of spend. Repeat-read
/// advisory (once per file per session, mtime-gated, disarmed by compaction)
/// plus symbol-slice hints for large indexed files.
/// The content a pending tool call would put into the world: a write's body,
/// or a shell command. Both are checked against declared prohibitions, because
/// a standing constraint is usually broken by a COMMAND rather than by prose —
/// `zfs set dedup=on` is typed, not written into a document.
fn pending_content(event: &HookEvent) -> Option<String> {
    let input = event.tool_input.as_ref()?;
    let field = match event.tool_name.as_deref()? {
        "Bash" => "command",
        "Write" => "content",
        "Edit" | "MultiEdit" | "NotebookEdit" => "new_string",
        "apply_patch" => "command",
        _ => return None,
    };
    input
        .get(field)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

/// Names a declared prohibition the pending call would violate, at the moment
/// it would be violated. Advisory, never a block: the rules are the operator's
/// and so is the decision to override one — but silence at the write site is
/// how a standing constraint gets broken by an agent that read it 200 messages
/// ago. At most once per rule per session, keyed through the reminder set so
/// the same advice cannot arrive twice by two routes.
fn prohibition_warning(
    cfg: Option<&Config>,
    state_dir: &Path,
    event: &HookEvent,
) -> Option<String> {
    let cfg = cfg?;
    let content = pending_content(event)?;
    if content.trim().is_empty() {
        return None;
    }
    let rules = govern::prohibitions(cfg, &SessionScope::from_event(event));
    let (hit, pattern) = govern::first_violation(&content, &rules)?;
    let key = format!("forbids:{}", hit.rule);
    let text = format!(
        "[cfetch: this would use `{pattern}`, which {} forbids — read that rule before proceeding]",
        hit.rule
    );
    let mut fired = false;
    let _ = session_state::update(state_dir, event.session(), |st| {
        fired = st.queue_reminder(&key, &text);
    });
    // The queue delivers at the next prompt; the write is happening NOW, so the
    // text is returned for immediate emission and the queue entry only holds
    // the once-per-rule key.
    fired.then_some(text)
}

fn pre_tool(event: &HookEvent) -> anyhow::Result<()> {
    // Subagents run in fresh context: the "earlier content" a warning points
    // at does not exist there. Never warn subagents.
    if event.is_subagent() {
        return Ok(());
    }
    let cfg_for_rules = Config::load().ok();
    let state_dir_for_rules = paths::state_dir();
    if let Some(warning) = prohibition_warning(cfg_for_rules.as_ref(), &state_dir_for_rules, event)
    {
        let mut emit = Emit::new("PreToolUse");
        emit.add_context(warning);
        let emitted = emit.finish();
        LedgerSink::of(cfg_for_rules.as_ref()).book(event.session(), "prohibition", emitted);
        return Ok(());
    }
    let Some((path, ranged)) = read_invocation(event) else {
        return Ok(());
    };
    if ranged {
        return Ok(());
    }

    let cfg = Config::load().ok();
    let state_dir = paths::state_dir();
    let mut emit = Emit::new("PreToolUse");
    let mtime = session_state::file_mtime(Path::new(&path));
    let hints = if is_large_file(Path::new(&path)) {
        symbol_slices(&state_dir.join("index.db"), &path, MAX_SLICE_HINTS)
    } else {
        Vec::new()
    };
    let (warn, show_hints) = session_state::update(&state_dir, event.session(), |st| {
        let warn = mtime.is_some_and(|mtime| st.should_warn_repeat_read(&path, mtime));
        let show_hints = !hints.is_empty() && st.should_hint_slices(&path);
        (warn, show_hints)
    })
    .unwrap_or((false, false));

    if warn {
        emit.add_context(format!(
            "[cfetch: {path} was already read this session (unchanged); prefer the earlier content or a narrower slice]"
        ));
    }
    if show_hints {
        emit.add_context(format!(
            "[cfetch: {path} is >{}KB; known symbol slices: {}]",
            LARGE_FILE_BYTES / 1024,
            hints.join(", ")
        ));
    }
    if let Some(redirect) = brain_read_redirect(cfg.as_ref(), &state_dir, event, &path) {
        emit.add_context(redirect);
    }

    let emitted = emit.finish();
    if emitted > 0 {
        // Book our own injection — advice that costs tokens is not free.
        LedgerSink::of(cfg.as_ref()).book(event.session(), "read-advisory", emitted);
    }
    Ok(())
}

/// PostToolUse: ALL halves run — ring-6 exhaust capture (wt/capture), cadence
/// counting (wt/govern), and session read/write tracking (wt/account). A
/// failure in one half must not starve the others; the first error is still
/// reported to the heartbeat.
fn post_tool(event: &HookEvent) -> anyhow::Result<()> {
    let mut first_err: Option<anyhow::Error> = None;
    let state_dir = paths::state_dir();
    let rewrite = match codex_condensed_output(&state_dir, event) {
        Ok(rewrite) => rewrite,
        Err(e) => {
            first_err = Some(e);
            None
        }
    };
    if event.tool_name.is_some() {
        match Config::load() {
            Ok(cfg) => {
                if let Err(e) = post_tool_capture(&cfg, event) {
                    first_err = Some(e);
                }
                if let Err(e) = post_tool_cadence(&state_dir, &cfg, event) {
                    first_err.get_or_insert(e);
                }
                if let Err(e) = post_tool_budget(&state_dir, &cfg, event) {
                    first_err.get_or_insert(e);
                }
                post_tool_self_read(&LedgerSink::of(Some(&cfg)), &cfg, event);
            }
            Err(e) => first_err = Some(e),
        }
    }
    if let Err(e) = post_tool_track(event) {
        first_err.get_or_insert(e);
    }
    // Exactly one of these can apply: Codex is identified by turn_id plus a
    // string response and takes `continue:false`; every object-shaped harness
    // takes hookSpecificOutput.updatedToolOutput instead. Both are booked the
    // same way — a saving is only defensible where both sides were seen.
    let object_rewrite = match object_condensed_output(&state_dir, event) {
        Ok(v) => v,
        Err(e) => {
            first_err.get_or_insert(e);
            None
        }
    };
    if rewrite.is_some() || object_rewrite.is_some() {
        let original_chars = rewrite
            .as_ref()
            .map(|r| r.original_chars)
            .or_else(|| object_rewrite.as_ref().map(|(_, chars)| *chars))
            .unwrap_or(0);
        let mut emit = Emit::new("PostToolUse");
        if let Some(rewrite) = rewrite {
            emit.replace_tool_output(rewrite.text);
        }
        if let Some((value, _)) = object_rewrite {
            emit.replace_claude_tool_output(value);
        }
        // What the harness actually received, not what condensation produced:
        // the pointer line rides along and is part of the bill.
        let entered = emit.finish();
        let cfg = Config::load().ok();
        book_rewrite(
            &LedgerSink::of(cfg.as_ref()),
            event.session(),
            original_chars,
            entered,
        );
    }
    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Session read/write tracking: full reads recorded with the file's mtime;
/// writes clear the read record (post-edit content is new content).
/// Paths this event wrote, across both shapes: Codex's `apply_patch` carries
/// them inside the patch body, the write tools name one directly.
fn written_targets(event: &HookEvent) -> Vec<String> {
    match event.tool_name.as_deref() {
        Some("apply_patch") => exhaust::written_paths(event),
        Some("Write" | "Edit" | "MultiEdit" | "NotebookEdit") => event
            .tool_input
            .as_ref()
            .and_then(|input| {
                input
                    .get("file_path")
                    .or_else(|| input.get("notebook_path"))
                    .and_then(|v| v.as_str())
            })
            .map(|p| vec![p.to_string()])
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// A brain file that outgrew its token budget costs every session that loads
/// it, for as long as it stays that size — so the warning belongs at the write,
/// where the author is still holding the context to act on it. One line, once
/// per file per session, never a block.
fn post_tool_budget(state_dir: &Path, cfg: &Config, event: &HookEvent) -> anyhow::Result<()> {
    if event.is_subagent() {
        return Ok(());
    }
    let budget = cfg.governance.state_file_budget_tokens;
    if budget == 0 {
        return Ok(());
    }
    for path in written_targets(event) {
        // Only the brain's own files: cfetch governs what it will be asked to
        // load, not the size of the user's source tree.
        if !Path::new(&path).starts_with(&cfg.brain_root) {
            continue;
        }
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        let tokens =
            crate::hook_io::estimate_tokens(usize::try_from(meta.len()).unwrap_or(usize::MAX));
        if tokens <= budget {
            continue;
        }
        let name = Path::new(&path)
            .strip_prefix(&cfg.brain_root)
            .unwrap_or(Path::new(&path))
            .display()
            .to_string();
        let _ = session_state::update(state_dir, event.session(), |st| {
            st.queue_reminder(
                &format!("budget:{path}"),
                &format!(
                    "[cfetch: {name} is ~{tokens} tokens, over the {budget}-token budget for one brain file — split it or distil it; every session that loads it pays this]"
                ),
            );
        });
    }
    Ok(())
}

/// Books what the agent spent reading brain files ITSELF. The ledger counted
/// every byte cfetch injected and none of the ones the agent fetched by hand
/// out of the same tree, which made the largest line item invisible precisely
/// where the memory system claims to help.
///
/// Whole-file reads only: a ranged read is the behaviour the redirect asks
/// for, and its true size is not knowable from a hook that must never read a
/// file body. Subagents are skipped for the reason measured usage skips them —
/// a fork's context is not this session's row.
fn post_tool_self_read(sink: &LedgerSink, cfg: &Config, event: &HookEvent) {
    if event.is_subagent() {
        return;
    }
    let Some((path, ranged)) = read_invocation(event) else {
        return;
    };
    if ranged || !Path::new(&path).starts_with(&cfg.brain_root) {
        return;
    }
    // The file's size at post-tool time, estimated the way every other ledger
    // line is: a metadata stat, never a read.
    let Ok(meta) = std::fs::metadata(&path) else {
        return;
    };
    sink.book_self_read(
        event.session(),
        usize::try_from(meta.len()).unwrap_or(usize::MAX),
    );
}

fn post_tool_track(event: &HookEvent) -> anyhow::Result<()> {
    if event.is_subagent() {
        return Ok(());
    }
    let (Some(tool), Some(input)) = (event.tool_name.as_deref(), event.tool_input.as_ref()) else {
        return Ok(());
    };
    let state_dir = paths::state_dir();
    match tool {
        "Read" | "Bash" => {
            if tool == "Bash"
                && input
                    .get("command")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(is_shell_write)
            {
                let _ = session_state::update(&state_dir, event.session(), |st| {
                    st.record_shell_write();
                });
            }
            let Some((path, ranged)) = read_invocation(event) else {
                return Ok(());
            };
            if ranged {
                return Ok(());
            }
            let Some(mtime) = session_state::file_mtime(Path::new(&path)) else {
                return Ok(());
            };
            let _ = session_state::update(&state_dir, event.session(), |st| {
                st.record_read(&path, mtime);
            });
        }
        "Write" | "Edit" | "MultiEdit" | "NotebookEdit" | "apply_patch" => {
            if tool == "apply_patch" {
                let paths = exhaust::written_paths(event);
                if paths.is_empty() {
                    return Ok(());
                }
                let _ = session_state::update(&state_dir, event.session(), |st| {
                    for path in paths {
                        st.record_write(&path);
                    }
                });
                return Ok(());
            }
            let target = input
                .get("file_path")
                .or_else(|| input.get("notebook_path"))
                .and_then(|v| v.as_str());
            let Some(path) = target else { return Ok(()) };
            let _ = session_state::update(&state_dir, event.session(), |st| {
                st.record_write(path);
            });
        }
        _ => {}
    }
    Ok(())
}

/// Stop: ALL halves — exhaust turn summary + traps (wt/capture), reminder
/// producers (wt/govern, after the traps so fresh flags count), and
/// measured-usage booking (wt/account).
fn stop(event: &HookEvent) -> anyhow::Result<()> {
    let mut first_err: Option<anyhow::Error> = None;
    let state_dir = paths::state_dir();
    let cfg = Config::load();
    match &cfg {
        Ok(cfg) => {
            if let Err(e) = stop_capture(cfg, event) {
                first_err = Some(e);
            }
            if let Err(e) = stop_govern(&state_dir, cfg, event) {
                first_err.get_or_insert(e);
            }
        }
        Err(e) => first_err = Some(anyhow::anyhow!("{e}")),
    }
    if let Err(e) = stop_measure(&LedgerSink::of(cfg.as_ref().ok()), event) {
        first_err.get_or_insert(e);
    }
    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Books measured transcript usage for this turn. The transcript's counters
/// are cumulative, so the ledger books only the delta above its watermark. An
/// unparseable transcript books NOTHING — status then labels the numbers
/// estimated instead of inventing measured zeros.
fn stop_measure(sink: &LedgerSink, event: &HookEvent) -> anyhow::Result<()> {
    // A subagent's usage belongs to its own transcript, not this ledger row.
    if event.is_subagent() {
        return Ok(());
    }
    let Some(tp) = event.transcript_path.as_deref() else {
        return Ok(());
    };
    let Some(usage) = transcript::scan(Path::new(tp)) else {
        return Ok(());
    };
    sink.book_measured(event.session(), &ledger::MeasuredUsage::from(&usage));
    Ok(())
}

/// Paths in the recap are shown relative to the session's directory when they
/// live under it: the same absolute prefix repeated on every line is context
/// the model already holds.
fn relative_to_cwd(path: &str, cwd: Option<&str>) -> String {
    cwd.map(Path::new)
        .and_then(|cwd| Path::new(path).strip_prefix(cwd).ok())
        .map_or_else(|| path.to_string(), |rel| rel.display().to_string())
}

/// What this session changed, for a conversation that has just lost its memory
/// of changing it.
///
/// Compaction is the one start where the repeat-read advisory is deliberately
/// switched OFF (`SessionState::compacted`) — the model no longer holds the
/// earlier reads, so warning it about them would be nonsense. That leaves the
/// summarized conversation with nothing standing between it and re-doing work
/// it already did, exactly when it is least able to notice. The written set is
/// the hook layer's own record, so this can be stated as fact rather than left
/// to the summary's recollection of it.
///
/// No snapshot is taken at PreCompact, because there is nothing to snapshot:
/// the session file already outlives the conversation (the harness keeps one
/// id across compaction, and `StartKind::Compact` deliberately does not reset),
/// so a copy would only be a staler second source for the same set.
///
/// Writes only, never reads: a repeated read costs tokens, a forgotten edit
/// costs correctness. Shell writes are COUNTED and not named, the same way
/// `record_shell_write` books them — a half-parsed redirect target asserted
/// here as a modified file would be worse than the bare tally.
fn compact_recap(st: &session_state::SessionState, cwd: Option<&str>) -> Option<String> {
    if st.written.is_empty() && st.shell_writes == 0 {
        return None;
    }
    let mut text = String::from(
        "[cfetch: what this session changed before compaction — the hook record, not the summary's recollection]",
    );
    for path in st.written.iter().take(COMPACT_RECAP_MAX_FILES) {
        text.push_str("\n- ");
        text.push_str(&relative_to_cwd(path, cwd));
    }
    let over_cap = st.written.len().saturating_sub(COMPACT_RECAP_MAX_FILES);
    if over_cap > 0 {
        text.push_str(&format!("\n- (+{over_cap} more)"));
    }
    if st.shell_writes > 0 {
        let shell = st.shell_writes;
        text.push_str(&format!(
            "\n- plus {shell} shell write(s), targets not recorded"
        ));
    }
    Some(text)
}

/// The recap, but only for the start that is owed one. Every other reason gets
/// nothing and pays no read: a start that begins a new conversation has just
/// had its records cleared, and a resume got the turns themselves back.
fn compact_recap_for(
    state_dir: &Path,
    event: &HookEvent,
    start: session_state::StartKind,
) -> Option<String> {
    if start != session_state::StartKind::Compact {
        return None;
    }
    compact_recap(
        &session_state::load(state_dir, event.session()),
        event.cwd.as_deref(),
    )
}

/// PreCompact: after compaction the model no longer remembers its earlier
/// reads, so repeat-read warnings are disarmed for the rest of the session.
fn precompact(event: &HookEvent) -> anyhow::Result<()> {
    if event.is_subagent() {
        return Ok(());
    }
    precompact_state(&paths::state_dir(), event);
    Ok(())
}

/// The state half of PreCompact, split out the way SessionStart's is, so the
/// two ends of the compaction contract — disarm here, hand the record back at
/// the next start — can be exercised together without the real state dir.
fn precompact_state(state_dir: &Path, event: &HookEvent) {
    let _ = session_state::update(state_dir, event.session(), |st| {
        st.compacted = true;
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn runtime_notices_are_codex_only() {
        let codex = HookEvent {
            transcript_path: Some("/tmp/.codex/sessions/2026/08/23/session.jsonl".into()),
            ..HookEvent::default()
        };
        let claude = HookEvent {
            transcript_path: Some("/tmp/.claude/projects/repo/session.jsonl".into()),
            ..HookEvent::default()
        };
        assert!(is_codex_event(&codex, None));
        assert!(!is_codex_event(&claude, None));
        assert!(is_codex_event(&HookEvent::default(), Some("codex")));
        assert!(!is_codex_event(&codex, Some("claude")));
    }

    /// The replacement must MIRROR the received response and change only
    /// stdout. A fresh object of the apparently-right shape is discarded by
    /// the harness, and silently — so every field the tool emitted, including
    /// ones cfetch does not understand, has to survive.
    #[test]
    fn the_replacement_mirrors_every_field_and_changes_only_stdout() {
        let state = tempfile::tempdir().unwrap();
        let flood = (0..400)
            .map(|i| format!("src/f{i}.rs:{i}: hit"))
            .collect::<Vec<_>>()
            .join("\n");
        let event: HookEvent = serde_json::from_value(serde_json::json!({
            "session_id": "s-obj",
            "tool_name": "Bash",
            "tool_use_id": "toolu_x",
            "tool_input": {"command": "grep -rn hit src/"},
            "tool_response": {
                "stdout": flood,
                "stderr": "a warning",
                "interrupted": false,
                "isImage": false,
                "some_future_field": {"nested": 1}
            }
        }))
        .unwrap();

        let (out, original) = object_condensed_output(state.path(), &event)
            .unwrap()
            .expect("condensed");
        assert_eq!(
            original,
            flood.len(),
            "the original size travels with the replacement"
        );
        let obj = out.as_object().unwrap();
        // Only stdout changed.
        assert_ne!(obj["stdout"].as_str().unwrap(), flood);
        assert!(obj["stdout"].as_str().unwrap().contains("preserved at"));
        // stderr is never touched — an error must reach the model verbatim.
        assert_eq!(obj["stderr"], "a warning");
        // Fields cfetch knows nothing about must survive, or the harness drops
        // the whole replacement.
        assert_eq!(obj["interrupted"], false);
        assert_eq!(obj["isImage"], false);
        assert_eq!(obj["some_future_field"]["nested"], 1);
    }

    /// Codex sends a bare string and takes continue:false; this path must not
    /// fire for it, or both channels would try to replace the same result.
    #[test]
    fn the_object_path_declines_codex_shaped_events() {
        let state = tempfile::tempdir().unwrap();
        let event: HookEvent = serde_json::from_value(serde_json::json!({
            "session_id": "s-codex",
            "turn_id": "t1",
            "tool_name": "Bash",
            "tool_input": {"command": "grep -rn x ."},
            "tool_response": "x".repeat(60_000),
        }))
        .unwrap();
        assert!(
            object_condensed_output(state.path(), &event)
                .unwrap()
                .is_none()
        );
    }

    /// The program parser selects the condensation strategy, so a prefix it
    /// does not know silently downgrades a known family to Generic. These are
    /// the forms zsh and fish users actually type.
    #[test]
    fn shell_prefixes_beyond_bash_still_find_the_program() {
        use crate::condense::{Family, classify};
        // zsh
        assert_eq!(classify("noglob cargo test"), Family::Verification);
        assert_eq!(classify("nocorrect cargo build"), Family::Verification);
        // fish and POSIX
        assert_eq!(classify("command cargo test"), Family::Verification);
        assert_eq!(classify("builtin cd /tmp"), classify("cd /tmp"));
        assert_eq!(classify("exec cargo test"), Family::Verification);
        // BSD sudo, and the scheduling wrappers
        assert_eq!(classify("doas cargo test"), Family::Verification);
        assert_eq!(classify("nice -n 19 cargo test"), Family::Verification);
        assert_eq!(classify("stdbuf -oL cargo test"), Family::Verification);
        // env assignment, still the subcommand off the same iterator
        assert_eq!(classify("RUST_LOG=debug cargo test"), Family::Verification);
    }

    #[test]
    fn shell_write_detection_survives_zsh_and_fish_forms() {
        // zsh clobber and both-stream redirects
        assert!(is_shell_write("cargo build >| out.log"));
        assert!(is_shell_write("cargo build &> out.log"));
        // fish conjunctions reduce to segments
        assert!(is_shell_write("mkdir -p /tmp/x && touch /tmp/x/a"));
        // prefixes the shared parser now steps over
        assert!(is_shell_write("doas rm -rf /tmp/x"));
        assert!(is_shell_write("command cp a b"));
        assert!(is_shell_write("noglob mv a b"));
        // still not fooled by quoted redirects or read-only commands
        assert!(!is_shell_write("echo \'a > b\'"));
        assert!(!is_shell_write("command cat file"));
    }

    fn budget_event(session: &str, path: &std::path::Path) -> HookEvent {
        serde_json::from_value(serde_json::json!({
            "session_id": session,
            "tool_name": "Write",
            "tool_input": {"file_path": path.to_string_lossy()},
        }))
        .unwrap()
    }

    #[test]
    fn an_oversized_brain_file_is_named_once_with_its_size_and_a_remedy() {
        let brain = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let big = brain.path().join("knowledge/huge.md");
        std::fs::create_dir_all(big.parent().unwrap()).unwrap();
        std::fs::write(&big, "x".repeat(40_000)).unwrap();
        let cfg = Config {
            brain_root: brain.path().to_path_buf(),
            ..Config::default()
        };
        let event = budget_event("s-budget", &big);

        post_tool_budget(state.path(), &cfg, &event).unwrap();
        // Drain the way UserPromptSubmit does — through `update`, so the
        // shown-key survives. Draining a loaded copy would throw it away and
        // the once-per-file guarantee would look broken when it is not.
        let mut texts = Vec::new();
        let _ = crate::session_state::update(state.path(), "s-budget", |st| {
            texts = st.drain_reminders();
        });
        assert_eq!(texts.len(), 1, "exactly one warning: {texts:?}");
        assert!(texts[0].contains("knowledge/huge.md"), "{}", texts[0]);
        assert!(
            texts[0].contains("tokens"),
            "the size must be stated: {}",
            texts[0]
        );
        assert!(
            texts[0].contains("split it or distil it"),
            "a remedy is required: {}",
            texts[0]
        );

        // Same file again in the same session must not warn twice.
        post_tool_budget(state.path(), &cfg, &event).unwrap();
        let mut again = Vec::new();
        let _ = crate::session_state::update(state.path(), "s-budget", |st| {
            again = st.drain_reminders();
        });
        assert!(
            again.is_empty(),
            "the warning fired twice for one file: {again:?}"
        );
    }

    #[test]
    fn a_small_file_and_a_file_outside_the_brain_are_both_silent() {
        let brain = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let cfg = Config {
            brain_root: brain.path().to_path_buf(),
            ..Config::default()
        };

        let small = brain.path().join("small.md");
        std::fs::write(&small, "x".repeat(100)).unwrap();
        post_tool_budget(state.path(), &cfg, &budget_event("s-small", &small)).unwrap();
        assert!(
            crate::session_state::load(state.path(), "s-small")
                .drain_reminders()
                .is_empty()
        );

        // cfetch governs what it will be asked to load, not the user's source.
        let outside = tempfile::tempdir().unwrap();
        let theirs = outside.path().join("vendor.rs");
        std::fs::write(&theirs, "x".repeat(40_000)).unwrap();
        post_tool_budget(state.path(), &cfg, &budget_event("s-out", &theirs)).unwrap();
        assert!(
            crate::session_state::load(state.path(), "s-out")
                .drain_reminders()
                .is_empty()
        );
    }

    #[test]
    fn a_zero_budget_disables_the_check() {
        let brain = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let big = brain.path().join("huge.md");
        std::fs::write(&big, "x".repeat(40_000)).unwrap();
        let cfg = Config {
            brain_root: brain.path().to_path_buf(),
            governance: crate::config::GovernanceConfig {
                state_file_budget_tokens: 0,
                ..Default::default()
            },
            ..Config::default()
        };
        post_tool_budget(state.path(), &cfg, &budget_event("s-zero", &big)).unwrap();
        assert!(
            crate::session_state::load(state.path(), "s-zero")
                .drain_reminders()
                .is_empty()
        );
    }

    fn start_event(session: &str, reason: &str) -> HookEvent {
        HookEvent {
            session_id: Some(session.into()),
            reason: Some(reason.into()),
            ..Default::default()
        }
    }

    /// A session file as a working conversation leaves it: something read,
    /// something written, caps spent, warnings disarmed by a compaction.
    fn seed_conversation(dir: &Path, session: &str) {
        let mut st = session_state::SessionState::default();
        st.record_read("/a.rs", 100);
        st.record_write("/b.rs");
        st.warned.insert("/a.rs".into());
        st.shown_keys.insert("status".into());
        st.compacted = true;
        st.tool_events = 7;
        session_state::store(dir, session, &st);
    }

    #[test]
    fn a_cleared_conversation_starts_without_the_previous_one_s_records() {
        let state = tempfile::tempdir().unwrap();
        seed_conversation(state.path(), "s-clear");
        session_start_state(state.path(), &start_event("s-clear", "clear"));

        let mut st = session_state::load(state.path(), "s-clear");
        assert!(
            st.reads.is_empty(),
            "the read history belongs to the conversation that is gone"
        );
        assert!(st.written.is_empty(), "so does the write history");
        assert_eq!(st.tool_events, 0);
        assert!(!st.compacted, "the new conversation has not been compacted");
        assert!(
            st.queue_reminder("status", "nudge"),
            "the spent caps are returned"
        );
    }

    #[test]
    fn resume_and_compaction_continue_the_conversation_they_arrive_in() {
        for reason in ["resume", "compact"] {
            let state = tempfile::tempdir().unwrap();
            seed_conversation(state.path(), "s");
            session_start_state(state.path(), &start_event("s", reason));

            let mut st = session_state::load(state.path(), "s");
            assert_eq!(
                st.reads.len(),
                1,
                "{reason} must not forget what was already read"
            );
            assert!(
                st.compacted,
                "{reason} must not re-arm what compaction disarmed"
            );
            assert!(
                !st.queue_reminder("status", "nudge"),
                "{reason} keeps spent caps spent"
            );
        }
    }

    #[test]
    fn compaction_hands_back_the_record_the_advisory_stops_covering() {
        let state = tempfile::tempdir().unwrap();
        let event = HookEvent {
            session_id: Some("s-recap".into()),
            reason: Some("compact".into()),
            cwd: Some("/work/repo".into()),
            ..Default::default()
        };
        let _ = session_state::update(state.path(), "s-recap", |st| {
            st.record_write("/work/repo/src/a.rs");
            st.record_write("/elsewhere/vendored.rs");
            st.record_shell_write();
        });

        precompact_state(state.path(), &event);
        let start = session_start_state(state.path(), &event);

        // The premise: from here on the repeat-read advisory says nothing, so
        // the file list is the only thing left that can.
        let mut st = session_state::load(state.path(), "s-recap");
        st.record_read("/work/repo/src/a.rs", 100);
        assert!(
            !st.should_warn_repeat_read("/work/repo/src/a.rs", 100),
            "compaction disarms it"
        );

        let recap = compact_recap_for(state.path(), &event, start).expect("compaction is recapped");
        assert!(
            recap.contains("src/a.rs"),
            "the written file must be named: {recap}"
        );
        assert!(
            !recap.contains("/work/repo/src/a.rs"),
            "the shared prefix is not repeated"
        );
        assert!(
            recap.contains("/elsewhere/vendored.rs"),
            "outside cwd stays absolute: {recap}"
        );
        assert!(
            recap.contains("1 shell write"),
            "unnamed writes are counted: {recap}"
        );
    }

    #[test]
    fn no_other_start_reason_receives_the_recap() {
        let state = tempfile::tempdir().unwrap();
        for reason in ["startup", "clear", "resume", "rewound"] {
            let event = HookEvent {
                session_id: Some(format!("s-{reason}")),
                reason: Some(reason.into()),
                cwd: Some("/work/repo".into()),
                ..Default::default()
            };
            let _ = session_state::update(state.path(), event.session(), |st| {
                st.record_write("/work/repo/src/a.rs");
            });
            let start = session_start_state(state.path(), &event);
            assert!(
                compact_recap_for(state.path(), &event, start).is_none(),
                "{reason} did not summarize the conversation away, so nothing is owed"
            );
        }
    }

    #[test]
    fn the_recap_names_a_bounded_prefix_and_counts_the_rest() {
        let mut st = session_state::SessionState::default();
        for i in 0..COMPACT_RECAP_MAX_FILES + 3 {
            st.record_write(&format!("/repo/f{i:02}.rs"));
        }
        let recap = compact_recap(&st, Some("/repo")).unwrap();
        assert_eq!(
            recap.lines().count(),
            COMPACT_RECAP_MAX_FILES + 2,
            "one header, the capped names, one overflow line: {recap}"
        );
        assert!(recap.contains("f00.rs") && recap.contains("f11.rs"));
        assert!(
            !recap.contains("f12.rs"),
            "past the cap the count carries the rest"
        );
        assert!(recap.contains("(+3 more)"), "{recap}");
    }

    #[test]
    fn a_session_that_changed_nothing_is_recapped_with_silence() {
        assert!(compact_recap(&session_state::SessionState::default(), Some("/repo")).is_none());
    }

    #[test]
    fn a_shell_only_session_is_recapped_as_a_count_it_can_stand_behind() {
        let mut st = session_state::SessionState::default();
        st.record_shell_write();
        st.record_shell_write();
        let recap = compact_recap(&st, None).expect("unnamed activity is still activity");
        assert!(recap.contains("2 shell write(s)"), "{recap}");
        assert!(
            recap.contains("targets not recorded"),
            "the gap is stated, not papered over"
        );
    }

    #[test]
    fn an_unrecognized_start_reason_keeps_what_it_cannot_judge() {
        let state = tempfile::tempdir().unwrap();
        seed_conversation(state.path(), "s-unknown");
        session_start_state(state.path(), &start_event("s-unknown", "rewound"));
        assert_eq!(
            session_state::load(state.path(), "s-unknown").reads.len(),
            1,
            "an unknown reason must take the recoverable branch, not the destructive one"
        );
    }

    #[test]
    fn the_historical_source_field_drives_the_same_decision() {
        let state = tempfile::tempdir().unwrap();
        seed_conversation(state.path(), "s-legacy");
        let event = HookEvent {
            session_id: Some("s-legacy".into()),
            source: Some("startup".into()),
            ..Default::default()
        };
        session_start_state(state.path(), &event);
        assert!(
            session_state::load(state.path(), "s-legacy")
                .reads
                .is_empty(),
            "the older field spelling names the same start"
        );
    }

    #[test]
    fn shell_writes_are_recognized_without_naming_the_target() {
        // Redirects, in either form.
        assert!(is_shell_write("cat x > /tmp/out"));
        assert!(is_shell_write("printf hi >> log.txt"));
        // Programs whose purpose is to modify the filesystem, incl. in a pipe.
        assert!(is_shell_write("sudo cp a b"));
        assert!(is_shell_write("echo hi | tee file"));
        assert!(is_shell_write("mkdir -p /tmp/x"));
        assert!(is_shell_write("sed -i 's/a/b/' f"));
        // A quoted '>' is data, not a redirect.
        assert!(!is_shell_write("echo \'a > b\'"));
        assert!(!is_shell_write("grep \"a > b\" file"));
        // Read-only commands, and sed without an in-place flag.
        assert!(!is_shell_write("cat file"));
        assert!(!is_shell_write("sed 's/a/b/' f"));
        assert!(!is_shell_write("ls -la"));
    }

    use super::*;
    use crate::config::CaptureConfig;
    use serde_json::json;

    /// Books straight into a tempdir, standing in for the tree's logs dir.
    fn test_sink(dir: &Path) -> LedgerSink {
        LedgerSink {
            dir: dir.to_path_buf(),
            host: "test-host".into(),
            cap: 1 << 20,
        }
    }

    fn bash_event(cmd: &str) -> HookEvent {
        HookEvent {
            session_id: Some("s1".into()),
            tool_name: Some("Bash".into()),
            tool_input: Some(json!({"command": cmd})),
            ..Default::default()
        }
    }

    /// A config whose brain tree is a tempdir, so capture writes into a
    /// throwaway `logs/cfetch` instead of the operator's tree.
    fn tree_cfg(brain: &Path, capture: bool) -> Config {
        Config {
            brain_root: brain.to_path_buf(),
            capture: CaptureConfig { enabled: capture },
            ..Config::default()
        }
    }

    #[test]
    fn capture_disabled_writes_nothing() {
        let brain = tempfile::tempdir().unwrap();
        let cfg = tree_cfg(brain.path(), false);
        post_tool_capture(&cfg, &bash_event("ls")).unwrap();
        stop_capture(&cfg, &bash_event("ls")).unwrap();
        assert!(
            !paths::logs_dir(brain.path()).exists(),
            "disabled capture must not even create the exhaust stream"
        );
    }

    #[test]
    fn capture_enabled_appends_one_line_to_the_tree() {
        let brain = tempfile::tempdir().unwrap();
        let cfg = tree_cfg(brain.path(), true);
        post_tool_capture(&cfg, &bash_event("cargo build")).unwrap();
        let records =
            crate::jsonl::read_all(&paths::logs_dir(brain.path()), exhaust::STREAM).records;
        assert_eq!(records.len(), 1, "one tool call, one appended line");
        assert_eq!(records[0].kind(), "bash");
        assert_eq!(
            records[0].value("payload").unwrap()["command"],
            "cargo build"
        );
    }

    #[test]
    fn codex_listing_output_is_condensed_and_preserved_privately() {
        let state = tempfile::tempdir().unwrap();
        let output = (0..200)
            .map(|i| format!("line {i} with enough content to make condensation worthwhile"))
            .collect::<Vec<_>>()
            .join("\n");
        let event = HookEvent {
            session_id: Some("s1".into()),
            turn_id: Some("turn-1".into()),
            tool_name: Some("Bash".into()),
            tool_input: Some(json!({"command": "API_TOKEN=not-a-real-secret rg needle ."})),
            tool_response: Some(json!(output)),
            tool_use_id: Some("call-1".into()),
            ..Default::default()
        };

        let rewrite = codex_condensed_output(state.path(), &event)
            .unwrap()
            .expect("long listing should be condensed");
        assert!(rewrite.text.contains("line 0 with enough content"));
        assert!(rewrite.text.contains("line 199 with enough content"));
        assert!(!rewrite.text.contains("line 100 with enough content"));
        assert!(
            rewrite
                .text
                .contains("full uncondensed output preserved at")
        );
        // The savings pair is only defensible if the original side is the
        // actual result that was replaced, taken at the rewrite itself.
        assert_eq!(rewrite.original_chars, output.len());
        assert!(rewrite.text.len() < rewrite.original_chars);

        let files = std::fs::read_dir(state.path().join("condensed-output"))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(files.len(), 1);
        let preserved = std::fs::read_to_string(files[0].path()).unwrap();
        assert!(preserved.contains("line 100 with enough content"));
        assert!(
            !preserved.contains("not-a-real-secret"),
            "the command header is redacted"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                files[0].metadata().unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn condensation_is_codex_only_and_never_rewrites_verification() {
        let state = tempfile::tempdir().unwrap();
        let output = (0..200)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");

        let mut event = bash_event("rg needle .");
        event.tool_response = Some(json!(output));
        assert!(
            codex_condensed_output(state.path(), &event)
                .unwrap()
                .is_none(),
            "an event without Codex's turn_id must not receive continue:false"
        );

        event.turn_id = Some("turn-1".into());
        event.tool_input = Some(json!({"command": "cargo test --all"}));
        assert!(
            codex_condensed_output(state.path(), &event)
                .unwrap()
                .is_none(),
            "test output is never rewritten"
        );
        assert!(!state.path().join("condensed-output").exists());
    }

    #[test]
    fn condensed_output_retention_keeps_the_current_pointer_and_caps_old_files() {
        let dir = tempfile::tempdir().unwrap();
        let old_a = dir.path().join("a.txt");
        let old_b = dir.path().join("b.txt");
        let keep = dir.path().join("keep.txt");
        std::fs::write(&old_a, "aaaaaaaaaa").unwrap();
        std::fs::write(&old_b, "bbbbbbbbbb").unwrap();
        std::fs::write(&keep, "kkkkkkkkkk").unwrap();

        prune_condensed_outputs(dir.path(), &keep, 15).unwrap();

        assert!(
            keep.exists(),
            "the path emitted to the model must survive pruning"
        );
        let bytes: u64 = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().metadata().unwrap().len())
            .sum();
        assert!(
            bytes <= 15,
            "old artifacts must be removed until the cap holds"
        );
    }

    #[test]
    fn stop_capture_stages_into_the_shared_tree() {
        let brain = tempfile::tempdir().unwrap();
        let cfg = tree_cfg(brain.path(), true);
        let hot = brain.path().join("knowledge/hot.md");
        for session in ["s1", "s2"] {
            let event = HookEvent {
                session_id: Some(session.into()),
                tool_name: Some("Write".into()),
                tool_input: Some(json!({"file_path": hot.to_string_lossy()})),
                ..Default::default()
            };
            post_tool_capture(&cfg, &event).unwrap();
        }
        stop_capture(&cfg, &bash_event("done")).unwrap();
        let staged = crate::staging::list(&paths::staging_dir(brain.path()));
        assert_eq!(
            staged.len(),
            1,
            "the hot-file trap staged a ring-5 candidate"
        );
        assert_eq!(staged[0].reason, "hot-file");
    }

    #[test]
    fn stop_capture_never_runs_legacy_state_migrations() {
        let state = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        let cfg = tree_cfg(brain.path(), true);
        let legacy = state.path().join("exhaust.db");
        let conn = rusqlite::Connection::open(&legacy).unwrap();
        conn.execute_batch(
            "CREATE TABLE events(
               id INTEGER PRIMARY KEY,
               session_id TEXT NOT NULL,
               ts INTEGER NOT NULL,
               kind TEXT NOT NULL,
               payload TEXT NOT NULL,
               flag INTEGER NOT NULL DEFAULT 0,
               flag_reason TEXT,
               consumed INTEGER NOT NULL DEFAULT 0);
             INSERT INTO events(session_id, ts, kind, payload)
             VALUES ('old-session', 1000, 'bash', '{\"command\":\"old\"}');",
        )
        .unwrap();
        drop(conn);

        stop_capture(&cfg, &bash_event("done")).unwrap();

        assert!(legacy.is_file(), "the old database remains operator data");
        assert!(
            !state.path().join("exhaust-db-imported").exists(),
            "a latency-critical Stop hook must never start an upgrade migration"
        );
        let records =
            crate::jsonl::read_all(&paths::logs_dir(brain.path()), crate::exhaust::STREAM).records;
        assert!(
            records
                .iter()
                .all(|record| record.str("session") != "old-session"),
            "legacy rows must be absent from the Stop-side append"
        );
        assert!(
            records.iter().any(|record| record.kind() == "turn"),
            "normal bounded Stop capture still runs"
        );
    }

    #[test]
    fn user_prompt_drains_queued_reminders_once_and_books_them() {
        let dir = tempfile::tempdir().unwrap();
        let mut st = session_state::SessionState::default();
        assert!(st.queue_reminder("status", "update the STATUS"));
        assert!(st.queue_reminder("staging", "look at staging"));
        session_state::store(dir.path(), "s1", &st);

        let event = HookEvent {
            session_id: Some("s1".into()),
            ..Default::default()
        };
        let sink = test_sink(dir.path());
        user_prompt_drain(dir.path(), &sink, &event, None).unwrap();

        let back = session_state::load(dir.path(), "s1");
        assert!(
            back.queued_reminders.is_empty(),
            "delivery must empty the queue"
        );
        assert!(back.shown_keys.contains("status") && back.shown_keys.contains("staging"));
        let ledger = ledger::load_from(dir.path());
        let booked = &ledger.sessions["s1"].by_source["reminders"];
        assert_eq!(booked.count, 1, "many reminders, ONE emit, one booking");
        assert!(booked.chars > 0);

        // A second prompt delivers (and books) nothing further.
        user_prompt_drain(dir.path(), &sink, &event, None).unwrap();
        let ledger = ledger::load_from(dir.path());
        assert_eq!(ledger.sessions["s1"].by_source["reminders"].count, 1);
    }

    #[test]
    fn user_prompt_never_serves_subagents() {
        let dir = tempfile::tempdir().unwrap();
        let mut st = session_state::SessionState::default();
        assert!(st.queue_reminder("status", "for the primary session"));
        session_state::store(dir.path(), "s1", &st);

        let sub = HookEvent {
            session_id: Some("s1".into()),
            agent_id: Some("a1".into()),
            ..Default::default()
        };
        user_prompt_drain(dir.path(), &test_sink(dir.path()), &sub, None).unwrap();
        let back = session_state::load(dir.path(), "s1");
        assert_eq!(
            back.queued_reminders.len(),
            1,
            "a subagent must not consume the queue"
        );
        assert!(
            ledger::load_from(dir.path()).sessions.is_empty(),
            "and books nothing"
        );
    }

    /// Config with governance settings and a tempdir brain holding one ring-0
    /// resident rules file.
    fn govern_cfg(brain: &Path, enabled: bool, reinject_every: u32) -> Config {
        std::fs::write(
            brain.join("rules.md"),
            "---\ndescription: never force off VMs\n---\nbody\n",
        )
        .unwrap();
        Config {
            brain_root: brain.to_path_buf(),
            resident: vec![crate::config::ResidentEntry {
                path: std::path::PathBuf::from("rules.md"),
                ring: 0,
                scope: crate::config::Scope::default(),
                weight: None,
            }],
            governance: crate::config::GovernanceConfig {
                enabled,
                reinject_every,
                ..Default::default()
            },
            ..Config::default()
        }
    }

    fn brain_writes(state_dir: &Path, session: &str, brain: &Path, n: usize) {
        let mut st = session_state::load(state_dir, session);
        for i in 0..n {
            st.record_write(&format!("{}/knowledge/f{i}.md", brain.display()));
        }
        session_state::store(state_dir, session, &st);
    }

    #[test]
    fn stop_producers_queue_only_when_governance_is_enabled() {
        let state = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        brain_writes(state.path(), "s1", brain.path(), 3);
        let event = HookEvent {
            session_id: Some("s1".into()),
            ..Default::default()
        };

        let off = govern_cfg(brain.path(), false, 25);
        stop_govern(state.path(), &off, &event).unwrap();
        assert!(
            session_state::load(state.path(), "s1")
                .queued_reminders
                .is_empty(),
            "disabled governance must queue nothing"
        );

        let on = govern_cfg(brain.path(), true, 25);
        stop_govern(state.path(), &on, &event).unwrap();
        let st = session_state::load(state.path(), "s1");
        assert_eq!(st.queued_reminders.len(), 1);
        assert_eq!(st.queued_reminders[0].0, "status");
    }

    #[test]
    fn stop_producers_exempt_subagents() {
        let state = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        brain_writes(state.path(), "s1", brain.path(), 3);
        let sub = HookEvent {
            session_id: Some("s1".into()),
            agent_type: Some("worker".into()),
            ..Default::default()
        };
        let cfg = govern_cfg(brain.path(), true, 25);
        stop_govern(state.path(), &cfg, &sub).unwrap();
        assert!(
            session_state::load(state.path(), "s1")
                .queued_reminders
                .is_empty()
        );
    }

    #[test]
    fn cadence_queues_rules_at_exactly_n_and_2n() {
        let state = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        let cfg = govern_cfg(brain.path(), true, 2);
        let event = bash_event("ls");

        for _ in 0..4 {
            post_tool_cadence(state.path(), &cfg, &event).unwrap();
        }
        let st = session_state::load(state.path(), "s1");
        assert_eq!(st.tool_events, 4);
        let keys: Vec<&str> = st
            .queued_reminders
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(
            keys,
            vec!["rules-2", "rules-4"],
            "fires at exactly N and 2N"
        );
        assert!(
            st.queued_reminders[0]
                .1
                .starts_with("[cfetch rule refresh]")
        );
        assert!(st.queued_reminders[0].1.contains("never force off VMs"));
    }

    #[test]
    fn cadence_skips_subagents_disabled_governance_and_non_tool_events() {
        let state = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();

        let off = govern_cfg(brain.path(), false, 2);
        post_tool_cadence(state.path(), &off, &bash_event("ls")).unwrap();

        let on = govern_cfg(brain.path(), true, 2);
        let mut sub = bash_event("ls");
        sub.agent_id = Some("a1".into());
        post_tool_cadence(state.path(), &on, &sub).unwrap();

        let mut no_tool = bash_event("ls");
        no_tool.tool_name = None;
        post_tool_cadence(state.path(), &on, &no_tool).unwrap();

        let st = session_state::load(state.path(), "s1");
        assert_eq!(st.tool_events, 0, "none of these may advance the counter");
        assert!(st.queued_reminders.is_empty());
    }

    #[test]
    fn symbol_slices_reads_the_index_readonly_and_fails_silent() {
        // Missing DB: no hints, no error, no DB file created.
        let empty = tempfile::tempdir().unwrap();
        let missing = empty.path().join("index.db");
        assert!(symbol_slices(&missing, "/x.rs", 5).is_empty());
        assert!(!missing.exists(), "read-only probe must not create a DB");

        // Real index: hints come back largest-slice-first as `name start-end`.
        let code = tempfile::tempdir().unwrap();
        let f = code.path().join("big.rs");
        std::fs::write(
            &f,
            "fn one() {\n}\nfn two() {\n    let a = 1;\n    let b = 2;\n}\n",
        )
        .unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut conn = crate::index::open(state.path()).unwrap();
        crate::code::scan_code(&mut conn, &[code.path().to_path_buf()]).unwrap();
        drop(conn);
        let path = f.to_string_lossy().to_string();
        let hints = symbol_slices(&state.path().join("index.db"), &path, 5);
        assert_eq!(hints, vec!["two 3-6".to_string(), "one 1-2".to_string()]);
        assert_eq!(
            symbol_slices(&state.path().join("index.db"), &path, 1).len(),
            1
        );
        assert!(symbol_slices(&state.path().join("index.db"), "/absent.rs", 5).is_empty());
    }

    #[test]
    fn large_file_gate_is_size_only_and_fails_quiet() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.txt");
        std::fs::write(&p, vec![b'x'; LARGE_FILE_BYTES as usize]).unwrap();
        assert!(!is_large_file(&p), "exactly the threshold is not large");
        std::fs::write(&p, vec![b'x'; LARGE_FILE_BYTES as usize + 1]).unwrap();
        assert!(is_large_file(&p));
        std::fs::write(&p, "small").unwrap();
        assert!(!is_large_file(&p));
        assert!(!is_large_file(Path::new("/nonexistent/cfetch-large-gate")));
    }

    #[test]
    fn ranged_reads_are_exempt_from_tracking() {
        let full: serde_json::Value = serde_json::json!({"file_path": "/a.rs"});
        let ranged: serde_json::Value =
            serde_json::json!({"file_path": "/a.rs", "offset": 10, "limit": 40});
        assert!(!is_ranged(&full));
        assert!(is_ranged(&ranged));
        assert_eq!(read_target(&full), Some("/a.rs"));
        assert_eq!(read_target(&serde_json::json!({"command": "ls"})), None);
    }

    #[test]
    fn codex_shell_reads_are_recognized_only_for_a_safe_subset() {
        let cwd = tempfile::tempdir().unwrap();
        let event = |command: &str| HookEvent {
            cwd: Some(cwd.path().to_string_lossy().into_owned()),
            tool_name: Some("Bash".into()),
            tool_input: Some(json!({"command": command})),
            ..Default::default()
        };
        assert_eq!(
            read_invocation(&event("cat 'file with spaces.rs'")),
            Some((
                cwd.path()
                    .join("file with spaces.rs")
                    .to_string_lossy()
                    .into_owned(),
                false
            ))
        );
        assert_eq!(
            read_invocation(&event("head -n 20 src/main.rs")),
            Some((
                cwd.path()
                    .join("src/main.rs")
                    .to_string_lossy()
                    .into_owned(),
                true
            ))
        );
        assert_eq!(
            read_invocation(&event("sed -n '10,30p' src/main.rs")),
            Some((
                cwd.path()
                    .join("src/main.rs")
                    .to_string_lossy()
                    .into_owned(),
                true
            ))
        );
        for unsafe_command in [
            "cat a.rs b.rs",
            "cat a.rs | sed -n 1p",
            "cat a.rs > copy.rs",
            "cat $(secret-path)",
        ] {
            assert_eq!(
                read_invocation(&event(unsafe_command)),
                None,
                "{unsafe_command}"
            );
        }

        let windows = HookEvent {
            tool_name: Some("Bash".into()),
            tool_input: Some(json!({"command": r"cat C:\Users\agent\brain\topic.md"})),
            ..Default::default()
        };
        assert_eq!(
            read_invocation(&windows),
            Some((r"C:\Users\agent\brain\topic.md".to_string(), false))
        );
    }

    #[test]
    fn a_rewrite_books_what_it_saved_next_to_what_it_cost() {
        let dir = tempfile::tempdir().unwrap();
        book_rewrite(&test_sink(dir.path()), "s1", 40_000, 1_500);
        let l = ledger::load_from(dir.path());

        // The injection line carries what ENTERED. Booking the original here
        // would price the session for output it never received.
        let booked = &l.sessions["s1"].by_source["output-condensation"];
        assert_eq!(booked.count, 1);
        assert_eq!(booked.chars, 1_500);

        // ... and the pair sits in its own field, never inside the injection
        // bill, where it would look like cfetch had injected 40k characters.
        let c = l.condensation["s1"];
        assert_eq!(c.rewrites, 1);
        assert_eq!(c.original_tokens, crate::hook_io::estimate_tokens(40_000));
        assert_eq!(c.entered_tokens, crate::hook_io::estimate_tokens(1_500));
        assert!(!l.sessions["s1"].by_source.contains_key("condensed"));
    }

    /// Config whose brain tree is a tempdir, for the self-read paths.
    fn brain_cfg(brain: &Path) -> Config {
        Config {
            brain_root: brain.to_path_buf(),
            ..Config::default()
        }
    }

    fn read_event(session: &str, path: &Path) -> HookEvent {
        HookEvent {
            session_id: Some(session.into()),
            tool_name: Some("Read".into()),
            tool_input: Some(json!({"file_path": path.to_string_lossy()})),
            ..Default::default()
        }
    }

    #[test]
    fn the_agents_own_brain_reads_are_booked_apart_from_the_injection_bill() {
        let logs = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        let cfg = brain_cfg(brain.path());
        let sink = test_sink(logs.path());
        let mine = brain.path().join("knowledge/topic.md");
        std::fs::create_dir_all(mine.parent().unwrap()).unwrap();
        std::fs::write(&mine, "x".repeat(7_000)).unwrap();

        post_tool_self_read(&sink, &cfg, &read_event("s1", &mine));
        let l = ledger::load_from(logs.path());
        let sr = l.self_reads["s1"];
        assert_eq!(sr.reads, 1);
        assert_eq!(sr.chars, 7_000, "the file's own size is the spend");
        assert_eq!(sr.tokens_estimated, crate::hook_io::estimate_tokens(7_000));
        assert!(
            l.sessions["s1"].by_source.is_empty(),
            "a read the AGENT performed is not an injection by cfetch"
        );

        // A shell read of the same file counts the same — 140 of 144 measured
        // duplicate reads upstream flowed through bash, not the Read tool.
        let mut shell = read_event("s1", &mine);
        shell.tool_name = Some("Bash".into());
        shell.tool_input = Some(json!({"command": format!("cat {}", mine.display())}));
        post_tool_self_read(&sink, &cfg, &shell);
        assert_eq!(ledger::load_from(logs.path()).self_reads["s1"].reads, 2);
    }

    #[test]
    fn self_read_booking_skips_narrow_reads_foreign_files_and_subagents() {
        let logs = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        let cfg = brain_cfg(brain.path());
        let sink = test_sink(logs.path());
        let mine = brain.path().join("big.md");
        std::fs::write(&mine, "x".repeat(7_000)).unwrap();

        // A ranged read is the behaviour the redirect asks for; charging for
        // it would punish exactly the habit this mechanism is buying.
        let mut ranged = read_event("s1", &mine);
        ranged.tool_input = Some(json!({"file_path": mine.to_string_lossy(), "limit": 40}));
        post_tool_self_read(&sink, &cfg, &ranged);

        // Source files are the user's business, not the brain's bill.
        let elsewhere = tempfile::tempdir().unwrap();
        let theirs = elsewhere.path().join("vendor.rs");
        std::fs::write(&theirs, "x".repeat(7_000)).unwrap();
        post_tool_self_read(&sink, &cfg, &read_event("s1", &theirs));

        // A fork reads into its own context, not this session's row.
        let mut sub = read_event("s1", &mine);
        sub.agent_id = Some("a1".into());
        post_tool_self_read(&sink, &cfg, &sub);

        assert!(ledger::load_from(logs.path()).self_reads.is_empty());
    }

    #[test]
    fn the_whole_file_redirect_names_the_cheap_path_once_per_session() {
        let state = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        let cfg = brain_cfg(brain.path());
        let big = brain.path().join("knowledge/huge.md");
        std::fs::create_dir_all(big.parent().unwrap()).unwrap();
        std::fs::write(&big, "x".repeat(40_000)).unwrap();
        let path = big.to_string_lossy().to_string();
        let event = read_event("s1", &big);

        let advice = brain_read_redirect(Some(&cfg), state.path(), &event, &path)
            .expect("an oversized brain file must be redirected");
        assert!(advice.contains("knowledge/huge.md"), "{advice}");
        assert!(
            advice.contains("cfetch recall"),
            "the cheap path must be named: {advice}"
        );
        assert!(
            advice.contains("--id"),
            "expanding one statement is the other half: {advice}"
        );

        // Once per session, not per file: a second oversized file in the same
        // session says nothing further.
        let other = brain.path().join("other.md");
        std::fs::write(&other, "x".repeat(40_000)).unwrap();
        assert!(
            brain_read_redirect(Some(&cfg), state.path(), &event, &other.to_string_lossy())
                .is_none(),
            "the redirect must not nag"
        );
        // A different session gets its own single shot.
        assert!(
            brain_read_redirect(Some(&cfg), state.path(), &read_event("s2", &big), &path).is_some()
        );
    }

    #[test]
    fn the_redirect_stays_silent_for_small_files_foreign_files_and_a_zero_budget() {
        let state = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        let cfg = brain_cfg(brain.path());

        let small = brain.path().join("small.md");
        std::fs::write(&small, "x".repeat(100)).unwrap();
        let event = read_event("s1", &small);
        assert!(
            brain_read_redirect(Some(&cfg), state.path(), &event, &small.to_string_lossy())
                .is_none(),
            "a file inside the budget is fine to read whole"
        );

        let elsewhere = tempfile::tempdir().unwrap();
        let theirs = elsewhere.path().join("vendor.rs");
        std::fs::write(&theirs, "x".repeat(40_000)).unwrap();
        assert!(
            brain_read_redirect(Some(&cfg), state.path(), &event, &theirs.to_string_lossy())
                .is_none(),
            "cfetch governs its own tree, not the user's source"
        );

        let big = brain.path().join("huge.md");
        std::fs::write(&big, "x".repeat(40_000)).unwrap();
        let off = Config {
            governance: crate::config::GovernanceConfig {
                state_file_budget_tokens: 0,
                ..Default::default()
            },
            ..brain_cfg(brain.path())
        };
        assert!(
            brain_read_redirect(Some(&off), state.path(), &event, &big.to_string_lossy()).is_none(),
            "zero budget is the off switch for both ends of the policy"
        );
        // The silent cases must not have burned the session's one shot.
        assert!(
            brain_read_redirect(Some(&cfg), state.path(), &event, &big.to_string_lossy()).is_some()
        );
    }
}
