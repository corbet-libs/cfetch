//! Coherence torture harness for serving mode.
//!
//! Proves the PRD's guarantees against a REAL daemon process (the compiled
//! binary, spawned per test against temp trees):
//!   (a) read-your-writes through the drain barrier, N concurrent writers,
//!       zero tolerance;
//!   (b) monotonic prefix across concurrent writers;
//!   (c) catalog determinism: fresh scan vs event-driven incremental builds
//!       yield EQUAL checksums;
//!   (d) crash-restart: the stat-fingerprint backstop catches up on writes
//!       made while the daemon was dead;
//!
//! The local channel is the platform's: a unix socket on unix, token-gated
//! loopback TCP on Windows (see `src/ipc.rs`). [`Local`] is the one place
//! that difference exists in this harness — every test below speaks to it
//! identically.
//!
use std::io::{BufRead as _, BufReader, Write as _};
#[cfg(windows)]
use std::net::TcpStream;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_cfetch");

struct Daemon {
    child: Child,
    state: PathBuf,
    // Kept alive for the daemon's lifetime.
    _home: tempfile::TempDir,
}

/// A handle on one daemon's LOCAL control channel. Cloneable and `Send` so
/// the concurrency tests can hand it to their writer threads, exactly as they
/// handed a socket path before.
#[cfg(unix)]
#[derive(Clone)]
struct Local(PathBuf);

#[cfg(windows)]
#[derive(Clone)]
struct Local {
    addr: String,
    token: String,
}

impl Local {
    /// Reads the endpoint a daemon publishes into its state dir. `None` until
    /// it has been published.
    #[cfg(unix)]
    fn published(state: &Path) -> Option<Local> {
        let p = state.join("daemon.sock");
        if p.exists() { Some(Local(p)) } else { None }
    }

    #[cfg(windows)]
    fn published(state: &Path) -> Option<Local> {
        let raw = std::fs::read_to_string(state.join("daemon.endpoint")).ok()?;
        let mut lines = raw.lines();
        let addr = lines.next()?.trim().to_string();
        let token = lines.next()?.trim().to_string();
        if addr.is_empty() || token.is_empty() {
            return None;
        }
        Some(Local { addr, token })
    }

    #[cfg(unix)]
    fn describe(&self) -> String {
        self.0.display().to_string()
    }

    #[cfg(windows)]
    fn describe(&self) -> String {
        format!("tcp {}", self.addr)
    }

    /// One request over the local channel; `None` when it does not answer.
    #[cfg(unix)]
    fn req_opt(&self, body: &Value) -> Option<Value> {
        let mut s = UnixStream::connect(&self.0).ok()?;
        s.set_read_timeout(Some(Duration::from_secs(15))).ok()?;
        s.set_write_timeout(Some(Duration::from_secs(15))).ok()?;
        writeln!(s, "{body}").ok()?;
        let mut line = String::new();
        BufReader::new(s).read_line(&mut line).ok()?;
        serde_json::from_str(&line).ok()
    }

    #[cfg(windows)]
    fn req_opt(&self, body: &Value) -> Option<Value> {
        let mut body = body.clone();
        body["token"] = Value::String(self.token.clone());
        let mut s = TcpStream::connect(&self.addr).ok()?;
        s.set_read_timeout(Some(Duration::from_secs(15))).ok()?;
        s.set_write_timeout(Some(Duration::from_secs(15))).ok()?;
        writeln!(s, "{body}").ok()?;
        let mut line = String::new();
        BufReader::new(s).read_line(&mut line).ok()?;
        serde_json::from_str(&line).ok()
    }

    fn req(&self, body: &Value) -> Value {
        self.req_opt(body)
            .unwrap_or_else(|| panic!("daemon did not answer on {}", self.describe()))
    }
}

impl Daemon {
    /// The daemon's local control channel.
    fn local(&self) -> Local {
        Local::published(&self.state).expect("daemon published its local endpoint")
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Spawns `cfetch daemon run` against its own state dir + config, waits until
/// the local control channel answers ping.
fn start_daemon(brain: &Path, state: &Path, serve_extra: Value) -> Daemon {
    start_daemon_cfg(brain, state, serve_extra, json!({}))
}

/// `start_daemon` with extra top-level config keys (e.g. `code_roots`).
fn start_daemon_cfg(brain: &Path, state: &Path, serve_extra: Value, cfg_extra: Value) -> Daemon {
    start_daemon_env(brain, state, serve_extra, cfg_extra, &[])
}

/// Spawn the daemon with explicit test environment overrides.
fn start_daemon_env(
    brain: &Path,
    state: &Path,
    _serve_extra: Value,
    cfg_extra: Value,
    env: &[(&str, &str)],
) -> Daemon {
    std::fs::create_dir_all(state).unwrap();
    let home = tempfile::tempdir().unwrap();
    let cfg_path = state.join("config.json");
    let mut cfg = json!({"resident": [], "capture": {"enabled": false}});
    if let Some(map) = cfg_extra.as_object() {
        for (k, v) in map {
            cfg[k] = v.clone();
        }
    }
    std::fs::write(&cfg_path, serde_json::to_string(&cfg).unwrap()).unwrap();
    let mut cmd = Command::new(BIN);
    cmd.args(["daemon", "run"])
        .env("CFETCH_STATE_DIR", state)
        .env("CFETCH_CONFIG", &cfg_path)
        .env("CFETCH_BRAIN", brain)
        .env("HOME", home.path())
        .env_remove("XDG_RUNTIME_DIR");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn daemon");
    let d = Daemon {
        child,
        state: state.to_path_buf(),
        _home: home,
    };
    for _ in 0..200 {
        if Local::published(&d.state)
            .and_then(|l| l.req_opt(&json!({"op": "ping"})))
            .is_some_and(|r| r["ok"] == true)
        {
            return d;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("daemon did not become ready in {}", d.state.display());
}

/// All hit snippets of a recall response, concatenated for containment checks.
fn snippet_blob(resp: &Value) -> String {
    resp["hits"]
        .as_array()
        .map(|hits| {
            hits.iter()
                .filter_map(|h| h["snippet"].as_str())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn append_line(path: &Path, line: &str) {
    use std::io::Write as _;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
        .unwrap();
    // One write syscall per line: a concurrent scan sees whole lines only.
    f.write_all(line.as_bytes()).unwrap();
}

// ---- (a) read-your-writes + observed-committed prefix, concurrent ----

#[test]
fn read_your_writes_under_concurrent_writers() {
    // All filesystems use the fingerprint barrier.
    concurrent_writer_torture(4, 75, &[]); // 300 barrier round-trips
}

/// N writers, each appending a uniquely tokenized line and then querying: a
/// fresh-labeled answer must contain EVERY statement whose write has already
/// completed — its own and every other writer's. Zero tolerance.
fn concurrent_writer_torture(writers: usize, iters: usize, env: &[(&str, &str)]) {
    let brain = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(brain.path().join("knowledge")).unwrap();
    for w in 0..writers {
        std::fs::write(brain.path().join(format!("knowledge/w{w}.md")), "").unwrap();
    }
    let state = tempfile::tempdir().unwrap();
    let daemon = start_daemon_env(
        brain.path(),
        state.path(),
        json!({"origin": "torture-origin"}),
        json!({}),
        env,
    );
    let local = daemon.local();

    // The mode must be VISIBLE to an operator, and must be the one asked for.
    let status = local.req(&json!({"op": "serve-status"}));
    let mode = status["serve"]["barrier_mode"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(
        !mode.is_empty(),
        "serve-status must name the barrier mode: {status}"
    );
    if let Some((_, want)) = env.iter().find(|(k, _)| *k == "CFETCH_BARRIER_MODE") {
        assert!(
            mode.starts_with(want),
            "forced {want}, daemon reports {mode}"
        );
    }

    // Every (writer, seq) whose append has COMPLETED. A query snapshotting
    // this set must see every member — that is the whole guarantee.
    let committed: Arc<Mutex<Vec<(usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
    let handles: Vec<_> = (0..writers)
        .map(|w| {
            let committed = committed.clone();
            let local = local.clone();
            let mode = mode.clone();
            let file = brain.path().join(format!("knowledge/w{w}.md"));
            std::thread::spawn(move || {
                for n in 1..=iters {
                    append_line(&file, &format!("- torture writer{w} seq{n} tk{w}x{n}\n"));
                    committed.lock().unwrap().push((w, n));
                    let snapshot = committed.lock().unwrap().clone();
                    let resp =
                        local.req(&json!({"op": "recall", "freshness": "strict", "query": "torture", "limit": 100000}));
                    assert_eq!(resp["ok"], true, "query failed: {resp}");
                    assert_eq!(
                        resp["fresh"], true,
                        "barrier must serve fresh under this load (writer {w} seq {n}, \
                         {mode}): {resp}"
                    );
                    assert!(
                        resp["origin"]
                            .as_str()
                            .is_some_and(|origin| !origin.is_empty())
                    );
                    let blob = snippet_blob(&resp);
                    for (cw, cn) in &snapshot {
                        assert!(
                            blob.contains(&format!("tk{cw}x{cn}")),
                            "writer {w} seq {n} ({mode}): committed statement tk{cw}x{cn} \
                             missing from a fresh-labeled answer (zero tolerance)"
                        );
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

// ---- (b) monotonic prefix seen by a concurrent reader ----

#[test]
fn monotonic_prefix_across_concurrent_writers() {
    const WRITERS: usize = 3;
    const ITERS: usize = 50;
    const READS: usize = 40;

    let brain = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(brain.path().join("knowledge")).unwrap();
    for w in 0..WRITERS {
        std::fs::write(brain.path().join(format!("knowledge/w{w}.md")), "").unwrap();
    }
    let state = tempfile::tempdir().unwrap();
    let daemon = start_daemon(brain.path(), state.path(), json!({}));
    let local = daemon.local();

    let writers: Vec<_> = (0..WRITERS)
        .map(|w| {
            let file = brain.path().join(format!("knowledge/w{w}.md"));
            std::thread::spawn(move || {
                for n in 1..=ITERS {
                    append_line(&file, &format!("- torture writer{w} seq{n} tk{w}x{n}\n"));
                    std::thread::sleep(Duration::from_millis(2));
                }
            })
        })
        .collect();

    for _ in 0..READS {
        let resp = local.req(
            &json!({"op": "recall", "freshness": "strict", "query": "torture", "limit": 100000}),
        );
        assert_eq!(resp["ok"], true, "{resp}");
        assert_eq!(resp["fresh"], true, "{resp}");
        let blob = snippet_blob(&resp);
        for w in 0..WRITERS {
            let max_seen = (1..=ITERS)
                .filter(|n| blob.contains(&format!("tk{w}x{n}")))
                .max()
                .unwrap_or(0);
            for n in 1..=max_seen {
                assert!(
                    blob.contains(&format!("tk{w}x{n}")),
                    "gap in writer {w}'s prefix: seq {n} missing while seq {max_seen} visible"
                );
            }
        }
    }
    for h in writers {
        h.join().unwrap();
    }
}

// ---- (c) determinism: fresh scan vs incremental-via-events ----

#[test]
fn checksum_deterministic_fresh_vs_incremental() {
    let brain = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(brain.path().join("knowledge")).unwrap();
    std::fs::write(brain.path().join("knowledge/base.md"), "- base fact\n").unwrap();

    // Daemon A builds INCREMENTALLY: initial scan, then five event batches.
    let state_a = tempfile::tempdir().unwrap();
    let daemon_a = start_daemon(brain.path(), state_a.path(), json!({}));
    for i in 0..5 {
        std::fs::write(
            brain.path().join(format!("knowledge/inc{i}.md")),
            format!("- incremental fact {i}\n\nparagraph {i}\n"),
        )
        .unwrap();
        std::thread::sleep(Duration::from_millis(20));
    }
    append_line(
        &brain.path().join("knowledge/base.md"),
        "- appended after start\n",
    );
    let a = daemon_a.local().req(&json!({"op": "checksum"}));
    assert_eq!(a["ok"], true, "{a}");
    assert_eq!(a["fresh"], true, "{a}");
    let checksum_a = a["checksum"].as_str().unwrap().to_string();
    assert!(!checksum_a.is_empty());

    // Daemon B derives FRESH from the finished tree in its own state dir.
    let state_b = tempfile::tempdir().unwrap();
    let daemon_b = start_daemon(brain.path(), state_b.path(), json!({}));
    let b = daemon_b.local().req(&json!({"op": "checksum"}));
    assert_eq!(b["ok"], true, "{b}");
    assert_eq!(
        b["checksum"].as_str().unwrap(),
        checksum_a,
        "incremental and fresh catalog derivations must agree byte-for-byte"
    );
    // Generations are per-holder histories and may differ; the CATALOG agrees.
    assert!(a["generation"].as_u64().unwrap() >= 1);
    assert!(b["generation"].as_u64().unwrap() >= 1);
}

// ---- (d) crash + restart: the fingerprint backstop catches up ----

#[test]
fn crash_restart_backstop_catches_up() {
    let brain = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(brain.path().join("knowledge")).unwrap();
    std::fs::write(brain.path().join("knowledge/a.md"), "- fact one\n").unwrap();

    let state = tempfile::tempdir().unwrap();
    let mut daemon = start_daemon(brain.path(), state.path(), json!({}));
    let before = daemon.local().req(&json!({"op": "checksum"}));
    assert_eq!(before["ok"], true);
    let checksum_before = before["checksum"].as_str().unwrap().to_string();

    // SIGKILL mid-life; write while nothing is watching.
    daemon.kill();
    append_line(
        &brain.path().join("knowledge/a.md"),
        "- fact two, written while daemon dead\n",
    );
    std::fs::write(
        brain.path().join("knowledge/new.md"),
        "- born during the outage\n",
    )
    .unwrap();

    // Restart on the SAME state dir: the startup fingerprint backstop must
    // reconcile before the first barrier releases.
    let daemon2 = start_daemon(brain.path(), state.path(), json!({}));
    let after = daemon2.local().req(&json!({"op": "checksum"}));
    assert_eq!(after["ok"], true, "{after}");
    assert_eq!(after["fresh"], true, "{after}");
    let checksum_after = after["checksum"].as_str().unwrap().to_string();
    assert_ne!(
        checksum_after, checksum_before,
        "outage writes must change the catalog"
    );

    // Ground truth: a fresh derivation over the final tree.
    let state_c = tempfile::tempdir().unwrap();
    let daemon_c = start_daemon(brain.path(), state_c.path(), json!({}));
    let fresh = daemon_c.local().req(&json!({"op": "checksum"}));
    assert_eq!(fresh["checksum"].as_str().unwrap(), checksum_after);

    // And recall actually surfaces the outage write.
    let resp = daemon2
        .local()
        .req(&json!({"op": "recall", "freshness": "strict", "query": "outage", "limit": 10}));
    assert_eq!(resp["ok"], true);
    assert!(snippet_blob(&resp).contains("born during the outage"));
}

// ---- the daemon scans code by itself, and serves `map` ----

/// The deployed defect: a fresh serving host answered `find`/`map` with "no
/// hits" forever, because a code scan only ever ran when someone sent
/// `scan-code` by hand. Nobody sends it here.
#[test]
fn daemon_scans_code_and_explains_local_dependencies() {
    let brain = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(brain.path().join("knowledge")).unwrap();
    std::fs::write(brain.path().join("knowledge/a.md"), "- seed statement\n").unwrap();

    // A small code tree for the code index: two files, one importing the other.
    let code = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(code.path().join("proj/src")).unwrap();
    std::fs::write(
        code.path().join("proj/src/lib.rs"),
        "pub fn alpha_helper() -> u32 { 1 }\npub struct AlphaThing;\n",
    )
    .unwrap();
    std::fs::write(
        code.path().join("proj/src/main.rs"),
        "mod lib;\nfn main() { let _ = lib::alpha_helper(); }\n",
    )
    .unwrap();

    let state = tempfile::tempdir().unwrap();
    let daemon = start_daemon_cfg(
        brain.path(),
        state.path(),
        json!({}),
        json!({"code_roots": [code.path().to_string_lossy()]}),
    );

    // Nobody sends `scan-code`: the daemon must kick its own scan once the
    // tree watches are registered.
    let mut counts = None;
    for _ in 0..300 {
        let s = daemon.local().req(&json!({"op": "scan-status"}));
        if s["scan"]["last_finished"].is_number() && !s["scan"]["running"].as_bool().unwrap_or(true)
        {
            counts = Some(s);
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let s = counts.expect("the daemon must run a code scan on its own, unasked");
    assert_eq!(
        s["scan"]["last_error"],
        Value::Null,
        "self-triggered scan failed: {s}"
    );
    assert!(
        s["scan"]["last_counts"]["files"].as_u64().unwrap_or(0) >= 2,
        "the self-triggered scan must have indexed the code tree: {s}"
    );

    // `find` now answers on a host nobody scanned by hand.
    let f = daemon
        .local()
        .req(&json!({"op": "find", "query": "alpha_helper"}));
    assert_eq!(f["ok"], true, "{f}");
    assert!(
        !f["code_hits"].as_array().unwrap().is_empty(),
        "a self-scanned host must answer find: {f}"
    );

    // The daemon and local CLI expose the same committed map.
    let sock_map = daemon
        .local()
        .req(&json!({"op": "map", "budget_tokens": 4000}));
    assert_eq!(sock_map["ok"], true, "{sock_map}");
    assert!(
        sock_map["origin"].is_string(),
        "map must carry the coherence labels: {sock_map}"
    );
    assert!(sock_map["generation"].is_number(), "{sock_map}");
    let lines: Vec<String> = sock_map["map"]["lines"]
        .as_array()
        .expect("map lines")
        .iter()
        .map(|l| l.as_str().unwrap().to_string())
        .collect();
    assert!(
        lines
            .iter()
            .any(|l| l.contains("proj/src/lib.rs") && l.contains("alpha_helper")),
        "map must list the indexed files with their symbols: {lines:?}"
    );

    // Dependency explanations use the same committed graph and coherence
    // envelope on both serving transports. Host-absolute paths must never
    // cross the wire.
    let path_request = json!({
        "op": "code-path",
        "from_path": "proj/src/main.rs",
        "to_path": "proj/src/lib.rs",
        "depth": 4,
    });
    let sock_path = daemon.local().req(&path_request);
    assert_eq!(sock_path["ok"], true, "{sock_path}");
    assert_eq!(sock_path["dependency_path"]["found"], true, "{sock_path}");
    assert_eq!(
        sock_path["dependency_path"]["edges"][0]["relation"],
        "imports"
    );
    assert_eq!(
        sock_path["dependency_path"]["edges"][0]["evidence"]["class"],
        "resolved"
    );
    assert_eq!(
        sock_path["dependency_path"]["edges"][0]["evidence"]["path"],
        "proj/src/main.rs"
    );
    assert_eq!(
        sock_path["dependency_path"]["edges"][0]["evidence"]["start_line"],
        1
    );
    let storage_root = code.path().to_string_lossy();
    assert!(
        !sock_path.to_string().contains(storage_root.as_ref()),
        "served dependency paths must not expose the storage host root: {sock_path}"
    );

    let impact_request = json!({
        "op": "code-impact",
        "path": "proj/src/lib.rs",
        "depth": 4,
        "limit": 10,
    });
    let sock_impact = daemon.local().req(&impact_request);
    assert_eq!(sock_impact["ok"], true, "{sock_impact}");
    assert_eq!(
        sock_impact["dependency_impact"]["total"], 1,
        "{sock_impact}"
    );
    assert_eq!(
        sock_impact["dependency_impact"]["nodes"][0]["path"],
        "proj/src/main.rs"
    );

    let context_request = json!({
        "op": "code-context",
        "path": "proj/src/lib.rs",
        "depth": 1,
        "limit": 10,
    });
    let sock_context = daemon.local().req(&context_request);
    assert_eq!(sock_context["ok"], true, "{sock_context}");
    assert_eq!(
        sock_context["dependency_context"]["total"], 1,
        "{sock_context}"
    );
    assert_eq!(
        sock_context["dependency_context"]["nodes"][0]["edge"]["source"],
        "proj/src/main.rs"
    );
    assert_eq!(
        sock_context["dependency_context"]["nodes"][0]["edge"]["target"],
        "proj/src/lib.rs"
    );
    assert!(
        !sock_context.to_string().contains(storage_root.as_ref()),
        "served dependency context must not expose the storage host root: {sock_context}"
    );

    let symbol_request = json!({"op": "code-symbol", "query": "main", "limit": 10});
    let sock_symbol = daemon.local().req(&symbol_request);
    assert_eq!(sock_symbol["ok"], true, "{sock_symbol}");
    assert_eq!(
        sock_symbol["symbol_context"]["total_symbols"], 1,
        "{sock_symbol}"
    );
    assert_eq!(
        sock_symbol["symbol_context"]["edges"]
            .as_array()
            .unwrap()
            .iter()
            .find(|edge| edge["relation"] == "calls")
            .unwrap()["target"]["name"],
        "alpha_helper"
    );
    assert!(
        !sock_symbol.to_string().contains(storage_root.as_ref()),
        "served symbol context must not expose the storage host root: {sock_symbol}"
    );

    // The serving host's own CLI, reading its local index directly.
    let cli_home = tempfile::tempdir().unwrap();
    let local_out = Command::new(BIN)
        .args(["map", "--budget-tokens", "4000"])
        .env("CFETCH_STATE_DIR", state.path())
        .env("CFETCH_CONFIG", state.path().join("config.json"))
        .env("HOME", cli_home.path())
        .env_remove("XDG_RUNTIME_DIR")
        .output()
        .unwrap();
    assert!(
        local_out.status.success(),
        "{}",
        String::from_utf8_lossy(&local_out.stderr)
    );
    let local_lines: Vec<String> = String::from_utf8_lossy(&local_out.stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_string)
        .collect();
    assert_eq!(
        local_lines, lines,
        "the local map and the served map must be the same lines"
    );
}
