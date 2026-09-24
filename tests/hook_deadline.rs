//! Real process deadlines with an intentionally blocked configuration FIFO.
//! These tests never read a host brain, install hooks or execute inference.
#![cfg(unix)]
use serde_json::{Value, json};
use std::io::{BufRead as _, BufReader, Write as _};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};
const BIN: &str = env!("CARGO_BIN_EXE_cfetch");
struct Fixture {
    dir: tempfile::TempDir,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("brain")).unwrap();
        std::fs::create_dir_all(dir.path().join("state")).unwrap();
        Self { dir }
    }
    fn config(&self) -> std::path::PathBuf {
        self.dir.path().join("config.json")
    }
    fn command(&self) -> Command {
        let mut command = Command::new(BIN);
        command
            .env("CFETCH_STATE_DIR", self.dir.path().join("state"))
            .env("CFETCH_CONFIG", self.config())
            .env("CFETCH_BRAIN", self.dir.path().join("brain"))
            .env("CFETCH_HOST", "freshness-test")
            .env("HOME", self.dir.path())
            .env_remove("XDG_RUNTIME_DIR");
        command
    }
    fn blocked_config(&self) {
        let _ = std::fs::remove_file(self.config());
        assert!(
            Command::new("mkfifo")
                .arg(self.config())
                .status()
                .unwrap()
                .success()
        );
    }
    fn hook(&self) -> (Output, Duration) {
        let started = Instant::now();
        let mut child = self
            .command()
            .args(["hook", "session-start", "--agent", "codex"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        writeln!(
            child.stdin.take().unwrap(),
            "{}",
            json!({"session_id":"deadline-test", "source":"startup", "cwd":self.dir.path()})
        )
        .unwrap();
        let deadline = started + Duration::from_secs(5);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("hook blocked beyond its process budget");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        (child.wait_with_output().unwrap(), started.elapsed())
    }
}
#[test]
fn warm_resident_ignores_blocked_config_and_preserves_daemon_ledger_cap() {
    let fixture = Fixture::new();
    fixture.blocked_config();
    let logs = fixture.dir.path().join("brain/logs/cfetch");
    std::fs::create_dir_all(&logs).unwrap();
    std::fs::write(logs.join("ledger-freshness-test.jsonl"), "{}\n".repeat(400)).unwrap();
    let listener = UnixListener::bind(fixture.dir.path().join("state/daemon.sock")).unwrap();
    let responder = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut line = String::new();
        BufReader::new(stream.try_clone().unwrap())
            .read_line(&mut line)
            .unwrap();
        let request: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(request["op"], "resident");
        writeln!(stream, "{}", json!({"ok":true,"digest":"trusted warm resident evidence","resident_ledger_max_bytes":900})).unwrap();
    });
    let (out, elapsed) = fixture.hook();
    responder.join().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("trusted warm resident evidence"), "{text}");
    assert!(!text.contains("deadline"), "{text}");
    assert!(
        elapsed < Duration::from_secs(1),
        "warm resident waited for config: {elapsed:?}"
    );
    assert!(
        logs.join("ledger-freshness-test.1.jsonl").exists(),
        "the daemon's900-byte cap must rotate the existing1200-byte ledger; the default cap would not"
    );
}

#[test]
fn missing_or_invalid_resident_policy_uses_bounded_fallback_without_guessing() {
    for cap in [None, Some(json!(-1))] {
        let fixture = Fixture::new();
        fixture.blocked_config();
        let listener = UnixListener::bind(fixture.dir.path().join("state/daemon.sock")).unwrap();
        let responder = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut line = String::new();
            BufReader::new(stream.try_clone().unwrap())
                .read_line(&mut line)
                .unwrap();
            let mut response =
                json!({"ok":true,"digest":"incomplete response must not be injected"});
            if let Some(cap) = cap {
                response["resident_ledger_max_bytes"] = cap;
            }
            writeln!(stream, "{response}").unwrap();
        });
        let (out, elapsed) = fixture.hook();
        responder.join().unwrap();
        assert!(
            out.status.success(),
            "hooks never block the host with a nonzero exit"
        );
        let text = String::from_utf8(out.stdout).unwrap();
        assert!(text.contains("2-second deadline"), "{text}");
        assert!(!text.contains("incomplete response must not be injected"));
        assert!(elapsed < Duration::from_secs(4), "{elapsed:?}");
    }
}

struct Daemon(Child);
impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn request(socket: &Path, body: Value) -> Option<Value> {
    let mut stream = UnixStream::connect(socket).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_millis(800)))
        .ok()?;
    stream
        .set_write_timeout(Some(Duration::from_millis(800)))
        .ok()?;
    writeln!(stream, "{body}").ok()?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line).ok()?;
    serde_json::from_str(&line).ok()
}
#[test]
fn resident_daemon_uses_its_validated_startup_policy_after_config_path_blocks() {
    let fixture = Fixture::new();
    std::fs::write(
        fixture.config(),
        json!({"resident":[],"capture":{"enabled":false},
        "embeddings":{"enabled":false},"ledger_max_bytes":12345})
        .to_string(),
    )
    .unwrap();
    let _daemon = Daemon(
        fixture
            .command()
            .args(["daemon", "run"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let socket = fixture.dir.path().join("state/daemon.sock");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !request(&socket, json!({"op":"ping"})).is_some_and(|response| response["ok"] == true) {
        assert!(Instant::now() < deadline, "daemon did not become ready");
        std::thread::sleep(Duration::from_millis(10));
    }
    fixture.blocked_config();
    let response =
        request(&socket, json!({"op":"resident"})).expect("resident reloaded blocked config");
    assert_eq!(response["ok"], true, "{response}");
    assert_eq!(response["resident_ledger_max_bytes"], 12345);
    assert!(response["digest"].is_string());
}
