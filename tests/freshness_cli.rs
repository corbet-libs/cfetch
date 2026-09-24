//! Frontend acceptance against a deliberately stale daemon and unreadable
//! query configuration: no CLI/MCP fallback may scan or load the source tree.
#![cfg(unix)]
use serde_json::{Value, json};
use std::io::{BufRead as _, BufReader, Write as _};
use std::os::unix::net::UnixListener;
use std::process::{Child, Command, Output, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_cfetch");
struct Fixture {
    dir: tempfile::TempDir,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), b"not valid configuration").unwrap();
        Self { dir }
    }
    fn command(&self) -> Command {
        let mut command = Command::new(BIN);
        command
            .env("CFETCH_STATE_DIR", self.dir.path())
            .env("CFETCH_CONFIG", self.dir.path().join("config.json"))
            .env("CFETCH_BRAIN", self.dir.path().join("unavailable-tree"))
            .env_remove("XDG_RUNTIME_DIR");
        command
    }
    fn mock(&self, replies: Vec<Value>) -> std::thread::JoinHandle<Vec<Value>> {
        let listener = UnixListener::bind(self.dir.path().join("daemon.sock")).unwrap();
        std::thread::spawn(move || {
            replies
                .into_iter()
                .map(|reply| {
                    let (mut stream, _) = listener.accept().unwrap();
                    stream
                        .set_read_timeout(Some(Duration::from_secs(3)))
                        .unwrap();
                    let mut request = String::new();
                    BufReader::new(stream.try_clone().unwrap())
                        .read_line(&mut request)
                        .unwrap();
                    writeln!(stream, "{reply}").unwrap();
                    serde_json::from_str(&request).unwrap()
                })
                .collect()
        })
    }
}
fn wait_output(mut child: Child) -> Output {
    let deadline = Instant::now() + Duration::from_secs(4);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("memory frontend exceeded its test deadline");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    child.wait_with_output().unwrap()
}
fn output(command: &mut Command) -> Output {
    wait_output(
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    )
}
fn stale() -> Value {
    json!({"ok":true, "origin":"fixture", "generation":42, "fresh":false,
        "stale_note":"cached snapshot; freshness unverified; background refresh requested",
        "hits":[{"cite":"r2-abcdef", "path":"knowledge/a.md", "ring":2, "start_line":1,
            "end_line":3, "snippet":"cached evidence", "mirrors":[]}],
        "blocks":[{"cite":"r2-abcdef", "path":"knowledge/a.md", "ring":2, "start_line":1,
            "end_line":3, "text":"cached evidence"}], "linked":[["knowledge/b.md",2]]})
}

#[test]
fn cli_cached_recall_id_and_link_expansion_are_daemon_first_and_labelled() {
    for args in [
        vec!["recall", "evidence", "--json", "--expand"],
        vec!["recall", "--id", "r2-abcdef", "--json"],
        vec!["recall", "evidence"],
        vec!["recall", "--id", "missing"],
    ] {
        let fixture = Fixture::new();
        let mut reply = stale();
        if args.contains(&"missing") {
            reply["blocks"] = json!([]);
        }
        let daemon = fixture.mock(vec![reply]);
        let out = output(fixture.command().args(&args));
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8(out.stdout).unwrap();
        assert!(text.contains("42"), "snapshot identity missing: {text}");
        assert!(
            text.contains("freshness unverified"),
            "stale note missing: {text}"
        );
        let requests = daemon.join().unwrap();
        assert_eq!(requests[0]["freshness"], "cached");
        if args.contains(&"--expand") {
            assert_eq!(requests[0]["expand"], true);
            assert!(text.contains("knowledge/b.md"));
        }
    }
}

#[test]
fn cli_strict_rejects_stale_data_instead_of_falling_back_to_a_scan() {
    let fixture = Fixture::new();
    let daemon = fixture.mock(vec![stale()]);
    let out = output(fixture.command().args(["recall", "evidence", "--fresh"]));
    assert!(!out.status.success());
    assert!(out.stdout.is_empty(), "strict must not output stale hits");
    assert!(String::from_utf8_lossy(&out.stderr).contains("freshness unverified"));
    assert_eq!(daemon.join().unwrap()[0]["freshness"], "strict");
}

#[test]
fn daemon_unavailable_never_creates_or_rebuilds_a_catalog() {
    let fixture = Fixture::new();
    let sentinel = fixture.dir.path().join("index.db");
    std::fs::write(&sentinel, b"preserve incompatible state").unwrap();
    for args in [
        vec!["recall", "evidence"],
        vec!["recall", "--id", "r2-abcdef"],
        vec!["recall", "evidence", "--expand"],
        vec!["recall", "evidence", "--fresh"],
    ] {
        let started = Instant::now();
        let out = output(fixture.command().args(args));
        assert!(!out.status.success());
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(String::from_utf8_lossy(&out.stderr).contains("memory daemon unavailable"));
        assert_eq!(
            std::fs::read(&sentinel).unwrap(),
            b"preserve incompatible state"
        );
    }
}

fn rpc(stdin: &mut impl std::io::Write, value: Value) {
    writeln!(stdin, "{value}").unwrap();
    stdin.flush().unwrap();
}
fn rpc_response(receiver: &mpsc::Receiver<Value>, id: u64) -> Value {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let value = receiver
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .unwrap();
        if value["id"] == id {
            return value;
        }
    }
}
#[test]
fn mcp_recall_and_expand_share_cached_and_strict_policy_without_loading_config() {
    let fixture = Fixture::new();
    let daemon = fixture.mock(vec![stale(), stale(), stale()]);
    let mut child = fixture
        .command()
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if tx
                .send(serde_json::from_str(&line.unwrap()).unwrap())
                .is_err()
            {
                break;
            }
        }
    });
    rpc(
        &mut stdin,
        json!({"jsonrpc":"2.0", "id":1,"method":"initialize", "params":{
        "protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"freshness-test","version":"1"}}}),
    );
    assert!(rpc_response(&rx, 1).get("result").is_some());
    rpc(
        &mut stdin,
        json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
    );
    for (id, name, args, strict) in [
        (
            2,
            "cfetch_recall",
            json!({"query":"evidence","mode":"lexical","expand":true}),
            false,
        ),
        (3, "cfetch_expand", json!({"cite":"r2-abcdef"}), false),
        (
            4,
            "cfetch_recall",
            json!({"query":"evidence","mode":"lexical","freshness":"strict"}),
            true,
        ),
    ] {
        rpc(
            &mut stdin,
            json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":name,"arguments":args}}),
        );
        let response = rpc_response(&rx, id);
        assert_eq!(
            response["result"]["isError"].as_bool().unwrap_or(false),
            strict,
            "{response}"
        );
        let text = response["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("freshness unverified"), "{text}");
        if !strict {
            assert!(text.contains("generation 42"), "{text}");
        }
    }
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();
    reader.join().unwrap();
    let requests = daemon.join().unwrap();
    assert_eq!(requests[0]["mode"], "lexical");
    assert_eq!(requests[1]["op"], "expand");
    assert_eq!(requests[2]["freshness"], "strict");
}
