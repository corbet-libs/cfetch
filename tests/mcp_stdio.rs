//! Wire-level acceptance test for the real stdio MCP transport.

use std::io::{BufRead as _, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

fn send(stdin: &mut impl std::io::Write, message: Value) {
    writeln!(stdin, "{message}").unwrap();
    stdin.flush().unwrap();
}

fn response(rx: &mpsc::Receiver<Value>, id: u64) -> Value {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let value = rx
            .recv_timeout(remaining)
            .unwrap_or_else(|error| panic!("MCP response {id} did not arrive: {error}"));
        if value.get("id").and_then(Value::as_u64) == Some(id) {
            return value;
        }
    }
}

#[test]
fn official_stdio_transport_negotiates_and_serves_tools() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_cfetch"))
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let line = line.unwrap();
            tx.send(serde_json::from_str(&line).unwrap()).unwrap();
        }
    });

    send(
        &mut stdin,
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "cfetch-test", "version": "1"}
            }
        }),
    );
    let initialized = response(&rx, 1);
    assert_eq!(initialized["result"]["serverInfo"]["name"], "cfetch");
    assert_eq!(
        initialized["result"]["protocolVersion"], "2025-06-18",
        "the SDK must negotiate the requested supported version"
    );

    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    );
    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
    );
    let listed = response(&rx, 2);
    let names: Vec<&str> = listed["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|tool| tool["name"].as_str())
        .collect();
    assert_eq!(
        names,
        [
            "cfetch_graph",
            "cfetch_graph_path",
            "cfetch_recall",
            "cfetch_expand",
            "cfetch_find",
            "cfetch_code_path",
            "cfetch_code_impact",
            "cfetch_code_context",
            "cfetch_code_symbol",
            "cfetch_runtime_status",
            "cfetch_maintenance_packet",
            "cfetch_maintenance_show",
            "cfetch_maintenance_propose",
            "cfetch_maintenance_review",
        ]
    );
    let tools = listed["result"]["tools"].as_array().unwrap();
    assert!(
        tools[..12]
            .iter()
            .all(|tool| tool["annotations"]["readOnlyHint"] == true)
    );
    assert!(
        tools[12..]
            .iter()
            .all(|tool| tool["annotations"]["readOnlyHint"] == false)
    );
    assert!(
        tools[12..]
            .iter()
            .all(|tool| tool["annotations"]["destructiveHint"] == false)
    );

    send(
        &mut stdin,
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "cfetch_delete_everything", "arguments": {}}
        }),
    );
    let unknown = response(&rx, 3);
    assert_eq!(unknown["error"]["code"], -32602);

    drop(stdin);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "MCP server exited with {status}");
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            panic!("MCP server did not exit when the client closed stdin");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    reader.join().unwrap();
}
