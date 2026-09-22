//! The diagnostic command is useful before networking or inference has ever
//! run, and observing that blank state must not create a host identity.

use std::process::Command;

#[test]
fn doctor_json_is_read_only_and_labels_unmeasured_state() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("home");
    let state = root.path().join("state");
    let brain = root.path().join("brain");
    for dir in [&home, &state, &brain] {
        std::fs::create_dir_all(dir).unwrap();
    }

    let output = Command::new(env!("CARGO_BIN_EXE_cfetch"))
        .args(["doctor", "--json", "--no-network"])
        .env("HOME", &home)
        .env("APPDATA", &home)
        .env("CFETCH_STATE_DIR", &state)
        .env("CFETCH_BRAIN", &brain)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "doctor failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["daemon"]["state"], "stopped");
    assert_eq!(report["inference"]["utilization"]["state"], "not_reported");
    assert_eq!(report["repositories"], serde_json::json!([]));
    assert!(report["memory"].get("peer_artifacts").is_none());
    assert!(
        report["hardware"]
            .as_array()
            .is_some_and(|rows| !rows.is_empty()),
        "the CPU floor must always be visible: {report}"
    );
    let build_backend = report["build"]["inference_backend"].as_str().unwrap();
    let cpu_binding = report["hardware"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["token"] == "cpu")
        .and_then(|row| row["binding"].as_str())
        .unwrap();
    let expected_cpu_binding = match build_backend {
        "cpu" | "openvino" | "coreml" => "available_not_selected",
        _ => "not_supported_by_build",
    };
    assert_eq!(cpu_binding, expected_cpu_binding);
    assert!(
        !state.join("endpoint.key").exists(),
        "reading diagnostics must not create a network identity"
    );

    let output = Command::new(env!("CARGO_BIN_EXE_cfetch"))
        .args(["doctor", "--no-network"])
        .env("HOME", &home)
        .env("APPDATA", &home)
        .env("CFETCH_STATE_DIR", &state)
        .env("CFETCH_BRAIN", &brain)
        .output()
        .unwrap();
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("Detected hardware"), "{text}");
    if matches!(build_backend, "cpu" | "openvino" | "coreml") {
        assert!(
            text.contains("CPU [cpu] — available, not selected"),
            "{text}"
        );
    } else {
        assert!(
            text.contains("CPU [cpu] — not supported by this build"),
            "{text}"
        );
    }
    assert!(text.contains("live utilization: not reported"), "{text}");
    assert!(text.contains("Git repositories"), "{text}");
}

#[test]
fn deep_doctor_uses_a_temporary_retrieval_fixture() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("home");
    let state = root.path().join("state");
    let brain = root.path().join("brain");
    for dir in [&home, &state, &brain] {
        std::fs::create_dir_all(dir).unwrap();
    }
    let sentinel = brain.join("knowledge/sentinel.md");
    std::fs::create_dir_all(sentinel.parent().unwrap()).unwrap();
    std::fs::write(&sentinel, "- user data must stay untouched\n").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_cfetch"))
        .args(["doctor", "--deep", "--json", "--no-network"])
        .env("HOME", &home)
        .env("APPDATA", &home)
        .env("CFETCH_STATE_DIR", &state)
        .env("CFETCH_BRAIN", &brain)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "deep doctor failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let probe = &report["retrieval_probe"];
    assert_eq!(probe["schema_version"], 2);
    assert_eq!(probe["temporary_data"], true);
    assert_eq!(probe["gates"]["production_ready"], false);
    assert_eq!(probe["gates"]["checks"][0]["id"], "bm25");
    assert_eq!(probe["gates"]["checks"][0]["status"], "pass");
    assert_eq!(probe["vector"]["active"], false);
    assert_eq!(
        probe["rankings"]["bm25"][0],
        "knowledge/deployment-metrics.md"
    );
    assert_eq!(
        probe["graph"]["neighbors"][0],
        "knowledge/recovery-checklist.md"
    );
    assert_eq!(probe["graph"]["fixture_edge_active"], true);
    assert_eq!(
        std::fs::read_to_string(&sentinel).unwrap(),
        "- user data must stay untouched\n"
    );
    assert!(!state.join("index.db").exists());

    let gated = Command::new(env!("CARGO_BIN_EXE_cfetch"))
        .args([
            "doctor",
            "--deep",
            "--json",
            "--no-network",
            "--require",
            "vector",
        ])
        .env("HOME", &home)
        .env("APPDATA", &home)
        .env("CFETCH_STATE_DIR", &state)
        .env("CFETCH_BRAIN", &brain)
        .output()
        .unwrap();
    assert!(!gated.status.success());
    let gated_report: serde_json::Value = serde_json::from_slice(&gated.stdout).unwrap();
    assert_eq!(
        gated_report["retrieval_probe"]["gates"]["production_ready"],
        false
    );
    assert!(
        String::from_utf8_lossy(&gated.stderr)
            .contains("required vector gate did not pass: vector_output (not run)")
    );
}
