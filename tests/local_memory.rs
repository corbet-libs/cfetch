//! Explicit real-model acceptance check, run only on a build worker with a
//! qualified model pack. No downloads or hosted inference.
#![cfg(feature = "embedded-embeddings")]
use serde_json::{Value, json};
use std::{path::Path, process::Command};

fn command(root: &Path, args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_cfetch"))
        .args(args)
        .env("CFETCH_BRAIN", root.join("brain"))
        .env("CFETCH_MIND", "test-mind")
        .env("CFETCH_STATE_DIR", root.join("state"))
        .env("CFETCH_CONFIG", root.join("config.json"))
        .env("HOME", root)
        .env("HF_ENDPOINT", "http://127.0.0.1:1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
#[ignore = "requires CFETCH_TEST_LOCAL_MODEL; invoked by the local-model CI check"]
fn offline_vectors_retrieve_meaning_and_drop_edited_sources() {
    let model = std::env::var("CFETCH_TEST_LOCAL_MODEL").expect("explicit qualified pack");
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let notes = root.join("brain/knowledge");
    std::fs::create_dir_all(&notes).unwrap();
    std::fs::write(
        notes.join("animals.md"),
        "A cat is a small domesticated feline animal that purrs.\n",
    )
    .unwrap();
    std::fs::write(
        notes.join("music.md"),
        "A violin is a musical instrument with four strings played using a bow.\n",
    )
    .unwrap();
    std::fs::write(
        root.join("config.json"),
        json!({
            "resident": [], "capture": {"enabled": false},
            "embeddings": {"enabled": true, "local_model": model}
        })
        .to_string(),
    )
    .unwrap();
    command(root, &["embed-index", "--batch", "1"]);
    let raw = command(
        root,
        &["recall", "purring household pet", "--semantic", "--json"],
    );
    let first: Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(first["note"], Value::Null, "{first}");
    assert_eq!(first["hits"][0]["path"], "knowledge/animals.md", "{first}");
    let old_cite = first["hits"][0]["cite"].as_str().unwrap();
    std::fs::write(
        notes.join("animals.md"),
        "The planet Saturn has prominent rings of ice and rock.\n",
    )
    .unwrap();
    let changed: Value = serde_json::from_str(&command(
        root,
        &["recall", "purring household pet", "--hybrid", "--json"],
    ))
    .unwrap();
    assert!(
        changed["note"].is_string(),
        "new content must expose incomplete vector coverage: {changed}"
    );
    assert!(
        !changed["hits"]
            .as_array()
            .unwrap()
            .iter()
            .any(|hit| hit["cite"] == old_cite)
    );
    assert!(command(root, &["recall", "--id", old_cite]).contains("no block"));
    command(root, &["embed-index", "--batch", "1"]);
    let next: Value = serde_json::from_str(&command(
        root,
        &["recall", "planet with icy rings", "--semantic", "--json"],
    ))
    .unwrap();
    assert_eq!(next["note"], Value::Null, "{next}");
    assert_eq!(next["hits"][0]["path"], "knowledge/animals.md", "{next}");
    assert!(
        next["hits"][0]["snippet"]
            .as_str()
            .unwrap()
            .contains("Saturn")
    );
}
