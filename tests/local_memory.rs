//! Explicit real-model acceptance check, run only on a build worker with a
//! qualified model pack. No downloads or hosted inference.
#![cfg(feature = "embedded-embeddings")]
use serde_json::{Value, json};
use std::{
    path::Path,
    process::{Child, Command, Stdio},
    time::Duration,
};

fn configured(root: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_cfetch"));
    cmd.env("CFETCH_BRAIN", root.join("brain"))
        .env("CFETCH_MIND", "test-mind")
        .env("CFETCH_STATE_DIR", root.join("state"))
        .env("CFETCH_CONFIG", root.join("config.json"))
        .env("HOME", root)
        .env("HF_ENDPOINT", "http://127.0.0.1:1")
        .env_remove("XDG_RUNTIME_DIR");
    cmd
}

struct Daemon(Child);
impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn start_daemon(root: &Path) -> Daemon {
    let child = configured(root)
        .args(["daemon", "run"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let daemon = Daemon(child);
    let endpoint = root.join("state").join(if cfg!(windows) {
        "daemon.endpoint"
    } else {
        "daemon.sock"
    });
    for _ in 0..200 {
        if endpoint.exists() {
            return daemon;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("test daemon did not publish an endpoint");
}

fn command(root: &Path, args: &[&str]) -> String {
    let mut cmd = configured(root);
    cmd.args(args);
    if args.first() == Some(&"recall") {
        cmd.arg("--fresh");
    }
    let output = cmd.output().unwrap();
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
    let _daemon = start_daemon(root);
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
    assert_eq!(changed["fresh"], true, "{changed}");
    // The daemon may already have embedded the replacement; regardless of
    // that race, the old content identity must never survive the strict pass.
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
