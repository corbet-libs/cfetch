//! Model diagnostics must never activate, rewrite, or infer compatibility
//! with the shared semantic profile.
#![cfg(feature = "embedded-embeddings")]

use std::process::Command;

fn diagnostic(root: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_cfetch"))
        .arg("embed-model")
        .args(args)
        .env("CFETCH_STATE_DIR", root.join("state"))
        .env("CFETCH_BRAIN", root.join("brain"))
        .env_remove("HF_HOME")
        .env("HF_ENDPOINT", "http://127.0.0.1:1")
        .output()
        .unwrap()
}

#[test]
fn status_reports_the_requested_model_without_loading_or_creating_state() {
    let root = tempfile::tempdir().unwrap();
    for model in ["MultilingualE5Base", "EmbeddingGemma300M", "AllMiniLML6V2"] {
        let output = diagnostic(root.path(), &["--model", model, "status"]);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.contains(model), "{text}");
        assert!(text.contains("diagnostic only"), "{text}");
        assert!(text.contains("not cached"), "{text}");
        assert!(text.contains("loadability: not checked"), "{text}");
    }
    assert!(!root.path().join("state").exists());
    assert!(!root.path().join("brain").exists());
}

#[test]
fn catalog_exposes_real_selectable_models_without_shared_compatibility_claims() {
    let root = tempfile::tempdir().unwrap();
    let output = diagnostic(root.path(), &["list"]);
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("EmbeddingGemma300M"), "{text}");
    assert!(text.contains("EmbeddingGemma300MQ4"), "{text}");
    assert!(text.contains("diagnostic only"), "{text}");
    assert!(!text.contains("cfetch canonical"), "{text}");
    assert!(!root.path().join("state").exists());
}

#[test]
fn status_respects_hf_home_without_claiming_an_incomplete_cache_is_loadable() {
    let root = tempfile::tempdir().unwrap();
    let cache = root.path().join("model-cache");
    let info =
        fastembed::TextEmbedding::get_model_info(&fastembed::EmbeddingModel::EmbeddingGemma300M)
            .unwrap();
    std::fs::create_dir_all(cache.join(format!("models--{}", info.model_code.replace('/', "--"))))
        .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_cfetch"))
        .args(["embed-model", "status", "--model", "EmbeddingGemma300M"])
        .env("CFETCH_STATE_DIR", root.path().join("state"))
        .env("HF_HOME", &cache)
        .env("HF_ENDPOINT", "http://127.0.0.1:1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        text.contains(&format!("cache dir: {}", cache.display())),
        "{text}"
    );
    assert!(
        text.contains("present (file completeness unverified)"),
        "{text}"
    );
    assert!(text.contains("loadability: not checked"), "{text}");
    assert!(!root.path().join("state").exists());
}

#[test]
fn unknown_models_and_store_switching_are_rejected_without_side_effects() {
    let root = tempfile::tempdir().unwrap();
    for args in [
        vec!["--model", "not-a-model", "download"],
        vec!["--model", "EmbeddingGemma300M-made-up", "download"],
        vec!["switch-to-shared"],
        vec!["check-compat"],
    ] {
        let output = diagnostic(root.path(), &args);
        assert!(!output.status.success(), "unexpected success for {args:?}");
    }
    assert!(!root.path().join("state").exists());
    assert!(!root.path().join("brain").exists());
}
