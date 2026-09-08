//! Offline evaluation must account for failed vectors and bind its evidence
//! without touching configured memory or invoking an inference runtime.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};
use sha2::Digest as _;

struct Fixture {
    root: tempfile::TempDir,
    corpus: PathBuf,
    representation: &'static str,
}

impl Fixture {
    fn new(corpus: &Value) -> Self {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("corpus.json");
        std::fs::write(&path, serde_json::to_vec_pretty(corpus).unwrap()).unwrap();
        Self { root, corpus: path, representation: "body" }
    }

    fn run(&self, export: Option<&Path>, vectors: Option<&Path>) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_cfetch"));
        command
            .args(["retrieval-eval", "--corpus"])
            .arg(&self.corpus)
            .args(["--representation", self.representation]);
        if let Some(path) = export {
            command.arg("--export").arg(path);
        }
        if let Some(path) = vectors {
            command.arg("--vectors").arg(path);
        }
        command
            .env("HOME", self.root.path().join("home"))
            .env("APPDATA", self.root.path().join("home"))
            .env("CFETCH_STATE_DIR", self.root.path().join("state"))
            .env("CFETCH_BRAIN", self.root.path().join("brain"))
            .output()
            .unwrap()
    }

    fn export(&self) -> Vec<u8> {
        let path = self.root.path().join("manifest.json");
        let output = self.run(Some(&path), None);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::read(path).unwrap()
    }

    fn bundle(&self, manifest_bytes: &[u8], refuse: impl Fn(&Value) -> bool) -> PathBuf {
        let manifest: Value = serde_json::from_slice(manifest_bytes).unwrap();
        let mut vector = vec![0.0_f32; manifest["dimensions"].as_u64().unwrap() as usize];
        vector[0] = 1.0;
        let outputs: Vec<Value> = manifest["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|input| {
                let refused = refuse(input);
                json!({
                    "id": input["id"], "token_count": 20, "bucket": 32,
                    "vector": if refused { None } else { Some(&vector) },
                    "error": if refused { Some("synthetic refusal; no inference attempted") } else { None }
                })
            })
            .collect();
        let path = self.root.path().join("vectors.json");
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&json!({
                "schema_version": 1, "manifest_sha256": digest(manifest_bytes),
                "provenance": {"fixture": "synthetic unit vectors; no inference"},
                "outputs": outputs
            }))
            .unwrap(),
        )
        .unwrap();
        path
    }

    fn assert_isolated(&self) {
        assert!(
            !self.root.path().join("state").exists(),
            "evaluation opened configured state"
        );
        assert!(
            !self.root.path().join("brain").exists(),
            "evaluation opened configured memory"
        );
    }
}

fn digest(bytes: &[u8]) -> String {
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn corpus() -> Value {
    json!({
        "schema_version": 1, "description": "synthetic CLI regression",
        "documents": [{"path": "knowledge/backups.md", "text": "- Keep backups.\n"}],
        "queries": [
            {"id": "available", "category": "policy", "text": "Keep backups", "critical": true,
             "relevant": [{"path": "knowledge/backups.md", "text": "- Keep backups.", "grade": 3}]},
            {"id": "refused", "category": "policy", "text": "How should archives be preserved", "critical": false,
             "relevant": [{"path": "knowledge/backups.md", "text": "- Keep backups.", "grade": 3}]}
        ]
    })
}

#[test]
fn context_payload_keeps_the_fixed_prefix_and_body_citations() {
    let mut fixture = Fixture::new(&json!({
        "schema_version": 1, "description": "context identity regression",
        "documents": [
            {"path": "knowledge/alpha.md", "text": "# Alpha\n\n- Keep backups.\n"},
            {"path": "knowledge/beta.md", "text": "# Beta\n\n- Keep backups.\n"}
        ],
        "queries": [{"id": "alpha", "category": "context", "text": "Alpha backups",
            "critical": false,
            "relevant": [{"path": "knowledge/alpha.md", "text": "- Keep backups.", "grade": 3}]}]
    }));
    fixture.representation = "context-payload";
    let manifest: Value = serde_json::from_slice(&fixture.export()).unwrap();
    let blocks: Vec<&Value> = manifest["blocks"].as_array().unwrap().iter()
        .filter(|block| block["body"] == "- Keep backups.").collect();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0]["cite"], blocks[1]["cite"]);
    assert_eq!(blocks[0]["source_hash"], blocks[1]["source_hash"]);
    assert_ne!(blocks[0]["input_id"], blocks[1]["input_id"]);
    let inputs = manifest["inputs"].as_array().unwrap();
    for (block, heading) in blocks.iter().zip(["Alpha", "Beta"]) {
        let input = inputs.iter().find(|input| input["id"] == block["input_id"]).unwrap();
        assert_eq!(input["text"], format!("title: none | text: {heading}\n\n- Keep backups."));
    }
    fixture.assert_isolated();
}

fn report(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn summary<'a>(report: &'a Value, category: &str, mode: &str) -> &'a Value {
    report["summary"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["category"] == category && row["mode"] == mode)
        .unwrap()
}

#[test]
fn refused_query_counts_as_zero_in_the_full_vector_and_hybrid_denominator() {
    let fixture = Fixture::new(&corpus());
    let bytes = fixture.export();
    let manifest: Value = serde_json::from_slice(&bytes).unwrap();
    let refused_id = &manifest["query_input_ids"]["refused"];
    let bundle = fixture.bundle(&bytes, |input| input["id"] == *refused_id);
    let report = report(fixture.run(None, Some(&bundle)));

    assert_eq!(report["vector_measurement"], "partial");
    assert_eq!(report["document_vectors_available"], 1);
    assert_eq!(report["refused"].as_array().unwrap().len(), 1);
    assert_eq!(report["production_admission"], false);
    assert_eq!(report["hybrid_rrf_k"], 2.0);
    assert_eq!(
        report["vector_bundle_sha256"],
        digest(&std::fs::read(bundle).unwrap())
    );
    for mode in ["vector", "hybrid"] {
        for category in ["all", "policy"] {
            let aggregate = summary(&report, category, mode);
            assert_eq!(aggregate["queries_evaluated"], 2);
            for metric in ["ndcg_at_10", "recall_at_5", "mrr_at_10"] {
                assert_eq!(
                    aggregate[metric], 0.5,
                    "{category}/{mode}/{metric}: {report}"
                );
            }
        }
        let refused = &report["results"][1]["modes"][mode];
        assert_eq!(refused["status"], "unavailable");
        assert_eq!(refused["reason"], "query embedding refused");
        assert!(refused["ranking"].as_array().unwrap().is_empty());
    }
    fixture.assert_isolated();
}

#[test]
fn no_document_vectors_cannot_be_reported_as_successful_lexical_hybrid() {
    let fixture = Fixture::new(&corpus());
    let bytes = fixture.export();
    let bundle = fixture.bundle(&bytes, |input| input["kind"] == "document");
    let report = report(fixture.run(None, Some(&bundle)));

    assert_eq!(report["vector_measurement"], "unavailable");
    assert_eq!(report["document_vectors_available"], 0);
    assert_eq!(
        report["results"][0]["modes"]["bm25"]["metrics"]["mrr_at_10"],
        1.0
    );
    for mode in ["vector", "hybrid"] {
        let aggregate = summary(&report, "all", mode);
        assert_eq!(aggregate["queries_evaluated"], 2);
        for metric in ["ndcg_at_10", "recall_at_5", "mrr_at_10"] {
            assert_eq!(aggregate[metric], 0.0, "{mode}/{metric}: {report}");
        }
        for query in report["results"].as_array().unwrap() {
            assert_eq!(query["modes"][mode]["status"], "unavailable");
            assert_eq!(query["modes"][mode]["reason"], "no document vectors");
            assert!(
                query["modes"][mode]["ranking"]
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
        }
    }
    fixture.assert_isolated();
}

#[test]
fn vectors_are_bound_to_exact_manifest_bytes_even_for_equivalent_corpus_json() {
    let fixture = Fixture::new(&corpus());
    let bytes = fixture.export();
    let bundle = fixture.bundle(&bytes, |_| false);
    assert_eq!(
        report(fixture.run(None, Some(&bundle)))["vector_measurement"],
        "measured"
    );

    let mut corpus_bytes = std::fs::read(&fixture.corpus).unwrap();
    corpus_bytes.push(b'\n');
    std::fs::write(&fixture.corpus, corpus_bytes).unwrap();
    let output = fixture.run(None, Some(&bundle));
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("does not match exact input manifest")
    );
    assert!(
        output.stdout.is_empty(),
        "mismatched evidence must not produce measurements"
    );
    fixture.assert_isolated();
}

#[test]
fn fixture_path_aliases_are_rejected_before_they_can_overwrite_documents() {
    for alias in [
        "knowledge//backups.md",
        "knowledge/./backups.md",
        "knowledge/backups.md",
    ] {
        let mut corpus = corpus();
        corpus["documents"].as_array_mut().unwrap().push(json!({
            "path": alias, "text": "- Replacement text must never overwrite the fixture.\n"
        }));
        let fixture = Fixture::new(&corpus);
        let output = fixture.run(None, None);
        assert!(!output.status.success(), "accepted alias {alias}");
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("document path must be") || error.contains("duplicate document path"),
            "{error}"
        );
        assert!(output.stdout.is_empty());
        fixture.assert_isolated();
    }
}

#[test]
fn mirror_labels_cannot_inflate_the_ideal_relevance_denominator() {
    let mut corpus = corpus();
    corpus["documents"].as_array_mut().unwrap().push(json!({
        "path": "knowledge/mirror.md", "text": "---\nring: 1\n---\n- Keep backups.\n"
    }));
    corpus["queries"][0]["relevant"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "path": "knowledge/mirror.md", "text": "- Keep backups.", "grade": 3
        }));
    let fixture = Fixture::new(&corpus);
    let output = fixture.run(None, None);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("labels multiple mirrors of one logical statement")
    );
    assert!(output.stdout.is_empty());
    fixture.assert_isolated();
}
