//! Offline qualification of labeled Markdown through the real index/rankers.
//! No configured brain, endpoint, profile activation, or shared store is opened.
//! Imported candidate outputs are confined to a disposable catalog.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use sha2::Digest;

use crate::config::{Precision, RingRules, VectorSpec};
use crate::{embedding_profile as profile, hashing, index};

#[derive(Clone, Copy, Debug, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Representation {
    Body,
    HeadingContext,
    ContextPayload,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    schema_version: u32,
    description: String,
    documents: Vec<Document>,
    queries: Vec<Query>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    path: String,
    text: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Query {
    id: String,
    category: String,
    text: String,
    relevant: Vec<Relevant>,
    critical: bool,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Relevant {
    path: String,
    text: String,
    grade: u8,
}

#[derive(Serialize)]
struct Block {
    path: String,
    start_line: usize,
    end_line: usize,
    ring: u8,
    cite: String,
    source_hash: String,
    body: String,
    context: String,
    chain: String,
    input_id: String,
    #[serde(skip)]
    payload_hash: String,
}

#[derive(Serialize)]
struct Input {
    id: String,
    kind: &'static str,
    text: String,
}

#[derive(Serialize)]
struct Manifest {
    schema_version: u32,
    purpose: &'static str,
    production_admission: bool,
    representation: Representation,
    corpus_sha256: String,
    description: String,
    baseline_profile_sha256: &'static str,
    dimensions: usize,
    sequence_buckets: &'static [usize],
    hybrid_rrf_k: f64,
    blocks: Vec<Block>,
    inputs: Vec<Input>,
    queries: Vec<Query>,
    query_input_ids: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Bundle {
    schema_version: u32,
    manifest_sha256: String,
    provenance: serde_json::Value,
    outputs: Vec<Output>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Output {
    id: String,
    token_count: usize,
    bucket: Option<usize>,
    vector: Option<Vec<f32>>,
    error: Option<String>,
}

#[derive(Clone, Default, Serialize)]
struct Metrics {
    ndcg_at_10: f64,
    recall_at_5: f64,
    mrr_at_10: f64,
}

fn digest(bytes: &[u8]) -> String {
    hashing::hex_lower(sha2::Sha256::digest(bytes))
}

fn input_id(text: &str) -> String {
    let mut hash = sha2::Sha256::new();
    hash.update(b"cfetch-evaluation-input-v1\0");
    hash.update(text.as_bytes());
    hashing::hex_lower(hash.finalize())
}

fn read_bounded(path: &Path, max: u64) -> anyhow::Result<Vec<u8>> {
    use std::io::Read;
    #[cfg(unix)]
    let file = {
        use rustix::fs::{Mode, OFlags};
        std::fs::File::from(
            rustix::fs::open(
                path,
                OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
                Mode::empty(),
            )
            .with_context(|| format!("open {}", path.display()))?,
        )
    };
    #[cfg(not(unix))]
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    ensure!(
        file.metadata()?.is_file(),
        "{} must be a regular file",
        path.display()
    );
    let mut bytes = Vec::new();
    file.take(max + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= max,
        "{} exceeds {max} bytes",
        path.display()
    );
    Ok(bytes)
}

fn build(
    path: &Path,
    representation: Representation,
) -> anyhow::Result<(
    Manifest,
    rusqlite::Connection,
    tempfile::TempDir,
    tempfile::TempDir,
)> {
    let bytes = read_bounded(path, 4 * 1024 * 1024)?;
    let corpus: Corpus = serde_json::from_slice(&bytes).context("parse labeled corpus")?;
    ensure!(corpus.schema_version == 1, "unsupported corpus schema");
    ensure!(
        !corpus.documents.is_empty() && corpus.documents.len() <= 128,
        "expected 1..128 documents"
    );
    ensure!(
        !corpus.queries.is_empty() && corpus.queries.len() <= 256,
        "expected 1..256 queries"
    );
    let brain = tempfile::tempdir()?;
    let state = tempfile::tempdir()?;
    let mut paths = BTreeSet::new();
    for doc in &corpus.documents {
        let path = Path::new(&doc.path);
        ensure!(
            !doc.path.contains(['\\', ':'])
                && !path.is_absolute()
                && doc
                    .path
                    .split('/')
                    .all(|part| !part.is_empty() && part != "." && part != "..")
                && path.components().all(|c| matches!(c, Component::Normal(_)))
                && path.extension().is_some_and(|ext| ext == "md"),
            "document path must be a relative Markdown path: {:?}",
            doc.path
        );
        ensure!(
            paths.insert(doc.path.clone()),
            "duplicate document path: {}",
            doc.path
        );
        let target = brain.path().join(path);
        std::fs::create_dir_all(target.parent().context("document needs a parent")?)?;
        use std::io::Write;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(target)?
            .write_all(doc.text.as_bytes())?;
    }
    let mut conn = index::open(state.path())?;
    index::scan(&mut conn, brain.path(), None, &RingRules::default())?;
    let mut blocks = Vec::new();
    let mut inputs = BTreeMap::new();
    {
        let mut stmt = conn.prepare(
            "SELECT b.id,d.path,b.start_line,b.end_line,d.ring,b.cite,b.hash,b.text,b.ctx,b.chain,b.embedding_text
             FROM blocks b JOIN docs d ON d.id=b.doc_id ORDER BY b.id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                Block {
                    path: row.get(1)?,
                    start_line: row.get::<_, i64>(2)? as usize,
                    end_line: row.get::<_, i64>(3)? as usize,
                    ring: row.get(4)?,
                    cite: row.get(5)?,
                    source_hash: row.get(6)?,
                    body: row.get(7)?,
                    context: row.get(8)?,
                    chain: row.get(9)?,
                    input_id: String::new(),
                    payload_hash: crate::embedding_input::hash(&row.get::<_, String>(10)?),
                },
                row.get::<_, String>(10)?,
            ))
        })?;
        for row in rows {
            let (id, mut block, payload) = row?;
            let text = match representation {
                Representation::Body => format!("{}{}", profile::DOCUMENT_PREFIX, block.body),
                Representation::HeadingContext if block.context.is_empty() => {
                    format!("{}{}", profile::DOCUMENT_PREFIX, block.body)
                }
                Representation::HeadingContext => {
                    format!("title: {} | text: {}", block.context, block.body)
                }
                Representation::ContextPayload => {
                    format!("{}{}", profile::DOCUMENT_PREFIX, payload)
                }
            };
            block.input_id = input_id(&text);
            inputs.entry(block.input_id.clone()).or_insert(Input {
                id: block.input_id.clone(),
                kind: "document",
                text,
            });
            // Only this temporary evaluation catalog uses the experimental
            // input identity. Cites/source_hash retain real source provenance.
            conn.execute(
                "UPDATE blocks SET embedding_hash=?1 WHERE id=?2",
                rusqlite::params![block.input_id, id],
            )?;
            blocks.push(block);
        }
    }
    ensure!(
        !blocks.is_empty() && blocks.len() <= 4096,
        "expected 1..4096 indexed blocks"
    );
    let mut query_input_ids = BTreeMap::new();
    for query in &corpus.queries {
        ensure!(
            !query.id.is_empty() && !query.text.trim().is_empty(),
            "query id/text must be nonempty"
        );
        ensure!(
            !query.relevant.is_empty(),
            "query {} has no relevance labels",
            query.id
        );
        let mut labels = BTreeSet::new();
        let mut logical_labels = BTreeSet::new();
        for relevant in &query.relevant {
            ensure!(
                (1..=3).contains(&relevant.grade),
                "relevance grade must be 1..3"
            );
            ensure!(
                labels.insert((&relevant.path, &relevant.text)),
                "duplicate relevance label"
            );
            ensure!(
                blocks
                    .iter()
                    .filter(|b| b.path == relevant.path && b.body == relevant.text)
                    .count()
                    == 1,
                "query {} relevance label does not identify exactly one indexed block: {} {:?}",
                query.id,
                relevant.path,
                relevant.text
            );
            let block = blocks
                .iter()
                .find(|b| b.path == relevant.path && b.body == relevant.text)
                .unwrap();
            ensure!(
                logical_labels.insert((&block.source_hash, &block.payload_hash)),
                "query {} labels multiple mirrors of one logical statement",
                query.id
            );
        }
        let text = format!("{}{}", profile::QUERY_PREFIX, query.text);
        let id = input_id(&text);
        ensure!(
            query_input_ids
                .insert(query.id.clone(), id.clone())
                .is_none(),
            "duplicate query id"
        );
        inputs.entry(id.clone()).or_insert(Input {
            id,
            kind: "query",
            text,
        });
    }
    Ok((
        Manifest {
            schema_version: 1,
            purpose: "isolated candidate qualification; not production admission",
            production_admission: false,
            representation,
            corpus_sha256: digest(&bytes),
            description: corpus.description,
            baseline_profile_sha256: profile::PROFILE_MANIFEST_SHA256,
            dimensions: profile::DIMENSIONS,
            sequence_buckets: profile::SEQUENCE_BUCKETS,
            hybrid_rrf_k: crate::config::RecallConfig::default().rrf_k,
            blocks,
            inputs: inputs.into_values().collect(),
            queries: corpus.queries,
            query_input_ids,
        },
        conn,
        brain,
        state,
    ))
}

fn measure(query: &Query, hits: &[index::Hit]) -> (Metrics, serde_json::Value) {
    let mut seen = BTreeSet::new();
    let mut gains = Vec::new();
    let mut ranks = Vec::new();
    let mut found_at_5 = 0usize;
    for (rank, hit) in hits.iter().take(10).enumerate() {
        let mut grade = 0u8;
        for (label, relevant) in query.relevant.iter().enumerate() {
            if hit.text == relevant.text
                && (hit.path == relevant.path || hit.mirrors.contains(&relevant.path))
                && seen.insert(label)
            {
                grade = grade.max(relevant.grade);
                if rank < 5 {
                    found_at_5 += 1;
                }
            }
        }
        gains.push(grade);
        ranks.push(
            serde_json::json!({"rank":rank+1,"grade":grade,"path":hit.path,
            "cite":hit.cite,"ring":hit.ring,"text":hit.text,"context":hit.chain}),
        );
    }
    let dcg = |values: &[u8]| {
        values
            .iter()
            .take(10)
            .enumerate()
            .map(|(rank, &grade)| ((1u32 << grade) - 1) as f64 / ((rank + 2) as f64).log2())
            .sum::<f64>()
    };
    let mut ideal: Vec<u8> = query.relevant.iter().map(|r| r.grade).collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let metrics = Metrics {
        ndcg_at_10: dcg(&gains) / dcg(&ideal),
        recall_at_5: found_at_5 as f64 / query.relevant.len() as f64,
        mrr_at_10: gains
            .iter()
            .position(|&g| g > 0)
            .map_or(0.0, |rank| 1.0 / (rank + 1) as f64),
    };
    (
        metrics.clone(),
        serde_json::json!({"metrics":metrics,"ranking":ranks}),
    )
}

fn validate_output(output: &Output) -> anyhow::Result<()> {
    let bucket = profile::SEQUENCE_BUCKETS
        .iter()
        .copied()
        .find(|&b| b >= output.token_count);
    ensure!(
        output.token_count > 0 && output.bucket == bucket,
        "incorrect token count/bucket for {}",
        output.id
    );
    match &output.vector {
        Some(vector) => {
            ensure!(
                bucket.is_some() && output.error.is_none(),
                "vector supplied for refused input"
            );
            ensure!(
                vector.len() == profile::DIMENSIONS,
                "wrong vector width for {}",
                output.id
            );
            ensure!(
                vector.iter().all(|v| v.is_finite()),
                "nonfinite vector for {}",
                output.id
            );
            let norm = vector.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
            ensure!(
                (0.99..=1.01).contains(&norm),
                "vector is not unit normalized for {}",
                output.id
            );
        }
        None => ensure!(
            output.error.as_ref().is_some_and(|e| !e.trim().is_empty()),
            "missing vector needs explicit reason"
        ),
    }
    Ok(())
}

pub fn run(
    corpus: &Path,
    representation: Representation,
    export: Option<&Path>,
    vectors: Option<&Path>,
) -> anyhow::Result<()> {
    let (manifest, conn, _brain, _state) = build(corpus, representation)?;
    let bytes = serde_json::to_vec_pretty(&manifest)?;
    let manifest_sha256 = digest(&bytes);
    if let Some(path) = export {
        use std::io::Write;
        // Existing evidence must not be silently replaced.
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(&bytes)?;
    }
    let mut outputs = BTreeMap::new();
    let mut provenance = serde_json::Value::Null;
    let mut vector_bundle_sha256 = None;
    let spec = VectorSpec {
        network_major: profile::NETWORK_MAJOR,
        profile_id: "isolated-memory-evaluation".into(),
        model: "unadmitted-candidate".into(),
        dim: profile::DIMENSIONS,
        precision: Precision::I8,
        doc_prefix: profile::DOCUMENT_PREFIX.into(),
    };
    if let Some(path) = vectors {
        let bundle_bytes = read_bounded(path, 64 * 1024 * 1024)?;
        vector_bundle_sha256 = Some(digest(&bundle_bytes));
        let bundle: Bundle = serde_json::from_slice(&bundle_bytes)?;
        ensure!(
            bundle.schema_version == 1 && bundle.manifest_sha256 == manifest_sha256,
            "vector bundle does not match exact input manifest"
        );
        ensure!(
            bundle.provenance.is_object() && !bundle.provenance.as_object().unwrap().is_empty(),
            "candidate provenance is required"
        );
        ensure!(
            bundle.outputs.len() == manifest.inputs.len(),
            "vector bundle must account for every input, including refusals"
        );
        for output in bundle.outputs {
            validate_output(&output)?;
            ensure!(
                outputs.insert(output.id.clone(), output).is_none(),
                "duplicate vector output"
            );
        }
        for input in &manifest.inputs {
            ensure!(
                outputs.contains_key(&input.id),
                "missing output for {}",
                input.id
            );
        }
        provenance = bundle.provenance;
        for block in &manifest.blocks {
            if let Some(vector) = &outputs[&block.input_id].vector {
                index::insert_vector(&conn, &block.input_id, &spec, vector)?;
            }
        }
    }
    let documents_available = manifest
        .inputs
        .iter()
        .filter(|input| {
            input.kind == "document"
                && outputs
                    .get(&input.id)
                    .is_some_and(|output| output.vector.is_some())
        })
        .count();
    let mut results = Vec::new();
    let mut aggregate: BTreeMap<(String, String), Vec<Metrics>> = BTreeMap::new();
    for query in &manifest.queries {
        let mut modes = BTreeMap::new();
        let lexical = index::recall(&conn, &query.text, 10)?;
        let mut rankings = vec![("bm25", lexical)];
        let query_vector = outputs
            .get(&manifest.query_input_ids[&query.id])
            .and_then(|o| o.vector.as_ref());
        if documents_available > 0
            && let Some(vector) = query_vector
        {
            rankings.push((
                "vector",
                index::semantic_recall(&conn, &spec, vector, 10, &[])?,
            ));
            rankings.push((
                "hybrid",
                index::hybrid_recall(
                    &conn,
                    &spec,
                    &query.text,
                    vector,
                    10,
                    manifest.hybrid_rrf_k,
                    &[],
                )?,
            ));
        } else if vectors.is_some() {
            for mode in ["vector", "hybrid"] {
                // Refused queries and an empty vector corpus are misses, not
                // successful lexical fallback or dropped measurement rows.
                for category in ["all", query.category.as_str()] {
                    aggregate
                        .entry((category.into(), mode.into()))
                        .or_default()
                        .push(Metrics::default());
                }
                modes.insert(mode,serde_json::json!({"status":"unavailable",
                    "reason":if documents_available==0 {"no document vectors"} else {"query embedding refused"},
                    "metrics":Metrics::default(),"ranking":[]}));
            }
        }
        for (mode, hits) in rankings {
            let (metrics, result) = measure(query, &hits);
            for category in ["all", query.category.as_str()] {
                aggregate
                    .entry((category.into(), mode.into()))
                    .or_default()
                    .push(metrics.clone());
            }
            modes.insert(mode, result);
        }
        results.push(serde_json::json!({"id":query.id,"category":query.category,"critical":query.critical,"query":query.text,"modes":modes}));
    }
    let means: Vec<_> = aggregate
        .into_iter()
        .map(|((category, mode), values)| {
            let n = values.len() as f64;
            serde_json::json!({"category":category,"mode":mode,"queries_evaluated":values.len(),
            "ndcg_at_10":values.iter().map(|m|m.ndcg_at_10).sum::<f64>()/n,
            "recall_at_5":values.iter().map(|m|m.recall_at_5).sum::<f64>()/n,
            "mrr_at_10":values.iter().map(|m|m.mrr_at_10).sum::<f64>()/n})
        })
        .collect();
    let refused: Vec<_> = outputs.values().filter(|o| o.vector.is_none()).collect();
    let report = serde_json::json!({"schema_version":1,"production_admission":false,
        "purpose":"candidate measurements using production segmentation, INT8 codec and rankers; no reranking or graph expansion",
        "representation":representation,"manifest_sha256":manifest_sha256,
        "vector_measurement":if vectors.is_none(){"not_run"}else if documents_available==0{"unavailable"}else if !refused.is_empty(){"partial"}else{"measured"},
        "hybrid_rrf_k":manifest.hybrid_rrf_k,"document_vectors_available":documents_available,
        "blocks":manifest.blocks.len(),"inputs":manifest.inputs.len(),"queries":manifest.queries.len(),
        "refused":refused,"vector_bundle_sha256":vector_bundle_sha256,
        "provenance_verification":"runner-reported; not admission or physical placement evidence",
        "provenance":provenance,"summary":means,"results":results});
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(root: &Path, doc_path: &str, relevant: &str) -> std::path::PathBuf {
        let path = root.join("corpus.json");
        std::fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({
                "schema_version":1,"description":"synthetic test",
                "documents":[{"path":doc_path,"text":"# Storage\n\n- Keep backups.\n"}],
                "queries":[{"id":"q","category":"policy","text":"backup","critical":true,
                    "relevant":[{"path":doc_path,"text":relevant,"grade":3}]}]
            }))
            .unwrap(),
        )
        .unwrap();
        path
    }

    #[test]
    fn refuses_path_escape_and_unindexed_labels() {
        let root = tempfile::tempdir().unwrap();
        assert!(
            build(
                &fixture(root.path(), "../escaped.md", "- Keep backups."),
                Representation::Body
            )
            .is_err()
        );
        assert!(
            build(
                &fixture(root.path(), "knowledge/ok.md", "invented answer"),
                Representation::Body
            )
            .is_err()
        );
        assert!(!root.path().join("escaped.md").exists());
    }

    #[test]
    fn exact_inputs_and_experimental_context_have_distinct_identity() {
        let root = tempfile::tempdir().unwrap();
        let path = fixture(root.path(), "knowledge/storage.md", "- Keep backups.");
        let (body, _, _, _) = build(&path, Representation::Body).unwrap();
        let (context, _, _, _) = build(&path, Representation::HeadingContext).unwrap();
        let b = body
            .blocks
            .iter()
            .find(|b| b.body == "- Keep backups.")
            .unwrap();
        let c = context
            .blocks
            .iter()
            .find(|block| block.body == "- Keep backups.")
            .unwrap();
        assert_eq!(b.source_hash, c.source_hash);
        assert_eq!(b.cite, c.cite);
        assert_ne!(b.input_id, c.input_id);
        assert_eq!(
            body.inputs
                .iter()
                .find(|i| i.id == b.input_id)
                .unwrap()
                .text,
            "title: none | text: - Keep backups."
        );
        assert_eq!(
            context
                .inputs
                .iter()
                .find(|i| i.id == c.input_id)
                .unwrap()
                .text,
            "title: Storage | text: - Keep backups."
        );
    }

    #[test]
    fn candidate_outputs_require_explicit_refusal_or_valid_normalized_vector() {
        let mut output = Output {
            id: "x".into(),
            token_count: 2050,
            bucket: None,
            vector: None,
            error: Some("overlong".into()),
        };
        assert!(validate_output(&output).is_ok());
        output.error = None;
        assert!(validate_output(&output).is_err());
        output.token_count = 20;
        output.bucket = Some(32);
        output.vector = Some(vec![0.0; 768]);
        assert!(validate_output(&output).is_err());
        output.vector.as_mut().unwrap()[0] = 1.0;
        assert!(validate_output(&output).is_ok());
        output.bucket = Some(64);
        assert!(validate_output(&output).is_err());
    }
}
