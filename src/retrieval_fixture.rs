//! A small, disposable retrieval fixture for checking the configured
//! embedding and reranking routes.
//!
//! The fixture never opens the configured brain or vector store. It builds a
//! temporary Markdown tree, runs the real retrieval components over it, and
//! reports each stage separately so an inactive vector path cannot look like
//! a successful hybrid result.

use anyhow::Context as _;
use serde::Serialize;
use sha2::Digest as _;

use crate::config::{Config, Precision};
use crate::{embed, embedding_profile, hashing, index, rerank};

const QUERY: &str = "database deployment failure rollback";
const LIMIT: usize = 3;
const PREVIEW_COMPONENTS: usize = 12;
const LEXICAL_DECOY: &str = "knowledge/deployment-metrics.md";
const LINK_SOURCE: &str = "knowledge/restore-release.md";
const LINK_TARGET: &str = "knowledge/recovery-checklist.md";
const SEMANTIC_TARGET: &str = "knowledge/schema-rollback.md";

struct FixtureDocument {
    path: &'static str,
    text: &'static str,
}

const DOCUMENTS: &[FixtureDocument] = &[
    FixtureDocument {
        path: "knowledge/restore-release.md",
        text: "- Restore the previous release after a failed deployment. See [[recovery-checklist]].\n",
    },
    FixtureDocument {
        path: "knowledge/schema-rollback.md",
        text: "- Roll back the schema migration and return to the last known-good version.\n",
    },
    FixtureDocument {
        path: "knowledge/deployment-metrics.md",
        text: "- The database records failed deployment counts for the weekly analytics report.\n",
    },
    FixtureDocument {
        path: "knowledge/music.md",
        text: "- A string quartet rehearses a new piece for the evening concert.\n",
    },
    FixtureDocument {
        path: "knowledge/recovery-checklist.md",
        text: "- Check service health and confirm users can complete requests before closing the incident.\n",
    },
];

#[derive(Debug, Clone, Serialize)]
pub struct RetrievalFixtureReport {
    schema_version: u32,
    temporary_data: bool,
    query: &'static str,
    gates: GateReport,
    vector: VectorReport,
    rankings: RankingReport,
    graph: GraphReport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Requirement {
    Bm25,
    Profile,
    Vector,
    Hybrid,
    Reranker,
    Graph,
    LocalAcceleration,
    Production,
}

impl Requirement {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bm25 => "bm25",
            Self::Profile => "profile",
            Self::Vector => "vector",
            Self::Hybrid => "hybrid",
            Self::Reranker => "reranker",
            Self::Graph => "graph",
            Self::LocalAcceleration => "local-acceleration",
            Self::Production => "production",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum GateStatus {
    Pass,
    Fail,
    NotRun,
}

impl GateStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::NotRun => "NOT RUN",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct GateReport {
    production_ready: bool,
    passed: usize,
    failed: usize,
    not_run: usize,
    checks: Vec<GateCheck>,
}

#[derive(Debug, Clone, Serialize)]
struct GateCheck {
    id: &'static str,
    status: GateStatus,
    production_required: bool,
    checks: &'static str,
    evidence: String,
}

#[derive(Debug, Clone, Serialize)]
struct VectorReport {
    purpose: &'static str,
    configured: bool,
    active: bool,
    route: &'static str,
    model: String,
    dimensions: usize,
    encoding: String,
    profile_status: &'static str,
    reason: Option<String>,
    execution: Option<ExecutionReport>,
    query: Option<VectorOutput>,
    documents: Vec<DocumentVector>,
}

#[derive(Debug, Clone, Serialize)]
struct DocumentVector {
    path: String,
    cosine_to_query: f32,
    output: VectorOutput,
}

#[derive(Debug, Clone, Serialize)]
struct ExecutionReport {
    backend: String,
    device_class: Option<String>,
    route: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct VectorOutput {
    sha256: String,
    norm: f32,
    preview: Components,
    #[serde(skip_serializing_if = "Option::is_none")]
    components: Option<Components>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum Components {
    SignedInt8(Vec<i8>),
    Float(Vec<f32>),
}

#[derive(Debug, Clone, Serialize)]
struct RankingReport {
    bm25_role: &'static str,
    bm25: Vec<String>,
    vector_role: &'static str,
    vector: Option<Vec<String>>,
    hybrid_role: &'static str,
    hybrid_rrf_k: f64,
    hybrid: Option<Vec<String>>,
    reranker: RerankerReport,
}

#[derive(Debug, Clone, Serialize)]
struct RerankerReport {
    purpose: &'static str,
    status: &'static str,
    input: &'static str,
    /// Size of the shortlist handed to the reranker. The gate requires one
    /// score per input item, which can legitimately be fewer than `LIMIT`
    /// when BM25 returns fewer hits.
    input_size: usize,
    model: Option<String>,
    reason: Option<String>,
    execution: Option<ExecutionReport>,
    ranking: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
struct GraphReport {
    role: &'static str,
    from: &'static str,
    neighbors: Vec<String>,
    fixture_edge_from: &'static str,
    fixture_edge_to: &'static str,
    fixture_edge_active: bool,
}

fn position(paths: &[String], target: &str) -> Option<usize> {
    paths.iter().position(|path| path == target)
}

fn gate(
    id: &'static str,
    status: GateStatus,
    production_required: bool,
    checks: &'static str,
    evidence: impl Into<String>,
) -> GateCheck {
    GateCheck {
        id,
        status,
        production_required,
        checks,
        evidence: evidence.into(),
    }
}

fn evaluate_gates(
    cfg: &Config,
    vector: &VectorReport,
    rankings: &RankingReport,
    graph: &GraphReport,
) -> GateReport {
    let bm25_passed = rankings
        .bm25
        .first()
        .is_some_and(|path| path == LEXICAL_DECOY)
        && rankings.bm25.iter().any(|path| path == LINK_SOURCE);
    let (profile_passed, profile_evidence) = if cfg.embeddings.model != embedding_profile::MODEL {
        (
            false,
            format!(
                "configured model {} is not the canonical shared-vector model {}",
                cfg.embeddings.model,
                embedding_profile::MODEL
            ),
        )
    } else {
        match embedding_profile::production_availability() {
            Ok(()) => (
                true,
                format!(
                    "model {} is active with at least one admitted backend",
                    cfg.embeddings.model
                ),
            ),
            Err(error) => (false, one_line(error)),
        }
    };
    let vector_passed =
        vector.active && vector.query.is_some() && vector.documents.len() == DOCUMENTS.len();

    let vector_status = if vector_passed {
        GateStatus::Pass
    } else if vector.configured {
        GateStatus::Fail
    } else {
        GateStatus::NotRun
    };
    let semantic_status = match rankings.vector.as_deref() {
        Some(paths)
            if position(paths, SEMANTIC_TARGET)
                .zip(position(paths, LEXICAL_DECOY))
                .is_some_and(|(semantic, decoy)| semantic < decoy) =>
        {
            GateStatus::Pass
        }
        Some(_) => GateStatus::Fail,
        None if vector.configured => GateStatus::Fail,
        None => GateStatus::NotRun,
    };
    let hybrid_status = match rankings.hybrid.as_deref() {
        Some(paths)
            if paths.len() == LIMIT
                && position(paths, SEMANTIC_TARGET).is_some()
                && position(paths, LEXICAL_DECOY).is_some() =>
        {
            GateStatus::Pass
        }
        Some(_) => GateStatus::Fail,
        None if vector.configured => GateStatus::Fail,
        None => GateStatus::NotRun,
    };
    let reranker_status = match rankings.reranker.status {
        "active"
            if rankings.reranker.input_size > 0
                && rankings
                    .reranker
                    .ranking
                    .as_ref()
                    .is_some_and(|rows| rows.len() == rankings.reranker.input_size) =>
        {
            GateStatus::Pass
        }
        "disabled" => GateStatus::NotRun,
        _ => GateStatus::Fail,
    };
    let graph_passed =
        graph.fixture_edge_active && graph.neighbors.iter().any(|path| path == LINK_TARGET);
    let local_acceleration_status = if !vector.active {
        GateStatus::NotRun
    } else {
        match &vector.execution {
            Some(execution)
                if vector.route == "local-package"
                    && execution.route.as_deref() == Some("local")
                    && execution.backend != "endpoint"
                    && execution
                        .device_class
                        .as_deref()
                        .is_some_and(|device| matches!(device, "npu" | "gpu" | "cpu")) =>
            {
                GateStatus::Pass
            }
            Some(_) => GateStatus::Fail,
            None => GateStatus::Fail,
        }
    };

    let checks = vec![
        gate(
            "bm25",
            if bm25_passed {
                GateStatus::Pass
            } else {
                GateStatus::Fail
            },
            true,
            "keyword retrieval returns the lexical control in deterministic order",
            if bm25_passed {
                "the lexical control ranked first and the linked recovery note was returned"
            } else {
                "the fixed lexical controls were missing or out of order"
            },
        ),
        gate(
            "profile_admission",
            if profile_passed {
                GateStatus::Pass
            } else {
                GateStatus::Fail
            },
            true,
            "the canonical shared-vector profile is active",
            profile_evidence,
        ),
        gate(
            "vector_output",
            vector_status,
            true,
            "the embedding route returns one valid query vector and every document vector",
            if vector_passed {
                format!(
                    "received one query vector and {} document vectors at {} dimensions",
                    vector.documents.len(),
                    vector.dimensions
                )
            } else {
                vector
                    .reason
                    .clone()
                    .unwrap_or_else(|| "complete vector output was not available".into())
            },
        ),
        gate(
            "semantic_ranking",
            semantic_status,
            true,
            "meaning-based ranking puts the rollback note above the keyword trap",
            match rankings.vector.as_deref() {
                Some(paths) => format!("vector order: {}", paths.join(" -> ")),
                None => "vector ranking was not produced".into(),
            },
        ),
        gate(
            "hybrid_fusion",
            hybrid_status,
            true,
            "RRF keeps evidence contributed by both BM25 and vector ranking",
            match rankings.hybrid.as_deref() {
                Some(paths) => format!(
                    "hybrid order at k={}: {}",
                    rankings.hybrid_rrf_k,
                    paths.join(" -> ")
                ),
                None => "hybrid ranking was not produced".into(),
            },
        ),
        gate(
            "reranker",
            reranker_status,
            cfg.rerank.enabled,
            "a configured reranker returns one score for every shortlist item",
            match rankings.reranker.status {
                "active" => format!(
                    "model {} reranked {} items",
                    rankings.reranker.model.as_deref().unwrap_or("unknown"),
                    rankings.reranker.ranking.as_ref().map_or(0, Vec::len)
                ),
                "disabled" => "reranking is optional and is not configured".into(),
                _ => rankings
                    .reranker
                    .reason
                    .clone()
                    .unwrap_or_else(|| "the configured reranker did not answer".into()),
            },
        ),
        gate(
            "graph_expansion",
            if graph_passed {
                GateStatus::Pass
            } else {
                GateStatus::Fail
            },
            true,
            "post-ranking expansion follows the fixture's known wikilink",
            if graph_passed {
                format!("followed {} -> {}", LINK_SOURCE, LINK_TARGET)
            } else {
                "the known fixture wikilink was not present in expanded results".into()
            },
        ),
        gate(
            "local_acceleration",
            local_acceleration_status,
            true,
            "the embedding call used an admitted local NPU, GPU, or accelerated CPU package",
            match &vector.execution {
                Some(execution) => format!(
                    "backend {}, route {}, device {}",
                    execution.backend,
                    execution.route.as_deref().unwrap_or("not reported"),
                    execution.device_class.as_deref().unwrap_or("not reported")
                ),
                None => "no local backend and device evidence was recorded".into(),
            },
        ),
    ];
    let passed = checks
        .iter()
        .filter(|check| check.status == GateStatus::Pass)
        .count();
    let failed = checks
        .iter()
        .filter(|check| check.status == GateStatus::Fail)
        .count();
    let not_run = checks
        .iter()
        .filter(|check| check.status == GateStatus::NotRun)
        .count();
    let production_ready = checks
        .iter()
        .filter(|check| check.production_required)
        .all(|check| check.status == GateStatus::Pass);
    GateReport {
        production_ready,
        passed,
        failed,
        not_run,
        checks,
    }
}

fn one_line(error: impl std::fmt::Display) -> String {
    error
        .to_string()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
fn current_execution() -> Option<ExecutionReport> {
    None
}

#[cfg(not(test))]
fn current_execution() -> Option<ExecutionReport> {
    let selected = crate::runtime_status::load_cached().inference.selected?;
    Some(ExecutionReport {
        backend: selected.backend,
        device_class: selected.device_class,
        route: selected.route.map(|route| match route {
            crate::runtime_status::InferenceRoute::Local => "local".into(),
            crate::runtime_status::InferenceRoute::Remote => "remote".into(),
        }),
    })
}

fn route(cfg: &Config) -> &'static str {
    if !cfg.embeddings.enabled {
        "disabled"
    } else if cfg.embeddings.endpoint.is_empty() {
        "local-package"
    } else {
        "configured-endpoint"
    }
}

fn fixture_tree() -> anyhow::Result<(tempfile::TempDir, tempfile::TempDir, rusqlite::Connection)> {
    let brain = tempfile::tempdir().context("create temporary fixture tree")?;
    for document in DOCUMENTS {
        let path = brain.path().join(document.path);
        std::fs::create_dir_all(path.parent().expect("fixture path has a parent"))?;
        std::fs::write(path, document.text)?;
    }
    let state = tempfile::tempdir().context("create temporary fixture index")?;
    let mut conn = index::open(state.path())?;
    index::scan(
        &mut conn,
        brain.path(),
        None,
        &crate::config::RingRules::default(),
    )?;
    Ok((brain, state, conn))
}

fn ranked_paths(hits: &[index::Hit]) -> Vec<String> {
    hits.iter().map(|hit| hit.path.clone()).collect()
}

fn path_for_hash(conn: &rusqlite::Connection, hash: &str) -> anyhow::Result<String> {
    conn.query_row(
        "SELECT d.path FROM blocks b JOIN docs d ON d.id = b.doc_id WHERE b.embedding_hash = ?1 ORDER BY d.path LIMIT 1",
        [hash],
        |row| row.get(0),
    )
    .context("fixture vector did not match an indexed statement")
}

fn prepared_for_ranking(mut vector: Vec<f32>, precision: Precision) -> Vec<f32> {
    if precision != Precision::I8 {
        index::l2_normalize(&mut vector);
    }
    vector
}

fn stored_bytes(vector: &[f32], precision: Precision) -> Vec<u8> {
    if precision == Precision::I8 {
        index::vec_to_blob(vector, precision)
    } else {
        let mut normalized = vector.to_vec();
        index::l2_normalize(&mut normalized);
        index::vec_to_blob(&normalized, precision)
    }
}

fn normalized_stored_view(vector: &[f32], precision: Precision) -> Vec<f32> {
    let bytes = stored_bytes(vector, precision);
    let mut decoded = index::blob_to_vec(&bytes, precision);
    index::l2_normalize(&mut decoded);
    decoded
}

fn components(bytes: &[u8], precision: Precision, limit: Option<usize>) -> Components {
    match precision {
        Precision::I8 => Components::SignedInt8(
            bytes
                .iter()
                .take(limit.unwrap_or(usize::MAX))
                .map(|byte| *byte as i8)
                .collect(),
        ),
        Precision::F16 | Precision::F32 => Components::Float(
            index::blob_to_vec(bytes, precision)
                .into_iter()
                .take(limit.unwrap_or(usize::MAX))
                .collect(),
        ),
    }
}

fn vector_output(vector: &[f32], precision: Precision, show_vectors: bool) -> VectorOutput {
    let bytes = stored_bytes(vector, precision);
    let decoded = normalized_stored_view(vector, precision);
    let norm = decoded
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    VectorOutput {
        sha256: hashing::hex_lower(sha2::Sha256::digest(&bytes)),
        norm,
        preview: components(&bytes, precision, Some(PREVIEW_COMPONENTS)),
        components: show_vectors.then(|| components(&bytes, precision, None)),
    }
}

fn unavailable_vector_report(cfg: &Config, reason: String) -> VectorReport {
    VectorReport {
        purpose: "turns the query and notes into vectors for meaning-based matching",
        configured: cfg.embeddings.enabled,
        active: false,
        route: route(cfg),
        model: cfg.embeddings.model.clone(),
        dimensions: cfg.embeddings.dimensions,
        encoding: cfg.embeddings.spec().vector_encoding(),
        profile_status: if cfg.embeddings.model == embedding_profile::MODEL {
            embedding_profile::PROFILE_STATUS
        } else {
            "custom"
        },
        reason: Some(reason),
        execution: None,
        query: None,
        documents: Vec::new(),
    }
}

fn rerank_fixture(
    cfg: &Config,
    conn: &rusqlite::Connection,
    vector_active: bool,
    spec: &crate::config::VectorSpec,
    query_vector: Option<&[f32]>,
) -> anyhow::Result<(RerankerReport, Vec<index::Hit>)> {
    let input = if vector_active { "hybrid" } else { "bm25" };
    let base = if let Some(query_vector) = query_vector {
        index::hybrid_recall(
            conn,
            spec,
            QUERY,
            query_vector,
            LIMIT,
            cfg.recall.rrf_k,
            &[],
        )?
    } else {
        index::recall(conn, QUERY, LIMIT)?
    };
    if !cfg.rerank.enabled {
        return Ok((
            RerankerReport {
                purpose: "reorders an existing shortlist; it does not find new notes",
                status: "disabled",
                input,
                input_size: base.len(),
                model: None,
                reason: None,
                execution: None,
                ranking: None,
            },
            base,
        ));
    }
    let client = match rerank::RerankClient::new(&cfg.rerank) {
        Ok(client) => client,
        Err(error) => {
            return Ok((
                RerankerReport {
                    purpose: "reorders an existing shortlist; it does not find new notes",
                    status: "unavailable",
                    input,
                    input_size: base.len(),
                    model: (!cfg.rerank.model.is_empty()).then(|| cfg.rerank.model.clone()),
                    reason: Some(one_line(format!("{error:#}"))),
                    execution: None,
                    ranking: Some(ranked_paths(&base)),
                },
                base,
            ));
        }
    };
    let model = client.model().to_string();
    let input_size = base.len();
    let result = rerank::apply(&client, QUERY, base, |hit: &index::Hit| hit.snippet.clone());
    let ranking = ranked_paths(&result.hits);
    let active = result.note.is_none();
    let report = RerankerReport {
        purpose: "reorders an existing shortlist; it does not find new notes",
        status: if active { "active" } else { "unavailable" },
        input,
        input_size,
        model: Some(model),
        reason: result.note,
        execution: active.then(current_execution).flatten(),
        ranking: Some(ranking),
    };
    Ok((report, result.hits))
}

pub(crate) fn gather_with_config(
    cfg: &Config,
    show_vectors: bool,
) -> anyhow::Result<RetrievalFixtureReport> {
    let (_brain, _state, conn) = fixture_tree()?;
    let bm25_hits = index::recall(&conn, QUERY, LIMIT)?;
    let bm25 = ranked_paths(&bm25_hits);
    let spec = cfg.embeddings.spec();

    let (client, inactive_reason) = match embed::EmbedClient::new(&cfg.embeddings) {
        Ok(client) => (Some(client), None),
        Err(error) => (None, Some(one_line(format!("{error:#}")))),
    };

    let mut prepared_query = None;
    let mut vector_ranking = None;
    let mut hybrid_ranking = None;
    let vector = if let Some(client) = client {
        let pending = index::hashes_without_vectors(&conn, &spec, usize::MAX)?;
        let texts: Vec<&str> = pending.iter().map(|(_, text)| text.as_str()).collect();
        match client
            .embed_documents_batch(&texts)
            .and_then(|documents| client.embed_query(QUERY).map(|query| (documents, query)))
        {
            Ok((document_vectors, query_vector)) => {
                let query_vector = prepared_for_ranking(query_vector, spec.precision);
                let query_view = normalized_stored_view(&query_vector, spec.precision);
                let mut documents = Vec::with_capacity(document_vectors.len());
                for ((hash, _), document_vector) in pending.iter().zip(document_vectors) {
                    index::insert_vector(&conn, hash, &spec, &document_vector)?;
                    let document_view = normalized_stored_view(&document_vector, spec.precision);
                    documents.push(DocumentVector {
                        path: path_for_hash(&conn, hash)?,
                        cosine_to_query: index::dot(&query_view, &document_view),
                        output: vector_output(&document_vector, spec.precision, show_vectors),
                    });
                }
                documents.sort_by(|left, right| {
                    right
                        .cosine_to_query
                        .total_cmp(&left.cosine_to_query)
                        .then(left.path.cmp(&right.path))
                });
                let semantic = index::semantic_recall(&conn, &spec, &query_vector, LIMIT, &[])?;
                let hybrid = index::hybrid_recall(
                    &conn,
                    &spec,
                    QUERY,
                    &query_vector,
                    LIMIT,
                    cfg.recall.rrf_k,
                    &[],
                )?;
                vector_ranking = Some(ranked_paths(&semantic));
                hybrid_ranking = Some(ranked_paths(&hybrid));
                let output = vector_output(&query_vector, spec.precision, show_vectors);
                prepared_query = Some(query_vector);
                VectorReport {
                    purpose: "turns the query and notes into vectors for meaning-based matching",
                    configured: true,
                    active: true,
                    route: route(cfg),
                    model: client.model().to_string(),
                    dimensions: client.dimensions(),
                    encoding: spec.vector_encoding(),
                    profile_status: if cfg.embeddings.model == embedding_profile::MODEL {
                        embedding_profile::PROFILE_STATUS
                    } else {
                        "custom"
                    },
                    reason: None,
                    execution: current_execution(),
                    query: Some(output),
                    documents,
                }
            }
            Err(error) => unavailable_vector_report(cfg, one_line(format!("{error:#}"))),
        }
    } else {
        unavailable_vector_report(
            cfg,
            inactive_reason.unwrap_or_else(|| "embedding route unavailable".to_string()),
        )
    };

    let (reranker, final_hits) =
        rerank_fixture(cfg, &conn, vector.active, &spec, prepared_query.as_deref())?;
    let top_paths: Vec<String> = final_hits
        .iter()
        .take(3)
        .map(|hit| hit.path.clone())
        .collect();
    let graph_neighbors = index::linked_docs(&conn, &top_paths, 8)?
        .into_iter()
        .map(|(path, _)| path)
        .collect();
    let fixture_edge_from = LINK_SOURCE;
    let fixture_edge_to = LINK_TARGET;
    let fixture_edge_active = index::linked_docs(&conn, &[fixture_edge_from.to_string()], 8)?
        .iter()
        .any(|(path, _)| path == fixture_edge_to);

    let rankings = RankingReport {
        bm25_role: "keyword matching; no model is used",
        bm25,
        vector_role: "meaning matching using the embedding model",
        vector: vector_ranking,
        hybrid_role: "BM25 and vector ranks combined with reciprocal rank fusion",
        hybrid_rrf_k: cfg.recall.rrf_k,
        hybrid: hybrid_ranking,
        reranker,
    };
    let graph = GraphReport {
        role: "post-ranking one-hop expansion; not a ranking input",
        from: "top three final hits",
        neighbors: graph_neighbors,
        fixture_edge_from,
        fixture_edge_to,
        fixture_edge_active,
    };
    let gates = evaluate_gates(cfg, &vector, &rankings, &graph);

    Ok(RetrievalFixtureReport {
        schema_version: 2,
        temporary_data: true,
        query: QUERY,
        gates,
        vector,
        rankings,
        graph,
    })
}

fn push_list(lines: &mut Vec<String>, label: &str, paths: Option<&[String]>) {
    lines.push(format!("{label}:"));
    match paths {
        Some([]) => lines.push("  no matches".into()),
        Some(paths) => {
            for (rank, path) in paths.iter().enumerate() {
                lines.push(format!("  {}. {path}", rank + 1));
            }
        }
        None => lines.push("  unavailable".into()),
    }
}

fn component_text(components: &Components) -> String {
    match components {
        Components::SignedInt8(values) => format!("{values:?}"),
        Components::Float(values) => format!("{values:?}"),
    }
}

pub fn vector_active(report: &RetrievalFixtureReport) -> bool {
    report.vector.active
}

pub fn reranker_status(report: &RetrievalFixtureReport) -> &str {
    report.rankings.reranker.status
}

fn required_gate_ids(
    report: &RetrievalFixtureReport,
    requirements: &[Requirement],
) -> std::collections::BTreeSet<&'static str> {
    let mut ids = std::collections::BTreeSet::new();
    for requirement in requirements {
        match requirement {
            Requirement::Bm25 => {
                ids.insert("bm25");
            }
            Requirement::Profile => {
                ids.insert("profile_admission");
            }
            Requirement::Vector => {
                ids.extend(["vector_output", "semantic_ranking"]);
            }
            Requirement::Hybrid => {
                ids.extend(["vector_output", "semantic_ranking", "hybrid_fusion"]);
            }
            Requirement::Reranker => {
                ids.insert("reranker");
            }
            Requirement::Graph => {
                ids.insert("graph_expansion");
            }
            Requirement::LocalAcceleration => {
                ids.extend(["vector_output", "local_acceleration"]);
            }
            Requirement::Production => {
                ids.extend(
                    report
                        .gates
                        .checks
                        .iter()
                        .filter(|check| check.production_required)
                        .map(|check| check.id),
                );
            }
        }
    }
    ids
}

pub fn enforce_requirements(
    report: &RetrievalFixtureReport,
    requirements: &[Requirement],
) -> anyhow::Result<()> {
    if requirements.is_empty() {
        return Ok(());
    }
    let required = required_gate_ids(report, requirements);
    let blockers: Vec<String> = report
        .gates
        .checks
        .iter()
        .filter(|check| required.contains(check.id) && check.status != GateStatus::Pass)
        .map(|check| {
            format!(
                "{} ({})",
                check.id,
                check.status.label().to_ascii_lowercase()
            )
        })
        .collect();
    anyhow::ensure!(
        blockers.is_empty(),
        "required {} gate did not pass: {}",
        requirements
            .iter()
            .map(|requirement| requirement.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        blockers.join(", ")
    );
    Ok(())
}

pub fn display_lines(report: &RetrievalFixtureReport) -> Vec<String> {
    let show_vectors = report
        .vector
        .query
        .as_ref()
        .is_some_and(|query| query.components.is_some());
    let mut lines = vec![
        "temporary retrieval test (your brain and vector store are untouched)".into(),
        format!("query: {}", report.query),
        String::new(),
        format!(
            "production retrieval gate: {}",
            if report.gates.production_ready {
                "PASS"
            } else {
                "BLOCKED"
            }
        ),
        format!(
            "  {} passed · {} failed · {} not run",
            report.gates.passed, report.gates.failed, report.gates.not_run
        ),
    ];
    for check in &report.gates.checks {
        lines.push(format!(
            "  {:<7} {:<20} {}",
            check.status.label(),
            check.id,
            check.evidence
        ));
    }
    lines.extend([
        String::new(),
        format!(
            "embedding model: {}",
            if report.vector.active {
                "ACTIVE"
            } else {
                "INACTIVE"
            }
        ),
        format!("  job: {}", report.vector.purpose),
        format!(
            "  configured: {}",
            if report.vector.configured {
                "yes"
            } else {
                "no"
            }
        ),
        format!("  route: {}", report.vector.route),
        format!("  model: {}", report.vector.model),
        format!("  profile: {}", report.vector.profile_status),
        format!(
            "  output: {} dimensions, {}",
            report.vector.dimensions, report.vector.encoding
        ),
    ]);
    if let Some(reason) = &report.vector.reason {
        lines.push(format!("  reason: {reason}"));
    }
    if report.vector.active {
        match &report.vector.execution {
            Some(execution) => lines.push(format!(
                "  execution: backend {} · route {} · device {}",
                execution.backend,
                execution.route.as_deref().unwrap_or("not reported"),
                execution.device_class.as_deref().unwrap_or("not reported"),
            )),
            None => lines.push(
                "  execution: model answered, but backend/device telemetry was not recorded".into(),
            ),
        }
    }
    if let Some(query) = &report.vector.query {
        lines.push(format!("  query vector sha256: {}", query.sha256));
        lines.push(format!("  query vector norm: {:.6}", query.norm));
        lines.push(format!(
            "  query vector {}: {}",
            if show_vectors {
                "components"
            } else {
                "first 12 components"
            },
            component_text(if show_vectors {
                query.components.as_ref().unwrap_or(&query.preview)
            } else {
                &query.preview
            })
        ));
        lines.push("  document cosine similarity:".into());
        for document in &report.vector.documents {
            lines.push(format!(
                "    {:+.6}  {}  sha256 {}",
                document.cosine_to_query, document.path, document.output.sha256
            ));
            if show_vectors {
                let values = document
                    .output
                    .components
                    .as_ref()
                    .unwrap_or(&document.output.preview);
                lines.push(format!("      {}", component_text(values)));
            }
        }
    }

    lines.push(String::new());
    push_list(
        &mut lines,
        &format!("BM25 ({})", report.rankings.bm25_role),
        Some(&report.rankings.bm25),
    );
    push_list(
        &mut lines,
        &format!("vector ({})", report.rankings.vector_role),
        report.rankings.vector.as_deref(),
    );
    push_list(
        &mut lines,
        &format!(
            "hybrid ({}; k {})",
            report.rankings.hybrid_role, report.rankings.hybrid_rrf_k
        ),
        report.rankings.hybrid.as_deref(),
    );
    lines.push(format!(
        "reranker model: {} (input: {})",
        report.rankings.reranker.status.to_ascii_uppercase(),
        report.rankings.reranker.input
    ));
    lines.push(format!("  job: {}", report.rankings.reranker.purpose));
    if let Some(model) = &report.rankings.reranker.model {
        lines.push(format!("  model: {model}"));
    }
    if let Some(reason) = &report.rankings.reranker.reason {
        lines.push(format!("  reason: {reason}"));
    }
    if report.rankings.reranker.status == "active" {
        match &report.rankings.reranker.execution {
            Some(execution) => lines.push(format!(
                "  execution: backend {} · route {} · device {}",
                execution.backend,
                execution.route.as_deref().unwrap_or("not reported"),
                execution.device_class.as_deref().unwrap_or("not reported"),
            )),
            None => lines.push(
                "  execution: model answered, but backend/device telemetry was not recorded".into(),
            ),
        }
    }
    if let Some(ranking) = &report.rankings.reranker.ranking {
        push_list(
            &mut lines,
            if report.rankings.reranker.status == "active" {
                "reranked"
            } else {
                "final ranking (reranker unavailable, order kept)"
            },
            Some(ranking),
        );
    }
    lines.push(format!(
        "graph expansion: {}",
        if report.graph.fixture_edge_active {
            "ACTIVE"
        } else {
            "INACTIVE"
        }
    ));
    lines.push(format!("  job: {}", report.graph.role));
    lines.push(format!(
        "  fixture link: {} -> {}",
        report.graph.fixture_edge_from, report.graph.fixture_edge_to
    ));
    lines.push(format!("  starts from: {}", report.graph.from));
    push_list(&mut lines, "graph neighbors", Some(&report.graph.neighbors));
    lines
}

pub fn render_text(report: &RetrievalFixtureReport) -> String {
    display_lines(report).join("\n")
}

pub fn gather(show_vectors: bool) -> anyhow::Result<RetrievalFixtureReport> {
    let cfg = Config::load()?;
    gather_with_config(&cfg, show_vectors)
}

pub fn run(json: bool, show_vectors: bool, requirements: &[Requirement]) -> anyhow::Result<()> {
    let report = gather(show_vectors)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{}", render_text(&report));
    }
    enforce_requirements(&report, requirements)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EmbeddingsConfig, RerankConfig};
    use crate::testhttp::{http_response, spawn_server};

    fn fixture_vector(text: &str) -> Vec<f32> {
        if text == QUERY || text.contains("schema migration") {
            vec![1.0, 0.0, 0.0]
        } else if text.contains("previous release") {
            vec![0.9, 0.1, 0.0]
        } else if text.contains("deployment counts") {
            vec![0.0, 1.0, 0.0]
        } else if text.contains("service health") {
            vec![-1.0, 0.0, 0.0]
        } else {
            vec![0.0, 0.0, 1.0]
        }
    }

    fn embedding_response(body: &str) -> String {
        let request: serde_json::Value = serde_json::from_str(body).unwrap();
        let model = request["model"].as_str().unwrap();
        let rows: Vec<serde_json::Value> = request["input"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, text)| {
                serde_json::json!({
                    "index": index,
                    "embedding": fixture_vector(text.as_str().unwrap()),
                })
            })
            .collect();
        http_response(
            200,
            &serde_json::json!({"model": model, "data": rows}).to_string(),
        )
    }

    fn active_config() -> Config {
        let (endpoint, _, _) = spawn_server(|_, body| embedding_response(body));
        let (rerank_endpoint, _, _) = spawn_server(|_, body| {
            let request: serde_json::Value = serde_json::from_str(body).unwrap();
            let count = request["documents"].as_array().unwrap().len();
            let rows: Vec<serde_json::Value> = (0..count)
                .map(|index| serde_json::json!({"index": index, "relevance_score": index as f32}))
                .collect();
            http_response(200, &serde_json::json!({"results": rows}).to_string())
        });
        Config {
            embeddings: EmbeddingsConfig {
                enabled: true,
                endpoint,
                model: "fixture-model".to_string(),
                endpoint_model: None,
                dimensions: 3,
                query_prefix: String::new(),
                document_prefix: String::new(),
                precision: Precision::I8,
                ..EmbeddingsConfig::default()
            },
            rerank: RerankConfig {
                enabled: true,
                endpoint: rerank_endpoint,
                model: "fixture-reranker".to_string(),
                candidates: LIMIT,
                ..RerankConfig::default()
            },
            ..Config::default()
        }
    }

    #[test]
    fn inactive_vectors_are_visible_while_bm25_and_graph_still_run() {
        let report = gather_with_config(&Config::default(), false).unwrap();
        assert_eq!(report.schema_version, 2);
        assert!(!report.vector.active);
        assert!(
            report
                .vector
                .reason
                .as_deref()
                .unwrap()
                .contains("embeddings disabled")
        );
        assert!(!report.rankings.bm25.is_empty());
        assert!(report.rankings.vector.is_none());
        assert!(report.rankings.hybrid.is_none());
        assert_eq!(report.rankings.reranker.status, "disabled");
        assert!(
            report
                .graph
                .neighbors
                .contains(&"knowledge/recovery-checklist.md".to_string())
        );
        assert!(report.graph.fixture_edge_active);
        assert!(!report.gates.production_ready);
        assert_eq!(
            report
                .gates
                .checks
                .iter()
                .find(|check| check.id == "bm25")
                .unwrap()
                .status,
            GateStatus::Pass
        );
        assert_eq!(
            report
                .gates
                .checks
                .iter()
                .find(|check| check.id == "vector_output")
                .unwrap()
                .status,
            GateStatus::NotRun
        );
        assert!(enforce_requirements(&report, &[Requirement::Bm25]).is_ok());
        assert!(enforce_requirements(&report, &[Requirement::Vector]).is_err());
        assert!(enforce_requirements(&report, &[Requirement::Production]).is_err());
    }

    #[test]
    fn active_vectors_show_outputs_and_every_ranking_stage() {
        let report = gather_with_config(&active_config(), true).unwrap();
        assert!(report.vector.active, "{:?}", report.vector.reason);
        assert_eq!(report.vector.dimensions, 3);
        assert_eq!(report.vector.documents.len(), DOCUMENTS.len());
        assert!(report.vector.query.as_ref().unwrap().components.is_some());
        assert_eq!(
            report.rankings.vector.as_ref().unwrap()[0],
            "knowledge/schema-rollback.md"
        );
        assert!(report.rankings.hybrid.is_some());
        assert_eq!(report.rankings.hybrid_rrf_k, 2.0);
        assert_eq!(report.rankings.reranker.status, "active");
        assert_eq!(
            report.rankings.reranker.model.as_deref(),
            Some("fixture-reranker")
        );
        assert!(report.rankings.reranker.ranking.is_some());
        assert!(report.graph.fixture_edge_active);
        for requirement in [
            Requirement::Bm25,
            Requirement::Vector,
            Requirement::Hybrid,
            Requirement::Reranker,
            Requirement::Graph,
        ] {
            enforce_requirements(&report, &[requirement]).unwrap();
        }
        assert!(enforce_requirements(&report, &[Requirement::Profile]).is_err());
        assert!(enforce_requirements(&report, &[Requirement::LocalAcceleration]).is_err());
        assert!(enforce_requirements(&report, &[Requirement::Production]).is_err());
    }

    #[test]
    fn reranker_gate_passes_when_a_shorter_shortlist_is_fully_reranked() {
        let cfg = active_config();
        let mut report = gather_with_config(&cfg, false).unwrap();
        // A shortlist shorter than LIMIT is legitimate: BM25 can return
        // fewer hits. The reranker still scored every submitted item, so
        // the gate must pass.
        report.rankings.reranker.input_size = 2;
        if let Some(rows) = report.rankings.reranker.ranking.as_mut() {
            rows.truncate(2);
        }
        report.gates = evaluate_gates(&cfg, &report.vector, &report.rankings, &report.graph);
        assert_eq!(
            report
                .gates
                .checks
                .iter()
                .find(|check| check.id == "reranker")
                .unwrap()
                .status,
            GateStatus::Pass
        );
        enforce_requirements(&report, &[Requirement::Reranker]).unwrap();

        // A dropped row is still a failure: two scored of three submitted.
        report.rankings.reranker.input_size = 3;
        report.gates = evaluate_gates(&cfg, &report.vector, &report.rankings, &report.graph);
        assert_eq!(
            report
                .gates
                .checks
                .iter()
                .find(|check| check.id == "reranker")
                .unwrap()
                .status,
            GateStatus::Fail
        );
        assert!(enforce_requirements(&report, &[Requirement::Reranker]).is_err());
    }

    #[test]
    fn compact_report_omits_full_vectors_but_keeps_the_preview_and_hash() {
        let report = gather_with_config(&active_config(), false).unwrap();
        let query = report.vector.query.unwrap();
        assert!(query.components.is_none());
        match query.preview {
            Components::SignedInt8(values) => assert_eq!(values.len(), 3),
            Components::Float(_) => panic!("fixture uses signed INT8"),
        }
        assert_eq!(query.sha256.len(), 64);
    }

    #[test]
    fn local_acceleration_needs_the_package_route_not_loopback_telemetry() {
        let cfg = active_config();
        let mut report = gather_with_config(&cfg, false).unwrap();
        report.vector.execution = Some(ExecutionReport {
            backend: "openvino".into(),
            device_class: Some("npu".into()),
            route: Some("local".into()),
        });
        report.gates = evaluate_gates(&cfg, &report.vector, &report.rankings, &report.graph);
        assert!(enforce_requirements(&report, &[Requirement::LocalAcceleration]).is_err());

        report.vector.route = "local-package";
        report.gates = evaluate_gates(&cfg, &report.vector, &report.rankings, &report.graph);
        enforce_requirements(&report, &[Requirement::LocalAcceleration]).unwrap();
    }

    #[test]
    fn valid_vectors_do_not_pass_when_the_semantic_control_loses() {
        let cfg = active_config();
        let mut report = gather_with_config(&cfg, false).unwrap();
        let vector = report.rankings.vector.as_mut().unwrap();
        let semantic = position(vector, SEMANTIC_TARGET).unwrap();
        let decoy = position(vector, LEXICAL_DECOY).unwrap();
        vector.swap(semantic, decoy);
        report.gates = evaluate_gates(&cfg, &report.vector, &report.rankings, &report.graph);

        assert_eq!(
            report
                .gates
                .checks
                .iter()
                .find(|check| check.id == "vector_output")
                .unwrap()
                .status,
            GateStatus::Pass
        );
        assert_eq!(
            report
                .gates
                .checks
                .iter()
                .find(|check| check.id == "semantic_ranking")
                .unwrap()
                .status,
            GateStatus::Fail
        );
        assert!(enforce_requirements(&report, &[Requirement::Vector]).is_err());
    }

    #[test]
    fn legacy_float_output_reports_the_same_normalized_bytes_as_the_index() {
        let output = vector_output(&[3.0, 4.0], Precision::F32, false);
        let expected = stored_bytes(&[3.0, 4.0], Precision::F32);
        assert_eq!(
            output.sha256,
            hashing::hex_lower(sha2::Sha256::digest(expected))
        );
        match output.preview {
            Components::Float(values) => assert_eq!(values, vec![0.6, 0.8]),
            Components::SignedInt8(_) => panic!("fixture requested f32"),
        }
    }
}
