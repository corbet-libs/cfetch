//! Cross-encoder reranking of a retrieved shortlist.
//!
//! Retrieval scores a query against a document the scorer never sees beside
//! it: BM25 counts terms, a bi-encoder compares two vectors that were built
//! independently. A cross-encoder reads query and document TOGETHER and is
//! markedly better at judging relevance — at the price of one forward pass
//! per candidate, so it can only ever run over a shortlist. Recall proposes,
//! rerank reorders.
//!
//! Reranking NEVER decides whether an answer comes back. Every failure —
//! endpoint down, unparseable response, a score list that does not line up
//! with what was sent — returns the retrieval order with the reason attached,
//! because a worse ordering is a far smaller harm than no answer, and a
//! silent fallback is the harm this codebase refuses everywhere else.

use crate::config::RerankConfig;
use crate::embed::{check_endpoint, resolve_auth, snippet};
use anyhow::Context;

pub struct RerankClient {
    agent: ureq::Agent,
    /// Full `…/rerank` URL, endpoint trailing slashes normalized away.
    url: String,
    model: String,
    auth: Option<String>,
    timeout: std::time::Duration,
    candidates: usize,
    /// Local cross-encoder reranker (feature-gated). When the endpoint is
    /// empty and the `embedded-embeddings` feature is enabled, reranking
    /// runs in-process via fastembed instead of over HTTP.
    #[cfg(feature = "embedded-embeddings")]
    embedded: Option<std::sync::Mutex<crate::embedded_embed::EmbeddedReranker>>,
}

impl std::fmt::Debug for RerankClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RerankClient")
            .field("url", &self.url)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl RerankClient {
    /// The one gate for every rerank path, so a misconfiguration is a single
    /// clear line rather than a surprise mid-query.
    ///
    /// With the `embedded-embeddings` feature an EMPTY endpoint selects the
    /// in-process cross-encoder (Jina reranker v2 multilingual via fastembed);
    /// a set endpoint keeps the HTTP path. Without the feature the endpoint
    /// remains required.
    pub fn new(cfg: &RerankConfig) -> anyhow::Result<RerankClient> {
        anyhow::ensure!(cfg.enabled, "rerank disabled (set rerank.enabled=true in config)");
        anyhow::ensure!(cfg.candidates > 0, "rerank.candidates must be at least 1");

        #[cfg(feature = "embedded-embeddings")]
        if cfg.endpoint.is_empty() {
            let reranker = crate::embedded_embed::EmbeddedReranker::load()
                .context("loading embedded reranker (rerank.endpoint is empty)")?;
            return Ok(RerankClient {
                embedded: Some(std::sync::Mutex::new(reranker)),
                agent: ureq::Agent::config_builder()
                    .max_redirects(0)
                    .timeout_global(Some(std::time::Duration::from_secs(cfg.timeout_secs.max(1))))
                    .http_status_as_error(false)
                    .build()
                    .new_agent(),
                url: String::new(),
                model: crate::embedded_embed::DEFAULT_RERANKER_MODEL_NAME.to_string(),
                auth: None,
                timeout: std::time::Duration::from_secs(cfg.timeout_secs.max(1)),
                candidates: cfg.candidates,
            });
        }

        anyhow::ensure!(
            !cfg.endpoint.is_empty() && !cfg.model.is_empty(),
            "rerank not configured (rerank.endpoint and rerank.model required)"
        );
        check_endpoint(&cfg.endpoint, &cfg.allow_hosts)?;
        let auth = resolve_auth(&cfg.api_key_env, "rerank")?;
        let timeout = std::time::Duration::from_secs(cfg.timeout_secs.max(1));
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .max_redirects(0)
            .timeout_global(Some(timeout))
            .http_status_as_error(false)
            .build()
            .new_agent();
        Ok(RerankClient {
            agent,
            url: format!("{}/rerank", cfg.endpoint.trim_end_matches('/')),
            model: cfg.model.clone(),
            auth,
            timeout,
            candidates: cfg.candidates,
            #[cfg(feature = "embedded-embeddings")]
            embedded: None,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn candidates(&self) -> usize {
        self.candidates
    }

    /// Scores every document against the query. Returns one `(index, score)`
    /// per input document, in INPUT order — ordering is the caller's job, so
    /// that a caller holding a stable sort keeps its own tie-breaking.
    ///
    /// Scores are whatever the model emits (cross-encoder logits are commonly
    /// unbounded and negative); only their ORDER is meaningful, and nothing
    /// here interprets them as probabilities.
    pub fn rank(&self, query: &str, documents: &[&str]) -> anyhow::Result<Vec<f32>> {
        #[cfg(feature = "embedded-embeddings")]
        if let Some(reranker) = &self.embedded {
            let result = reranker
                .lock()
                .expect("reranker mutex poisoned")
                .rank(query, documents);
            crate::runtime_status::record_inference_result(
                crate::runtime_status::InferenceMode::Local,
                crate::runtime_status::InferenceRoute::Local,
                "rerank-embedded",
                None,
                &result,
            );
            return result;
        }
        let result = self.rank_request(query, documents);
        crate::runtime_status::record_inference_result(
            crate::runtime_status::InferenceMode::Endpoint,
            crate::runtime_status::endpoint_route(&self.url),
            "rerank-endpoint",
            None,
            &result,
        );
        result
    }

    fn rank_request(&self, query: &str, documents: &[&str]) -> anyhow::Result<Vec<f32>> {
        #[derive(serde::Deserialize)]
        struct Response {
            results: Vec<Row>,
        }
        #[derive(serde::Deserialize)]
        struct Row {
            index: Option<usize>,
            relevance_score: f32,
        }
        anyhow::ensure!(!documents.is_empty(), "rerank called with no documents");
        let body =
            serde_json::json!({ "model": self.model, "query": query, "documents": documents })
                .to_string();
        let mut req = self
            .agent
            .post(&self.url)
            .config()
            .timeout_global(Some(self.timeout))
            .build()
            .header("content-type", "application/json");
        if let Some(auth) = &self.auth {
            req = req.header("authorization", auth);
        }
        let mut resp = req.send(body.as_bytes()).with_context(|| format!("POST {}", self.url))?;
        let status = resp.status();
        // Same bound as the maintenance client: an endpoint that streams an
        // unbounded body must not be able to OOM the process. A rerank
        // response is one float per document — 2 MiB is orders of magnitude
        // beyond any legitimate one.
        const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
        let text = resp
            .body_mut()
            .with_config()
            .limit(MAX_RESPONSE_BYTES as u64)
            .read_to_string()
            .with_context(|| format!("read response from {}", self.url))?;
        anyhow::ensure!(
            text.len() <= MAX_RESPONSE_BYTES,
            "rerank endpoint response exceeds {MAX_RESPONSE_BYTES} bytes"
        );
        anyhow::ensure!(status.is_success(), "rerank endpoint returned {status}: {}", snippet(&text));
        let parsed: Response = serde_json::from_str(&text)
            .with_context(|| format!("unparseable rerank response: {}", snippet(&text)))?;
        anyhow::ensure!(
            parsed.results.len() == documents.len(),
            "rerank endpoint returned {} score(s) for {} document(s)",
            parsed.results.len(),
            documents.len()
        );
        // Place by the response's `index` where present: array order is not
        // promised, and a misplaced score attaches the wrong relevance to the
        // wrong statement — the reranking equivalent of a mis-aligned vector.
        // A duplicate or out-of-range index means the response cannot be
        // trusted at all, so it is an error rather than a partial application.
        // An OMITTED index is the same defect (the embeddings client refuses
        // the identical shape): falling back to array position silently
        // reorders best-first responses into wrong scores.
        let mut scores: Vec<Option<f32>> = vec![None; documents.len()];
        for row in parsed.results.into_iter() {
            let i = row.index.context("rerank endpoint omitted an index; array order is not promised")?;
            anyhow::ensure!(
                i < scores.len(),
                "rerank endpoint returned index {i} for {} document(s)",
                scores.len()
            );
            anyhow::ensure!(scores[i].is_none(), "rerank endpoint returned index {i} twice");
            scores[i] = Some(row.relevance_score);
        }
        scores
            .into_iter()
            .enumerate()
            .map(|(i, s)| s.ok_or_else(|| anyhow::anyhow!("rerank endpoint scored no document {i}")))
            .collect()
    }
}

/// Flattens a multi-line error into one operator-readable line: these notes
/// are printed beside hits and carried in `--json`, where an embedded newline
/// would break the shape of both.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A rerank attempt's result: the list, and what went wrong if anything did.
pub struct Reranked<T> {
    pub hits: Vec<T>,
    /// Present exactly when the order is NOT the cross-encoder's — the reason
    /// belongs in front of the operator, never swallowed.
    pub note: Option<String>,
}

/// Reorders the head of `hits` by cross-encoder relevance.
///
/// Only the first `client.candidates()` entries are sent; the tail keeps its
/// retrieval order and follows the reranked head. That bound is the whole
/// reason reranking is affordable, so it is enforced here rather than trusted
/// to the caller.
pub fn apply<T>(
    client: &RerankClient,
    query: &str,
    hits: Vec<T>,
    text_of: impl Fn(&T) -> String,
) -> Reranked<T> {
    if hits.len() < 2 {
        // Nothing to reorder, and no reason to spend a forward pass saying so.
        return Reranked { hits, note: None };
    }
    let head = client.candidates().min(hits.len());
    let texts: Vec<String> = hits[..head].iter().map(&text_of).collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    match client.rank(query, &refs) {
        Ok(scores) => {
            let mut tail = hits;
            let head_hits: Vec<T> = tail.drain(..head).collect();
            // Sort by score descending, ties keeping retrieval order: a
            // cross-encoder that cannot separate two candidates must not be
            // allowed to shuffle them either.
            let mut ordered: Vec<(usize, T)> = head_hits.into_iter().enumerate().collect();
            ordered.sort_by(|(ia, _), (ib, _)| {
                scores[*ib].total_cmp(&scores[*ia]).then(ia.cmp(ib))
            });
            let mut out: Vec<T> = ordered.into_iter().map(|(_, h)| h).collect();
            out.append(&mut tail);
            Reranked { hits: out, note: None }
        }
        Err(e) => Reranked {
            hits,
            note: Some(format!(
                "rerank unavailable ({}) — answering in retrieval order",
                one_line(&format!("{e:#}"))
            )),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testhttp::{http_response, spawn_server};

    fn cfg_for(url: &str, candidates: usize) -> RerankConfig {
        RerankConfig {
            enabled: true,
            endpoint: url.to_string(),
            model: "test-reranker".into(),
            candidates,
            ..RerankConfig::default()
        }
    }

    fn client_for(url: &str, candidates: usize) -> RerankClient {
        RerankClient::new(&cfg_for(url, candidates)).unwrap()
    }

    /// Scores by position, emitted in REVERSED array order, so a client that
    /// trusts array order instead of `index` fails this immediately.
    fn canned(body: &str, scores: &[f32]) -> String {
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        let n = v["documents"].as_array().unwrap().len();
        let rows: Vec<String> = (0..n)
            .rev()
            .map(|i| format!(r#"{{"index":{i},"relevance_score":{}}}"#, scores[i]))
            .collect();
        http_response(200, &format!(r#"{{"object":"list","results":[{}]}}"#, rows.join(",")))
    }

    #[test]
    fn scores_are_placed_by_index_not_by_array_order() {
        let (url, bodies, _) = spawn_server(move |_, body| canned(body, &[0.5, -2.0, 9.0]));
        let c = client_for(&url, 10);
        let got = c.rank("q", &["a", "b", "c"]).unwrap();
        assert_eq!(got, vec![0.5, -2.0, 9.0]);
        let sent: serde_json::Value = serde_json::from_str(&bodies.lock().unwrap()[0]).unwrap();
        assert_eq!(sent["query"], "q");
        assert_eq!(sent["model"], "test-reranker");
        assert_eq!(sent["documents"], serde_json::json!(["a", "b", "c"]));
    }

    #[test]
    fn a_short_score_list_is_an_error_never_a_partial_order() {
        let (url, _, _) = spawn_server(|_, _| {
            http_response(200, r#"{"results":[{"index":0,"relevance_score":1.0}]}"#)
        });
        let e = client_for(&url, 10).rank("q", &["a", "b"]).unwrap_err().to_string();
        assert!(e.contains("1 score(s) for 2 document(s)"), "{e}");
    }

    #[test]
    fn an_out_of_range_index_is_refused() {
        let (url, _, _) = spawn_server(|_, _| {
            http_response(
                200,
                r#"{"results":[{"index":0,"relevance_score":1.0},{"index":7,"relevance_score":2.0}]}"#,
            )
        });
        let e = client_for(&url, 10).rank("q", &["a", "b"]).unwrap_err().to_string();
        assert!(e.contains("index 7"), "{e}");
    }

    #[test]
    fn a_repeated_index_is_refused_rather_than_silently_overwritten() {
        let (url, _, _) = spawn_server(|_, _| {
            http_response(
                200,
                r#"{"results":[{"index":1,"relevance_score":1.0},{"index":1,"relevance_score":2.0}]}"#,
            )
        });
        let e = client_for(&url, 10).rank("q", &["a", "b"]).unwrap_err().to_string();
        assert!(e.contains("index 1 twice"), "{e}");
    }

    #[test]
    fn non_2xx_and_unparseable_are_errors() {
        let (url, _, _) = spawn_server(|_, _| http_response(503, "upstream is down"));
        let e = client_for(&url, 10).rank("q", &["a", "b"]).unwrap_err().to_string();
        assert!(e.contains("503"), "{e}");

        let (url2, _, _) = spawn_server(|_, _| http_response(200, "not json at all"));
        let e2 = client_for(&url2, 10).rank("q", &["a", "b"]).unwrap_err().to_string();
        assert!(e2.contains("unparseable"), "{e2}");
    }

    #[test]
    fn a_private_endpoint_is_refused_without_an_allow_entry() {
        let mut cfg = cfg_for("http://10.1.2.3:8080/v1", 10);
        assert!(RerankClient::new(&cfg).is_err(), "private range must be refused");
        cfg.endpoint = "http://not-a-url".into();
        assert!(RerankClient::new(&cfg).is_err());
    }

    #[test]
    fn disabled_or_unconfigured_never_constructs() {
        let mut cfg = cfg_for("http://127.0.0.1:1/v1", 10);
        cfg.enabled = false;
        assert!(RerankClient::new(&cfg).unwrap_err().to_string().contains("disabled"));
        let mut cfg = cfg_for("http://127.0.0.1:1/v1", 10);
        cfg.model = String::new();
        assert!(RerankClient::new(&cfg).unwrap_err().to_string().contains("not configured"));
        let mut cfg = cfg_for("http://127.0.0.1:1/v1", 10);
        cfg.candidates = 0;
        assert!(RerankClient::new(&cfg).unwrap_err().to_string().contains("at least 1"));
    }

    // ---- apply(): ordering, the candidate bound, and honest degradation ----

    #[test]
    fn apply_reorders_the_head_and_leaves_the_tail_in_retrieval_order() {
        // 3 candidates of 5. Head scores promote "c" over "a".
        let (url, _, _) = spawn_server(move |_, body| canned(body, &[1.0, 0.0, 5.0]));
        let c = client_for(&url, 3);
        let hits = vec!["a", "b", "c", "d", "e"];
        let out = apply(&c, "q", hits, |h: &&str| h.to_string());
        assert!(out.note.is_none(), "{:?}", out.note);
        assert_eq!(out.hits, vec!["c", "a", "b", "d", "e"], "reranked head, then the untouched tail");
    }

    #[test]
    fn only_the_candidate_window_is_ever_sent() {
        let (url, bodies, _) = spawn_server(move |_, body| canned(body, &[1.0, 2.0]));
        let c = client_for(&url, 2);
        let out = apply(&c, "q", vec!["a", "b", "c", "d"], |h: &&str| h.to_string());
        assert_eq!(out.hits, vec!["b", "a", "c", "d"]);
        let sent: serde_json::Value = serde_json::from_str(&bodies.lock().unwrap()[0]).unwrap();
        assert_eq!(sent["documents"], serde_json::json!(["a", "b"]), "the tail must cost nothing");
    }

    #[test]
    fn equal_scores_keep_retrieval_order() {
        // A model that cannot separate candidates must not be allowed to
        // shuffle them: ties fall back to what retrieval decided.
        let (url, _, _) = spawn_server(move |_, body| canned(body, &[1.0, 1.0, 1.0]));
        let c = client_for(&url, 3);
        let out = apply(&c, "q", vec!["a", "b", "c"], |h: &&str| h.to_string());
        assert_eq!(out.hits, vec!["a", "b", "c"]);
    }

    #[test]
    fn an_unavailable_endpoint_keeps_the_answer_and_says_why() {
        // Nothing is listening on this port.
        let c = client_for("http://127.0.0.1:1/v1", 10);
        let out = apply(&c, "q", vec!["a", "b", "c"], |h: &&str| h.to_string());
        assert_eq!(out.hits, vec!["a", "b", "c"], "retrieval order survives");
        let note = out.note.expect("degradation must be reported, never silent");
        assert!(note.contains("rerank unavailable"), "{note}");
        assert!(!note.contains('\n'), "the note rides in --json and beside hits: {note}");
    }

    #[test]
    fn a_list_too_short_to_reorder_spends_no_request() {
        let (url, bodies, _) = spawn_server(|_, _| http_response(500, "must not be called"));
        let c = client_for(&url, 10);
        let out = apply(&c, "q", vec!["only"], |h: &&str| h.to_string());
        assert_eq!(out.hits, vec!["only"]);
        assert!(out.note.is_none());
        assert!(bodies.lock().unwrap().is_empty(), "a single hit needs no forward pass");
    }
}
