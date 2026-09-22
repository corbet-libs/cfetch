//! The ranking pipeline: retrieve, optionally rank by vectors, optionally
//! rerank with a cross-encoder.
//!
//! This exists so there is exactly ONE answer to "how does cfetch rank?".
//! Two callers need it and they are on opposite sides of the network: the
//! local CLI on a host that holds its own index, and the serving daemon
//! answering for a host that holds nothing. If each carried its own copy, the
//! same query against the same tree could be ranked two different ways
//! depending on who happened to answer — the coherence failure this codebase
//! exists to prevent, in the ranking rather than in the catalog.
//!
//! Both stages past retrieval are optional and both degrade loudly: the note
//! this returns is the caller's to surface, never to swallow. So does the
//! third, which removes rather than reorders: the precision gate below.

use crate::config::Config;
use crate::index::Hit;
use crate::{embed, index, rerank};

/// Query words that carry no topic. They occur in nearly every block, so
/// counting one as evidence would admit exactly the hits the gate exists to
/// refuse. Deliberately short and English: a word that is merely FREQUENT in
/// a particular corpus is still evidence, and only structural words —
/// articles, pronouns, auxiliaries, the question openers a recall query is
/// phrased with — belong here.
const STOPWORDS: &[&str] = &[
    "a", "about", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by", "can", "did",
    "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it",
    "its", "me", "my", "no", "not", "of", "on", "or", "our", "should", "so", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
];

/// Splits text into the tokens the gate compares, lowercased.
///
/// The split is on every non-alphanumeric character, which is what FTS5's
/// tokenizer does — and deliberately NOT what `index::fts_query` does, since
/// that keeps `-` and `_` inside a term. A query for "cross-encoder" reaches
/// the index as a two-token phrase and matches a block storing the same two
/// tokens; treating it here as one term no block can carry would drop a hit
/// retrieval had every right to find.
fn tokens(text: &str) -> impl Iterator<Item = String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
}

/// The query's distinct topical terms, in order — what a hit is measured
/// against.
fn content_terms(query: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for t in tokens(query) {
        if !STOPWORDS.contains(&t.as_str()) && !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

/// How many of `terms` the hit carries.
///
/// A term counts when some word of the hit STARTS WITH it — the same prefix
/// rule `index::fts_query` asks FTS5 for, so the gate can never contradict
/// the retrieval that proposed the hit. The heading-chain context counts as
/// well: FTS indexes it as its own column, so a hit may legitimately owe its
/// match to it alone.
fn covered(hit: &Hit, terms: &[String]) -> usize {
    // The snippet carries the context and the body carries the evidence past
    // the snippet's 160-character cap; both are needed, neither alone.
    let words: Vec<String> = tokens(&hit.snippet).chain(tokens(&hit.text)).collect();
    terms
        .iter()
        .filter(|t| words.iter().any(|w| w.starts_with(t.as_str())))
        .count()
}

/// Drops hits carrying fewer than `floor` of the query's terms, and reports
/// how many went.
///
/// The best-ranked hit survives whatever it scores. An operator who asked a
/// question and got nothing back cannot tell a gate from an empty brain, and
/// one weak hit they can dismiss themselves is the cheaper failure.
fn gate(hits: Vec<Hit>, terms: &[String], floor: usize) -> (Vec<Hit>, usize) {
    let before = hits.len();
    let mut kept: Vec<Hit> = Vec::with_capacity(before);
    let mut best_refused: Option<Hit> = None;
    for hit in hits {
        if covered(&hit, terms) >= floor {
            kept.push(hit);
        } else if best_refused.is_none() {
            best_refused = Some(hit);
        }
    }
    if kept.is_empty()
        && let Some(hit) = best_refused
    {
        kept.push(hit);
    }
    let dropped = before - kept.len();
    (kept, dropped)
}

/// Ranked hits plus the single note describing every degradation that
/// happened on the way — joined rather than nested, because the caller has
/// exactly one line to show a human and one JSON field to fill.
#[derive(Debug)]
pub struct Ranked {
    pub hits: Vec<Hit>,
    pub note: Option<String>,
}

/// Ranks `query` to at most `limit` hits using whatever stages this host is
/// configured for.
///
/// Retrieval widens to the reranker's candidate window when one is
/// configured, because a cross-encoder can only promote what retrieval
/// proposed, and to twice `limit` when the precision gate is armed, because
/// the gate can only remove; the answer is cut back to `limit` at the end.
pub fn ranked(
    cfg: &Config,
    conn: &rusqlite::Connection,
    query: &str,
    limit: usize,
    semantic: bool,
    hybrid: bool,
    slice: Option<&str>,
) -> anyhow::Result<Ranked> {
    // Resolve the slice ONCE, before any retrieval: an unknown name must fail
    // loudly here rather than silently answer from the whole tree.
    let model = cfg.slice_model()?;
    let prefixes: Vec<String> = match slice {
        None => Vec::new(),
        Some(name) => match model.prefixes_of(name) {
            None => Vec::new(), // the root slice restricts nothing
            Some([]) => anyhow::bail!(
                "no slice named {name:?} (configured: {})",
                if model.is_empty() {
                    "none".to_string()
                } else {
                    model.names().collect::<Vec<_>>().join(", ")
                }
            ),
            Some(p) => p.to_vec(),
        },
    };
    let mut notes: Vec<String> = Vec::new();

    // A misconfigured reranker can only be fixed by a human, so it is named
    // once and the answer still comes back — the same stance as an
    // unreachable one, which `rerank::apply` reports.
    let reranker = if cfg.rerank.enabled {
        match rerank::RerankClient::new(&cfg.rerank) {
            Ok(c) => Some(c),
            Err(e) => {
                crate::runtime_status::record_inference_initialization_failure();
                notes.push(format!(
                    "rerank misconfigured ({e}) — answering in retrieval order"
                ));
                None
            }
        }
    } else {
        None
    };
    // The gate judges LEXICAL overlap, so it only ever runs on an answer
    // that lexical ranking alone produced. A vector hit sharing no word with
    // the query is precisely what semantic recall is for; a fused hybrid hit
    // cannot be traced back to the half that proposed it; and a cross-encoder
    // that read query and block together is a better judge than a word count
    // on its best day. In all three cases the gate stands down.
    let min_terms = cfg.recall.gate.min_terms;
    let lexical_only = !semantic && !hybrid && reranker.is_none();
    // Retrieval widens for the gate exactly as it does for the reranker: the
    // gate can only remove, so without a wider pool a gated answer would come
    // back short of the caller's limit while good hits sat just past the cut.
    // Same shape as the doubled candidate pool `index::recall_in` keeps so
    // duplicate suppression never shrinks a result.
    let retrieve = match &reranker {
        Some(client) => client.candidates().max(limit),
        None if min_terms > 1 && lexical_only => limit.saturating_mul(2),
        None => limit,
    };

    let hits = if semantic || hybrid {
        let out = embed::semantic_hits(cfg, conn, query, retrieve, hybrid, &prefixes)?;
        if let Some(n) = out.note {
            notes.push(n);
        }
        out.hits
    } else {
        index::recall_in(conn, query, retrieve, &prefixes)?
    };

    let mut hits = match &reranker {
        Some(client) => {
            let out = rerank::apply(client, query, hits, |h: &Hit| h.snippet.clone());
            if let Some(n) = out.note {
                notes.push(n);
            }
            out.hits
        }
        None => hits,
    };

    // Last, and before the cut to `limit`: admission is a different question
    // from rank, and filtering ahead of the truncation is what lets the hits
    // that pass refill the slots the dropped ones free.
    if min_terms > 1 {
        if lexical_only {
            let terms = content_terms(query);
            // Below two terms there is nothing to corroborate a hit with, and
            // a query that short is unambiguous by construction.
            let floor = min_terms.min(terms.len());
            if floor > 1 {
                let (kept, dropped) = gate(hits, &terms, floor);
                hits = kept;
                if dropped > 0 {
                    notes.push(format!(
                        "precision gate dropped {dropped} hit(s) carrying under {floor} of the query's {} term(s)",
                        terms.len()
                    ));
                }
            }
        } else {
            notes.push(format!(
                "precision gate (min_terms {min_terms}) not applied: it weighs lexical overlap only"
            ));
        }
    }
    hits.truncate(limit);

    let note = (!notes.is_empty()).then(|| notes.join("; "));
    let degraded = crate::runtime_status::retrieval_note_is_degraded(note.as_deref());
    crate::runtime_status::record_retrieval(
        if hybrid {
            crate::runtime_status::RetrievalMode::Hybrid
        } else if semantic {
            crate::runtime_status::RetrievalMode::Semantic
        } else {
            crate::runtime_status::RetrievalMode::Lexical
        },
        degraded,
    );
    crate::runtime_status::record_memory_answer(
        crate::runtime_status::MemoryRoute::Local,
        Some(index::generation(conn)),
        None,
        true,
    );
    Ok(Ranked { hits, note })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RerankConfig;
    use crate::testhttp::{http_response, spawn_server};

    /// Five single-line blocks, all matching the query "item", so ranking —
    /// not retrieval — decides what comes back.
    fn five_block_index() -> (tempfile::TempDir, tempfile::TempDir, rusqlite::Connection) {
        let brain = tempfile::tempdir().unwrap();
        let p = brain.path().join("knowledge/a.md");
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(
            p,
            "- item one\n- item two\n- item three\n- item four\n- item five\n",
        )
        .unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut conn = index::open(state.path()).unwrap();
        index::scan(
            &mut conn,
            brain.path(),
            None,
            &crate::config::RingRules::default(),
        )
        .unwrap();
        (brain, state, conn)
    }

    fn cfg_with_rerank(brain: &std::path::Path, url: &str, candidates: usize) -> Config {
        Config {
            brain_root: brain.to_path_buf(),
            rerank: RerankConfig {
                enabled: true,
                endpoint: url.to_string(),
                model: "test-reranker".into(),
                candidates,
                ..RerankConfig::default()
            },
            ..Config::default()
        }
    }

    /// Scores every document by its position, LAST document best, so a
    /// reranked answer is the exact reverse of retrieval order.
    fn reverse_scores(body: &str) -> String {
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        let n = v["documents"].as_array().unwrap().len();
        let rows: Vec<String> = (0..n)
            .map(|i| format!(r#"{{"index":{i},"relevance_score":{}}}"#, i as f32))
            .collect();
        http_response(200, &format!(r#"{{"results":[{}]}}"#, rows.join(",")))
    }

    #[test]
    fn retrieval_widens_to_the_candidate_window_and_the_answer_is_cut_back() {
        let (brain, _state, conn) = five_block_index();
        let (url, bodies, _) = spawn_server(|_, body| reverse_scores(body));
        let cfg = cfg_with_rerank(brain.path(), &url, 5);

        // limit 2, candidates 5: the reranker must SEE all five, or it could
        // never promote the fifth — which is the entire point of the window.
        let out = ranked(&cfg, &conn, "item", 2, false, false, None).unwrap();
        assert!(out.note.is_none(), "{:?}", out.note);
        let sent: serde_json::Value = serde_json::from_str(&bodies.lock().unwrap()[0]).unwrap();
        assert_eq!(
            sent["documents"].as_array().unwrap().len(),
            5,
            "the window is what gets sent"
        );
        assert_eq!(out.hits.len(), 2, "the caller asked for 2");
        assert!(
            out.hits[0].snippet.contains("five"),
            "last by retrieval, first by rerank: {:?}",
            out.hits[0]
        );
    }

    #[test]
    fn without_a_reranker_retrieval_never_widens() {
        let (brain, _state, conn) = five_block_index();
        let cfg = Config {
            brain_root: brain.path().to_path_buf(),
            ..Config::default()
        };
        let out = ranked(&cfg, &conn, "item", 2, false, false, None).unwrap();
        assert_eq!(out.hits.len(), 2);
        assert!(out.note.is_none());
        assert!(
            out.hits[0].snippet.contains("one"),
            "plain retrieval order: {:?}",
            out.hits[0]
        );
    }

    #[test]
    fn an_unreachable_reranker_still_answers_and_says_so() {
        let (brain, _state, conn) = five_block_index();
        let cfg = cfg_with_rerank(brain.path(), "http://127.0.0.1:1/v1", 5);
        let out = ranked(&cfg, &conn, "item", 3, false, false, None).unwrap();
        assert_eq!(
            out.hits.len(),
            3,
            "the answer survives the second stage failing"
        );
        let note = out.note.expect("degradation is never silent");
        assert!(note.contains("rerank unavailable"), "{note}");
    }

    #[test]
    fn a_misconfigured_reranker_is_named_once_and_the_answer_comes_back() {
        let (brain, _state, conn) = five_block_index();
        // Enabled but with no model: only a human can fix this, so it is
        // reported rather than retried per query.
        let mut cfg = cfg_with_rerank(brain.path(), "http://127.0.0.1:1/v1", 5);
        cfg.rerank.model = String::new();
        let out = ranked(&cfg, &conn, "item", 3, false, false, None).unwrap();
        assert_eq!(out.hits.len(), 3);
        let note = out
            .note
            .expect("a misconfiguration must reach the operator");
        assert!(note.contains("rerank misconfigured"), "{note}");
        assert!(!note.contains('\n'), "one line: {note}");
    }

    // ---- slices (S1 acceptance)

    /// Two slices over one tree, each holding a block that matches the query.
    fn two_slice_index() -> (tempfile::TempDir, tempfile::TempDir, rusqlite::Connection) {
        let brain = tempfile::tempdir().unwrap();
        for (rel, body) in [
            ("knowledge/hosts/server.md", "- item about the server\n"),
            ("knowledge/world/vendor.md", "- item about a vendor\n"),
            ("mind/memories/pref.md", "- item about a preference\n"),
        ] {
            let p = brain.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        }
        let state = tempfile::tempdir().unwrap();
        let mut conn = index::open(state.path()).unwrap();
        index::scan(
            &mut conn,
            brain.path(),
            None,
            &crate::config::RingRules::default(),
        )
        .unwrap();
        (brain, state, conn)
    }

    fn sliced_cfg(brain: &std::path::Path) -> Config {
        Config {
            brain_root: brain.to_path_buf(),
            slices: vec![
                crate::config::SliceRule {
                    name: "work".into(),
                    prefixes: vec!["knowledge".into()],
                },
                crate::config::SliceRule {
                    name: "hosts".into(),
                    prefixes: vec!["knowledge/hosts".into()],
                },
            ],
            ..Config::default()
        }
    }

    fn paths_of(r: &Ranked) -> Vec<String> {
        r.hits.iter().map(|h| h.path.clone()).collect()
    }

    #[test]
    fn a_slice_query_returns_only_that_slice_and_what_nests_inside_it() {
        let (brain, _state, conn) = two_slice_index();
        let cfg = sliced_cfg(brain.path());

        let inner = ranked(&cfg, &conn, "item", 10, false, false, Some("hosts")).unwrap();
        assert_eq!(paths_of(&inner), vec!["knowledge/hosts/server.md"]);

        // `hosts` is nested inside `work`, so restricting to `work` reaches
        // both — and still never leaves it.
        let outer = ranked(&cfg, &conn, "item", 10, false, false, Some("work")).unwrap();
        let mut got = paths_of(&outer);
        got.sort();
        assert_eq!(
            got,
            vec!["knowledge/hosts/server.md", "knowledge/world/vendor.md"]
        );
        assert!(
            !got.iter().any(|p| p.starts_with("mind/")),
            "the slice is a boundary"
        );
    }

    #[test]
    fn the_root_slice_is_the_whole_tree_and_matches_no_filter_at_all() {
        let (brain, _state, conn) = two_slice_index();
        let cfg = sliced_cfg(brain.path());
        let root = ranked(
            &cfg,
            &conn,
            "item",
            10,
            false,
            false,
            Some(crate::config::ROOT_SLICE),
        )
        .unwrap();
        let unfiltered = ranked(&cfg, &conn, "item", 10, false, false, None).unwrap();
        assert_eq!(paths_of(&root), paths_of(&unfiltered));
        assert_eq!(root.hits.len(), 3);
    }

    #[test]
    fn a_slice_that_does_not_exist_is_an_error_not_the_whole_tree() {
        // Silently widening a typo to the entire brain is how a private slice
        // leaks; refuse instead.
        let (brain, _state, conn) = two_slice_index();
        let cfg = sliced_cfg(brain.path());
        let e = ranked(&cfg, &conn, "item", 10, false, false, Some("hsots"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("no slice named"), "{e}");
        assert!(
            e.contains("work") && e.contains("hosts"),
            "the real names are offered: {e}"
        );
    }

    #[test]
    fn a_brain_with_no_slices_answers_exactly_as_it_did_before() {
        // The S1 acceptance test: a single-slice brain is byte-identical.
        let (brain, _state, conn) = two_slice_index();
        let plain = Config {
            brain_root: brain.path().to_path_buf(),
            ..Config::default()
        };
        let before = index::recall(&conn, "item", 10).unwrap();
        let after = ranked(&plain, &conn, "item", 10, false, false, None).unwrap();
        assert_eq!(
            paths_of(&after),
            before.iter().map(|h| h.path.clone()).collect::<Vec<_>>()
        );
        assert_eq!(
            after
                .hits
                .iter()
                .map(|h| h.cite.clone())
                .collect::<Vec<_>>(),
            before.iter().map(|h| h.cite.clone()).collect::<Vec<_>>()
        );
    }

    // ---- the precision gate

    /// A brain of one-line files, scanned into a fresh index.
    fn brain_of(
        files: &[(&str, &str)],
    ) -> (tempfile::TempDir, tempfile::TempDir, rusqlite::Connection) {
        let brain = tempfile::tempdir().unwrap();
        for (rel, body) in files {
            let p = brain.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        }
        let state = tempfile::tempdir().unwrap();
        let mut conn = index::open(state.path()).unwrap();
        index::scan(
            &mut conn,
            brain.path(),
            None,
            &crate::config::RingRules::default(),
        )
        .unwrap();
        (brain, state, conn)
    }

    fn gated_cfg(brain: &std::path::Path, min_terms: usize) -> Config {
        Config {
            brain_root: brain.to_path_buf(),
            recall: crate::config::RecallConfig {
                gate: crate::config::GateConfig { min_terms },
                ..crate::config::RecallConfig::default()
            },
            ..Config::default()
        }
    }

    /// One block about the query's subject, one that shares a single ordinary
    /// word with it — the shape `index::fts_query`'s OR-join retrieves and
    /// nothing else refuses.
    const MIXED: &[(&str, &str)] = &[
        (
            "knowledge/storage.md",
            "- zfs snapshot retention runs nightly\n",
        ),
        (
            "knowledge/people.md",
            "- policy on contributor onboarding\n",
        ),
    ];

    #[test]
    fn the_gate_is_off_by_default_and_the_weak_hit_still_comes_back() {
        let (brain, _state, conn) = brain_of(MIXED);
        let cfg = Config {
            brain_root: brain.path().to_path_buf(),
            ..Config::default()
        };
        let out = ranked(
            &cfg,
            &conn,
            "zfs snapshot retention policy",
            10,
            false,
            false,
            None,
        )
        .unwrap();
        let mut got = paths_of(&out);
        got.sort();
        assert_eq!(got, vec!["knowledge/people.md", "knowledge/storage.md"]);
        assert!(
            out.note.is_none(),
            "an unarmed gate says nothing: {:?}",
            out.note
        );
    }

    #[test]
    fn an_armed_gate_drops_the_one_word_hit_and_names_the_drop() {
        let (brain, _state, conn) = brain_of(MIXED);
        let cfg = gated_cfg(brain.path(), 2);
        let out = ranked(
            &cfg,
            &conn,
            "zfs snapshot retention policy",
            10,
            false,
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            paths_of(&out),
            vec!["knowledge/storage.md"],
            "one shared word is not evidence"
        );
        let note = out.note.expect("a removal is never silent");
        assert!(note.contains("precision gate dropped 1 hit"), "{note}");
        assert!(note.contains("under 2 of the query's 4 term(s)"), "{note}");
    }

    #[test]
    fn the_gate_never_empties_an_answer() {
        // Nothing in this brain reaches the floor. An operator who cannot
        // tell a gate from an empty brain is worse off than one holding a
        // weak hit they can dismiss themselves.
        let (brain, _state, conn) = brain_of(&[
            (
                "knowledge/people.md",
                "- policy on contributor onboarding\n",
            ),
            ("knowledge/pools.md", "- the zfs pool layout diagram\n"),
        ]);
        let cfg = gated_cfg(brain.path(), 2);
        let out = ranked(
            &cfg,
            &conn,
            "zfs snapshot retention policy",
            10,
            false,
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            out.hits.len(),
            1,
            "the best-ranked hit survives whatever it scores"
        );
        assert!(
            out.note
                .expect("still said out loud")
                .contains("precision gate dropped 1 hit")
        );
    }

    #[test]
    fn a_query_too_short_to_corroborate_is_never_gated_against_itself() {
        let (brain, _state, conn) = brain_of(MIXED);
        let cfg = gated_cfg(brain.path(), 3);
        // "the" is structural, so this query has exactly one term to spend —
        // and a floor of 3 against it would answer nothing but the floor.
        let out = ranked(&cfg, &conn, "the policy", 10, false, false, None).unwrap();
        assert_eq!(paths_of(&out), vec!["knowledge/people.md"]);
        assert!(out.note.is_none(), "{:?}", out.note);
    }

    #[test]
    fn the_gate_weighs_the_whole_block_not_the_snippet() {
        // The second term sits past the snippet's 160-character cap. A gate
        // reading the display form would drop a block that carries both.
        let long = format!("- zfs {}retention nightly\n", "padding ".repeat(30));
        let (brain, _state, conn) = brain_of(&[
            ("knowledge/short.md", "- zfs retention basics\n"),
            ("knowledge/long.md", &long),
        ]);
        let cfg = gated_cfg(brain.path(), 2);
        let out = ranked(&cfg, &conn, "zfs retention", 10, false, false, None).unwrap();
        let mut got = paths_of(&out);
        got.sort();
        assert_eq!(got, vec!["knowledge/long.md", "knowledge/short.md"]);
        assert!(out.note.is_none(), "nothing was dropped: {:?}", out.note);
    }

    #[test]
    fn an_armed_gate_widens_retrieval_so_the_answer_is_not_short() {
        // "policy" occurs once and the other terms six times each, so BM25
        // ranks the one-word hit first. Retrieving exactly `limit` would hand
        // the gate nothing but that hit; the answer must still be the block
        // about the subject.
        let mut files: Vec<(String, String)> = vec![(
            "knowledge/people.md".to_string(),
            "- policy on contributor onboarding\n".to_string(),
        )];
        for i in 0..6 {
            files.push((
                format!("knowledge/storage-{i}.md"),
                format!("- zfs snapshot retention on pool {i}\n"),
            ));
        }
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let (brain, _state, conn) = brain_of(&refs);

        let plain = Config {
            brain_root: brain.path().to_path_buf(),
            ..Config::default()
        };
        let ungated = ranked(
            &plain,
            &conn,
            "zfs snapshot retention policy",
            1,
            false,
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            paths_of(&ungated),
            vec!["knowledge/people.md"],
            "the weak hit outranks"
        );

        let cfg = gated_cfg(brain.path(), 2);
        let out = ranked(
            &cfg,
            &conn,
            "zfs snapshot retention policy",
            1,
            false,
            false,
            None,
        )
        .unwrap();
        assert_eq!(out.hits.len(), 1, "the caller's limit is still filled");
        assert!(
            out.hits[0].path.starts_with("knowledge/storage-"),
            "a hit past the cut refilled the freed slot: {:?}",
            out.hits[0]
        );
    }

    #[test]
    fn the_gate_stands_down_for_a_reranked_answer_and_says_so() {
        // A cross-encoder read the query and the block together. Letting a
        // word count overrule it would be the crude judge vetoing the good
        // one — but an operator who armed the gate must not be left assuming
        // it filtered.
        let (brain, _state, conn) = brain_of(MIXED);
        let (url, _bodies, _) = spawn_server(|_, body| reverse_scores(body));
        let mut cfg = cfg_with_rerank(brain.path(), &url, 5);
        cfg.recall.gate.min_terms = 2;
        let out = ranked(
            &cfg,
            &conn,
            "zfs snapshot retention policy",
            10,
            false,
            false,
            None,
        )
        .unwrap();
        assert_eq!(out.hits.len(), 2, "nothing is dropped behind a reranker");
        let note = out.note.expect("standing down is a fact the caller needs");
        assert!(
            note.contains("precision gate (min_terms 2) not applied"),
            "{note}"
        );
    }

    #[test]
    fn content_terms_keeps_only_what_a_hit_can_be_measured_against() {
        assert_eq!(
            content_terms("how do I configure the rerank endpoint"),
            vec!["configure", "rerank", "endpoint"]
        );
        // Hyphens split, matching FTS5's tokenizer rather than `fts_query`'s
        // term splitter, so a term no block could ever carry is never counted.
        assert_eq!(content_terms("cross-encoder"), vec!["cross", "encoder"]);
        // Repeats are one term: a query cannot corroborate itself.
        assert_eq!(content_terms("backup backup BACKUP"), vec!["backup"]);
        assert!(content_terms("what is the").is_empty());
    }
}
