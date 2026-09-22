use super::*;

#[test]
fn ignored_nested_repositories_form_one_current_graph_without_secret_nodes() {
    let brain = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    for (path, content) in [
        (
            "knowledge/personal/project.md",
            "# Project\n\n[[knowledge/shared/guide]]\n",
        ),
        (
            "knowledge/shared/guide.md",
            "# Guide\n\n[[knowledge/behaviours/skills/team/runbook]]\n",
        ),
        (
            "knowledge/behaviours/skills/team/runbook.md",
            "# Runbook\n\nA verified procedure.\n",
        ),
        (
            "knowledge/secrets/hidden.md",
            "# Secret\n\n[[knowledge/shared/guide]]\n",
        ),
        (".gitignore", "/knowledge/\n"),
        (
            "knowledge/.gitignore",
            "/personal/\n/shared/\n/behaviours/\n",
        ),
    ] {
        let path = brain.path().join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }
    for repo in [
        "",
        "knowledge",
        "knowledge/shared",
        "knowledge/behaviours/skills/team",
    ] {
        std::fs::create_dir_all(brain.path().join(repo).join(".git")).unwrap();
    }
    let mut conn = crate::index::open(state.path()).unwrap();
    let rules = crate::config::RingRules::default();
    crate::index::scan(&mut conn, brain.path(), None, &rules).unwrap();
    let route = trace(&conn, "project", "runbook", 3).unwrap();
    assert!(route.found);
    assert_eq!(route.paths.len(), 3);
    assert_eq!(build(&conn, None, 40).unwrap().total_nodes, 3);
    // An independently removed checkout must not remain as a ghost graph.
    std::fs::remove_file(brain.path().join("knowledge/shared/guide.md")).unwrap();
    crate::index::scan(&mut conn, brain.path(), None, &rules).unwrap();
    assert!(!trace(&conn, "project", "runbook", 3).unwrap().found);
    assert_eq!(build(&conn, None, 40).unwrap().unresolved_references, 1);
}

fn graph_db() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
             CREATE TABLE docs(
               id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, ring INTEGER NOT NULL,
               mtime INTEGER NOT NULL, size INTEGER NOT NULL
             );
             CREATE TABLE blocks(
               id INTEGER PRIMARY KEY, cite TEXT NOT NULL, doc_id INTEGER NOT NULL,
               start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, text TEXT NOT NULL,
               ctx TEXT NOT NULL DEFAULT '', chain TEXT NOT NULL DEFAULT '',
               hash TEXT NOT NULL DEFAULT '',
               embedding_text TEXT NOT NULL, embedding_hash TEXT NOT NULL
             );
             CREATE TABLE links(from_doc INTEGER NOT NULL, to_doc INTEGER NOT NULL);
             CREATE TABLE doc_links(doc_id INTEGER NOT NULL, target TEXT NOT NULL);
             INSERT INTO meta(key, value) VALUES('generation', '7');",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO docs(id, path, ring, mtime, size) VALUES
             (1, 'mind/overview.md', 1, 1, 1),
             (2, 'projects/alpha.md', 3, 1, 1),
             (3, 'knowledge/rust.md', 4, 1, 1),
             (4, 'knowledge/isolated.md', 4, 1, 1)",
        [],
    )
    .unwrap();
    conn.execute(
            "INSERT INTO blocks(cite, doc_id, start_line, end_line, text, ctx, chain, hash, embedding_text, embedding_hash)
             VALUES ('r1-a', 1, 1, 1, 'overview', '', '', 'a', 'overview', 'payload-a'),
                    ('r3-b', 2, 1, 1, 'alpha', '', '', 'b', 'alpha', 'payload-b')",
            [],
        )
        .unwrap();
    conn.execute(
        "INSERT INTO links(from_doc, to_doc) VALUES (1, 2), (2, 3)",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO doc_links(doc_id, target) VALUES
             (1, 'alpha'), (2, 'rust'), (3, 'missing')",
        [],
    )
    .unwrap();
    conn
}

#[test]
fn routes_cross_topics_and_follow_backlinks_with_original_edge_direction() {
    let conn = graph_db();
    let route = trace(&conn, "rust", "overview", 2).unwrap();
    assert!(route.found);
    assert_eq!(
        route.paths,
        ["knowledge/rust.md", "projects/alpha.md", "mind/overview.md"]
    );
    assert_eq!(route.edges[0].from, "projects/alpha.md");
    assert_eq!(route.edges[0].to, "knowledge/rust.md");
    assert!(!trace(&conn, "rust", "overview", 1).unwrap().found);
    assert!(!trace(&conn, "rust", "isolated", 32).unwrap().found);
    assert!(trace(&conn, "rust", "rust", 1).unwrap().found);
}

#[test]
fn paths_never_guess_missing_or_ambiguous_endpoints() {
    let conn = graph_db();
    conn.execute(
        "INSERT INTO docs VALUES (5, 'knowledge/other/rust.md', 3, 1, 1)",
        [],
    )
    .unwrap();
    let ambiguous = trace(&conn, "rust", "overview", 6).unwrap();
    assert!(!ambiguous.found);
    assert_eq!(ambiguous.from_matches.len(), 2);
    assert!(ambiguous.paths.is_empty());
    let missing = trace(&conn, "absent", "overview", 6).unwrap();
    assert!(missing.from_matches.is_empty());
    assert!(!missing.found);
    assert!(trace(&conn, "knowledge/rust", "overview", 6).unwrap().found);
}

#[test]
fn path_cannot_traverse_a_provisional_memory_even_if_an_old_index_contains_it() {
    let conn = graph_db();
    conn.execute("UPDATE docs SET ring = 5 WHERE id = 2", [])
        .unwrap();
    assert!(!trace(&conn, "rust", "overview", 6).unwrap().found);
}

#[test]
fn focus_resolves_a_doc_and_walks_its_curated_neighborhood() {
    let graph = build(&graph_db(), Some("alpha"), 3).unwrap();
    assert!(graph.focus_matched);
    assert_eq!(graph.resolved_focus.as_deref(), Some("projects/alpha.md"));
    assert_eq!(graph.nodes.len(), 3);
    assert!(graph.nodes[0].focused);
    assert_eq!(graph.edges.len(), 2);
    assert_eq!(graph.unresolved_references, 1);
}

#[test]
fn slice_filter_removes_hidden_nodes_and_incident_edges() {
    let graph =
        build_matching(&graph_db(), None, 20, |path| path.starts_with("knowledge/")).unwrap();
    assert_eq!(graph.total_nodes, 2);
    assert!(graph.edges.is_empty());
    assert!(
        graph
            .nodes
            .iter()
            .all(|node| node.path.starts_with("knowledge/"))
    );
}

#[test]
fn missing_focus_is_truthful_and_falls_back_to_ranked_overview() {
    let graph = build(&graph_db(), Some("absent"), 2).unwrap();
    assert!(!graph.focus_matched);
    assert!(graph.resolved_focus.is_none());
    assert_eq!(graph.nodes.len(), 2);
}

#[test]
fn ambiguous_note_name_is_reported_instead_of_guessed() {
    let conn = graph_db();
    conn.execute(
        "INSERT INTO docs(id, path, ring, mtime, size) VALUES
             (5, 'projects/other/alpha.md', 2, 1, 1)",
        [],
    )
    .unwrap();

    let graph = build(&conn, Some("alpha"), 3).unwrap();

    assert!(!graph.focus_matched);
    assert!(graph.resolved_focus.is_none());
    assert_eq!(
        graph.ambiguous_focus,
        vec!["projects/alpha.md", "projects/other/alpha.md"]
    );
    assert!(!graph.nodes.iter().any(|node| node.focused));
}

#[test]
fn repeated_links_are_edges_not_false_unresolved_references() {
    let conn = graph_db();
    conn.execute("INSERT INTO links(from_doc, to_doc) VALUES (1, 2)", [])
        .unwrap();
    conn.execute(
        "INSERT INTO doc_links(doc_id, target) VALUES (1, 'alpha')",
        [],
    )
    .unwrap();

    let graph = build(&conn, None, 20).unwrap();

    assert_eq!(graph.total_edges, 2);
    assert_eq!(graph.unresolved_references, 1);
}
