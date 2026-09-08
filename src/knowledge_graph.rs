//! Read-only knowledge graph derived from the Markdown catalog.
//!
//! Nodes are indexed Markdown documents and edges are the same resolved,
//! human-authored Obsidian wikilinks used by recall expansion. The graph is a
//! disposable view: Markdown remains the record and an unresolved or
//! ambiguous link never becomes a guessed edge.

use std::collections::{BTreeSet, HashMap, VecDeque};

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

const MAX_NODES: usize = 200;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeNode {
    pub path: String,
    pub ring: u8,
    pub kind: String,
    pub blocks: usize,
    pub inbound: usize,
    pub outbound: usize,
    pub focused: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeEdge {
    pub from: String,
    pub to: String,
    pub relation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    pub generation: u64,
    pub requested_focus: Option<String>,
    pub resolved_focus: Option<String>,
    /// Equally specific matches when a note name is ambiguous. cfetch never
    /// chooses one by ring or lexical order because that would turn a display
    /// convenience into a guessed relationship.
    #[serde(default)]
    pub ambiguous_focus: Vec<String>,
    pub focus_matched: bool,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub unresolved_references: usize,
    pub nodes: Vec<KnowledgeNode>,
    pub edges: Vec<KnowledgeEdge>,
    pub omitted_edges: usize,
}

#[derive(Debug)]
struct Doc {
    id: i64,
    path: String,
    ring: u8,
    blocks: usize,
}

fn kind(path: &str) -> &'static str {
    if path.starts_with("native:") {
        "agent-memory"
    } else if path.starts_with("mind/") {
        "memory"
    } else if path.starts_with("projects/") {
        "project"
    } else if path.starts_with("knowledge/") {
        "knowledge"
    } else if path.starts_with("todo/") {
        "task"
    } else {
        "document"
    }
}

fn focus_score(path: &str, query: &str) -> Option<u8> {
    let path = path.to_ascii_lowercase();
    let stemless = path.strip_suffix(".md").unwrap_or(&path);
    let query = query
        .trim()
        .trim_end_matches(".md")
        .trim_matches('/')
        .to_ascii_lowercase();
    if query.is_empty() {
        return None;
    }
    let stem = stemless.rsplit('/').next().unwrap_or(stemless);
    if stemless == query {
        Some(4)
    } else if stem == query {
        Some(3)
    } else if stemless.ends_with(&format!("/{query}")) {
        Some(2)
    } else if stemless.contains(&query) {
        Some(1)
    } else {
        None
    }
}

pub fn build(
    conn: &Connection,
    focus: Option<&str>,
    limit: usize,
) -> anyhow::Result<KnowledgeGraph> {
    build_matching(conn, focus, limit, |_| true)
}

/// Builds a graph over only the documents accepted by `visible`. This is the
/// slice boundary for peer queries: hidden nodes and their incident edges are
/// absent rather than summarized.
pub fn build_matching(
    conn: &Connection,
    focus: Option<&str>,
    limit: usize,
    visible: impl Fn(&str) -> bool,
) -> anyhow::Result<KnowledgeGraph> {
    let limit = limit.clamp(1, MAX_NODES);
    let mut stmt = conn.prepare(
        "SELECT d.id, d.path, d.ring, COUNT(b.id)
         FROM docs d LEFT JOIN blocks b ON b.doc_id = d.id
         GROUP BY d.id ORDER BY d.path",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(Doc {
            id: row.get(0)?,
            path: row.get(1)?,
            ring: row.get::<_, i64>(2)? as u8,
            blocks: row.get::<_, i64>(3)? as usize,
        })
    })?;
    let docs: Vec<Doc> = rows
        .filter_map(Result::ok)
        .filter(|doc| visible(&doc.path))
        .collect();
    let by_id: HashMap<i64, usize> = docs
        .iter()
        .enumerate()
        .map(|(index, doc)| (doc.id, index))
        .collect();

    let mut link_stmt = conn.prepare("SELECT from_doc, to_doc FROM links")?;
    let link_rows = link_stmt.query_map([], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
    })?;
    let resolved_links: Vec<(usize, usize)> = link_rows
        .filter_map(Result::ok)
        .filter_map(|(from, to)| Some((*by_id.get(&from)?, *by_id.get(&to)?)))
        .collect();
    // Self-references (`[[overview]]` inside `overview.md`) resolve to the
    // same doc: they are valid links (resolve_links deliberately refuses to
    // insert them into `links`), but counting them as unresolved inflated
    // the broken-link metric by exactly the self-reference count.
    let self_references = resolved_links.iter().filter(|(from, to)| from == to).count();
    let resolved_references = resolved_links.len().saturating_sub(self_references);
    let mut links: Vec<(usize, usize)> = resolved_links
        .into_iter()
        .filter(|(from, to)| from != to)
        .collect();
    links.sort_unstable();
    links.dedup();

    let mut inbound = vec![0usize; docs.len()];
    let mut outbound = vec![0usize; docs.len()];
    let mut adjacency = vec![Vec::new(); docs.len()];
    for &(from, to) in &links {
        outbound[from] += 1;
        inbound[to] += 1;
        adjacency[from].push(to);
        adjacency[to].push(from);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let requested_focus = focus.map(str::trim).filter(|value| !value.is_empty());
    let focus_matches = requested_focus
        .map(|query| {
            let scored: Vec<(usize, u8)> = docs
                .iter()
                .enumerate()
                .filter_map(|(index, doc)| focus_score(&doc.path, query).map(|score| (index, score)))
                .collect();
            let best = scored.iter().map(|(_, score)| *score).max();
            scored
                .into_iter()
                .filter_map(|(index, score)| (Some(score) == best).then_some(index))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let focused = (focus_matches.len() == 1).then(|| focus_matches[0]);
    let mut ambiguous_focus = if focus_matches.len() > 1 {
        focus_matches
            .iter()
            .map(|index| docs[*index].path.clone())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    ambiguous_focus.sort();

    let degree = |index: usize| inbound[index] + outbound[index];
    let mut selected = BTreeSet::new();
    if let Some(seed) = focused {
        let mut queue = VecDeque::from([seed]);
        selected.insert(seed);
        while let Some(current) = queue.pop_front() {
            let mut neighbors = adjacency[current].clone();
            neighbors.sort_by(|left, right| {
                degree(*right)
                    .cmp(&degree(*left))
                    .then_with(|| docs[*left].ring.cmp(&docs[*right].ring))
                    .then_with(|| docs[*left].path.cmp(&docs[*right].path))
            });
            for neighbor in neighbors {
                if selected.len() >= limit {
                    break;
                }
                if selected.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
            if selected.len() >= limit {
                break;
            }
        }
    } else {
        let mut ranked: Vec<usize> = (0..docs.len()).collect();
        ranked.sort_by(|left, right| {
            degree(*right)
                .cmp(&degree(*left))
                .then_with(|| docs[*left].ring.cmp(&docs[*right].ring))
                .then_with(|| docs[*left].path.cmp(&docs[*right].path))
        });
        selected.extend(ranked.into_iter().take(limit));
    }

    let mut nodes: Vec<KnowledgeNode> = selected
        .iter()
        .map(|&index| KnowledgeNode {
            path: docs[index].path.clone(),
            ring: docs[index].ring,
            kind: kind(&docs[index].path).to_string(),
            blocks: docs[index].blocks,
            inbound: inbound[index],
            outbound: outbound[index],
            focused: focused == Some(index),
        })
        .collect();
    nodes.sort_by(|left, right| {
        right
            .focused
            .cmp(&left.focused)
            .then_with(|| (right.inbound + right.outbound).cmp(&(left.inbound + left.outbound)))
            .then_with(|| left.ring.cmp(&right.ring))
            .then_with(|| left.path.cmp(&right.path))
    });

    let mut edges: Vec<KnowledgeEdge> = links
        .iter()
        .filter(|(from, to)| selected.contains(from) && selected.contains(to))
        .map(|(from, to)| KnowledgeEdge {
            from: docs[*from].path.clone(),
            to: docs[*to].path.clone(),
            relation: "curated_link".to_string(),
        })
        .collect();
    edges.sort_by(|left, right| left.from.cmp(&right.from).then(left.to.cmp(&right.to)));
    let edge_limit = limit.saturating_mul(4);
    let omitted_edges = edges.len().saturating_sub(edge_limit);
    edges.truncate(edge_limit);

    let raw_references = conn
        .prepare(
            "SELECT d.path FROM doc_links dl JOIN docs d ON d.id = dl.doc_id",
        )
        .and_then(|mut statement| {
            let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
            Ok(rows
                .filter_map(Result::ok)
                .filter(|path| visible(path))
                .count())
        })
        .unwrap_or(0);
    Ok(KnowledgeGraph {
        generation: crate::index::generation(conn),
        requested_focus: requested_focus.map(str::to_string),
        resolved_focus: focused.map(|index| docs[index].path.clone()),
        ambiguous_focus,
        focus_matched: requested_focus.is_none() || focused.is_some(),
        total_nodes: docs.len(),
        total_edges: links.len(),
        unresolved_references: raw_references.saturating_sub(resolved_references),
        nodes,
        edges,
        omitted_edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let graph = build_matching(&graph_db(), None, 20, |path| path.starts_with("knowledge/"))
            .unwrap();
        assert_eq!(graph.total_nodes, 2);
        assert!(graph.edges.is_empty());
        assert!(graph.nodes.iter().all(|node| node.path.starts_with("knowledge/")));
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
}
