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

/// A shortest explanation through visible, explicitly authored links.
/// Traversal may follow a backlink; each returned edge retains its original
/// direction. Missing or ambiguous endpoints produce no invented route.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgePath {
    pub generation: u64,
    pub from_matches: Vec<String>,
    pub to_matches: Vec<String>,
    pub max_depth: usize,
    pub found: bool,
    pub paths: Vec<String>,
    pub edges: Vec<KnowledgeEdge>,
}

pub fn trace(
    conn: &Connection,
    from: &str,
    to: &str,
    max_depth: usize,
) -> anyhow::Result<KnowledgePath> {
    anyhow::ensure!(
        (1..=32).contains(&max_depth),
        "graph depth must be between 1 and 32"
    );
    let mut statement = conn.prepare("SELECT id, path FROM docs WHERE ring <= 4 ORDER BY path")?;
    let docs = statement
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let matches = |query: &str| {
        let scored: Vec<_> = docs
            .iter()
            .enumerate()
            .filter_map(|(index, (_, path))| focus_score(path, query).map(|score| (index, score)))
            .collect();
        let best = scored.iter().map(|(_, score)| *score).max();
        scored
            .into_iter()
            .filter_map(|(index, score)| (Some(score) == best).then_some(index))
            .collect::<Vec<_>>()
    };
    let from_matches = matches(from);
    let to_matches = matches(to);
    let mut result = KnowledgePath {
        generation: crate::index::generation(conn),
        from_matches: from_matches.iter().map(|&i| docs[i].1.clone()).collect(),
        to_matches: to_matches.iter().map(|&i| docs[i].1.clone()).collect(),
        max_depth,
        found: false,
        paths: Vec::new(),
        edges: Vec::new(),
    };
    if from_matches.len() != 1 || to_matches.len() != 1 {
        return Ok(result);
    }
    let (start, end) = (from_matches[0], to_matches[0]);
    let by_id: HashMap<_, _> = docs
        .iter()
        .enumerate()
        .map(|(index, (id, _))| (*id, index))
        .collect();
    let mut statement =
        conn.prepare("SELECT from_doc, to_doc FROM links ORDER BY from_doc, to_doc")?;
    let edges = statement
        .query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    let mut adjacency = vec![Vec::new(); docs.len()];
    let mut directed = BTreeSet::new();
    for (from, to) in edges {
        if let (Some(&from), Some(&to)) = (by_id.get(&from), by_id.get(&to)) {
            adjacency[from].push(to);
            adjacency[to].push(from);
            directed.insert((from, to));
        }
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    let mut visited = vec![false; docs.len()];
    let mut parent = vec![None; docs.len()];
    let mut queue = VecDeque::from([(start, 0)]);
    visited[start] = true;
    while let Some((node, depth)) = queue.pop_front() {
        if node == end {
            let mut route = vec![end];
            let mut current = end;
            while let Some(previous) = parent[current] {
                route.push(previous);
                current = previous;
            }
            route.reverse();
            result.paths = route.iter().map(|&index| docs[index].1.clone()).collect();
            for pair in route.windows(2) {
                let (a, b) = if directed.contains(&(pair[0], pair[1])) {
                    (pair[0], pair[1])
                } else {
                    (pair[1], pair[0])
                };
                result.edges.push(KnowledgeEdge {
                    from: docs[a].1.clone(),
                    to: docs[b].1.clone(),
                    relation: "curated_link".into(),
                });
            }
            result.found = true;
            return Ok(result);
        }
        if depth == max_depth {
            continue;
        }
        for &next in &adjacency[node] {
            if !visited[next] {
                visited[next] = true;
                parent[next] = Some(node);
                queue.push_back((next, depth + 1));
            }
        }
    }
    Ok(result)
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
    let link_rows =
        link_stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?;
    let resolved_links: Vec<(usize, usize)> = link_rows
        .filter_map(Result::ok)
        .filter_map(|(from, to)| Some((*by_id.get(&from)?, *by_id.get(&to)?)))
        .collect();
    // Self-references (`[[overview]]` inside `overview.md`) resolve to the
    // same doc: they are valid links (resolve_links deliberately refuses to
    // insert them into `links`), but counting them as unresolved inflated
    // the broken-link metric by exactly the self-reference count.
    let self_references = resolved_links
        .iter()
        .filter(|(from, to)| from == to)
        .count();
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
                .filter_map(|(index, doc)| {
                    focus_score(&doc.path, query).map(|score| (index, score))
                })
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
        .prepare("SELECT d.path FROM doc_links dl JOIN docs d ON d.id = dl.doc_id")
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
mod tests;
