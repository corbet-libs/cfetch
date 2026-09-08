//! The recall index: SQLite + FTS5 over the brain's markdown, rings 0-4.
//!
//! The DB is a per-host DERIVED, DISPOSABLE cache of the shared tree â€” it
//! lives in the local state dir (SQLite WAL cannot live on NFS), is rebuilt
//! whenever the tree's (path, mtime, size) set changes, and is deleted and
//! recreated on any corruption. The git-tracked markdown stays the only source
//! of truth.
//!
//! Citations are content-addressed: `r<ring>-<prefix of the versioned exact
//! block-body hash>`. They survive reordering and unrelated edits; an edited
//! entry becomes a new citation by construction. The ring prefix makes the trust
//! level of a hit visible in the id itself.
//!
//! The FULL digest behind that prefix is the block's content address, and it
//! is what every derived artifact is keyed by â€” so vectors survive a rebuild
//! that recycles every rowid in the file, and an edit costs the embeddings of
//! exactly the blocks that changed.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use rusqlite::Connection;
use sha2::Digest as _;

use crate::config::{Precision, RingRules, VectorSpec};
use crate::resident::blank_private_checked;

/// Rings 5-6 are never indexed: staging and exhaust must not surface in
/// recall. A file declaring itself ring 5+ is skipped entirely.
const MAX_INDEXED_RING: u8 = 4;

pub struct Block {
    pub cite: String,
    pub path: String,
    pub ring: u8,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

#[derive(Debug)]
pub struct Hit {
    pub cite: String,
    pub path: String,
    pub ring: u8,
    pub start_line: usize,
    pub end_line: usize,
    pub snippet: String,
    /// Paths of suppressed duplicate copies of this logical block: same
    /// content hash AND same heading chain on a higher (or equal) ring â€”
    /// e.g. the native auto-memory mirror of a brain file. Identical short
    /// blocks under DIFFERENT sections are different statements, never
    /// mirrors. Empty for a block that exists in exactly one place.
    pub mirrors: Vec<String>,
    /// Enclosing heading chain ("H1 > H2"), the second half of the mirror
    /// dedup key. Internal â€” the snippet already displays it.
    pub(crate) chain: String,
    /// The block's whole body. Internal â€” `snippet` is the display form, and
    /// it is capped at 160 characters. A consumer that only REORDERS hits can
    /// live with that cap; one that DROPS them cannot, or it would suppress a
    /// block whose evidence sat past the cut.
    pub(crate) text: String,
}

/// Rewrites `sep` to `/` â€” the canonical separator of every brain-relative
/// doc path.
///
/// Doc paths are matched by PREFIX against `mind/secrets/`, `logs/`,
/// `projects/`, `knowledge/archive/` and `.git/`, are stored in the catalog
/// and are printed in citations. A platform whose separator is `\` produces
/// `mind\secrets\age.key` from the same file, which matches none of those
/// prefixes â€” the secrets boundary would open by pure string accident. On a
/// platform whose separator is already `/`, a backslash is an ordinary
/// filename character and is left untouched.
pub(crate) fn normalize_separators(rel: &str, sep: char) -> String {
    if sep == '/' { rel.to_string() } else { rel.replace(sep, "/") }
}

/// Canonical doc path for a brain-root-relative path, on any platform.
pub(crate) fn rel_doc_path(rel: &Path) -> String {
    normalize_separators(&rel.to_string_lossy(), std::path::MAIN_SEPARATOR)
}

/// Whether a doc path names a file in Claude Code's native auto-memory store
/// rather than the brain tree. The `native:` tag is the only thing that
/// separates the two stores downstream of the scan, and each store has its
/// own contract â€” a clamped ring, and its own link dialect.
fn is_native(doc_path: &str) -> bool {
    doc_path.starts_with("native:")
}

/// THE taxonomy entry point: the configured location default for a
/// brain-root-relative path. Frontmatter `ring: N` still overrides it at
/// scan time. Everything that needs a path's ring â€” the scan, the watcher,
/// ring-6 capture â€” comes through here, so the mapping lives in the config
/// and nowhere else.
pub fn default_ring(rel: &str, rules: &RingRules) -> u8 {
    rules.ring_for(rel)
}

/// Directory names whose markdown is installed or generated, never authored.
///
/// Built in rather than configurable, because a file under one of these is
/// machine output wherever the tree lives and nobody would choose to recall
/// it. `exclude_prefixes` could not express it anyway: those prefixes anchor
/// at the brain root, and a dependency tree appears at arbitrary depth â€” a
/// documentation vault built with a static-site generator is an npm project,
/// so one `npm install` inside it buries the operator's own notes under
/// thousands of package READMEs that are prose-shaped, plausible, and enough
/// to dominate FTS for ordinary words.
///
/// Deliberately NOT here: `build`, `dist`, `target`, `vendor`. Those are
/// credible topic names in a knowledge tree, and a built-in nobody can switch
/// off would make an operator's own notes silently unrecallable â€” a far worse
/// failure than indexing a generated copy of prose that mirror dedup already
/// collapses.
const GENERATED_DIRS: &[&str] = &["node_modules", "__pycache__"];

/// Whether any path component names a [`GENERATED_DIRS`] tree. Component-wise,
/// not substring: `knowledge/node_modules-migration.md` is a real note.
fn under_generated_dir(rel: &str) -> bool {
    rel.trim_end_matches('/').split('/').any(|c| GENERATED_DIRS.contains(&c))
}

/// Paths that must never enter the index: the compiled-in boundary (secrets,
/// logs, git internals) plus the operator's `exclude_prefixes` â€” by default
/// `projects/` (repo clones, owned by the code index) and
/// `knowledge/archive/` (retired knowledge, not recallable by accident) â€”
/// plus the generated-directory built-ins.
fn excluded(rel: &str, rules: &RingRules) -> bool {
    rules.excluded(rel) || under_generated_dir(rel)
}

/// Directory form of [`excluded`]: true when nothing under `rel` can ever be
/// indexed, so the serving watcher can skip the whole subtree. One predicate
/// serves both forms â€” the watch set is the index set by construction, never
/// by two hand-maintained lists agreeing.
pub(crate) fn excluded_dir(rel: &str, rules: &RingRules) -> bool {
    rules.excluded_dir(rel) || under_generated_dir(rel)
}

/// Secret-shaped file names are refused even outside mind/secrets/ â€” capture
/// the guard at the earliest point so every downstream store inherits it.
fn secret_shaped(rel: &str) -> bool {
    let base = rel.rsplit('/').next().unwrap_or(rel).to_ascii_lowercase();
    base.contains("secret")
        || base.contains("credential")
        || base.contains("password")
        || base.starts_with(".env")
        || base.ends_with(".pem")
        || base.ends_with(".key")
}

/// Longest line, in bytes, that can still be something a person wrote.
///
/// Measured rather than guessed: across a real brain of thousands of files
/// the longest hand-authored line was a wide memory-index table row at ~3 KB,
/// and nothing came within a factor of two of this cap. The shapes on the
/// other side â€” a minified bundle pasted into a page, a base64 data URI, a
/// one-line data dump saved as `.md` â€” sit orders of magnitude above it.
const MAX_PROSE_LINE: usize = 8 * 1024;

/// Whether a statement is generated rather than written.
///
/// Such a statement is not merely useless in recall, it is corrosive: FTS5
/// tokenizes it into thousands of terms, so it becomes a candidate for
/// almost any query while carrying no meaning, and it would claim an
/// embedding of its own. Bytes, not characters, because bytes are what the
/// tokenizer and the embedder are charged for.
///
/// One pass over a body already in memory â€” the walk that reads the file
/// pays nothing extra for this.
fn generated_blob(body: &str) -> bool {
    body.lines().any(|l| l.len() > MAX_PROSE_LINE)
}

/// Parses a leading `---` frontmatter for `ring: N`. Returns (ring override,
/// line count of the frontmatter block including fences).
///
/// FAIL CLOSED: a `ring:` key whose value does not parse cleanly yields 255
/// (= skip the file). A malformed declaration on quarantined content must
/// never fall back to an indexable default.
pub(crate) fn frontmatter_ring(text: &str) -> (Option<u8>, usize) {
    let mut lines = text.lines();
    // A UTF-8 BOM is legal UTF-8 and editors on Windows emit it by default.
    // It is not whitespace, so an unstripped BOM would make the first `---`
    // comparison fail and the whole frontmatter - including a `ring: 5`
    // quarantine marker - silently invisible: fail-open by encoding
    // accident. Strip it before the comparison.
    let first = lines.next().map(|l| l.trim_start_matches('\u{FEFF}').trim());
    if first != Some("---") {
        return (None, 0);
    }
    let mut ring = None;
    for (i, line) in lines.enumerate() {
        let t = line.trim();
        if t == "---" {
            return (ring, i + 2);
        }
        let lower = t.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("ring:") {
            // Only a TOP-LEVEL key counts: a nested `ring:` (under
            // `template:` or any other key, indented or inline) is a value
            // of that key, not a ring declaration. Matching any line
            // starting with `ring:` after a trim let
            // `template:\n  ring: 0` self-promote a knowledge file into the
            // policy band from what YAML says is a nested value.
            if !line.starts_with(' ') && !line.starts_with('\t') {
                let token = v.split_whitespace().next().unwrap_or("");
                ring = Some(token.parse::<u8>().unwrap_or(255));
            }
        }
    }
    // Unterminated frontmatter: treat as content.
    (None, 0)
}

/// Extracts `[[wikilink]]` targets: alias (`|â€¦`) and heading (`#â€¦`) parts
/// dropped, lowercased; slash-qualified targets (`[[hosts/zfs]]`) survive
/// whole. Fenced code blocks are skipped first (brain-lint parity): a
/// `[[link]]` inside a ``` or ~~~ fence is an example, not an edge. The brain
/// is an Obsidian vault â€” these are human-curated edges, the graph we trust
/// most.
pub fn wikilinks(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut fence: Option<(char, usize)> = None;
    for line in text.lines() {
        let t = line.trim_start();
        match fence {
            Some(open) => {
                if fence_closes(t, open) {
                    fence = None;
                }
                continue;
            }
            None => {
                if let Some(open) = fence_open(t) {
                    fence = Some(open);
                    continue;
                }
            }
        }
        let mut rest = line;
        while let Some(start) = rest.find("[[") {
            let after = &rest[start + 2..];
            let Some(end) = after.find("]]") else { break };
            let target = after[..end].split(['|', '#']).next().unwrap_or("").trim();
            if !target.is_empty() {
                out.push(target.to_ascii_lowercase());
            }
            rest = &after[end + 2..];
        }
    }
    out
}

/// Extracts `[label](target.md)` targets, in the same shape [`wikilinks`]
/// returns them: link title and `#anchor` dropped, `./` prefix stripped,
/// lowercased, fenced code skipped.
///
/// This is the NATIVE store's dialect, not the brain's. The brain is an
/// Obsidian vault whose edges are wikilinks and whose resolution keeps
/// brain-lint parity; Claude Code's memory store has no wikilinks at all â€”
/// its index file points at its entries with ordinary markdown links. Reading
/// one dialect per store is what keeps those pointers from being edges nobody
/// can see, without inventing edges in the vault.
///
/// Only `.md` targets are taken: one rule that keeps images and web links out
/// of the pointer set. Whatever survives it and still resolves to no indexed
/// doc yields no edge, exactly as an unresolvable wikilink does.
pub fn markdown_links(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut fence: Option<(char, usize)> = None;
    for line in text.lines() {
        let t = line.trim_start();
        match fence {
            Some(open) => {
                if fence_closes(t, open) {
                    fence = None;
                }
                continue;
            }
            None => {
                if let Some(open) = fence_open(t) {
                    fence = Some(open);
                    continue;
                }
            }
        }
        let mut rest = line;
        while let Some(start) = rest.find("](") {
            let after = &rest[start + 2..];
            let Some(end) = after.find(')') else { break };
            let target = &after[..end];
            rest = &after[end + 1..];
            // `[a](b.md "Title")`: the title is not part of the target.
            let target = target.split_whitespace().next().unwrap_or("");
            let target = target.split('#').next().unwrap_or("");
            let target = target.strip_prefix("./").unwrap_or(target);
            if target.ends_with(".md") {
                out.push(target.to_ascii_lowercase());
            }
        }
    }
    out
}

/// THE content address of a statement: the full SHA-256 hex of a versioned
/// domain followed by its exact segmented body bytes. Case, indentation and
/// internal line breaks are preserved, just as they are in embedding input.
/// The domain separates these keys from legacy normalized-body hashes, whose
/// vectors may have been derived from different raw text. Every derived
/// artifact uses this key, stable across hosts, rescans and reorderings.
///
/// ONE hashing site: [`cite_from_hash`] shows a truncated PREFIX of this same
/// digest, so a citation and the vector stored under its hash can never end
/// up describing different content.
pub fn content_hash(text: &str) -> String {
    let mut hash = sha2::Sha256::new();
    hash.update(b"cfetch-statement-body-v1\0");
    hash.update(text.as_bytes());
    crate::hashing::hex_lower(hash.finalize())
}

/// 40 hash bits: at ~20k blocks the birthday collision expectation is ~0.0002
/// â€” the 24-bit version measurably collided in the real corpus. The citation
/// TRUNCATES the content address; the full digest keys the artifacts.
const CITE_HASH_HEX: usize = 10;

/// Citation id of a block: its ring, then a prefix of its content address.
/// Takes the hash rather than the text so no caller ever hashes twice â€” the
/// citation and the block's derived artifacts come from one digest.
pub fn cite_from_hash(ring: u8, hash: &str) -> String {
    format!("r{ring}-{}", &hash[..CITE_HASH_HEX])
}

/// A fence opener: a run of at least three backticks or tildes (CommonMark
/// treats both characters as code fences). Returns (fence char, run length).
fn fence_open(t: &str) -> Option<(char, usize)> {
    let c = t.chars().next()?;
    if c != '`' && c != '~' {
        return None;
    }
    let run = t.chars().take_while(|&x| x == c).count();
    (run >= 3).then_some((c, run))
}

/// CommonMark closing rule: only a run of the SAME character at least as long
/// as the opener closes a fence, and a closing fence may be followed only by
/// spaces â€” ```` ```bash ```` INSIDE a fence is an info string on an opening
/// fence, not a closer. Without the trailing-text check, a note documenting
/// another fence's language tag closed the outer fence early, promoting
/// fenced wikilinks into curated graph edges.
fn fence_closes(t: &str, (ch, len): (char, usize)) -> bool {
    let run = t.chars().take_while(|&x| x == ch).count();
    if run < len {
        return false;
    }
    // Everything after the closing run must be whitespace (CommonMark 4.5).
    t.chars().skip(run).all(char::is_whitespace)
}

/// Setext heading underline: a line of only `=` (level 1) or only `-`
/// (level 2). Two characters minimum â€” a lone `-` in running text is far more
/// often a stray bullet than an underline.
fn setext_level(t: &str) -> Option<u8> {
    let t = t.trim();
    if t.len() >= 2 && t.chars().all(|c| c == '=') {
        Some(1)
    } else if t.len() >= 2 && t.chars().all(|c| c == '-') {
        Some(2)
    } else {
        None
    }
}

/// Continuation indentation: spaces or tabs both indent.
fn indented(line: &str) -> bool {
    line.starts_with(' ') || line.starts_with('\t')
}

/// Heading level and text of a block when the block IS a heading: ATX
/// (`## title`, closing hashes tolerated) or setext (text + `===`/`---`
/// underline as produced by [`segment`]).
fn heading_of(body: &str) -> Option<(u8, String)> {
    let mut it = body.lines();
    let first = it.next()?.trim_start();
    if first.starts_with('#') {
        let level = first.chars().take_while(|&c| c == '#').count();
        if (1..=6).contains(&level) && it.next().is_none() {
            let text = first[level..].trim().trim_end_matches('#').trim();
            return Some((level as u8, text.to_string()));
        }
        return None;
    }
    let second = it.next()?;
    if it.next().is_none()
        && let Some(level) = setext_level(second)
    {
        return Some((level, first.trim().to_string()));
    }
    None
}

/// Splits markdown into logical blocks: heading (ATX or setext underline),
/// list item (with indented continuation, including blank-separated indented
/// continuation paragraphs), table row, fenced code block (``` or ~~~), or
/// paragraph. Line numbers are 1-indexed into the ORIGINAL file (the caller
/// passes blanked text of equal line structure).
pub fn segment(text: &str, skip_lines: usize) -> Vec<(usize, usize, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut blocks = Vec::new();
    let mut i = skip_lines;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            i += 1;
            continue;
        }
        let start = i;
        if let Some(open) = fence_open(trimmed) {
            i += 1;
            while i < lines.len() && !fence_closes(lines[i].trim_start(), open) {
                i += 1;
            }
            i = (i + 1).min(lines.len());
        } else if trimmed.starts_with('#') {
            i += 1;
        } else if trimmed.starts_with('|') {
            i += 1; // one table row per block: rows are independent statements
        } else if is_list_item(trimmed) {
            i += 1;
            loop {
                // Tight continuation: indented, non-blank, not itself an item.
                while i < lines.len()
                    && !lines[i].trim_start().is_empty()
                    && indented(lines[i])
                    && !is_list_item(lines[i].trim_start())
                {
                    i += 1;
                }
                // Loose continuation: blank line(s) followed by an indented
                // non-item line â€” an indented continuation paragraph still
                // belongs to the item (numbered items especially are written
                // this way).
                let mut j = i;
                while j < lines.len() && lines[j].trim().is_empty() {
                    j += 1;
                }
                if j > i
                    && j < lines.len()
                    && indented(lines[j])
                    && !is_list_item(lines[j].trim_start())
                {
                    i = j + 1;
                } else {
                    break;
                }
            }
        } else if i + 1 < lines.len() && setext_level(lines[i + 1]).is_some() {
            i += 2; // setext heading: the text line plus its underline
        } else {
            i += 1;
            while i < lines.len() {
                let t = lines[i].trim_start();
                if t.is_empty()
                    || t.starts_with('#')
                    || t.starts_with('|')
                    || is_list_item(t)
                    || fence_open(t).is_some()
                {
                    break;
                }
                // The upcoming line underlines THIS line into a setext
                // heading: the heading starts its own block.
                if i + 1 < lines.len() && setext_level(lines[i + 1]).is_some() {
                    break;
                }
                i += 1;
            }
        }
        let body = lines[start..i].join("\n");
        if !body.trim().is_empty() {
            blocks.push((start + 1, i, body));
        }
    }
    blocks
}

fn is_list_item(t: &str) -> bool {
    let numbered = |sep: char| {
        t.split_once(sep).is_some_and(|(n, rest)| {
            !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()) && rest.starts_with(' ')
        })
    };
    t.starts_with("- ")
        || t.starts_with("* ")
        || t.starts_with("+ ")
        || numbered('.')
        || numbered(')')
}

fn db_path(state_dir: &Path) -> PathBuf {
    state_dir.join("index.db")
}

/// Whether a catalog database file exists at all — the cheap first half of
/// a diagnostic's liveness probe, so a read-only look can be reserved for
/// the case where there is something to read.
pub(crate) fn db_exists(state_dir: &Path) -> bool {
    db_path(state_dir).is_file()
}

/// READ-ONLY open for diagnostics: never creates the file, never repairs a
/// corrupt one (`open`'s delete-and-rebuild is the writer's recovery, and a
/// diagnostic must not be the thing that first writes an index). The schema
/// check rejects a corrupt or outdated catalog without attempting repairs.
pub fn open_read_only(state_dir: &Path) -> anyhow::Result<Connection> {
    let path = db_path(state_dir);
    let conn = Connection::open_with_flags(&path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| anyhow::anyhow!("open {} read-only: {e}", path.display()))?;
    ensure_current_schema(&conn)?;
    Ok(conn)
}

/// Bump whenever tables/columns/id formats change: an old DB with a new
/// binary is silently wrong (e.g. stale cite widths), and the cache is
/// disposable â€” mismatches are handled by delete-and-rebuild in `open()`.
const SCHEMA_VERSION: i64 = 8; // 8: versioned exact-body hashes for blocks, cites and vectors

fn ensure_current_schema(conn: &Connection) -> anyhow::Result<()> {
    let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    anyhow::ensure!(
        version == SCHEMA_VERSION,
        "index schema v{version} != v{SCHEMA_VERSION}; rebuild required"
    );
    Ok(())
}

fn open_at(path: &Path) -> anyhow::Result<Connection> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let conn = Connection::open(path)?;
    conn.busy_timeout(std::time::Duration::from_millis(1000))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    // The ON DELETE CASCADE clauses below (and in code.rs) are load-bearing:
    // enforce them explicitly instead of relying on the bundled build's
    // compile-time default.
    conn.pragma_update(None, "foreign_keys", true)?;
    let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version != SCHEMA_VERSION {
        let tables: i64 =
            conn.query_row("SELECT count(*) FROM sqlite_master WHERE type='table'", [], |r| r.get(0))?;
        if tables > 0 {
            anyhow::bail!("index schema v{version} != v{SCHEMA_VERSION}; rebuild required");
        }
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    }
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
         CREATE TABLE IF NOT EXISTS docs(
           id INTEGER PRIMARY KEY,
           path TEXT UNIQUE NOT NULL,
           ring INTEGER NOT NULL,
           mtime INTEGER NOT NULL,
           size INTEGER NOT NULL
         );
         CREATE TABLE IF NOT EXISTS blocks(
           id INTEGER PRIMARY KEY,
           cite TEXT NOT NULL,
           doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
           start_line INTEGER NOT NULL,
           end_line INTEGER NOT NULL,
           text TEXT NOT NULL,
           ctx TEXT NOT NULL DEFAULT '',
           chain TEXT NOT NULL DEFAULT '',
           hash TEXT NOT NULL DEFAULT ''
         );
         CREATE INDEX IF NOT EXISTS blocks_cite ON blocks(cite);
         CREATE INDEX IF NOT EXISTS blocks_hash ON blocks(hash);
         CREATE INDEX IF NOT EXISTS blocks_doc ON blocks(doc_id);
         CREATE TABLE IF NOT EXISTS links(
           from_doc INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
           to_doc INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE
         );
         CREATE INDEX IF NOT EXISTS links_from ON links(from_doc);
         CREATE INDEX IF NOT EXISTS links_to ON links(to_doc);
         CREATE TABLE IF NOT EXISTS doc_links(
           doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
           target TEXT NOT NULL
         );
         CREATE INDEX IF NOT EXISTS doc_links_doc ON doc_links(doc_id);
         CREATE TABLE IF NOT EXISTS skipped_docs(
           path TEXT PRIMARY KEY,
           mtime INTEGER NOT NULL,
           size INTEGER NOT NULL
         );
         CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts USING fts5(text, ctx, content='blocks', content_rowid='id');
         CREATE TABLE IF NOT EXISTS vectors(
           content_hash TEXT PRIMARY KEY,
           model TEXT NOT NULL,
           dim INTEGER NOT NULL,
           embedding BLOB NOT NULL
         );",
    )?;
    Ok(conn)
}

/// Opens the index; a corrupt file is deleted and recreated (derived cache â€”
/// the tree is the truth).
pub fn open(state_dir: &Path) -> anyhow::Result<Connection> {
    let path = db_path(state_dir);
    match open_at(&path) {
        Ok(c) => Ok(c),
        Err(_) => {
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(path.with_extension("db-wal"));
            let _ = std::fs::remove_file(path.with_extension("db-shm"));
            open_at(&path)
        }
    }
}

/// One indexable file: the doc path stored in the DB, where to read it, stat
/// info for staleness, and the ring to assume when no frontmatter overrides.
struct SourceFile {
    doc_path: String,
    abs: PathBuf,
    mtime: u64,
    size: u64,
    default_ring: u8,
}

fn stat_of(meta: &std::fs::Metadata) -> (u64, u64) {
    // Nanosecond precision: same-second same-size edits must flip staleness.
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    (mtime, meta.len())
}

/// Per-directory overlay that scopes a subtree out of cfetch WITHOUT hiding
/// it from git. Gitignore syntax, gitignore precedence, its own file name:
/// until this existed `.gitignore` was the only file-based control the
/// walkers read, so keeping generated markdown out of the brain index â€” or a
/// vendored subtree out of the code index â€” meant untracking it first.
pub const IGNORE_FILE: &str = ".cfetchignore";

/// THE walker every enumeration of a tree is built from: the brain scan, the
/// code scan, and the serving watcher alike.
///
/// The ignore policy lives here and nowhere else on purpose. The watch set
/// must stay exactly the index set â€” a directory the watcher skips is a
/// directory whose writes never wake the daemon, which is silent staleness,
/// the one failure serving mode refuses â€” and an overlay honored by one
/// walker but not another would reintroduce that drift one `.cfetchignore`
/// at a time. Callers add their own traversal concerns (parallelism,
/// `filter_entry`) on top of what is returned.
pub fn tree_walker(root: &Path) -> ignore::WalkBuilder {
    let mut builder = ignore::WalkBuilder::new(root);
    builder
        .hidden(true)
        .git_ignore(true)
        // Highest precedence of every ignore source, and read whether or not
        // the tree is a git checkout â€” a brain that is not a repo can be
        // scoped too.
        .add_custom_ignore_filename(IGNORE_FILE)
        .follow_links(false);
    builder
}

/// A managed nixcards checkout is knowledge even when the outer brain keeps
/// the nested repository out of its own Git history. Walk it explicitly so
/// `.gitignore` remains a version-control boundary rather than silently
/// becoming a trust/index boundary. Configured cfetch exclusions still apply
/// after the walk, and hidden `.git` internals never enter it.
pub fn managed_cards_walker(root: &Path) -> Option<ignore::WalkBuilder> {
    let cards = root.join("knowledge/cards");
    if !cards.join(".git").exists() || !cards.join("catalog.json").is_file() {
        return None;
    }
    let mut builder = ignore::WalkBuilder::new(cards);
    builder
        .hidden(true)
        .parents(false)
        .git_global(false)
        .git_ignore(false)
        .git_exclude(false)
        .add_custom_ignore_filename(IGNORE_FILE)
        .follow_links(false);
    let _ = builder.add_ignore(root.join(IGNORE_FILE));
    Some(builder)
}

/// Whether the indexer would actually READ this brain-relative path:
/// markdown, past the exclusion boundary, not secret-shaped. The scan and
/// the cost census below both come through here, so "what would be indexed"
/// has one answer instead of two lists that agree until they don't.
pub(crate) fn indexable_doc(rel: &str, rules: &RingRules) -> bool {
    rel.ends_with(".md") && !excluded(rel, rules) && !secret_shaped(rel)
}

/// A directory holding at least 9/10 of everything the indexer reads IS the
/// index, not a subtree of it: excluding it would not scope cfetch, it would
/// switch it off. Such a directory is never offered as a candidate â€” on a
/// tree whose whole corpus sits under one `knowledge/`, the only honest
/// answer is that there is nothing to scope out.
const WHOLE_TREE_NUM: usize = 9;
const WHOLE_TREE_DEN: usize = 10;

/// Where the index's cost actually sits: how many files the INDEXER would
/// read under each directory of the tree.
///
/// The measure is deliberately neither bytes on disk nor directory size. A
/// 40 GiB directory of build artifacts costs the index nothing â€” the walker
/// passes over every file in it â€” while a 3 MiB directory of generated
/// markdown costs a doc, a watch and a block set per file, and is what makes
/// a tree outgrow the watch budget. Advice derived from `du` would name the
/// first and never the second.
#[derive(Debug, Default)]
pub struct DocCensus {
    total: usize,
    /// Directory prefix (no trailing slash) -> indexable docs beneath it.
    /// Keyed by directory rather than by file, so a tree of any size costs a
    /// map the size of its directory count.
    by_prefix: std::collections::HashMap<String, usize>,
}

impl DocCensus {
    /// Counts one indexable doc against every directory that contains it.
    pub fn record(&mut self, rel: &str) {
        self.total += 1;
        for (i, _) in rel.match_indices('/') {
            *self.by_prefix.entry(rel[..i].to_string()).or_default() += 1;
        }
    }

    /// Everything the indexer would read, tree-wide.
    pub fn total(&self) -> usize {
        self.total
    }

    /// The subtrees worth scoping out: DISJOINT directories, each holding at
    /// least `min_share` of what the indexer reads, heaviest first.
    ///
    /// Depth decides whether an answer is useful at all. "`knowledge/` holds
    /// 60%" is not advice â€” nobody can exclude their knowledge tree â€” while
    /// "`knowledge/archive/2019/` holds 55% of everything" is something to
    /// act on. So the deepest directory that clears the bar wins and its
    /// ancestors are dropped, and a directory that already IS the index (see
    /// [`WHOLE_TREE_NUM`]) is never named at all. Disjointness is what makes
    /// the numbers readable: no doc is counted in two candidates.
    pub fn concentrations(&self, min_share: f64) -> Vec<(String, usize)> {
        let threshold = ((self.total as f64) * min_share).ceil().max(1.0) as usize;
        // Deepest first, so a nested directory claims its cost before any
        // ancestor is considered for the same docs.
        let mut candidates: Vec<(&String, usize)> =
            self.by_prefix.iter().map(|(p, &n)| (p, n)).collect();
        candidates.sort_by(|a, b| depth_of(b.0).cmp(&depth_of(a.0)).then_with(|| a.0.cmp(b.0)));
        let mut claimed: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        let mut out: Vec<(String, usize)> = Vec::new();
        for (prefix, count) in candidates {
            let already_named = claimed.get(prefix.as_str()).copied().unwrap_or(0);
            if already_named > 0
                || count < threshold
                || count * WHOLE_TREE_DEN >= self.total * WHOLE_TREE_NUM
            {
                continue;
            }
            out.push((prefix.clone(), count));
            for (i, _) in prefix.match_indices('/') {
                *claimed.entry(&prefix[..i]).or_default() += count;
            }
        }
        // Count first, path second: the same tree must always yield the same
        // advice, and a HashMap hands its entries back in no order at all.
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }
}

/// Directory nesting depth of a `/`-separated prefix; `knowledge` is 0.
fn depth_of(prefix: &str) -> usize {
    prefix.matches('/').count()
}

/// Brain tree + (optionally) Claude Code's native auto-memory stores.
/// Native memory (`<native_root>/<project-slug>/memory/*.md`) is indexed as
/// ring 2 with doc paths `native:<slug>/<file>` â€” cfetch reads and surfaces
/// the native store, it never writes to it.
fn collect_files(
    brain_root: &Path,
    native_root: Option<&Path>,
    rules: &RingRules,
) -> Vec<SourceFile> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut walkers = vec![tree_walker(brain_root).build()];
    if let Some(builder) = managed_cards_walker(brain_root) {
        walkers.push(builder.build());
    }
    for walker in walkers {
        for entry in walker.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_file() || !seen.insert(entry.path().to_path_buf()) {
                continue;
            }
            let Ok(rel) = entry.path().strip_prefix(brain_root) else { continue };
            let rel = rel_doc_path(rel);
            if !indexable_doc(&rel, rules) {
                continue;
            }
            let (mtime, size) = stat_of(&meta);
            out.push(SourceFile {
                doc_path: rel.clone(),
                abs: entry.path().to_path_buf(),
                mtime,
                size,
                default_ring: default_ring(&rel, rules),
            });
        }
    }
    if let Some(projects) = native_root.and_then(|nr| std::fs::read_dir(nr).ok()) {
        for project in projects.flatten() {
            let slug = project.file_name().to_string_lossy().to_string();
            let mem_dir = project.path().join("memory");
            let Ok(files) = std::fs::read_dir(&mem_dir) else { continue };
            for f in files.flatten() {
                let Ok(meta) = f.metadata() else { continue };
                let name = f.file_name().to_string_lossy().to_string();
                if !meta.is_file() || !name.ends_with(".md") || secret_shaped(&name) {
                    continue;
                }
                let (mtime, size) = stat_of(&meta);
                out.push(SourceFile {
                    doc_path: format!("native:{slug}/{name}"),
                    abs: f.path(),
                    mtime,
                    size,
                    default_ring: 2,
                });
            }
        }
    }
    out.sort_by(|a, b| a.doc_path.cmp(&b.doc_path));
    out
}

/// One value answering "does this index describe these sources": sha256 over
/// the sorted (doc_path, mtime, size) list â€” INCLUDING files the scan later
/// skips by ring, so a ring-frontmatter edit or a skipped file's change flips
/// staleness like any other.
fn source_fingerprint(files: &[SourceFile]) -> String {
    let mut hasher = sha2::Sha256::new();
    for f in files {
        hasher.update(f.doc_path.as_bytes());
        hasher.update([0u8]);
        hasher.update(f.mtime.to_le_bytes());
        hasher.update(f.size.to_le_bytes());
        hasher.update([0xffu8]);
    }
    crate::hashing::hex_lower(hasher.finalize())
}

/// The stat fingerprint of the tree AS IT IS NOW â€” one stat walk, no file
/// bodies read. This is the value the 60s backstop compares against, and the
/// coverage token the serving daemon's unordered drain barrier takes at query
/// entry (see `serve::BarrierMode`).
pub fn tree_fingerprint(
    brain_root: &Path,
    native_root: Option<&Path>,
    rules: &RingRules,
) -> String {
    source_fingerprint(&collect_files(brain_root, native_root, rules))
}

/// The fingerprint of the tree state the COMMITTED catalog describes, written
/// inside each scan's transaction. `None` = never scanned.
pub fn stored_fingerprint(conn: &Connection) -> Option<String> {
    conn.query_row("SELECT value FROM meta WHERE key='source_fingerprint'", [], |r| r.get(0)).ok()
}

/// One stat walk answering both questions the backstop asks: is the catalog
/// stale, and what does the tree look like right now. A caller that needs the
/// fingerprint too must never pay for a second walk.
pub fn staleness(
    conn: &Connection,
    brain_root: &Path,
    native_root: Option<&Path>,
    rules: &RingRules,
) -> anyhow::Result<(bool, String)> {
    let current = tree_fingerprint(brain_root, native_root, rules);
    let root_meta: Option<String> = conn
        .query_row("SELECT value FROM meta WHERE key='brain_root'", [], |r| r.get(0))
        .ok();
    if root_meta.as_deref() != Some(brain_root.to_string_lossy().as_ref()) {
        return Ok((true, current));
    }
    let stale = stored_fingerprint(conn).as_deref() != Some(current.as_str());
    Ok((stale, current))
}

/// Cheap staleness decision: stat-only fingerprint comparison â€” no file
/// bodies are read.
pub fn stale(
    conn: &Connection,
    brain_root: &Path,
    native_root: Option<&Path>,
    rules: &RingRules,
) -> anyhow::Result<bool> {
    Ok(staleness(conn, brain_root, native_root, rules)?.0)
}

pub struct ScanReport {
    pub docs: usize,
    pub blocks: usize,
    pub skipped_high_ring: usize,
    /// Doc paths of the files skipped as ring 5+ â€” including fail-closed
    /// unparseable ring frontmatter â€” surfaced so a file quarantined by
    /// accident is visible instead of silently absent from recall.
    pub skipped: Vec<String>,
    /// Doc paths that could not be READ (invalid UTF-8, or a Windows sharing
    /// violation). A distinct reason from `skipped`: an unreadable file is
    /// not quarantined by policy, and reporting it as ring 5+ sends the
    /// operator looking for a frontmatter that is not the problem.
    pub unreadable: Vec<String>,
    /// Doc paths where an unclosed `<private>` blanked to end of file â€”
    /// fail-closed by design, but a tail leaving the index silently has
    /// cost real trees most of a file to a mere mention of the tag.
    pub private_swallowed: Vec<String>,
    /// Statements dropped as generated blobs (a line over [`MAX_PROSE_LINE`]
    /// bytes). The rest of their file indexes normally, so the drop is
    /// invisible unless counted.
    pub blob_dropped: usize,
    /// Catalog generation this scan committed (see [`generation`]).
    pub generation: u64,
}

/// Joined heading texts of a chain, empty texts (blanked private headings)
/// skipped.
fn chain_text(chain: &[(u8, String)]) -> String {
    chain
        .iter()
        .map(|(_, t)| t.as_str())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" > ")
}

/// Inserts one source file's rows â€” doc, blocks (with heading-chain context),
/// FTS rows, wikilink targets â€” or records it in `skipped_docs` when its ring
/// is 5+. Generated statements ([`generated_blob`]) are dropped individually
/// while the rest of their file is indexed normally. Shared by the full scan
/// and the incremental rescan so both derive byte-identical catalogs from the
/// same tree.
fn insert_doc(
    tx: &rusqlite::Transaction<'_>,
    src: &SourceFile,
    raw: &str,
    report: &mut ScanReport,
) -> anyhow::Result<()> {
    let (fm_ring, fm_lines) = frontmatter_ring(raw);
    let mut ring = fm_ring.unwrap_or(src.default_ring);
    let native = is_native(&src.doc_path);
    // The native store's contract ring is 2: honor demotion to 5+ (skip),
    // never self-promotion into the resident/policy rings.
    if native {
        ring = ring.max(2);
    }
    // Quarantine locations (ring 5+: staging, logs) are decided by WHERE the
    // file lives, not by what it claims about itself: a candidate declaring
    // `ring: 2` in its frontmatter stays unindexed, exactly as one whose
    // frontmatter was stripped or mangled â€” the taxonomy's "cannot be edited
    // away file by file" promise has to hold against a well-formed lie too.
    // Promotion BELOW the location ring stays available everywhere else: a
    // knowledge file declaring `ring: 1` is the documented, intended override.
    if src.default_ring > MAX_INDEXED_RING {
        ring = ring.max(src.default_ring);
    }
    if ring > MAX_INDEXED_RING {
        report.skipped_high_ring += 1;
        report.skipped.push(src.doc_path.clone());
        tx.execute(
            "INSERT INTO skipped_docs(path, mtime, size) VALUES(?1, ?2, ?3)
             ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size",
            rusqlite::params![src.doc_path, src.mtime as i64, src.size as i64],
        )?;
        return Ok(());
    }
    // Blank (not strip) private regions so line numbers stay accurate. An
    // unclosed tag blanks to end of file fail-closed â€” that is the design â€”
    // but the file is named in the report so the loss is a line of output,
    // not a silent absence.
    let (blanked, private_outcome) = blank_private_checked(raw);
    if private_outcome.unbalanced {
        report.private_swallowed.push(src.doc_path.clone());
    }
    tx.execute(
        "INSERT INTO docs(path, ring, mtime, size) VALUES(?1, ?2, ?3, ?4)",
        rusqlite::params![src.doc_path, ring, src.mtime as i64, src.size as i64],
    )?;
    let doc_id = tx.last_insert_rowid();
    let targets = if native { markdown_links(&blanked) } else { wikilinks(&blanked) };
    for target in targets {
        tx.execute(
            "INSERT INTO doc_links(doc_id, target) VALUES(?1, ?2)",
            rusqlite::params![doc_id, target],
        )?;
    }
    // Citation context: every block carries its enclosing heading chain
    // (H1 > H2 > â€¦); a heading block carries its ANCESTOR chain (a bare
    // heading has no other context); a table row additionally carries its
    // table's header row. `chain` is the dedup key part, `ctx` the searchable
    // and displayed form.
    let mut chain: Vec<(u8, String)> = Vec::new();
    let mut table_header: Option<String> = None;
    let mut last_row_end = 0usize;
    for (start, end, body) in segment(&blanked, fm_lines) {
        let chain_key;
        let ctx;
        if let Some((level, text)) = heading_of(&body) {
            chain.retain(|(l, _)| *l < level);
            chain_key = chain_text(&chain);
            ctx = chain_key.clone();
            chain.push((level, text));
            table_header = None;
        } else if body.trim_start().starts_with('|') {
            chain_key = chain_text(&chain);
            // A gap in line numbers separates two adjacent tables.
            if start > last_row_end + 1 {
                table_header = None;
            }
            last_row_end = end;
            match &table_header {
                Some(header) => {
                    ctx = if chain_key.is_empty() {
                        header.clone()
                    } else {
                        format!("{chain_key} > {header}")
                    };
                }
                None => {
                    // This row IS the table's header.
                    table_header = Some(snippet_of(&body));
                    ctx = chain_key.clone();
                }
            }
        } else {
            table_header = None;
            chain_key = chain_text(&chain);
            ctx = chain_key.clone();
        }
        // Refused AFTER the chain and table bookkeeping above, never before:
        // every neighboring statement then keeps byte-identical context, so
        // a blob appearing beside a paragraph can never move that
        // paragraph's citation. Refusing the statement and not the file also
        // keeps a knowledge page that embeds one data URI fully recallable.
        if generated_blob(&body) {
            report.blob_dropped += 1;
            continue;
        }
        let hash = content_hash(&body);
        let cite = cite_from_hash(ring, &hash);
        tx.prepare_cached(
            "INSERT INTO blocks(cite, doc_id, start_line, end_line, text, ctx, chain, hash)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )?
        .execute(rusqlite::params![cite, doc_id, start as i64, end as i64, body, ctx, chain_key, hash])?;
        let block_id = tx.last_insert_rowid();
        tx.prepare_cached("INSERT INTO blocks_fts(rowid, text, ctx) VALUES(?1, ?2, ?3)")?
            .execute(rusqlite::params![block_id, body, ctx])?;
        report.blocks += 1;
    }
    report.docs += 1;
    Ok(())
}

/// Rebuilds the `links` table from `doc_links` + the doc registry. Every doc
/// is registered under ALL its path suffixes (stem, parent/stem, â€¦, full
/// path; `.md` stripped, lowercased) and a target resolves only when its key
/// is unambiguous â€” a slash-qualified target resolves a stem collision
/// (brain-lint parity). A native-store link is tried against its own
/// project's directory first, because a markdown link is a relative path.
/// A pure function of (docs, doc_links): full and incremental scans converge
/// on identical edges.
fn resolve_links(tx: &rusqlite::Transaction<'_>) -> anyhow::Result<()> {
    tx.execute("DELETE FROM links", [])?;
    let mut by_suffix: std::collections::HashMap<String, Option<i64>> =
        std::collections::HashMap::new();
    {
        let mut stmt = tx.prepare("SELECT id, path FROM docs")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        for (id, path) in rows.filter_map(Result::ok) {
            let stemless =
                path.strip_suffix(".md").unwrap_or(&path).to_ascii_lowercase();
            let parts: Vec<&str> = stemless.split('/').collect();
            for k in 1..=parts.len() {
                let key = parts[parts.len() - k..].join("/");
                by_suffix
                    .entry(key)
                    .and_modify(|e| *e = None) // ambiguous
                    .or_insert(Some(id));
            }
        }
    }
    let pending: Vec<(i64, String, String)> = {
        let mut stmt = tx.prepare(
            "SELECT dl.doc_id, d.path, dl.target FROM doc_links dl JOIN docs d ON d.id = dl.doc_id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
        })?;
        rows.filter_map(Result::ok).collect()
    };
    for (from_doc, from_path, target) in pending {
        let key = target.strip_suffix(".md").unwrap_or(&target);
        // A markdown link is a RELATIVE path by definition: `[x](y.md)` in
        // one project's memory index means THAT project's y.md. Every project
        // in the native store names the same generic files, so the vault-wide
        // stem key is ambiguous exactly where the pointers matter, and the
        // whole index would resolve to nothing on any host with more than one
        // project. The vault keeps stem resolution untouched: there a name is
        // global, and brain-lint reads the same edges from the same rule.
        let sibling = is_native(&from_path)
            .then(|| from_path.to_ascii_lowercase())
            .and_then(|p| p.rsplit_once('/').map(|(dir, _)| format!("{dir}/{key}")));
        let resolved = sibling
            .and_then(|k| by_suffix.get(&k).copied().flatten())
            .or_else(|| by_suffix.get(key).copied().flatten());
        if let Some(to_doc) = resolved
            && to_doc != from_doc
        {
            tx.execute(
                "INSERT INTO links(from_doc, to_doc) VALUES(?1, ?2)",
                rusqlite::params![from_doc, to_doc],
            )?;
        }
    }
    Ok(())
}

/// Bumps the persisted catalog generation INSIDE the given transaction: a
/// reader can never observe a new catalog under an old generation or vice
/// versa.
fn bump_generation(tx: &rusqlite::Transaction<'_>) -> anyhow::Result<u64> {
    let generation: u64 = tx
        .query_row("SELECT value FROM meta WHERE key='generation'", [], |r| r.get::<_, String>(0))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
        + 1;
    tx.execute(
        "INSERT INTO meta(key, value) VALUES('generation', ?1)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [generation.to_string()],
    )?;
    Ok(generation)
}

fn set_fingerprint(tx: &rusqlite::Transaction<'_>, fingerprint: &str) -> anyhow::Result<()> {
    tx.execute(
        "INSERT INTO meta(key, value) VALUES('source_fingerprint', ?1)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [fingerprint],
    )?;
    Ok(())
}

/// Full rebuild inside one transaction. The corpus is small markdown; a
/// rebuild is cheap, and the incremental path ([`rescan_changed`]) exists for
/// the serving daemon's event batches.
pub fn scan(
    conn: &mut Connection,
    brain_root: &Path,
    native_root: Option<&Path>,
    rules: &RingRules,
) -> anyhow::Result<ScanReport> {
    let files = collect_files(brain_root, native_root, rules);
    let fingerprint = source_fingerprint(&files);
    // Read every body BEFORE the write transaction: the tree may be NFS, and
    // holding SQLite's writer lock across slow I/O starves concurrent readers
    // into SQLITE_BUSY failures. A read that fails (invalid UTF-8, or a
    // Windows sharing violation from an editor or antivirus) must be VISIBLE,
    // not invisible: the file stays in the fingerprint, so stale() reports
    // fresh forever and nothing would ever retry it. Report it as skipped.
    let mut report = ScanReport {
        docs: 0,
        blocks: 0,
        skipped_high_ring: 0,
        skipped: Vec::new(),
        unreadable: Vec::new(),
        private_swallowed: Vec::new(),
        blob_dropped: 0,
        generation: 0,
    };
    let bodies: Vec<(SourceFile, String)> = files
        .into_iter()
        .filter_map(|src| match std::fs::read_to_string(&src.abs) {
            Ok(raw) => Some((src, raw)),
            Err(_) => {
                report.unreadable.push(src.doc_path.clone());
                None
            }
        })
        .collect();
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    // Vectors deliberately SURVIVE the rebuild: they are keyed by content
    // hash, not by a block rowid, so a rebuilt catalog re-joins every vector
    // whose text is still in the tree. They used to be dropped here because
    // rowids are recycled â€” which made one markdown edit cost 100% of the
    // embeddings. `prune_vectors` below drops exactly the hashes that left.
    tx.execute_batch(
        "DELETE FROM doc_links; DELETE FROM links;
         DELETE FROM blocks; DELETE FROM docs; DELETE FROM skipped_docs;
         INSERT INTO blocks_fts(blocks_fts) VALUES('delete-all');",
    )?;
    tx.execute(
        "INSERT INTO meta(key, value) VALUES('brain_root', ?1)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [brain_root.to_string_lossy().as_ref()],
    )?;
    set_fingerprint(&tx, &fingerprint)?;
    let generation = bump_generation(&tx)?;
    report.generation = generation;
    for (src, raw) in bodies {
        insert_doc(&tx, &src, &raw, &mut report)?;
    }
    resolve_links(&tx)?;
    prune_vectors(&tx)?;
    tx.commit()?;
    Ok(report)
}

/// Removes one doc's rows everywhere: FTS rows (external-content FTS5 must be
/// told each removed row's old values), blocks, the doc row (links and
/// doc_links cascade), and any `skipped_docs` entry. Vectors are NOT touched
/// here â€” they belong to content, not to a doc, and the same text in another
/// file keeps them alive; [`prune_vectors`] settles that at the end of a scan.
fn delete_doc(tx: &rusqlite::Transaction<'_>, path: &str) -> anyhow::Result<()> {
    tx.execute("DELETE FROM skipped_docs WHERE path=?1", [path])?;
    let doc_id: Option<i64> =
        tx.query_row("SELECT id FROM docs WHERE path=?1", [path], |r| r.get(0)).ok();
    let Some(doc_id) = doc_id else { return Ok(()) };
    {
        let mut stmt = tx.prepare("SELECT id, text, ctx FROM blocks WHERE doc_id=?1")?;
        let rows = stmt.query_map([doc_id], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
        })?;
        let blocks: Vec<(i64, String, String)> = rows.filter_map(Result::ok).collect();
        for (id, text, ctx) in blocks {
            tx.execute(
                "INSERT INTO blocks_fts(blocks_fts, rowid, text, ctx) VALUES('delete', ?1, ?2, ?3)",
                rusqlite::params![id, text, ctx],
            )?;
        }
    }
    tx.execute("DELETE FROM blocks WHERE doc_id=?1", [doc_id])?;
    tx.execute("DELETE FROM docs WHERE id=?1", [doc_id])?;
    Ok(())
}

/// Changed+vanished files an incremental rescan will handle; larger diffs
/// fall back to the full scan, whose bulk delete-all path is faster there.
const INCREMENTAL_MAX_CHANGES: usize = 32;

/// Incremental catalog update: re-reads ONLY the files whose (mtime, size)
/// changed â€” each changed doc's rows are deleted and reinserted, vanished
/// docs are pruned, links re-resolved â€” inside one transaction that also
/// advances the generation, exactly like [`scan`]. Returns `None` when the
/// incremental step has no valid basis (never scanned, different brain root)
/// or the diff exceeds [`INCREMENTAL_MAX_CHANGES`]; the caller then runs the
/// full scan. A no-op diff (fingerprint already current) commits nothing and
/// reports the existing generation.
pub fn rescan_changed(
    conn: &mut Connection,
    brain_root: &Path,
    native_root: Option<&Path>,
    rules: &RingRules,
) -> anyhow::Result<Option<ScanReport>> {
    let root_meta: Option<String> = conn
        .query_row("SELECT value FROM meta WHERE key='brain_root'", [], |r| r.get(0))
        .ok();
    if root_meta.as_deref() != Some(brain_root.to_string_lossy().as_ref())
        || generation(conn) == 0
    {
        return Ok(None);
    }
    let files = collect_files(brain_root, native_root, rules);
    let fingerprint = source_fingerprint(&files);
    let stored: Option<String> = conn
        .query_row("SELECT value FROM meta WHERE key='source_fingerprint'", [], |r| r.get(0))
        .ok();
    if stored.as_deref() == Some(fingerprint.as_str()) {
        // The committed catalog already describes this tree.
        return Ok(Some(ScanReport {
            docs: 0,
            blocks: 0,
            skipped_high_ring: 0,
            skipped: Vec::new(),
            unreadable: Vec::new(),
            private_swallowed: Vec::new(),
            blob_dropped: 0,
            generation: generation(conn),
        }));
    }
    // Stat diff against the stored per-file stats (indexed docs + skipped
    // ring-5+ files) â€” the same (mtime, size) basis the fingerprint uses.
    let mut stored_stats: std::collections::HashMap<String, (i64, i64)> =
        std::collections::HashMap::new();
    for table in ["docs", "skipped_docs"] {
        let mut stmt = conn.prepare(&format!("SELECT path, mtime, size FROM {table}"))?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, (r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)))
        })?;
        for (path, stat) in rows.filter_map(Result::ok) {
            stored_stats.insert(path, stat);
        }
    }
    let changed: Vec<&SourceFile> = files
        .iter()
        .filter(|f| stored_stats.get(&f.doc_path) != Some(&(f.mtime as i64, f.size as i64)))
        .collect();
    let current: std::collections::HashSet<&str> =
        files.iter().map(|f| f.doc_path.as_str()).collect();
    let vanished: Vec<String> = stored_stats
        .keys()
        .filter(|p| !current.contains(p.as_str()))
        .cloned()
        .collect();
    if changed.len() + vanished.len() > INCREMENTAL_MAX_CHANGES {
        return Ok(None);
    }
    // Changed bodies read BEFORE the write transaction (same NFS rationale
    // as scan). A file that vanished between stat and read is treated as
    // deleted; the fingerprint diff surfaces it again on the next pass.
    let bodies: Vec<(&SourceFile, Option<String>)> = changed
        .iter()
        .map(|f| (*f, std::fs::read_to_string(&f.abs).ok()))
        .collect();
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    for path in &vanished {
        delete_doc(&tx, path)?;
    }
    let mut unreadable: Vec<&str> = Vec::new();
    for (src, raw) in &bodies {
        // A read that failed mid-rescan (transient lock, encoding) must NOT
        // delete the existing rows: the file still exists, its last good
        // content is a better answer than nothing, and the next mtime change
        // retries the read. Deleting here is how a one-second editor lock
        // turned into a silently vanished document.
        if raw.is_none() {
            unreadable.push(&src.doc_path);
            continue;
        }
        delete_doc(&tx, &src.doc_path)?;
    }
    set_fingerprint(&tx, &fingerprint)?;
    let generation = bump_generation(&tx)?;
    let mut report = ScanReport {
        docs: 0,
        blocks: 0,
        skipped_high_ring: 0,
        skipped: Vec::new(),
        unreadable: Vec::new(),
        private_swallowed: Vec::new(),
        blob_dropped: 0,
        generation,
    };
    for (src, raw) in &bodies {
        if let Some(raw) = raw {
            insert_doc(&tx, src, raw, &mut report)?;
        }
    }
    for path in unreadable {
        report.unreadable.push(path.to_string());
    }
    resolve_links(&tx)?;
    prune_vectors(&tx)?;
    tx.commit()?;
    Ok(Some(report))
}

/// Current catalog generation: the count of committed scans of this index.
/// Monotonic and persisted in meta; incremented INSIDE each scan's
/// transaction, so any reader always sees a (catalog, generation) pair from
/// one commit. 0 = never scanned.
pub fn generation(conn: &Connection) -> u64 {
    conn.query_row("SELECT value FROM meta WHERE key='generation'", [], |r| r.get::<_, String>(0))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Deterministic digest of the catalog: sha256 over the sorted
/// (cite, path, ring) rows. Two catalogs built from the same tree â€” by ANY
/// derivation path (fresh scan, event-driven rebuild, post-crash backstop) â€”
/// must produce the same checksum: the coherence invariant's cross-holder
/// verification value. Generation is deliberately NOT part of the digest.
pub fn catalog_checksum(conn: &Connection) -> anyhow::Result<String> {
    catalog_checksum_matching(conn, |_| true)
}

/// Catalog checksum restricted by path. Remote slice holders use this rather
/// than learning a digest over documents outside their grant.
pub fn catalog_checksum_matching(
    conn: &Connection,
    include: impl Fn(&str) -> bool,
) -> anyhow::Result<String> {
    let mut stmt = conn.prepare(
        "SELECT b.cite, d.path, d.ring FROM blocks b JOIN docs d ON d.id = b.doc_id
         ORDER BY b.cite, d.path, d.ring",
    )?;
    let rows =
        stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?)))?;
    let mut hasher = sha2::Sha256::new();
    for (cite, path, ring) in rows.filter_map(Result::ok) {
        if !include(&path) {
            continue;
        }
        hasher.update(cite.as_bytes());
        hasher.update([0u8]);
        hasher.update(path.as_bytes());
        hasher.update([0u8]);
        hasher.update(ring.to_le_bytes());
        hasher.update([0xffu8]);
    }
    Ok(crate::hashing::hex_lower(hasher.finalize()))
}

/// Read-only open for serving-side query threads: never contends for the
/// write lock, never creates or migrates schema. Fails when no current-schema
/// index exists yet â€” the caller reports that instead of racing the builder.
pub fn open_ro(state_dir: &Path) -> anyhow::Result<Connection> {
    use rusqlite::OpenFlags;
    let conn = Connection::open_with_flags(
        db_path(state_dir),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.busy_timeout(std::time::Duration::from_millis(1000))?;
    ensure_current_schema(&conn)?;
    Ok(conn)
}

/// Docs linked (either direction, human-curated wikilinks) to the docs of the
/// given citation paths â€” the deterministic 1-hop graph expansion of a recall
/// result. Returns (path, ring) sorted by ring then path, deduped.
pub fn linked_docs(conn: &Connection, hit_paths: &[String], limit: usize) -> anyhow::Result<Vec<(String, u8)>> {
    let mut out: Vec<(String, u8)> = Vec::new();
    for path in hit_paths {
        let mut stmt = conn.prepare(
            "SELECT d2.path, d2.ring FROM docs d1
             JOIN links l ON l.from_doc = d1.id JOIN docs d2 ON d2.id = l.to_doc
             WHERE d1.path = ?1
             UNION
             SELECT d2.path, d2.ring FROM docs d1
             JOIN links l ON l.to_doc = d1.id JOIN docs d2 ON d2.id = l.from_doc
             WHERE d1.path = ?1",
        )?;
        let rows = stmt.query_map([path], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as u8)))?;
        for row in rows.filter_map(Result::ok) {
            if !hit_paths.contains(&row.0) && !out.contains(&row) {
                out.push(row);
            }
        }
    }
    out.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    out.truncate(limit);
    Ok(out)
}

/// FTS5 query string: each term becomes a quoted prefix token, OR-joined â€”
/// recall-heavy on purpose (precision gates come at the consumer).
fn fts_query(user_query: &str) -> String {
    // Ordered de-dup: the same term twice in one OR would double-count it
    // in bm25 for no information gain.
    let mut terms: Vec<String> = Vec::new();
    let push = |terms: &mut Vec<String>, term: String| {
        if !term.is_empty() && !terms.contains(&term) {
            terms.push(term);
        }
    };
    user_query
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|t| !t.is_empty())
        // The FTS5 tokenizer splits on hyphens, so a hyphenated term inside
        // one quoted token becomes an ADJACENCY phrase query. That phrase is
        // also the only place a compound's identity survives: `bug-400` as
        // bare prefix terms is `bug` (near-zero signal once a tree holds
        // hundreds of bug notes) OR `400` (prefix-matches 400002, 17.400,
        // 86_400_000), and the id note sinks. So a hyphenated term becomes
        // BOTH: the compound as an exact-adjacency phrase, and its parts as
        // independent prefix terms â€” the recall contract stays (`state-
        // machine` still finds "the machine's state"), and an id lookup
        // finds the id first because adjacency is high signal.
        .for_each(|t| {
            let cleaned = t.replace('"', "");
            if cleaned.contains('-') && cleaned.chars().all(|c| c.is_alphanumeric() || c == '-') {
                push(&mut terms, format!("\"{cleaned}\""));
                for part in cleaned.split('-').filter(|s| !s.is_empty()) {
                    push(&mut terms, format!("\"{part}\"*"));
                }
            } else {
                push(&mut terms, format!("\"{cleaned}\"*"));
            }
        });
    terms.join(" OR ")
}

/// One-line, length-capped preview of a block for hit listings.
fn snippet_of(text: &str) -> String {
    let one = text.split_whitespace().collect::<Vec<_>>().join(" ");
    one.chars().take(160).collect()
}

/// Snippet with the citation context prepended: `[H1 > H2] text` (and, for
/// table rows, the table's header row inside the context). A bare heading or
/// a lone table row is unreadable without it.
fn snippet_with_ctx(ctx: &str, text: &str) -> String {
    let s = snippet_of(text);
    if ctx.is_empty() { s } else { format!("[{ctx}] {s}") }
}

/// The content-hash part of a citation id (after the `r<ring>-` prefix): the
/// key every copy of one logical block shares across stores and rings.
fn cite_hash(cite: &str) -> &str {
    cite.split_once('-').map_or(cite, |(_, hash)| hash)
}

/// Collapses hits carrying the same content hash UNDER THE SAME heading
/// chain: the native auto-memory store mirrors brain files, so one logical
/// block would otherwise surface once per store. Identical short blocks in
/// DIFFERENT sections (a bare "yes", a repeated table row) are different
/// statements and are NOT collapsed. The lowest-ring copy survives, at the
/// first occurrence's rank; the suppressed copies' paths land on the keeper's
/// `mirrors`.
fn dedup_by_content(hits: Vec<Hit>) -> Vec<Hit> {
    let mut out: Vec<Hit> = Vec::new();
    let mut slot_by_key: std::collections::HashMap<(String, String), usize> =
        std::collections::HashMap::new();
    for hit in hits {
        let key = (cite_hash(&hit.cite).to_string(), hit.chain.clone());
        match slot_by_key.get(&key) {
            Some(&slot) => {
                let kept = &mut out[slot];
                if hit.ring < kept.ring {
                    let old = std::mem::replace(kept, hit);
                    kept.mirrors = old.mirrors;
                    kept.mirrors.push(old.path);
                } else if hit.path != kept.path && !kept.mirrors.contains(&hit.path) {
                    kept.mirrors.push(hit.path);
                }
            }
            None => {
                slot_by_key.insert(key, out.len());
                out.push(hit);
            }
        }
    }
    out
}

/// Ring folded into the BM25 order as a SMALL additive prior (bm25 is
/// negative-better): enough to break near-ties toward the more trusted ring,
/// far below the gaps between genuinely different lexical matches.
const RING_PRIOR: f64 = 0.05;
/// Weight of the heading-chain context column in BM25: context terms help
/// find a block but must never outrank the same terms in the block itself.
const CTX_WEIGHT: f64 = 0.3;

/// The ranked SELECT shared by [`recall`] and [`bm25_block_ids`].
/// The slice filter, as SQL over `prefix_count` bound prefixes starting at
/// `?3`.
///
/// Compared by component rather than with LIKE or GLOB: a path is under a
/// prefix when it IS the prefix or continues with `/`. That is the same rule
/// `config::under_prefix` applies â€” so `drafts` never claims `draftsman` â€”
/// and it cannot be confused by a `%`, `_` or `[` living in a filename.
fn slice_filter_sql(prefix_count: usize) -> String {
    if prefix_count == 0 {
        return String::new();
    }
    let terms: Vec<String> = (0..prefix_count)
        .map(|i| {
            let p = i + 3;
            format!(
                "(substr(d.path,1,length(?{p})) = ?{p} \
                  AND (length(d.path) = length(?{p}) OR substr(d.path,length(?{p})+1,1) = '/'))"
            )
        })
        .collect();
    format!(" AND ({})", terms.join(" OR "))
}

fn ranked_match_sql(select: &str, prefix_count: usize) -> String {
    let slice = slice_filter_sql(prefix_count);
    format!(
        "SELECT {select}
         FROM blocks_fts f
         JOIN blocks b ON b.id = f.rowid
         JOIN docs d ON d.id = b.doc_id
         WHERE blocks_fts MATCH ?1{slice}
         ORDER BY bm25(blocks_fts, 1.0, {CTX_WEIGHT}) + {RING_PRIOR} * d.ring, d.ring ASC
         LIMIT ?2"
    )
}

/// Binds `(fts, limit, prefixesâ€¦)` in the order [`ranked_match_sql`] numbers
/// them. Prefixes lose a trailing slash here so the SQL's length arithmetic
/// matches `config::under_prefix` exactly.
fn ranked_params(fts: &str, limit: usize, prefixes: &[String]) -> Vec<rusqlite::types::Value> {
    use rusqlite::types::Value;
    let mut p = vec![Value::Text(fts.to_string()), Value::Integer(limit as i64)];
    p.extend(prefixes.iter().map(|s| Value::Text(s.trim_end_matches('/').to_string())));
    p
}

pub fn recall(conn: &Connection, query: &str, limit: usize) -> anyhow::Result<Vec<Hit>> {
    recall_in(conn, query, limit, &[])
}

/// Reserve only the first lexical slot for the best matching ring-0/1
/// candidate. Every other candidate keeps its relative BM25(+prior) order.
/// Both displayed recall and the lexical input to hybrid use this policy.
fn reserve_top_trust<T>(candidates: &mut Vec<T>, ring: impl Fn(&T) -> u8) {
    if let Some(pos) = candidates.iter().position(|candidate| ring(candidate) <= 1)
        && pos > 0
    {
        let reserved = candidates.remove(pos);
        candidates.insert(0, reserved);
    }
}

/// Recall restricted to documents under `prefixes` â€” the slice filter. An
/// empty slice restricts nothing, which is what the root slice is.
pub fn recall_in(
    conn: &Connection,
    query: &str,
    limit: usize,
    prefixes: &[String],
) -> anyhow::Result<Vec<Hit>> {
    let fts = fts_query(query);
    if fts.is_empty() {
        return Ok(Vec::new());
    }
    // Candidate pool: `limit * 2` covers the common mirror case (one brain +
    // one native twin), but a pathological block with >2*limit copies can
    // exhaust the pool with duplicates and shrink below limit. The pool
    // expands on demand â€” capped at a sane multiple â€” so the dedup never
    // silently drops a distinct block that ranked below the duplicate wall.
    let mut pool = limit * 2;
    let mut hits;
    loop {
        let mut stmt = conn.prepare(&ranked_match_sql(
            "b.cite, d.path, d.ring, b.start_line, b.end_line, b.text, b.ctx, b.chain",
            prefixes.len(),
        ))?;
        let rows = stmt.query_map(rusqlite::params_from_iter(ranked_params(&fts, pool, prefixes)), |r| {
            let text: String = r.get(5)?;
            Ok(Hit {
                cite: r.get(0)?,
                path: r.get(1)?,
                ring: r.get::<_, i64>(2)? as u8,
                start_line: r.get::<_, i64>(3)? as usize,
                end_line: r.get::<_, i64>(4)? as usize,
                snippet: snippet_with_ctx(&r.get::<_, String>(6)?, &text),
                mirrors: Vec::new(),
                chain: r.get(7)?,
                text,
            })
        })?;
        hits = dedup_by_content(rows.filter_map(Result::ok).collect());
        if hits.len() >= limit || pool >= limit * 16 {
            break;
        }
        pool *= 2;
    }
    reserve_top_trust(&mut hits, |hit| hit.ring);
    hits.truncate(limit);
    Ok(hits)
}

/// Every indexed document with its block count, for classification in Rust.
///
/// Slice membership is a pure function of (path, config), so it is derived
/// here rather than stored: a stored slice would silently disagree with the
/// configuration the moment someone edited it and had not rescanned.
pub fn doc_block_counts(conn: &Connection) -> anyhow::Result<Vec<(String, usize)>> {
    let mut stmt = conn.prepare(
        "SELECT d.path, COUNT(b.id) FROM docs d
         LEFT JOIN blocks b ON b.doc_id = d.id
         GROUP BY d.id ORDER BY d.path",
    )?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as usize)))?;
    Ok(rows.filter_map(Result::ok).collect())
}

/// Expands a citation id to its full block(s) â€” the second disclosure layer.
/// Content-addressing means the HASH names the logical block while the ring
/// prefix only labels one copy's trust level: expansion matches every copy
/// sharing the hash (a mirror suppressed in recall stays reachable through
/// the kept hit's cite), lowest ring first.
pub fn expand(conn: &Connection, cite: &str) -> anyhow::Result<Vec<Block>> {
    let mut stmt = conn.prepare(
        "SELECT b.cite, d.path, d.ring, b.start_line, b.end_line, b.text
         FROM blocks b JOIN docs d ON d.id = b.doc_id
         WHERE substr(b.cite, instr(b.cite, '-') + 1) = ?1
         ORDER BY d.ring ASC, d.path ASC, b.start_line ASC",
    )?;
    let rows = stmt.query_map([cite_hash(cite)], |r| {
        Ok(Block {
            cite: r.get(0)?,
            path: r.get(1)?,
            ring: r.get::<_, i64>(2)? as u8,
            start_line: r.get::<_, i64>(3)? as usize,
            end_line: r.get::<_, i64>(4)? as usize,
            text: r.get(5)?,
        })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

// ---- vector store (semantic + hybrid recall) ----
//
// The RECORD for embeddings is the shared artifact store in the tree (see
// `vectors.rs`). What lives here is a per-host CACHE of the same vectors,
// keyed by CONTENT HASH so it survives every rebuild of the catalog, joined
// to blocks by hash at query time. Rows are L2-normalized at insert, so
// cosine similarity reduces to a dot product. No vector-index dependency: at
// ~20k blocks a linear scan in Rust is milliseconds and exact.
//
// A missing row means "not yet embedded" â€” that single fact makes both the
// hydrate from the shared store and `embed-index` resumable for free.

/// IEEE binary32 -> binary16, round-to-nearest-even, subnormals included.
/// Written out rather than pulled in: it is thirty lines of well-specified
/// bit work, and the crate that would supply it is a dependency in every
/// build of a tool whose bill of materials is deliberately short.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        // Inf, or NaN (kept NaN by a non-zero payload).
        return sign | 0x7c00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exponent - 127 + 15;
    if unbiased >= 0x1f {
        return sign | 0x7c00; // overflows half range -> infinity
    }
    if unbiased <= 0 {
        if unbiased < -10 {
            return sign; // below the smallest subnormal -> signed zero
        }
        // Subnormal: restore the implicit leading one, then shift into place.
        let full = mantissa | 0x0080_0000;
        let shift = (14 - unbiased) as u32;
        let half = full >> shift;
        let round_bit = 1u32 << (shift - 1);
        let round_up =
            (full & round_bit) != 0 && ((full & (round_bit - 1)) != 0 || (half & 1) != 0);
        return sign | (half + u32::from(round_up)) as u16;
    }
    let half = ((unbiased as u32) << 10) | (mantissa >> 13);
    let round_bit = 1u32 << 12;
    let round_up = (mantissa & round_bit) != 0 && ((mantissa & (round_bit - 1)) != 0 || (half & 1) != 0);
    sign | (half + u32::from(round_up)) as u16
}

/// Inverse of [`f32_to_f16`] â€” exact, every binary16 is a binary32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x03ff) as u32;
    let out = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            // Subnormal: normalize by shifting the leading one into place.
            let mut m = mantissa;
            let mut shifts = 0u32;
            while m & 0x0400 == 0 {
                m <<= 1;
                shifts += 1;
            }
            sign | ((113 - shifts) << 23) | ((m & 0x03ff) << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(out)
}

/// Canonical vector encoding, used byte-for-byte by both the local cache and
/// the shared artifact files.
///
/// INT8 uses the full signed symmetric range `[-127, 127]`. Each vector's
/// maximum absolute component maps to 127 with round-to-nearest-even. The
/// scale is intentionally not stored: semantic search uses cosine, which is
/// invariant under that positive per-vector scale. One 768-d v1 vector is
/// therefore exactly 768 bytes, with no hidden float metadata.
pub fn vec_to_blob(v: &[f32], precision: Precision) -> Vec<u8> {
    let mut out = Vec::with_capacity(precision.record_bytes(v.len()));
    if precision == Precision::I8 {
        let max = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let scale = if max > 0.0 { max } else { 1.0 };
        for x in v {
            let q = (x / scale * 127.0).round_ties_even().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
        return out;
    }
    for x in v {
        match precision {
            Precision::F16 => out.extend_from_slice(&f32_to_f16(*x).to_le_bytes()),
            Precision::F32 => out.extend_from_slice(&x.to_le_bytes()),
            Precision::I8 => unreachable!("handled above"),
        }
    }
    out
}

/// Inverse of [`vec_to_blob`], widened and normalized for callers that need a
/// float view. The canonical record remains the INT8 bytes; no scale is
/// reconstructed because cosine does not need it.
pub fn blob_to_vec(b: &[u8], precision: Precision) -> Vec<f32> {
    match precision {
        Precision::I8 => {
            let mut v: Vec<f32> = b.iter().map(|q| (*q as i8) as f32).collect();
            l2_normalize(&mut v);
            v
        }
        Precision::F16 => {
            let (chunks, _remainder) = b.as_chunks::<2>();
            chunks.iter().map(|c| f16_to_f32(u16::from_le_bytes(*c))).collect()
        }
        Precision::F32 => {
            let (chunks, _remainder) = b.as_chunks::<4>();
            chunks.iter().map(|c| f32::from_le_bytes(*c)).collect()
        }
    }
}

/// L2-normalizes in place; the zero vector is left untouched (it can never
/// rank anyway).
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Plain dot product â€” cosine similarity, given both sides are normalized.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn meta_get(conn: &Connection, key: &str) -> Option<String> {
    conn.query_row("SELECT value FROM meta WHERE key=?1", [key], |r| r.get(0)).ok()
}

fn meta_set(conn: &Connection, key: &str, value: &str) -> anyhow::Result<()> {
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [key, value],
    )?;
    Ok(())
}

/// The artifact spec the cached vectors were written under, if any.
pub fn stored_vector_spec(conn: &Connection) -> Option<VectorSpec> {
    let network_major = meta_get(conn, "embed_network_major")?.parse().ok()?;
    let profile_id = meta_get(conn, "embed_profile_id")?;
    let model = meta_get(conn, "embed_model")?;
    let dim = meta_get(conn, "embed_dim")?.parse().ok()?;
    let precision = match meta_get(conn, "embed_precision")?.as_str() {
        "f16" => Precision::F16,
        "f32" => Precision::F32,
        "i8" => Precision::I8,
        _ => return None,
    };
    // Absent means the empty prefix: a cache written before document
    // prefixes existed held raw-document vectors, which is exactly what an
    // empty prefix means today.
    let doc_prefix = meta_get(conn, "embed_doc_prefix").unwrap_or_default();
    Some(VectorSpec { network_major, profile_id, model, dim, precision, doc_prefix })
}

/// Records `(model, dim, precision, doc_prefix)` in meta; a DIFFERENT stored spec drops
/// every cached vector â€” vectors of two models, widths or precisions produce
/// numbers that look like similarity and are not. Returns true when a drop
/// happened. Re-filling is a first-class path: the shared store still holds
/// the artifacts, so the cost is usually a hydrate, not an embed run.
pub fn ensure_vector_spec(conn: &Connection, spec: &VectorSpec) -> anyhow::Result<bool> {
    let stored = stored_vector_spec(conn);
    if stored.as_ref() == Some(spec) {
        return Ok(false);
    }
    let dropping = stored.is_some();
    conn.execute("DELETE FROM vectors", [])?;
    meta_set(conn, "embed_network_major", &spec.network_major.to_string())?;
    meta_set(conn, "embed_profile_id", &spec.profile_id)?;
    meta_set(conn, "embed_model", &spec.model)?;
    meta_set(conn, "embed_dim", &spec.dim.to_string())?;
    meta_set(conn, "embed_precision", spec.precision.as_str())?;
    meta_set(conn, "embed_doc_prefix", &spec.doc_prefix)?;
    Ok(dropping)
}

/// Drops cached vectors whose content is no longer anywhere in the catalog.
/// Called at the end of every scan: an edited block's old vector goes, every
/// unchanged block's vector stays.
fn prune_vectors(tx: &rusqlite::Transaction<'_>) -> anyhow::Result<()> {
    tx.execute("DELETE FROM vectors WHERE content_hash NOT IN (SELECT hash FROM blocks)", [])?;
    Ok(())
}

/// The embed work queue: content hashes with no cached vector, each with one
/// representative text, in document order. DISTINCT by hash â€” the same
/// statement in two files is one artifact, embedded once.
pub fn hashes_without_vectors(
    conn: &Connection,
    spec: &VectorSpec,
    limit: usize,
) -> anyhow::Result<Vec<(String, String)>> {
    let mut stmt = conn.prepare(
        "SELECT b.hash, min(b.text), min(b.id) FROM blocks b
         LEFT JOIN vectors v ON v.content_hash = b.hash AND v.model = ?1 AND v.dim = ?2
         WHERE v.content_hash IS NULL
         GROUP BY b.hash ORDER BY min(b.id) LIMIT ?3",
    )?;
    // SQLite reads a negative LIMIT as unbounded â€” which is exactly what a
    // caller asking for usize::MAX (a full hydrate) means.
    let bound = i64::try_from(limit).unwrap_or(-1);
    let rows = stmt.query_map(
        rusqlite::params![spec.model, spec.dim as i64, bound],
        |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
    )?;
    Ok(rows.filter_map(Result::ok).collect())
}

/// (blocks reachable by semantic recall, blocks total) FOR THIS SPEC â€” the
/// coverage number every degradation warning quotes. Spec-relative on
/// purpose: vectors of another model are not coverage, they are ballast.
pub fn vector_coverage(conn: &Connection, spec: &VectorSpec) -> anyhow::Result<(usize, usize)> {
    let (v, b): (i64, i64) = conn.query_row(
        "SELECT (SELECT count(*) FROM blocks b JOIN vectors v
                   ON v.content_hash = b.hash AND v.model = ?1 AND v.dim = ?2),
                (SELECT count(*) FROM blocks)",
        rusqlite::params![spec.model, spec.dim as i64],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    Ok((v as usize, b as usize))
}

/// Caches one content hash's embedding at the spec's width. INT8 must use the
/// exact same max-absolute codec as the shared record; an extra floating-point
/// normalization here could otherwise put different bytes in the disposable
/// cache than in the authoritative store. Legacy F16/F32 caches retain their
/// normalized representation.
pub fn insert_vector(
    conn: &Connection,
    content_hash: &str,
    spec: &VectorSpec,
    embedding: &[f32],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        embedding.len() == spec.dim,
        "vector has {} components, profile {} requires {}",
        embedding.len(),
        spec.profile_id,
        spec.dim
    );
    let encoded = if spec.precision == Precision::I8 {
        vec_to_blob(embedding, spec.precision)
    } else {
        let mut normalized = embedding.to_vec();
        l2_normalize(&mut normalized);
        vec_to_blob(&normalized, spec.precision)
    };
    conn.execute(
        "INSERT INTO vectors(content_hash, model, dim, embedding) VALUES(?1, ?2, ?3, ?4)
         ON CONFLICT(content_hash) DO UPDATE SET
           model=excluded.model, dim=excluded.dim, embedding=excluded.embedding",
        rusqlite::params![content_hash, spec.model, spec.dim as i64, encoded],
    )?;
    Ok(())
}

/// Hits for the given block ids, preserving the ids' order. Ids that no
/// longer exist (index moved on) are silently skipped.
fn hits_for_block_ids(conn: &Connection, ids: &[i64]) -> anyhow::Result<Vec<Hit>> {
    let mut stmt = conn.prepare(
        "SELECT b.cite, d.path, d.ring, b.start_line, b.end_line, b.text, b.ctx, b.chain
         FROM blocks b JOIN docs d ON d.id = b.doc_id WHERE b.id = ?1",
    )?;
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        let hit = stmt.query_row([id], |r| {
            let text: String = r.get(5)?;
            Ok(Hit {
                cite: r.get(0)?,
                path: r.get(1)?,
                ring: r.get::<_, i64>(2)? as u8,
                start_line: r.get::<_, i64>(3)? as usize,
                end_line: r.get::<_, i64>(4)? as usize,
                snippet: snippet_with_ctx(&r.get::<_, String>(6)?, &text),
                mirrors: Vec::new(),
                chain: r.get(7)?,
                text,
            })
        });
        if let Ok(hit) = hit {
            out.push(hit);
        }
    }
    Ok(out)
}

/// Block ids ranked by dot product against a normalized query vector â€” a full
/// linear scan, exact by construction. Vectors join blocks BY CONTENT HASH,
/// so a rebuilt catalog re-attaches every surviving vector to its new rowid.
/// Rows whose width does not match the query are skipped (a transitional
/// state while a spec change is being re-embedded).
fn semantic_block_ids(
    conn: &Connection,
    spec: &VectorSpec,
    query_vec: &[f32],
    limit: usize,
    prefixes: &[String],
) -> anyhow::Result<Vec<i64>> {
    // The slice filter belongs HERE and not after ranking: a hybrid answer
    // fuses this list with the lexical one, so a vector hit from outside the
    // slice would leak in through the fusion no matter what the lexical half
    // was restricted to.
    let mut stmt = conn.prepare(&format!(
        "SELECT b.id, v.embedding FROM vectors v
         JOIN blocks b ON b.hash = v.content_hash
         JOIN docs d ON d.id = b.doc_id
         WHERE v.model = ?1 AND v.dim = ?2{}",
        slice_filter_sql(prefixes.len())
    ))?;
    let mut params: Vec<rusqlite::types::Value> = vec![
        rusqlite::types::Value::Text(spec.model.clone()),
        rusqlite::types::Value::Integer(spec.dim as i64),
    ];
    params.extend(
        prefixes.iter().map(|p| rusqlite::types::Value::Text(p.trim_end_matches('/').to_string())),
    );
    let rows = stmt.query_map(rusqlite::params_from_iter(params), |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
    })?;
    if spec.precision == Precision::I8 {
        let query = vec_to_blob(query_vec, Precision::I8);
        anyhow::ensure!(query.len() == spec.dim, "query vector width does not match {}", spec.dim);
        let mut scored: Vec<I8Score> = rows
            .filter_map(Result::ok)
            .filter_map(|(id, blob)| i8_score(id, &query, &blob))
            .collect();
        scored.sort_by(|a, b| i8_cosine_desc(a, b).then(a.id.cmp(&b.id)));
        scored.truncate(limit);
        return Ok(scored.into_iter().map(|score| score.id).collect());
    }
    let mut scored: Vec<(i64, f32)> = rows
        .filter_map(Result::ok)
        .filter_map(|(id, blob)| {
            let v = blob_to_vec(&blob, spec.precision);
            (v.len() == query_vec.len()).then(|| (id, dot(&v, query_vec)))
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    scored.truncate(limit);
    Ok(scored.into_iter().map(|(id, _)| id).collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct I8Score {
    id: i64,
    dot: i64,
    document_norm_sq: u64,
}

fn i8_score(id: i64, query: &[u8], document: &[u8]) -> Option<I8Score> {
    if query.len() != document.len() {
        return None;
    }
    let mut dot = 0i64;
    let mut document_norm_sq = 0u64;
    for (&q, &d) in query.iter().zip(document) {
        let q = (q as i8) as i64;
        let d = (d as i8) as i64;
        dot += q * d;
        document_norm_sq += (d * d) as u64;
    }
    (document_norm_sq > 0).then_some(I8Score { id, dot, document_norm_sq })
}

/// Exact descending cosine order for INT8 documents. The query norm is the
/// same for every candidate and cancels; squaring and cross-multiplying in
/// u128 avoids square roots and floating-point differences across machines.
fn i8_cosine_desc(a: &I8Score, b: &I8Score) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.dot.signum(), b.dot.signum()) {
        (sa, sb) if sa != sb => sb.cmp(&sa),
        (0, 0) => Ordering::Equal,
        (1, 1) => {
            let left = (a.dot as u128).pow(2) * b.document_norm_sq as u128;
            let right = (b.dot as u128).pow(2) * a.document_norm_sq as u128;
            right.cmp(&left)
        }
        (-1, -1) => {
            let left = (a.dot.unsigned_abs() as u128).pow(2) * b.document_norm_sq as u128;
            let right = (b.dot.unsigned_abs() as u128).pow(2) * a.document_norm_sq as u128;
            left.cmp(&right)
        }
        _ => unreachable!("signum is -1, 0, or 1"),
    }
}

/// Pure cosine ranking over all stored vectors (query vector must be
/// normalized). Blocks without vectors simply cannot appear.
pub fn semantic_recall(
    conn: &Connection,
    spec: &VectorSpec,
    query_vec: &[f32],
    limit: usize,
    prefixes: &[String],
) -> anyhow::Result<Vec<Hit>> {
    let ids = semantic_block_ids(conn, spec, query_vec, limit, prefixes)?;
    // Same mirror suppression as lexical recall: a block mirrored in the
    // native store has the SAME content hash and therefore the SAME vector -
    // it ranks adjacent to itself, and without this pass the fused result
    // carries one logical statement twice while displacing a distinct block.
    Ok(dedup_by_content(hits_for_block_ids(conn, &ids)?))
}

/// Lexical candidate ids in BM25(+prior) order with the same single top-trust
/// slot reservation as [`recall`]. Fusion needs block ids; mirror suppression
/// still happens after fusion, whereas standalone recall suppresses first.
fn bm25_block_ids(
    conn: &Connection,
    query: &str,
    limit: usize,
    prefixes: &[String],
) -> anyhow::Result<Vec<i64>> {
    let fts = fts_query(query);
    if fts.is_empty() {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare(&ranked_match_sql("b.id, d.ring", prefixes.len()))?;
    let rows = stmt.query_map(
        rusqlite::params_from_iter(ranked_params(&fts, limit, prefixes)),
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)? as u8)),
    )?;
    let mut candidates: Vec<(i64, u8)> = rows.filter_map(Result::ok).collect();
    reserve_top_trust(&mut candidates, |(_, ring)| *ring);
    Ok(candidates.into_iter().map(|(id, _)| id).collect())
}

/// Reciprocal rank fusion over ranked id lists: score(d) = Î£ 1/(k + rank),
/// rank starting at 1. Ties break by id for determinism.
pub fn rrf_fuse(lists: &[Vec<i64>], k: f64) -> Vec<i64> {
    let mut score: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
    for list in lists {
        for (i, id) in list.iter().enumerate() {
            *score.entry(*id).or_default() += 1.0 / (k + (i + 1) as f64);
        }
    }
    let mut items: Vec<(i64, f64)> = score.into_iter().collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    items.into_iter().map(|(id, _)| id).collect()
}

/// BM25 list âŠ• semantic list via RRF â€” each fetched with a wider pool than
/// the final limit so fusion has something to reorder.
pub fn hybrid_recall(
    conn: &Connection,
    spec: &VectorSpec,
    query: &str,
    query_vec: &[f32],
    limit: usize,
    rrf_k: f64,
    prefixes: &[String],
) -> anyhow::Result<Vec<Hit>> {
    let pool = (limit * 4).max(20);
    let lexical = bm25_block_ids(conn, query, pool, prefixes)?;
    let semantic = semantic_block_ids(conn, spec, query_vec, pool, prefixes)?;
    let mut fused = rrf_fuse(&[lexical, semantic], rrf_k);
    fused.truncate(limit);
    // Mirror suppression for the same reason as semantic_recall: both input
    // lists can contain the native twin of a brain block, and fusion scores
    // the twin pair twice.
    Ok(dedup_by_content(hits_for_block_ids(conn, &fused)?))
}

/// Ensures the index exists and is fresh, rebuilding when stale. Rebuilds are
/// serialized by a lockfile: when another process is already rebuilding, this
/// one serves the still-valid committed snapshot instead of failing with
/// SQLITE_BUSY or duplicating the work.
pub fn ensure_fresh(
    state_dir: &Path,
    brain_root: &Path,
    native_root: Option<&Path>,
    rules: &RingRules,
) -> anyhow::Result<Connection> {
    let mut conn = open(state_dir).context("open index")?;
    if stale(&conn, brain_root, native_root, rules)? {
        // `None` = another rebuilder is active; serve the committed snapshot.
        let lock = crate::lockfile::acquire(&state_dir.join("scan.lock"), 500, 120);
        // Re-check under the lock: the previous holder may have rebuilt
        // exactly what we were about to.
        if lock.is_some() && stale(&conn, brain_root, native_root, rules)? {
            scan(&mut conn, brain_root, native_root, rules)?;
        }
    }
    Ok(conn)
}

#[cfg(test)]
mod path_shape_tests {
    use super::*;

    #[test]
    fn doc_paths_are_slash_separated_whatever_the_platform_uses() {
        assert_eq!(normalize_separators(r"mind\secrets\age.key", '\\'), "mind/secrets/age.key");
        assert_eq!(normalize_separators("mind/secrets/age.key", '/'), "mind/secrets/age.key");
        // A backslash is a legal filename character on unix and must survive.
        assert_eq!(normalize_separators(r"odd\name.md", '/'), r"odd\name.md");
    }

    #[test]
    fn the_exclusion_boundary_holds_for_backslash_separated_paths() {
        // Without normalization every one of these slips past the prefix
        // match and lands in the catalog â€” secrets first.
        for raw in [r"mind\secrets\age.key.md", r"logs\session.md", r"projects\repo\a.md", r"knowledge\archive\old.md", r".git\COMMIT_EDITMSG.md"] {
            let rel = normalize_separators(raw, '\\');
            assert!(excluded(&rel, &RingRules::default()), "{raw} normalized to {rel} must be excluded");
        }
        for raw in [r"mind\secrets", r"logs", r"projects", r"knowledge\archive"] {
            assert!(excluded_dir(&normalize_separators(raw, '\\'), &RingRules::default()), "{raw}");
        }
    }

    #[test]
    fn ring_defaults_survive_the_separator() {
        let rules = RingRules::default();
        assert_eq!(default_ring(&normalize_separators(r"mind\memories\MEMORY.md", '\\'), &rules), 1);
        assert_eq!(default_ring(&normalize_separators(r"mind\memories\topic.md", '\\'), &rules), 2);
        assert_eq!(default_ring(&normalize_separators(r"todo\active\x\STATUS.md", '\\'), &rules), 4);
    }

    #[test]
    fn secret_shaped_names_are_caught_after_normalization() {
        assert!(secret_shaped(&normalize_separators(r"knowledge\hosts\my.pem", '\\')));
        assert!(secret_shaped(&normalize_separators(r"knowledge\hosts\password-notes.md", '\\')));
    }
}

#[cfg(test)]
mod scoping_tests {
    use super::*;

    #[test]
    fn cfetchignore_scopes_a_subtree_out_of_the_brain_scan() {
        let brain = tempfile::tempdir().unwrap();
        let root = brain.path();
        std::fs::create_dir_all(root.join("knowledge/generated")).unwrap();
        std::fs::write(root.join("knowledge/real.md"), "# real\n").unwrap();
        std::fs::write(root.join("knowledge/generated/api.md"), "# generated\n").unwrap();
        // Nothing here is gitignored, and nothing may become gitignored: the
        // generated docs are committed on purpose. Before the overlay there
        // was no way to express "index everything but this".
        std::fs::write(root.join(".cfetchignore"), "knowledge/generated/\n").unwrap();

        let docs: Vec<String> = collect_files(root, None, &RingRules::default())
            .into_iter()
            .map(|f| f.doc_path)
            .collect();
        assert_eq!(docs, vec!["knowledge/real.md".to_string()]);
    }

    #[test]
    fn managed_cards_remain_knowledge_when_the_outer_repo_ignores_the_checkout() {
        let brain = tempfile::tempdir().unwrap();
        let root = brain.path();
        let cards = root.join("knowledge/cards");
        std::fs::create_dir_all(cards.join(".git")).unwrap();
        std::fs::create_dir_all(cards.join("cloud/certificates/example")).unwrap();
        std::fs::write(root.join(".gitignore"), "knowledge/cards/\n").unwrap();
        std::fs::write(cards.join("catalog.json"), "{\"schema_version\":1,\"sets\":[]}")
            .unwrap();
        std::fs::write(
            cards.join("cloud/certificates/example/question.md"),
            "# Question?\n\nAnswer.\n",
        )
        .unwrap();
        std::fs::write(cards.join(".git/config.md"), "# must stay excluded\n").unwrap();

        let docs: Vec<_> = collect_files(root, None, &RingRules::default())
            .into_iter()
            .map(|file| (file.doc_path, file.default_ring))
            .collect();
        assert_eq!(
            docs,
            vec![(
                "knowledge/cards/cloud/certificates/example/question.md".to_string(),
                crate::config::UNMATCHED_RING,
            )]
        );
    }

    #[test]
    fn cfetchignore_cannot_reopen_the_hard_exclusion_boundary() {
        let brain = tempfile::tempdir().unwrap();
        let root = brain.path();
        std::fs::create_dir_all(root.join("mind/secrets")).unwrap();
        std::fs::create_dir_all(root.join("mind/memories")).unwrap();
        std::fs::write(root.join("mind/secrets/token.md"), "# token\n").unwrap();
        std::fs::write(root.join("mind/memories/how.md"), "# how\n").unwrap();
        std::fs::write(root.join("keep.md"), "# keep\n").unwrap();
        // Gitignore syntax carries negation, so the overlay can be written to
        // ASK for secrets back. It gets `mind/` excluded and nothing else:
        // the boundary is applied after the walk, not by it.
        std::fs::write(root.join(".cfetchignore"), "mind/\n!mind/secrets/\n").unwrap();

        let docs: Vec<String> = collect_files(root, None, &RingRules::default())
            .into_iter()
            .map(|f| f.doc_path)
            .collect();
        assert_eq!(docs, vec!["keep.md".to_string()]);
    }

    #[test]
    fn writes_under_a_scoped_out_subtree_do_not_churn_the_index() {
        let brain = tempfile::tempdir().unwrap();
        let root = brain.path();
        std::fs::create_dir_all(root.join("scratch")).unwrap();
        std::fs::write(root.join("keep.md"), "# keep\n").unwrap();
        std::fs::write(root.join(".cfetchignore"), "scratch/\n").unwrap();
        let rules = RingRules::default();
        let before = tree_fingerprint(root, None, &rules);

        // The staleness basis is the same file list, so scoping a subtree out
        // buys more than a smaller catalog: a directory that churns every few
        // seconds stops forcing rebuilds of everything else.
        std::fs::write(root.join("scratch/churn.md"), "# churn\n").unwrap();
        assert_eq!(
            tree_fingerprint(root, None, &rules),
            before,
            "a write under a scoped-out subtree must not make the catalog stale"
        );
        std::fs::write(root.join("keep.md"), "# keep, expanded\n").unwrap();
        assert_ne!(tree_fingerprint(root, None, &rules), before, "an indexed write still must");
    }

    /// Fixture: 20 generated API docs, a real knowledge subtree, some
    /// memories, and a directory of binaries that outweighs all of it on
    /// disk by three orders of magnitude.
    fn mixed_census() -> DocCensus {
        let rules = RingRules::default();
        let mut census = DocCensus::default();
        // The heavy directory the index never opens: not one entry of it is
        // recordable, whatever it weighs.
        for i in 0..4 {
            assert!(!indexable_doc(&format!("bulk/blob{i}.bin"), &rules));
        }
        for i in 0..20 {
            let rel = format!("knowledge/generated/api/p{i}.md");
            assert!(indexable_doc(&rel, &rules));
            census.record(&rel);
        }
        for i in 0..5 {
            census.record(&format!("knowledge/hosts/h{i}.md"));
        }
        for i in 0..4 {
            census.record(&format!("mind/memories/m{i}.md"));
        }
        census.record("AGENT.md");
        census
    }

    #[test]
    fn census_ranks_by_what_the_indexer_reads_not_by_bytes_on_disk() {
        let census = mixed_census();
        assert_eq!(census.total(), 30);
        let hits = census.concentrations(0.15);
        // Disjoint and deepest: the generated docs are named where they sit,
        // and no candidate contains another.
        assert_eq!(
            hits,
            vec![
                ("knowledge/generated/api".to_string(), 20),
                ("knowledge/hosts".to_string(), 5),
            ]
        );
        // `knowledge/` holds 25 of 30 and is still not the answer: naming it
        // would tell the operator to exclude their whole knowledge tree.
        assert!(hits.iter().all(|(p, _)| p != "knowledge"));
        // Nothing on the ranking came from disk usage.
        assert!(hits.iter().all(|(p, _)| !p.starts_with("bulk")));
    }

    #[test]
    fn census_names_the_parent_when_the_cost_is_spread_across_it() {
        let mut census = DocCensus::default();
        for quarter in 0..4 {
            for note in 0..5 {
                census.record(&format!("todo/done/q{quarter}/n{note}.md"));
            }
        }
        for i in 0..10 {
            census.record(&format!("knowledge/hosts/h{i}.md"));
        }
        // No child of `todo/done` clears the bar alone, so `todo/done` is the
        // deepest honest name â€” descending further would advise excluding one
        // quarter and leaving three behind.
        assert_eq!(census.concentrations(0.5), vec![("todo/done".to_string(), 20)]);
    }

    #[test]
    fn a_tree_that_is_all_one_subtree_yields_no_advice_at_all() {
        let mut census = DocCensus::default();
        for area in ["hosts", "world", "projects", "coding", "health", "finance", "career"] {
            census.record(&format!("knowledge/{area}/one.md"));
        }
        // `knowledge/` holds everything and every child is equal. There is
        // nothing to scope out, and "exclude your entire brain" is worse than
        // saying nothing at all.
        assert!(census.concentrations(0.15).is_empty(), "{:?}", census.concentrations(0.15));
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn int8_round_trips_within_its_quantisation_step() {
        // A per-vector scale means the largest component always maps to
        // exactly 127, so the error is bounded by half a step of THIS
        // vector's range rather than of an assumed [-1,1].
        let v: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.017).sin() * 0.31).collect();
        let blob = vec_to_blob(&v, Precision::I8);
        assert_eq!(blob.len(), Precision::I8.record_bytes(256), "one byte per component");
        let back = blob_to_vec(&blob, Precision::I8);
        assert_eq!(back.len(), 256);
        let mut expected = v.clone();
        l2_normalize(&mut expected);
        assert!(dot(&expected, &back) > 0.9999, "INT8 record preserves direction");
    }

    #[test]
    fn int8_preserves_direction_which_is_all_cosine_uses() {
        let v: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.031).cos()).collect();
        let mut n = v.clone();
        l2_normalize(&mut n);
        let back = blob_to_vec(&vec_to_blob(&n, Precision::I8), Precision::I8);
        let dot: f32 = n.iter().zip(&back).map(|(a, b)| a * b).sum();
        let nb: f32 = back.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Cosine between the original and its quantised self.
        assert!(dot / nb > 0.9999, "direction drifted: cos = {}", dot / nb);
    }

    #[test]
    fn int8_codec_is_byte_exact_ties_even_and_idempotent() {
        let blob = vec_to_blob(&[1.0, 0.5, -1.0, 0.0], Precision::I8);
        assert_eq!(blob, vec![127, 64, 129, 0], "-127 is stored as its two's-complement byte");
        assert_eq!(
            vec_to_blob(&blob_to_vec(&blob, Precision::I8), Precision::I8),
            blob,
            "hydrate/cache cycles must never rewrite canonical bytes"
        );
    }

    #[test]
    fn int8_cosine_order_is_exact_across_norms_and_signs() {
        let query = vec![127u8, 0];
        let mut scores = [
            i8_score(4, &query, &[156, 100]).unwrap(), // [-100, 100]
            i8_score(3, &query, &[0, 127]).unwrap(),
            i8_score(2, &query, &[100, 100]).unwrap(),
            i8_score(1, &query, &[127, 0]).unwrap(),
        ];
        scores.sort_by(|a, b| i8_cosine_desc(a, b).then(a.id.cmp(&b.id)));
        assert_eq!(scores.map(|score| score.id), [1, 2, 3, 4]);
    }

    #[test]
    fn an_int8_record_has_no_hidden_scale_trailer() {
        assert_eq!(blob_to_vec(&[1, 2], Precision::I8).len(), 2);
        assert!(blob_to_vec(&[], Precision::I8).is_empty());
    }

    #[test]
    fn each_precision_reports_the_bytes_it_actually_occupies() {
        assert_eq!(Precision::F32.record_bytes(10), 40);
        assert_eq!(Precision::F16.record_bytes(10), 20);
        assert_eq!(Precision::I8.record_bytes(10), 10, "exactly one byte per component");
        // The reason i8 is worth having: half of f16 on the scan path.
        assert!(Precision::I8.record_bytes(1024) * 2 < Precision::F16.record_bytes(1024) * 2 + 8);
    }
    use super::*;
    use crate::config::{Precision, RingRule, VectorSpec};

    fn brain(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (rel, content) in files {
            let p = dir.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, content).unwrap();
        }
        dir
    }

    #[test]
    fn ring_defaults_follow_design_mapping() {
        // Regression lock: with the shipped rules the taxonomy entry point
        // answers exactly what the hardcoded version answered.
        let r = RingRules::default();
        assert_eq!(default_ring("AGENT.md", &r), 1);
        assert_eq!(default_ring("README.md", &r), 1);
        assert_eq!(default_ring("mind/memories/MEMORY.md", &r), 1);
        assert_eq!(default_ring("mind/memories/feedback_x.md", &r), 2);
        assert_eq!(default_ring("knowledge/hosts/example/storage.md", &r), 3);
        assert_eq!(default_ring("todo/active/task/STATUS.md", &r), 4);
        assert_eq!(default_ring("todo/staging/hot-file-1234abcd.md", &r), 5);
        assert_eq!(default_ring("todo/staging/dismissed/hot-file-1234abcd.md", &r), 5);
        // An unmigrated tree must stay quarantined: falling through to the
        // unmatched ring would promote candidates into recall on upgrade.
        assert_eq!(default_ring("staging/cfetch/hot-file-1234abcd.md", &r), 5);
        // Disposable working material is excluded outright, not ringed.
        assert!(r.excluded("todo/scratch/dump.md"));
    }

    #[test]
    fn staged_ring5_candidates_are_never_recalled() {
        // Ring 5 is the ladder's quarantine: staging files sit in the same
        // tree as everything else and must still be invisible to recall and
        // injection. Fabricate a tree with a REAL staged candidate, scan it,
        // and prove nothing of it comes back.
        let dir = brain(&[("knowledge/live.md", "the reachable fact zulqar\n")]);
        let staging_dir = dir.path().join("staging/cfetch");
        crate::staging::write(
            &staging_dir,
            &crate::staging::Candidate {
                id: "hot-file-deadbeef".into(),
                reason: "hot-file".into(),
                session: "s1".into(),
                host: "host-alpha".into(),
                ts: 1,
                kind: "write".into(),
                payload: serde_json::json!({"file_path": "/b/knowledge/x.md", "quarantined": "zulqar"}),
            },
        )
        .unwrap();
        // A dismissed candidate is kept in the tree too, and is just as quiet.
        crate::staging::write(
            &staging_dir,
            &crate::staging::Candidate {
                id: "hot-file-cafebabe".into(),
                reason: "hot-file".into(),
                session: "s1".into(),
                host: "host-beta".into(),
                ts: 2,
                kind: "write".into(),
                payload: serde_json::json!({"note": "zulqar"}),
            },
        )
        .unwrap();
        crate::staging::dismiss(&staging_dir, "hot-file-cafebabe").unwrap();

        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.docs, 1, "only the ring-3 knowledge file is indexed");
        assert_eq!(report.skipped_high_ring, 2, "both candidates are skipped as ring 5");
        assert!(report.skipped.iter().all(|p| p.starts_with("staging/")));
        assert_eq!(recall(&conn, "zulqar", 10).unwrap().len(), 1, "only the ring-3 hit");
        assert_eq!(recall(&conn, "zulqar", 10).unwrap()[0].path, "knowledge/live.md");
        assert!(
            recall(&conn, "hot-file", 10).unwrap().is_empty(),
            "no staged candidate is recallable, by any of its words"
        );
    }

    #[test]
    fn custom_ring_rules_drive_the_indexed_rings() {
        // A tree with none of the shipped names: rings come from config only.
        let dir = brain(&[
            ("laws.md", "kwenty is the hard law\n"),
            ("handbook/style.md", "trakkel is the house style\n"),
            ("misc/loose.md", "vurpel is a loose fact\n"),
            ("scratch/dump.md", "pargast is scratch\n"),
        ]);
        let rules = RingRules {
            rules: vec![
                RingRule { prefix: "laws.md".into(), ring: 0 },
                RingRule { prefix: "handbook/".into(), ring: 1 },
                RingRule { prefix: "scratch/".into(), ring: 5 },
                RingRule { prefix: String::new(), ring: 4 },
            ],
            exclude_prefixes: Vec::new(),
        };
        let mut conn = open(dir.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &rules).unwrap();
        assert_eq!(report.skipped, vec!["scratch/dump.md".to_string()], "ring 5 is never indexed");
        let ring_of = |q: &str| recall(&conn, q, 5).unwrap().first().map(|h| h.ring);
        assert_eq!(ring_of("kwenty"), Some(0));
        assert_eq!(ring_of("trakkel"), Some(1));
        assert_eq!(ring_of("vurpel"), Some(4), "the catch-all rule applies");
        assert_eq!(ring_of("pargast"), None);
    }

    #[test]
    fn configured_exclusions_apply_while_hard_ones_cannot_be_lifted() {
        let dir = brain(&[
            ("mind/secrets/tokens.md", "hyllvar token\n"),
            ("drafts/wip.md", "nogrant draft\n"),
            ("knowledge/live.md", "kavender fact\n"),
        ]);
        let rules = RingRules {
            rules: RingRules::default().rules,
            exclude_prefixes: vec!["drafts/".into()],
        };
        let mut conn = open(dir.path()).unwrap();
        scan(&mut conn, dir.path(), None, &rules).unwrap();
        assert!(recall(&conn, "hyllvar", 5).unwrap().is_empty(), "secrets stay out, always");
        assert!(recall(&conn, "nogrant", 5).unwrap().is_empty(), "configured exclusion applies");
        assert_eq!(recall(&conn, "kavender", 5).unwrap().len(), 1);
    }

    fn cite_id(ring: u8, text: &str) -> String {
        cite_from_hash(ring, &content_hash(text))
    }

    #[test]
    fn citation_preserves_exact_body_identity_and_ring() {
        let a = cite_id(3, "The  Quick   Fox");
        let b = cite_id(3, "the quick fox");
        assert_ne!(a, b, "different embedding input must not share a citation hash");
        assert_eq!(a, cite_id(3, "The  Quick   Fox"));
        assert!(a.starts_with("r3-"));
        assert_ne!(a, cite_id(3, "the quick foxes"));
        assert_ne!(a, cite_id(2, "The  Quick   Fox"), "ring is part of the id");
    }

    #[test]
    fn segmentation_splits_list_items_and_tables() {
        let text = "# H\n\n- item one\n  continued\n- item two\n\n| a | b |\n| c | d |\n\npara one\npara two\n";
        let blocks = segment(text, 0);
        let bodies: Vec<&str> = blocks.iter().map(|(_, _, b)| b.as_str()).collect();
        assert!(bodies.contains(&"# H"));
        assert!(bodies.contains(&"- item one\n  continued"));
        assert!(bodies.contains(&"- item two"));
        assert!(bodies.contains(&"| a | b |"));
        assert!(bodies.contains(&"para one\npara two"));
        let (start, end, _) = blocks.iter().find(|(_, _, b)| b.starts_with("- item one")).unwrap();
        assert_eq!((*start, *end), (3, 4));
    }

    #[test]
    fn scan_and_recall_end_to_end() {
        let dir = brain(&[
            ("AGENT.md", "# Rules\n\n- never rsync zfs to zfs\n"),
            ("knowledge/world/opentofu.md", "OpenTofu state is AES encrypted.\n\nPulumi is retired.\n"),
            ("todo/active/x/STATUS.md", "current quest: ship the recall index\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.docs, 3);
        assert!(report.blocks >= 4);

        let hits = recall(&conn, "pulumi", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].ring, 3);
        assert!(hits[0].cite.starts_with("r3-"));
        assert!(hits[0].path.ends_with("opentofu.md"));

        // prefix matching: "encrypt" hits "encrypted"
        let hits = recall(&conn, "encrypt", 5).unwrap();
        assert_eq!(hits.len(), 1);

        let full = expand(&conn, &hits[0].cite).unwrap();
        assert_eq!(full.len(), 1);
        assert!(full[0].text.contains("AES encrypted"));
    }

    #[test]
    fn private_content_never_enters_the_index() {
        let dir = brain(&[(
            "knowledge/x.md",
            "public fact\n\n<private>hunter2 is the password</private>\n\nanother fact\n",
        )]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert!(recall(&conn, "hunter2", 5).unwrap().is_empty());
        assert_eq!(recall(&conn, "public", 5).unwrap().len(), 1);
    }

    #[test]
    fn an_unclosed_private_tag_is_reported_not_silent() {
        // A mere mention of the tag in prose blanks everything after it,
        // fail-closed — the hiding is the design; the silence was the bug.
        let dir = brain(&[
            ("knowledge/good.md", "public fact\n"),
            ("knowledge/mentioned.md", "we use <private> blocks here\nand this tail is gone too\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.private_swallowed, vec!["knowledge/mentioned.md".to_string()]);
        // Balanced private regions stay unreported.
        assert!(report.unreadable.is_empty());
        // The tail is really gone from recall (fail-closed held) ...
        assert!(recall(&conn, "tail", 5).unwrap().is_empty());
        // ... and the mention itself never entered the index either.
        assert!(recall(&conn, "blocks here", 5).unwrap().is_empty());
    }

    #[test]
    fn an_unreadable_file_is_reported_under_its_own_reason() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("knowledge")).unwrap();
        std::fs::write(dir.path().join("knowledge").join("ok.md"), "fine\n").unwrap();
        let bad = dir.path().join("knowledge").join("binary.md");
        std::fs::create_dir_all(bad.parent().unwrap()).unwrap();
        std::fs::write(&bad, b"not utf8: \xff\xfe broken\n").unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.unreadable, vec!["knowledge/binary.md".to_string()]);
        assert!(report.skipped.is_empty(), "not under the ring-5+ reason: {:?}", report.skipped);
        assert_eq!(report.docs, 1, "the readable file indexed normally");
    }

    #[test]
    fn dropped_blobs_are_counted() {
        let long = "x".repeat(9 * 1024);
        let dir = brain(&[(
            "knowledge/blob.md",
            format!("short intro\n\n{long}\n\nshort outro\n").as_str(),
        )]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.docs, 1);
        assert_eq!(report.blob_dropped, 1);
        // The neighbors of the blob stay recallable.
        assert_eq!(recall(&conn, "intro", 5).unwrap().len(), 1);
        assert_eq!(recall(&conn, "outro", 5).unwrap().len(), 1);
    }

    #[test]
    fn frontmatter_ring_overrides_and_high_rings_are_skipped() {
        let dir = brain(&[
            ("knowledge/promoted.md", "---\nring: 1\n---\nlocked decision here\n"),
            ("knowledge/staged.md", "---\nring: 5\n---\nquarantined candidate\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.docs, 1);
        assert_eq!(report.skipped_high_ring, 1);
        let hits = recall(&conn, "locked decision", 5).unwrap();
        assert_eq!(hits[0].ring, 1);
        assert!(recall(&conn, "quarantined", 5).unwrap().is_empty());
    }

    #[test]
    fn quarantine_locations_ignore_a_self_promoting_frontmatter() {
        // A staging candidate declaring `ring: 2` is a well-formed lie: the
        // location decides for ring-5+ directories, so the file must stay
        // unindexed exactly like one whose frontmatter was stripped.
        let dir = brain(&[
            ("todo/staging/smuggled.md", "---\nring: 2\n---\nself promoted candidate\n"),
            ("knowledge/honest.md", "---\nring: 1\n---\nlegitimate promotion\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.docs, 1, "only the knowledge file is indexed");
        assert_eq!(report.skipped_high_ring, 1, "the staging candidate is skipped, not promoted");
        assert!(recall(&conn, "self promoted", 5).unwrap().is_empty());
        let hits = recall(&conn, "legitimate promotion", 5).unwrap();
        assert_eq!(hits[0].ring, 1, "knowledge-directory promotion keeps working");
    }

    #[test]
    fn a_bom_does_not_hide_the_frontmatter_ring() {
        // Windows editors emit UTF-8 with BOM by default; the BOM must not
        // make the `---` comparison fail-open. A BOM'd `ring: 5` quarantine
        // marker has to quarantine, and a BOM'd `ring: 1` promotion has to
        // promote.
        let dir = brain(&[
            ("todo/staging/bomq.md", "\u{FEFF}---\nring: 5\n---\nzebraxq quarantined payload\n"),
            ("knowledge/bomp.md", "\u{FEFF}---\nring: 1\n---\npluvixq promoted decision\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.docs, 1, "only the promoted knowledge file is indexed");
        assert_eq!(report.skipped_high_ring, 1, "the BOM'd quarantine marker still quarantines");
        assert!(recall(&conn, "zebraxq", 5).unwrap().is_empty());
        assert_eq!(recall(&conn, "pluvixq", 5).unwrap()[0].ring, 1);
    }

    #[test]
    fn an_unreadable_file_is_reported_and_kept_on_incremental_rescan() {
        // Invalid UTF-8 is the reproducible shape of "read failed" (an
        // editor's sharing violation is the transient one). The full scan
        // must REPORT the file instead of silently dropping it, and the
        // incremental rescan (the daemon's steady-state path) must KEEP the
        // last good rows: a one-second editor lock must not delete a
        // document.
        let dir = brain(&[("knowledge/gone.md", "remember the alamo incident\n")]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(recall(&conn, "alamo", 5).unwrap().len(), 1);
        std::fs::write(dir.path().join("knowledge/gone.md"), b"remember \xff\xfe broken").unwrap();

        // Full rebuild: reported, not invisible.
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert!(
            report.unreadable.iter().any(|p| p.contains("gone.md")),
            "unreadable file must be reported: {:?}",
            report.unreadable
        );

        // Incremental rescan of a previously-indexed file that became
        // unreadable: keep the last good content.
        let dir2 = brain(&[("knowledge/gone2.md", "remember the alamo incident\n")]);
        scan(&mut conn, dir2.path(), None, &RingRules::default()).unwrap();
        std::fs::write(dir2.path().join("knowledge/gone2.md"), b"remember \xff\xfe broken").unwrap();
        let report = rescan_changed(&mut conn, dir2.path(), None, &RingRules::default())
            .unwrap()
            .expect("incremental rescan should handle one changed file");
        assert!(report.unreadable.iter().any(|p| p.contains("gone2.md")));
        assert_eq!(
            recall(&conn, "alamo", 5).unwrap().len(),
            1,
            "incremental rescan keeps last good content for an unreadable file"
        );
    }

    #[test]
    fn secrets_and_archive_are_excluded() {
        let dir = brain(&[
            ("mind/secrets/README.md", "index of all tokens\n"),
            ("knowledge/archive/old.md", "retired design\n"),
            ("knowledge/live.md", "live fact\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.docs, 1);
        assert!(recall(&conn, "tokens", 5).unwrap().is_empty());
        assert!(recall(&conn, "retired", 5).unwrap().is_empty());
    }

    #[test]
    fn installed_dependency_trees_are_never_indexed() {
        // A documentation vault built with a static-site generator is an npm
        // project: `npm install` inside the brain drops thousands of package
        // READMEs into it, at arbitrary depth, and every one of them is
        // markdown the extension gate happily takes.
        let dir = brain(&[
            ("node_modules/left-pad/README.md", "hurbeck pads a string\n"),
            ("site/node_modules/dep/README.md", "hurbeck nested copy\n"),
            ("scripts/__pycache__/notes.md", "quelber cached\n"),
            ("knowledge/node_modules-migration.md", "vantrel dropping the dep tree\n"),
            ("knowledge/live.md", "kavender fact\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.docs, 2, "only the two authored knowledge files");
        assert!(recall(&conn, "hurbeck", 5).unwrap().is_empty(), "at any depth");
        assert!(recall(&conn, "quelber", 5).unwrap().is_empty());
        assert_eq!(
            recall(&conn, "vantrel", 5).unwrap().len(),
            1,
            "a note ABOUT the dep tree is not in it: components, not substrings"
        );
        // The watcher derives its subtrees from the directory form, so the
        // two predicates must agree or the daemon watches what it cannot index.
        assert!(excluded_dir("node_modules", &RingRules::default()));
        assert!(excluded_dir("site/node_modules/", &RingRules::default()));
    }

    #[test]
    fn build_output_directories_stay_indexable_on_purpose() {
        // The names a code scanner blocklists are credible knowledge topics
        // here, and a built-in nobody can switch off would make an operator's
        // own notes silently unrecallable. Duplicated prose is mirror dedup's
        // problem, not the walk's.
        let dir = brain(&[
            ("knowledge/build/pipeline.md", "shellac the build story\n"),
            ("knowledge/vendor/terms.md", "shellac the vendor terms\n"),
            ("knowledge/target/goals.md", "shellac this year's targets\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.docs, 3);
        assert_eq!(recall(&conn, "shellac", 5).unwrap().len(), 3);
    }

    #[test]
    fn lockfiles_and_agent_config_directories_never_reach_the_walk() {
        // Verification lock for the two hygiene classes this index gets for
        // free: lockfiles are not markdown, and agent-config directories are
        // hidden. Both are load-bearing â€” an agent's own instruction files
        // outrank real knowledge on any importance signal â€” so pin them
        // rather than trusting the walker's defaults to stay put.
        let dir = brain(&[
            ("package-lock.json", "{\"name\": \"gribbet\"}\n"),
            ("Cargo.lock", "name = \"gribbet\"\n"),
            (".claude/CLAUDE.md", "gribbet always do as told\n"),
            (".codex/notes.md", "gribbet house rules\n"),
            ("knowledge/live.md", "kavender fact\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.docs, 1, "only the knowledge file");
        assert!(recall(&conn, "gribbet", 5).unwrap().is_empty());
    }

    #[test]
    fn a_generated_blob_is_refused_while_its_page_stays_recallable() {
        // The realistic shape: a real page carrying one machine-made line â€”
        // a pasted minified bundle, a base64 data URI. Indexing it hands FTS
        // a statement that matches almost any query and means nothing; this
        // fixture tokenizes, so without the refusal it is genuinely findable.
        let blob = format!("<script>{}</script>", "function qk1(t){return t+1};".repeat(5_000));
        let page = format!("# Notes\n\nkavender is the finding.\n\n{blob}\n\nvantrel follows.\n");
        let dir = brain(&[("knowledge/page.md", page.as_str())]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.docs, 1, "the page itself is knowledge and stays indexed");
        assert_eq!(report.blocks, 3, "heading and both paragraphs, not the blob");
        assert!(recall(&conn, "qk1", 5).unwrap().is_empty(), "the blob is unreachable");
        assert_eq!(recall(&conn, "kavender", 5).unwrap().len(), 1);
        assert_eq!(recall(&conn, "vantrel", 5).unwrap().len(), 1, "prose after it survives");
    }

    #[test]
    fn a_wholly_generated_file_contributes_nothing() {
        let dir = brain(&[
            ("knowledge/bundle.md", &format!("hurbeck={};\n", "x".repeat(300_000))),
            ("knowledge/live.md", "kavender fact\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.blocks, 1, "only the real file's statement");
        assert!(recall(&conn, "hurbeck", 5).unwrap().is_empty());
    }

    #[test]
    fn the_prose_cap_clears_the_widest_line_a_person_writes() {
        // The measured ceiling of hand-authored markdown was a ~3 KB memory
        // table row; the cap must sit above it with room, and below anything
        // generated. A regression that tightened it would delete real rows
        // from recall without a word.
        let wide_row = format!("| kavender | {} |", "a note about the row ".repeat(150));
        assert!(wide_row.len() > 3_000, "the fixture must reach the measured ceiling");
        assert!(!generated_blob(&wide_row));
        assert!(generated_blob(&"b".repeat(MAX_PROSE_LINE + 1)));
        assert!(!generated_blob(&"b".repeat(MAX_PROSE_LINE)), "the cap itself is prose");
        assert!(
            !generated_blob(&format!("{}\n{}", "c".repeat(MAX_PROSE_LINE), "d".repeat(MAX_PROSE_LINE))),
            "a long block of ordinary lines is prose; only a single long LINE is not"
        );
    }

    #[test]
    fn native_auto_memory_is_indexed_as_ring2() {
        let brain = brain(&[("knowledge/a.md", "brain fact\n")]);
        let native = tempfile::tempdir().unwrap();
        let mem = native.path().join("-home-user/memory");
        std::fs::create_dir_all(&mem).unwrap();
        std::fs::write(mem.join("MEMORY.md"), "# Memory\n\n- [zvol trap](f.md) nossd required\n").unwrap();
        std::fs::write(
            mem.join("feedback_zvol.md"),
            "---\nname: feedback_zvol\ndescription: x\n---\nzvol on btrfs needs nossd mount option\n",
        )
        .unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, brain.path(), Some(native.path()), &RingRules::default()).unwrap();
        assert_eq!(report.docs, 3);
        let hits = recall(&conn, "nossd", 5).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|h| h.ring == 2));
        assert!(hits.iter().any(|h| h.path == "native:-home-user/feedback_zvol.md"));
        // frontmatter of native files has no `ring:` â€” must not shift line numbers wrongly
        let fb = hits.iter().find(|h| h.path.ends_with("feedback_zvol.md")).unwrap();
        assert_eq!(fb.start_line, 5);
    }

    #[test]
    fn native_staleness_is_tracked() {
        let brain = brain(&[("knowledge/a.md", "alpha\n")]);
        let native = tempfile::tempdir().unwrap();
        let mem = native.path().join("p1/memory");
        std::fs::create_dir_all(&mem).unwrap();
        std::fs::write(mem.join("MEMORY.md"), "first\n").unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, brain.path(), Some(native.path()), &RingRules::default()).unwrap();
        assert!(!stale(&conn, brain.path(), Some(native.path()), &RingRules::default()).unwrap());
        std::fs::write(mem.join("MEMORY.md"), "second, and quite a bit longer\n").unwrap();
        assert!(stale(&conn, brain.path(), Some(native.path()), &RingRules::default()).unwrap());
    }

    #[test]
    fn missing_native_root_is_fine() {
        let brain = brain(&[("knowledge/a.md", "alpha\n")]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let absent = brain.path().join("no-such-dir");
        let report = scan(&mut conn, brain.path(), Some(&absent), &RingRules::default()).unwrap();
        assert_eq!(report.docs, 1);
        assert!(!stale(&conn, brain.path(), Some(&absent), &RingRules::default()).unwrap());
    }

    #[test]
    fn malformed_ring_frontmatter_fails_closed() {
        let dir = brain(&[
            ("knowledge/bad.md", "---\nring: banana\n---\nzweptahl must stay hidden\n"),
            ("knowledge/spaced.md", "---\nRing: 1 # promoted\n---\nquorvex is promoted\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.skipped_high_ring, 1);
        assert!(recall(&conn, "zweptahl", 5).unwrap().is_empty());
        let hits = recall(&conn, "quorvex", 5).unwrap();
        assert_eq!(hits[0].ring, 1, "case-insensitive key + trailing token tolerated");
    }

    #[test]
    fn native_files_cannot_promote_above_ring2() {
        let brain_dir = brain(&[("knowledge/a.md", "x\n")]);
        let native = tempfile::tempdir().unwrap();
        let mem = native.path().join("p/memory");
        std::fs::create_dir_all(&mem).unwrap();
        std::fs::write(mem.join("sneaky.md"), "---\nring: 0\n---\ni claim to be an invariant\n").unwrap();
        std::fs::write(mem.join("quarantined.md"), "---\nring: 5\n---\nhidden\n").unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, brain_dir.path(), Some(native.path()), &RingRules::default()).unwrap();
        let hits = recall(&conn, "invariant", 5).unwrap();
        assert_eq!(hits[0].ring, 2, "promotion clamped to the store's contract ring");
        assert_eq!(report.skipped_high_ring, 1, "demotion to 5+ is honored");
    }

    #[test]
    fn native_index_pointers_become_edges_in_both_directions() {
        // The native store's index is a hook table: one line per entry,
        // pointing at the file that explains it. The pointers are markdown
        // links, which the vault's wikilink reader cannot see â€” so a hit on a
        // hook used to be a dead end, with the hook itself all the reader got.
        let brain = brain(&[("knowledge/a.md", "brain fact\n")]);
        let native = tempfile::tempdir().unwrap();
        let mem = native.path().join("proj-a/memory");
        std::fs::create_dir_all(&mem).unwrap();
        std::fs::write(
            mem.join("MEMORY.md"),
            "# Memory\n\n- [zvol trap](feedback_zvol.md) nossd required\n",
        )
        .unwrap();
        std::fs::write(mem.join("feedback_zvol.md"), "zvol on btrfs needs nossd\n").unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, brain.path(), Some(native.path()), &RingRules::default()).unwrap();

        let index = "native:proj-a/MEMORY.md".to_string();
        let entry = "native:proj-a/feedback_zvol.md".to_string();
        assert_eq!(
            linked_docs(&conn, std::slice::from_ref(&index), 8).unwrap(),
            vec![(entry.clone(), 2)],
            "the index descends to the file its pointer names"
        );
        assert_eq!(
            linked_docs(&conn, &[entry], 8).unwrap(),
            vec![(index, 2)],
            "and an entry names the index it hangs under"
        );
    }

    #[test]
    fn a_native_pointer_resolves_inside_its_own_project() {
        // Each project's store names the same generic files, so a vault-wide
        // stem lookup sees two candidates and gives up: without relative
        // resolution the pointers are dead on any host with more than one
        // project, which is every host.
        let brain = brain(&[("knowledge/a.md", "brain fact\n")]);
        let native = tempfile::tempdir().unwrap();
        for slug in ["proj-a", "proj-b"] {
            let mem = native.path().join(slug).join("memory");
            std::fs::create_dir_all(&mem).unwrap();
            std::fs::write(mem.join("MEMORY.md"), "- [rule](feedback_rule.md)\n").unwrap();
            std::fs::write(mem.join("feedback_rule.md"), format!("the rule of {slug}\n")).unwrap();
        }
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, brain.path(), Some(native.path()), &RingRules::default()).unwrap();

        assert_eq!(
            linked_docs(&conn, &["native:proj-a/MEMORY.md".to_string()], 8).unwrap(),
            vec![("native:proj-a/feedback_rule.md".to_string(), 2)],
            "the sibling entry, never the other project's file of the same name"
        );
    }

    #[test]
    fn the_vault_graph_stays_wikilinks_only() {
        // Reading markdown links in the brain too would silently add edges
        // brain-lint does not see, to the graph this project trusts most.
        let dir = brain(&[
            ("knowledge/zfs.md", "see [the shares doc](hosts/shares.md)\n"),
            ("knowledge/hosts/shares.md", "SMB share layout\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert!(
            linked_docs(&conn, &["knowledge/zfs.md".to_string()], 8).unwrap().is_empty(),
            "the vault's edges are its curated wikilinks, nothing else"
        );
    }

    #[test]
    fn markdown_link_extraction_keeps_only_doc_pointers() {
        let links = markdown_links(concat!(
            "- [Alias](Feedback_One.md \"title\") and [anchored](two.md#section)\n",
            "- [relative](./three.md), [web](https://example.com/), ![shot](x.png)\n",
            "```\n[fenced](never.md)\n```\n",
        ));
        assert_eq!(links, vec!["feedback_one.md", "two.md", "three.md"]);
    }

    #[test]
    fn skipped_file_changes_still_flip_staleness() {
        let dir = brain(&[
            ("knowledge/a.md", "visible\n"),
            ("knowledge/staged.md", "---\nring: 5\n---\nv1\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert!(!stale(&conn, dir.path(), None, &RingRules::default()).unwrap());
        // Editing the SKIPPED file (e.g. removing its ring-5 marker = promotion)
        // must be noticed â€” the old subset comparison was blind to this.
        std::fs::write(dir.path().join("knowledge/staged.md"), "now public\n").unwrap();
        assert!(stale(&conn, dir.path(), None, &RingRules::default()).unwrap());
    }

    #[test]
    fn long_fence_containing_short_fence_does_not_close_early() {
        let text = "````\ncode\n```\nstill code\n````\n\nafter\n";
        let blocks = segment(text, 0);
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].2.contains("still code"));
        assert_eq!(blocks[1].2, "after");
    }

    #[test]
    fn generation_advances_with_every_scan_and_persists() {
        let dir = brain(&[("knowledge/a.md", "alpha\n")]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        assert_eq!(generation(&conn), 0, "unscanned index has generation 0");
        let r1 = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(r1.generation, 1);
        assert_eq!(generation(&conn), 1);
        let r2 = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(r2.generation, 2, "generation is monotonic per committed scan");
        drop(conn);
        let conn = open(state.path()).unwrap();
        assert_eq!(generation(&conn), 2, "generation is persisted in index meta");
    }

    #[test]
    fn catalog_checksum_is_a_pure_function_of_the_tree() {
        let dir = brain(&[
            ("knowledge/a.md", "alpha fact\n\nbeta fact\n"),
            ("knowledge/b.md", "- gamma\n- delta\n"),
        ]);
        // Two independent derivations (separate state dirs) over the same tree.
        let s1 = tempfile::tempdir().unwrap();
        let s2 = tempfile::tempdir().unwrap();
        let mut c1 = open(s1.path()).unwrap();
        let mut c2 = open(s2.path()).unwrap();
        scan(&mut c1, dir.path(), None, &RingRules::default()).unwrap();
        scan(&mut c2, dir.path(), None, &RingRules::default()).unwrap();
        // Different generations must not leak into the digest.
        scan(&mut c2, dir.path(), None, &RingRules::default()).unwrap();
        let k1 = catalog_checksum(&c1).unwrap();
        let k2 = catalog_checksum(&c2).unwrap();
        assert!(!k1.is_empty());
        assert_eq!(k1, k2, "same tree must yield the same catalog checksum");
        // A content change must change the digest.
        std::fs::write(dir.path().join("knowledge/b.md"), "- gamma\n- delta prime\n").unwrap();
        scan(&mut c1, dir.path(), None, &RingRules::default()).unwrap();
        assert_ne!(catalog_checksum(&c1).unwrap(), k2);
    }

    #[test]
    fn a_slice_checksum_cannot_observe_changes_outside_its_paths() {
        let dir = brain(&[
            ("knowledge/shared/a.md", "visible fact\n"),
            ("knowledge/private/b.md", "private fact\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let shared = |path: &str| path == "knowledge/shared" || path.starts_with("knowledge/shared/");
        let before = catalog_checksum_matching(&conn, shared).unwrap();
        std::fs::write(dir.path().join("knowledge/private/b.md"), "changed private fact\n").unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(catalog_checksum_matching(&conn, shared).unwrap(), before);
        std::fs::write(dir.path().join("knowledge/shared/a.md"), "changed visible fact\n").unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_ne!(catalog_checksum_matching(&conn, shared).unwrap(), before);
    }

    #[test]
    fn open_ro_serves_the_committed_snapshot() {
        let dir = brain(&[("knowledge/a.md", "royw fact\n")]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let ro = open_ro(state.path()).unwrap();
        assert_eq!(recall(&ro, "royw", 5).unwrap().len(), 1);
        assert!(
            ro.execute("INSERT INTO meta(key,value) VALUES('x','y')", []).is_err(),
            "read-only handle must not be able to write"
        );
    }

    #[test]
    fn normalized_body_schema_is_rejected_read_only_and_rebuilt_from_markdown() {
        let dir = brain(&[("knowledge/a.md", "- US\n")]);
        let state = tempfile::tempdir().unwrap();
        let spec = test_spec(2);
        let legacy_hash = crate::hashing::hex_lower(sha2::Sha256::digest(b"- us"));
        {
            let mut conn = open(state.path()).unwrap();
            scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
            conn.execute(
                "UPDATE blocks SET hash=?1, cite=?2",
                rusqlite::params![legacy_hash, cite_from_hash(3, &legacy_hash)],
            )
            .unwrap();
            ensure_vector_spec(&conn, &spec).unwrap();
            insert_vector(&conn, &legacy_hash, &spec, &[1.0, 0.0]).unwrap();
            conn.execute("INSERT INTO meta(key,value) VALUES('marker','old')", []).unwrap();
            conn.pragma_update(None, "user_version", 7i64).unwrap();

            for result in [open_ro(state.path()), open_read_only(state.path())] {
                let error = result.expect_err("read-only opens must reject normalized-body schema");
                assert!(error.to_string().contains("index schema v7 != v8"), "{error}");
            }
            let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
            assert_eq!(version, 7, "read-only opens do not migrate the catalog");
            assert_eq!(vector_coverage(&conn, &spec).unwrap(), (1, 1));
        }
        let conn = ensure_fresh(state.path(), dir.path(), None, &RingRules::default()).unwrap();
        let v: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        let marker: Option<String> = conn
            .query_row("SELECT value FROM meta WHERE key='marker'", [], |r| r.get(0))
            .ok();
        assert!(marker.is_none(), "old-schema DB must be discarded, not reused");
        let hits = recall(&conn, "US", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].cite, cite_id(3, "- US"));
        assert!(expand(&conn, &cite_from_hash(3, &legacy_hash)).unwrap().is_empty());
        assert_eq!(vector_coverage(&conn, &spec).unwrap(), (0, 1));
        assert_eq!(
            hashes_without_vectors(&conn, &spec, 10).unwrap(),
            vec![(content_hash("- US"), "- US".to_string())]
        );
    }

    #[test]
    fn wikilink_extraction_handles_alias_and_anchor() {
        let links = wikilinks("see [[Zfs-Dataset|the dataset doc]] and [[shares#SMB]] but not [[]]");
        assert_eq!(links, vec!["zfs-dataset", "shares"]);
    }

    #[test]
    fn recall_expansion_follows_curated_links_both_directions() {
        let dir = brain(&[
            ("knowledge/zfs.md", "pools and datasets, see [[shares]]\n"),
            ("knowledge/hosts/shares.md", "SMB share layout\n"),
            ("knowledge/backup.md", "backups mirror [[zfs]] snapshots\n"),
            ("knowledge/unrelated.md", "nothing here\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let linked = linked_docs(&conn, &["knowledge/zfs.md".to_string()], 8).unwrap();
        let paths: Vec<&str> = linked.iter().map(|(p, _)| p.as_str()).collect();
        assert!(paths.contains(&"knowledge/hosts/shares.md"), "outgoing link followed");
        assert!(paths.contains(&"knowledge/backup.md"), "incoming link followed");
        assert!(!paths.contains(&"knowledge/unrelated.md"));
    }

    #[test]
    fn ambiguous_wikilink_stems_are_skipped() {
        let dir = brain(&[
            ("knowledge/a/readme2.md", "x\n"),
            ("knowledge/b/readme2.md", "y\n"),
            ("knowledge/linker.md", "see [[readme2]]\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let linked = linked_docs(&conn, &["knowledge/linker.md".to_string()], 8).unwrap();
        assert!(linked.is_empty(), "ambiguous stem must create no edge");
    }

    #[test]
    fn f32_blob_roundtrip_is_little_endian() {
        let v = vec![0.25f32, -1.5, 3.0];
        let blob = vec_to_blob(&v, Precision::F32);
        assert_eq!(blob.len(), 12);
        assert_eq!(&blob[0..4], &0.25f32.to_le_bytes());
        assert_eq!(blob_to_vec(&blob, Precision::F32), v);
        // corrupt trailing partial chunk is dropped, not misread
        assert_eq!(blob_to_vec(&blob[..10], Precision::F32), vec![0.25f32, -1.5]);
        // f16 is little-endian too, and exact for these values
        let blob = vec_to_blob(&v, Precision::F16);
        assert_eq!(blob.len(), 6);
        assert_eq!(blob_to_vec(&blob, Precision::F16), v);
    }

    #[test]
    fn cosine_on_known_vectors() {
        let mut a = vec![3.0f32, 4.0];
        l2_normalize(&mut a);
        assert!((a[0] - 0.6).abs() < 1e-6);
        assert!((a[1] - 0.8).abs() < 1e-6);
        assert!((dot(&a, &a) - 1.0).abs() < 1e-6);
        let mut x = vec![1.0f32, 0.0];
        let mut y = vec![0.0f32, 1.0];
        l2_normalize(&mut x);
        l2_normalize(&mut y);
        assert!(dot(&x, &y).abs() < 1e-6, "orthogonal vectors have cosine 0");
        let mut d = vec![1.0f32, 1.0];
        l2_normalize(&mut d);
        assert!((dot(&d, &x) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        // zero vector: normalization must not divide by zero
        let mut z = vec![0.0f32, 0.0];
        l2_normalize(&mut z);
        assert_eq!(z, vec![0.0, 0.0]);
    }

    #[test]
    fn rrf_fusion_on_hand_built_rankings_k2() {
        // list 1: a b c   list 2: c a d
        // K=2 scores: a = 1/3 + 1/4 = 0.583, c = 1/5 + 1/3 = 0.533,
        //             b = 1/4 = 0.25,        d = 1/5 = 0.2
        let fused = rrf_fuse(&[vec![1, 2, 3], vec![3, 1, 4]], 2.0);
        assert_eq!(fused, vec![1, 3, 2, 4]);
    }

    #[test]
    fn fts_query_keeps_the_compound_as_a_phrase_and_splits_for_recall() {
        // An id lookup: the compound is the signal...
        assert_eq!(fts_query("bug-400"), "\"bug-400\" OR \"bug\"* OR \"400\"*");
        // ...while the recall contract survives: a hyphenated concept still
        // matches its words scattered across a sentence.
        assert_eq!(fts_query("state-machine"), "\"state-machine\" OR \"state\"* OR \"machine\"*");
        // Plain terms are unchanged prefix terms.
        assert_eq!(fts_query("database migration"), "\"database\"* OR \"migration\"*");
        // Duplicate terms are not double-counted.
        assert_eq!(fts_query("bug bug-400"), "\"bug\"* OR \"bug-400\" OR \"400\"*");
    }

    #[test]
    fn an_id_lookup_ranks_the_id_note_first_even_among_prefix_traps() {
        // The shape of the reported failure, scaled down: hundreds of bug
        // notes flatten the term `bug`, and prefix matching makes `400` hit
        // 400002, 17.400 and 86_400_000 elsewhere in the tree. The note
        // carrying the literal compound must come back first.
        let brain = tempfile::tempdir().unwrap();
        let bugs = brain.path().join("knowledge/bugs");
        std::fs::create_dir_all(&bugs).unwrap();
        for i in 1..=60 {
            let id = format!("bug-{i:03}");
            std::fs::write(
                bugs.join(format!("{id}.md")),
                format!("# {id} â€” failure number {i}\n\nroot cause and fix for {id}\n"),
            )
            .unwrap();
        }
        // The id under test: round number, far outside the sequential range,
        // so nothing else collides with `400` except the traps below.
        std::fs::write(
            bugs.join("bug-400.md"),
            "# bug-400 â€” frames decoded to noise\n\nroot cause and fix for bug-400\n",
        )
        .unwrap();
        std::fs::write(
            brain.path().join("knowledge/budget.md"),
            "# Budget\n\n- allocation 400002 units\n- ratio 17.400 measured\n- seconds 86_400_000 total\n",
        )
        .unwrap();
        std::fs::write(
            brain.path().join("knowledge/log.md"),
            "# Log\n\n- see bug-225 and bug-400 in the bug list\n",
        )
        .unwrap();

        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, brain.path(), None, &RingRules::default()).unwrap();

        let hits = recall(&conn, "bug-400", 8).unwrap();
        assert!(!hits.is_empty(), "the id note must be findable at all");
        assert_eq!(
            hits[0].path, "knowledge/bugs/bug-400.md",
            "the literal compound outranks everything; got: {}",
            hits.iter().map(|h| h.path.clone()).collect::<Vec<_>>().join(", ")
        );
        // The recall contract holds on the same shape: any other bug note is
        // found by its id too, not just the one the phrase boosted hardest.
        let scatter = recall(&conn, "bug-060", 8).unwrap();
        assert!(scatter.iter().any(|h| h.path == "knowledge/bugs/bug-060.md"));
    }

    #[test]
    fn rrf_ties_break_by_id_deterministically() {
        // two disjoint single-item lists: equal scores, id order decides
        let fused = rrf_fuse(&[vec![9], vec![4]], 2.0);
        assert_eq!(fused, vec![4, 9]);
    }

    fn test_spec(dim: usize) -> VectorSpec {
        VectorSpec {
            network_major: 1,
            profile_id: "test-profile".into(),
            model: "test-model".into(),
            dim,
            precision: Precision::I8,
            doc_prefix: String::new(),
        }
    }

    fn embed_everything(conn: &Connection, spec: &VectorSpec, vector: &[f32]) {
        for (hash, _) in hashes_without_vectors(conn, spec, 1000).unwrap() {
            insert_vector(conn, &hash, spec, vector).unwrap();
        }
    }

    #[test]
    fn content_hash_is_the_full_digest_the_citation_truncates() {
        // ONE hashing site: the citation is a prefix of the content address,
        // so a vector keyed by hash always belongs to the cited block.
        let h = content_hash("The  Quick   Fox");
        assert_eq!(h.len(), 64, "full sha256 hex");
        assert_ne!(h, content_hash("the quick fox"), "embedding input is case and whitespace sensitive");
        assert_eq!(cite_id(3, "The  Quick   Fox"), format!("r3-{}", &h[..10]));
        assert_eq!(cite_id(1, "The  Quick   Fox"), format!("r1-{}", &h[..10]), "ring labels, hash addresses");
        assert_ne!(h, content_hash("the quick foxes"));
    }

    #[test]
    fn vector_queue_preserves_case_spacing_and_line_breaks_but_deduplicates_exact_copies() {
        let dir = brain(&[
            ("knowledge/upper.md", "- US\n"),
            ("knowledge/lower.md", "- us\n"),
            ("knowledge/spaced.md", "- us  value\n"),
            ("knowledge/single.md", "- us value\n"),
            ("knowledge/tab.md", "- us\tvalue\n"),
            ("knowledge/lines.md", "US\nstate\n"),
            ("knowledge/inline.md", "US state\n"),
            ("knowledge/copy.md", "---\nring: 1\n---\n- US\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let spec = test_spec(2);
        let pending: std::collections::BTreeMap<String, String> =
            hashes_without_vectors(&conn, &spec, 20).unwrap().into_iter().collect();
        let bodies = [
            "- US", "- us", "- us  value", "- us value", "- us\tvalue", "US\nstate", "US state",
        ];
        assert_eq!(pending.len(), bodies.len());
        for body in bodies {
            assert_eq!(pending.get(&content_hash(body)).map(String::as_str), Some(body));
        }
        insert_vector(&conn, &content_hash("- US"), &spec, &[1.0, 0.0]).unwrap();
        assert_eq!(
            vector_coverage(&conn, &spec).unwrap(),
            (2, 8),
            "identical copies share one artifact across rings"
        );
        assert_eq!(hashes_without_vectors(&conn, &spec, 20).unwrap().len(), 6);
        let copies = expand(&conn, &cite_id(3, "- US")).unwrap();
        assert_eq!(copies.len(), 2);
        assert_eq!(copies[0].ring, 1);
        assert_eq!(copies[1].ring, 3);
    }

    #[test]
    fn exact_body_edits_invalidate_only_the_changed_vector_on_full_and_incremental_scans() {
        for incremental in [false, true] {
            let dir = brain(&[("knowledge/a.md", "- US\n- unchanged\n")]);
            let path = dir.path().join("knowledge/a.md");
            let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
            let state = tempfile::tempdir().unwrap();
            let mut conn = open(state.path()).unwrap();
            scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
            let spec = test_spec(2);
            embed_everything(&conn, &spec, &[1.0, 0.0]);
            let mut previous = "- US";
            for (step, body) in ["- us", "- us  ", "- us\t ", "- us\n  value"].into_iter().enumerate() {
                std::fs::write(&path, format!("{body}\n- unchanged\n")).unwrap();
                // Make same-size edits visible on filesystems with coarse mtimes.
                std::fs::File::options()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_times(std::fs::FileTimes::new().set_modified(
                        modified + std::time::Duration::from_secs(step as u64 + 1),
                    ))
                    .unwrap();
                if incremental {
                    rescan_changed(&mut conn, dir.path(), None, &RingRules::default()).unwrap().unwrap();
                } else {
                    scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
                }
                assert_eq!(
                    hashes_without_vectors(&conn, &spec, 10).unwrap(),
                    vec![(content_hash(body), body.to_string())],
                    "only the exact edited body is queued, incremental={incremental}"
                );
                assert_eq!(vector_coverage(&conn, &spec).unwrap(), (1, 2));
                assert!(expand(&conn, &cite_id(3, previous)).unwrap().is_empty());
                let rows: i64 = conn.query_row("SELECT count(*) FROM vectors", [], |r| r.get(0)).unwrap();
                assert_eq!(rows, 1, "the superseded vector is pruned");
                embed_everything(&conn, &spec, &[0.0, 1.0]);
                previous = body;
            }
            // Moving an unchanged body does not change its address or requeue it.
            std::fs::write(&path, format!("- unchanged\n{previous}\n")).unwrap();
            scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
            assert!(hashes_without_vectors(&conn, &spec, 10).unwrap().is_empty());
            assert_eq!(expand(&conn, &cite_id(3, previous)).unwrap().len(), 1);
        }
    }

    #[test]
    fn f16_conversion_matches_the_ieee_oracle_bit_for_bit() {
        // Expected bit patterns produced by an INDEPENDENT implementation
        // (CPython's struct "e" format, IEEE binary16, round-to-nearest-even)
        // â€” this is the check that the hand-written narrowing is a real f16
        // and not merely self-consistent. Subnormals, the rounding floor
        // (2.98e-8 ties to zero, 5.96e-8 to the smallest subnormal), the
        // largest finite half, and the awkward decimal fractions are all in.
        let cases: [(f32, u16); 25] = [
            (0.0e+00, 0x0000),
            (-0.0e+00, 0x8000),
            (1.0e+00, 0x3c00),
            (-1.0e+00, 0xbc00),
            (5.0e-01, 0x3800),
            (-3.125e-02, 0xa800),
            (1.0e-04, 0x068e),
            (-1.0e-04, 0x868e),
            (6.0e-08, 0x0001),
            (5.9604645e-08, 0x0001),
            (2.9802322e-08, 0x0000),
            (9.995117e-01, 0x3bff),
            (3.3333334e-01, 0x3555),
            (6.5504e+04, 0x7bff),
            (-6.5504e+04, 0xfbff),
            (6.1e-05, 0x03ff),
            (6.0e-05, 0x03ef),
            (1.0e-07, 0x0002),
            (1.0e-01, 0x2e66),
            (2.0e-01, 0x3266),
            (7.0710677e-01, 0x39a8),
            (9.765625e-04, 0x1400),
            (1.0009766e+00, 0x3c01),
            (2.4414062e-04, 0x0c00),
            (1.1e-05, 0x00b9),
        ];
        for (value, expected) in cases {
            let got = f32_to_f16(value);
            assert_eq!(got, expected, "f32_to_f16({value:e}) = {got:#06x}, expected {expected:#06x}");
            // And back: widening a half is exact, so the round trip is the
            // half's own value, sign of zero included.
            let back = f16_to_f32(got);
            assert_eq!(
                back.to_bits(),
                f16_to_f32(expected).to_bits(),
                "widening {expected:#06x} disagreed"
            );
        }
        // Values beyond the half range saturate to infinity rather than
        // wrapping into a finite lie.
        assert_eq!(f32_to_f16(1e30), 0x7c00);
        assert_eq!(f32_to_f16(-1e30), 0xfc00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn f16_blobs_halve_the_bytes_and_round_trip_within_tolerance() {
        let v: Vec<f32> = (0..8).map(|i| (i as f32 - 3.5) / 16.0).collect();
        let blob = vec_to_blob(&v, Precision::F16);
        assert_eq!(blob.len(), v.len() * 2, "half floats, half the bytes");
        for (a, b) in v.iter().zip(blob_to_vec(&blob, Precision::F16).iter()) {
            assert!((a - b).abs() < 1e-3, "{a} -> {b}");
        }
        // f32 stays available and stays exact.
        let blob32 = vec_to_blob(&v, Precision::F32);
        assert_eq!(blob32.len(), v.len() * 4);
        assert_eq!(blob_to_vec(&blob32, Precision::F32), v);
        // The values a normalized component actually reaches, plus the edges.
        for x in [0.0f32, 1.0, -1.0, 0.5, -0.03125, 1e-4, -1e-4, 6e-8] {
            let back = blob_to_vec(&vec_to_blob(&[x], Precision::F16), Precision::F16)[0];
            assert!((back - x).abs() <= 1e-3 * x.abs().max(1e-3), "{x} -> {back}");
        }
        // What actually matters: the ranking score survives the narrowing.
        let mut a: Vec<f32> = (0..1024).map(|i| ((i % 17) as f32 - 8.0) / 9.0).collect();
        let mut b: Vec<f32> = (0..1024).map(|i| ((i % 23) as f32 - 11.0) / 12.0).collect();
        l2_normalize(&mut a);
        l2_normalize(&mut b);
        let exact = dot(&a, &b);
        let narrowed = dot(
            &blob_to_vec(&vec_to_blob(&a, Precision::F16), Precision::F16),
            &blob_to_vec(&vec_to_blob(&b, Precision::F16), Precision::F16),
        );
        assert!((exact - narrowed).abs() < 1e-3, "cosine {exact} vs {narrowed}");
    }

    #[test]
    fn corrupt_blobs_are_dropped_never_misread() {
        // A trailing partial component means the blob is not what it claims.
        assert!(blob_to_vec(&[0u8; 3], Precision::F16).len() == 1);
        assert!(blob_to_vec(&[0u8; 5], Precision::F32).len() == 1);
    }

    #[test]
    fn int8_insert_uses_the_exact_shared_record_codec() {
        let dir = brain(&[("knowledge/a.md", "- one\n")]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let spec = test_spec(2);
        let missing = hashes_without_vectors(&conn, &spec, 10).unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].1, "- one");
        assert_eq!(missing[0].0, content_hash("- one"), "the queue is keyed by content address");
        insert_vector(&conn, &missing[0].0, &spec, &[3.0, 4.0]).unwrap();
        let blob: Vec<u8> = conn
            .query_row("SELECT embedding FROM vectors WHERE content_hash=?1", [&missing[0].0], |r| r.get(0))
            .unwrap();
        assert_eq!(
            blob,
            vec_to_blob(&[3.0, 4.0], Precision::I8),
            "the disposable cache must preserve the authoritative shared bytes"
        );
        let stored = blob_to_vec(&blob, Precision::I8);
        assert!((stored[0] - 0.6).abs() < 2e-3, "decoded view is normalized");
        assert!((stored[1] - 0.8).abs() < 2e-3);
        assert!(hashes_without_vectors(&conn, &spec, 10).unwrap().is_empty());
        assert_eq!(vector_coverage(&conn, &spec).unwrap(), (1, 1));
    }

    #[test]
    fn a_spec_change_drops_vectors_and_a_repeat_keeps_them() {
        let dir = brain(&[("knowledge/a.md", "- one\n- two\n")]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let spec = test_spec(2);
        assert!(!ensure_vector_spec(&conn, &spec).unwrap(), "first run: nothing to drop");
        embed_everything(&conn, &spec, &[1.0, 0.0]);
        assert_eq!(vector_coverage(&conn, &spec).unwrap(), (2, 2));
        assert!(!ensure_vector_spec(&conn, &spec).unwrap(), "same spec keeps vectors");
        assert_eq!(vector_coverage(&conn, &spec).unwrap(), (2, 2));

        let other_model = VectorSpec { model: "other".into(), ..spec.clone() };
        assert!(ensure_vector_spec(&conn, &other_model).unwrap(), "model change drops");
        assert_eq!(vector_coverage(&conn, &other_model).unwrap(), (0, 2));
        embed_everything(&conn, &other_model, &[1.0, 0.0]);

        let other_dim = VectorSpec { dim: 4, ..other_model.clone() };
        assert!(ensure_vector_spec(&conn, &other_dim).unwrap(), "dimension change drops");
        embed_everything(&conn, &other_dim, &[1.0, 0.0, 0.0, 0.0]);

        let other_precision = VectorSpec { precision: Precision::F32, ..other_dim.clone() };
        assert!(ensure_vector_spec(&conn, &other_precision).unwrap(), "precision change drops");
        assert_eq!(vector_coverage(&conn, &other_precision).unwrap(), (0, 2));
        assert_eq!(stored_vector_spec(&conn), Some(other_precision));
    }

    #[test]
    fn rescan_preserves_vectors_for_unchanged_blocks() {
        // THE regression: vectors used to be keyed by a volatile rowid, so
        // every scan wiped the whole table and any markdown edit destroyed
        // 100% of the embeddings. Content addressing makes an edit cost
        // exactly the blocks that changed.
        let dir = brain(&[("knowledge/a.md", "- one\n- two\n")]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let spec = test_spec(2);
        ensure_vector_spec(&conn, &spec).unwrap();
        embed_everything(&conn, &spec, &[1.0, 0.0]);
        assert_eq!(vector_coverage(&conn, &spec).unwrap(), (2, 2));

        std::fs::write(dir.path().join("knowledge/a.md"), "- one\n- two\n- three\n").unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(
            vector_coverage(&conn, &spec).unwrap(),
            (2, 3),
            "a full rebuild keeps every unchanged block's vector"
        );
        let missing = hashes_without_vectors(&conn, &spec, 10).unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].1, "- three", "only the new block is queued");

        // The incremental path must agree â€” it is the daemon's hot path.
        embed_everything(&conn, &spec, &[0.0, 1.0]);
        std::fs::write(dir.path().join("knowledge/a.md"), "- one\n- two\n- three\n- four\n").unwrap();
        rescan_changed(&mut conn, dir.path(), None, &RingRules::default())
            .unwrap()
            .expect("small diff -> incremental");
        assert_eq!(vector_coverage(&conn, &spec).unwrap(), (3, 4));
    }

    #[test]
    fn an_edited_block_re_enters_the_queue_and_its_stale_vector_is_pruned() {
        let dir = brain(&[("knowledge/a.md", "- one\n- two\n")]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let spec = test_spec(2);
        embed_everything(&conn, &spec, &[1.0, 0.0]);
        let gone = content_hash("- two");

        std::fs::write(dir.path().join("knowledge/a.md"), "- one\n- two, corrected\n").unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let missing = hashes_without_vectors(&conn, &spec, 10).unwrap();
        assert_eq!(missing.len(), 1, "the edited block is queued again");
        assert_eq!(missing[0].1, "- two, corrected");
        assert_eq!(vector_coverage(&conn, &spec).unwrap(), (1, 2));
        let rows: i64 = conn.query_row("SELECT count(*) FROM vectors", [], |r| r.get(0)).unwrap();
        assert_eq!(rows, 1, "the vector of a hash no longer present is pruned");
        assert!(
            conn.query_row("SELECT 1 FROM vectors WHERE content_hash=?1", [&gone], |r| r.get::<_, i64>(0))
                .is_err(),
            "the superseded vector is gone, not orphaned"
        );
    }

    #[test]
    fn semantic_recall_ranks_by_cosine() {
        let dir = brain(&[
            ("knowledge/zfs.md", "pools and datasets\n"),
            ("knowledge/mail.md", "stalwart smtp\n"),
            ("knowledge/unembedded.md", "no vector here\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let spec = test_spec(2);
        let assign = |text_frag: &str, v: &[f32]| {
            let hash: String = conn
                .query_row(
                    "SELECT hash FROM blocks WHERE text LIKE '%' || ?1 || '%'",
                    [text_frag],
                    |r| r.get(0),
                )
                .unwrap();
            insert_vector(&conn, &hash, &spec, v).unwrap();
        };
        assign("pools", &[1.0, 0.0]);
        assign("stalwart", &[0.0, 1.0]);
        let mut q = vec![0.9f32, 0.1];
        l2_normalize(&mut q);
        let hits = semantic_recall(&conn, &spec, &q, 10, &[]).unwrap();
        assert_eq!(hits.len(), 2, "unembedded block cannot appear");
        assert!(hits[0].path.ends_with("zfs.md"), "closest vector first");
        assert!(hits[1].path.ends_with("mail.md"));
        assert!(hits[0].cite.starts_with("r3-"), "hits carry the normal citation shape");
        // limit applies
        assert_eq!(semantic_recall(&conn, &spec, &q, 1, &[]).unwrap().len(), 1);
    }

    #[test]
    fn hybrid_recall_fuses_lexical_and_semantic() {
        // "zfs" appears lexically only in doc A; doc B is only semantically
        // close to the query vector. Hybrid must surface both, lexical hit
        // first (it leads on BOTH its list rank and the fused score here).
        let dir = brain(&[
            ("knowledge/a.md", "zfs pools and datasets\n"),
            ("knowledge/b.md", "storage volumes explained\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let spec = test_spec(2);
        let mut stmt = conn.prepare("SELECT hash FROM blocks ORDER BY id").unwrap();
        let hashes: Vec<String> = stmt.query_map([], |r| r.get(0)).unwrap().filter_map(Result::ok).collect();
        drop(stmt);
        insert_vector(&conn, &hashes[0], &spec, &[1.0, 0.0]).unwrap(); // a.md: far from query
        insert_vector(&conn, &hashes[1], &spec, &[0.0, 1.0]).unwrap(); // b.md: close to query
        let q = vec![0.0f32, 1.0];
        let hits = hybrid_recall(&conn, &spec, "zfs", &q, 10, 2.0, &[]).unwrap();
        let paths: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        assert!(paths.contains(&"knowledge/a.md"), "lexical-only hit present");
        assert!(paths.contains(&"knowledge/b.md"), "semantic-only hit present");
        // a.md: rank 1 lexical (1/3) + rank 2 semantic (1/4) = 0.583
        // b.md: rank 1 semantic (1/3) = 0.333
        assert_eq!(paths[0], "knowledge/a.md");
    }

    #[test]
    fn hybrid_preserves_agreement_between_reserved_lexical_and_semantic_top_policy() {
        let obsolete = "- approval";
        let current = "- A production restart needs explicit approval for this particular action, even during maintenance.";
        for ring in [0, 1] {
            let dir = brain(&[
                ("knowledge/a-old.md", &format!("---\nring: 4\n---\n{obsolete}\n")),
                ("knowledge/z-current.md", &format!("---\nring: {ring}\n---\n{current}\n")),
                // Keep IDF positive so the short obsolete block wins raw BM25.
                ("knowledge/fill.md", "- one\n- two\n- three\n- four\n- five\n- six\n- seven\n- eight\n"),
            ]);
            let state = tempfile::tempdir().unwrap();
            let mut conn = open(state.path()).unwrap();
            scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
            let spec = test_spec(2);
            insert_vector(&conn, &content_hash(current), &spec, &[1.0, 0.0]).unwrap();
            insert_vector(&conn, &content_hash(obsolete), &spec, &[0.8, 0.6]).unwrap();
            let query = "approval";
            let vector = [1.0, 0.0];

            let raw_first: String = conn.query_row(
                &ranked_match_sql("d.path", 0),
                rusqlite::params_from_iter(ranked_params(&fts_query(query), 20, &[])),
                |row| row.get(0),
            ).unwrap();
            assert_eq!(raw_first, "knowledge/a-old.md", "fixture must exercise the reservation");
            let lexical = recall(&conn, query, 10).unwrap();
            let semantic = semantic_recall(&conn, &spec, &vector, 10, &[]).unwrap();
            assert_eq!(lexical[0].path, "knowledge/z-current.md");
            assert_eq!(semantic[0].path, "knowledge/z-current.md");

            let lexical_ids = bm25_block_ids(&conn, query, 20, &[]).unwrap();
            let fusion_input = hits_for_block_ids(&conn, &lexical_ids).unwrap();
            assert_eq!(
                fusion_input.iter().map(|hit| &hit.path).collect::<Vec<_>>(),
                lexical.iter().map(|hit| &hit.path).collect::<Vec<_>>(),
                "hybrid must receive the same reserved lexical order"
            );
            let hybrid = hybrid_recall(&conn, &spec, query, &vector, 10, 2.0, &[]).unwrap();
            assert_eq!(hybrid[0].path, "knowledge/z-current.md", "both ranking inputs agree on current policy");
            assert_eq!(hybrid[1].path, "knowledge/a-old.md");
        }
    }

    #[test]
    fn recall_suppresses_native_mirror_duplicates() {
        // The native auto-memory store mirrors brain files: the same logical
        // block must surface ONCE, as its lowest-ring copy, with the
        // suppressed paths recorded as mirrors.
        let brain = brain(&[(
            "mind/memories/MEMORY.md",
            "- zvol on btrfs needs the nossd mount option\n",
        )]);
        let native = tempfile::tempdir().unwrap();
        let mem = native.path().join("p/memory");
        std::fs::create_dir_all(&mem).unwrap();
        std::fs::write(mem.join("MEMORY.md"), "- zvol on btrfs needs the nossd mount option\n")
            .unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, brain.path(), Some(native.path()), &RingRules::default()).unwrap();
        let hits = recall(&conn, "nossd", 5).unwrap();
        assert_eq!(hits.len(), 1, "one logical block, one hit");
        assert_eq!(hits[0].ring, 1, "the lowest-ring copy survives");
        assert_eq!(hits[0].path, "mind/memories/MEMORY.md");
        assert_eq!(hits[0].mirrors, vec!["native:p/MEMORY.md".to_string()]);
    }

    #[test]
    fn dedup_keeps_lowest_ring_regardless_of_rank_order() {
        let h = |cite: &str, path: &str, ring: u8, chain: &str| Hit {
            cite: cite.into(),
            path: path.into(),
            ring,
            start_line: 1,
            end_line: 1,
            snippet: String::new(),
            mirrors: Vec::new(),
            chain: chain.into(),
            text: String::new(),
        };
        let hits = vec![
            h("r2-aabbccddee", "native:p/MEMORY.md", 2, "Memory"),
            h("r1-aabbccddee", "mind/memories/MEMORY.md", 1, "Memory"),
            h("r3-0123456789", "knowledge/x.md", 3, ""),
        ];
        let out = dedup_by_content(hits);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].cite, "r1-aabbccddee", "lowest ring wins even when ranked second");
        assert_eq!(out[0].mirrors, vec!["native:p/MEMORY.md".to_string()]);
        assert_eq!(out[1].cite, "r3-0123456789");
        assert!(out[1].mirrors.is_empty(), "a hit without duplicates is untouched");
    }

    #[test]
    fn identical_blocks_in_different_sections_are_not_mirrors() {
        let h = |cite: &str, path: &str, ring: u8, chain: &str| Hit {
            cite: cite.into(),
            path: path.into(),
            ring,
            start_line: 1,
            end_line: 1,
            snippet: String::new(),
            mirrors: Vec::new(),
            chain: chain.into(),
            text: String::new(),
        };
        let hits = vec![
            h("r3-aabbccddee", "knowledge/a.md", 3, "Hosts > Server"),
            h("r3-aabbccddee", "knowledge/b.md", 3, "Backups"),
        ];
        let out = dedup_by_content(hits);
        assert_eq!(out.len(), 2, "same hash under different heading chains = two statements");
        assert!(out.iter().all(|h| h.mirrors.is_empty()));
    }

    #[test]
    fn dedup_refills_freed_limit_slots() {
        // The mirrored block is the shortest (= best BM25), so a naive
        // LIMIT 2 would return both copies of it and lose the second logical
        // block. Suppression must not shrink the result count.
        let brain = brain(&[
            ("mind/memories/MEMORY.md", "- flumox\n"),
            ("knowledge/other.md", "flumox beta fact with more words\n"),
        ]);
        let native = tempfile::tempdir().unwrap();
        let mem = native.path().join("p/memory");
        std::fs::create_dir_all(&mem).unwrap();
        std::fs::write(mem.join("MEMORY.md"), "- flumox\n").unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, brain.path(), Some(native.path()), &RingRules::default()).unwrap();
        let hits = recall(&conn, "flumox", 2).unwrap();
        assert_eq!(hits.len(), 2, "suppression must not shrink the result count");
        let paths: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        assert!(paths.contains(&"mind/memories/MEMORY.md"));
        assert!(paths.contains(&"knowledge/other.md"));
        let kept = hits.iter().find(|h| h.path == "mind/memories/MEMORY.md").unwrap();
        assert_eq!(kept.mirrors, vec!["native:p/MEMORY.md".to_string()]);
    }

    #[test]
    fn expand_matches_hash_across_rings_lowest_first() {
        let dir = brain(&[
            ("knowledge/copy.md", "shared statement body\n"),
            ("knowledge/promoted.md", "---\nring: 1\n---\nshared statement body\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let hits = recall(&conn, "shared statement", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].ring, 1);
        assert_eq!(hits[0].mirrors, vec!["knowledge/copy.md".to_string()]);
        let blocks = expand(&conn, &hits[0].cite).unwrap();
        assert_eq!(blocks.len(), 2, "every copy of the logical block surfaces");
        assert_eq!(blocks[0].ring, 1, "lowest ring first");
        assert_eq!(blocks[0].path, "knowledge/promoted.md");
        assert_eq!(blocks[1].ring, 3);
        assert_eq!(blocks[1].path, "knowledge/copy.md");
    }

    #[test]
    fn scan_reports_skipped_paths() {
        let dir = brain(&[
            ("knowledge/ok.md", "fine\n"),
            ("knowledge/staged.md", "---\nring: 5\n---\nquarantined\n"),
            ("knowledge/broken.md", "---\nring: banana\n---\nunparseable\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let report = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(report.skipped_high_ring, 2);
        assert_eq!(
            report.skipped,
            vec!["knowledge/broken.md".to_string(), "knowledge/staged.md".to_string()],
            "skipped paths listed in doc-path order"
        );
    }

    #[test]
    fn tilde_fences_segment_like_backticks() {
        let text = "~~~\ncode line\nstill code\n~~~\n\nafter\n";
        let blocks = segment(text, 0);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].2, "~~~\ncode line\nstill code\n~~~");
        assert_eq!((blocks[0].0, blocks[0].1), (1, 4));
        assert_eq!(blocks[1].2, "after");
        // A tilde fence containing backtick runs must not close early, and
        // vice versa.
        let text = "~~~\n```\ninner\n```\n~~~\nafter\n";
        let blocks = segment(text, 0);
        assert_eq!(blocks.len(), 2, "backtick runs inside a tilde fence do not close it");
        assert_eq!((blocks[0].0, blocks[0].1), (1, 5));
        let text = "```\n~~~\ninner\n~~~\n```\n";
        let blocks = segment(text, 0);
        assert_eq!(blocks.len(), 1, "tilde runs inside a backtick fence do not close it");
        assert_eq!((blocks[0].0, blocks[0].1), (1, 5));
    }

    #[test]
    fn setext_headings_are_heading_blocks() {
        let text = "Title\n=====\n\nintro paragraph\n\nSection\n-------\n\nbody text\n";
        let blocks = segment(text, 0);
        let bodies: Vec<&str> = blocks.iter().map(|(_, _, b)| b.as_str()).collect();
        assert_eq!(bodies, vec!["Title\n=====", "intro paragraph", "Section\n-------", "body text"]);
        assert_eq!((blocks[0].0, blocks[0].1), (1, 2), "H1 spans text + underline");
        assert_eq!((blocks[2].0, blocks[2].1), (6, 7), "H2 spans text + underline");
        assert_eq!(heading_of(blocks[0].2.as_str()), Some((1, "Title".to_string())));
        assert_eq!(heading_of(blocks[2].2.as_str()), Some((2, "Section".to_string())));
        // A multi-line paragraph before an underline: only the LAST line is
        // underlined into the heading; earlier lines stay a paragraph.
        let text = "para line\nTitle\n====\n";
        let blocks = segment(text, 0);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].2, "para line");
        assert_eq!((blocks[0].0, blocks[0].1), (1, 1));
        assert_eq!(blocks[1].2, "Title\n====");
        assert_eq!((blocks[1].0, blocks[1].1), (2, 3));
    }

    #[test]
    fn numbered_items_keep_blank_separated_indented_continuation() {
        let text = "1. item one\n\n   continuation para of item one\n\n2. item two\n";
        let blocks = segment(text, 0);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].2, "1. item one\n\n   continuation para of item one");
        assert_eq!((blocks[0].0, blocks[0].1), (1, 3), "exact line span including the blank");
        assert_eq!(blocks[1].2, "2. item two");
        assert_eq!((blocks[1].0, blocks[1].1), (5, 5));
    }

    #[test]
    fn paren_numbered_items_split_like_dotted_ones() {
        let blocks = segment("1) item one\n2) item two\n", 0);
        assert_eq!(blocks.len(), 2, "CommonMark `1)` ordered markers are list items");
        assert_eq!(blocks[0].2, "1) item one");
        assert_eq!(blocks[1].2, "2) item two");
    }

    #[test]
    fn tab_indented_continuation_stays_with_the_item() {
        let blocks = segment("1. item one\n\tcontinued with a tab\n", 0);
        assert_eq!(blocks.len(), 1);
        assert_eq!((blocks[0].0, blocks[0].1), (1, 2));
    }

    #[test]
    fn heading_chain_prepended_to_snippets() {
        let dir = brain(&[(
            "knowledge/h.md",
            "# ProjectX\n\n## Setup\n\nInstall the flurbium package.\n",
        )]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let hits = recall(&conn, "flurbium", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].snippet, "[ProjectX > Setup] Install the flurbium package.");
        // A bare heading block carries its ANCESTOR chain.
        let hits = recall(&conn, "setup", 5).unwrap();
        let heading = hits.iter().find(|h| h.snippet.contains("## Setup")).unwrap();
        assert_eq!(heading.snippet, "[ProjectX] ## Setup");
    }

    #[test]
    fn table_rows_carry_header_and_chain() {
        let dir = brain(&[(
            "knowledge/t.md",
            "# Hosts\n\n| Name | IP |\n|---|---|\n| serverx | 192.0.2.1 |\n",
        )]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let hits = recall(&conn, "serverx", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].snippet, "[Hosts > | Name | IP |] | serverx | 192.0.2.1 |");
        // The header row itself gets only the heading chain.
        let hits = recall(&conn, "name", 5).unwrap();
        let header = hits.iter().find(|h| h.snippet.contains("| Name | IP |")).unwrap();
        assert_eq!(header.snippet, "[Hosts] | Name | IP |");
    }

    #[test]
    fn chain_matches_recall_but_rank_below_text_matches() {
        // Filler blocks keep the term's IDF positive (BM25 inverts its
        // ranking when a term appears in most blocks of a tiny corpus).
        let dir = brain(&[
            ("knowledge/a.md", "glorpnik\n"),
            ("knowledge/b.md", "# glorpnik\n\nunrelated words here\n"),
            ("knowledge/fill.md", "- one\n- two\n- three\n- four\n- five\n- six\n- seven\n- eight\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let hits = recall(&conn, "glorpnik", 10).unwrap();
        let ctx_only = hits
            .iter()
            .position(|h| h.snippet.contains("unrelated words"))
            .expect("a block whose HEADING CHAIN matches must still be recalled");
        assert!(
            hits[0].snippet.contains("glorpnik"),
            "a text match leads: {:?}",
            hits.iter().map(|h| &h.snippet).collect::<Vec<_>>()
        );
        assert_eq!(
            ctx_only,
            hits.len() - 1,
            "the chain-only match ranks below every text match (lower column weight)"
        );
    }

    #[test]
    fn same_text_under_different_headings_is_two_hits_not_mirrors() {
        let dir = brain(&[
            ("knowledge/a.md", "# Server\n\n- restart the service\n"),
            ("knowledge/b.md", "# Laptop\n\n- restart the service\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let hits = recall(&conn, "restart", 5).unwrap();
        assert_eq!(hits.len(), 2, "different sections = different statements");
        assert!(hits.iter().all(|h| h.mirrors.is_empty()));
    }

    #[test]
    fn wikilinks_inside_code_fences_are_ignored() {
        let text = "```\n[[nope]]\n```\nsee [[yes]]\n~~~\n[[also-nope]]\n~~~\n";
        assert_eq!(wikilinks(text), vec!["yes"]);
    }

    #[test]
    fn slash_qualified_wikilink_resolves_an_ambiguous_stem() {
        let dir = brain(&[
            ("knowledge/a/readme2.md", "x\n"),
            ("knowledge/b/readme2.md", "y\n"),
            ("knowledge/linker.md", "see [[a/readme2]] and [[knowledge/b/readme2]]\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let linked = linked_docs(&conn, &["knowledge/linker.md".to_string()], 8).unwrap();
        let paths: Vec<&str> = linked.iter().map(|(p, _)| p.as_str()).collect();
        assert!(paths.contains(&"knowledge/a/readme2.md"), "parent/stem suffix resolves: {paths:?}");
        assert!(paths.contains(&"knowledge/b/readme2.md"), "full-path suffix resolves: {paths:?}");
    }

    #[test]
    fn ring0_1_match_takes_the_top_slot() {
        // The ring-3 block is the stronger lexical match (shortest); the
        // ring-1 statement must still hold the top slot.
        let dir = brain(&[
            ("AGENT.md", "- the fenwick rule with several more words around it\n"),
            ("knowledge/deep.md", "fenwick\n"),
            // Fillers keep the IDF positive so BM25 genuinely prefers the
            // shorter ring-3 block before the reservation kicks in.
            ("knowledge/fill.md", "- one\n- two\n- three\n- four\n- five\n- six\n- seven\n- eight\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let hits = recall(&conn, "fenwick", 5).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].ring, 1, "top slot reserved for the ring 0-1 band");
        assert_eq!(hits[1].ring, 3, "the rest stays in BM25 order");
    }

    #[test]
    fn ring_prior_breaks_ties_but_never_strong_lexical_wins() {
        // Equal lexical evidence: the lower ring wins the near-tie.
        let dir = brain(&[
            ("knowledge/a.md", "zelkova pattern alpha\n"),
            ("todo/x/b.md", "zelkova pattern bravo\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let hits = recall(&conn, "zelkova", 5).unwrap();
        assert_eq!(hits[0].ring, 3, "prior breaks the tie toward the lower ring");
        assert_eq!(hits[1].ring, 4);
        // Strong lexical win: the much better match stays first even from a
        // higher ring (no ring 0-1 hit involved, so no reservation either).
        let dir = brain(&[
            ("knowledge/a.md", "korvat is mentioned once amid many other words in this long statement about something else\n"),
            ("todo/x/b.md", "korvat\n"),
            ("knowledge/fill.md", "- one\n- two\n- three\n- four\n- five\n- six\n- seven\n- eight\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        let hits = recall(&conn, "korvat", 5).unwrap();
        assert_eq!(hits[0].ring, 4, "a strong lexical win is never overridden by the prior");
    }

    #[test]
    fn rescan_changed_matches_a_fresh_scan_exactly() {
        let dir = brain(&[
            ("knowledge/a.md", "# Hosts\n\n| Name | IP |\n| serverx | 192.0.2.1 |\n"),
            ("knowledge/b.md", "- keep this\n"),
            ("knowledge/c.md", "doomed content\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        let r1 = scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(r1.generation, 1);
        // Vectors on every block: unchanged CONTENT must keep theirs.
        let spec = test_spec(2);
        embed_everything(&conn, &spec, &[1.0, 0.0]);
        let vectors_before = vector_coverage(&conn, &spec).unwrap().0;

        // Change one file, add one (with a wikilink), delete one.
        std::fs::write(dir.path().join("knowledge/b.md"), "- keep this\n- brand new fact\n").unwrap();
        std::fs::write(dir.path().join("knowledge/d.md"), "arrived later, see [[a]]\n").unwrap();
        std::fs::remove_file(dir.path().join("knowledge/c.md")).unwrap();

        let r2 = rescan_changed(&mut conn, dir.path(), None, &RingRules::default()).unwrap().expect("small diff â†’ incremental");
        assert_eq!(r2.generation, 2, "incremental commit advances the generation");
        assert!(!stale(&conn, dir.path(), None, &RingRules::default()).unwrap());

        // Byte-identical catalog vs an independent fresh derivation.
        let state2 = tempfile::tempdir().unwrap();
        let mut fresh = open(state2.path()).unwrap();
        scan(&mut fresh, dir.path(), None, &RingRules::default()).unwrap();
        assert_eq!(
            catalog_checksum(&conn).unwrap(),
            catalog_checksum(&fresh).unwrap(),
            "incremental and fresh derivations must agree byte-for-byte"
        );

        // Content moved correctly.
        assert_eq!(recall(&conn, "brand", 5).unwrap().len(), 1);
        assert_eq!(recall(&conn, "arrived", 5).unwrap().len(), 1);
        assert!(recall(&conn, "doomed", 5).unwrap().is_empty());
        // The changed doc's snippet still carries its rebuilt table context.
        let hits = recall(&conn, "serverx", 5).unwrap();
        assert_eq!(hits[0].snippet, "[Hosts > | Name | IP |] | serverx | 192.0.2.1 |");
        // Links of the NEW doc resolved (full re-resolution each rescan).
        let linked = linked_docs(&conn, &["knowledge/d.md".to_string()], 8).unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].0, "knowledge/a.md");
        // Unchanged BLOCKS kept their vectors â€” including the untouched line
        // of the doc that changed. Only genuinely new text needs embedding.
        let (vectors_after, blocks_after) = vector_coverage(&conn, &spec).unwrap();
        let a_blocks = 3; // heading + 2 table rows
        assert_eq!(vectors_after, a_blocks + 1, "a.md's blocks plus b.md's kept line");
        assert_eq!(blocks_after, a_blocks + 3, "b.md's two lines and d.md's one");
        assert_eq!(vectors_before, a_blocks + 2, "c.md's block was embedded too, then pruned");
        let queued: Vec<String> =
            hashes_without_vectors(&conn, &spec, 10).unwrap().into_iter().map(|(_, t)| t).collect();
        assert_eq!(queued, vec!["- brand new fact", "arrived later, see [[a]]"]);

        // A no-op rescan commits nothing and keeps the generation.
        let r3 = rescan_changed(&mut conn, dir.path(), None, &RingRules::default()).unwrap().unwrap();
        assert_eq!(r3.generation, 2, "fingerprint already current â†’ no new commit");
        assert_eq!(generation(&conn), 2);
    }

    #[test]
    fn rescan_falls_back_without_basis_or_on_large_diffs() {
        let dir = brain(&[("knowledge/a.md", "alpha\n")]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        // Never scanned: no basis.
        assert!(rescan_changed(&mut conn, dir.path(), None, &RingRules::default()).unwrap().is_none());
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        // Different brain root: no basis.
        let other = brain(&[("knowledge/a.md", "alpha\n")]);
        assert!(rescan_changed(&mut conn, other.path(), None, &RingRules::default()).unwrap().is_none());
        // A diff beyond the incremental cap falls back to the full scan.
        for i in 0..40 {
            std::fs::write(dir.path().join(format!("knowledge/bulk{i}.md")), "bulk\n").unwrap();
        }
        assert!(rescan_changed(&mut conn, dir.path(), None, &RingRules::default()).unwrap().is_none());
        assert!(stale(&conn, dir.path(), None, &RingRules::default()).unwrap(), "fallback leaves the full scan to do it");
    }

    #[test]
    fn rescan_handles_ring5_transitions_both_ways() {
        let dir = brain(&[
            ("knowledge/staged.md", "---\nring: 5\n---\nquarantined zulqar\n"),
            ("knowledge/normal.md", "public wembly fact\n"),
        ]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert!(recall(&conn, "zulqar", 5).unwrap().is_empty());

        // Promotion: the ring-5 marker is removed.
        std::fs::write(dir.path().join("knowledge/staged.md"), "promoted zulqar\n").unwrap();
        // Demotion: a public file becomes staged.
        std::fs::write(dir.path().join("knowledge/normal.md"), "---\nring: 5\n---\npublic wembly fact\n").unwrap();
        let r = rescan_changed(&mut conn, dir.path(), None, &RingRules::default()).unwrap().expect("incremental");
        assert_eq!(r.skipped_high_ring, 1);
        assert_eq!(recall(&conn, "zulqar", 5).unwrap().len(), 1, "promoted file is indexed");
        assert!(recall(&conn, "wembly", 5).unwrap().is_empty(), "demoted file left the index");
        assert!(!stale(&conn, dir.path(), None, &RingRules::default()).unwrap(), "skipped-file stats are tracked too");
        // The tracked skipped file changing again is still an incremental step.
        std::fs::write(dir.path().join("knowledge/normal.md"), "---\nring: 5\n---\nstill hidden, edited\n").unwrap();
        let r = rescan_changed(&mut conn, dir.path(), None, &RingRules::default()).unwrap().expect("incremental");
        assert_eq!(r.skipped_high_ring, 1);
        assert!(!stale(&conn, dir.path(), None, &RingRules::default()).unwrap());
    }

    #[test]
    fn staleness_flips_on_edit_and_rescan_clears_it() {
        let dir = brain(&[("knowledge/a.md", "alpha\n")]);
        let state = tempfile::tempdir().unwrap();
        let mut conn = open(state.path()).unwrap();
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert!(!stale(&conn, dir.path(), None, &RingRules::default()).unwrap());
        std::fs::write(dir.path().join("knowledge/a.md"), "alpha beta, much longer now\n").unwrap();
        assert!(stale(&conn, dir.path(), None, &RingRules::default()).unwrap());
        scan(&mut conn, dir.path(), None, &RingRules::default()).unwrap();
        assert!(!stale(&conn, dir.path(), None, &RingRules::default()).unwrap());
        assert_eq!(recall(&conn, "beta", 5).unwrap().len(), 1);
    }
}
