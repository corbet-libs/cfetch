//! Ring-5 staging as FILES in the tree: `<brain_root>/scratch/cfetch-staging/<id>.md`.
//!
//! A staged candidate is the ladder's only inward crossing, so it must be
//! visible to whichever host runs maintenance — a row in one machine's
//! database never was. Each candidate is one markdown file with frontmatter
//! (`ring: 5`, the trap reason, the session and host it came from) and the
//! captured payload in the body: readable in Obsidian, greppable, and
//! deletable by hand.
//!
//! Ring 5 keeps them out of recall and injection by the ordinary ring rules —
//! `scratch/cfetch-staging/` resolves to ring 5 by LOCATION (see `index::default_ring`), and
//! every file also declares `ring: 5` in its frontmatter, so neither a moved
//! file nor a stripped default can make a candidate recallable.
//!
//! Ids are content-addressed over the trap's key (`<reason>-<8 hex>`), which
//! makes staging idempotent across hosts for free: two hosts that notice the
//! same recurring failure derive the same id, and the second one finds the
//! file already there.
//!
//! `consume` deletes the file — maintenance has taken the content into a
//! curated ring-2/3 file, so the candidate is redundant. `dismiss` MOVES it to
//! `dismissed/`, because nothing in the ladder may be silently destroyed.
//!
//! Two hosts noticing the same pattern at the same moment can both write the
//! same id; because the id IS the identity, they write the same candidate and
//! the later rename simply wins. There is nothing to reconcile.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use sha2::{Digest as _, Sha256};

/// Subdirectory holding dismissed candidates. Kept, never deleted.
pub const DISMISSED: &str = "dismissed";

/// One ring-5 candidate awaiting autonomous maintenance or manual debugging.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Candidate {
    pub id: String,
    /// Which trap fired (`hot-file`, `fix-discovered`, `recurring-failure`, …).
    pub reason: String,
    pub session: String,
    /// The host whose exhaust produced it.
    pub host: String,
    pub ts: i64,
    /// The ring-6 event kind the candidate was distilled from.
    pub kind: String,
    pub payload: serde_json::Value,
}

/// Deterministic candidate id: reason plus a short hash of the trap's key.
/// Same key on any host, same id — which is exactly what makes cross-host
/// staging idempotent.
pub fn id_for(reason: &str, key: &str) -> String {
    let mut h = Sha256::new();
    h.update(reason.as_bytes());
    h.update([0u8]);
    h.update(key.as_bytes());
    let digest = crate::hashing::hex_lower(h.finalize());
    format!("{}-{}", slug(reason), &digest[..8])
}

/// Filename-safe reason token.
fn slug(s: &str) -> String {
    let out: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c.to_ascii_lowercase() } else { '-' })
        .collect();
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() { "candidate".to_string() } else { trimmed }
}

pub fn path_of(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.md"))
}

/// A staging id is a single filename segment by construction (`slug-<8hex>`).
/// Every caller that takes an id from OUTSIDE this module (CLI arguments,
/// maintenance proposal targets) validates here, at the edge, so a
/// hand-typed `cfetch staging consume ..\..\foo` names nothing at all.
pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

pub fn dismissed_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(DISMISSED).join(format!("{id}.md"))
}

/// Has this candidate already been staged — pending OR dismissed? A dismissed
/// candidate must never come back: maintenance already settled it.
pub fn exists(dir: &Path, id: &str) -> bool {
    path_of(dir, id).exists() || dismissed_path(dir, id).exists()
}

/// Writes a candidate unless it already exists. Returns whether a new file was
/// created. The write is atomic (temp file + rename) so a watching indexer
/// never sees a half-written markdown file.
pub fn write(dir: &Path, c: &Candidate) -> anyhow::Result<bool> {
    if exists(dir, &c.id) {
        return Ok(false);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create staging dir {}", dir.display()))?;
    let path = path_of(dir, &c.id);
    let tmp = dir.join(format!(".{}.{}.tmp", c.id, std::process::id()));
    std::fs::write(&tmp, render(c)).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("stage {}", path.display()))?;
    Ok(true)
}

/// Pending candidates from EVERY host, newest first. Unparseable files are
/// skipped — a direct Markdown edit is authoritative, not a reason to crash.
pub fn list(dir: &Path) -> Vec<Candidate> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<Candidate> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("md"))
        .filter_map(|p| {
            let text = std::fs::read_to_string(&p).ok()?;
            let id = p.file_stem()?.to_string_lossy().to_string();
            parse(&text, &id)
        })
        .collect();
    out.sort_by(|a, b| b.ts.cmp(&a.ts).then_with(|| a.id.cmp(&b.id)));
    out
}

/// Number of pending candidates, without parsing any of them.
pub fn pending_count(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    e.path().is_file()
                        && e.file_name().to_str().is_some_and(|n| {
                            n.ends_with(".md") && !n.starts_with('.')
                        })
                })
                .count()
        })
        .unwrap_or(0)
}

/// Pending counts by reason, in trap order, plus the total.
#[derive(Debug, Default)]
pub struct Stats {
    pub total: usize,
    pub by_reason: Vec<(String, usize)>,
}

pub fn stats(dir: &Path) -> Stats {
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let all = list(dir);
    for c in &all {
        *counts.entry(c.reason.clone()).or_default() += 1;
    }
    let order = |r: &str| match r {
        "fix-discovered" => 0,
        "recurring-failure" => 1,
        "hot-file" => 2,
        _ => 3,
    };
    let mut by_reason: Vec<(String, usize)> = counts.into_iter().collect();
    by_reason.sort_by(|a, b| order(&a.0).cmp(&order(&b.0)).then_with(|| a.0.cmp(&b.0)));
    Stats { total: all.len(), by_reason }
}

/// Distillation took the candidate into a curated file: the staged copy is
/// redundant and goes. It MOVES to `dismissed/` (like [`dismiss`]) instead of
/// being deleted: the dismissed copy is the permanent marker that keeps the
/// staging trap from re-staging the same candidate once its `consume` record
/// has rotated out of the trap's tail window - a delete would let a still-live
/// write pattern resurrect the id forever.
pub fn consume(dir: &Path, id: &str) -> anyhow::Result<bool> {
    anyhow::ensure!(valid_id(id), "invalid staging id {id:?}");
    let path = path_of(dir, id);
    if !path.is_file() {
        return Ok(false);
    }
    let target = dismissed_path(dir, id);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::rename(&path, &target).with_context(|| format!("consume {}", target.display()))?;
    Ok(true)
}

/// Not worth promoting — but never destroyed: the file MOVES to `dismissed/`,
/// where it also serves as the marker that keeps the trap from re-staging it.
pub fn dismiss(dir: &Path, id: &str) -> anyhow::Result<bool> {
    anyhow::ensure!(valid_id(id), "invalid staging id {id:?}");
    let path = path_of(dir, id);
    if !path.is_file() {
        return Ok(false);
    }
    let target = dismissed_path(dir, id);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::rename(&path, &target).with_context(|| format!("dismiss {}", target.display()))?;
    Ok(true)
}

/// YAML scalar for a string value: JSON encoding is valid YAML and survives
/// colons, quotes and newlines in session ids or commands.
fn yaml_str(s: &str) -> String {
    serde_json::Value::from(s).to_string()
}

/// The markdown form. Frontmatter carries the coordinates, the fenced JSON
/// block carries the payload verbatim so the file round-trips exactly.
pub fn render(c: &Candidate) -> String {
    let payload = serde_json::to_string_pretty(&c.payload).unwrap_or_else(|_| "{}".into());
    format!(
        "---\nring: 5\nid: {}\nflag_reason: {}\nsession: {}\nhost: {}\nts: {}\nkind: {}\n---\n\n\
         Auto-flagged ring-5 candidate from session exhaust. Never injected and never\n\
         recalled; autonomous maintenance reviews and settles it after deterministic gates.\n\
         Manual debugging remains available with `cfetch maintain packet {}` or\n\
         `cfetch staging dismiss {}`.\n\n\
         ```json\n{payload}\n```\n",
        yaml_str(&c.id),
        yaml_str(&c.reason),
        yaml_str(&c.session),
        yaml_str(&c.host),
        c.ts,
        yaml_str(&c.kind),
        c.id,
        c.id,
    )
}

fn unquote(v: &str) -> String {
    let v = v.trim();
    serde_json::from_str::<String>(v).unwrap_or_else(|_| v.to_string())
}

/// Parses a candidate back out of its markdown. `fallback_id` is the file
/// stem, used when the frontmatter has no id (a hand-written candidate).
pub fn parse(text: &str, fallback_id: &str) -> Option<Candidate> {
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("---") {
        return None;
    }
    let mut fields: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    let mut in_frontmatter = true;
    let mut body = String::new();
    for line in lines {
        if in_frontmatter {
            if line.trim() == "---" {
                in_frontmatter = false;
                continue;
            }
            if let Some((k, v)) = line.split_once(':') {
                fields.insert(k.trim().to_ascii_lowercase(), v.to_string());
            }
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }
    if in_frontmatter {
        return None; // unterminated frontmatter: not a candidate
    }
    let payload = fenced_json(&body).unwrap_or(serde_json::Value::Null);
    Some(Candidate {
        id: fields.get("id").map(|v| unquote(v)).filter(|v| !v.is_empty())
            .unwrap_or_else(|| fallback_id.to_string()),
        reason: fields.get("flag_reason").map(|v| unquote(v)).unwrap_or_default(),
        session: fields.get("session").map(|v| unquote(v)).unwrap_or_default(),
        host: fields.get("host").map(|v| unquote(v)).unwrap_or_default(),
        ts: fields.get("ts").and_then(|v| v.trim().parse::<i64>().ok()).unwrap_or(0),
        kind: fields.get("kind").map(|v| unquote(v)).unwrap_or_default(),
        payload,
    })
}

/// First ```json fenced block of a body, decoded.
fn fenced_json(body: &str) -> Option<serde_json::Value> {
    let mut collecting = false;
    let mut buf = String::new();
    for line in body.lines() {
        if collecting {
            if line.trim_start().starts_with("```") {
                break;
            }
            buf.push_str(line);
            buf.push('\n');
        } else if line.trim() == "```json" {
            collecting = true;
        }
    }
    serde_json::from_str(&buf).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn an_id_that_is_not_a_single_segment_can_name_nothing() {
        // consume/dismiss build a path from the id; the edge check must make
        // traversal-shaped and empty ids errors, not deletions.
        let dir = tempfile::tempdir().unwrap();
        for bad in ["", "..", "..\\..\\..\\brain", "../../knowledge/x", "/abs", "a/b", "a\\b", "."] {
            assert!(!valid_id(bad), "{bad:?} must not validate");
            assert!(consume(dir.path(), bad).is_err(), "consume {bad:?} must be an error");
            assert!(dismiss(dir.path(), bad).is_err(), "dismiss {bad:?} must be an error");
        }
        assert!(valid_id("hot-file-1a2b3c4d"));
    }

    fn candidate(id: &str, host: &str, ts: i64) -> Candidate {
        Candidate {
            id: id.to_string(),
            reason: "hot-file".into(),
            session: "s1".into(),
            host: host.into(),
            ts,
            kind: "write".into(),
            payload: json!({"file_path": "/brain/knowledge/one.md", "ring": 3}),
        }
    }

    #[test]
    fn candidate_files_round_trip_through_the_tree() {
        let dir = tempfile::tempdir().unwrap();
        let c = candidate("hot-file-aabbccdd", "h1", 100);
        assert!(write(dir.path(), &c).unwrap());
        assert!(!write(dir.path(), &c).unwrap(), "staging the same id twice is a no-op");

        let back = list(dir.path());
        assert_eq!(back.len(), 1);
        assert_eq!(back[0], c, "everything survives the markdown round trip");
        let text = std::fs::read_to_string(path_of(dir.path(), &c.id)).unwrap();
        assert!(text.starts_with("---\nring: 5\n"), "ring 5 is declared first: {text}");
        assert!(text.contains("flag_reason: \"hot-file\""));
        assert!(text.contains("\"file_path\": \"/brain/knowledge/one.md\""));
    }

    #[test]
    fn consume_deletes_and_dismiss_preserves() {
        let dir = tempfile::tempdir().unwrap();
        let keep = candidate("hot-file-11111111", "h1", 1);
        let drop = candidate("hot-file-22222222", "h1", 2);
        write(dir.path(), &keep).unwrap();
        write(dir.path(), &drop).unwrap();

        assert!(consume(dir.path(), &keep.id).unwrap());
        assert!(!consume(dir.path(), &keep.id).unwrap(), "consuming twice reports nothing to do");
        assert!(!path_of(dir.path(), &keep.id).exists(), "distillation took it: the copy goes");
        assert!(
            dismissed_path(dir.path(), &keep.id).is_file(),
            "a consumed candidate moves to dismissed/, the permanent anti-resurrection marker"
        );

        assert!(dismiss(dir.path(), &drop.id).unwrap());
        assert!(!dismiss(dir.path(), &drop.id).unwrap());
        assert!(!path_of(dir.path(), &drop.id).exists());
        assert!(
            dismissed_path(dir.path(), &drop.id).is_file(),
            "a dismissed candidate is moved aside, never destroyed"
        );
        assert!(list(dir.path()).is_empty());
        assert_eq!(pending_count(dir.path()), 0);
        // A dismissal is final: the trap must never re-stage that id.
        assert!(exists(dir.path(), &drop.id), "the dismissed file is the do-not-restage marker");
        // And so is a consume: the moved-aside copy is the permanent marker,
        // surviving the trap window's rotation (the old delete let a still-
        // live write pattern resurrect the candidate once its consume record
        // left the 1 MiB tail - contradicting the stream's own promise).
        assert!(exists(dir.path(), &keep.id), "consumed ids stay decided forever");
        assert!(!write(dir.path(), &keep).unwrap(), "re-staging a consumed id is a no-op");
    }

    #[test]
    fn staging_lists_every_host_newest_first() {
        // The defect this whole change fixes: a candidate flagged on one host
        // must be visible to a distillation session on another.
        let dir = tempfile::tempdir().unwrap();
        let mut a = candidate("hot-file-aaaaaaaa", "host-alpha", 100);
        a.session = "sa".into();
        let mut b = candidate("hot-file-bbbbbbbb", "host-beta", 200);
        b.session = "sb".into();
        write(dir.path(), &a).unwrap();
        write(dir.path(), &b).unwrap();

        let listed = list(dir.path());
        assert_eq!(listed.len(), 2, "both hosts' candidates are listed");
        assert_eq!(listed[0].host, "host-beta", "newest first");
        assert_eq!(listed[1].host, "host-alpha");
        assert_eq!(pending_count(dir.path()), 2);
        // …and either host can act on either candidate.
        assert!(consume(dir.path(), &a.id).unwrap());
        assert!(dismiss(dir.path(), &b.id).unwrap());
    }

    #[test]
    fn ids_are_deterministic_across_hosts_and_filename_safe() {
        assert_eq!(id_for("hot-file", "/brain/x.md"), id_for("hot-file", "/brain/x.md"));
        assert_ne!(id_for("hot-file", "/brain/x.md"), id_for("hot-file", "/brain/y.md"));
        assert_ne!(id_for("hot-file", "k"), id_for("recurring-failure", "k"));
        let id = id_for("recurring failure!", "cargo test");
        assert!(
            id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
            "ids become file names: {id}"
        );
        assert!(id.starts_with("recurring-failure-"));
    }

    #[test]
    fn stats_count_pending_by_reason_in_trap_order() {
        let dir = tempfile::tempdir().unwrap();
        for (id, reason) in [
            ("a-1", "hot-file"),
            ("b-2", "fix-discovered"),
            ("c-3", "fix-discovered"),
            ("d-4", "recurring-failure"),
        ] {
            let mut c = candidate(id, "h1", 1);
            c.id = id.to_string();
            c.reason = reason.to_string();
            write(dir.path(), &c).unwrap();
        }
        let s = stats(dir.path());
        assert_eq!(s.total, 4);
        assert_eq!(
            s.by_reason,
            vec![
                ("fix-discovered".to_string(), 2),
                ("recurring-failure".to_string(), 1),
                ("hot-file".to_string(), 1),
            ]
        );
    }

    #[test]
    fn malformed_and_absent_files_are_survivable() {
        let dir = tempfile::tempdir().unwrap();
        assert!(list(dir.path()).is_empty());
        assert_eq!(pending_count(dir.path()), 0);
        assert!(!consume(dir.path(), "nope").unwrap());
        assert!(!dismiss(dir.path(), "nope").unwrap());
        assert!(!dir.path().join(DISMISSED).exists(), "a no-op must not create state");

        std::fs::write(dir.path().join("hand-written.md"), "no frontmatter here\n").unwrap();
        std::fs::write(dir.path().join("unterminated.md"), "---\nring: 5\nstill open\n").unwrap();
        assert!(list(dir.path()).is_empty(), "unparseable candidates are skipped, not fatal");
    }

    #[test]
    fn payload_survives_colons_quotes_and_newlines() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = candidate("fix-discovered-99999999", "h1", 7);
        c.reason = "fix-discovered".into();
        c.session = "id: with \"quotes\"".into();
        c.kind = "bash".into();
        c.payload = json!({"command": "sh -c 'echo a\nb'", "norm": "sh -c <path>"});
        write(dir.path(), &c).unwrap();
        assert_eq!(list(dir.path())[0], c);
    }
}
