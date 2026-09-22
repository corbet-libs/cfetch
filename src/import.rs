//! One-way importer from openwolf-enhanced `.wolf/` directories into the
//! cfetch brain tree. Preserves every markdown file an operator or agent
//! wrote; skips everything cfetch regenerates (code index, token ledger,
//! exhaust, embeddings, hooks, cron state); lists everything it does not
//! recognise instead of dropping it silently.
//!
//! The mapping is deliberate:
//!
//! | openwolf | cfetch | ring | why |
//! |---|---|---|---|
//! | `OPENWOLF.md` | `knowledge/openwolf-protocol.md` | 3 | the old operating protocol, preserved verbatim but recall-only: its instructions point at `.wolf/` paths that do not exist here and at `openwolf recall`; injecting them with ring-1 authority would redirect every session's work outside the brain |
//! | *(authored)* | `AGENT.md` | 1 | a migration bridge written by cfetch (never overwriting an operator's file): where knowledge, bugs, memories, and staging live now, and where the old protocol is kept |
//! | `cerebrum.md` | `knowledge/cerebrum.md` | 3 | curated, recallable |
//! | `memory.md` | `knowledge/behaviours/MEMORY.md` | 2 | behaviour, scoped injection |
//! | `identity.md` | `knowledge/identity.md` | 3 | project identity |
//! | `reframe-frameworks.md` | `knowledge/reframe-frameworks.md` | 3 | reference |
//! | `STATUS.md` | `knowledge/handoff.md` | 0 | snapshot of the project state at migration time; a one-way import makes it history, and it is the natural ring-0 handoff |
//! | `buglog.json` | `knowledge/bugs/<id>.md` | 3 | one note per curated bug; the ids stay greppable so `bug-NNN` references inside imported files keep resolving |
//! | archive `*.md` | `knowledge/archive/` | excl | out of the index |
//! | `todo/staging/*` | `scratch/cfetch-staging/*` | 5 | quarantine, same contract |
//!
//! Files cfetch derives on its own are skipped with a note: `anatomy.md`
//! (code index), `token-ledger.json`, `config.json` (cfetch has its own
//! two-layer config), `recall-embeddings.*` (cfetch will re-embed), hooks,
//! logs, cron state. Anything else found at the top level is reported
//! as `unrecognized` and left in place — ring placement for somebody else's
//! conventions is the operator's call, not the importer's.
//!
//! The dry run and the real run share one code path: `collect()` builds the
//! same report either way, and only the real run writes. The two cannot
//! disagree with each other.
//!
//! A migrated tree must also reach the model, not just exist on disk: the
//! real run writes a starter tree config (`<brain>/.cfetch/config.json`,
//! never overwriting an existing one) that makes `AGENT.md` — and the
//! migrated handoff, when present — resident, so the session hooks inject
//! from the first `scan` onward instead of silently injecting nothing.

use std::path::Path;

pub struct ImportReport {
    pub imported: Vec<(String, String)>,
    pub skipped: Vec<(String, String)>,
    /// Top-level entries that matched no table — left in place, but named.
    pub unrecognized: Vec<String>,
    pub errors: Vec<(String, String)>,
    /// What the import did about resident injection: confirmation of the
    /// starter config it wrote, or a warning that the digest is empty and
    /// the file that would fix it. `None` when an existing non-empty
    /// resident config answers the question already.
    pub resident_note: Option<String>,
}

/// Files that become brain content, with their destination and a ring
/// frontmatter to prepend (None = no frontmatter needed, the location
/// default already applies).
pub const MIGRATIONS: &[(&str, &str, Option<u8>)] = &[
    ("OPENWOLF.md", "knowledge/openwolf-protocol.md", Some(3)),
    ("cerebrum.md", "knowledge/cerebrum.md", Some(3)),
    ("memory.md", "knowledge/behaviours/MEMORY.md", Some(2)),
    ("identity.md", "knowledge/identity.md", Some(3)),
    (
        "reframe-frameworks.md",
        "knowledge/reframe-frameworks.md",
        Some(3),
    ),
    ("STATUS.md", "knowledge/handoff.md", Some(0)),
];

/// The ring-1 file cfetch authors when a migration would otherwise leave
/// the resident channel to the old tool's protocol (or to nothing). Facts
/// only: where the migrated conventions live now and where the old protocol
/// is kept. An operator's own AGENT.md is never overwritten.
const AGENT_BRIDGE: &str = "---\nring: 1\n---\n\n# Migration bridge (authored by `cfetch import openwolf`)\n\nThis brain was migrated from an openwolf-enhanced `.wolf/` tree. The\nprevious operating protocol is preserved verbatim at\n`knowledge/openwolf-protocol.md` — recall-only. Its instructions reference\n`.wolf/` paths (`.wolf/STATUS.md`, `.wolf/memory.md`, `.wolf/buglog.json`)\nand the `openwolf recall` command; none of these exist in a cfetch brain.\nFollowing them here sends work outside the tree.\n\nWhere things live now:\n\n- curated facts and project knowledge → `knowledge/`\n- one note per bug or failure conclusion → `knowledge/bugs/`\n- behaviour and working agreements → `knowledge/behaviours/`\n- quarantine for anything unreviewed → the selected mind’s `memories/`\n- the migrated project-state snapshot (ring-0 handoff) → `knowledge/handoff.md`\n";

/// Files cfetch regenerates or tracks differently — skipped with a reason.
const SKIPPED: &[(&str, &str)] = &[
    ("anatomy.md", "cfetch builds its own code index"),
    ("anatomy-graph.json", "derived from the code index"),
    ("anatomy-symbols.json", "derived from the code index"),
    ("token-ledger.json", "cfetch has its own token accounting"),
    ("config.json", "cfetch has its own two-layer config"),
    ("cron-manifest.json", "cfetch has its own cron engine"),
    ("cron-state.json", "cfetch has its own cron engine"),
    ("designqc-report.json", "cfetch generates its own"),
    ("suggestions.json", "cfetch generates its own"),
    (
        "recall-embeddings.json",
        "cfetch will re-embed with its model",
    ),
    (
        "recall-embeddings.vec",
        "cfetch will re-embed with its model",
    ),
];

/// Patterns for archive files — old content kept but excluded from the index
/// by cfetch's default `knowledge/archive/` exclude prefix.
const ARCHIVE_SUFFIXES: &[&str] = &[".vor-aufraeumen", "-archiv-", "-backup-", "-old-"];

fn is_archive(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    // The suffixes are specific enough (German "vor-aufräumen", dated
    // archives, backups) that they cannot false-positive on ordinary
    // brain files — no .md extension check needed, since backup naming
    // appends AFTER the extension (memory.md.vor-aufraeumen).
    ARCHIVE_SUFFIXES.iter().any(|s| lower.contains(s))
}

/// Prepends a `ring:` frontmatter block if the file doesn't already have one.
fn with_ring(raw: &str, ring: Option<u8>) -> String {
    let Some(ring) = ring else {
        return raw.to_string();
    };
    if raw.starts_with("---") {
        // Already has frontmatter; don't duplicate the fence.
        return raw.to_string();
    }
    format!("---\nring: {}\n---\n\n{}", ring, raw)
}

/// The openwolf-enhanced bug log: one hand-curated entry per failure, each
/// written down right after losing an afternoon to it. The schema was read
/// from exported `buglog.json` data files, never from openwolf source code.
#[derive(serde::Deserialize, Default)]
struct Buglog {
    #[serde(default)]
    bugs: Vec<BuglogEntry>,
}

#[derive(serde::Deserialize, Default)]
struct BuglogEntry {
    #[serde(default)]
    id: String,
    #[serde(default)]
    timestamp: String,
    #[serde(default)]
    error_message: String,
    #[serde(default)]
    file: String,
    #[serde(default)]
    root_cause: String,
    #[serde(default)]
    fix: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    related_bugs: Vec<String>,
    /// openwolf-enhanced writes this as a number; hand-edited logs have
    /// been seen with strings. Accept either.
    #[serde(default)]
    occurrences: serde_json::Value,
    #[serde(default)]
    last_seen: String,
}

impl BuglogEntry {
    fn occurrences_text(&self) -> String {
        match &self.occurrences {
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::String(s) => s.clone(),
            _ => String::new(),
        }
    }

    /// A safe note filename for the entry's id, or None when the id cannot
    /// become one. Ids come from an external tool's data file and were never
    /// checked as filenames: `../../x` and `/abs/path/y` traversed out of
    /// `knowledge/bugs/` and wrote outside the brain with exit 0. A filename
    /// id is `[A-Za-z0-9._-]+`, does not start with a dot, and is not a
    /// `..` hop — anything else is reported, not written.
    fn safe_filename_id(&self, index: usize) -> Result<String, String> {
        let trimmed = self.id.trim();
        let candidate = if trimmed.is_empty() {
            format!("bug-{}", index + 1)
        } else {
            trimmed.to_string()
        };
        let ok = !candidate.is_empty()
            && candidate
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
            && !candidate.starts_with('.')
            && !candidate
                .split('.')
                .any(|part| part.is_empty() || part == "..")
            && !candidate.ends_with(".md");
        if ok {
            Ok(candidate)
        } else {
            Err(self.id.clone())
        }
    }
}

/// One bug entry becomes one ring-3 note. The `bug-NNN` id is kept in the
/// filename and the title so references inside imported files stay greppable.
fn bug_note(entry: &BuglogEntry) -> String {
    let mut note = String::new();
    note.push_str("---\nring: 3\n---\n\n");
    note.push_str(&format!("# {} — {}\n\n", entry.id, entry.error_message));
    note.push_str(&format!("- **error**: {}\n", entry.error_message));
    if !entry.file.is_empty() {
        note.push_str(&format!("- **file**: {}\n", entry.file));
    }
    if !entry.timestamp.is_empty() {
        note.push_str(&format!("- **date**: {}", entry.timestamp));
        if !entry.last_seen.is_empty() {
            note.push_str(&format!(" (last seen {})", entry.last_seen));
        }
        note.push('\n');
    }
    if !entry.occurrences_text().is_empty() {
        note.push_str(&format!(
            "- **occurrences**: {}\n",
            entry.occurrences_text()
        ));
    }
    if !entry.tags.is_empty() {
        note.push_str(&format!("- **tags**: {}\n", entry.tags.join(", ")));
    }
    if !entry.related_bugs.is_empty() {
        note.push_str(&format!("\nrelated: {}\n", entry.related_bugs.join(", ")));
    }
    if !entry.root_cause.is_empty() {
        note.push_str("\n## Root cause\n\n");
        note.push_str(&entry.root_cause);
        note.push_str("\n\n");
    }
    if !entry.fix.is_empty() {
        note.push_str("## Fix\n\n");
        note.push_str(&entry.fix);
        note.push('\n');
    }
    note
}

/// Builds the complete import report. `execute == false` is the dry run:
/// identical report, nothing written. This is the ONE code path both runs
/// share, so the preview cannot disagree with the act.
fn collect(wolf_dir: &Path, brain_root: &Path, execute: bool) -> anyhow::Result<ImportReport> {
    anyhow::ensure!(
        wolf_dir.is_dir(),
        "{} is not a directory; point at the .wolf/ directory",
        wolf_dir.display()
    );
    let mut report = ImportReport {
        imported: Vec::new(),
        skipped: Vec::new(),
        unrecognized: Vec::new(),
        errors: Vec::new(),
        resident_note: None,
    };

    // Migrate the named content files.
    for (src_name, dest_rel, ring) in MIGRATIONS {
        let src = wolf_dir.join(src_name);
        if !src.is_file() {
            continue;
        }
        let dest = brain_root.join(dest_rel);
        if dest.exists() {
            report.skipped.push((
                src_name.to_string(),
                "destination already exists".to_string(),
            ));
            continue;
        }
        if execute {
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| anyhow::anyhow!("create {}: {e}", parent.display()))?;
            }
            let raw = std::fs::read_to_string(&src)
                .map_err(|e| anyhow::anyhow!("read {}: {e}", src.display()))?;
            std::fs::write(&dest, with_ring(&raw, *ring))
                .map_err(|e| anyhow::anyhow!("write {}: {e}", dest.display()))?;
        }
        report
            .imported
            .push((src_name.to_string(), dest_rel.to_string()));
    }

    // A migration with no AGENT.md would leave the resident channel to the
    // old protocol — or, once that is demoted to ring 3, to nothing. cfetch
    // authors the bridge instead; an operator's own file is never touched.
    let agent_md = brain_root.join("AGENT.md");
    if !agent_md.exists() {
        if execute {
            std::fs::write(&agent_md, AGENT_BRIDGE)
                .map_err(|e| anyhow::anyhow!("write {}: {e}", agent_md.display()))?;
        }
        report.imported.push((
            "cfetch import (authored)".to_string(),
            "AGENT.md".to_string(),
        ));
    }

    // Migrate the curated bug log: one note per entry under knowledge/bugs/.
    // The ids stay in the filenames so `bug-NNN` references inside imported
    // files keep resolving; a broken log is an error entry, not an abort.
    let buglog = wolf_dir.join("buglog.json");
    if buglog.is_file() {
        let parsed = std::fs::read_to_string(&buglog)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", buglog.display()))
            .and_then(|raw| {
                serde_json::from_str::<Buglog>(&raw)
                    .map_err(|e| anyhow::anyhow!("parse buglog.json: {e}"))
            });
        match parsed {
            Ok(log) if !log.bugs.is_empty() => {
                let mut written = 0usize;
                for (index, entry) in log.bugs.iter().enumerate() {
                    let id = match entry.safe_filename_id(index) {
                        Ok(id) => id,
                        Err(raw) => {
                            let label = if raw.is_empty() {
                                format!("buglog.json: entry {index} (empty id)")
                            } else {
                                format!("buglog.json entry {raw:?}")
                            };
                            report.errors.push((
                                label,
                                "id cannot be a note filename (path separators or '..'); entry left behind".to_string(),
                            ));
                            continue;
                        }
                    };
                    let dest = brain_root
                        .join("knowledge")
                        .join("bugs")
                        .join(format!("{id}.md"));
                    if dest.exists() {
                        report
                            .skipped
                            .push((format!("buglog.json:{id}"), "already exists".to_string()));
                        continue;
                    }
                    if execute {
                        std::fs::create_dir_all(dest.parent().expect("bugs has a parent"))
                            .map_err(|e| anyhow::anyhow!("create knowledge/bugs: {e}"))?;
                        std::fs::write(&dest, bug_note(entry))
                            .map_err(|e| anyhow::anyhow!("write {}: {e}", dest.display()))?;
                    }
                    written += 1;
                }
                report.imported.push((
                    "buglog.json".to_string(),
                    format!("knowledge/bugs/ ({written} entries)"),
                ));
            }
            Ok(_) => {
                report
                    .skipped
                    .push(("buglog.json".to_string(), "no entries".to_string()));
            }
            Err(error) => {
                report
                    .errors
                    .push(("buglog.json".to_string(), error.to_string()));
            }
        }
    }

    // Migrate archive markdown files (handoff archives, re-notes, backups).
    for entry in std::fs::read_dir(wolf_dir).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if is_archive(&name) {
            let src = entry.path();
            let dest = brain_root.join("knowledge/archive").join(&name);
            if dest.exists() {
                report
                    .skipped
                    .push((name, "archive destination already exists".to_string()));
                continue;
            }
            if execute {
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                if let Err(e) = std::fs::copy(&src, &dest) {
                    report.errors.push((name.clone(), e.to_string()));
                    continue;
                }
            }
            report
                .imported
                .push((name.clone(), format!("knowledge/archive/{}", name)));
        }
    }

    // Migrate staging candidates (quarantine in cfetch has the same contract).
    let staging_src = wolf_dir.join("todo").join("staging");
    if staging_src.is_dir() {
        for entry in std::fs::read_dir(&staging_src)
            .into_iter()
            .flatten()
            .flatten()
        {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.ends_with(".md") {
                continue;
            }
            let dest_rel = format!("mind/{}/memories/{name}", crate::paths::mind_id());
            let dest = crate::paths::staging_dir(brain_root).join(&name);
            if dest.exists() {
                report
                    .skipped
                    .push((dest_rel.clone(), "already exists".to_string()));
                continue;
            }
            if execute {
                std::fs::create_dir_all(dest.parent().expect("staging has a parent")).ok();
                if let Err(e) = std::fs::copy(entry.path(), &dest) {
                    report.errors.push((dest_rel.clone(), e.to_string()));
                    continue;
                }
            }
            report
                .imported
                .push((format!("todo/staging/{name}"), dest_rel));
        }
    }

    // Report skipped files with their reasons.
    for (name, reason) in SKIPPED {
        if wolf_dir.join(name).is_file() {
            report.skipped.push((name.to_string(), reason.to_string()));
        }
    }

    // Anything else at the top level is neither imported nor skipped: name it
    // and leave it in place, so the report claims completeness by listing,
    // not by omission.
    let known = |name: &str| {
        MIGRATIONS.iter().any(|(m, _, _)| *m == name)
            || name == "buglog.json"
            || SKIPPED.iter().any(|(s, _)| *s == name)
            || name == "todo"
            || is_archive(name)
    };
    let mut unrecognized: Vec<String> = std::fs::read_dir(wolf_dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            if known(&name) {
                String::new()
            } else if entry.path().is_dir() {
                format!("{name}/")
            } else {
                name
            }
        })
        .filter(|name| !name.is_empty())
        .collect();
    unrecognized.sort();
    report.unrecognized = unrecognized;

    // A migrated tree that injects nothing is a tree the model never sees.
    // If no tree config exists, the real run writes one that makes the files
    // this import just landed resident; an existing config is never
    // overwritten, but an empty resident list is named, because the import
    // output is the one place a migrating user is guaranteed to be reading.
    // The note is identical in both modes — the dry run already speaks in
    // the real run's voice ("imported:") for everything else.
    let tree_config = crate::paths::tree_config_path(brain_root);
    if tree_config.exists() {
        let existing = std::fs::read_to_string(&tree_config)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", tree_config.display()))?;
        let has_resident = serde_json::from_str::<serde_json::Value>(&existing)
            .ok()
            .and_then(|v| v.get("resident").and_then(|r| r.as_array()).cloned())
            .is_some_and(|entries| !entries.is_empty());
        if !has_resident {
            report.resident_note = Some(format!(
                "resident digest is empty — nothing will be injected; add resident entries to {}",
                tree_config.display()
            ));
        }
    } else {
        // Mode-independent so dry run and real run agree on an empty brain:
        // the handoff counts when it already exists (a previous import) or
        // when STATUS.md is present and will land. Checking "will land" via
        // dest-not-exists AFTER the migration loop would invert on the real
        // run, which has just created the dest.
        let with_handoff = brain_root.join("knowledge/handoff.md").exists()
            || wolf_dir.join("STATUS.md").is_file();
        let mut resident = vec![serde_json::json!({"path": "AGENT.md", "ring": 1})];
        if with_handoff {
            resident.push(serde_json::json!({"path": "knowledge/handoff.md", "ring": 0}));
        }
        if execute {
            let starter = serde_json::json!({
                "resident": resident,
                "budget_chars": crate::config::default_budget_chars(),
            });
            if let Some(parent) = tree_config.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| anyhow::anyhow!("create {}: {e}", parent.display()))?;
            }
            std::fs::write(&tree_config, format!("{starter:#}\n"))
                .map_err(|e| anyhow::anyhow!("write {}: {e}", tree_config.display()))?;
        }
        let entries = if with_handoff {
            "2 resident entries (AGENT.md, knowledge/handoff.md)"
        } else {
            "1 resident entry (AGENT.md)"
        };
        report.resident_note = Some(format!(
            "starter tree config {} — {entries}; the session hooks will inject after `cfetch scan`",
            tree_config.display()
        ));
    }

    Ok(report)
}

/// The dry run: the exact report the real run would produce, nothing written.
pub fn plan_openwolf(wolf_dir: &Path, brain_root: &Path) -> anyhow::Result<ImportReport> {
    collect(wolf_dir, brain_root, false)
}

pub fn import_openwolf(wolf_dir: &Path, brain_root: &Path) -> anyhow::Result<ImportReport> {
    collect(wolf_dir, brain_root, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_frontmatter_is_prepended_only_when_absent() {
        assert_eq!(with_ring("hello", Some(2)), "---\nring: 2\n---\n\nhello");
        assert_eq!(
            with_ring("---\ntitle: x\n---\n\nbody", Some(2)),
            "---\ntitle: x\n---\n\nbody"
        );
        assert_eq!(with_ring("no ring needed", None), "no ring needed");
    }

    #[test]
    fn archive_files_are_recognised() {
        assert!(is_archive("handoff-archiv-bis-2026-08-06.md"));
        assert!(is_archive("memory.md.vor-aufraeumen"));
        assert!(is_archive("re-notes-archiv-2026-08-06.md"));
        assert!(!is_archive("cerebrum.md"));
        assert!(!is_archive("memory.md"));
        assert!(!is_archive("OPENWOLF.md"));
    }

    #[test]
    fn import_copies_and_skips_correctly() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        // Content files.
        std::fs::write(wolf.path().join("OPENWOLF.md"), "# Context\nRules here.").unwrap();
        std::fs::write(wolf.path().join("cerebrum.md"), "# Knowledge\nFacts here.").unwrap();
        std::fs::write(wolf.path().join("memory.md"), "# Memory\nBehaviours.").unwrap();
        // A file cfetch skips.
        std::fs::write(wolf.path().join("anatomy.md"), "# Anatomy\ncode index").unwrap();
        // An archive file.
        std::fs::write(
            wolf.path().join("handoff-archiv-2026-01-01.md"),
            "# Old handoff",
        )
        .unwrap();
        // A staging candidate.
        std::fs::create_dir_all(wolf.path().join("todo").join("staging")).unwrap();
        std::fs::write(
            wolf.path().join("todo/staging/candidate-abc.md"),
            "# Candidate",
        )
        .unwrap();

        let report = import_openwolf(wolf.path(), brain.path()).unwrap();

        // The old protocol is recall-only...
        assert!(
            report
                .imported
                .iter()
                .any(|(s, d)| s == "OPENWOLF.md" && d == "knowledge/openwolf-protocol.md")
        );
        let protocol =
            std::fs::read_to_string(brain.path().join("knowledge/openwolf-protocol.md")).unwrap();
        assert!(protocol.starts_with("---\nring: 3\n---"));
        assert!(protocol.contains("# Context"));
        // ...and the ring-1 file is the cfetch-authored bridge.
        assert!(
            report
                .imported
                .iter()
                .any(|(s, d)| s == "cfetch import (authored)" && d == "AGENT.md")
        );
        let agent = std::fs::read_to_string(brain.path().join("AGENT.md")).unwrap();
        assert!(agent.starts_with("---\nring: 1\n---"));
        assert!(agent.contains("knowledge/openwolf-protocol.md"));
        assert!(agent.contains("knowledge/bugs/"));
        assert!(
            report
                .imported
                .iter()
                .any(|(s, d)| s == "cerebrum.md" && d == "knowledge/cerebrum.md")
        );
        assert!(
            report
                .imported
                .iter()
                .any(|(s, d)| s == "memory.md" && d == "knowledge/behaviours/MEMORY.md")
        );
        assert!(
            report
                .imported
                .iter()
                .any(|(s, _)| s.starts_with("handoff-archiv"))
        );
        assert!(
            report
                .imported
                .iter()
                .any(|(s, _)| s.starts_with("todo/staging/"))
        );
        assert!(
            report
                .skipped
                .iter()
                .any(|(s, r)| s == "anatomy.md" && r.contains("code index"))
        );

        // Ring frontmatter was applied.
        let memory =
            std::fs::read_to_string(brain.path().join("knowledge/behaviours/MEMORY.md")).unwrap();
        assert!(
            memory.starts_with("---\nring: 2\n---"),
            "got: {}",
            &memory[..40.min(memory.len())]
        );
    }

    #[test]
    fn the_old_protocol_never_reaches_ring1_authority() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        // The protocol full of instructions for a world that no longer exists.
        std::fs::write(
            wolf.path().join("OPENWOLF.md"),
            "# OpenWolf Operating Protocol\n\n- Read .wolf/STATUS.md first.\n- Append to .wolf/memory.md after every action.\n- Log bugs to .wolf/buglog.json.\n- Never re-read a file already read this session.\n",
        )
        .unwrap();

        import_openwolf(wolf.path(), brain.path()).unwrap();

        // Whatever lands resident at ring 1 is cfetch's bridge, and none of
        // the old protocol's instructions or verbatim rules carry over: the
        // bridge may NAME the dead paths (to say they do not exist), but it
        // never instructs through them.
        let agent = std::fs::read_to_string(brain.path().join("AGENT.md")).unwrap();
        assert!(!agent.contains("Read .wolf/STATUS.md first"));
        assert!(!agent.contains("Append to .wolf/memory.md after every action"));
        assert!(!agent.contains("Log bugs to .wolf/buglog.json"));
        assert!(
            !agent.contains("Never re-read a file already read"),
            "old hard rules must not carry verbatim"
        );
        assert!(
            agent.contains("none of these exist"),
            "the bridge names the dead paths to close them"
        );
        // The protocol itself stays findable, unmodified, recall-only.
        let protocol =
            std::fs::read_to_string(brain.path().join("knowledge/openwolf-protocol.md")).unwrap();
        assert!(protocol.contains(".wolf/buglog.json"));
        assert!(protocol.contains("Never re-read"));
    }

    #[test]
    fn an_existing_agent_md_is_never_overwritten() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(wolf.path().join("OPENWOLF.md"), "# Old protocol").unwrap();
        std::fs::write(
            brain.path().join("AGENT.md"),
            "---\nring: 1\n---\n\n# My own rules",
        )
        .unwrap();

        let report = import_openwolf(wolf.path(), brain.path()).unwrap();
        let agent = std::fs::read_to_string(brain.path().join("AGENT.md")).unwrap();
        assert!(
            agent.contains("My own rules"),
            "operator file must stay untouched: {agent}"
        );
        assert!(
            !report
                .imported
                .iter()
                .any(|(s, _)| s == "cfetch import (authored)")
        );
        // The old protocol still lands recall-only.
        assert!(
            brain
                .path()
                .join("knowledge/openwolf-protocol.md")
                .is_file()
        );
    }

    #[test]
    fn dry_run_reports_exactly_what_the_real_run_does() {
        let wolf = tempfile::tempdir().unwrap();
        std::fs::write(wolf.path().join("cerebrum.md"), "# Knowledge\nsee bug-001.").unwrap();
        std::fs::write(
            wolf.path().join("STATUS.md"),
            "# Project state\nlast known good",
        )
        .unwrap();
        std::fs::write(
            wolf.path().join("buglog.json"),
            r#"{"version":1,"bugs":[{"id":"bug-001","timestamp":"2026-07-24","error_message":"frames decoded to noise","file":"loader.py","root_cause":"offsets were direct","fix":"use raw offsets","tags":["cwr"],"related_bugs":[],"occurrences":2,"last_seen":"2026-07-25"}]}"#,
        )
        .unwrap();
        std::fs::write(wolf.path().join("anatomy.md"), "index").unwrap();
        std::fs::write(wolf.path().join("ENGINE.md"), "live instructions").unwrap();

        // Same brain for both: the dry run writes nothing, so the real run
        // sees the same starting state — exactly the reported repro. The
        // two reports must be identical, paths included.
        let brain = tempfile::tempdir().unwrap();
        let planned = plan_openwolf(wolf.path(), brain.path()).unwrap();
        assert!(!brain.path().join(".cfetch/config.json").exists());
        let executed = import_openwolf(wolf.path(), brain.path()).unwrap();

        assert_eq!(planned.imported, executed.imported);
        assert_eq!(planned.skipped, executed.skipped);
        assert_eq!(planned.unrecognized, executed.unrecognized);
        assert_eq!(planned.errors, executed.errors);
        assert_eq!(planned.resident_note, executed.resident_note);
        // Both runs see the skip and the unknown file.
        assert!(planned.skipped.iter().any(|(s, _)| s == "anatomy.md"));
        assert_eq!(planned.unrecognized, vec!["ENGINE.md".to_string()]);
        // Both name the starter config with the handoff included.
        let note = planned.resident_note.expect("starter note");
        assert!(note.contains("starter tree config"), "{note}");
        assert!(
            note.contains("2 resident entries (AGENT.md, knowledge/handoff.md)"),
            "{note}"
        );
    }

    #[test]
    fn dry_run_writes_nothing() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(wolf.path().join("cerebrum.md"), "# Knowledge").unwrap();
        let report = plan_openwolf(wolf.path(), brain.path()).unwrap();
        // cerebrum.md plus the authored AGENT.md the plan announces.
        assert_eq!(report.imported.len(), 2);
        assert!(!brain.path().join("knowledge/cerebrum.md").exists());
        assert!(!brain.path().join("AGENT.md").exists());
    }

    #[test]
    fn buglog_becomes_one_greppable_note_per_entry() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(
            wolf.path().join("buglog.json"),
            r#"{"version":1,"bugs":[
                {"id":"bug-001","timestamp":"2026-07-24","error_message":"frames decoded to noise","file":"loader.py","root_cause":"offsets were direct","fix":"use raw offsets","tags":["cwr","sprites"],"related_bugs":["bug-002"],"occurrences":1,"last_seen":"2026-07-24"},
                {"id":"bug-002","timestamp":"2026-07-25","error_message":"stale assembly","file":"Godot","root_cause":"stale build","fix":"rebuild with --build-solutions","tags":[],"related_bugs":[],"occurrences":1,"last_seen":"2026-07-25"}
            ]}"#,
        )
        .unwrap();

        let report = import_openwolf(wolf.path(), brain.path()).unwrap();
        assert!(
            report
                .imported
                .iter()
                .any(|(s, d)| s == "buglog.json" && d == "knowledge/bugs/ (2 entries)")
        );

        let note = std::fs::read_to_string(brain.path().join("knowledge/bugs/bug-001.md")).unwrap();
        assert!(note.starts_with("---\nring: 3\n---"));
        assert!(note.contains("# bug-001 — frames decoded to noise"));
        assert!(note.contains("**file**: loader.py"));
        assert!(note.contains("**tags**: cwr, sprites"));
        assert!(note.contains("related: bug-002"));
        assert!(note.contains("## Root cause"));
        assert!(note.contains("## Fix"));
        assert!(note.contains("use raw offsets"));
        assert!(brain.path().join("knowledge/bugs/bug-002.md").exists());
    }

    #[test]
    fn a_broken_buglog_is_an_error_not_an_abort() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(wolf.path().join("buglog.json"), "not json").unwrap();
        std::fs::write(wolf.path().join("memory.md"), "# Memory").unwrap();

        let report = import_openwolf(wolf.path(), brain.path()).unwrap();
        assert!(report.errors.iter().any(|(s, _)| s == "buglog.json"));
        assert!(report.imported.iter().any(|(s, _)| s == "memory.md"));
    }

    #[test]
    fn buglog_ids_that_cannot_be_filenames_are_reported_not_traversed() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(
            wolf.path().join("buglog.json"),
            r#"{"version":1,"bugs":[
                {"id":"bug-001","error_message":"fine"},
                {"id":"../../escape","error_message":"traversal"},
                {"id":"/abs/path/y","error_message":"absolute"},
                {"id":"bug-004.md","error_message":"would double the extension"},
                {"id":"..","error_message":"dot hop"}
            ]}"#,
        )
        .unwrap();

        let report = import_openwolf(wolf.path(), brain.path()).unwrap();

        // The good entry lands.
        assert!(brain.path().join("knowledge/bugs/bug-001.md").is_file());
        // Nothing escaped knowledge/bugs/: no file at the brain root, none
        // outside it.
        assert_eq!(
            std::fs::read_dir(brain.path().join("knowledge/bugs"))
                .unwrap()
                .count(),
            1
        );
        assert!(!brain.path().join("escape.md").exists());
        // Every refused id is named in the report.
        let error_names: Vec<&str> = report.errors.iter().map(|(s, _)| s.as_str()).collect();
        for expected in [
            "buglog.json entry \"../../escape\"",
            "buglog.json entry \"/abs/path/y\"",
            "buglog.json entry \"bug-004.md\"",
            "buglog.json entry \"..\"",
        ] {
            let needle = expected.trim_matches('"');
            assert!(
                error_names.iter().any(|e| e.contains(needle)),
                "missing error for {expected}; got {error_names:?}"
            );
        }
    }

    #[test]
    fn an_empty_buglog_id_falls_back_to_position_and_stays_a_filename() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(
            wolf.path().join("buglog.json"),
            r#"{"version":1,"bugs":[{"error_message":"no id given"},{"id":"  ","error_message":"whitespace id"}]}"#,
        )
        .unwrap();
        let report = import_openwolf(wolf.path(), brain.path()).unwrap();
        assert!(brain.path().join("knowledge/bugs/bug-1.md").is_file());
        assert!(brain.path().join("knowledge/bugs/bug-2.md").is_file());
        assert!(report.errors.is_empty(), "{:?}", report.errors);
    }

    #[test]
    fn an_empty_buglog_is_a_skip_not_an_import() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(
            wolf.path().join("buglog.json"),
            r#"{"version":1,"bugs":[]}"#,
        )
        .unwrap();
        let report = import_openwolf(wolf.path(), brain.path()).unwrap();
        assert!(
            report
                .skipped
                .iter()
                .any(|(s, r)| s == "buglog.json" && r == "no entries")
        );
        assert!(!report.imported.iter().any(|(s, _)| s == "buglog.json"));
    }

    #[test]
    fn unknown_top_level_entries_are_named_not_dropped() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(wolf.path().join("cerebrum.md"), "# Knowledge").unwrap();
        std::fs::write(wolf.path().join("anatomy.md"), "index").unwrap();
        std::fs::write(wolf.path().join("ENGINE.md"), "live instructions").unwrap();
        std::fs::write(wolf.path().join("PREREG_2026-08-01_abc.md"), "# prereg").unwrap();
        std::fs::write(wolf.path().join("PREREG_hashes.txt"), "abc123").unwrap();
        std::fs::create_dir_all(wolf.path().join("proposals")).unwrap();
        std::fs::write(wolf.path().join("proposals/p1.md"), "# Proposal").unwrap();

        let report = import_openwolf(wolf.path(), brain.path()).unwrap();
        assert!(report.unrecognized.contains(&"ENGINE.md".to_string()));
        assert!(
            report
                .unrecognized
                .contains(&"PREREG_2026-08-01_abc.md".to_string())
        );
        assert!(
            report
                .unrecognized
                .contains(&"PREREG_hashes.txt".to_string())
        );
        assert!(report.unrecognized.contains(&"proposals/".to_string()));
        // Known names are not reported as unknown.
        assert!(!report.unrecognized.iter().any(|n| n == "cerebrum.md"));
        assert!(!report.unrecognized.iter().any(|n| n == "anatomy.md"));
        // Nothing unrecognized was touched.
        assert!(wolf.path().join("ENGINE.md").is_file());
        assert!(wolf.path().join("proposals/p1.md").is_file());
    }

    #[test]
    fn status_md_migrates_as_the_ring0_handoff() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(
            wolf.path().join("STATUS.md"),
            "# Project state\nlast known-good build",
        )
        .unwrap();

        let report = import_openwolf(wolf.path(), brain.path()).unwrap();
        assert!(
            report
                .imported
                .iter()
                .any(|(s, d)| s == "STATUS.md" && d == "knowledge/handoff.md")
        );
        let handoff = std::fs::read_to_string(brain.path().join("knowledge/handoff.md")).unwrap();
        assert!(handoff.starts_with("---\nring: 0\n---"));
        assert!(handoff.contains("last known-good build"));
    }

    #[test]
    fn a_fresh_import_writes_a_starter_tree_config() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(wolf.path().join("OPENWOLF.md"), "# Context").unwrap();
        std::fs::write(wolf.path().join("STATUS.md"), "# Project state").unwrap();

        let report = import_openwolf(wolf.path(), brain.path()).unwrap();
        let cfg_path = brain.path().join(".cfetch/config.json");
        assert!(cfg_path.is_file());
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        let resident = cfg["resident"].as_array().unwrap();
        assert_eq!(resident.len(), 2);
        assert_eq!(resident[0]["path"], "AGENT.md");
        assert_eq!(resident[0]["ring"], 1);
        assert_eq!(resident[1]["path"], "knowledge/handoff.md");
        assert_eq!(resident[1]["ring"], 0);
        assert_eq!(cfg["budget_chars"], crate::config::default_budget_chars());
        // The written config must itself load as a valid cfetch config.
        let loaded = crate::config::Config::load_from(&cfg_path);
        loaded.unwrap_or_else(|e| panic!("starter config does not load: {e:#}"));
        // The note names the starter.
        assert!(
            report
                .resident_note
                .as_deref()
                .unwrap_or_default()
                .contains("starter tree config")
        );
    }

    #[test]
    fn a_fresh_import_without_status_writes_a_single_entry_starter() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(wolf.path().join("OPENWOLF.md"), "# Context").unwrap();
        let report = import_openwolf(wolf.path(), brain.path()).unwrap();
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(brain.path().join(".cfetch/config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(cfg["resident"].as_array().unwrap().len(), 1);
        assert!(
            report
                .resident_note
                .as_deref()
                .unwrap_or_default()
                .contains("1 resident entry (AGENT.md)")
        );
    }

    #[test]
    fn an_existing_tree_config_is_never_overwritten() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(wolf.path().join("OPENWOLF.md"), "# Context").unwrap();
        std::fs::create_dir_all(brain.path().join(".cfetch")).unwrap();
        std::fs::write(
            brain.path().join(".cfetch/config.json"),
            r#"{"resident": [], "budget_chars": 1234}"#,
        )
        .unwrap();

        let report = import_openwolf(wolf.path(), brain.path()).unwrap();
        let raw = std::fs::read_to_string(brain.path().join(".cfetch/config.json")).unwrap();
        assert!(raw.contains("1234"), "starter must not overwrite: {raw}");
        // An empty resident list is named, not left silent.
        assert!(
            report
                .resident_note
                .as_deref()
                .unwrap_or_default()
                .contains("resident digest is empty")
        );
    }

    #[test]
    fn an_existing_config_with_residents_stays_silent() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(wolf.path().join("OPENWOLF.md"), "# Context").unwrap();
        std::fs::create_dir_all(brain.path().join(".cfetch")).unwrap();
        std::fs::write(
            brain.path().join(".cfetch/config.json"),
            r#"{"resident": [{"path": "AGENT.md", "ring": 1}]}"#,
        )
        .unwrap();

        let report = import_openwolf(wolf.path(), brain.path()).unwrap();
        assert!(report.resident_note.is_none());
    }

    #[test]
    fn the_dry_run_writes_no_starter_config() {
        let wolf = tempfile::tempdir().unwrap();
        let brain = tempfile::tempdir().unwrap();
        std::fs::write(wolf.path().join("OPENWOLF.md"), "# Context").unwrap();
        let report = plan_openwolf(wolf.path(), brain.path()).unwrap();
        assert!(!brain.path().join(".cfetch/config.json").exists());
        assert!(
            !brain.path().join("AGENT.md").exists(),
            "the authored bridge is a write too"
        );
        // The plan names both the migration and the authored bridge.
        assert_eq!(report.imported.len(), 2);
        assert!(report.resident_note.is_some());
    }
}
