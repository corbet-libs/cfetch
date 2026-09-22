//! `cfetch init`: create the brain tree cfetch's defaults already describe.
//!
//! Until now those defaults — `mind/memories/` is ring 2, `todo/` is ring 4,
//! `mind/secrets/` is never indexed — described a layout cfetch did not
//! create. That made them a description of ONE tree that happened to exist,
//! and left a new user with a binary whose rules matched nothing they had.
//! Creating the tree turns the same constants into a specification.
//!
//! Two rules govern what is created:
//!
//! - The TOP-LEVEL directories are the standard. They are what a ring rule, a
//!   slice and a grant can name, so they must mean the same thing on every
//!   machine or none of those three is portable.
//! - The SUBDIRECTORIES are not. `knowledge/` holds whatever subjects a person
//!   has, `todo/` whatever lanes they work in, `logs/` whichever agent clients
//!   they run. Inventing those would be prescribing someone's life.
//!
//! A few subdirectory NAMES are nonetheless reserved, because a rule already
//! keys on them. They are documented rather than created: if they exist the
//! rule applies, and if they never exist nothing is lost.

use std::path::{Path, PathBuf};

/// One directory in the standard tree.
///
/// `reserved` names are documented in the README but never created: a rule
/// keys on each of them, so the rule has to be visible where someone would
/// look for it, while creating the directory would prescribe a workflow
/// nobody asked for.
struct Dir {
    name: &'static str,
    readme: &'static str,
    children: &'static [&'static str],
    reserved: &'static [(&'static str, &'static str)],
}

const DIRS: &[Dir] = &[
    Dir {
        name: "knowledge",
        children: &["decisions", "people", "projects", "hosts", "world", "vocabulary"],
        reserved: &[("archive", "retired knowledge: indexed, held out of ordinary recall")],
        readme: "\
# knowledge

What is true, by subject. Ring 3 — recallable, never injected unasked.

- `decisions/` — why things are the way they are. The question agents ask most
  and can answer least: a decision recorded only in the thread that made it is
  a decision nobody can revisit without re-arguing it.
- `people/` — who is involved, and what they need.
- `projects/` — per-project knowledge that outlives any one task.
- `hosts/` — machines you operate.
- `world/` — external services and vendors: what is true about someone else's
  system.
- `vocabulary/` — the terms an agent must use correctly. The cheapest fix
  there is for a whole class of repeated error: a word used loosely once is
  used loosely for the rest of the session.

Add subjects freely; these are the ones worth having before you have any.

Reserved: `archive/` — retired knowledge. Indexed, but held out of ordinary
recall, so a superseded decision cannot answer a question as though it were
current.
",
    },
    Dir {
        name: "mind",
        children: &["memories", "skills", "tools", "identities", "policy"],
        reserved: &[("secrets", "never indexed, recalled or injected, at ANY configuration")],
        readme: "\
# mind

How to behave, who you are, and what grants access.

- `memories/` — distilled behaviour, ring 2. `memories/MEMORY.md` is the index
  and is ring 1: read every session, so its size is a standing cost.
- `skills/` — procedures, written to be followed.
- `tools/` — executables. Keep build artifacts out: a `target/` directory here
  is indexed as prose and poisons retrieval for everyone.
- `identities/` — who the agent acts as, and who the operator is to a service.
- `secrets/` — RESERVED. Never indexed, recalled or injected, at ANY
  configuration: cfetch refuses this prefix rather than merely defaulting to
  skipping it.
- `policy/` — standing conventions, declared rather than learned: where memory
  lives, how each agent client is wired, what is never done. `memories/` is
  case law; this is the constitution. Named for ring 1 rather than \"config\",
  because machine-readable tool settings live in `.cfetch/` and two things
  called config is the confusion this tree exists to remove.
",
    },
    Dir {
        name: "todo",
        children: &["active", "backlog", "blocked", "done"],
        reserved: &[],
        readme: "\
# todo

Shared task state. Ring 4 — tasks survive changes of agent or session.

- `backlog/` — all new tasks start here.
- `active/` — work in progress; keep `active/<task>/STATUS.md` current.
- `blocked/` — work waiting for a named dependency or decision.
- `done/` — closed tasks with an outcome and any reusable findings.

Move the same task between these four states. The containing directory is its
state; update its handoff and commit and push each transition when Git is used.
Working material and generated evidence belong in `../scratch/`.
",
    },
    Dir {
        name: "scratch",
        children: &[],
        reserved: &[("cfetch-staging", "ring-5 maintenance evidence and history; never recalled")],
        readme: "\
# scratch

Working material outside Git, never indexed for recall.

`cfetch-staging/` holds generated ring-5 candidates, reviews and maintenance
history. It is shared across hosts, preserved during migration, and never
injected or recalled as curated knowledge. Snapshot retention is separate
from task Git history; do not discard maintenance evidence as a cache.
",
    },
    Dir {
        name: "logs",
        children: &["audits"],
        reserved: &[],
        readme: "\
# logs

Ring 6: raw exhaust. Never injected, never committed.

- One directory per agent client you run — `claude/`, `codex/`, `gemini/`, and
  so on. Which exist is yours; the shape is the standard, so a question like
  \"which client costs least for this work\" has one answer across machines.
- `audits/` — periodic findings, distinct from per-session exhaust: an audit
  is a conclusion, a transcript is evidence.
- `cfetch/` — this tool's own exhaust and injection ledger.

Transcripts may embed plaintext secrets verbatim. Excluded from git by the
.gitignore written alongside, and exclusion means \"no history\", not \"no
backup\".
",
    },
    Dir {
        name: "state",
        children: &[],
        reserved: &[],
        readme: "\
# state

No ring: nothing here is a statement, so nothing here can be recalled,
injected, or contradict anything.

Derived artifacts expensive enough to share but which are not the record —
vector embeddings keyed by content hash, so a document is embedded once per
storage group rather than once per machine. Namespaced by tool (`cfetch/`).

Delete any of it and it rebuilds. The markdown is the truth.
",
    },
    Dir {
        name: ".cfetch",
        children: &[],
        reserved: &[],
        readme: "\
# .cfetch

Machine-readable tool configuration: ring rules, slices, endpoints.

Hidden on purpose, and for two reasons. The walker skips hidden paths, so
nothing here is indexed without a rule having to say so. And it sits beside
`.git/` and `.obsidian/` — tool-owned, tool-written, not something to browse.

Config DECLARES; it does not contain. A resident entry names a visible file
and assigns it a ring; the ring-0 text itself lives in the tree where it can
be read and edited. Anything here that would be worth recalling is in the
wrong place — prose conventions belong in `mind/config/`.
",
    },
];

/// Every reserved name, as `("parent/child", why)`. Derived from the table so
/// the list cfetch reports and the list its READMEs document cannot drift.
pub fn reserved() -> Vec<(String, &'static str)> {
    DIRS.iter()
        .flat_map(|d| d.reserved.iter().map(move |(c, why)| (format!("{}/{c}", d.name), *why)))
        .collect()
}

const GITIGNORE: &str = "\
# Written by `cfetch init`. Exclusion here means \"no history\", not \"no backup\":
# everything below still lives on the tree and rides whatever snapshots it has.

# Ring 6. Per-client transcripts may embed plaintext secrets verbatim.
/logs/

# Derived, and large. Rebuilt from the markdown at any time.
/state/

# Ring 5 candidates and disposable working material. Both are working
# material, not record: candidates cross into the tree only through the
# evidence-grounded maintenance gates, and until then they are not history.
/scratch/
# Keep legacy locations excluded until explicit migration completes.
/todo/staging/
/todo/scratch/
";

const ROOT_README: &str = "\
# The brain

A markdown tree that agents read, write and recall from. The markdown IS the
record: everything derived from it — indexes, embeddings, graphs — is a cache
that can be deleted and rebuilt.

Privilege runs by ring. 0 and 1 are operator invariants and policy, 2 is
distilled behaviour, 3 is knowledge, 4 is work state, 5 is unpromoted
candidates, 6 is raw exhaust. A lower ring wins a contradiction, and the ring
of a statement is visible in the citation that carries it.

The top-level directories and the subdirectories listed in each README are a
STANDARD: they are what a ring rule, a slice and a grant each name, so unless
they mean the same thing on every machine none of those three is portable
between people. What you add beside them is yours.

By convention `projects/` holds code and is excluded from the prose index —
source is reached through `cfetch find`, which reads symbols, not paragraphs.
";

pub struct Created {
    pub root: PathBuf,
    pub dirs: Vec<(String, bool)>,
    pub files: Vec<(String, bool)>,
}

/// Create the tree at `root`. Additive and idempotent: an existing directory
/// is left alone and an existing file is NEVER overwritten, because this may
/// be run against a tree someone already keeps and losing their AGENT.md to a
/// scaffold would be unforgivable.
pub fn run(root: &Path) -> anyhow::Result<Created> {
    let mut created = Created { root: root.to_path_buf(), dirs: Vec::new(), files: Vec::new() };
    std::fs::create_dir_all(root)?;

    for dir in DIRS {
        let path = root.join(dir.name);
        let fresh = !path.exists();
        std::fs::create_dir_all(&path)?;
        created.dirs.push((dir.name.to_string(), fresh));
        created.files.push(write_if_absent(&path.join("README.md"), dir.readme)?);
        for child in dir.children {
            let child_path = path.join(child);
            let fresh = !child_path.exists();
            std::fs::create_dir_all(&child_path)?;
            created.dirs.push((format!("{}/{child}", dir.name), fresh));
        }
    }
    created.files.push(write_if_absent(&root.join("README.md"), ROOT_README)?);
    created.files.push(write_if_absent(&root.join(".gitignore"), GITIGNORE)?);
    Ok(created)
}

/// `(relative name, was written)`. Never truncates: a false here means the
/// file was already someone's.
fn write_if_absent(path: &Path, body: &str) -> anyhow::Result<(String, bool)> {
    let label = path.file_name().map_or_else(String::new, |n| n.to_string_lossy().to_string());
    let shown = path
        .parent()
        .and_then(Path::file_name)
        .map_or(label.clone(), |p| format!("{}/{}", p.to_string_lossy(), label));
    if path.exists() {
        return Ok((shown, false));
    }
    crate::fsutil::atomic_write(path, body)?;
    Ok((shown, true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_top_level_directory_gets_a_readme_naming_its_ring_or_its_rule() {
        let dir = tempfile::tempdir().unwrap();
        let out = run(dir.path()).unwrap();
        let expected: usize = DIRS.len() + DIRS.iter().map(|d| d.children.len()).sum::<usize>();
        assert_eq!(out.dirs.len(), expected, "every top-level dir and canonical child is reported");
        for d in DIRS {
            let readme = dir.path().join(d.name).join("README.md");
            assert!(readme.exists(), "{} has no README", d.name);
            let body = std::fs::read_to_string(&readme).unwrap();
            assert!(
                body.contains("Ring") || body.contains("ring") || body.contains("indexed"),
                "{}'s README explains neither its ring nor its rule",
                d.name
            );
        }
    }

    /// The whole point of the standard: a rule, a slice and a grant all name
    /// a top-level directory, so the set cfetch creates and the set its rules
    /// key on cannot be allowed to drift apart.
    #[test]
    fn the_created_tree_covers_every_directory_the_default_rules_name() {
        let dir = tempfile::tempdir().unwrap();
        run(dir.path()).unwrap();
        for named in ["mind", "todo", "scratch", "logs", "knowledge", "state"] {
            assert!(dir.path().join(named).is_dir(), "{named} is named by a rule but not created");
        }
    }

    /// Reserved names are documented, never created — creating them would
    /// prescribe a workflow, and leaving them undocumented would make the rule
    /// that keys on them invisible.
    #[test]
    fn reserved_subdirectories_are_documented_but_not_created() {
        let dir = tempfile::tempdir().unwrap();
        run(dir.path()).unwrap();
        for (path, _) in reserved() {
            assert!(!dir.path().join(&path).exists(), "{path} must not be created");
            let (parent, child) = path.split_once('/').unwrap();
            let readme = std::fs::read_to_string(dir.path().join(parent).join("README.md")).unwrap();
            assert!(readme.contains(child), "{parent}/README.md never mentions {child}");
        }
    }

    /// A canonical child that no README mentions is a directory people will
    /// put the wrong things in, which is the failure this whole table exists
    /// to prevent.
    #[test]
    fn every_canonical_child_is_created_and_explained() {
        let dir = tempfile::tempdir().unwrap();
        run(dir.path()).unwrap();
        for d in DIRS {
            let readme = std::fs::read_to_string(dir.path().join(d.name).join("README.md")).unwrap();
            for child in d.children {
                assert!(dir.path().join(d.name).join(child).is_dir(), "{}/{child} missing", d.name);
                assert!(readme.contains(child), "{}/README.md never mentions {child}", d.name);
            }
        }
    }

    /// This may be run against a tree someone already keeps.
    #[test]
    fn a_second_run_changes_nothing_and_never_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let mine = dir.path().join("knowledge/README.md");
        std::fs::create_dir_all(mine.parent().unwrap()).unwrap();
        std::fs::write(&mine, "my own words").unwrap();

        let first = run(dir.path()).unwrap();
        assert_eq!(std::fs::read_to_string(&mine).unwrap(), "my own words");
        assert!(
            first.files.iter().any(|(n, written)| n.ends_with("README.md") && !*written),
            "an existing file must be reported as untouched"
        );

        let second = run(dir.path()).unwrap();
        assert!(second.dirs.iter().all(|(_, fresh)| !fresh), "a second run creates no directory");
        assert!(second.files.iter().all(|(_, written)| !*written), "a second run writes no file");
    }
}
