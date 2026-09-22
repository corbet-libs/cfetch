//! Additive scaffolding for composed knowledge repositories and a selected mind.

use std::io::Write as _;
use std::path::{Path, PathBuf};

const DIRECTORIES: &[(&str, &str)] = &[
    (
        "knowledge",
        "# Knowledge\n\nHuman-readable, Obsidian-compatible knowledge. Ring 0 rules, ring 2 behaviours, ring 3 topics. Compose independently shared repositories here; parent repositories ignore child checkouts. Links can cross repository boundaries.\n",
    ),
    (
        "knowledge/rules",
        "# Rules\n\nRing 0: shared operator invariants. Never changed by automatic maintenance.\n",
    ),
    (
        "knowledge/behaviours",
        "# Behaviours\n\nRing 2: shared skills, tools, preferences and procedures. Skills may own independent repositories with different collaborators.\n",
    ),
    (
        "knowledge/behaviours/skills",
        "# Skills\n\nRing 2: reusable procedures and their supporting files.\n",
    ),
    (
        "knowledge/behaviours/tools",
        "# Tools\n\nRing 2: reusable tools and their documentation.\n",
    ),
    (
        "knowledge/secrets",
        "# Secrets\n\nEncrypted credentials in a separately restricted repository. Never indexed or injected. Decryption keys stay outside the brain.\n",
    ),
    (
        "mind",
        "# Minds\n\nSelect a mind with CFETCH_MIND; the default is the host identity. Multiple environments may select the same directory. Ring 1 contains guidance, identity and policy; ring 5 memories are provisional. Model weights remain outside Git.\n",
    ),
    (
        "mind/models",
        "# Models\n\nBulk model assets, outside the rings and Git. cfetch does not prescribe the backing storage.\n",
    ),
    (
        "todo",
        "# Tasks\n\nRing 4: independent Git repository. Tasks start in backlog, move through active or blocked, and finish in done. Move the same record and publish each transition. Scratch work belongs in ../scratch/.\n",
    ),
    (
        "scratch",
        "# Scratch\n\nDisposable work outside Git and retrieval. No ring.\n",
    ),
    (
        "logs",
        "# Logs\n\nRing 6: session and tool records, outside Git and retrieval.\n",
    ),
    (
        "projects",
        "# Projects\n\nIndependent source checkouts. Use the code index and dependency graph; curated project knowledge belongs in knowledge/projects/.\n",
    ),
];

const ROOT_README: &str = "# The brain\n\nMarkdown is the record; Obsidian is a normal editor. Knowledge, tasks and minds compose ordinary Git repositories. Each repository owns its permissions and history. Physical placement is external to cfetch.\n\nRings: 0 knowledge/rules; 1 selected mind guidance, identity and policy; 2 knowledge/behaviours; 3 knowledge topics; 4 todo; 5 selected mind memories; 6 logs. Scratch and models are outside retrieval.\n\nUse text search, vectors and the explicit link graph together. Missing or ambiguous links are never invented. Indexes are disposable local data. Shared files need no Git exchange; independent checkouts synchronize with ordinary remotes and may temporarily differ.\n";
const GITIGNORE: &str = "/logs/\n/scratch/\n/projects/\n/mind/models/\n# Independently owned checkouts; give each its own Git history.\n/knowledge/\n/todo/\n/mind/*/\n";
const CFETCHIGNORE: &str = "# Retrieval exclusions, independent of Git repository ownership.\n**/target/\n**/node_modules/\n**/.venv/\n";

#[derive(Debug)]
pub struct Created {
    pub root: PathBuf,
    pub dirs: Vec<(String, bool)>,
    pub files: Vec<(String, bool)>,
}

pub fn reserved() -> Vec<(String, &'static str)> {
    vec![(
        "knowledge/archive".into(),
        "retired knowledge, excluded from ordinary recall",
    )]
}

fn directory(root: &Path, relative: &str, body: &str, created: &mut Created) -> anyhow::Result<()> {
    let path = root.join(relative);
    let fresh = !path.exists();
    std::fs::create_dir_all(&path)?;
    created.dirs.push((relative.to_string(), fresh));
    let readme = format!("{relative}/README.md");
    created
        .files
        .push((readme.clone(), write_if_absent(&root.join(readme), body)?));
    Ok(())
}

pub fn run(root: &Path) -> anyhow::Result<Created> {
    crate::paths::validate_mind_id()?;
    let mut created = Created {
        root: root.to_path_buf(),
        dirs: Vec::new(),
        files: Vec::new(),
    };
    std::fs::create_dir_all(root)?;
    for &(relative, body) in DIRECTORIES {
        directory(root, relative, body, &mut created)?;
    }
    let mind = format!("mind/{}", crate::paths::mind_id());
    directory(
        root,
        &mind,
        "# Mind\n\nRing 1 guidance, identity and policy; ring 5 provisional memories. Select this directory independently of the machine running cfetch.\n",
        &mut created,
    )?;
    for child in ["guidance", "identity", "policy", "memories"] {
        let ring = if child == "memories" { 5 } else { 1 };
        directory(
            root,
            &format!("{mind}/{child}"),
            &format!("# {child}\n\nRing {ring}.\n"),
            &mut created,
        )?;
    }
    for state in ["backlog", "active", "blocked", "done"] {
        directory(
            root,
            &format!("todo/{state}"),
            &format!("# {state}\n\nRing 4 tasks in this state.\n"),
            &mut created,
        )?;
    }
    for (name, body) in [
        ("README.md", ROOT_README),
        (".gitignore", GITIGNORE),
        (".cfetchignore", CFETCHIGNORE),
    ] {
        created
            .files
            .push((name.into(), write_if_absent(&root.join(name), body)?));
    }
    let entry = format!(
        "# Agent entry point\n\nRead knowledge/rules/README.md, knowledge/behaviours/README.md, {mind}/README.md and the active task's STATUS.md. Navigate category indexes before opening routed documents. Direct Markdown edits are authoritative.\n"
    );
    created.files.push((
        "AGENT.md".into(),
        write_if_absent(&root.join("AGENT.md"), &entry)?,
    ));
    Ok(created)
}

fn write_if_absent(path: &Path, body: &str) -> anyhow::Result<bool> {
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(mut file) => {
            file.write_all(body.as_bytes())?;
            file.sync_all()?;
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_selected_mind_and_exactly_four_task_states_without_retired_folders() {
        let root = tempfile::tempdir().unwrap();
        run(root.path()).unwrap();
        for state in ["backlog", "active", "blocked", "done"] {
            assert!(root.path().join("todo").join(state).is_dir());
        }
        for old in [
            "state",
            "staging",
            "todo/staging",
            "todo/server-planning",
            "scratch/cfetch-staging",
            "mind/skills",
        ] {
            assert!(!root.path().join(old).exists());
        }
        assert!(
            crate::paths::mind_dir(root.path())
                .join("memories")
                .is_dir()
        );
        assert!(root.path().join("knowledge/behaviours/skills").is_dir());
    }

    #[test]
    fn repeated_initialization_preserves_all_existing_edits() {
        let root = tempfile::tempdir().unwrap();
        run(root.path()).unwrap();
        std::fs::write(root.path().join("AGENT.md"), "operator text\n").unwrap();
        std::fs::write(root.path().join(".gitignore"), "operator ignores\n").unwrap();
        let second = run(root.path()).unwrap();
        assert!(second.files.iter().all(|(_, fresh)| !fresh));
        assert_eq!(
            std::fs::read_to_string(root.path().join("AGENT.md")).unwrap(),
            "operator text\n"
        );
        assert_eq!(
            std::fs::read_to_string(root.path().join(".gitignore")).unwrap(),
            "operator ignores\n"
        );
    }
}
