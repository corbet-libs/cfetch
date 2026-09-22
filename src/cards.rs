//! Optional nixcards catalogue integration.
//!
//! The checkout itself is the rendezvous contract: Git's sparse-checkout
//! list is the one local selection record. cfetch and the nixcards TUI are
//! equal clients of that state; neither keeps a shadow selection.

use crate::{CardsAction, config, lockfile};
use anyhow::{Context as _, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const OFFICIAL_REPOSITORY: &str = "https://github.com/julian-corbet/nixcards-corbet-ch.git";
const CATALOG_BRANCH: &str = "cards";
const CATALOG_INDEX_FILE: &str = "catalog.json";
const CATALOG_SCHEMA: u32 = 1;
const STORE_LOCK: &str = ".nixcards-store.lock";

#[derive(Debug, Deserialize, Serialize)]
struct CatalogIndex {
    schema_version: u32,
    sets: Vec<CatalogSet>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CatalogSet {
    id: String,
    title: String,
    language: String,
    tags: Vec<String>,
    path: String,
    card_count: usize,
}

#[derive(Debug, Serialize)]
struct CardSetStatus {
    id: String,
    title: String,
    language: String,
    tags: Vec<String>,
    path: String,
    card_count: usize,
    selected: bool,
}

#[derive(Debug, Serialize)]
struct CardsStatus {
    store: String,
    initialized: bool,
    repository: Option<String>,
    revision: Option<String>,
    partial_clone_filter: Option<String>,
    selected_sets: Vec<String>,
}

pub fn run(action: CardsAction) -> anyhow::Result<()> {
    let cfg = config::Config::load()?;
    let store = store_root(&cfg.brain_root);
    match action {
        CardsAction::Init { repository } => {
            initialize(&store, &repository)?;
            println!("initialized nixcards catalogue at {}", store.display());
        }
        CardsAction::List { json } => list(&store, json)?,
        CardsAction::Select { selectors, json } => select(&store, &selectors, json)?,
        CardsAction::Sync { json } => sync(&store, json)?,
        CardsAction::Status { json } => status(&store, json)?,
        CardsAction::Tui => tui(&store)?,
    }
    Ok(())
}

pub(crate) fn store_root(brain_root: &Path) -> PathBuf {
    brain_root.join("knowledge/cards")
}

fn initialize(store: &Path, repository: &str) -> anyhow::Result<()> {
    if store.exists() {
        let mut entries = fs::read_dir(store)
            .with_context(|| format!("read existing store {}", store.display()))?;
        ensure!(
            entries.next().is_none(),
            "{} already exists and is not an initialized nixcards store",
            store.display()
        );
    }
    let parent = store.parent().context("card store has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let _lock = acquire_store_lock(store)?;
    run_command(
        Command::new("git")
            .arg("clone")
            .arg("--filter=blob:none")
            .arg("--sparse")
            .arg("--single-branch")
            .arg("--branch")
            .arg(CATALOG_BRANCH)
            .arg(repository)
            .arg(store),
        "initialize nixcards catalogue",
    )?;
    validate_checkout(store)?;
    Ok(())
}

fn list(store: &Path, json: bool) -> anyhow::Result<()> {
    // The store lock is held for reads too: `catalog_index` reads
    // catalog.json non-atomically and `selected_paths` runs git commands,
    // both of which can tear or contend while a concurrent `select`/`sync`
    // rewrites the worktree. The lock exists for exactly this rendezvous.
    let _lock = acquire_store_lock(store)?;
    let index = catalog_index(store)?;
    let selected: BTreeSet<_> = selected_paths(store)?.into_iter().collect();
    let rows: Vec<_> = index
        .sets
        .into_iter()
        .map(|set| CardSetStatus {
            selected: selected.contains(&set.path),
            id: set.id,
            title: set.title,
            language: set.language,
            tags: set.tags,
            path: set.path,
            card_count: set.card_count,
        })
        .collect();
    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
    } else {
        for row in rows {
            let marker = if row.selected { "x" } else { " " };
            println!("[{marker}]\t{}\t{}\t{}", row.id, row.language, row.title);
        }
    }
    Ok(())
}

fn select(store: &Path, selectors: &[String], json: bool) -> anyhow::Result<()> {
    let index = catalog_index(store)?;
    let ids = resolve_selectors(&index, selectors)?;
    let paths: BTreeMap<_, _> = index
        .sets
        .into_iter()
        .map(|set| (set.id, set.path))
        .collect();
    let input = ids
        .iter()
        .map(|id| paths.get(id).expect("resolved ID exists").as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let _lock = acquire_store_lock(store)?;
    git_input(
        store,
        ["sparse-checkout", "set", "--cone", "--stdin"],
        &input,
    )?;
    if json {
        println!("{}", serde_json::json!({"selected_sets": ids}));
    } else {
        println!("selected {} nixcards set(s)", ids.len());
        for id in ids {
            println!("{id}");
        }
    }
    Ok(())
}

fn sync(store: &Path, json: bool) -> anyhow::Result<()> {
    validate_checkout(store)?;
    let _lock = acquire_store_lock(store)?;
    let dirty = git_output(store, ["status", "--porcelain", "--untracked-files=no"])?;
    ensure!(
        dirty.trim().is_empty(),
        "catalogue checkout has local changes; commit them on a topic branch or restore them before syncing"
    );
    git(
        store,
        ["fetch", "--filter=blob:none", "origin", CATALOG_BRANCH],
    )?;
    git(store, ["merge", "--ff-only", "FETCH_HEAD"])?;
    // The report runs UNDER the lock this function already holds — calling
    // `status()` here deadlocked the command on its own store lock after
    // fetch and merge were already done.
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&status_report_locked(store)?)?
        );
    } else {
        println!("synchronized nixcards catalogue");
        let report = status_report_locked(store)?;
        if !report.initialized {
            println!("not initialized: {}", report.store);
        } else {
            println!(
                "revision: {}",
                report.revision.as_deref().unwrap_or("unknown")
            );
            println!("selected sets: {}", report.selected_sets.len());
        }
    }
    Ok(())
}

fn status(store: &Path, json: bool) -> anyhow::Result<()> {
    let report = status_report(store)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else if !report.initialized {
        println!("not initialized: {}", report.store);
    } else {
        println!("store: {}", report.store);
        println!(
            "repository: {}",
            report.repository.as_deref().unwrap_or("unknown")
        );
        println!(
            "revision: {}",
            report.revision.as_deref().unwrap_or("unknown")
        );
        println!(
            "partial clone filter: {}",
            report.partial_clone_filter.as_deref().unwrap_or("none")
        );
        println!("selected sets: {}", report.selected_sets.len());
        for id in report.selected_sets {
            println!("{id}");
        }
    }
    Ok(())
}

fn status_report(store: &Path) -> anyhow::Result<CardsStatus> {
    // Same store lock as the mutators: a status snapshot concurrent with a
    // sync/select would otherwise be a torn view of the worktree.
    let _lock = acquire_store_lock(store)?;
    status_report_locked(store)
}

/// The report body for a caller that already holds the store lock.
fn status_report_locked(store: &Path) -> anyhow::Result<CardsStatus> {
    if !store.join(".git").exists() {
        return Ok(CardsStatus {
            store: store.display().to_string(),
            initialized: false,
            repository: None,
            revision: None,
            partial_clone_filter: None,
            selected_sets: Vec::new(),
        });
    }
    let index = catalog_index(store)?;
    let selected_paths: BTreeSet<_> = selected_paths(store)?.into_iter().collect();
    let selected_sets = index
        .sets
        .into_iter()
        .filter(|set| selected_paths.contains(&set.path))
        .map(|set| set.id)
        .collect();
    Ok(CardsStatus {
        store: store.display().to_string(),
        initialized: true,
        repository: git_output_optional(store, ["remote", "get-url", "origin"])?
            .map(|value| value.trim().to_owned()),
        revision: git_output_optional(store, ["rev-parse", "HEAD"])?
            .map(|value| value.trim().to_owned()),
        partial_clone_filter: git_output_optional(
            store,
            ["config", "--get", "remote.origin.partialclonefilter"],
        )?
        .map(|value| value.trim().to_owned()),
        selected_sets,
    })
}

fn tui(store: &Path) -> anyhow::Result<()> {
    let status = Command::new("nixcards")
        .arg("--store")
        .arg(store)
        .status()
        .context(
            "start nixcards; install it or manage the catalogue with `cfetch cards` commands",
        )?;
    ensure!(status.success(), "nixcards exited with {status}");
    Ok(())
}

fn catalog_index(store: &Path) -> anyhow::Result<CatalogIndex> {
    validate_checkout(store)?;
    let path = store.join(CATALOG_INDEX_FILE);
    let raw = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let index: CatalogIndex =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    ensure!(
        index.schema_version == CATALOG_SCHEMA,
        "unsupported nixcards catalogue schema {}; expected {}",
        index.schema_version,
        CATALOG_SCHEMA
    );
    let mut ids = BTreeSet::new();
    let mut paths = BTreeSet::new();
    for set in &index.sets {
        let expected = set.id.replace('.', "/");
        ensure!(
            set.path == expected,
            "card set {} uses path {}; expected {}",
            set.id,
            set.path,
            expected
        );
        ensure!(ids.insert(&set.id), "duplicate card set ID {}", set.id);
        ensure!(
            paths.insert(&set.path),
            "duplicate card set path {}",
            set.path
        );
    }
    Ok(index)
}

fn validate_checkout(store: &Path) -> anyhow::Result<()> {
    ensure!(
        store.join(".git").exists(),
        "{} is not a Git checkout",
        store.display()
    );
    ensure!(
        store.join(CATALOG_INDEX_FILE).is_file(),
        "{} is not a nixcards catalogue checkout",
        store.display()
    );
    Ok(())
}

fn selected_paths(store: &Path) -> anyhow::Result<Vec<String>> {
    let output = git_output(store, ["sparse-checkout", "list"])?;
    let mut paths: Vec<_> = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(normalize_path)
        .collect();
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn resolve_selectors(index: &CatalogIndex, selectors: &[String]) -> anyhow::Result<Vec<String>> {
    let mut selected = BTreeSet::new();
    for raw in selectors {
        let selector = raw.trim().trim_end_matches(".*");
        ensure!(
            !selector.is_empty(),
            "an empty catalogue selector is not valid"
        );
        let matches: Vec<_> = index
            .sets
            .iter()
            .filter(|set| {
                set.id == selector
                    || set
                        .id
                        .strip_prefix(selector)
                        .is_some_and(|suffix| suffix.starts_with('.'))
            })
            .map(|set| set.id.clone())
            .collect();
        ensure!(
            !matches.is_empty(),
            "catalogue selector {raw:?} matches no card sets"
        );
        selected.extend(matches);
    }
    Ok(selected.into_iter().collect())
}

fn acquire_store_lock(store: &Path) -> anyhow::Result<lockfile::Lock> {
    let parent = store.parent().context("card store has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    lockfile::acquire(&parent.join(STORE_LOCK), 5_000, 0)
        .context("another nixcards store operation is still active")
}

fn normalize_path(path: &str) -> String {
    path.trim().replace('\\', "/").trim_matches('/').to_owned()
}

fn git<const N: usize>(root: &Path, args: [&str; N]) -> anyhow::Result<()> {
    run_command(
        Command::new("git").arg("-C").arg(root).args(args),
        "run Git",
    )
}

fn git_output<const N: usize>(root: &Path, args: [&str; N]) -> anyhow::Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .context("run git")?;
    ensure!(
        output.status.success(),
        "git failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    String::from_utf8(output.stdout).context("git returned invalid UTF-8")
}

fn git_output_optional<const N: usize>(
    root: &Path,
    args: [&str; N],
) -> anyhow::Result<Option<String>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .context("run git")?;
    if output.status.success() {
        return String::from_utf8(output.stdout)
            .map(Some)
            .context("git returned invalid UTF-8");
    }
    // "Absence" is any failure shape here, not only exit 1: `git remote
    // get-url origin` exits 2 when the remote does not exist, and treating
    // that as a hard error killed `cards status` on stores whose origin was
    // removed - exactly the degraded case the Option exists for.
    if matches!(output.status.code(), Some(1) | Some(2)) {
        return Ok(None);
    }
    anyhow::bail!(
        "git failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    )
}

fn git_input<const N: usize>(root: &Path, args: [&str; N], input: &str) -> anyhow::Result<()> {
    let mut child = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("run git")?;
    child
        .stdin
        .take()
        .context("open git stdin")?
        .write_all(input.as_bytes())
        .context("write git input")?;
    let output = child.wait_with_output().context("wait for git")?;
    ensure!(
        output.status.success(),
        "git failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

fn run_command(command: &mut Command, action: &str) -> anyhow::Result<()> {
    let output = command
        .output()
        .with_context(|| format!("cannot {action}"))?;
    ensure!(
        output.status.success(),
        "cannot {action}: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn fixture_index() -> CatalogIndex {
        CatalogIndex {
            schema_version: CATALOG_SCHEMA,
            sets: vec![
                CatalogSet {
                    id: "cloud.bearingpoint.interview".into(),
                    title: "Interview".into(),
                    language: "en".into(),
                    tags: vec!["cloud".into()],
                    path: "cloud/bearingpoint/interview".into(),
                    card_count: 1,
                },
                CatalogSet {
                    id: "cloud.certificates.databricks.introduction".into(),
                    title: "Databricks".into(),
                    language: "en".into(),
                    tags: vec!["certificate".into()],
                    path: "cloud/certificates/databricks/introduction".into(),
                    card_count: 1,
                },
            ],
        }
    }

    #[test]
    fn dotted_category_selects_only_descendant_sets() {
        let selected =
            resolve_selectors(&fixture_index(), &["cloud.certificates.databricks".into()]).unwrap();
        assert_eq!(selected, ["cloud.certificates.databricks.introduction"]);
    }

    #[test]
    fn card_store_is_always_inside_knowledge_and_uses_its_path_ring() {
        let brain = Path::new("/brain");
        assert_eq!(store_root(brain), brain.join("knowledge/cards"));
        assert_eq!(
            config::RingRules::default().ring_for("knowledge/cards/cloud/certificates/card.md"),
            config::UNMATCHED_RING
        );
    }

    #[test]
    fn cfetch_materializes_only_the_selected_catalogue_branch() {
        let fixture = TempDir::new().unwrap();
        let remote = fixture.path().join("catalog.git");
        let source = fixture.path().join("source");
        git_ok(Command::new("git").arg("init").arg("--bare").arg(&remote));
        git_ok(Command::new("git").arg("-C").arg(&remote).args([
            "config",
            "uploadpack.allowFilter",
            "true",
        ]));
        git_ok(
            Command::new("git")
                .arg("init")
                .arg("--initial-branch=cards")
                .arg(&source),
        );
        git_ok(Command::new("git").arg("-C").arg(&source).args([
            "config",
            "user.email",
            "fixture@example.invalid",
        ]));
        git_ok(
            Command::new("git")
                .arg("-C")
                .arg(&source)
                .args(["config", "user.name", "Fixture"]),
        );
        fs::write(
            source.join(CATALOG_INDEX_FILE),
            serde_json::to_vec_pretty(&fixture_index()).unwrap(),
        )
        .unwrap();
        for set in &fixture_index().sets {
            let directory = source.join(&set.path);
            fs::create_dir_all(&directory).unwrap();
            fs::write(directory.join("question.md"), "# Question?\n\nAnswer.\n").unwrap();
        }
        git_ok(
            Command::new("git")
                .arg("-C")
                .arg(&source)
                .args(["add", "."]),
        );
        git_ok(
            Command::new("git")
                .arg("-C")
                .arg(&source)
                .args(["commit", "-m", "fixture"]),
        );
        git_ok(
            Command::new("git")
                .arg("-C")
                .arg(&source)
                .arg("remote")
                .arg("add")
                .arg("origin")
                .arg(&remote),
        );
        git_ok(
            Command::new("git")
                .arg("-C")
                .arg(&source)
                .args(["push", "origin", "cards"]),
        );

        let store = fixture.path().join("brain/knowledge/cards");
        initialize(&store, &file_url(&remote)).unwrap();
        assert!(!store.join("cloud/bearingpoint").exists());
        assert!(!store.join("cloud/certificates").exists());
        select(&store, &["cloud.certificates.databricks".into()], false).unwrap();
        assert!(
            store
                .join("cloud/certificates/databricks/introduction/question.md")
                .is_file()
        );
        assert!(!store.join("cloud/bearingpoint").exists());
        let report = status_report(&store).unwrap();
        assert_eq!(report.partial_clone_filter.as_deref(), Some("blob:none"));
        assert_eq!(
            report.selected_sets,
            ["cloud.certificates.databricks.introduction"]
        );
    }

    fn git_ok(command: &mut Command) {
        let output = command.output().expect("run git fixture command");
        assert!(
            output.status.success(),
            "git fixture command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn file_url(path: &Path) -> String {
        let normalized = path.to_string_lossy().replace('\\', "/");
        if cfg!(windows) {
            format!("file:///{normalized}")
        } else {
            format!("file://{normalized}")
        }
    }
}
