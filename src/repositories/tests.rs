use super::*;

#[test]
#[cfg(unix)]
fn timed_out_git_helpers_cannot_keep_mutating_after_the_lock_is_released() {
    let root = tempfile::tempdir().unwrap();
    let error = git_output(
        root.path(),
        &[
            "-c",
            "alias.slow=!sleep 1; printf orphan > orphan-marker",
            "slow",
        ],
        Duration::from_millis(100),
    )
    .unwrap_err();
    assert!(error.to_string().contains("timed out"));
    std::thread::sleep(Duration::from_millis(1200));
    assert!(!root.path().join("orphan-marker").exists());
}
const TIMEOUT: Duration = Duration::from_secs(10);

struct Fixture {
    _dir: tempfile::TempDir,
    remote: PathBuf,
    a: PathBuf,
    b: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let remote = dir.path().join("remote.git");
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        git(
            dir.path(),
            &[
                "init",
                "--bare",
                "--initial-branch=main",
                remote.to_str().unwrap(),
            ],
            TIMEOUT,
        )
        .unwrap();
        git(
            dir.path(),
            &["clone", remote.to_str().unwrap(), a.to_str().unwrap()],
            TIMEOUT,
        )
        .unwrap();
        for path in [&a] {
            git(path, &["config", "user.name", "Test Writer"], TIMEOUT).unwrap();
            git(
                path,
                &["config", "user.email", "test@example.invalid"],
                TIMEOUT,
            )
            .unwrap();
        }
        std::fs::write(a.join("note.md"), "original\n").unwrap();
        commit(&a, "note.md");
        git(&a, &["push", "-u", "origin", "main"], TIMEOUT).unwrap();
        git(
            dir.path(),
            &["clone", remote.to_str().unwrap(), b.to_str().unwrap()],
            TIMEOUT,
        )
        .unwrap();
        git(&b, &["config", "user.name", "Test Writer"], TIMEOUT).unwrap();
        git(
            &b,
            &["config", "user.email", "test@example.invalid"],
            TIMEOUT,
        )
        .unwrap();
        Self {
            _dir: dir,
            remote,
            a,
            b,
        }
    }
}

fn commit(root: &Path, path: &str) {
    git(root, &["add", "--", path], TIMEOUT).unwrap();
    git(root, &["commit", "-qm", "record change"], TIMEOUT).unwrap();
}

#[test]
fn independent_edits_merge_and_publish_without_a_special_transport() {
    let f = Fixture::new();
    std::fs::write(f.a.join("first.md"), "first\n").unwrap();
    commit(&f.a, "first.md");
    std::fs::write(f.b.join("second.md"), "second\n").unwrap();
    commit(&f.b, "second.md");
    synchronize(&f.a, TIMEOUT).unwrap();
    assert_eq!(synchronize(&f.b, TIMEOUT).unwrap().state, "synchronized");
    synchronize(&f.a, TIMEOUT).unwrap();
    assert_eq!(
        std::fs::read_to_string(f.a.join("second.md")).unwrap(),
        "second\n"
    );
    assert!(f.b.join("first.md").is_file());
    assert_eq!(
        inspect(&f.a, TIMEOUT).unwrap().head,
        inspect(&f.b, TIMEOUT).unwrap().head
    );
}

#[test]
fn conflicting_edits_leave_worktree_and_index_unchanged() {
    let f = Fixture::new();
    std::fs::write(f.a.join("note.md"), "alpha\n").unwrap();
    commit(&f.a, "note.md");
    std::fs::write(f.b.join("note.md"), "beta\n").unwrap();
    commit(&f.b, "note.md");
    synchronize(&f.a, TIMEOUT).unwrap();
    let before = inspect(&f.b, TIMEOUT).unwrap().head;
    assert_eq!(synchronize(&f.b, TIMEOUT).unwrap().state, "conflict");
    assert_eq!(
        std::fs::read_to_string(f.b.join("note.md")).unwrap(),
        "beta\n"
    );
    assert_eq!(inspect(&f.b, TIMEOUT).unwrap().head, before);
    assert!(!dirty(&f.b, TIMEOUT).unwrap());
    assert!(!in_progress(&f.b, TIMEOUT).unwrap());
}

#[test]
fn dirty_and_staged_work_are_never_committed_or_replaced() {
    let f = Fixture::new();
    std::fs::write(f.a.join("note.md"), "unpublished\n").unwrap();
    git(&f.a, &["add", "--", "note.md"], TIMEOUT).unwrap();
    std::fs::write(f.a.join("note.md"), "still editing\n").unwrap();
    assert_eq!(synchronize(&f.a, TIMEOUT).unwrap().state, "dirty");
    assert_eq!(
        git(&f.a, &["show", ":note.md"], TIMEOUT).unwrap(),
        "unpublished\n"
    );
    assert_eq!(
        std::fs::read_to_string(f.a.join("note.md")).unwrap(),
        "still editing\n"
    );
}

#[test]
fn nested_ignored_repositories_are_discovered_and_locked_independently() {
    let f = Fixture::new();
    let child = f.a.join("skills/shared");
    std::fs::create_dir_all(&child).unwrap();
    git(&child, &["init", "--initial-branch=main"], TIMEOUT).unwrap();
    std::fs::write(f.a.join(".gitignore"), "/skills/shared/\n").unwrap();
    let repos = discover(std::slice::from_ref(&f.a)).unwrap();
    assert_eq!(repos.len(), 2);
    let _parent_lock = lock(&f.a, TIMEOUT).unwrap();
    assert!(lock(&f.a, TIMEOUT).is_err());
    assert!(lock(&child, TIMEOUT).is_ok());
}

#[test]
fn unavailable_remote_does_not_block_another_repository() {
    let f = Fixture::new();
    git(
        &f.a,
        &["remote", "set-url", "origin", "/missing/example.git"],
        TIMEOUT,
    )
    .unwrap();
    let statuses = run(&[f.a.clone(), f.b.clone()], true, TIMEOUT).unwrap();
    assert_eq!(statuses[0].state, "error");
    assert_eq!(statuses[1].state, "synchronized");
    assert!(f.remote.is_dir());
}
