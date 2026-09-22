//! Ordinary Git repositories are the sharing boundary. Each checkout is
//! independent; an offline or conflicted repository never blocks its peers.
//! Filesystem placement is outside this module's contract.

use anyhow::{Context as _, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GitConfig {
    /// Enroll existing checkouts explicitly. A directory enrolls repositories
    /// below it as well, including ignored independently owned child repos.
    pub roots: Vec<PathBuf>,
    pub enabled: bool,
    pub interval_secs: u64,
    pub timeout_secs: u64,
}

impl Default for GitConfig {
    fn default() -> Self {
        Self {
            roots: Vec::new(),
            enabled: false,
            interval_secs: 60,
            timeout_secs: 30,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RepositoryStatus {
    pub path: PathBuf,
    pub state: String,
    pub branch: Option<String>,
    pub head: Option<String>,
    pub ahead: usize,
    pub behind: usize,
    pub detail: Option<String>,
}

impl RepositoryStatus {
    fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            state: "unknown".into(),
            branch: None,
            head: None,
            ahead: 0,
            behind: 0,
            detail: None,
        }
    }
}

/// Never follow directory symlinks or enter Git internals. Filesystem mount
/// points behave like any other directory. Unreadable roots are errors,
/// rather than silently turning a missing checkout into an empty brain.
pub fn discover(roots: &[PathBuf]) -> anyhow::Result<Vec<PathBuf>> {
    let mut seen = BTreeSet::new();
    let mut repositories = BTreeSet::new();
    let mut pending: Vec<PathBuf> = roots.to_vec();
    while let Some(path) = pending.pop() {
        let path = path
            .canonicalize()
            .with_context(|| format!("open repository root {}", path.display()))?;
        if !seen.insert(path.clone()) {
            continue;
        }
        ensure!(
            seen.len() <= 100_000,
            "repository discovery exceeds 100000 directories; enroll narrower roots"
        );
        if path.join(".git").exists() {
            repositories.insert(path.clone());
        }
        for entry in std::fs::read_dir(&path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir()
                && !entry.file_name().to_string_lossy().starts_with('.')
                && !matches!(
                    entry.file_name().to_str(),
                    Some("node_modules" | "target" | "models")
                )
            {
                pending.push(entry.path());
            }
        }
    }
    Ok(repositories.into_iter().collect())
}

struct RepositoryLock {
    path: PathBuf,
}

impl Drop for RepositoryLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir(&self.path);
    }
}

/// Atomic directory creation coordinates all cfetch processes seeing this
/// Git common directory. A crash leaves a visible lock requiring inspection;
/// no clock-based lease ever steals a live writer's lock.
fn lock(root: &Path, timeout: Duration) -> anyhow::Result<RepositoryLock> {
    let common = git(root, &["rev-parse", "--git-common-dir"], timeout)?;
    let path = root.join(common.trim()).join("cfetch-repository.lock");
    std::fs::create_dir(&path)
        .with_context(|| format!("repository busy or lock inaccessible: {}", path.display()))?;
    Ok(RepositoryLock { path })
}

fn git_output(root: &Path, args: &[&str], timeout: Duration) -> anyhow::Result<(bool, String)> {
    use std::io::{Read as _, Seek as _};
    let mut output = tempfile::tempfile()?;
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(root)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GCM_INTERACTIVE", "Never")
        .env("GIT_MERGE_AUTOEDIT", "no")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .env_remove("GIT_COMMON_DIR")
        .env_remove("GIT_OBJECT_DIRECTORY")
        .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES")
        .stdin(Stdio::null())
        .stdout(Stdio::from(output.try_clone()?))
        // Remote diagnostics can contain credential-bearing URLs. State and
        // operation are reported, never raw transport stderr.
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }
    let mut child = command.spawn().context("start Git")?;
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() >= timeout {
            #[cfg(unix)]
            {
                let _ = rustix::process::kill_process_group(
                    rustix::process::Pid::from_child(&child),
                    rustix::process::Signal::KILL,
                );
            }
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!(
                "Git {} timed out; inspect repository state before retrying",
                args[0]
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    output.rewind()?;
    let mut bytes = Vec::new();
    output.take(4 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 4 * 1024 * 1024, "Git response exceeds 4 MiB");
    Ok((
        status.success(),
        String::from_utf8(bytes).context("Git returned non-UTF-8 output")?,
    ))
}

fn git(root: &Path, args: &[&str], timeout: Duration) -> anyhow::Result<String> {
    let (ok, output) = git_output(root, args, timeout)?;
    ensure!(
        ok,
        "Git {} failed; repository changes were not discarded",
        args[0]
    );
    Ok(output)
}

fn optional(root: &Path, args: &[&str], timeout: Duration) -> anyhow::Result<Option<String>> {
    let (ok, value) = git_output(root, args, timeout)?;
    Ok(ok.then(|| value.trim().to_string()))
}

fn in_progress(root: &Path, timeout: Duration) -> anyhow::Result<bool> {
    for name in [
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "rebase-merge",
        "rebase-apply",
    ] {
        let path = git(root, &["rev-parse", "--git-path", name], timeout)?;
        if root.join(path.trim()).exists() {
            return Ok(true);
        }
    }
    Ok(false)
}

fn dirty(root: &Path, timeout: Duration) -> anyhow::Result<bool> {
    Ok(!git(
        root,
        &["status", "--porcelain=v1", "--untracked-files=normal"],
        timeout,
    )?
    .is_empty())
}

fn counts(root: &Path, remote: &str, timeout: Duration) -> anyhow::Result<(usize, usize)> {
    let value = git(
        root,
        &[
            "rev-list",
            "--left-right",
            "--count",
            &format!("HEAD...{remote}"),
        ],
        timeout,
    )?;
    let mut parts = value.split_whitespace();
    Ok((
        parts.next().context("missing ahead count")?.parse()?,
        parts.next().context("missing behind count")?.parse()?,
    ))
}

fn inspect(root: &Path, timeout: Duration) -> anyhow::Result<RepositoryStatus> {
    let mut result = RepositoryStatus::new(root);
    result.branch = optional(
        root,
        &["symbolic-ref", "--quiet", "--short", "HEAD"],
        timeout,
    )?;
    result.head = optional(root, &["rev-parse", "--verify", "HEAD^{commit}"], timeout)?;
    result.state = if in_progress(root, timeout)? {
        "conflict"
    } else if result.head.is_none() {
        "unborn"
    } else if result.branch.is_none() {
        "detached"
    } else if dirty(root, timeout)? {
        "dirty"
    } else if let Some(upstream) = optional(
        root,
        &["rev-parse", "--verify", "@{upstream}^{commit}"],
        timeout,
    )? {
        (result.ahead, result.behind) = counts(root, &upstream, timeout)?;
        match (result.ahead, result.behind) {
            (0, 0) => "current",
            (_, 0) => "ahead",
            (0, _) => "behind",
            _ => "diverged",
        }
    } else {
        "untracked"
    }
    .into();
    Ok(result)
}

/// Sync only already committed work on an attached branch with an upstream.
/// User/editor changes are never swept into an automatic commit. Clean text
/// merges are previewed in Git's object store so conflicts do not write
/// conflict markers into live Markdown. No reset, stash, rebase or force push.
fn synchronize(root: &Path, timeout: Duration) -> anyhow::Result<RepositoryStatus> {
    let _lock = lock(root, timeout)?;
    let mut result = inspect(root, timeout)?;
    if matches!(
        result.state.as_str(),
        "conflict" | "dirty" | "unborn" | "detached" | "untracked"
    ) {
        return Ok(result);
    }
    let branch = result.branch.as_deref().context("missing branch")?;
    let remote = git(
        root,
        &["config", "--get", &format!("branch.{branch}.remote")],
        timeout,
    )?;
    let target = git(
        root,
        &["config", "--get", &format!("branch.{branch}.merge")],
        timeout,
    )?;
    let (remote, target) = (remote.trim(), target.trim());
    ensure!(
        !remote.is_empty() && target.starts_with("refs/heads/") && !target.contains('\n'),
        "upstream must identify one branch"
    );
    git(
        root,
        &[
            "fetch",
            "--no-tags",
            "--no-recurse-submodules",
            "--",
            remote,
            target,
        ],
        timeout,
    )?;
    let fetched = git(
        root,
        &["rev-parse", "--verify", "FETCH_HEAD^{commit}"],
        timeout,
    )?;
    let fetched = fetched.trim();
    // Re-read after the network wait: another editor may have started work.
    if dirty(root, timeout)? || in_progress(root, timeout)? {
        return inspect(root, timeout);
    }
    let observed = inspect(root, timeout)?;
    if observed.branch != result.branch || observed.head != result.head {
        return Ok(RepositoryStatus {
            state: "changed".into(),
            detail: Some("checkout changed during fetch; retrying later".into()),
            ..observed
        });
    }
    (result.ahead, result.behind) = counts(root, fetched, timeout)?;
    if result.behind > 0 {
        if result.ahead > 0 {
            let (clean, _) = git_output(
                root,
                &["merge-tree", "--write-tree", "HEAD", fetched],
                timeout,
            )?;
            if !clean {
                result.state = "conflict".into();
                result.detail = Some(
                    "upstream cannot be merged automatically; working files are unchanged".into(),
                );
                return Ok(result);
            }
        }
        git(
            root,
            &["merge", "--no-edit", "--no-stat", "--no-autostash", fetched],
            timeout,
        )?;
    }
    if counts(root, fetched, timeout)?.0 > 0 {
        let refspec = format!("refs/heads/{branch}:{target}");
        git(
            root,
            &[
                "push",
                "--porcelain",
                "--no-follow-tags",
                "--",
                remote,
                &refspec,
            ],
            timeout,
        )?;
    }
    result = inspect(root, timeout)?;
    // A successful fetch/merge/push establishes this state even for remotes
    // whose custom fetch refspec does not update an upstream tracking ref.
    result.state = "synchronized".into();
    result.ahead = 0;
    result.behind = 0;
    Ok(result)
}

pub fn run(
    roots: &[PathBuf],
    sync: bool,
    timeout: Duration,
) -> anyhow::Result<Vec<RepositoryStatus>> {
    ensure!(!timeout.is_zero(), "Git timeout must be positive");
    let repositories = discover(roots)?;
    Ok(repositories
        .iter()
        .map(|root| {
            let result = if sync {
                synchronize(root, timeout)
            } else {
                inspect(root, timeout)
            };
            result.unwrap_or_else(|error| {
                let mut status = RepositoryStatus::new(root);
                status.state = "error".into();
                status.detail = Some(format!("{error:#}"));
                status
            })
        })
        .collect())
}

pub fn worker(cfg: crate::config::Config, stopping: impl Fn() -> bool) {
    let timeout = Duration::from_secs(cfg.git.timeout_secs);
    let roots: Vec<_> = cfg.git.roots.iter().map(|path| cfg.resolve(path)).collect();
    while !stopping() {
        match run(&roots, true, timeout) {
            Ok(status) => {
                if let Ok(body) = serde_json::to_string_pretty(&status) {
                    let _ = crate::fsutil::atomic_write(
                        &crate::paths::state_dir().join("repositories.json"),
                        &body,
                    );
                }
            }
            Err(error) => eprintln!("cfetch Git: {error:#}"),
        }
        for _ in 0..cfg.git.interval_secs {
            if stopping() {
                return;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}

#[cfg(test)]
mod tests;
