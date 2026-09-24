//! Local index lifecycle and bounded freshness checks.
//! File notifications and coalesced background fingerprint passes update the
//! catalog. Query threads never walk the source tree.
//! Only disposable local indexes are updated here. Git handles sharing.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError, mpsc};
use std::time::{Duration, Instant};

use notify::Watcher as _;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::{index, paths};

/// Bounded wait for the drain barrier; on expiry the query answers anyway,
/// labeled stale.
pub const BARRIER_TIMEOUT: Duration = Duration::from_secs(5);
/// Event batches settle for this long before a rebuild picks them up.
const DEBOUNCE: Duration = Duration::from_millis(50);
/// Cadence of the stat-fingerprint correctness backstop — and of the daemon's
/// own code-scan refresh, which rides the same tick.
pub const FINGERPRINT_INTERVAL: Duration = Duration::from_secs(60);
// ---- watcher ordering capability ----

/// Does this host's watcher backend deliver events in an order the drain
/// barrier may reason about?
///
/// Every production filesystem uses content fingerprints. The former ordered
/// barrier remains only as a regression fixture for the shared event worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierMode {
    /// FIFO event delivery: the sentinel proves coverage.
    #[cfg(test)]
    Ordered,
    /// No usable ordering: the tree fingerprint proves coverage.
    Unordered,
}

impl BarrierMode {
    /// Operator-facing label. Names the MECHANISM in force, never a guarantee
    /// the platform cannot give.
    pub fn label(self) -> &'static str {
        match self {
            #[cfg(test)]
            BarrierMode::Ordered => "ordered (sentinel)",
            BarrierMode::Unordered => "unordered (fingerprint)",
        }
    }
}

/// Select the filesystem-independent production freshness barrier.
pub fn detected_mode() -> BarrierMode {
    BarrierMode::Unordered
}

/// Why the rebuild worker was woken. `Barrier` additionally FORCES the
/// stat-fingerprint pass and skips the debounce: a query is blocked on it, and
/// on an unordered backend it cannot wait for an event that may never arrive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wake {
    Event,
    Barrier,
}

// ---- wire types (shared by server responses and remote clients) ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireHit {
    pub cite: String,
    pub path: String,
    pub ring: u8,
    pub start_line: usize,
    pub end_line: usize,
    pub snippet: String,
    #[serde(default)]
    pub mirrors: Vec<String>,
}

impl From<index::Hit> for WireHit {
    fn from(h: index::Hit) -> Self {
        WireHit {
            cite: h.cite,
            path: h.path,
            ring: h.ring,
            start_line: h.start_line,
            end_line: h.end_line,
            snippet: h.snippet,
            mirrors: h.mirrors,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireBlock {
    pub cite: String,
    pub path: String,
    pub ring: u8,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

impl From<index::Block> for WireBlock {
    fn from(b: index::Block) -> Self {
        WireBlock {
            cite: b.cite,
            path: b.path,
            ring: b.ring,
            start_line: b.start_line,
            end_line: b.end_line,
            text: b.text,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireFindHit {
    pub path: String,
    pub name: Option<String>,
    pub kind: Option<String>,
    pub start_line: usize,
    pub end_line: usize,
    pub token_estimate: u64,
}

impl From<crate::code::FindHit> for WireFindHit {
    fn from(h: crate::code::FindHit) -> Self {
        WireFindHit {
            path: h.path,
            name: h.name,
            kind: h.kind,
            start_line: h.start_line,
            end_line: h.end_line,
            token_estimate: h.token_estimate,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireMap {
    pub lines: Vec<String>,
    pub total_files: usize,
    pub focus_matched: bool,
}

impl From<crate::graph::RepoMap> for WireMap {
    fn from(m: crate::graph::RepoMap) -> Self {
        WireMap {
            lines: m.lines,
            total_files: m.total_files,
            focus_matched: m.focus_matched,
        }
    }
}

// ---- server state ----

#[derive(Debug, Default)]
struct Progress {
    /// Real fs event batches observed by the watcher.
    pending: u64,
    /// `pending` value covered by the last committed index update.
    applied: u64,
    /// Highest barrier sentinel sequence the watcher has observed.
    sentinel_seen: u64,
    /// The startup backstop (or first rebuild) has completed — before this,
    /// nothing may claim freshness: events may have been missed while down.
    settled: bool,
    /// Tree watch registration has finished. Freshness additionally requires
    /// this: between listener bind and full registration, a write in a
    /// not-yet-watched directory would be invisible to the barrier.
    watches_ready: bool,
    /// Last index-update failure; cleared by the next success.
    last_error: Option<String>,
    /// Stat fingerprint of the tree state the COMMITTED catalog describes —
    /// the value stored in the index meta, republished here so an unordered
    /// barrier can test coverage without opening the catalog.
    applied_fingerprint: Option<String>,
    /// Lower bound on when the stat walk behind the latest committed pass
    /// BEGAN. Strict queries require this start to follow their entry time;
    /// a pass begun earlier cannot establish read-your-writes coverage.
    /// Only ever set by a pass that actually walked.
    applied_walk_start: Option<Instant>,
}

pub struct ServeState {
    pub origin: String,
    state_dir: PathBuf,
    barrier_dir: PathBuf,
    /// Which coverage proof this host's watcher backend can support.
    mode: BarrierMode,
    /// Lets the unordered barrier ask the rebuild worker for a pass NOW.
    wake: Option<mpsc::Sender<Wake>>,
    refresh_queued: AtomicBool,
    progress: Mutex<Progress>,
    cv: Condvar,
    #[cfg(test)]
    barrier_seq: AtomicU64,
    pub generation: AtomicU64,
    pub last_barrier_ms: AtomicU64,
    /// Most recent measured background walk cost (0 = never measured).
    /// A past fast walk never licenses synchronous query-side filesystem I/O.
    last_walk_ms: AtomicU64,
}

pub struct BarrierOutcome {
    pub fresh: bool,
    pub waited_ms: u64,
    pub note: Option<String>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl ServeState {
    pub(crate) fn new(origin: String, state_dir: PathBuf, barrier_dir: PathBuf) -> Self {
        // CANONICALIZE: macOS reports fs events under the canonical path
        // (/var is a symlink to /private/var), so a sentinel written to the
        // as-given path would never match the event path and every barrier
        // would time out — the exact failure the macOS CI matrix caught on
        // its first run. inotify reports paths as-watched, so canonicalizing
        // both sides is correct on every platform.
        let barrier_dir = std::fs::canonicalize(&barrier_dir).unwrap_or(barrier_dir);
        ServeState {
            origin,
            state_dir,
            barrier_dir,
            mode: detected_mode(),
            wake: None,
            refresh_queued: AtomicBool::new(false),
            progress: Mutex::new(Progress::default()),
            cv: Condvar::new(),
            #[cfg(test)]
            barrier_seq: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            last_barrier_ms: AtomicU64::new(0),
            last_walk_ms: AtomicU64::new(0),
        }
    }

    /// Select a regression fixture's barrier.
    #[cfg(test)]
    pub(crate) fn with_mode(mut self, mode: BarrierMode) -> Self {
        self.mode = mode;
        self
    }

    fn with_wake(mut self, wake: mpsc::Sender<Wake>) -> Self {
        self.wake = Some(wake);
        self
    }

    /// Which coverage proof is in force here — reported by `serve-status` so
    /// an operator sees whether this platform is on the fast or the
    /// sound-and-slower path.
    pub fn mode(&self) -> BarrierMode {
        self.mode
    }

    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }

    /// The drain barrier: serve-fresh-or-wait, bounded. Which coverage proof
    /// it uses is the platform's answer, not a preference — see
    /// [`BarrierMode`].
    pub fn barrier(&self, timeout: Duration) -> BarrierOutcome {
        match self.mode {
            #[cfg(test)]
            BarrierMode::Ordered => self.barrier_ordered(timeout),
            BarrierMode::Unordered => self.barrier_unordered(timeout),
        }
    }

    /// Coverage by ORDER. Unchanged from the original barrier: two condvar
    /// waits and a sentinel write, no tree walk. Nothing on this path may
    /// acquire a cost that only the unordered path needs.
    #[cfg(test)]
    fn barrier_ordered(&self, timeout: Duration) -> BarrierOutcome {
        let start = Instant::now();
        let deadline = start + timeout;
        let seq = self.barrier_seq.fetch_add(1, Ordering::SeqCst) + 1;
        let sentinel = self.barrier_dir.join(format!("b{seq}"));
        let mut fresh = true;
        let mut note = None;
        if std::fs::write(&sentinel, seq.to_string()).is_ok() {
            if !self.wait_until(deadline, |p| p.sentinel_seen >= seq) {
                fresh = false;
                note = Some(
                    "barrier timeout: the fs watcher has not observed the barrier sentinel"
                        .to_string(),
                );
            }
        } else {
            fresh = false;
            note = Some("barrier unavailable: could not write the barrier sentinel".to_string());
        }
        if fresh {
            let target = lock(&self.progress).pending;
            if !self.wait_until(deadline, |p| {
                p.settled && p.watches_ready && p.applied >= target
            }) {
                fresh = false;
                let err = lock(&self.progress).last_error.clone();
                note = Some(match err {
                    Some(e) => format!(
                        "barrier timeout: pending changes not applied (last index error: {e})"
                    ),
                    None => "barrier timeout: pending changes not yet applied".to_string(),
                });
            }
        }
        let _ = std::fs::remove_file(&sentinel);
        self.finish(start, fresh, note)
    }

    /// Strict freshness is proved only by a successful background walk that
    /// began after this request. Waiting is bounded; filesystem work never
    /// runs on the query thread, even if the previous walk was fast.
    fn barrier_unordered(&self, timeout: Duration) -> BarrierOutcome {
        let start = Instant::now();
        let target = lock(&self.progress).pending;
        self.request_pass();
        let fresh = self.wait_until(start + timeout, |p| {
            p.settled
                && p.watches_ready
                && p.applied >= target
                && p.last_error.is_none()
                && p.applied_walk_start.is_some_and(|walk| walk >= start)
        });
        let note = if fresh {
            None
        } else {
            let error = lock(&self.progress).last_error.clone();
            Some(match error {
                Some(e) => format!("freshness unavailable: background scan has not covered this request (last index error: {e})"),
                None => "freshness unavailable: background scan has not covered this request before the deadline".to_string(),
            })
        };
        self.finish(start, fresh, note)
    }

    /// A cached snapshot makes no source-tree freshness claim. At most one
    /// additional background pass is queued while the worker is busy.
    pub(crate) fn cached(&self) -> BarrierOutcome {
        self.request_pass();
        let progress = lock(&self.progress);
        let note = match &progress.last_error {
            Some(e) => format!(
                "cached snapshot; freshness unverified; background refresh requested (last index error: {e})"
            ),
            None => {
                "cached snapshot; freshness unverified; background refresh requested".to_string()
            }
        };
        BarrierOutcome {
            fresh: false,
            waited_ms: 0,
            note: Some(note),
        }
    }

    fn finish(&self, start: Instant, fresh: bool, note: Option<String>) -> BarrierOutcome {
        let waited_ms = start.elapsed().as_millis() as u64;
        self.last_barrier_ms.store(waited_ms, Ordering::Relaxed);
        BarrierOutcome {
            fresh,
            waited_ms,
            note,
        }
    }

    /// Records the cost of a background stat walk.
    pub(crate) fn note_walk_cost(&self, walk: Duration) {
        self.last_walk_ms
            .store(walk.as_millis() as u64, Ordering::Relaxed);
    }

    /// Asks the rebuild worker for an immediate stat-fingerprint pass. A no-op
    /// on a state with no worker (unit tests): the barrier then times out and
    /// labels the answer stale, which is the correct answer.
    fn request_pass(&self) {
        if let Some(tx) = &self.wake
            && !self.refresh_queued.swap(true, Ordering::AcqRel)
            && tx.send(Wake::Barrier).is_err()
        {
            self.refresh_queued.store(false, Ordering::Release);
        }
    }

    fn wait_until(&self, deadline: Instant, done: impl Fn(&Progress) -> bool) -> bool {
        let mut guard = lock(&self.progress);
        loop {
            if done(&guard) {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            guard = self
                .cv
                .wait_timeout(guard, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    /// Watcher glue: a real (non-sentinel) event batch was observed.
    pub(crate) fn note_event(&self) {
        let mut p = lock(&self.progress);
        p.pending += 1;
        drop(p);
        self.cv.notify_all();
    }

    /// Watcher glue: barrier sentinel `seq` was observed.
    pub(crate) fn note_sentinel(&self, seq: u64) {
        let mut p = lock(&self.progress);
        if seq > p.sentinel_seen {
            p.sentinel_seen = seq;
        }
        drop(p);
        self.cv.notify_all();
    }

    /// Worker glue: an index update covering `target` committed at
    /// `generation`, with no tree walk to report.
    pub(crate) fn mark_applied(&self, target: u64, generation: u64) {
        self.mark_pass(target, generation, None, None);
    }

    /// Worker glue for a pass that WALKED the tree: `fingerprint` is the tree
    /// state the committed catalog now describes, `walk_start` a lower bound
    /// on when that walk began.
    ///
    /// `walk_start` must be `None` for any pass that did not stat the whole
    /// tree — claiming a walk that did not happen is exactly the silent
    /// staleness the unordered barrier exists to prevent.
    pub(crate) fn mark_pass(
        &self,
        target: u64,
        generation: u64,
        fingerprint: Option<String>,
        walk_start: Option<Instant>,
    ) {
        self.generation.store(generation, Ordering::Relaxed);
        let mut p = lock(&self.progress);
        if target > p.applied {
            p.applied = target;
        }
        if fingerprint.is_some() {
            p.applied_fingerprint = fingerprint;
        }
        if let Some(w) = walk_start
            && p.applied_walk_start.is_none_or(|seen| w > seen)
        {
            p.applied_walk_start = Some(w);
        }
        p.settled = true;
        p.last_error = None;
        drop(p);
        crate::runtime_status::record_generation(
            crate::runtime_status::MemoryRoute::Local,
            generation,
        );
        self.cv.notify_all();
    }

    /// Worker glue: an index update failed. `applied` does not advance and
    /// `settled` is not set — barriers will (correctly) label answers stale.
    pub(crate) fn mark_error(&self, err: String) {
        let mut p = lock(&self.progress);
        p.last_error = Some(err);
        drop(p);
        self.cv.notify_all();
    }

    fn pending_now(&self) -> u64 {
        lock(&self.progress).pending
    }

    /// The startup backstop has applied AND the tree watches are registered:
    /// the index describes the tree and nothing can slip past the barrier.
    pub(crate) fn is_settled(&self) -> bool {
        let p = lock(&self.progress);
        p.settled && p.watches_ready
    }

    /// Registration thread glue: every tree watch is in place.
    pub(crate) fn mark_watches_ready(&self) {
        lock(&self.progress).watches_ready = true;
        self.cv.notify_all();
    }

    pub(crate) fn watches_ready(&self) -> bool {
        lock(&self.progress).watches_ready
    }

    fn applied_now(&self) -> u64 {
        lock(&self.progress).applied
    }
}

pub fn origin_of(_cfg: &Config) -> String {
    crate::paths::host_id()
}

/// Keeps the watcher alive for the daemon's lifetime.
pub struct ServeHandle {
    pub state: Arc<ServeState>,
    _watcher: Arc<Mutex<notify::RecommendedWatcher>>,
}

/// Directories the INDEXER would walk — the exact set worth watching.
///
/// Three defects made a recursive watch on the tree root undeployable on a
/// real brain, all found by deploying it (2026-08-21):
///   1. it followed symlinks — a wineprefix `dosdevices/z: -> /` inside a
///      checkout sent the watch walking the entire root filesystem;
///   2. it watched subtrees the indexer excludes (`projects/`, gitignored
///      scratch dumps), needing ~524k watches against a 524,288 default;
///   3. registration ran before the listener bound, so a restart was a
///      multi-minute serving outage.
///
/// Reusing the indexer's own walker settles 1 and 2 by construction: the same
/// `ignore` walker, the same gitignore/hidden rules, `follow_links(false)`,
/// and the same exclusion predicate — so the watch set is exactly the index
/// set, never a byte more. 3 is settled by registering in the background
/// (see `start`), with `settled` withheld until registration completes so no
/// answer claims freshness it cannot have.
///
/// The same walk also tallies what the indexer would READ, because defect 2
/// is a recurring condition and not a one-time bug: a tree keeps growing.
/// When registration fails anyway, [`scoping_advice`] names the subtrees to
/// scope out — measured, not guessed at from disk usage.
fn watchable_dirs(brain_root: &Path, rules: &crate::config::RingRules) -> WatchScope {
    let mut scope = WatchScope {
        dirs: vec![brain_root.to_path_buf()],
        census: index::DocCensus::default(),
    };
    let mut seen_dirs = std::collections::HashSet::from([brain_root.to_path_buf()]);
    let mut seen_files = std::collections::HashSet::new();
    let mut walkers = vec![crate::index::brain_walker(brain_root, rules).build()];
    if let Some(builder) = crate::index::managed_cards_walker(brain_root) {
        walkers.push(builder.build());
    }
    for walker in walkers {
        for entry in walker.flatten() {
            let Some(kind) = entry.file_type() else {
                continue;
            };
            let Ok(rel) = entry.path().strip_prefix(brain_root) else {
                continue;
            };
            // Same canonical `/`-separated form the indexer derives, so the watch
            // set stays EXACTLY the index set on every platform — a
            // backslash-separated `mind\secrets` would match no exclusion.
            let rel = crate::index::rel_doc_path(rel);
            if !kind.is_dir() {
                // The census rides along on the walk the watcher has to do
                // anyway. File entries carry their kind from the directory read,
                // so counting them costs no stat — and a stat per file is the one
                // thing that must not be added to a startup already slow enough
                // to need this advice.
                if kind.is_file()
                    && seen_files.insert(entry.path().to_path_buf())
                    && crate::index::indexable_doc(&rel, rules)
                {
                    scope.census.record(&rel);
                }
                continue;
            }
            if rel.is_empty()
                || crate::index::excluded_dir(&rel, rules)
                || !seen_dirs.insert(entry.path().to_path_buf())
            {
                continue;
            }
            scope.dirs.push(entry.path().to_path_buf());
        }
    }
    scope
}

/// The watch enumeration's two products: the directories to register, and
/// what the indexer would read under each of them.
struct WatchScope {
    dirs: Vec<PathBuf>,
    census: index::DocCensus,
}

/// A subtree holding this much of everything the indexer reads is worth
/// naming as a scoping candidate. Low enough that several can be named at
/// once, high enough that a healthy tree yields nothing.
const SCOPE_MIN_SHARE: f64 = 0.15;

/// Turns "watch registration failed" into something to do about it.
///
/// This is the one moment a tree's size becomes a correctness problem the
/// operator can see: writes in unwatched directories now wait for the 60s
/// backstop instead of waking the daemon. The candidates are ranked by what
/// the indexer READS (see [`index::DocCensus`]) rather than by bytes on disk,
/// because those two orders disagree exactly where it matters — the directory
/// that exhausts a watch table is full of small markdown files, which is the
/// directory `du` ranks last.
fn scoping_advice(census: &index::DocCensus) -> Option<String> {
    let hits = census.concentrations(SCOPE_MIN_SHARE);
    if hits.is_empty() {
        return None;
    }
    let total = census.total();
    let named: Vec<String> = hits
        .iter()
        .map(|(prefix, n)| format!("{prefix}/ ({n} of {total})"))
        .collect();
    Some(format!(
        "cfetch serve: most of what is indexed sits under {} — a `{}` naming those scopes them \
         out of cfetch without hiding them from git",
        named.join(", "),
        index::IGNORE_FILE
    ))
}

/// Registers one non-recursive watch per indexable directory. Non-recursive
/// is what keeps a symlinked directory from dragging its target in: notify
/// only ever sees the paths we hand it.
fn register_watches(
    watcher: &Arc<Mutex<notify::RecommendedWatcher>>,
    dirs: &[PathBuf],
) -> (usize, usize) {
    let mut ok = 0usize;
    let mut failed = 0usize;
    for dir in dirs {
        let res = lock(watcher).watch(dir, notify::RecursiveMode::NonRecursive);
        match res {
            Ok(()) => ok += 1,
            // A directory that vanished between walk and watch, or a watch
            // limit hit: the 60s fingerprint backstop still covers it.
            Err(_) => failed += 1,
        }
    }
    (ok, failed)
}

#[cfg(test)]
mod watch_scope_tests {
    use super::*;

    /// The three deployment defects, as tests: never follow a symlink out of
    /// the tree, never watch what the indexer excludes, and watch only
    /// directories that exist.
    #[test]
    fn watchable_dirs_skips_symlinks_and_excluded_subtrees() {
        let brain = tempfile::tempdir().unwrap();
        for d in [
            "knowledge/hosts",
            "knowledge/behaviours",
            "mind/secrets",
            "logs/x",
            "projects/repo/src",
            "knowledge/archive/old",
        ] {
            std::fs::create_dir_all(brain.path().join(d)).unwrap();
        }
        // The wineprefix-style escape hatch that walked the whole rootfs.
        // Creating a directory symlink is unprivileged on unix only; the
        // exclusion half of this test runs on every platform.
        #[cfg(unix)]
        let outside = {
            let outside = tempfile::tempdir().unwrap();
            std::fs::create_dir_all(outside.path().join("deep/deeper")).unwrap();
            std::os::unix::fs::symlink(outside.path(), brain.path().join("knowledge/z")).unwrap();
            outside
        };

        let dirs = watchable_dirs(brain.path(), &crate::config::RingRules::default()).dirs;
        let rel: Vec<String> = dirs
            .iter()
            .map(|d| crate::index::rel_doc_path(d.strip_prefix(brain.path()).unwrap()))
            .collect();

        assert!(rel.iter().any(|r| r == "knowledge/hosts"));
        assert!(rel.iter().any(|r| r == "knowledge/behaviours"));
        assert!(
            !rel.iter().any(|r| r.starts_with("mind/secrets")),
            "secrets never watched: {rel:?}"
        );
        assert!(
            !rel.iter().any(|r| r.starts_with("logs")),
            "logs excluded: {rel:?}"
        );
        assert!(
            !rel.iter().any(|r| r.starts_with("projects")),
            "projects excluded: {rel:?}"
        );
        assert!(
            !rel.iter().any(|r| r.starts_with("knowledge/archive")),
            "archive excluded: {rel:?}"
        );
        #[cfg(unix)]
        assert!(
            !dirs.iter().any(|d| d.starts_with(outside.path())),
            "a symlinked directory must never drag its target in: {dirs:?}"
        );
    }

    /// The overlay has to reach the watcher, not only the indexer: a subtree
    /// scoped out of cfetch must be neither indexed nor watched, or the watch
    /// set and the index set part ways again.
    ///
    /// And the advice that names such a subtree is measured from what the
    /// indexer READS. The binary dump here outweighs every markdown file in
    /// the tree by three orders of magnitude and must not appear anywhere in
    /// the ranking.
    #[test]
    fn watch_scope_honors_the_overlay_and_ranks_by_what_the_indexer_reads() {
        let brain = tempfile::tempdir().unwrap();
        let root = brain.path();
        let dirs = [
            "knowledge/hosts",
            "knowledge/generated/api",
            "knowledge/behaviours",
            "bulk",
            "scratch/deep",
        ];
        for d in dirs {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        for i in 0..20 {
            std::fs::write(
                root.join(format!("knowledge/generated/api/p{i}.md")),
                "# p\n",
            )
            .unwrap();
        }
        for i in 0..5 {
            std::fs::write(root.join(format!("knowledge/hosts/h{i}.md")), "# h\n").unwrap();
        }
        for i in 0..4 {
            std::fs::write(root.join(format!("knowledge/behaviours/m{i}.md")), "# m\n").unwrap();
        }
        std::fs::write(root.join("AGENT.md"), "# agent\n").unwrap();
        for i in 0..3 {
            std::fs::write(root.join(format!("bulk/blob{i}.bin")), vec![0u8; 64 * 1024]).unwrap();
        }
        std::fs::write(root.join("scratch/deep/notes.md"), "# scratch\n").unwrap();
        std::fs::write(root.join(".cfetchignore"), "scratch/\n").unwrap();

        let scope = watchable_dirs(root, &crate::config::RingRules::default());
        let rel: Vec<String> = scope
            .dirs
            .iter()
            .map(|d| crate::index::rel_doc_path(d.strip_prefix(root).unwrap()))
            .collect();
        assert!(rel.iter().any(|r| r == "knowledge/hosts"));
        assert!(
            !rel.iter().any(|r| r.starts_with("scratch")),
            "the overlay must reach the watcher too: {rel:?}"
        );

        assert_eq!(
            scope.census.total(),
            30,
            "only markdown the indexer would read is counted"
        );
        let advice = scoping_advice(&scope.census).expect("one subtree dominates this tree");
        assert!(
            advice.contains("knowledge/generated/api/ (20 of 30)"),
            "{advice}"
        );
        assert!(
            !advice.contains("bulk"),
            "bytes on disk must not enter the ranking: {advice}"
        );
        assert!(
            advice.contains(".cfetchignore"),
            "the advice must name the lever: {advice}"
        );
    }

    #[test]
    fn every_watchable_dir_is_a_real_directory() {
        let brain = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(brain.path().join("knowledge")).unwrap();
        std::fs::write(brain.path().join("knowledge/a.md"), "x\n").unwrap();
        for d in watchable_dirs(brain.path(), &crate::config::RingRules::default()).dirs {
            assert!(d.is_dir(), "{d:?} is not a directory");
        }
    }

    #[test]
    fn managed_cards_are_watched_even_when_the_outer_repo_ignores_them() {
        let brain = tempfile::tempdir().unwrap();
        let root = brain.path();
        let cards = root.join("knowledge/cards");
        std::fs::create_dir_all(cards.join(".git")).unwrap();
        std::fs::create_dir_all(cards.join("cloud/certificates/example")).unwrap();
        std::fs::write(root.join(".gitignore"), "knowledge/cards/\n").unwrap();
        std::fs::write(cards.join("catalog.json"), "{}").unwrap();
        std::fs::write(
            cards.join("cloud/certificates/example/question.md"),
            "# Question?\n",
        )
        .unwrap();

        let scope = watchable_dirs(root, &crate::config::RingRules::default());
        assert!(scope.dirs.contains(&cards));
        assert!(
            scope
                .dirs
                .contains(&cards.join("cloud/certificates/example"))
        );
        assert_eq!(scope.census.total(), 1);
    }

    #[test]
    fn freshness_requires_watch_registration() {
        // A state whose backstop settled but whose watches are not yet
        // registered must NOT be treated as settled: a write in an
        // unregistered directory would be invisible to the barrier.
        let dir = tempfile::tempdir().unwrap();
        let state = ServeState::new(
            "o".to_string(),
            dir.path().to_path_buf(),
            dir.path().join("barrier"),
        );
        state.mark_applied(0, 1);
        assert!(!state.is_settled(), "settled must require watches_ready");
        state.mark_watches_ready();
        assert!(state.is_settled());
    }
}

/// Parses `<barrier_dir>/b<seq>`; None for anything else (a real tree path).
fn sentinel_seq_of(barrier_dir: &Path, path: &Path) -> Option<u64> {
    let rel = path.strip_prefix(barrier_dir).ok()?;
    rel.to_str()?.strip_prefix('b')?.parse().ok()
}

fn on_event(state: &ServeState, wake: &mpsc::Sender<Wake>, event: &notify::Event) {
    // Access events fire for every file the indexer itself reads; counting
    // them would make each rebuild trigger the next.
    if matches!(event.kind, notify::EventKind::Access(_)) {
        return;
    }
    let mut sentinel_max = 0u64;
    let mut real = false;
    for path in &event.paths {
        match sentinel_seq_of(&state.barrier_dir, path) {
            Some(seq) => sentinel_max = sentinel_max.max(seq),
            None => real = true,
        }
    }
    if real {
        state.note_event();
        let _ = wake.send(Wake::Event);
    }
    if sentinel_max > 0 {
        state.note_sentinel(sentinel_max);
    }
}

/// Starts watcher + rebuild worker. The returned handle must live as long as
/// serving does.
pub fn start(cfg: &Config) -> anyhow::Result<ServeHandle> {
    let state_dir = paths::state_dir();
    let barrier_dir = state_dir.join("barrier");
    std::fs::create_dir_all(&barrier_dir)?;
    // Canonical from here on: the watch path and the sentinel path must agree
    // with what the platform's event stream reports (see ServeState::new).
    let barrier_dir = std::fs::canonicalize(&barrier_dir).unwrap_or(barrier_dir);
    // Leftover sentinels from a previous run would satisfy this run's low
    // sequence numbers; clear them.
    if let Ok(rd) = std::fs::read_dir(&barrier_dir) {
        for e in rd.flatten() {
            let _ = std::fs::remove_file(e.path());
        }
    }
    let (wake_tx, wake_rx) = mpsc::channel::<Wake>();
    let state = Arc::new(
        ServeState::new(origin_of(cfg), state_dir, barrier_dir.clone())
            // The unordered barrier needs both: the tree inputs to fingerprint
            // and a way to ask the worker for a pass. The ordered barrier
            // touches neither.
            .with_wake(wake_tx.clone()),
    );
    let watcher = notify::recommended_watcher({
        let state = state.clone();
        let wake_tx = wake_tx.clone();
        move |res: notify::Result<notify::Event>| {
            if let Ok(ev) = res {
                on_event(&state, &wake_tx, &ev);
            }
        }
    })?;
    let watcher = Arc::new(Mutex::new(watcher));
    // The barrier directory is tiny and always watchable: register it inline
    // so a barrier taken during startup still resolves.
    lock(&watcher).watch(&barrier_dir, notify::RecursiveMode::NonRecursive)?;

    // Tree watches are registered in the BACKGROUND: enumerating a large
    // brain takes seconds to minutes, and the caller (daemon::run) must reach
    // its listener bind immediately — a restart may never be a serving
    // outage. Until this completes the worker withholds `settled`, so answers
    // are labeled stale rather than wrongly fresh.
    std::thread::spawn({
        let watcher = watcher.clone();
        let state = state.clone();
        let brain_root = cfg.brain_root.clone();
        let rules = cfg.rings();
        let wake_tx = wake_tx.clone();
        move || {
            let scope = watchable_dirs(&brain_root, &rules);
            let (ok, failed) = register_watches(&watcher, &scope.dirs);
            if failed > 0 {
                eprintln!(
                    "cfetch serve: watching {ok} director(ies); {failed} could not be watched \
                     (the 60s fingerprint backstop still covers them)"
                );
                if let Some(advice) = scoping_advice(&scope.census) {
                    eprintln!("{advice}");
                }
            }
            state.mark_watches_ready();
            // A directory may have appeared while we were registering.
            let _ = wake_tx.send(Wake::Event);
        }
    });

    std::thread::spawn({
        let state = state.clone();
        let cfg = cfg.clone();
        let watcher = watcher.clone();
        move || worker(&state, &cfg, &wake_rx, &watcher)
    });
    Ok(ServeHandle {
        state,
        _watcher: watcher,
    })
}

/// The rebuild worker: drains event batches into index updates and runs the
/// fingerprint backstop at start and every 60s.
fn worker(
    state: &ServeState,
    cfg: &Config,
    wake: &mpsc::Receiver<Wake>,
    watcher: &Arc<Mutex<notify::RecommendedWatcher>>,
) {
    let mut conn: Option<rusqlite::Connection> = None;
    // Startup backstop: whatever happened while the daemon was down is
    // invisible to the watcher.
    apply(state, &mut conn, cfg, true);
    let mut last_backstop = Instant::now();
    let mut watched: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    loop {
        match wake.recv_timeout(FINGERPRINT_INTERVAL) {
            Ok(first) => {
                // Event bursts settle behind the debounce; a barrier request
                // does NOT wait — a query is blocked on it. Concurrent
                // requests still collapse into one pass through the drain
                // below, and every barrier that entered before that pass
                // starts is covered by it.
                let mut forced = first == Wake::Barrier;
                if !forced {
                    std::thread::sleep(DEBOUNCE);
                }
                while let Ok(w) = wake.try_recv() {
                    forced |= w == Wake::Barrier;
                }
                if forced {
                    state.refresh_queued.store(false, Ordering::Release);
                }
                apply(state, &mut conn, cfg, forced);
                if forced {
                    // That pass WAS the fingerprint sweep; do not repeat it.
                    last_backstop = Instant::now();
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
        if last_backstop.elapsed() >= FINGERPRINT_INTERVAL {
            last_backstop = Instant::now();
            apply(state, &mut conn, cfg, true);
            // Directories created since the last sweep carry no watch of
            // their own (watches are per-directory, non-recursive by design).
            // Re-enumerating on the backstop cadence keeps the watch set
            // convergent without ever following a symlink.
            if state.watches_ready() {
                let dirs = watchable_dirs(&cfg.brain_root, &cfg.rings()).dirs;
                if watched.is_empty() {
                    watched = dirs.iter().cloned().collect();
                } else {
                    let fresh: Vec<PathBuf> = dirs
                        .iter()
                        .filter(|d| !watched.contains(*d))
                        .cloned()
                        .collect();
                    if !fresh.is_empty() {
                        register_watches(watcher, &fresh);
                        watched.extend(fresh);
                    }
                }
            }
        }
    }
}

/// One update pass. `backstop` additionally runs the stat-fingerprint check
/// (correctness floor, and the coverage token the unordered barrier waits on);
/// the event path trusts the watcher's dirty marker. A barrier request always
/// forces `backstop`.
fn apply(
    state: &ServeState,
    conn: &mut Option<rusqlite::Connection>,
    cfg: &Config,
    backstop: bool,
) {
    let pass_start = Instant::now();
    if conn.is_none() {
        match index::open(&state.state_dir) {
            Ok(c) => {
                // The rebuild may queue behind the daemon's background code
                // scan; give the writer more patience than a default reader.
                let _ = c.busy_timeout(Duration::from_secs(10));
                // The read-only query connections cannot create the code
                // tables lazily; make sure they exist before serving `find`.
                if let Err(e) = crate::code::ensure_schema(&c) {
                    state.mark_error(format!("code schema: {e}"));
                    return;
                }
                *conn = Some(c);
            }
            Err(e) => {
                state.mark_error(format!("open index: {e}"));
                return;
            }
        }
    }
    let Some(c) = conn.as_mut() else { return };
    let rules = cfg.rings();
    let target = state.pending_now();
    let dirty = target > state.applied_now();
    // Every stat this pass takes happens after `pass_start`, so reporting it
    // as the walk's start is a LOWER bound — the safe direction: a barrier
    // accepts coverage only when the reported start is at or after its own
    // entry, and the real walk is then later still.
    let mut walked: Option<String> = None;
    let need_scan = if backstop {
        let result = index::staleness(c, &cfg.brain_root, None, &rules);
        // Record background I/O cost. Query threads never perform this walk,
        // regardless of its previous measured latency.
        state.note_walk_cost(pass_start.elapsed());
        match result {
            Ok((stale, fingerprint)) => {
                walked = Some(fingerprint);
                dirty || stale
            }
            Err(_) => true,
        }
    } else {
        dirty
    };
    if need_scan {
        // Incremental first: re-scan only the changed files. `None` (diff too
        // large, basis changed) and errors alike fall back to the full scan —
        // the correctness floor either way.
        let result = match index::rescan_changed(c, &cfg.brain_root, None, &rules) {
            Ok(Some(r)) => Ok(r),
            Ok(None) | Err(_) => index::scan(c, &cfg.brain_root, None, &rules),
        };
        match result {
            // Both scan paths stat the whole tree and commit the fingerprint
            // of what they saw, so this pass DID walk — report it.
            Ok(r) => state.mark_pass(
                target,
                r.generation,
                index::stored_fingerprint(c),
                Some(pass_start),
            ),
            Err(e) => state.mark_error(e.to_string()),
        }
    } else if let Some(fingerprint) = walked {
        // The stat walk just proved the committed catalog already describes
        // the tree — coverage without a rebuild, and the value an unordered
        // barrier tests against.
        state.mark_pass(
            target,
            index::generation(c),
            Some(fingerprint),
            Some(pass_start),
        );
    } else if !state.is_settled() {
        // Event path, nothing dirty: no walk happened, so nothing may be
        // claimed about tree coverage — only that startup is past.
        state.mark_applied(target, index::generation(c));
    }
}

// ---- token handling ----

/// Constant-time-ish comparison: never early-returns on the first differing
/// byte, so response timing does not leak the token prefix.
pub fn token_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= usize::from(x ^ y);
    }
    diff == 0
}

// ---- client side (none-tier host) ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_eq_matches_only_exact() {
        assert!(token_eq("abc123", "abc123"));
        assert!(!token_eq("abc123", "abc124"));
        assert!(!token_eq("abc123", "abc12"));
        assert!(!token_eq("", "x"));
        assert!(token_eq("", ""));
    }

    #[test]
    fn the_serving_origin_is_never_empty() {
        // "unknown-host" was the macOS symptom of reading /proc; an empty
        // origin would be a worse one — every coherence label carries it.
        assert!(!crate::paths::hostname().is_empty());
        assert!(!origin_of(&Config::default()).is_empty());
    }

    #[test]
    fn barrier_times_out_stale_when_nothing_applies() {
        let dir = tempfile::tempdir().unwrap();
        // Pinned to the sentinel path: this test simulates a watcher, so it
        // must not depend on what the HOST's watcher backend can promise.
        let state = ServeState::new(
            "test".into(),
            dir.path().to_path_buf(),
            dir.path().join("barrier"),
        )
        .with_mode(BarrierMode::Ordered);
        std::fs::create_dir_all(dir.path().join("barrier")).unwrap();
        // No watcher, no worker: the sentinel is never observed.
        let out = state.barrier(Duration::from_millis(80));
        assert!(
            !out.fresh,
            "an unserviced barrier must label the answer stale"
        );
        assert!(out.note.is_some());
        assert!(out.waited_ms >= 80);
    }

    #[test]
    fn barrier_waits_for_sentinel_then_applied() {
        let dir = tempfile::tempdir().unwrap();
        let barrier_dir = dir.path().join("barrier");
        std::fs::create_dir_all(&barrier_dir).unwrap();
        let state = Arc::new(
            ServeState::new("test".into(), dir.path().to_path_buf(), barrier_dir)
                .with_mode(BarrierMode::Ordered),
        );
        // A fully started server: watches registered (the registration
        // thread's job), pending work exists and is not yet applied.
        state.mark_watches_ready();
        state.note_event();
        // A fake watcher+worker: observes the sentinel, then applies.
        let sim = std::thread::spawn({
            let state = state.clone();
            move || {
                std::thread::sleep(Duration::from_millis(30));
                state.note_sentinel(1);
                std::thread::sleep(Duration::from_millis(30));
                state.mark_applied(1, 7);
            }
        });
        let out = state.barrier(Duration::from_secs(2));
        sim.join().unwrap();
        assert!(out.fresh, "note: {:?}", out.note);
        assert!(
            out.waited_ms >= 60,
            "must have waited for sentinel AND apply"
        );
        assert_eq!(state.generation.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn barrier_is_stale_until_startup_settles() {
        let dir = tempfile::tempdir().unwrap();
        let barrier_dir = dir.path().join("barrier");
        std::fs::create_dir_all(&barrier_dir).unwrap();
        let state = Arc::new(
            ServeState::new("test".into(), dir.path().to_path_buf(), barrier_dir)
                .with_mode(BarrierMode::Ordered),
        );
        // Sentinel observed immediately, zero pending — but the startup
        // backstop has not settled yet: events may have been missed while
        // the daemon was down, so freshness must not be claimed.
        let sim = std::thread::spawn({
            let state = state.clone();
            move || {
                std::thread::sleep(Duration::from_millis(10));
                state.note_sentinel(1);
            }
        });
        let out = state.barrier(Duration::from_millis(100));
        sim.join().unwrap();
        assert!(!out.fresh, "unsettled startup must not serve fresh");
        // Applying is not enough while tree watches are still registering:
        // a write in an unwatched directory would be invisible.
        state.mark_applied(0, 1);
        std::thread::spawn({
            let state = state.clone();
            move || {
                std::thread::sleep(Duration::from_millis(10));
                state.note_sentinel(2);
            }
        })
        .join()
        .unwrap();
        let out = state.barrier(Duration::from_millis(100));
        assert!(!out.fresh, "unregistered watches must not serve fresh");
        state.mark_watches_ready();
        std::thread::spawn({
            let state = state.clone();
            move || {
                std::thread::sleep(Duration::from_millis(10));
                state.note_sentinel(3);
            }
        })
        .join()
        .unwrap();
        let out = state.barrier(Duration::from_secs(1));
        assert!(out.fresh);
    }

    #[test]
    fn failed_update_makes_barrier_stale_with_the_error_named() {
        let dir = tempfile::tempdir().unwrap();
        let barrier_dir = dir.path().join("barrier");
        std::fs::create_dir_all(&barrier_dir).unwrap();
        let state = Arc::new(
            ServeState::new("test".into(), dir.path().to_path_buf(), barrier_dir)
                .with_mode(BarrierMode::Ordered),
        );
        state.mark_applied(0, 1); // settled
        state.note_event();
        state.mark_error("disk on fire".to_string());
        let sim = std::thread::spawn({
            let state = state.clone();
            move || {
                std::thread::sleep(Duration::from_millis(10));
                state.note_sentinel(1);
            }
        });
        let out = state.barrier(Duration::from_millis(120));
        sim.join().unwrap();
        assert!(!out.fresh);
        assert!(out.note.as_deref().unwrap().contains("disk on fire"));
    }

    #[test]
    fn sentinel_paths_are_distinguished_from_tree_paths() {
        let bd = PathBuf::from("/state/barrier");
        assert_eq!(
            sentinel_seq_of(&bd, Path::new("/state/barrier/b17")),
            Some(17)
        );
        assert_eq!(sentinel_seq_of(&bd, Path::new("/state/barrier/bx")), None);
        assert_eq!(
            sentinel_seq_of(&bd, Path::new("/brain/knowledge/b17")),
            None
        );
        assert_eq!(sentinel_seq_of(&bd, Path::new("/state/barrier")), None);
    }

    // ---- barrier mode: the platform capability, and the two paths ----

    /// A state wired the way `start` wires the real one, minus watcher and
    /// worker — the tests below play those parts themselves.
    #[test]
    fn strict_requires_a_walk_started_after_request_not_an_old_matching_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let (tx, rx) = mpsc::channel();
        let state = Arc::new(
            ServeState::new(
                "test".into(),
                dir.path().to_path_buf(),
                dir.path().join("barrier"),
            )
            .with_wake(tx),
        );
        state.mark_watches_ready();
        let old_start = Instant::now();
        state.mark_pass(0, 1, Some("same tree".into()), Some(old_start));
        let worker = std::thread::spawn({
            let state = state.clone();
            move || {
                assert_eq!(rx.recv().unwrap(), Wake::Barrier);
                state.mark_pass(0, 2, Some("same tree".into()), Some(old_start));
                std::thread::sleep(Duration::from_millis(40));
                state.mark_pass(0, 3, Some("a moving tree".into()), Some(Instant::now()));
            }
        });
        let outcome = state.barrier(Duration::from_secs(1));
        worker.join().unwrap();
        assert!(outcome.fresh, "{:?}", outcome.note);
        assert!(
            outcome.waited_ms >= 30,
            "an earlier walk cannot prove freshness"
        );
    }

    #[test]
    fn blocked_background_worker_does_not_block_cached_answers_and_refreshes_coalesce() {
        let dir = tempfile::tempdir().unwrap();
        let (tx, rx) = mpsc::channel();
        let state = ServeState::new(
            "test".into(),
            dir.path().to_path_buf(),
            dir.path().join("barrier"),
        )
        .with_wake(tx);
        let start = Instant::now();
        for _ in 0..1000 {
            let outcome = state.cached();
            assert!(!outcome.fresh);
            assert!(outcome.note.unwrap().contains("cached snapshot"));
        }
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(
            rx.try_iter().count(),
            1,
            "one queued refresh, no per-query workers"
        );
        let start = Instant::now();
        let outcome = state.barrier(Duration::from_millis(30));
        assert!(!outcome.fresh);
        assert!(outcome.note.unwrap().contains("freshness unavailable"));
        assert!(start.elapsed() < Duration::from_millis(500));
        assert_eq!(
            rx.try_iter().count(),
            0,
            "blocked worker still owns the pending pass"
        );
    }

    #[test]
    fn the_ordered_barrier_still_proves_coverage_by_sentinel_and_pays_nothing_else() {
        // The Linux fast path is measured at ~30-50 ms end to end INCLUDING
        // process spawn; it may not acquire a tree walk. Two proofs here: it
        // still waits for the sentinel (a state whose catalog covers the tree
        // perfectly stays stale until the sentinel is observed), and it never
        // reads the tree; only the sentinel is involved in this fixture.
        let brain = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(brain.path().join("knowledge")).unwrap();
        std::fs::write(brain.path().join("knowledge/a.md"), "one\n").unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("barrier")).unwrap();
        let state = Arc::new(
            ServeState::new(
                "test".into(),
                dir.path().to_path_buf(),
                dir.path().join("barrier"),
            )
            .with_mode(BarrierMode::Ordered),
        );
        state.mark_watches_ready();
        state.mark_pass(
            0,
            1,
            Some(index::tree_fingerprint(
                brain.path(),
                None,
                &crate::config::RingRules::default(),
            )),
            Some(Instant::now()),
        );
        // Coverage by content is fully established — the ordered path does not
        // look at it, so no sentinel means no freshness.
        let out = state.barrier(Duration::from_millis(120));
        assert!(
            !out.fresh,
            "the ordered path must still wait for its sentinel"
        );
        assert!(out.note.unwrap().contains("sentinel"));

        let sim = std::thread::spawn({
            let state = state.clone();
            move || {
                std::thread::sleep(Duration::from_millis(10));
                state.note_sentinel(2);
            }
        });
        let out = state.barrier(Duration::from_secs(1));
        sim.join().unwrap();
        assert!(
            out.fresh,
            "sentinel observed + nothing pending = fresh: {:?}",
            out.note
        );
    }

    /// The measurement the unordered path's cost decision rests on: what does
    /// ONE entry fingerprint cost on a real brain tree, as opposed to a
    /// tempdir with four files?
    ///
    /// Ignored by default — it needs a tree, and CI has none. Point
    /// `CFETCH_FINGERPRINT_BENCH` at one and run with `--ignored --nocapture`.
    #[test]
    #[ignore = "measurement against a real tree; set CFETCH_FINGERPRINT_BENCH"]
    fn fingerprint_walk_cost_on_a_real_tree() {
        let Ok(root) = std::env::var("CFETCH_FINGERPRINT_BENCH") else {
            return;
        };
        let root = PathBuf::from(root);
        let rules = crate::config::RingRules::default();
        let dirs = watchable_dirs(&root, &rules).dirs.len();
        let mut times = Vec::new();
        let mut previous: Option<String> = None;
        for _ in 0..10 {
            let t = Instant::now();
            let fingerprint = index::tree_fingerprint(&root, None, &rules);
            times.push(t.elapsed());
            // A fingerprint that is not stable on an idle tree would make the
            // unordered barrier force a rebuild pass on every single query.
            if let Some(prev) = &previous {
                assert_eq!(
                    prev, &fingerprint,
                    "the fingerprint must be stable on an idle tree"
                );
            }
            previous = Some(fingerprint);
        }
        times.sort();
        println!(
            "entry fingerprint over {dirs} indexable director(ies): p50 {:?}, max {:?}",
            times[times.len() / 2],
            times[times.len() - 1]
        );
    }
}
