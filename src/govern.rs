//! The governance loop's producers: what gets queued, and with which text.
//!
//! Economics (see DESIGN, "Reminder queue drained at UserPromptSubmit"): a
//! Stop-level `additionalContext` forces a whole extra model turn, so nothing
//! here ever emits at Stop. Producers only QUEUE into the session state; the
//! UserPromptSubmit hook drains the queue piggybacked on the next user prompt
//! at zero extra turns. Cadence re-injection is grounded in the
//! instruction-decay study cited in DESIGN (~5.6% compliance decay per
//! generated function): every N-th post-tool event re-surfaces the top ring-0
//! rules in ~100 tokens.

use std::path::{Component, Path, PathBuf};

use crate::config::Config;
use crate::resident::SessionScope;
use crate::session_state::SessionState;

/// ~100 tokens at the chars/3.5 heuristic — hard cap on one rule refresh.
const RULES_MAX_CHARS: usize = 350;
/// At most this many ring-0 descriptions per refresh.
const RULES_MAX_LINES: usize = 3;
/// Brain-tree writes at which the stale-STATUS nudge arms.
const STATUS_NUDGE_MIN_WRITES: usize = 3;

/// Stale-STATUS nudge: the session wrote `STATUS_NUDGE_MIN_WRITES`+ files
/// under the brain tree, none of them a `todo/active/*/STATUS.md` — queue a
/// reminder to update the task STATUS. Returns whether anything was queued.
pub fn queue_status_nudge(st: &mut SessionState, brain_root: &Path) -> bool {
    let mut brain_writes = 0usize;
    let mut out_of_tree = 0usize;
    let mut touched_status = false;
    for p in &st.written {
        let path = Path::new(p);
        match path.strip_prefix(brain_root) {
            Ok(rel) => {
                brain_writes += 1;
                touched_status |= is_active_status(rel);
            }
            // Counted, not named: a write outside the tree is still evidence
            // the session did work, and dropping it was what made a session
            // that edits elsewhere look idle.
            Err(_) => out_of_tree += 1,
        }
    }
    if touched_status {
        return false;
    }
    let shell = usize::try_from(st.shell_writes).unwrap_or(usize::MAX);
    let unnamed = out_of_tree.saturating_add(shell);
    // Unnamed activity is a FALLBACK, never a booster. A session with named
    // brain writes is judged on those alone — a write to /elsewhere must not
    // push a two-file session over a three-file threshold. But a session that
    // edits only through the shell names nothing at all, and arming on named
    // writes alone left that case permanently silent.
    let armed = if brain_writes > 0 {
        brain_writes >= STATUS_NUDGE_MIN_WRITES
    } else {
        unnamed >= STATUS_NUDGE_MIN_WRITES
    };
    if !armed {
        return false;
    }
    let seen = if brain_writes > 0 {
        format!("{brain_writes} brain file(s) written this session")
    } else {
        // Nothing nameable was seen, so the count is all this may claim.
        format!("{unnamed} write(s) this session, none of them nameable")
    };
    st.queue_reminder(
        "status",
        &format!(
            "[cfetch: {seen}, but no todo/active/*/STATUS.md among them — update the task STATUS]"
        ),
    )
}

/// Capture-visibility: staged candidates are invisible by design (rings 5-6
/// are never injected), so their COUNT is surfaced instead — with the command
/// that shows them. The count covers the whole shared staging directory, so a
/// candidate another host flagged is announced here too. Returns whether
/// anything was queued.
pub fn queue_staging_visibility(st: &mut SessionState, staging_dir: &Path) -> bool {
    let n = crate::staging::pending_count(staging_dir);
    if n < 1 {
        return false;
    }
    st.queue_reminder(
        "staging",
        &format!("[cfetch: {n} staged candidate(s) await distillation — cfetch staging list]"),
    )
}

/// The top ring-0 rules as one "[cfetch rule refresh]" block: description
/// frontmatter lines of the resident config's ring-0 files THIS session is
/// entitled to (or, with no resident config at all, of the brain's
/// `knowledge/behaviours` files declaring `ring: 0`), first `RULES_MAX_LINES`,
/// capped at `RULES_MAX_CHARS`.
pub fn top_ring0_rules(cfg: &Config, scope: &SessionScope) -> Option<String> {
    let descriptions = if cfg.resident.is_empty() {
        memories_ring0_descriptions(&cfg.brain_root, &behavior_dirs(cfg))
    } else {
        resident_ring0_descriptions(cfg, scope)
    };
    if descriptions.is_empty() {
        return None;
    }
    let mut text = String::from("[cfetch rule refresh]");
    for d in descriptions {
        text.push_str("\n- ");
        text.push_str(&d);
    }
    if text.len() > RULES_MAX_CHARS {
        let mut cut = RULES_MAX_CHARS;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        text.truncate(cut);
    }
    Some(text)
}

/// Descriptions of the resident config's ring-0 files, in config order,
/// filtered by the session's injection scope. Unreadable files and files
/// without a description contribute nothing.
fn resident_ring0_descriptions(cfg: &Config, scope: &SessionScope) -> Vec<String> {
    let mut out = Vec::new();
    for entry in cfg
        .resident
        .iter()
        .filter(|e| e.ring == 0 && e.scope.matches(&scope.host, scope.repo.as_deref()))
    {
        if out.len() == RULES_MAX_LINES {
            break;
        }
        let Ok(raw) = std::fs::read_to_string(cfg.resolve(&entry.path)) else { continue };
        if let Some(d) = frontmatter_value(&raw, "description") {
            out.push(d);
        }
    }
    out
}

/// The directories the configured taxonomy puts behavioral memory in: every
/// ring rule naming a SUBTREE (a prefix ending in `/`) on rings 0-2. With the
/// shipped rules that is the distilled-memories directory; with a custom
/// taxonomy it is whatever that taxonomy calls the same thing — the fallback
/// follows the config instead of assuming one tree's layout.
fn behavior_dirs(cfg: &Config) -> Vec<PathBuf> {
    cfg.ring_rules
        .iter()
        .filter(|r| r.ring <= 2 && r.prefix.ends_with('/'))
        .map(|r| PathBuf::from(r.prefix.trim_end_matches('/')))
        .collect()
}

/// Descriptions of files declaring `ring: 0` under `dirs`, in path order
/// (read_dir order is arbitrary; the "first 3" must be deterministic).
fn memories_ring0_descriptions(brain_root: &Path, dirs: &[PathBuf]) -> Vec<String> {
    let mut files: Vec<PathBuf> = Vec::new();
    for dir in dirs {
        let Ok(rd) = std::fs::read_dir(brain_root.join(dir)) else { continue };
        files.extend(
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("md")),
        );
    }
    files.sort();
    let mut out = Vec::new();
    for f in files {
        if out.len() == RULES_MAX_LINES {
            break;
        }
        let Ok(raw) = std::fs::read_to_string(&f) else { continue };
        let ring0 = frontmatter_value(&raw, "ring")
            .is_some_and(|v| v.split_whitespace().next() == Some("0"));
        if !ring0 {
            continue;
        }
        if let Some(d) = frontmatter_value(&raw, "description") {
            out.push(d);
        }
    }
    out
}

/// `rel` (brain-root-relative) is exactly `todo/active/<task>/STATUS.md`.
fn is_active_status(rel: &Path) -> bool {
    let parts: Vec<&str> = rel
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => s.to_str(),
            _ => None,
        })
        .collect();
    matches!(parts.as_slice(), ["todo", "active", _, "STATUS.md"])
}

/// Single-line value of `key:` from a leading `---` frontmatter block. Key
/// match is ASCII-case-insensitive; the first non-empty value wins. An
/// unterminated block yields nothing (mirrors the index's fail-closed
/// frontmatter handling).
fn frontmatter_value(text: &str, key: &str) -> Option<String> {
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("---") {
        return None;
    }
    let mut value = None;
    for line in lines {
        let t = line.trim();
        if t == "---" {
            return value;
        }
        if value.is_none()
            && let Some((k, v)) = t.split_once(':')
            && k.trim().eq_ignore_ascii_case(key)
        {
            let v = v.trim();
            if !v.is_empty() {
                value = Some(v.to_string());
            }
        }
    }
    None // unterminated frontmatter: fail closed
}

#[cfg(test)]
mod tests {
    /// End-to-end discovery, in the shape a real brain has: no resident config,
    /// rules found by walking the ring-2 behaviour directory.
    #[test]
    fn prohibitions_are_discovered_from_the_behaviour_directory() {
        let dir = tempfile::tempdir().unwrap();
        let mem = dir.path().join("knowledge/behaviours");
        std::fs::create_dir_all(&mem).unwrap();
        std::fs::write(
            mem.join("feedback_never_enable_dedup.md"),
            "---\nname: feedback_never_enable_dedup\ndescription: \"Never set dedup=\"\nring: 0\n\
forbids:\n  - \"dedup=on|sha256|blake3|edonr\"\nmetadata: \n  type: feedback\n---\n\nbody\n",
        )
        .unwrap();
        // A rule that declares nothing must contribute nothing.
        std::fs::write(mem.join("feedback_other.md"), "---\nring: 2\n---\n\nprose\n").unwrap();

        let cfg = Config { brain_root: dir.path().to_path_buf(), ..Config::default() };
        let found = prohibitions(&cfg, &SessionScope::from_cwd(None));
        assert_eq!(found.len(), 1, "exactly the declaring rule: {found:?}");
        assert_eq!(found[0].rule, "feedback_never_enable_dedup");
        assert!(first_violation("zfs set dedup=on tank", &found).is_some());
        assert!(first_violation("ls -la", &found).is_none());
    }

    /// The rule this mechanism exists for, in the declared form.
    #[test]
    fn a_declared_prohibition_expands_its_alternation_with_the_shared_prefix() {
        let raw = "---\nname: feedback_never_enable_dedup\nring: 0\n\
forbids:\n  - \"dedup=on|sha256|blake3|edonr\"\nmetadata:\n  type: feedback\n---\n\nbody\n";
        let p = prohibitions_in(raw, "feedback_never_enable_dedup").expect("a prohibition");
        assert!(p.patterns.contains(&"dedup=on".to_string()), "{:?}", p.patterns);
        assert!(p.patterns.contains(&"dedup=sha256".to_string()), "{:?}", p.patterns);
        // The bare algorithm name must never become a pattern: it appears in
        // ordinary code constantly and would fire on almost every write.
        assert!(!p.patterns.contains(&"sha256".to_string()), "bare sha256 leaked in");
        // The list ends at the next key.
        assert!(!p.patterns.iter().any(|x| x.contains("feedback")), "{:?}", p.patterns);
    }

    #[test]
    fn a_violation_names_the_rule_that_forbade_it() {
        let rules = vec![Prohibition {
            rule: "feedback_never_enable_dedup".into(),
            patterns: vec!["dedup=on".into(), "dedup=sha256".into()],
        }];
        let (hit, pat) = first_violation("zfs set dedup=on solid/agents", &rules).expect("caught");
        assert_eq!(hit.rule, "feedback_never_enable_dedup");
        assert_eq!(pat, "dedup=on");
        assert!(first_violation("ZFS SET DEDUP=ON tank", &rules).is_some(), "case is no excuse");
        // Ordinary work is left alone — this is the property that decides
        // whether the mechanism survives contact with a real session.
        assert!(first_violation("zfs set compression=zstd tank", &rules).is_none());
        assert!(first_violation("let h = sha256(bytes);", &rules).is_none());
        assert!(first_violation("ls -la && zfs list", &rules).is_none());
    }

    /// A rule that declares nothing contributes nothing. Inferring
    /// prohibitions from prose was tried and rejected: see prohibitions_in.
    #[test]
    fn a_rule_without_a_forbids_key_is_skipped() {
        let raw = "---\nring: 0\n---\n\nNever enable ZFS dedup (`dedup=on`) on any pool.\n";
        assert!(prohibitions_in(raw, "feedback_never_enable_dedup").is_none());
    }

    use super::*;

    /// A session matching no particular host or repo — enough for every
    /// unscoped entry, which is what these fixtures use.
    fn any_session() -> SessionScope {
        SessionScope { host: "any-host".into(), repo: None }
    }
    use crate::config::{Config, GovernanceConfig, ResidentEntry, Scope};
    use crate::exhaust;
    use serde_json::json;

    fn write_event(session: &str, path: &str) -> crate::hook_io::HookEvent {
        crate::hook_io::HookEvent {
            session_id: Some(session.into()),
            tool_name: Some("Write".into()),
            tool_input: Some(json!({"file_path": path})),
            ..Default::default()
        }
    }

    /// A staging directory holding exactly `n` candidates, produced through
    /// the real capture + trap path (hot-file: the same ring-3 brain file
    /// written in two distinct sessions).
    fn staged_tree(dir: &Path, n: usize) -> PathBuf {
        let brain = Path::new("/b/agents");
        let ex = exhaust::Exhaust::new(
            dir.join("logs/cfetch"),
            dir.join("staging/cfetch"),
            "h1".into(),
            1 << 20,
        );
        for i in 0..n {
            let path = format!("/b/agents/knowledge/hot{i}.md");
            for s in ["s1", "s2"] {
                ex.capture_post_tool(
                    &write_event(s, &path),
                    brain,
                    &crate::config::RingRules::default(),
                )
                .unwrap();
            }
        }
        ex.record_stop("s2").unwrap();
        ex.staging_dir
    }

    /// A session that edits only through the shell names nothing, so the
    /// nudge used to stay silent no matter how much work it did.
    #[test]
    fn shell_only_sessions_still_arm_the_nudge() {
        let brain = Path::new("/b/agents");
        let mut st = SessionState::default();
        for _ in 0..3 {
            st.record_shell_write();
        }
        assert!(queue_status_nudge(&mut st, brain));
        let texts = st.drain_reminders();
        assert!(texts[0].contains("none of them nameable"), "{}", texts[0]);
        // The STATUS pattern is a hint, not a path — what must never appear is
        // anything the session actually touched.
        assert!(!texts[0].contains("/b/agents"), "a real path leaked: {}", texts[0]);
        assert!(!texts[0].contains("/tmp"), "a real path leaked: {}", texts[0]);
    }

    /// Unnamed activity is a fallback, not a booster: it must never top up a
    /// named brain count that is short of the threshold.
    #[test]
    fn unnamed_writes_do_not_inflate_a_named_brain_count() {
        let brain = Path::new("/b/agents");
        let mut st = SessionState::default();
        st.record_write("/b/agents/knowledge/a.md");
        st.record_write("/b/agents/knowledge/b.md");
        for _ in 0..9 {
            st.record_shell_write();
        }
        st.record_write("/elsewhere/c.md");
        assert!(
            !queue_status_nudge(&mut st, brain),
            "two brain writes stay below the threshold however much unnamed activity there was"
        );
    }

    #[test]
    fn status_nudge_requires_three_brain_writes() {
        let brain = Path::new("/b/agents");
        let mut st = SessionState::default();
        st.record_write("/b/agents/knowledge/a.md");
        st.record_write("/b/agents/knowledge/b.md");
        st.record_write("/elsewhere/c.md");
        assert!(!queue_status_nudge(&mut st, brain), "two brain writes are below the threshold");
        assert!(st.drain_reminders().is_empty());
        st.record_write("/b/agents/knowledge/behaviours/c.md");
        assert!(queue_status_nudge(&mut st, brain));
        let texts = st.drain_reminders();
        assert_eq!(texts.len(), 1);
        assert!(texts[0].contains("STATUS"), "the text must point at the task STATUS: {}", texts[0]);
    }

    #[test]
    fn status_nudge_suppressed_when_status_was_written() {
        let brain = Path::new("/b/agents");
        let mut st = SessionState::default();
        st.record_write("/b/agents/knowledge/a.md");
        st.record_write("/b/agents/knowledge/b.md");
        st.record_write("/b/agents/todo/active/cfetch/STATUS.md");
        assert!(!queue_status_nudge(&mut st, brain));
        assert!(st.drain_reminders().is_empty());
    }

    #[test]
    fn status_nudge_ignores_lookalike_status_paths() {
        // A STATUS.md outside todo/active/<task>/ does not count as updating
        // the task STATUS.
        let brain = Path::new("/b/agents");
        let mut st = SessionState::default();
        st.record_write("/b/agents/knowledge/a.md");
        st.record_write("/b/agents/todo/done/old/STATUS.md");
        st.record_write("/b/agents/todo/active/STATUS.md"); // no task dir
        assert!(queue_status_nudge(&mut st, brain));
    }

    #[test]
    fn active_status_path_shape() {
        assert!(is_active_status(Path::new("todo/active/cfetch/STATUS.md")));
        assert!(!is_active_status(Path::new("todo/active/STATUS.md")));
        assert!(!is_active_status(Path::new("todo/done/cfetch/STATUS.md")));
        assert!(!is_active_status(Path::new("todo/active/cfetch/notes/STATUS.md")));
        assert!(!is_active_status(Path::new("todo/active/cfetch/status.md")));
    }

    #[test]
    fn staging_visibility_counts_pending_candidates() {
        let dir = tempfile::tempdir().unwrap();
        let staging = staged_tree(dir.path(), 2);
        let mut st = SessionState::default();
        assert!(queue_staging_visibility(&mut st, &staging));
        assert_eq!(
            st.drain_reminders(),
            vec!["[cfetch: 2 staged candidate(s) await distillation — cfetch staging list]"
                .to_string()]
        );
    }

    #[test]
    fn staging_visibility_silent_when_nothing_is_staged() {
        // No staging directory at all: nothing queued, and the probe must not
        // create one.
        let dir = tempfile::tempdir().unwrap();
        let staging = dir.path().join("staging/cfetch");
        let mut st = SessionState::default();
        assert!(!queue_staging_visibility(&mut st, &staging));
        assert!(!staging.exists(), "a read-only probe must not create tree state");
        // A queue whose candidates were all consumed: also silent.
        let staging = staged_tree(dir.path(), 1);
        let id = crate::staging::list(&staging)[0].id.clone();
        crate::staging::consume(&staging, &id).unwrap();
        assert!(!queue_staging_visibility(&mut st, &staging));
        assert!(st.drain_reminders().is_empty());
    }

    #[test]
    fn frontmatter_value_parses_and_fails_closed() {
        let doc = "---\nname: feedback_x\ndescription: never do the thing\nring: 0\n---\nbody\n";
        assert_eq!(frontmatter_value(doc, "description"), Some("never do the thing".into()));
        assert_eq!(frontmatter_value(doc, "ring"), Some("0".into()));
        assert_eq!(frontmatter_value(doc, "absent"), None);
        assert_eq!(frontmatter_value("no frontmatter here", "description"), None);
        assert_eq!(
            frontmatter_value("---\ndescription: unterminated\nbody", "description"),
            None,
            "an unterminated block must yield nothing"
        );
        assert_eq!(frontmatter_value("---\ndescription:\n---\n", "description"), None);
        assert_eq!(
            frontmatter_value("---\nDescription: mixed case\n---\n", "description"),
            Some("mixed case".into())
        );
    }

    #[test]
    fn ring0_rules_come_from_resident_ring0_files_only() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("guards.md"),
            "---\ndescription: destruction is human-gated\n---\nbody\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("policy.md"),
            "---\ndescription: policy line must not appear\n---\nbody\n",
        )
        .unwrap();
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            resident: vec![
                ResidentEntry { path: PathBuf::from("guards.md"), ring: 0, scope: Scope::default(), weight: None },
                ResidentEntry { path: PathBuf::from("policy.md"), ring: 1, scope: Scope::default(), weight: None },
            ],
            ..Config::default()
        };
        let rules = top_ring0_rules(&cfg, &any_session()).unwrap();
        assert!(rules.starts_with("[cfetch rule refresh]"));
        assert!(rules.contains("destruction is human-gated"));
        assert!(!rules.contains("policy line"), "ring-1 files are not rule-refresh material");
    }

    #[test]
    fn ring0_rules_fall_back_to_memories_when_resident_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let memories = dir.path().join("knowledge/behaviours");
        std::fs::create_dir_all(&memories).unwrap();
        std::fs::write(
            memories.join("feedback_a.md"),
            "---\nring: 0\ndescription: rule alpha\n---\n",
        )
        .unwrap();
        std::fs::write(
            memories.join("feedback_b.md"),
            "---\nring: 2\ndescription: behavior beta\n---\n",
        )
        .unwrap();
        std::fs::write(memories.join("feedback_c.md"), "---\nring: 0\n---\nno description\n")
            .unwrap();
        std::fs::write(
            memories.join("feedback_d.md"),
            "---\nring: 0\ndescription: rule delta\n---\n",
        )
        .unwrap();
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            resident: Vec::new(),
            ..Config::default()
        };
        let rules = top_ring0_rules(&cfg, &any_session()).unwrap();
        assert!(rules.contains("rule alpha"));
        assert!(rules.contains("rule delta"));
        assert!(!rules.contains("behavior beta"), "ring-2 memories must not surface");
    }

    #[test]
    fn ring0_rules_fallback_follows_custom_ring_rules() {
        // A tree that calls its behavioral memory something else: the
        // fallback reads the config's subtree, not a hardcoded one.
        let dir = tempfile::tempdir().unwrap();
        let handbook = dir.path().join("handbook");
        std::fs::create_dir_all(&handbook).unwrap();
        std::fs::write(
            handbook.join("a.md"),
            "---\nring: 0\ndescription: house rule alpha\n---\n",
        )
        .unwrap();
        let memories = dir.path().join("knowledge/behaviours");
        std::fs::create_dir_all(&memories).unwrap();
        std::fs::write(
            memories.join("b.md"),
            "---\nring: 0\ndescription: not this tree\n---\n",
        )
        .unwrap();
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            resident: Vec::new(),
            ring_rules: vec![crate::config::RingRule { prefix: "handbook/".into(), ring: 1 }],
            ..Config::default()
        };
        let rules = top_ring0_rules(&cfg, &any_session()).unwrap();
        assert!(rules.contains("house rule alpha"));
        assert!(!rules.contains("not this tree"), "the shipped layout has no privilege here");
    }

    #[test]
    fn ring0_rules_cap_lines_and_chars() {
        let dir = tempfile::tempdir().unwrap();
        let memories = dir.path().join("knowledge/behaviours");
        std::fs::create_dir_all(&memories).unwrap();
        for i in 0..5 {
            std::fs::write(
                memories.join(format!("feedback_{i}.md")),
                format!("---\nring: 0\ndescription: rule number {i} {}\n---\n", "x".repeat(200)),
            )
            .unwrap();
        }
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            resident: Vec::new(),
            ..Config::default()
        };
        let rules = top_ring0_rules(&cfg, &any_session()).unwrap();
        assert!(rules.len() <= 350, "rule refresh was {} chars for a ~100-token cap", rules.len());
        assert!(!rules.contains("rule number 3"), "only the first 3 descriptions are taken");
        assert!(!rules.contains("rule number 4"));
    }

    #[test]
    fn no_ring0_material_yields_none() {
        let dir = tempfile::tempdir().unwrap();
        // Resident configured but nothing at ring 0.
        std::fs::write(dir.path().join("p.md"), "---\ndescription: d\n---\n").unwrap();
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            resident: vec![ResidentEntry { path: PathBuf::from("p.md"), ring: 1, scope: Scope::default(), weight: None }],
            ..Config::default()
        };
        assert_eq!(top_ring0_rules(&cfg, &any_session()), None);
        // No resident config and no memories dir at all.
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            resident: Vec::new(),
            ..Config::default()
        };
        assert_eq!(top_ring0_rules(&cfg, &any_session()), None);
    }

    #[test]
    fn governance_config_is_carried_by_default() {
        // Compile-time guard that the block exists on Config; behavior gating
        // lives in the hooks layer tests.
        let g = GovernanceConfig::default();
        assert!(g.enabled);
        assert_eq!(g.reinject_every, 25);
    }
}

/// A prohibition lifted out of a rule, with the rule that stated it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prohibition {
    /// Rule file stem, so a violation can be traced back to its source.
    pub rule: String,
    /// Literal fragments, any one of which matching means the rule is in play.
    pub patterns: Vec<String>,
}

/// Shortest fragment worth matching. Below this a pattern is a common word and
/// every write trips it; a hook that cries wolf gets switched off, which costs
/// more than the warnings were ever worth.
const MIN_PATTERN: usize = 4;

/// Patterns a rule DECLARES it forbids, from a `forbids:` frontmatter list.
///
/// Declared, never inferred. Extracting prohibitions from prose was tried
/// against the operator's real 203-rule corpus and failed on a specific shape:
/// "don't guess — run `zfs list`" is a negated sentence whose quoted token is
/// the REMEDY. The heuristic could not separate those from prohibitions, and
/// the surviving false positives were `ls -la`, `zfs list`, `stat -c` — the
/// commands an agent runs most. A hook that objects to `ls -la` is a hook
/// switched off within a day, and then the real guards go unenforced too.
///
/// So a rule opts in:
///
/// ```text
/// forbids:
///   - "dedup=on|sha256|blake3|edonr"
///   - "systemd-cryptenroll --tpm2-device"
/// ```
///
/// `|` is alternation sharing the prefix before `=`, because that is how the
/// rules already write it: `dedup=on|sha256` means two prohibitions, not a
/// prohibition on the bare word `sha256`.
pub fn prohibitions_in(raw: &str, rule: &str) -> Option<Prohibition> {
    let mut patterns = Vec::new();
    let mut in_list = false;
    for line in frontmatter(raw).lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("forbids:") {
            in_list = true;
            // Inline form: `forbids: "a|b"`.
            for span in unquote(rest.trim()) {
                push_expanded(&mut patterns, &span);
            }
            continue;
        }
        if in_list {
            if let Some(item) = trimmed.strip_prefix("- ") {
                for span in unquote(item.trim()) {
                    push_expanded(&mut patterns, &span);
                }
                continue;
            }
            // Any other key ends the list.
            if !trimmed.is_empty() && !trimmed.starts_with('-') {
                in_list = false;
            }
        }
    }
    (!patterns.is_empty()).then(|| Prohibition { rule: rule.to_string(), patterns })
}

/// The frontmatter block, or the empty string when a file has none.
fn frontmatter(raw: &str) -> &str {
    let rest = raw.strip_prefix("---").unwrap_or("");
    rest.split_once("\n---").map_or("", |(fm, _)| fm)
}

/// One YAML scalar, minus surrounding quotes. Empty input yields nothing.
fn unquote(s: &str) -> Vec<String> {
    let s = s.trim().trim_matches('"').trim_matches('\'').trim();
    if s.is_empty() { Vec::new() } else { vec![s.to_string()] }
}

fn push_expanded(out: &mut Vec<String>, span: &str) {
    for alt in expand_alternation(span) {
        if alt.len() >= MIN_PATTERN && !out.contains(&alt) {
            out.push(alt);
        }
    }
}

/// Expands `dedup=on|sha256|blake3|edonr` into patterns that each keep the
/// `dedup=` prefix.
///
/// Splitting on `|` alone would yield `sha256`, `blake3` and `edonr` as
/// standalone patterns — tokens that appear in ordinary code constantly, so the
/// rule would fire on almost every write. The prefix is what makes the match
/// specific, and the rules write it once and share it across alternatives.
fn expand_alternation(span: &str) -> Vec<String> {
    let Some((head, tail)) = span.split_once('|') else {
        return vec![span.trim().to_string()];
    };
    let prefix = head.rfind('=').map(|i| &head[..=i]).unwrap_or("");
    let mut out = vec![head.trim().to_string()];
    for alt in tail.split('|') {
        let alt = alt.trim();
        if alt.is_empty() {
            continue;
        }
        out.push(if alt.contains('=') || prefix.is_empty() {
            alt.to_string()
        } else {
            format!("{prefix}{alt}")
        });
    }
    out
}

/// Every prohibition declared by the rules this session is entitled to.
/// Ring 0-2: invariants, policy and distilled behaviour — the rings that carry
/// standing constraints. Ring 3+ is knowledge, which describes rather than
/// forbids.
pub fn prohibitions(cfg: &Config, scope: &SessionScope) -> Vec<Prohibition> {
    let mut out: Vec<Prohibition> = Vec::new();
    let mut take = |raw: &str, stem: &str| {
        if let Some(p) = prohibitions_in(raw, stem)
            && !out.iter().any(|existing| existing.rule == p.rule)
        {
            out.push(p);
        }
    };

    // Resident entries AND the behaviour directories. Both, never one or the
    // other: a brain has a resident set naming a handful of entry points and a
    // directory of distilled rules, and the constraints live in the second.
    // Returning early on a non-empty resident list meant the default config —
    // which names AGENT.md and nothing else — hid every declared prohibition.
    for entry in cfg
        .resident
        .iter()
        .filter(|e| e.ring <= 2 && e.scope.matches(&scope.host, scope.repo.as_deref()))
    {
        let path = cfg.resolve(&entry.path);
        let Ok(raw) = std::fs::read_to_string(&path) else { continue };
        let stem = path.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into());
        take(&raw, &stem);
    }

    for dir in behavior_dirs(cfg) {
        let Ok(rd) = std::fs::read_dir(cfg.brain_root.join(&dir)) else { continue };
        let mut files: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("md"))
            .collect();
        files.sort();
        for f in files {
            let Ok(raw) = std::fs::read_to_string(&f) else { continue };
            let ring = frontmatter_value(&raw, "ring")
                .and_then(|v| v.split_whitespace().next()?.parse::<u8>().ok())
                .unwrap_or(2);
            if ring > 2 {
                continue;
            }
            let stem = f.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into());
            take(&raw, &stem);
        }
    }
    out
}

/// The first rule this content violates, if any. Case-insensitive: a rule about
/// `dedup=on` means the same shouted.
pub fn first_violation<'a>(
    content: &str,
    rules: &'a [Prohibition],
) -> Option<(&'a Prohibition, &'a str)> {
    let hay = content.to_ascii_lowercase();
    rules.iter().find_map(|r| {
        r.patterns
            .iter()
            .find(|p| hay.contains(&p.to_ascii_lowercase()))
            .map(|p| (r, p.as_str()))
    })
}
