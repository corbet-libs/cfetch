//! The resident set: ring-0/1 content injected at session start, per POLICY —
//! each entry declares the sessions it belongs to and only those receive it.
//! Private blocks are removed fail-closed (an unclosed <private> swallows to
//! end of input — degrade to MORE private, never less).
//!
//! What arrives is a DIGEST, not a dump. Disclosure is progressive: an entry
//! the budget can carry whole arrives whole, an entry it cannot is reduced to
//! one index line — what the file is, roughly what reading it costs, where it
//! is — with its hard rules still carried verbatim, and `scope.always` is the
//! escape hatch for content that must arrive in full whatever it costs. The
//! digest closes with a PRICE for the rest of the catalog, per ring, so a
//! session knows what it could still ask for instead of being blind to it.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::hook_io::HookEvent;

const OPEN: &str = "<private>";
const CLOSE: &str = "</private>";

/// Byte ranges (tags inclusive) of private regions, DEPTH-AWARE: a nested
/// `<private>` does not let the first `</private>` end the region — trailing
/// private content must never leak. Unbalanced opens fail closed to end of
/// input; a stray close outside any region is plain text.
fn private_regions(s: &str) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut depth = 0usize;
    let mut region_start = 0usize;
    let mut pos = 0usize;
    while pos < s.len() {
        let next_open = s[pos..].find(OPEN).map(|i| pos + i);
        let next_close = s[pos..].find(CLOSE).map(|i| pos + i);
        let open_first = match (next_open, next_close) {
            (None, None) => break,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (Some(o), Some(c)) => o < c,
        };
        if open_first {
            let o = next_open.unwrap();
            if depth == 0 {
                region_start = o;
            }
            depth += 1;
            pos = o + OPEN.len();
        } else {
            let c = next_close.unwrap();
            if depth > 0 {
                depth -= 1;
                if depth == 0 {
                    regions.push((region_start, c + CLOSE.len()));
                }
            }
            pos = c + CLOSE.len();
        }
    }
    if depth > 0 {
        regions.push((region_start, s.len())); // fail closed
    }
    regions
}

/// What blanking did to a text, so the scan can say it instead of letting
/// content vanish in silence. `unbalanced` is the sharp case: an unclosed
/// `<private>` — often a mere mention of the tag in prose — blanks
/// everything after it, fail-closed, and the file's tail leaves the index
/// without a word unless the caller reports this.
pub struct PrivateOutcome {
    /// True when a `<private>` was never closed: the region extends to end
    /// of input by design ("degrade to MORE private, never less").
    pub unbalanced: bool,
}

/// Blanks `<private>` regions (spaces, newlines preserved — line numbers in
/// citations stay accurate) and reports what the blanking did.
pub fn blank_private_checked(s: &str) -> (String, PrivateOutcome) {
    let regions = private_regions(s);
    let unbalanced = regions
        .last()
        .is_some_and(|&(start, end)| end == s.len() && !s[start..end].ends_with(CLOSE));
    let mut out = String::with_capacity(s.len());
    let mut cursor = 0usize;
    for (start, end) in &regions {
        out.push_str(&s[cursor..*start]);
        for c in s[*start..*end].chars() {
            out.push(if c == '\n' { '\n' } else { ' ' });
        }
        cursor = *end;
    }
    out.push_str(&s[cursor..]);
    (out, PrivateOutcome { unbalanced })
}

/// Removes `<private>...</private>` regions (nesting-aware, fail-closed).
pub fn strip_private(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut cursor = 0usize;
    for (start, end) in private_regions(s) {
        out.push_str(&s[cursor..start]);
        cursor = end;
    }
    out.push_str(&s[cursor..]);
    out
}

/// The session an injection decision is made for. Two coordinates, because
/// those are the two an agent session actually has at SessionStart: which
/// machine it runs on, and what it is working on.
#[derive(Debug, Clone)]
pub struct SessionScope {
    pub host: String,
    /// The working directory's own name — the repo, as the agent sees it.
    pub repo: Option<String>,
}

impl SessionScope {
    /// From a hook event. The repo name is the LAST component of the event's
    /// `cwd`: no filesystem walk, because the hook path must not stat its way
    /// up a tree that may be an NFS mount.
    pub fn from_event(event: &HookEvent) -> SessionScope {
        SessionScope::from_cwd(event.cwd.as_deref())
    }

    pub fn from_cwd(cwd: Option<&str>) -> SessionScope {
        SessionScope { host: crate::paths::hostname(), repo: repo_name(cwd) }
    }

    /// For non-hook callers (selfcheck): this process's own working directory.
    pub fn current() -> SessionScope {
        let cwd = std::env::current_dir().ok();
        SessionScope::from_cwd(cwd.as_deref().and_then(Path::to_str))
    }
}

fn repo_name(cwd: Option<&str>) -> Option<String> {
    let trimmed = cwd?.trim_end_matches('/');
    let name = Path::new(trimmed).file_name()?.to_string_lossy().to_string();
    (!name.is_empty()).then_some(name)
}

pub struct ResidentDigest {
    pub text: String,
    /// (source label, chars contributed) — selfcheck reporting and, from
    /// Milestone 5, per-source injection booking. An entry the digest only
    /// indexed is booked for its index line: the point of the line is that it
    /// costs a fortieth of the file, and an unbooked saving is a claim.
    pub sources: Vec<(String, usize)>,
    /// Labels of the entries this session's scope excluded. Reported rather
    /// than dropped: a resident file that stops arriving must be explainable
    /// without reading the config.
    pub skipped_by_scope: Vec<String>,
    /// Labels of the entries whose FILE is missing. The digest still carries
    /// the one placeholder line (a session should learn its contract broke)
    /// — but the placeholder must not pass as health: selfcheck reports
    /// these, because a configured resident file that injects nothing but
    /// `[resident file missing: …]` is a broken setup wearing a green line.
    pub missing: Vec<String>,
}

/// Prohibitions carried verbatim out of the files the digest only indexed.
/// Three, because a fourth starts rebuilding the wall the index replaced.
const MAX_HARD_RULES: usize = 3;

/// Longest prohibition kept verbatim. A rule that does not fit is DROPPED
/// rather than cut: "never delete X unless Y" truncated after X states a
/// broader rule than the file does, and a rule invented by clipping is worse
/// than a rule left in its file.
const RULE_MAX_CHARS: usize = 140;

/// Longest "what it is" on an index line.
const SUMMARY_MAX_CHARS: usize = 80;

/// Names the second disclosure level where the model reads it, so a one-line
/// entry cannot be mistaken for the whole of a file.
const INDEX_HEADER: &str = "== index: not injected in full ==";

/// Last-resort header for a crowded index. Resident labels already carry the
/// configured brain-relative path, so repeating a long platform-specific
/// absolute prefix on every line is pure overhead.
const COMPACT_INDEX_HEADER: &str =
    "== index: not injected in full; paths are relative to the configured brain root ==";

/// One resident entry that reached this session, resolved and ready to be
/// disclosed at one of the two levels below.
struct Section {
    label: String,
    /// Absolute path — an index line has to say where the rest actually is.
    path: PathBuf,
    /// How the catalog names this file (brain-root-relative, `/`-separated),
    /// or `None` for a file outside the tree. The key the availability index
    /// subtracts by, so nothing is advertised and injected at once.
    doc: Option<String>,
    body: String,
    ring: u8,
    weight: f32,
    /// `scope.always`: this entry arrives in full even under budget pressure.
    pinned: bool,
}

/// Splits `usable` chars across the entries by WEIGHT, then hands back what
/// the small ones do not want. Each entry is `(chars it wants, its weight)`.
///
/// An equal split wastes budget in the common case: one short invariants file
/// and one long status file get the same share, the short one uses a fraction
/// of its allowance, and the long one is clipped anyway with the remainder
/// unspent. So this water-fills — every round distributes the remaining budget
/// among the entries that are still short, in proportion to their weight, and
/// an entry that needs less than its share takes only what it needs and
/// releases the rest into the next round.
///
/// It terminates: each round either satisfies at least one entry (strictly
/// shrinking the unsatisfied set) or satisfies none, which breaks immediately.
fn allocate(entries: &[(usize, f32)], usable: usize) -> Vec<usize> {
    let mut granted = vec![0usize; entries.len()];
    let mut settled = vec![false; entries.len()];
    let mut pool = usable;
    loop {
        let open: Vec<usize> = (0..entries.len()).filter(|i| !settled[*i]).collect();
        if open.is_empty() || pool == 0 {
            break;
        }
        let total: f64 = open.iter().map(|i| f64::from(entries[*i].1)).sum();
        if total <= 0.0 {
            break;
        }
        let mut progressed = false;
        let pool_at_round_start = pool;
        for &i in &open {
            let share = ((pool_at_round_start as f64) * f64::from(entries[i].1) / total) as usize;
            let want = entries[i].0.saturating_sub(granted[i]);
            let take = share.min(want).min(pool);
            granted[i] += take;
            pool -= take;
            if take == want {
                settled[i] = true;
                progressed = true;
            }
        }
        if !progressed {
            // Nobody was satisfied this round, so every remaining entry is
            // clipped. Hand the rest out once, by weight, and stop.
            for &i in &open {
                let share =
                    ((pool_at_round_start as f64) * f64::from(entries[i].1) / total) as usize;
                granted[i] += share.min(pool);
                pool = pool.saturating_sub(share);
            }
            break;
        }
    }
    granted
}

/// The entries this session is entitled to, read and cleaned, plus the labels
/// of the ones its scope left out.
fn collect(cfg: &Config, scope: &SessionScope) -> (Vec<Section>, Vec<String>, Vec<String>) {
    let mut sections = Vec::new();
    let mut skipped_by_scope = Vec::new();
    let mut missing = Vec::new();
    for entry in &cfg.resident {
        let label = format!("ring-{} {}", entry.ring, entry.path.display());
        if !entry.scope.matches(&scope.host, scope.repo.as_deref()) {
            skipped_by_scope.push(label);
            continue;
        }
        let path = cfg.resolve(&entry.path);
        let body = match std::fs::read_to_string(&path) {
            Ok(raw) => {
                let clean = strip_private(&raw).trim().to_string();
                if clean.is_empty() {
                    continue;
                }
                clean
            }
            // A missing resident file is worth one short line, not silence:
            // the resident set is the contract the operator configured.
            // The line is for the SESSION; `missing` is for the operator's
            // diagnostics — the placeholder must never pass as health.
            Err(_) => {
                missing.push(label.clone());
                format!("[resident file missing: {}]", path.display())
            }
        };
        sections.push(Section {
            label,
            doc: doc_path(&cfg.brain_root, &path),
            path,
            body,
            ring: entry.ring,
            weight: entry.budget_weight(),
            pinned: entry.scope.always,
        });
    }
    (sections, skipped_by_scope, missing)
}

/// How the catalog would name this file: brain-root-relative, `/`-separated.
/// A path outside the tree has no catalog name and cannot be double-counted.
fn doc_path(brain_root: &Path, abs: &Path) -> Option<String> {
    let rel = abs.strip_prefix(brain_root).ok()?;
    let joined: Vec<String> =
        rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    (!joined.is_empty()).then(|| joined.join("/"))
}

/// Builds the injected digest for ONE session. Entries whose scope does not
/// match the session are left out entirely — they cost no budget and are not
/// booked.
///
/// Disclosure is PROGRESSIVE, and the budget decides the level. An entry whose
/// body fits the share the water-fill gives it arrives in full, as before. One
/// that does not is INDEXED instead: a single line naming what it is, roughly
/// what reading it costs, and where it is. That trade is the whole mechanism —
/// half a rules file costs what its bytes cost and teaches only the rules it
/// happens to open with, while forty tokens of index let the model decide for
/// itself whether the rest is worth fetching. `scope.always` is the escape
/// hatch for content that must arrive whatever it costs; a pinned entry is
/// clipped rather than indexed, exactly as every entry was before.
///
/// The two things an index cannot delegate travel with it: the hard rules of
/// the indexed files, verbatim, and a price for everything the catalog holds
/// and this digest did not carry.
pub fn build(cfg: &Config, scope: &SessionScope) -> ResidentDigest {
    build_in(cfg, scope, &crate::paths::state_dir())
}

fn build_in(cfg: &Config, scope: &SessionScope, state_dir: &Path) -> ResidentDigest {
    let (sections, skipped_by_scope, missing) = collect(cfg, scope);
    if sections.is_empty() {
        return ResidentDigest { text: String::new(), sources: Vec::new(), skipped_by_scope, missing };
    }

    // The budget is a HARD cap on the whole digest: headers, index lines, clip
    // markers and the availability line are charged against it, not added on
    // top.
    let budget = cfg.budget_chars.max(200);
    let tail = tail_line(&recallable_tail(cfg, state_dir, &sections));

    // Render progressively: summaries yield first, then repeated absolute
    // brain-root prefixes yield to a single relative-path notice. The entry
    // name and read price survive every pass.
    let (mut text, mut sources) = render(&sections, budget, &tail, true, false);
    if text.len() > budget {
        let (terse, terse_sources) = render(&sections, budget, &tail, false, false);
        if terse.len() < text.len() {
            text = terse;
            sources = terse_sources;
        }
    }
    if text.len() > budget {
        let (compact, compact_sources) = render(&sections, budget, &tail, false, true);
        if compact.len() < text.len() {
            text = compact;
            sources = compact_sources;
        }
    }
    ResidentDigest { text: text.trim_end().to_string(), sources, skipped_by_scope, missing }
}

/// Decides the disclosure level of every entry and renders the digest.
///
/// The level is settled by demotion, not by a threshold: allocate, and if any
/// unpinned entry would arrive half-written, index the least load-bearing of
/// them and allocate again — its whole share is then free for the rest, which
/// is how one file being too big buys another file the room to arrive whole.
/// The loop ends after at most one pass per entry.
fn render(
    sections: &[Section],
    budget: usize,
    tail: &str,
    summaries: bool,
    compact_paths: bool,
) -> (String, Vec<(String, usize)>) {
    let mut indexed = vec![false; sections.len()];
    let (full, granted, lines, rules) = loop {
        let full: Vec<usize> = (0..sections.len()).filter(|i| !indexed[*i]).collect();
        let lines: Vec<(usize, String)> = (0..sections.len())
            .filter(|i| indexed[*i])
            .map(|i| (i, index_line(&sections[i], summaries, compact_paths)))
            .collect();
        let rules = hard_rules_block(sections, &indexed);
        let fixed: usize = lines.iter().map(|(_, l)| l.len() + 1).sum::<usize>()
            + if lines.is_empty() {
                0
            } else if compact_paths {
                COMPACT_INDEX_HEADER.len() + 2
            } else {
                INDEX_HEADER.len() + 2
            }
            + rules.len()
            + if tail.is_empty() { 0 } else { tail.len() + 1 };
        let headers: usize = full.iter().map(|i| sections[*i].label.len() + 8).sum();
        // The floor keeps every full entry represented even when the budget is
        // far too small for any of them: silence about a configured resident
        // file is the failure this whole path exists to avoid.
        let usable =
            budget.saturating_sub(fixed.saturating_add(headers)).max(full.len().saturating_mul(60));
        let wants: Vec<(usize, f32)> =
            full.iter().map(|i| (sections[*i].body.len(), sections[*i].weight)).collect();
        let granted = allocate(&wants, usable);
        let victim = full
            .iter()
            .zip(&granted)
            .filter(|(i, g)| !sections[**i].pinned && **g < sections[**i].body.len())
            // Least load-bearing first, and among equals the biggest — that is
            // the demotion that frees the most room for the fewest lost words.
            .min_by(|a, b| {
                sections[*a.0]
                    .weight
                    .total_cmp(&sections[*b.0].weight)
                    .then(sections[*b.0].body.len().cmp(&sections[*a.0].body.len()))
            })
            .map(|(i, _)| *i);
        match victim {
            Some(i) => indexed[i] = true,
            None => break (full, granted, lines, rules),
        }
    };

    let mut text = String::new();
    let mut sources = Vec::new();
    for (i, share) in full.into_iter().zip(granted) {
        let s = &sections[i];
        let clipped = clip(&s.body, share, &s.label);
        let _ = write!(text, "== {} ==\n{clipped}\n\n", s.label);
        sources.push((s.label.clone(), clipped.len()));
    }
    if !lines.is_empty() {
        let header = if compact_paths { COMPACT_INDEX_HEADER } else { INDEX_HEADER };
        let _ = writeln!(text, "{header}");
        for (i, line) in &lines {
            let _ = writeln!(text, "{line}");
            sources.push((sections[*i].label.clone(), line.len()));
        }
        text.push('\n');
    }
    if !rules.is_empty() {
        text.push_str(&rules);
        sources.push(("hard rules (verbatim)".to_string(), rules.len()));
    }
    if !tail.is_empty() {
        let _ = writeln!(text, "{tail}");
        sources.push(("availability index".to_string(), tail.len()));
    }
    (text, sources)
}

/// Clips one body to its share, marking the cut and naming where the rest is.
fn clip(body: &str, share: usize, label: &str) -> String {
    if body.len() <= share {
        return body.to_string();
    }
    let marker_reserve = 60 + label.len();
    let mut cut = share.saturating_sub(marker_reserve).max(40).min(body.len());
    while cut < body.len() && !body.is_char_boundary(cut) {
        cut += 1;
    }
    if cut < body.len() {
        format!("{}\n[clipped at {cut} of {} chars — full content: {label}]", &body[..cut], body.len())
    } else {
        body.to_string()
    }
}

/// The one line that stands in for a whole file: what it is, what reading it
/// costs, and where it is.
fn index_line(s: &Section, summary: bool, compact_path: bool) -> String {
    let mut line = format!("- {}", s.label);
    if summary && let Some(what) = summarize(&s.body) {
        let _ = write!(line, " — {what}");
    }
    if compact_path && s.doc.is_some() {
        let _ = write!(line, " — {}", fmt_tokens(s.body.len()));
    } else {
        let _ = write!(line, " — {} — read {}", fmt_tokens(s.body.len()), s.path.display());
    }
    line
}

/// `~840 tok` / `~3.4k tok` — a magnitude, and honest about being one. The
/// model is deciding whether to spend, not reconciling an invoice.
fn fmt_tokens(chars: usize) -> String {
    let tokens = crate::hook_io::estimate_tokens(chars);
    if tokens >= 1000 {
        format!("~{:.1}k tok", tokens as f64 / 1000.0)
    } else {
        format!("~{tokens} tok")
    }
}

/// What the file says it is: its frontmatter description, else its title, else
/// its first line of prose.
fn summarize(body: &str) -> Option<String> {
    let (frontmatter, rest) = split_frontmatter(body);
    let raw = frontmatter
        .and_then(|fm| frontmatter_value(fm, "description"))
        .or_else(|| rest.lines().find_map(first_prose_line))?;
    let text: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let text = text.trim_matches(|c: char| c == '#' || c == '*' || c == '_' || c.is_whitespace());
    if text.is_empty() {
        return None;
    }
    Some(truncate(text, SUMMARY_MAX_CHARS))
}

/// The first line worth showing as a title: headings keep their text, list
/// scaffolding and frontmatter fences are not prose.
fn first_prose_line(line: &str) -> Option<String> {
    let t = line.trim();
    if t.is_empty() || t == "---" {
        return None;
    }
    Some(undecorate(t).to_string())
}

/// `(frontmatter body, everything after it)`. An unterminated block yields no
/// frontmatter at all — the same fail-closed reading the index applies, so a
/// mangled fence cannot make the whole file read as metadata.
fn split_frontmatter(body: &str) -> (Option<&str>, &str) {
    let Some(rest) = body.strip_prefix("---\n") else { return (None, body) };
    match rest.find("\n---") {
        Some(end) => {
            let after = &rest[end + 4..];
            (Some(&rest[..end]), after.strip_prefix('\n').unwrap_or(after))
        }
        None => (None, body),
    }
}

/// Single-line value of `key:` from a frontmatter block; the first non-empty
/// value wins, key match is ASCII-case-insensitive.
fn frontmatter_value(frontmatter: &str, key: &str) -> Option<String> {
    frontmatter.lines().find_map(|line| {
        let (k, v) = line.trim().split_once(':')?;
        (k.trim().eq_ignore_ascii_case(key) && !v.trim().is_empty())
            .then(|| v.trim().to_string())
    })
}

/// The hard rules of the files that were only indexed, verbatim, lowest ring
/// first — the one thing a pointer cannot stand in for. "Read AGENT.md for the
/// rules" is advice; "never force-stop a VM" is the rule, and a session that
/// only got the advice has not been told.
fn hard_rules_block(sections: &[Section], indexed: &[bool]) -> String {
    let mut candidates: Vec<(u8, usize, String)> = Vec::new();
    for (i, s) in sections.iter().enumerate() {
        if !indexed[i] {
            // A file that arrived whole already carries its own rules.
            continue;
        }
        for rule in hard_rules(&s.body) {
            candidates.push((s.ring, i, rule));
        }
    }
    candidates.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut out = String::new();
    let mut kept: Vec<String> = Vec::new();
    for (_, _, rule) in candidates {
        if kept.len() == MAX_HARD_RULES {
            break;
        }
        // Two files stating the same rule must not spend two of three slots.
        if kept.iter().any(|k| k.eq_ignore_ascii_case(&rule)) {
            continue;
        }
        if out.is_empty() {
            out.push_str("their hard rules, verbatim:\n");
        }
        let _ = writeln!(out, "- {rule}");
        kept.push(rule);
    }
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

/// Lines of `body` that state a prohibition, in file order, undecorated.
/// Frontmatter is skipped: a description is a label, not a rule.
fn hard_rules(body: &str) -> Vec<String> {
    let (_, rest) = split_frontmatter(body);
    rest.lines()
        .map(undecorate)
        // CHARS, not bytes: the cap is a readability limit, and measuring
        // bytes made any non-ASCII rule count heavier than it reads — a
        // 120-character German rule is 208 bytes and silently dropped
        // while its English twin passed.
        .filter(|l| l.chars().count() <= RULE_MAX_CHARS && is_prohibition(l))
        .map(str::to_string)
        .collect()
}

/// Strips list bullets, headings, quote markers and emphasis from around a
/// line, so `1. **NEVER** …` and `- never …` read the same to everything
/// downstream. Only decoration goes: the words are never touched, because a
/// line that survives this is quoted verbatim as a rule.
fn undecorate(line: &str) -> &str {
    const MARKS: [char; 10] = ['-', '*', '+', '>', '#', ' ', '\t', '`', '"', '_'];
    let s = line.trim().trim_end_matches(['*', '_', '`', ' ', '\t']).trim_start_matches(MARKS);
    let s = match s.split_once(' ') {
        Some((first, rest))
            if first.len() <= 3
                && first.starts_with(|c: char| c.is_ascii_digit())
                && first.trim_end_matches(['.', ')']).chars().all(|c| c.is_ascii_digit()) =>
        {
            rest
        }
        _ => s,
    };
    s.trim_start_matches(MARKS)
}

/// Whether a line opens with a negative imperative. Deliberately a small,
/// closed set of openers rather than a reading of the sentence: a false
/// positive spends one of three lines on a real sentence from a rules file,
/// while a clever matcher that misfires on "no longer" would spend it on
/// nothing.
fn is_prohibition(text: &str) -> bool {
    let mut words = text.split_whitespace();
    let first = words
        .next()
        .unwrap_or_default()
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
        .to_ascii_lowercase();
    match first.as_str() {
        "never" | "no" | "don't" | "dont" => true,
        "do" => words.next().is_some_and(|w| w.eq_ignore_ascii_case("not")),
        _ => false,
    }
}

/// Prices what recall can still reach and this digest did not carry: per ring,
/// how many files and roughly what reading them costs.
///
/// The numbers come from the derived catalog rather than from a tree walk — a
/// walk at session start would stat the whole brain on what may be an NFS
/// mount, and the catalog is what recall itself answers from, so the price is
/// exactly as true as the answers it advertises. No catalog (a none-tier host,
/// or one that has never scanned) means NO line: "0 files" would state an
/// absence nobody measured.
fn recallable_tail(cfg: &Config, state_dir: &Path, injected: &[Section]) -> Vec<(u8, usize, u64)> {
    if cfg.client.serving.is_some() {
        // Opening a local catalog here would build the second, silently stale
        // truth a none-tier host exists to avoid.
        return Vec::new();
    }
    let Ok(conn) = crate::index::open_ro(state_dir) else { return Vec::new() };
    // A session start must never wait out someone else's write transaction;
    // an unpriced digest is a small loss, a stalled hook is a large one.
    let _ = conn.busy_timeout(std::time::Duration::from_millis(150));
    let Ok(mut stmt) =
        conn.prepare("SELECT ring, COUNT(*), COALESCE(SUM(size), 0) FROM docs GROUP BY ring")
    else {
        return Vec::new();
    };
    let Ok(rows) = stmt.query_map([], |r| {
        Ok((r.get::<_, i64>(0)? as u8, r.get::<_, i64>(1)? as usize, r.get::<_, i64>(2)? as u64))
    }) else {
        return Vec::new();
    };
    let mut per_ring: Vec<(u8, usize, u64)> = rows.filter_map(Result::ok).collect();
    // Whatever this digest already accounts for — in full or as an index line
    // — is not something to go and recall.
    if let Ok(mut one) = conn.prepare("SELECT ring, size FROM docs WHERE path = ?1") {
        for doc in injected.iter().filter_map(|s| s.doc.as_deref()) {
            let Ok((ring, size)) = one
                .query_row([doc], |r| Ok((r.get::<_, i64>(0)? as u8, r.get::<_, i64>(1)? as u64)))
            else {
                continue;
            };
            if let Some(e) = per_ring.iter_mut().find(|e| e.0 == ring) {
                e.1 = e.1.saturating_sub(1);
                e.2 = e.2.saturating_sub(size);
            }
        }
    }
    per_ring.retain(|e| e.1 > 0);
    per_ring.sort_by_key(|e| e.0);
    per_ring
}

/// The availability index itself: what exists, what it costs, how to ask.
fn tail_line(per_ring: &[(u8, usize, u64)]) -> String {
    if per_ring.is_empty() {
        return String::new();
    }
    let priced: Vec<String> = per_ring
        .iter()
        .map(|(ring, files, chars)| {
            format!("ring {ring} · {files} file(s) {}", fmt_tokens(*chars as usize))
        })
        .collect();
    format!("[not injected, recallable: {} — cfetch recall \"<topic>\"]", priced.join(" · "))
}

/// Cuts `text` to `max` chars on a char boundary, saying so when it cuts.
fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{}…", cut.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ResidentEntry, Scope};
    use std::path::PathBuf;

    /// A brain with four resident files and the scopes the injection policy
    /// is meant to discriminate on.
    fn scoped_cfg(dir: &std::path::Path) -> Config {
        for name in ["everywhere.md", "on-host.md", "in-repo.md", "elsewhere.md"] {
            std::fs::write(dir.join(name), format!("body of {name}\n")).unwrap();
        }
        Config {
            brain_root: dir.to_path_buf(),
            resident: vec![
                ResidentEntry {
                    path: PathBuf::from("everywhere.md"),
                    ring: 1,
                    scope: Scope::default(),
                    weight: None,
                },
                ResidentEntry {
                    path: PathBuf::from("on-host.md"),
                    ring: 1,
                    scope: Scope { hosts: vec!["build-box".into()], ..Scope::default() },
                    weight: None,
                },
                ResidentEntry {
                    path: PathBuf::from("in-repo.md"),
                    ring: 1,
                    scope: Scope { repos: vec!["widget".into()], ..Scope::default() },
                    weight: None,
                },
                ResidentEntry {
                    path: PathBuf::from("elsewhere.md"),
                    ring: 1,
                    scope: Scope {
                        hosts: vec!["other-box".into()],
                        repos: vec!["gadget".into()],
                        always: false,
                    },
                    weight: None,
                },
            ],
            ..Config::default()
        }
    }

    fn entry(path: &str, ring: u8, weight: Option<f32>) -> ResidentEntry {
        ResidentEntry { path: PathBuf::from(path), ring, scope: Scope::default(), weight }
    }

    /// Builds against a state dir that holds no catalog. The availability
    /// index has its own tests; every other test would otherwise price
    /// whatever catalog the machine running it happens to have.
    fn build(cfg: &Config, scope: &SessionScope) -> ResidentDigest {
        build_in(cfg, scope, &PathBuf::from("/nonexistent/cfetch-state"))
    }

    #[test]
    fn a_missing_resident_file_is_injected_as_placeholder_but_reported_missing() {
        // The session still learns its contract broke (one placeholder
        // line); the OPERATOR's diagnostics must not count that line as
        // health — `missing` is what selfcheck warns on.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("AGENT.md"), "real rules\n").unwrap();
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            resident: vec![
                ResidentEntry { path: "AGENT.md".into(), ring: 1, scope: Scope::default(), weight: None },
                ResidentEntry { path: "knowledge/handoff.md".into(), ring: 0, scope: Scope::default(), weight: None },
            ],
            ..Config::default()
        };
        let digest = build(&cfg, &SessionScope { host: "h".into(), repo: None });
        assert!(digest.text.contains("[resident file missing:"), "session sees the placeholder");
        assert!(digest.text.contains("real rules"), "the present entry still arrives");
        assert_eq!(digest.missing.len(), 1, "the missing entry is reported: {:?}", digest.missing);
        assert!(digest.missing[0].contains("knowledge/handoff.md"));
    }

    #[test]
    fn a_present_resident_set_reports_nothing_missing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("knowledge")).unwrap();
        std::fs::write(dir.path().join("knowledge/handoff.md"), "state\n").unwrap();
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            resident: vec![ResidentEntry {
                path: "knowledge/handoff.md".into(),
                ring: 0,
                scope: Scope::default(),
                weight: None,
            }],
            ..Config::default()
        };
        let digest = build(&cfg, &SessionScope { host: "h".into(), repo: None });
        assert!(digest.missing.is_empty());
        assert!(!digest.text.contains("resident file missing"));
    }

    /// The index line the digest prints instead of a whole file.
    fn index_line_for(d: &ResidentDigest, name: &str) -> Option<String> {
        d.text.lines().find(|l| l.starts_with("- ") && l.contains(name)).map(str::to_string)
    }

    /// Two entries, one tiny and one long. Under an equal split the tiny file
    /// reserves half the budget, uses a sliver of it, and the long file no
    /// longer fits its share — which now costs it the whole disclosure, not
    /// just its tail. Water-filling must hand that slack over instead.
    #[test]
    fn a_small_entry_releases_its_unused_budget_to_a_large_one() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("small.md"), "tiny\n").unwrap();
        std::fs::write(dir.path().join("big.md"), "B".repeat(1500)).unwrap();
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            budget_chars: 2000,
            resident: vec![entry("small.md", 1, None), entry("big.md", 1, None)],
            ..Config::default()
        };
        let d = build(&cfg, &SessionScope::from_cwd(None));
        let big = d.sources.iter().find(|(l, _)| l.contains("big.md")).unwrap().1;
        // An equal split caps this near half the usable budget, and a body
        // that does not fit its share is indexed rather than injected.
        assert_eq!(big, 1500, "big.md arrived as {big} chars; the small entry's slack was not released");
        assert!(d.text.contains(&"B".repeat(1500)), "big.md did not arrive whole");
        assert!(d.text.len() <= 2000, "the budget is still a hard cap: {}", d.text.len());
    }

    /// The ring is the default statement of how load-bearing an entry is, so
    /// when only one of two entries can arrive whole, the behavior note is the
    /// one that gives way to the invariant. Under an equal split neither would
    /// fit its share and the FIRST entry would be demoted instead.
    #[test]
    fn a_lower_ring_outbids_a_higher_one_when_both_cannot_fit() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("inv.md"), "I".repeat(1200)).unwrap();
        std::fs::write(dir.path().join("beh.md"), "B".repeat(1200)).unwrap();
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            budget_chars: 2000,
            resident: vec![
                entry("inv.md", 0, None),
                ResidentEntry {
                    scope: Scope { repos: vec!["widget".into()], ..Scope::default() },
                    ..entry("beh.md", 2, None)
                },
            ],
            ..Config::default()
        };
        let d = build(&cfg, &SessionScope { host: "any-host".into(), repo: Some("widget".into()) });
        assert!(d.text.contains(&"I".repeat(1200)), "the invariant did not arrive whole");
        assert!(!d.text.contains(&"B".repeat(1200)), "both bodies fit — the budget was not the constraint");
        assert!(index_line_for(&d, "beh.md").is_some(), "the behavior note lost its index line too");
    }

    #[test]
    fn an_explicit_weight_overrides_the_ring_and_degenerate_values_fall_back() {
        // Explicit weight wins over the ring's default.
        let e = entry("x.md", 2, Some(9.0));
        assert_eq!(e.budget_weight(), 9.0);
        // A NaN would poison the whole allocation; a zero would silently
        // delete an entry the operator asked for. Both fall back to the ring.
        assert_eq!(entry("x.md", 0, Some(f32::NAN)).budget_weight(), 4.0);
        assert_eq!(entry("x.md", 1, Some(0.0)).budget_weight(), 2.0);
        assert_eq!(entry("x.md", 1, Some(-3.0)).budget_weight(), 2.0);
        assert_eq!(entry("x.md", 2, None).budget_weight(), 1.0);
    }

    /// Every entry must still be represented even when the budget is far too
    /// small for any of them — silence about a configured resident file is the
    /// failure this whole path exists to avoid.
    #[test]
    fn a_tiny_budget_still_names_every_entry() {
        let dir = tempfile::tempdir().unwrap();
        for n in ["a.md", "b.md", "c.md"] {
            std::fs::write(dir.path().join(n), "X".repeat(4000)).unwrap();
        }
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            budget_chars: 200,
            resident: vec![entry("a.md", 0, None), entry("b.md", 1, None), entry("c.md", 2, None)],
            ..Config::default()
        };
        let d = build(&cfg, &SessionScope::from_cwd(None));
        assert_eq!(d.sources.len(), 3);
        for n in ["a.md", "b.md", "c.md"] {
            assert!(d.text.contains(n), "{n} vanished from the digest");
        }
    }

    #[test]
    fn session_scope_reads_the_repo_from_the_hook_event_cwd() {
        let event: HookEvent =
            serde_json::from_str(r#"{"session_id":"s1","cwd":"/srv/work/widget"}"#).unwrap();
        let scope = SessionScope::from_event(&event);
        assert_eq!(scope.repo.as_deref(), Some("widget"));
        assert!(!scope.host.is_empty(), "the host is always known");

        let trailing: HookEvent = serde_json::from_str(r#"{"cwd":"/srv/work/widget/"}"#).unwrap();
        assert_eq!(SessionScope::from_event(&trailing).repo.as_deref(), Some("widget"));

        let no_cwd: HookEvent = serde_json::from_str(r#"{"session_id":"s1"}"#).unwrap();
        assert!(SessionScope::from_event(&no_cwd).repo.is_none());
    }

    #[test]
    fn injection_selects_by_host_scope() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = scoped_cfg(dir.path());
        let scope = SessionScope { host: "build-box".into(), repo: Some("sprocket".into()) };
        let d = build(&cfg, &scope);
        assert!(d.text.contains("body of everywhere.md"), "an unscoped entry is always in");
        assert!(d.text.contains("body of on-host.md"), "the host matches");
        assert!(!d.text.contains("body of in-repo.md"), "wrong repo");
        assert!(!d.text.contains("body of elsewhere.md"), "neither host nor repo matches");
        assert_eq!(d.skipped_by_scope.len(), 2, "skips are reported, never silent");
    }

    #[test]
    fn injection_selects_by_repo_scope() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = scoped_cfg(dir.path());
        let scope = SessionScope { host: "laptop".into(), repo: Some("widget".into()) };
        let d = build(&cfg, &scope);
        assert!(d.text.contains("body of everywhere.md"));
        assert!(d.text.contains("body of in-repo.md"), "the repo matches");
        assert!(!d.text.contains("body of on-host.md"));
        assert!(!d.text.contains("body of elsewhere.md"));
    }

    #[test]
    fn a_session_matching_nothing_still_gets_the_unscoped_entries() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = scoped_cfg(dir.path());
        let scope = SessionScope { host: "laptop".into(), repo: None };
        let d = build(&cfg, &scope);
        assert!(d.text.contains("body of everywhere.md"));
        assert_eq!(d.sources.len(), 1, "only the unscoped entry is booked");
        assert_eq!(d.skipped_by_scope.len(), 3);
    }

    #[test]
    fn scoped_out_entries_do_not_consume_the_budget() {
        // The share each injected file gets is computed over the entries that
        // actually reach the session, never over the whole configured list —
        // so the same file that has to be indexed among three arrives whole
        // when it is the only one the session is entitled to.
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = scoped_cfg(dir.path());
        for name in ["everywhere.md", "on-host.md", "in-repo.md", "elsewhere.md"] {
            std::fs::write(dir.path().join(name), "w".repeat(1500)).unwrap();
        }
        cfg.budget_chars = 2000;
        let all = build(&cfg, &SessionScope { host: "build-box".into(), repo: Some("widget".into()) });
        let one = build(&cfg, &SessionScope { host: "laptop".into(), repo: None });
        assert_eq!(all.sources.len(), 3, "everywhere + host + repo; `elsewhere` matches neither");
        assert_eq!(one.sources.len(), 1);
        assert!(one.text.len() <= 2000, "the budget is still a hard cap");
        assert!(
            index_line_for(&all, "everywhere.md").is_some(),
            "three bodies cannot fit 2000 chars, so some had to be indexed:\n{}",
            all.text
        );
        assert!(
            one.text.contains(&"w".repeat(1500)),
            "alone, the surviving entry gets the whole budget:\n{}",
            one.text
        );
    }

    #[test]
    fn private_blocks_are_removed() {
        assert_eq!(strip_private("a<private>secret</private>b"), "ab");
        assert_eq!(strip_private("plain"), "plain");
    }

    #[test]
    fn unclosed_private_swallows_to_end() {
        assert_eq!(strip_private("keep<private>oops no close\nmore"), "keep");
    }

    #[test]
    fn multiple_private_blocks() {
        assert_eq!(strip_private("a<private>x</private>b<private>y</private>c"), "abc");
    }

    #[test]
    fn nested_private_blocks_do_not_leak_the_tail() {
        // The first </private> must NOT close the outer region.
        let s = "a<private>x<private>y</private>STILL-PRIVATE</private>b";
        assert_eq!(strip_private(s), "ab");
        let b = blank_private_checked(s).0;
        assert!(!b.contains("STILL-PRIVATE"));
        assert_eq!(b.len(), s.len());
    }

    #[test]
    fn stray_close_tag_is_plain_text() {
        assert_eq!(strip_private("a</private>b"), "a</private>b");
    }

    #[test]
    fn digest_budget_is_a_hard_cap() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["a.md", "b.md", "c.md"] {
            std::fs::write(dir.path().join(name), "word ".repeat(3000)).unwrap();
        }
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            resident: ["a.md", "b.md", "c.md"]
                .iter()
                .map(|n| ResidentEntry { path: PathBuf::from(n), ring: 1, scope: Scope::default(), weight: None })
                .collect(),
            code_roots: Vec::new(),
            budget_chars: 2000,
            ..Config::default()
        };
        let d = build(&cfg, &SessionScope { host: "any-host".into(), repo: None });
        assert!(d.text.len() <= 2000, "digest was {} chars for a 2000 budget", d.text.len());
        // Nothing arrives half-written to make the cap: the three of them are
        // indexed, and each is still named.
        assert_eq!(d.text.matches("[clipped at ").count(), 0);
        for name in ["a.md", "b.md", "c.md"] {
            assert!(index_line_for(&d, name).is_some(), "{name} lost its index line");
        }
    }

    #[test]
    fn blanking_preserves_length_and_newlines() {
        let s = "keep\n<private>zqx\nwvy</private>\ntail";
        let b = blank_private_checked(s).0;
        assert_eq!(b.len(), s.len());
        assert_eq!(b.matches('\n').count(), s.matches('\n').count());
        assert!(b.starts_with("keep\n"));
        assert!(b.ends_with("\ntail"));
        assert!(!b.contains("zqx") && !b.contains("wvy"));
    }

    #[test]
    fn blanking_unclosed_blanks_to_end_keeping_newlines() {
        let s = "keep\n<private>x\ny";
        let b = blank_private_checked(s).0;
        assert_eq!(b.len(), s.len());
        assert!(b.starts_with("keep\n"));
        assert!(!b.contains('x'));
        assert!(!b.contains('y'));
        assert_eq!(b.matches('\n').count(), 2);
    }

    /// `always` is the escape hatch: this content arrives whatever it costs,
    /// so it is clipped at the budget rather than reduced to a pointer.
    #[test]
    fn a_pinned_entry_is_clipped_to_budget_with_a_marker() {
        let dir = tempfile::tempdir().unwrap();
        let big = dir.path().join("big.md");
        std::fs::write(&big, "line\n".repeat(5000)).unwrap();
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            resident: vec![ResidentEntry {
                path: PathBuf::from("big.md"),
                ring: 0,
                scope: Scope { always: true, ..Scope::default() },
                weight: None,
            }],
            code_roots: Vec::new(),
            budget_chars: 1000,
            ..Config::default()
        };
        let d = build(&cfg, &SessionScope { host: "any-host".into(), repo: None });
        assert!(d.text.len() < 1400, "digest was {} chars", d.text.len());
        assert!(d.text.contains("[clipped at "));
        assert!(index_line_for(&d, "big.md").is_none(), "a pinned entry is never demoted");
    }

    #[test]
    fn missing_file_yields_one_line_not_silence() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            resident: vec![ResidentEntry { path: PathBuf::from("absent.md"), ring: 1, scope: Scope::default(), weight: None }],
            code_roots: Vec::new(),
            budget_chars: 1000,
            ..Config::default()
        };
        let d = build(&cfg, &SessionScope { host: "any-host".into(), repo: None });
        assert!(d.text.contains("resident file missing"));
    }

    /// A file the budget cannot carry whole is advertised, not sawn in half:
    /// one line saying what it is, what reading it costs, and where it is.
    #[test]
    fn an_oversized_entry_is_indexed_instead_of_half_injected() {
        let dir = tempfile::tempdir().unwrap();
        let body = format!("# The operating contract\n\n{}", "prose ".repeat(4000));
        std::fs::write(dir.path().join("AGENT.md"), &body).unwrap();
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            budget_chars: 1000,
            resident: vec![entry("AGENT.md", 1, None)],
            ..Config::default()
        };
        let d = build(&cfg, &SessionScope::from_cwd(None));
        assert!(!d.text.contains("prose prose"), "the body was injected anyway:\n{}", d.text);
        assert!(!d.text.contains("[clipped at "), "half a file is not a disclosure level");
        let line = index_line_for(&d, "AGENT.md").expect("no index line");
        assert!(line.contains("The operating contract"), "the line does not say what it is: {line}");
        assert!(line.contains("~6.9k tok"), "the line does not price the file: {line}");
        assert!(
            line.contains(dir.path().join("AGENT.md").to_str().unwrap()),
            "the line does not say where the rest is: {line}"
        );
        assert_eq!(
            d.sources.iter().find(|(l, _)| l.contains("AGENT.md")).unwrap().1,
            line.len(),
            "an indexed entry is booked for what it actually cost"
        );
    }

    /// The one thing a pointer cannot stand in for. The prohibitions sit at
    /// the END of the file, where no clip would have reached them.
    #[test]
    fn the_hard_rules_of_an_indexed_file_arrive_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        let body = format!(
            "---\ndescription: how this host is run\n---\n{}\n## Hard Rules\n\n\
             1. **NEVER FORCE OFF VMs** — the guests do not survive it\n\
             - Never rsync between two ZFS pools\n\
             - No force-pushes to the default branch\n\
             - Do not enable dedup, ever\n\
             - Prefer Rust for systems daemons\n",
            "filler ".repeat(2000)
        );
        std::fs::write(dir.path().join("rules.md"), &body).unwrap();
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            budget_chars: 1200,
            resident: vec![entry("rules.md", 0, None)],
            ..Config::default()
        };
        let d = build(&cfg, &SessionScope::from_cwd(None));
        assert!(d.text.contains("NEVER FORCE OFF VMs"), "the first hard rule was lost:\n{}", d.text);
        assert!(d.text.contains("Never rsync between two ZFS pools"));
        assert!(d.text.contains("No force-pushes to the default branch"));
        assert!(!d.text.contains("Do not enable dedup"), "only the top few rules travel");
        assert!(!d.text.contains("Prefer Rust"), "a preference is not a prohibition");
        assert!(!d.text.contains("filler filler"), "the file itself was injected anyway");
        assert!(
            index_line_for(&d, "rules.md").is_some_and(|l| l.contains("how this host is run")),
            "the index line lost the file's own description"
        );
    }

    /// A file that arrived whole already carries its rules; repeating three of
    /// them under the index would be paying twice for the same sentence.
    #[test]
    fn a_fully_injected_entry_contributes_no_rule_lines() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("rules.md"), "- Never force off VMs\n").unwrap();
        let cfg = Config {
            brain_root: dir.path().to_path_buf(),
            resident: vec![entry("rules.md", 0, None)],
            ..Config::default()
        };
        let d = build(&cfg, &SessionScope::from_cwd(None));
        assert_eq!(d.text.matches("Never force off VMs").count(), 1);
        assert!(!d.text.contains("hard rules, verbatim"));
    }

    #[test]
    fn a_prohibition_too_long_for_the_line_is_dropped_not_truncated() {
        let long = format!("- Never delete {} unless the mirror is verified", "x".repeat(140));
        assert!(hard_rules(&long).is_empty(), "a truncated rule would state a broader one");
        assert_eq!(hard_rules("- Never delete the pool\n"), vec!["Never delete the pool"]);
        assert_eq!(
            hard_rules("1. **NO UNSUPERVISED INSTALLS**\n"),
            vec!["NO UNSUPERVISED INSTALLS"]
        );
        assert!(
            hard_rules("---\ndescription: never do this\n---\nbody\n").is_empty(),
            "frontmatter is a label, not a rule"
        );
        assert!(hard_rules("The rule is: do not do that\n").is_empty(), "only the opener decides");
    }

    #[test]
    fn the_rule_cap_counts_chars_not_bytes() {
        // The reporting shape: a German rule of 120 characters is 208 UTF-8
        // bytes and was silently dropped while an English rule of the same
        // visible length passed. The cap is a readability limit — it must
        // measure what the operator reads.
        let rule = format!("never {} gravierend", "ä".repeat(100));
        assert_eq!(rule.chars().count(), 117);
        assert!(rule.len() > 140, "the fixture must be over the cap in bytes");
        assert_eq!(hard_rules(&rule), vec![rule], "a readable-length rule survives");
        // And the cap itself still holds, in chars.
        let over = format!("never {}", "x".repeat(140));
        assert!(over.chars().count() > 140);
        assert!(hard_rules(&over).is_empty(), "past the cap in chars it drops, whatever its byte length");
    }

    /// A brain whose catalog holds far more than any digest could carry. The
    /// session must be told what it can still ask for — and never offered the
    /// file it was just handed.
    #[test]
    fn the_availability_index_prices_what_recall_can_still_reach() {
        let brain = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(brain.path().join("AGENT.md"), "# contract\nbe good\n").unwrap();
        for (dir, name) in [
            ("knowledge", "a.md"),
            ("knowledge", "b.md"),
            ("knowledge/behaviours", "m.md"),
            ("todo", "t.md"),
        ] {
            std::fs::create_dir_all(brain.path().join(dir)).unwrap();
            std::fs::write(brain.path().join(dir).join(name), "x".repeat(3500)).unwrap();
        }
        let cfg = Config {
            brain_root: brain.path().to_path_buf(),
            resident: vec![entry("AGENT.md", 1, None)],
            ..Config::default()
        };
        let mut conn = crate::index::open(state.path()).unwrap();
        crate::index::scan(&mut conn, brain.path(), None, &cfg.rings()).unwrap();

        let d = build_in(&cfg, &SessionScope::from_cwd(None), state.path());
        let tail = d.text.lines().last().unwrap();
        assert!(
            tail.starts_with("[not injected, recallable:"),
            "no availability index:\n{}",
            d.text
        );
        assert!(tail.contains("ring 2 · 1 file(s) ~1.0k tok"), "{tail}");
        assert!(tail.contains("ring 3 · 2 file(s) ~2.0k tok"), "{tail}");
        assert!(tail.contains("ring 4 · 1 file(s) ~1.0k tok"), "{tail}");
        assert!(tail.contains("cfetch recall"), "the price without the way to pay it: {tail}");
        assert!(
            !tail.contains("ring 1"),
            "the only ring-1 file was just injected; advertising it asks for it twice: {tail}"
        );
        assert_eq!(
            d.sources.iter().filter(|(l, _)| l == "availability index").count(),
            1,
            "the advertisement costs tokens too and is booked like any other source"
        );
    }

    /// Absence of a catalog is not a catalog of nothing.
    #[test]
    fn a_host_with_no_catalog_advertises_nothing() {
        let brain = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(brain.path().join("AGENT.md"), "# contract\nbe good\n").unwrap();
        let cfg = Config {
            brain_root: brain.path().to_path_buf(),
            resident: vec![entry("AGENT.md", 1, None)],
            ..Config::default()
        };
        let d = build_in(&cfg, &SessionScope::from_cwd(None), state.path());
        assert!(!d.text.contains("not injected, recallable"), "priced from nothing:\n{}", d.text);
    }

    /// A none-tier host answers recall from its serving host. Pricing a local
    /// catalog here would build the second, silently stale truth that host
    /// exists to avoid.
    #[test]
    fn a_none_tier_client_never_prices_a_local_catalog() {
        let brain = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::write(brain.path().join("AGENT.md"), "# contract\nbe good\n").unwrap();
        std::fs::create_dir_all(brain.path().join("knowledge")).unwrap();
        std::fs::write(brain.path().join("knowledge/a.md"), "x".repeat(3500)).unwrap();
        let mut cfg = Config {
            brain_root: brain.path().to_path_buf(),
            resident: vec![entry("AGENT.md", 1, None)],
            ..Config::default()
        };
        let mut conn = crate::index::open(state.path()).unwrap();
        crate::index::scan(&mut conn, brain.path(), None, &cfg.rings()).unwrap();
        assert!(
            build_in(&cfg, &SessionScope::from_cwd(None), state.path())
                .text
                .contains("not injected, recallable"),
            "the catalog is there to be priced"
        );

        cfg.client.serving = Some(crate::config::ClientServingConfig {
            addr: "storage.example:9737".into(),
            token_file: PathBuf::from("/var/empty/token"),
        });
        let d = build_in(&cfg, &SessionScope::from_cwd(None), state.path());
        assert!(
            !d.text.contains("not injected, recallable"),
            "a client priced a local catalog:\n{}",
            d.text
        );
    }

    /// The index has to fit its own budget. When it cannot, the summaries go —
    /// they are the only convenience on the line; the name, the price and the
    /// path are the reason it exists.
    #[test]
    fn a_crowded_index_drops_its_summaries_before_its_files() {
        let dir = tempfile::tempdir().unwrap();
        let brain_root = dir
            .path()
            .join("a-deliberately-long-brain-root-that-is-stable-across-platforms");
        std::fs::create_dir_all(&brain_root).unwrap();
        let mut resident = Vec::new();
        for i in 0..20 {
            let name = format!("file-{i:02}.md");
            std::fs::write(
                brain_root.join(&name),
                format!("# a title long enough to matter for entry {i}\n{}", "x".repeat(4000)),
            )
            .unwrap();
            resident.push(entry(&name, 1, None));
        }
        let cfg = Config {
            brain_root,
            budget_chars: 1600,
            resident,
            ..Config::default()
        };
        let d = build(&cfg, &SessionScope::from_cwd(None));
        assert!(d.text.len() <= 1600, "digest was {} chars for a 1600 budget", d.text.len());
        assert!(!d.text.contains("a title long enough"), "the summaries survived the squeeze");
        assert!(
            d.text.contains("paths are relative to the configured brain root"),
            "the crowded index kept repeating its absolute root:\n{}",
            d.text
        );
        for i in 0..20 {
            let name = format!("file-{i:02}.md");
            assert!(index_line_for(&d, &name).is_some(), "{name} was dropped, not summarized");
        }
    }

    #[test]
    fn a_summary_prefers_the_description_then_the_title_then_the_prose() {
        assert_eq!(
            summarize("---\nname: x\ndescription: how this host is run\n---\n# Title\n").as_deref(),
            Some("how this host is run")
        );
        assert_eq!(summarize("# Title\n\nbody\n").as_deref(), Some("Title"));
        assert_eq!(summarize("just prose\nmore\n").as_deref(), Some("just prose"));
        assert_eq!(summarize("\n\n").as_deref(), None);
        let long = summarize(&format!("# {}", "word ".repeat(40))).unwrap();
        assert!(long.chars().count() <= SUMMARY_MAX_CHARS, "{long}");
        assert!(long.ends_with('…'), "a cut summary says it was cut: {long}");
    }
}
