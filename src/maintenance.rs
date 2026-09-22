//! Autonomous, evidence-grounded maintenance for the 6 -> 5 -> 2/3/4 trust crossing.
//!
//! The model-facing half is deliberately narrow: build a bounded evidence
//! packet and accept a typed proposal into ring-5 quarantine. A separate model
//! pass reviews the proposal, deterministic gates bind it to current bytes,
//! and the normal maintenance loop applies it without routine human approval.
//! Direct Markdown edits remain authoritative: stale work is rejected rather
//! than rebased over the user's bytes. Manual CLI operations remain available
//! for inspection, debugging, and exact reversion.

use std::collections::{BTreeSet, HashSet};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::config::{Config, RingRules};
use crate::{exhaust, fsutil, index, jsonl, paths, staging};

pub const SCHEMA_VERSION: u64 = 1;
const MAX_EVENTS: usize = 96;
const MAX_RELEVANT: usize = 12;
const MAX_SNAPSHOT_BYTES: usize = 64 * 1024;
const MAX_EVENT_BYTES: usize = 128 * 1024;
const MAX_CONTEXT_BYTES: usize = 64 * 1024;
const MAX_JOURNAL_TEXT_BYTES: usize = 16 * 1024;
const LOCK_WAIT_MS: u64 = 2_000;

const PENDING: &str = "pending";
const APPLIED: &str = "applied";
const FINALIZED: &str = "finalized";
const REJECTED: &str = "rejected";
const REVERTED: &str = "reverted";
const REVIEWS: &str = "reviews";
const HISTORY: &str = "history";
const PAUSE_FILE: &str = "PAUSED.md";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transition {
    Add,
    Fold,
    Supersede,
    Revalidate,
    Dismiss,
    Noop,
}

impl Transition {
    pub(crate) fn changes_memory(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Fold | Self::Supersede | Self::Revalidate
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Authority {
    /// A direct operator instruction authorizes the proposed claim.
    Authorized,
    /// The claim is supported by independently observable evidence.
    Attested,
    /// A third party or model inferred it; it may be reviewed but not trusted.
    Unendorsed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewVerdict {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewMethod {
    IndependentAgent,
    Human,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewInput {
    pub verdict: ReviewVerdict,
    pub method: ReviewMethod,
    pub evidence_coverage: bool,
    pub factual_faithfulness: bool,
    pub preservation: bool,
    pub authority_fit: bool,
    pub target_fit: bool,
    pub contradiction_checked: bool,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Review {
    pub schema_version: u64,
    pub id: String,
    pub proposal_id: String,
    pub proposal_sha256: String,
    pub created_at: i64,
    pub created_by_host: String,
    pub verdict: ReviewVerdict,
    pub method: ReviewMethod,
    pub evidence_coverage: bool,
    pub factual_faithfulness: bool,
    pub preservation: bool,
    pub authority_fit: bool,
    pub target_fit: bool,
    pub contradiction_checked: bool,
    pub notes: String,
}

/// What an agent submits. Current bytes, candidate hashes, and citation
/// snapshots are captured by cfetch itself rather than trusted from the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalInput {
    pub candidate_ids: Vec<String>,
    pub transition: Transition,
    pub target: Option<String>,
    pub after: Option<String>,
    pub authority: Authority,
    #[serde(default)]
    pub valid_until: Option<i64>,
    pub rationale: String,
    #[serde(default)]
    pub evidence: Vec<String>,
    #[serde(default)]
    pub related_citations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateSource {
    pub id: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CitationSource {
    pub cite: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proposal {
    pub schema_version: u64,
    pub id: String,
    pub created_at: i64,
    pub created_by_host: String,
    pub candidates: Vec<CandidateSource>,
    pub transition: Transition,
    pub target: Option<String>,
    pub authority: Authority,
    pub valid_until: Option<i64>,
    pub rationale: String,
    pub evidence: Vec<String>,
    pub related_citations: Vec<CitationSource>,
    pub before_sha256: Option<String>,
    pub before: Option<String>,
    pub after: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CandidateSnapshot {
    pub candidate: staging::Candidate,
    pub sha256: String,
    pub evidence_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidenceEvent {
    pub id: String,
    pub ts: i64,
    pub host: String,
    pub kind: String,
    pub session: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RelevantStatement {
    pub cite: String,
    pub path: String,
    pub ring: u8,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub sha256: String,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TargetSnapshot {
    pub path: String,
    pub ring: u8,
    pub exists: bool,
    pub sha256: Option<String>,
    pub content: Option<String>,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidencePacket {
    pub schema_version: u64,
    pub generated_at: i64,
    pub candidate: CandidateSnapshot,
    pub events: Vec<EvidenceEvent>,
    pub events_truncated: bool,
    pub unreadable_streams: Vec<String>,
    pub relevant_statements: Vec<RelevantStatement>,
    pub context_note: Option<String>,
    pub target_snapshot: Option<TargetSnapshot>,
    pub proposal_contract: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Check {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Verification {
    pub proposal_id: String,
    pub valid: bool,
    pub checks: Vec<Check>,
    pub review_id: Option<String>,
    pub approval_token: Option<String>,
    pub diff: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProposalSummary {
    pub id: String,
    pub state: String,
    pub transition: Transition,
    pub target: Option<String>,
    pub candidates: Vec<String>,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmitResult {
    pub proposal: Proposal,
    pub created: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalizeResult {
    pub proposal_id: String,
    pub candidate_ids: Vec<String>,
    pub already_finalized: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventOutcome {
    Applied,
    Dismissed,
    Noop,
    Reverted,
    Exception,
}

/// One immutable, human-readable maintenance event. Proposal and review files
/// retain the exact evidence and bytes; this is the compact timeline that the
/// CLI/TUI can read without reconstructing state transitions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaintenanceEvent {
    pub schema_version: u64,
    pub id: String,
    pub created_at: i64,
    pub created_by_host: String,
    pub proposal_id: Option<String>,
    pub candidate_ids: Vec<String>,
    pub outcome: EventOutcome,
    pub target: Option<String>,
    pub before_sha256: Option<String>,
    pub after_sha256: Option<String>,
    pub review_id: Option<String>,
    pub checks: Vec<Check>,
    pub detail: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct RunReport {
    pub paused: bool,
    pub examined: usize,
    pub applied: usize,
    pub dismissed: usize,
    pub noops: usize,
    pub exceptions: usize,
}

/// The inference boundary is intentionally smaller than the transaction
/// engine. Production may use local hardware or a configured remote endpoint;
/// tests and integrations can supply any implementation without weakening the
/// deterministic gates below it.
pub trait MaintenanceModel {
    fn propose(&mut self, packet: &EvidencePacket) -> anyhow::Result<ProposalInput>;
    fn review(
        &mut self,
        packet: &EvidencePacket,
        proposal: &Proposal,
    ) -> anyhow::Result<ReviewInput>;
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn hash_bytes(bytes: impl AsRef<[u8]>) -> String {
    crate::hashing::hex_lower(Sha256::digest(bytes.as_ref()))
}

fn candidate_hash(candidate: &staging::Candidate) -> String {
    hash_bytes(staging::render(candidate))
}

fn candidate_evidence_id(candidate: &staging::Candidate) -> String {
    format!("c5-{}", &candidate_hash(candidate)[..16])
}

fn event_id(record: &jsonl::Record) -> String {
    let value = serde_json::json!({
        "v": jsonl::FORMAT_VERSION,
        "ts": record.ts,
        "host": record.host,
        "fields": record.fields,
    });
    format!(
        "e6-{}",
        &hash_bytes(serde_json::to_vec(&value).unwrap_or_default())[..16]
    )
}

fn maintenance_root(brain_root: &Path) -> PathBuf {
    paths::staging_dir(brain_root).join("maintenance")
}

fn state_dir(brain_root: &Path, state: &str) -> PathBuf {
    maintenance_root(brain_root).join(state)
}

fn proposal_path(brain_root: &Path, state: &str, id: &str) -> PathBuf {
    state_dir(brain_root, state).join(format!("{id}.md"))
}

fn review_path(brain_root: &Path, proposal_id: &str) -> PathBuf {
    state_dir(brain_root, REVIEWS).join(format!("{proposal_id}.md"))
}

fn history_path(brain_root: &Path, event_id: &str) -> PathBuf {
    state_dir(brain_root, HISTORY).join(format!("{event_id}.md"))
}

fn pause_path(brain_root: &Path) -> PathBuf {
    maintenance_root(brain_root).join(PAUSE_FILE)
}

fn lock(cfg: &Config) -> anyhow::Result<crate::lockfile::Lock> {
    let logs = paths::logs_dir(&cfg.brain_root);
    std::fs::create_dir_all(&logs).with_context(|| format!("create {}", logs.display()))?;
    crate::lockfile::acquire(&logs.join("maintenance.lock"), LOCK_WAIT_MS, 0)
        .context("another cfetch maintenance transaction is active")
}

fn find_candidate(dir: &Path, id: &str) -> anyhow::Result<staging::Candidate> {
    anyhow::ensure!(
        id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
        "invalid candidate id"
    );
    let path = staging::path_of(dir, id);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("no pending staging candidate {id}"))?;
    let candidate = staging::parse(&text, id)
        .with_context(|| format!("pending staging candidate {id} is malformed"))?;
    anyhow::ensure!(
        candidate.id == id,
        "candidate id inside {} does not match its filename",
        path.display()
    );
    Ok(candidate)
}

fn matching_event(candidate: &staging::Candidate, record: &jsonl::Record) -> bool {
    let payload = record
        .value("payload")
        .and_then(serde_json::Value::as_object);
    match candidate.reason.as_str() {
        "fix-discovered" | "recurring-failure" => {
            let wanted = candidate
                .payload
                .get("norm")
                .and_then(serde_json::Value::as_str);
            wanted.is_some()
                && record.kind() == "bash"
                && payload
                    .and_then(|p| p.get("norm"))
                    .and_then(serde_json::Value::as_str)
                    == wanted
        }
        "hot-file" => {
            let wanted = candidate
                .payload
                .get("file_path")
                .and_then(serde_json::Value::as_str);
            wanted.is_some()
                && record.kind() == "write"
                && payload
                    .and_then(|p| p.get("file_path"))
                    .and_then(serde_json::Value::as_str)
                    == wanted
        }
        _ => {
            record.host == candidate.host
                && record.str("session") == candidate.session
                && record.kind() == candidate.kind
        }
    }
}

fn evidence_events(
    candidate: &staging::Candidate,
    records: &[jsonl::Record],
) -> Vec<EvidenceEvent> {
    let mut bytes = 0usize;
    records
        .iter()
        .filter(|record| matching_event(candidate, record))
        .filter_map(|record| {
            let payload = record
                .value("payload")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let cost = serde_json::to_vec(&payload)
                .map(|value| value.len())
                .unwrap_or(0);
            if bytes.saturating_add(cost) > MAX_EVENT_BYTES {
                return None;
            }
            bytes += cost;
            Some(EvidenceEvent {
                id: event_id(record),
                ts: record.ts,
                host: record.host.clone(),
                kind: record.kind().to_string(),
                session: record.str("session").to_string(),
                payload,
            })
        })
        .take(MAX_EVENTS)
        .collect()
}

fn evidence_coverage(
    candidate: &staging::Candidate,
    records: &[jsonl::Record],
    selected: &HashSet<&str>,
    // At apply time: the proposal's created_at. New matching events that
    // arrived AFTER the proposal was submitted could not have been cited
    // by the proposer; requiring it to have cited them is impossible.
    // `None` at submit time (first evaluation, no proposal yet).
    proposal_created_at: Option<i64>,
) -> anyhow::Result<String> {
    let matching: Vec<&jsonl::Record> = records
        .iter()
        .filter(|record| matching_event(candidate, record))
        .collect();
    if matching.is_empty() {
        let fallback = candidate_evidence_id(candidate);
        anyhow::ensure!(
            selected.contains(fallback.as_str()),
            "candidate {} has no raw match and its candidate evidence id {fallback} was not cited",
            candidate.id
        );
        return Ok(
            "candidate record covers the source whose raw event has rotated out".to_string(),
        );
    }
    // Regime-flip guard: at APPLY time, if the fallback was cited and every
    // matching event postdates the proposal, the fallback stays sufficient.
    // The proposer cited the candidate record because the raw events HAD
    // rotated out at submit time; new events arriving during the review
    // round-trip do not retroactively invalidate that decision.
    if let Some(created_at) = proposal_created_at {
        let fallback = candidate_evidence_id(candidate);
        if selected.contains(fallback.as_str())
            && matching.iter().all(|record| record.ts > created_at)
        {
            return Ok(format!(
                "candidate record cited; all {} matching raw event(s) arrived after the proposal was submitted",
                matching.len()
            ));
        }
    }
    let chosen: Vec<&jsonl::Record> = matching
        .into_iter()
        .filter(|record| {
            let id = event_id(record);
            selected.contains(id.as_str())
        })
        .collect();
    match candidate.reason.as_str() {
        "fix-discovered" => {
            let failed = chosen.iter().any(|record| {
                record
                    .value("payload")
                    .and_then(|payload| payload.get("failed"))
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
            });
            let fixed = chosen.iter().any(|record| {
                record
                    .value("payload")
                    .and_then(|payload| payload.get("failed"))
                    .and_then(serde_json::Value::as_bool)
                    == Some(false)
            });
            anyhow::ensure!(
                failed && fixed,
                "fix-discovered candidate {} requires both its failure and successful recovery evidence",
                candidate.id
            );
        }
        "recurring-failure" => {
            let sessions: BTreeSet<&str> =
                chosen.iter().map(|record| record.str("session")).collect();
            anyhow::ensure!(
                sessions.len() >= 2,
                "recurring-failure candidate {} requires failing evidence from at least two sessions",
                candidate.id
            );
        }
        "hot-file" => {
            let sessions: BTreeSet<&str> =
                chosen.iter().map(|record| record.str("session")).collect();
            let current = chosen
                .iter()
                .filter(|record| record.str("session") == candidate.session)
                .count();
            anyhow::ensure!(
                sessions.len() >= 2 || current >= 10,
                "hot-file candidate {} requires two sessions or ten writes in its stopping session",
                candidate.id
            );
        }
        _ => anyhow::ensure!(
            !chosen.is_empty(),
            "candidate {} is not covered by matching raw evidence",
            candidate.id
        ),
    }
    Ok(format!(
        "{} raw event(s) satisfy the {} trap",
        chosen.len(),
        candidate.reason
    ))
}

fn query_for(candidate: &staging::Candidate) -> String {
    let mut terms = Vec::new();
    for key in [
        "norm",
        "command",
        "failed_command",
        "fixed_command",
        "file_path",
    ] {
        if let Some(value) = candidate
            .payload
            .get(key)
            .and_then(serde_json::Value::as_str)
        {
            for term in value.split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-') {
                let term = term.trim();
                if term.len() >= 3 && !terms.iter().any(|seen| seen == term) {
                    terms.push(term.to_string());
                }
                // The cap applies to the WHOLE term list, not per key: the
                // inner `break` only exits this key's loop, and the next key
                // restarts it — `== 12` never fires again and every unique
                // token from every later payload key joins the query.
                if terms.len() >= 12 {
                    break;
                }
            }
            if terms.len() >= 12 {
                break;
            }
        }
    }
    if terms.is_empty() {
        candidate.reason.replace('-', " ")
    } else {
        terms.join(" ")
    }
}

fn safe_relative(raw: &str) -> anyhow::Result<String> {
    anyhow::ensure!(!raw.trim().is_empty(), "target path is empty");
    anyhow::ensure!(!raw.contains('\\'), "target path must use '/' separators");
    let path = Path::new(raw);
    anyhow::ensure!(!path.is_absolute(), "target path must be brain-relative");
    for component in path.components() {
        anyhow::ensure!(
            matches!(component, Component::Normal(_)),
            "target path may not contain '.', '..', a root, or a drive prefix"
        );
    }
    let normalized = path.to_string_lossy().to_string();
    anyhow::ensure!(
        normalized.ends_with(".md"),
        "maintenance targets Markdown files only"
    );
    Ok(normalized)
}

fn reject_symlink_path(brain_root: &Path, rel: &str) -> anyhow::Result<PathBuf> {
    let root = std::fs::canonicalize(brain_root)
        .with_context(|| format!("resolve brain root {}", brain_root.display()))?;
    let mut current = root.clone();
    for component in Path::new(rel).components() {
        let Component::Normal(name) = component else {
            unreachable!()
        };
        current.push(name);
        match std::fs::symlink_metadata(&current) {
            Ok(meta) => anyhow::ensure!(
                !meta.file_type().is_symlink(),
                "maintenance target crosses symlink {}",
                current.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).with_context(|| format!("stat {}", current.display())),
        }
    }
    Ok(root.join(rel))
}

fn effective_ring(rel: &str, after: &str, rules: &RingRules) -> u8 {
    index::frontmatter_ring(after)
        .0
        .unwrap_or_else(|| rules.ring_for(rel))
}

fn target_policy(
    rel: &str,
    after: &str,
    authority: Authority,
    rules: &RingRules,
) -> anyhow::Result<u8> {
    anyhow::ensure!(
        index::indexable_doc(rel, rules),
        "target is excluded from the recallable Markdown tree"
    );
    let ring = effective_ring(rel, after, rules);
    anyhow::ensure!(
        (2..=4).contains(&ring),
        "maintenance may write only rings 2-4; {rel} resolves to ring {ring}"
    );
    match (ring, authority) {
        (2, Authority::Authorized) => {}
        (2, _) => {
            anyhow::bail!("ring 2 requires authorized authority from a direct operator instruction")
        }
        (3, Authority::Authorized | Authority::Attested) => {}
        (_, Authority::Unendorsed) => {
            anyhow::bail!("unendorsed model or third-party claims cannot cross into trusted memory")
        }
        _ => {}
    }
    Ok(ring)
}

fn secret_shape(text: &str) -> Option<&'static str> {
    let lower = text.to_ascii_lowercase();
    if lower.contains("-----begin private key-----")
        || lower.contains("-----begin rsa private key-----")
        || lower.contains("-----begin openssh private key-----")
    {
        return Some("private-key block");
    }
    for token in text
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| "'\"`,;()[]{}".contains(c)))
    {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() == 3
            && parts.iter().all(|part| {
                part.len() >= 16
                    && part
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
            })
        {
            return Some("JWT-shaped token");
        }
        if (token.starts_with("sk-")
            || token.starts_with("ghp_")
            || token.starts_with("github_pat_"))
            && token.len() >= 20
        {
            return Some("API-token-shaped value");
        }
        if token.starts_with("AKIA")
            && token.len() == 20
            && token
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
        {
            return Some("AWS-key-shaped value");
        }
    }
    for line in text.lines() {
        let Some((name, value)) = line.split_once('=').or_else(|| line.split_once(':')) else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase().replace(['-', '_'], "");
        let secret_name = [
            "password",
            "passwd",
            "secret",
            "apikey",
            "accesstoken",
            "authtoken",
            "privatekey",
        ]
        .iter()
        .any(|needle| name.ends_with(needle));
        let value = value.trim().trim_matches(['\'', '"']);
        let placeholder = value.is_empty()
            || value.contains("<redacted>")
            || value.contains("${")
            || value.contains("{{")
            || value.eq_ignore_ascii_case("from environment")
            || (value.len() >= 3
                && value.chars().all(|character| {
                    character.is_ascii_uppercase() || character.is_ascii_digit() || character == '_'
                }));
        if secret_name && !placeholder {
            return Some("literal assigned to a secret-shaped name");
        }
    }
    None
}

fn candidate_target(candidate: &staging::Candidate, brain_root: &Path) -> Option<String> {
    let raw = candidate.payload.get("file_path")?.as_str()?;
    let path = Path::new(raw);
    let rel = if path.is_absolute() {
        path.strip_prefix(brain_root).ok()?
    } else {
        path
    };
    safe_relative(&rel.to_string_lossy()).ok()
}

fn snapshot_target(candidate: &staging::Candidate, cfg: &Config) -> Option<TargetSnapshot> {
    let rel = candidate_target(candidate, &cfg.brain_root)?;
    let path = reject_symlink_path(&cfg.brain_root, &rel).ok()?;
    let ring = cfg.rings().ring_for(&rel);
    match std::fs::read(&path) {
        Ok(bytes) => {
            let sha256 = hash_bytes(&bytes);
            let truncated = bytes.len() > MAX_SNAPSHOT_BYTES;
            let shown = &bytes[..bytes.len().min(MAX_SNAPSHOT_BYTES)];
            Some(TargetSnapshot {
                path: rel,
                ring,
                exists: true,
                sha256: Some(sha256),
                content: Some(String::from_utf8_lossy(shown).to_string()),
                truncated,
            })
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Some(TargetSnapshot {
            path: rel,
            ring,
            exists: false,
            sha256: None,
            content: None,
            truncated: false,
        }),
        Err(_) => None,
    }
}

pub fn proposal_contract() -> serde_json::Value {
    serde_json::json!({
        "candidate_ids": ["the exact candidate id from this packet"],
        "transition": "add | fold | supersede | revalidate | dismiss | noop",
        "target": "brain-relative .md path; required for content transitions, otherwise null",
        "after": "complete proposed Markdown bytes; required for content transitions, otherwise null",
        "authority": "authorized | attested | unendorsed",
        "valid_until": "optional Unix timestamp",
        "rationale": "why this transition follows from the cited evidence",
        "evidence": ["candidate c5-* or raw e6-* ids; include raw ids when present"],
        "related_citations": ["current rN-* citations considered for conflicts or folding"]
    })
}

pub fn packet(cfg: &Config, candidate_id: &str) -> anyhow::Result<EvidencePacket> {
    packet_at(
        cfg,
        candidate_id,
        &paths::state_dir(),
        &paths::native_projects_root(),
    )
}

fn packet_at(
    cfg: &Config,
    candidate_id: &str,
    local_state: &Path,
    native_projects: &Path,
) -> anyhow::Result<EvidencePacket> {
    let staging_dir = paths::staging_dir(&cfg.brain_root);
    let candidate = find_candidate(&staging_dir, candidate_id)?;
    let all = jsonl::read_all(&paths::logs_dir(&cfg.brain_root), exhaust::STREAM);
    let matched = evidence_events(&candidate, &all.records);
    let events_truncated = all
        .records
        .iter()
        .filter(|record| matching_event(&candidate, record))
        .count()
        > matched.len();

    let mut relevant_statements = Vec::new();
    let mut context_note = None;
    match index::ensure_fresh(
        local_state,
        &cfg.brain_root,
        Some(native_projects),
        &cfg.rings(),
    )
    .and_then(|conn| index::recall(&conn, &query_for(&candidate), MAX_RELEVANT))
    {
        Ok(hits) => {
            let mut remaining = MAX_CONTEXT_BYTES;
            for hit in hits {
                if remaining == 0 {
                    break;
                }
                let full_hash = hash_bytes(hit.text.as_bytes());
                let shown = hit.text.len().min(remaining);
                let truncated = shown < hit.text.len();
                let text = String::from_utf8_lossy(&hit.text.as_bytes()[..shown]).to_string();
                remaining = remaining.saturating_sub(shown);
                relevant_statements.push(RelevantStatement {
                    cite: hit.cite,
                    path: hit.path,
                    ring: hit.ring,
                    start_line: hit.start_line,
                    end_line: hit.end_line,
                    text,
                    sha256: full_hash,
                    truncated,
                });
            }
        }
        Err(error) => context_note = Some(format!("current-statement lookup unavailable: {error}")),
    }

    Ok(EvidencePacket {
        schema_version: SCHEMA_VERSION,
        generated_at: now(),
        candidate: CandidateSnapshot {
            evidence_id: candidate_evidence_id(&candidate),
            sha256: candidate_hash(&candidate),
            candidate: candidate.clone(),
        },
        events: matched,
        events_truncated,
        unreadable_streams: all.unreadable,
        relevant_statements,
        context_note,
        target_snapshot: snapshot_target(&candidate, cfg),
        proposal_contract: proposal_contract(),
    })
}

fn render(proposal: &Proposal) -> String {
    let json = serde_json::to_string_pretty(proposal).unwrap_or_else(|_| "{}".to_string());
    format!(
        "---\nring: 5\ntype: cfetch-maintenance-proposal\nid: {:?}\ntransition: {:?}\n---\n\n\
         Quarantined maintenance proposal. It is not recalled or injected. The automatic and\n\
         manual paths re-run exact evidence, authority, and target checks before applying it.\n\
         Inspect those checks with `cfetch maintain verify {}`.\n\n\
         ```json\n{}\n```\n",
        proposal.id,
        format!("{:?}", proposal.transition).to_ascii_lowercase(),
        proposal.id,
        json,
    )
}

fn render_review(review: &Review) -> String {
    let json = serde_json::to_string_pretty(review).unwrap_or_else(|_| "{}".to_string());
    format!(
        "---\nring: 5\ntype: cfetch-maintenance-review\nid: {:?}\nproposal: {:?}\nverdict: {:?}\n---\n\n\
         Immutable semantic review of a quarantined maintenance proposal. Deterministic\n\
         gates are re-run separately before either automatic or manual application.\n\n\
         ```json\n{}\n```\n",
        review.id,
        review.proposal_id,
        format!("{:?}", review.verdict).to_ascii_lowercase(),
        json,
    )
}

fn render_event(event: &MaintenanceEvent) -> String {
    let json = serde_json::to_string_pretty(event).unwrap_or_else(|_| "{}".to_string());
    format!(
        "---\nring: 5\ntype: cfetch-maintenance-event\nid: {:?}\noutcome: {:?}\n---\n\n\
         Immutable maintenance history. The proposal and review named below retain the\n\
         complete evidence, snapshots, and rationale.\n\n\
         ```json\n{}\n```\n",
        event.id,
        format!("{:?}", event.outcome).to_ascii_lowercase(),
        json,
    )
}

fn load_event_at(path: &Path) -> anyhow::Result<MaintenanceEvent> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let event: MaintenanceEvent =
        serde_json::from_value(fenced_json(&text)?).context("decode maintenance event")?;
    let expected = event_identity(&event)?;
    anyhow::ensure!(
        event.id == expected,
        "maintenance event content address mismatch: expected {expected}"
    );
    anyhow::ensure!(
        path.file_stem().and_then(|name| name.to_str()) == Some(event.id.as_str()),
        "maintenance event filename does not match its content address"
    );
    Ok(event)
}

fn event_identity(event: &MaintenanceEvent) -> anyhow::Result<String> {
    let mut identity = event.clone();
    identity.id.clear();
    Ok(format!(
        "event-{}",
        &hash_bytes(serde_json::to_vec(&identity)?)[..16]
    ))
}

fn write_event(cfg: &Config, mut event: MaintenanceEvent) -> anyhow::Result<MaintenanceEvent> {
    event.detail = journal_text(&event.detail);
    for check in &mut event.checks {
        check.detail = journal_text(&check.detail);
    }
    event.id = event_identity(&event)?;
    let path = history_path(&cfg.brain_root, &event.id);
    if path.exists() {
        let existing = load_event_at(&path)?;
        anyhow::ensure!(
            existing == event,
            "maintenance event id collision at {}",
            path.display()
        );
        return Ok(existing);
    }
    fsutil::atomic_write(&path, render_event(&event))?;
    Ok(event)
}

fn journal_text(value: &str) -> String {
    if let Some(shape) = secret_shape(value) {
        return format!("detail redacted by the maintenance journal ({shape})");
    }
    if value.len() <= MAX_JOURNAL_TEXT_BYTES {
        return value.to_string();
    }
    let mut end = MAX_JOURNAL_TEXT_BYTES;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… [journal detail truncated]", &value[..end])
}

pub fn history(cfg: &Config) -> Vec<MaintenanceEvent> {
    let Ok(entries) = std::fs::read_dir(state_dir(&cfg.brain_root, HISTORY)) else {
        return Vec::new();
    };
    let mut events: Vec<_> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("md"))
        .filter_map(|path| load_event_at(&path).ok())
        .collect();
    events.sort_by(|a, b| {
        b.created_at
            .cmp(&a.created_at)
            .then_with(|| a.id.cmp(&b.id))
    });
    events
}

/// Corrupt or edited history is never accepted as an automatic outcome. The
/// doctor uses these bounded, brain-relative diagnostics to make tampering or
/// storage damage visible instead of silently lowering the event count.
pub fn history_issues(cfg: &Config) -> Vec<String> {
    let dir = state_dir(&cfg.brain_root, HISTORY);
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(error) => return vec![format!("history directory unreadable: {error}")],
    };
    let mut issues = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("md"))
        .filter_map(|path| {
            load_event_at(&path).err().map(|error| {
                format!(
                    "{}: {error}",
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("history record")
                )
            })
        })
        .collect::<Vec<_>>();
    issues.sort();
    issues
}

fn event_for(
    proposal: Option<&Proposal>,
    outcome: EventOutcome,
    review_id: Option<String>,
    checks: Vec<Check>,
    detail: impl Into<String>,
    fallback_candidates: Vec<String>,
) -> MaintenanceEvent {
    MaintenanceEvent {
        schema_version: SCHEMA_VERSION,
        id: String::new(),
        created_at: now(),
        created_by_host: paths::host_id(),
        proposal_id: proposal.map(|proposal| proposal.id.clone()),
        candidate_ids: proposal
            .map(|proposal| {
                proposal
                    .candidates
                    .iter()
                    .map(|source| source.id.clone())
                    .collect()
            })
            .unwrap_or(fallback_candidates),
        outcome,
        target: proposal.and_then(|proposal| proposal.target.clone()),
        before_sha256: proposal.and_then(|proposal| proposal.before_sha256.clone()),
        after_sha256: proposal.and_then(|proposal| proposal.after.as_ref().map(hash_bytes)),
        review_id,
        checks,
        detail: detail.into(),
    }
}

fn fenced_json(text: &str) -> anyhow::Result<serde_json::Value> {
    let mut in_json = false;
    let mut body = String::new();
    for line in text.lines() {
        if in_json {
            if line.trim_start().starts_with("```") {
                break;
            }
            body.push_str(line);
            body.push('\n');
        } else if line.trim() == "```json" {
            in_json = true;
        }
    }
    anyhow::ensure!(in_json, "proposal has no fenced JSON record");
    serde_json::from_str(&body).context("parse proposal JSON")
}

fn load_at(path: &Path) -> anyhow::Result<Proposal> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let value = fenced_json(&text)?;
    serde_json::from_value(value).context("decode maintenance proposal")
}

fn load_review_at(path: &Path) -> anyhow::Result<Review> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let value = fenced_json(&text)?;
    serde_json::from_value(value).context("decode maintenance review")
}

fn locate(
    brain_root: &Path,
    id: &str,
    states: &[&str],
) -> anyhow::Result<(String, PathBuf, Proposal)> {
    anyhow::ensure!(
        id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
        "invalid proposal id"
    );
    for state in states {
        let path = proposal_path(brain_root, state, id);
        if path.is_file() {
            return Ok(((*state).to_string(), path.clone(), load_at(&path)?));
        }
    }
    anyhow::bail!("no maintenance proposal {id} in {}", states.join(" or "))
}

fn resolve_citations(cfg: &Config, cites: &[String]) -> anyhow::Result<Vec<CitationSource>> {
    if cites.is_empty() {
        return Ok(Vec::new());
    }
    let native = paths::native_projects_root();
    let conn = index::ensure_fresh(
        &paths::state_dir(),
        &cfg.brain_root,
        Some(&native),
        &cfg.rings(),
    )?;
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for cite in cites {
        if !seen.insert(cite.clone()) {
            continue;
        }
        let blocks = index::expand(&conn, cite)?;
        anyhow::ensure!(
            blocks.len() == 1,
            "related citation {cite} does not resolve uniquely"
        );
        out.push(CitationSource {
            cite: cite.clone(),
            sha256: hash_bytes(blocks[0].text.as_bytes()),
        });
    }
    Ok(out)
}

fn canonical_input(mut input: ProposalInput) -> ProposalInput {
    input.candidate_ids.sort();
    input.candidate_ids.dedup();
    input.evidence.sort();
    input.evidence.dedup();
    input.related_citations.sort();
    input.related_citations.dedup();
    input
}

fn validate_input(input: &ProposalInput) -> anyhow::Result<()> {
    anyhow::ensure!(
        !input.candidate_ids.is_empty(),
        "proposal names no staging candidates"
    );
    anyhow::ensure!(
        input.candidate_ids.len() <= 32,
        "one proposal may reference at most 32 candidates"
    );
    anyhow::ensure!(
        !input.rationale.trim().is_empty(),
        "proposal rationale is empty"
    );
    anyhow::ensure!(
        input.rationale.len() <= 16 * 1024,
        "proposal rationale is too large"
    );
    if let Some(shape) = secret_shape(&input.rationale) {
        anyhow::bail!("proposal rationale contains a {shape}");
    }
    if input.transition.changes_memory() {
        anyhow::ensure!(input.target.is_some(), "content transition requires target");
        anyhow::ensure!(
            input.after.is_some(),
            "content transition requires complete after bytes"
        );
        // The model's output is capped at 16384 tokens (~24-65 KB of text,
        // less with JSON escaping). A proposal whose `after` exceeds what
        // the model can echo back is structurally unproposable through the
        // autonomous path: the response truncates, the JSON parse fails,
        // and the candidate throws an exception every retry. Refuse here,
        // at the gate, instead of sending the model an impossible task.
        // 3 bytes/token is conservative for English and code; JSON string
        // escaping inflates by another ~2x on markdown-heavy content.
        const MODEL_AFTER_BUDGET: usize = 16_384 * 3 / 2; // ~24 KB
        if let Some(after) = &input.after {
            anyhow::ensure!(
                after.len() <= MODEL_AFTER_BUDGET,
                "proposed Markdown is {} bytes; the model's output budget ({ } tokens) can echo at most ~{} bytes. \
                 Split the change or apply it manually with `cfetch maintain apply`",
                after.len(),
                crate::maintenance_model::MAX_OUTPUT_TOKENS,
                MODEL_AFTER_BUDGET,
            );
        }
    } else {
        anyhow::ensure!(
            input.target.is_none() && input.after.is_none(),
            "dismiss/noop may not carry a target or content"
        );
    }
    Ok(())
}

pub fn submit(cfg: &Config, input: ProposalInput) -> anyhow::Result<SubmitResult> {
    let input = canonical_input(input);
    validate_input(&input)?;
    let _lock = lock(cfg)?;
    let staging_dir = paths::staging_dir(&cfg.brain_root);
    let candidates: Vec<staging::Candidate> = input
        .candidate_ids
        .iter()
        .map(|id| find_candidate(&staging_dir, id))
        .collect::<anyhow::Result<_>>()?;

    let all = jsonl::read_all(&paths::logs_dir(&cfg.brain_root), exhaust::STREAM);
    let selected: HashSet<&str> = input.evidence.iter().map(String::as_str).collect();
    for candidate in &candidates {
        evidence_coverage(candidate, &all.records, &selected, None)?;
    }

    let (target, before, before_sha256, after) = if input.transition.changes_memory() {
        let rel = safe_relative(input.target.as_deref().unwrap_or_default())?;
        let after = input.after.clone().unwrap_or_default();
        anyhow::ensure!(
            after.len() <= 2 * 1024 * 1024,
            "proposed Markdown exceeds the 2 MiB maintenance limit"
        );
        if let Some(shape) = secret_shape(&after) {
            anyhow::bail!("proposed Markdown contains a {shape}");
        }
        target_policy(&rel, &after, input.authority, &cfg.rings())?;
        let path = reject_symlink_path(&cfg.brain_root, &rel)?;
        let before = match std::fs::read_to_string(&path) {
            Ok(content) => Some(content),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
        };
        anyhow::ensure!(
            before.as_ref().map_or(0, String::len) <= 2 * 1024 * 1024,
            "current Markdown exceeds the 2 MiB maintenance limit"
        );
        match input.transition {
            Transition::Add => anyhow::ensure!(before.is_none(), "add target {rel} already exists"),
            _ => anyhow::ensure!(
                before.is_some(),
                "{:?} target {rel} does not exist",
                input.transition
            ),
        }
        anyhow::ensure!(
            before.as_deref() != Some(after.as_str()),
            "proposal makes no byte-level change"
        );
        let before_sha256 = before.as_ref().map(hash_bytes);
        (Some(rel), before, before_sha256, Some(after))
    } else {
        (None, None, None, None)
    };

    let related_citations = resolve_citations(cfg, &input.related_citations)?;
    let candidate_sources = candidates
        .iter()
        .map(|candidate| CandidateSource {
            id: candidate.id.clone(),
            sha256: candidate_hash(candidate),
        })
        .collect();
    let mut proposal = Proposal {
        schema_version: SCHEMA_VERSION,
        id: String::new(),
        created_at: now(),
        created_by_host: paths::host_id(),
        candidates: candidate_sources,
        transition: input.transition,
        target,
        authority: input.authority,
        valid_until: input.valid_until,
        rationale: input.rationale,
        evidence: input.evidence,
        related_citations,
        before_sha256,
        before,
        after,
    };
    let mut identity = proposal.clone();
    identity.created_at = 0;
    identity.id.clear();
    proposal.id = format!(
        "maintenance-{}",
        &hash_bytes(serde_json::to_vec(&identity)?)[..12]
    );

    let path = proposal_path(&cfg.brain_root, PENDING, &proposal.id);
    if path.exists() {
        let existing = load_at(&path)?;
        anyhow::ensure!(
            existing == proposal || {
                let mut existing_identity = existing.clone();
                existing_identity.created_at = 0;
                let mut proposal_identity = proposal.clone();
                proposal_identity.created_at = 0;
                existing_identity == proposal_identity
            },
            "proposal id collision at {}",
            path.display()
        );
        return Ok(SubmitResult {
            proposal: existing,
            created: false,
        });
    }
    // A proposal whose content-addressed id already sits in a TERMINAL state
    // must not resurrect: the autonomous loop re-derives deterministic ids
    // every cycle, and a rejected twin in PENDING would fail the same gate
    // forever (the loop's reject is now idempotent, but the spend and the
    // exception history are not free). Terminal REJECTED blocks re-submission
    // of identical content; REVERTED means the operator rolled an applied
    // proposal back - also a decision, also respected.
    for state in [REJECTED, REVERTED] {
        let terminal = proposal_path(&cfg.brain_root, state, &proposal.id);
        if terminal.exists() {
            let existing = load_at(&terminal)?;
            let same = existing == proposal || {
                let mut a = existing.clone();
                a.created_at = 0;
                let mut b = proposal.clone();
                b.created_at = 0;
                a == b
            };
            anyhow::ensure!(
                !same,
                "proposal {id} was already {state}; identical content will not be re-proposed",
                id = proposal.id
            );
        }
    }
    fsutil::atomic_write(&path, render(&proposal))?;
    Ok(SubmitResult {
        proposal,
        created: true,
    })
}

pub fn get(cfg: &Config, id: &str) -> anyhow::Result<(String, Proposal)> {
    let (state, _, proposal) = locate(
        &cfg.brain_root,
        id,
        &[PENDING, APPLIED, FINALIZED, REJECTED, REVERTED],
    )?;
    Ok((state, proposal))
}

pub fn get_review(cfg: &Config, proposal_id: &str) -> anyhow::Result<Option<Review>> {
    // Same gate as `locate`: every current caller validates first, but this
    // function builds a path from the id, so the check belongs here too
    // rather than in the caller's discipline.
    anyhow::ensure!(
        proposal_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-'),
        "invalid proposal id"
    );
    let path = review_path(&cfg.brain_root, proposal_id);
    if !path.is_file() {
        return Ok(None);
    }
    load_review_at(&path).map(Some)
}

pub fn list(cfg: &Config) -> Vec<ProposalSummary> {
    let mut out = Vec::new();
    for state in [PENDING, APPLIED, FINALIZED, REJECTED, REVERTED] {
        let Ok(entries) = std::fs::read_dir(state_dir(&cfg.brain_root, state)) else {
            continue;
        };
        for path in entries.flatten().map(|entry| entry.path()) {
            if path.extension().and_then(|ext| ext.to_str()) != Some("md") {
                continue;
            }
            let Ok(proposal) = load_at(&path) else {
                continue;
            };
            out.push(ProposalSummary {
                id: proposal.id,
                state: state.to_string(),
                transition: proposal.transition,
                target: proposal.target,
                candidates: proposal
                    .candidates
                    .into_iter()
                    .map(|candidate| candidate.id)
                    .collect(),
                created_at: proposal.created_at,
            });
        }
    }
    out.sort_by(|a, b| {
        b.created_at
            .cmp(&a.created_at)
            .then_with(|| a.id.cmp(&b.id))
    });
    out
}

pub fn pending_count(cfg: &Config) -> usize {
    proposal_count(cfg, PENDING)
}

pub fn applied_count(cfg: &Config) -> usize {
    proposal_count(cfg, APPLIED)
}

fn proposal_count(cfg: &Config, state: &str) -> usize {
    std::fs::read_dir(state_dir(&cfg.brain_root, state))
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("md"))
                .count()
        })
        .unwrap_or(0)
}

fn push_check(checks: &mut Vec<Check>, name: &str, result: anyhow::Result<String>) {
    match result {
        Ok(detail) => checks.push(Check {
            name: name.to_string(),
            ok: true,
            detail,
        }),
        Err(error) => checks.push(Check {
            name: name.to_string(),
            ok: false,
            detail: error.to_string(),
        }),
    }
}

fn verify_identity(proposal: &Proposal) -> anyhow::Result<String> {
    anyhow::ensure!(
        proposal.schema_version == SCHEMA_VERSION,
        "unsupported proposal schema {}",
        proposal.schema_version
    );
    let mut identity = proposal.clone();
    identity.id.clear();
    identity.created_at = 0;
    let expected = format!(
        "maintenance-{}",
        &hash_bytes(serde_json::to_vec(&identity)?)[..12]
    );
    anyhow::ensure!(
        proposal.id == expected,
        "content address mismatch: expected {expected}"
    );
    Ok(format!(
        "content address {} matches proposal bytes",
        proposal.id
    ))
}

fn verify_candidates(cfg: &Config, proposal: &Proposal) -> anyhow::Result<String> {
    let dir = paths::staging_dir(&cfg.brain_root);
    for source in &proposal.candidates {
        let candidate = find_candidate(&dir, &source.id)?;
        anyhow::ensure!(
            candidate_hash(&candidate) == source.sha256,
            "candidate {} changed after proposal",
            source.id
        );
    }
    Ok(format!(
        "{} candidate(s) are still pending and byte-identical",
        proposal.candidates.len()
    ))
}

fn verify_evidence(cfg: &Config, proposal: &Proposal) -> anyhow::Result<String> {
    let dir = paths::staging_dir(&cfg.brain_root);
    let all = jsonl::read_all(&paths::logs_dir(&cfg.brain_root), exhaust::STREAM);
    anyhow::ensure!(
        all.unreadable.is_empty(),
        "{} exhaust stream(s) are unreadable",
        all.unreadable.len()
    );
    let selected: HashSet<&str> = proposal.evidence.iter().map(String::as_str).collect();
    for source in &proposal.candidates {
        let candidate = find_candidate(&dir, &source.id)?;
        evidence_coverage(
            &candidate,
            &all.records,
            &selected,
            Some(proposal.created_at),
        )?;
    }
    let known: HashSet<String> = all.records.iter().map(event_id).collect();
    for id in proposal.evidence.iter().filter(|id| id.starts_with("e6-")) {
        anyhow::ensure!(
            known.contains(id),
            "cited raw evidence {id} is no longer present"
        );
    }
    Ok(format!(
        "{} cited evidence item(s) resolve and cover every candidate",
        proposal.evidence.len()
    ))
}

fn verify_citations(cfg: &Config, proposal: &Proposal) -> anyhow::Result<String> {
    if proposal.related_citations.is_empty() {
        return Ok("no related citation snapshots were claimed".to_string());
    }
    let native = paths::native_projects_root();
    let conn = index::ensure_fresh(
        &paths::state_dir(),
        &cfg.brain_root,
        Some(&native),
        &cfg.rings(),
    )?;
    for source in &proposal.related_citations {
        let blocks = index::expand(&conn, &source.cite)?;
        anyhow::ensure!(
            blocks.len() == 1,
            "related citation {} is stale or ambiguous",
            source.cite
        );
        anyhow::ensure!(
            hash_bytes(blocks[0].text.as_bytes()) == source.sha256,
            "related citation {} changed after proposal",
            source.cite
        );
    }
    Ok(format!(
        "{} related citation snapshot(s) are current",
        proposal.related_citations.len()
    ))
}

fn verify_target(cfg: &Config, proposal: &Proposal, applied: bool) -> anyhow::Result<String> {
    if !proposal.transition.changes_memory() {
        anyhow::ensure!(
            proposal.target.is_none() && proposal.before.is_none() && proposal.after.is_none(),
            "dismiss/noop proposal carries content"
        );
        anyhow::ensure!(
            proposal.authority == Authority::Unendorsed
                || proposal.authority == Authority::Attested
                || proposal.authority == Authority::Authorized,
            "invalid authority"
        );
        return Ok("decision-only transition cannot write memory".to_string());
    }
    let rel = safe_relative(proposal.target.as_deref().unwrap_or_default())?;
    let after = proposal
        .after
        .as_deref()
        .context("content transition has no after bytes")?;
    target_policy(&rel, after, proposal.authority, &cfg.rings())?;
    if let Some(shape) = secret_shape(after) {
        anyhow::bail!("proposed Markdown contains a {shape}");
    }
    let path = reject_symlink_path(&cfg.brain_root, &rel)?;
    let current = match std::fs::read_to_string(&path) {
        Ok(content) => Some(content),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
    };
    let expected = if applied {
        proposal.after.as_ref()
    } else {
        proposal.before.as_ref()
    };
    anyhow::ensure!(
        current.as_ref() == expected,
        "target {rel} changed since the proposal's {} snapshot",
        if applied { "after" } else { "before" }
    );
    anyhow::ensure!(
        proposal.before.as_ref().map(hash_bytes) == proposal.before_sha256,
        "stored before hash does not match stored bytes"
    );
    Ok(format!(
        "{rel} is symlink-free, resolves to ring {}, and matches the {} snapshot",
        effective_ring(&rel, after, &cfg.rings()),
        if applied { "applied" } else { "captured" }
    ))
}

fn verify_validity(proposal: &Proposal) -> anyhow::Result<String> {
    if let Some(until) = proposal.valid_until {
        anyhow::ensure!(
            until > now(),
            "proposal validity expired at Unix timestamp {until}"
        );
        Ok(format!(
            "proposal remains valid until Unix timestamp {until}"
        ))
    } else {
        Ok("proposal declares no time limit".to_string())
    }
}

fn verify_no_competing_apply(cfg: &Config, proposal: &Proposal) -> anyhow::Result<String> {
    let Some(target) = proposal.target.as_deref() else {
        return Ok("decision-only transition has no write target".to_string());
    };
    for summary in list(cfg)
        .into_iter()
        .filter(|summary| summary.state == APPLIED)
    {
        if summary.id != proposal.id && summary.target.as_deref() == Some(target) {
            anyhow::bail!(
                "another applied proposal {} owns target {target}",
                summary.id
            );
        }
    }
    Ok(format!("no other applied proposal owns {target}"))
}

fn whole_file_diff(proposal: &Proposal) -> Option<String> {
    if !proposal.transition.changes_memory() {
        return None;
    }
    let path = proposal.target.as_deref().unwrap_or("memory.md");
    let before = proposal.before.as_deref().unwrap_or("");
    let after = proposal.after.as_deref().unwrap_or("");
    let old_lines: Vec<&str> = before.lines().collect();
    let new_lines: Vec<&str> = after.lines().collect();
    let mut out = format!(
        "--- a/{path}\n+++ b/{path}\n@@ -1,{} +1,{} @@\n",
        old_lines.len(),
        new_lines.len()
    );
    for line in old_lines {
        out.push('-');
        out.push_str(line);
        out.push('\n');
    }
    for line in new_lines {
        out.push('+');
        out.push_str(line);
        out.push('\n');
    }
    Some(out)
}

fn proposal_sha256(proposal: &Proposal) -> anyhow::Result<String> {
    Ok(hash_bytes(serde_json::to_vec(proposal)?))
}

fn valid_review(cfg: &Config, proposal: &Proposal) -> anyhow::Result<Review> {
    let path = review_path(&cfg.brain_root, &proposal.id);
    let review = load_review_at(&path).with_context(|| {
        format!(
            "no independent semantic review for {}; run `cfetch maintain review {}`",
            proposal.id, proposal.id
        )
    })?;
    anyhow::ensure!(
        review.schema_version == SCHEMA_VERSION,
        "unsupported review schema {}",
        review.schema_version
    );
    anyhow::ensure!(
        review.proposal_id == proposal.id,
        "review names proposal {}, expected {}",
        review.proposal_id,
        proposal.id
    );
    anyhow::ensure!(
        review.proposal_sha256 == proposal_sha256(proposal)?,
        "review was made against different proposal bytes"
    );
    let mut identity = review.clone();
    identity.id.clear();
    identity.created_at = 0;
    let expected = format!(
        "review-{}",
        &hash_bytes(serde_json::to_vec(&identity)?)[..12]
    );
    anyhow::ensure!(
        review.id == expected,
        "review content address mismatch: expected {expected}"
    );
    anyhow::ensure!(
        review.verdict == ReviewVerdict::Pass,
        "semantic review failed: {}",
        review.notes
    );
    let checks = [
        ("evidence coverage", review.evidence_coverage),
        ("factual faithfulness", review.factual_faithfulness),
        ("preservation", review.preservation),
        ("authority fit", review.authority_fit),
        ("target fit", review.target_fit),
        ("contradiction check", review.contradiction_checked),
    ];
    let failed: Vec<&str> = checks
        .into_iter()
        .filter_map(|(name, ok)| (!ok).then_some(name))
        .collect();
    anyhow::ensure!(
        failed.is_empty(),
        "semantic review did not pass: {}",
        failed.join(", ")
    );
    Ok(review)
}

pub fn submit_review(
    cfg: &Config,
    proposal_id: &str,
    input: ReviewInput,
) -> anyhow::Result<(Review, bool)> {
    anyhow::ensure!(!input.notes.trim().is_empty(), "review notes are empty");
    anyhow::ensure!(input.notes.len() <= 16 * 1024, "review notes are too large");
    if let Some(shape) = secret_shape(&input.notes) {
        anyhow::bail!("review notes contain a {shape}");
    }
    let _lock = lock(cfg)?;
    let (_, _, proposal) = locate(&cfg.brain_root, proposal_id, &[PENDING])?;
    let deterministic = verify_proposal(cfg, &proposal, false, false);
    anyhow::ensure!(
        deterministic.valid,
        "proposal fails deterministic checks before semantic review: {}",
        deterministic
            .checks
            .iter()
            .filter(|check| !check.ok)
            .map(|check| format!("{}: {}", check.name, check.detail))
            .collect::<Vec<_>>()
            .join("; ")
    );
    let mut review = Review {
        schema_version: SCHEMA_VERSION,
        id: String::new(),
        proposal_id: proposal.id.clone(),
        proposal_sha256: proposal_sha256(&proposal)?,
        created_at: now(),
        created_by_host: paths::host_id(),
        verdict: input.verdict,
        method: input.method,
        evidence_coverage: input.evidence_coverage,
        factual_faithfulness: input.factual_faithfulness,
        preservation: input.preservation,
        authority_fit: input.authority_fit,
        target_fit: input.target_fit,
        contradiction_checked: input.contradiction_checked,
        notes: input.notes,
    };
    let mut identity = review.clone();
    identity.id.clear();
    identity.created_at = 0;
    review.id = format!(
        "review-{}",
        &hash_bytes(serde_json::to_vec(&identity)?)[..12]
    );
    let path = review_path(&cfg.brain_root, proposal_id);
    if path.exists() {
        let existing = load_review_at(&path)?;
        let mut existing_identity = existing.clone();
        existing_identity.created_at = 0;
        let mut review_identity = review.clone();
        review_identity.created_at = 0;
        anyhow::ensure!(
            existing_identity == review_identity,
            "proposal {proposal_id} already has an immutable review; revise the proposal to get a new id"
        );
        return Ok((existing, false));
    }
    fsutil::atomic_write(&path, render_review(&review))?;
    Ok((review, true))
}

fn approval_token(proposal: &Proposal, review: &Review) -> anyhow::Result<String> {
    let bytes = serde_json::to_vec(&(proposal, review))?;
    let mut hash = Sha256::new();
    hash.update(b"cfetch-maintenance-approval-v1\0");
    hash.update(bytes);
    let digest = crate::hashing::hex_lower(hash.finalize());
    Ok(format!("approve-{}", &digest[..20]))
}

fn verify_proposal(
    cfg: &Config,
    proposal: &Proposal,
    applied: bool,
    require_review: bool,
) -> Verification {
    let mut checks = Vec::new();
    push_check(&mut checks, "proposal_identity", verify_identity(proposal));
    if !applied {
        push_check(
            &mut checks,
            "candidate_integrity",
            verify_candidates(cfg, proposal),
        );
        push_check(
            &mut checks,
            "evidence_coverage",
            verify_evidence(cfg, proposal),
        );
        push_check(
            &mut checks,
            "citation_freshness",
            verify_citations(cfg, proposal),
        );
    }
    push_check(
        &mut checks,
        "target_boundary",
        verify_target(cfg, proposal, applied),
    );
    push_check(&mut checks, "validity", verify_validity(proposal));
    push_check(
        &mut checks,
        "single_writer",
        verify_no_competing_apply(cfg, proposal),
    );
    let review = if require_review {
        match valid_review(cfg, proposal) {
            Ok(review) => {
                checks.push(Check {
                    name: "semantic_review".to_string(),
                    ok: true,
                    detail: format!("{} passed via {:?}", review.id, review.method)
                        .to_ascii_lowercase(),
                });
                Some(review)
            }
            Err(error) => {
                checks.push(Check {
                    name: "semantic_review".to_string(),
                    ok: false,
                    detail: error.to_string(),
                });
                None
            }
        }
    } else {
        None
    };
    let valid = checks.iter().all(|check| check.ok);
    Verification {
        proposal_id: proposal.id.clone(),
        valid,
        checks,
        review_id: review.as_ref().map(|review| review.id.clone()),
        approval_token: if valid {
            review
                .as_ref()
                .and_then(|review| approval_token(proposal, review).ok())
        } else {
            None
        },
        diff: whole_file_diff(proposal),
    }
}

pub fn verify(cfg: &Config, id: &str) -> anyhow::Result<Verification> {
    let (_, _, proposal) = locate(&cfg.brain_root, id, &[PENDING])?;
    Ok(verify_proposal(cfg, &proposal, false, true))
}

/// Re-run the deterministic gates that still have meaning for an active
/// proposal without granting the caller any mutation authority. Pending
/// proposals are checked against their captured before bytes; applied ones
/// are checked against their exact after bytes. Terminal lifecycle records
/// remain inspectable, but are not misleadingly presented as currently
/// actionable.
pub fn inspect_verification(cfg: &Config, id: &str) -> anyhow::Result<Option<Verification>> {
    let (state, _, proposal) = locate(
        &cfg.brain_root,
        id,
        &[PENDING, APPLIED, FINALIZED, REJECTED, REVERTED],
    )?;
    match state.as_str() {
        PENDING => Ok(Some(verify_proposal(cfg, &proposal, false, true))),
        APPLIED => Ok(Some(verify_proposal(cfg, &proposal, true, true))),
        _ => Ok(None),
    }
}

/// Exact full-file diff captured by a proposal. Decision-only transitions
/// return no diff because they can never write trusted memory.
pub fn exact_diff(proposal: &Proposal) -> Option<String> {
    whole_file_diff(proposal)
}

fn move_proposal(brain_root: &Path, id: &str, from: &str, to: &str) -> anyhow::Result<()> {
    let source = proposal_path(brain_root, from, id);
    let target = proposal_path(brain_root, to, id);
    anyhow::ensure!(!target.exists(), "proposal {id} already exists in {to}");
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::rename(&source, &target)
        .with_context(|| format!("move proposal {id} from {from} to {to}"))
}

pub fn apply(cfg: &Config, id: &str, token: &str) -> anyhow::Result<Proposal> {
    let _lock = lock(cfg)?;
    let (_, _, proposal) = locate(&cfg.brain_root, id, &[PENDING])?;
    let verification = verify_proposal(cfg, &proposal, false, true);
    anyhow::ensure!(
        verification.valid,
        "proposal does not pass verification: {}",
        verification
            .checks
            .iter()
            .filter(|check| !check.ok)
            .map(|check| format!("{}: {}", check.name, check.detail))
            .collect::<Vec<_>>()
            .join("; ")
    );
    let expected = verification.approval_token.as_deref().unwrap_or_default();
    anyhow::ensure!(
        token == expected,
        "approval token does not match the current verified proposal; run `cfetch maintain verify {id}` again"
    );

    if proposal.transition.changes_memory() {
        let rel = proposal.target.as_deref().unwrap_or_default();
        let path = reject_symlink_path(&cfg.brain_root, rel)?;
        fsutil::atomic_write(&path, proposal.after.as_deref().unwrap_or_default())?;
        if let Err(error) = move_proposal(&cfg.brain_root, id, PENDING, APPLIED) {
            match proposal.before.as_deref() {
                Some(before) => fsutil::atomic_write(&path, before)?,
                None => {
                    let _ = std::fs::remove_file(&path);
                }
            }
            return Err(error.context("proposal state move failed; target was rolled back"));
        }
    } else {
        move_proposal(&cfg.brain_root, id, PENDING, APPLIED)?;
    }
    Ok(proposal)
}

fn frontmatter_pauses_maintenance(text: &str) -> bool {
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("---") {
        return false;
    }
    for line in lines {
        let line = line.trim();
        if line == "---" {
            break;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if matches!(key.trim(), "cfetch-maintenance" | "cfetch_maintenance")
            && matches!(
                value
                    .trim()
                    .trim_matches(['\'', '"'])
                    .to_ascii_lowercase()
                    .as_str(),
                "pause" | "paused" | "manual" | "off"
            )
        {
            return true;
        }
    }
    false
}

fn direct_user_evidence(cfg: &Config, proposal: &Proposal) -> anyhow::Result<bool> {
    let staging_dir = paths::staging_dir(&cfg.brain_root);
    let mut found = false;
    for source in &proposal.candidates {
        let candidate = find_candidate(&staging_dir, &source.id)?;
        let direct = matches!(
            candidate.reason.as_str(),
            "correction" | "direct-instruction"
        ) || candidate
            .payload
            .get("authority")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|authority| matches!(authority, "operator" | "user"));
        if !direct {
            return Ok(false);
        }
        found = true;
    }
    Ok(found)
}

fn verify_automatic_policy(cfg: &Config, proposal: &Proposal) -> anyhow::Result<()> {
    if let Some(before) = proposal.before.as_deref()
        && frontmatter_pauses_maintenance(before)
    {
        anyhow::bail!(
            "target is paused by frontmatter; edit or remove cfetch-maintenance before retrying"
        );
    }
    if let Some(after) = proposal.after.as_deref()
        && frontmatter_pauses_maintenance(after)
    {
        anyhow::bail!(
            "proposed target is paused by frontmatter and cannot be written automatically"
        );
    }
    if proposal.transition.changes_memory() {
        let rel = proposal.target.as_deref().unwrap_or_default();
        let after = proposal.after.as_deref().unwrap_or_default();
        if effective_ring(rel, after, &cfg.rings()) == 2 {
            anyhow::ensure!(
                direct_user_evidence(cfg, proposal)?,
                "automatic ring 2 maintenance requires direct user evidence; model-selected authorized authority is insufficient"
            );
        }
    }
    Ok(())
}

/// Apply a reviewed proposal through the normal autonomous path. There is no
/// human token and no git dependency: the independent semantic review and all
/// deterministic gates are re-run under the transaction lock immediately
/// before the exact bytes are written. The proposal record is retained for
/// exact reversion.
pub fn automatic_apply(cfg: &Config, id: &str) -> anyhow::Result<Proposal> {
    let _lock = lock(cfg)?;
    let (_, _, proposal) = locate(&cfg.brain_root, id, &[PENDING])?;
    let verification = verify_proposal(cfg, &proposal, false, true);
    anyhow::ensure!(
        verification.valid,
        "proposal does not pass verification: {}",
        verification
            .checks
            .iter()
            .filter(|check| !check.ok)
            .map(|check| format!("{}: {}", check.name, check.detail))
            .collect::<Vec<_>>()
            .join("; ")
    );
    verify_automatic_policy(cfg, &proposal)?;

    let mut written_path = None;
    if proposal.transition.changes_memory() {
        let path = reject_symlink_path(
            &cfg.brain_root,
            proposal.target.as_deref().unwrap_or_default(),
        )?;
        fsutil::atomic_write(&path, proposal.after.as_deref().unwrap_or_default())?;
        written_path = Some(path);
    }
    if let Err(error) = move_proposal(&cfg.brain_root, id, PENDING, FINALIZED) {
        if let Some(path) = written_path.as_deref() {
            match proposal.before.as_deref() {
                Some(before) => fsutil::atomic_write(path, before)?,
                None => {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
        return Err(error.context("automatic state move failed; target was rolled back"));
    }

    let outcome = match proposal.transition {
        Transition::Dismiss => EventOutcome::Dismissed,
        Transition::Noop => EventOutcome::Noop,
        _ => EventOutcome::Applied,
    };
    let review_id = verification.review_id.clone();
    if let Err(error) = write_event(
        cfg,
        event_for(
            Some(&proposal),
            outcome,
            review_id,
            verification.checks.clone(),
            "independent semantic review and deterministic gates passed; exact bytes applied automatically",
            Vec::new(),
        ),
    ) {
        if let Some(path) = written_path.as_deref() {
            match proposal.before.as_deref() {
                Some(before) => fsutil::atomic_write(path, before)?,
                None => {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
        if let Err(move_error) = move_proposal(&cfg.brain_root, id, FINALIZED, PENDING) {
            if let Some(path) = written_path.as_deref() {
                fsutil::atomic_write(path, proposal.after.as_deref().unwrap_or_default())?;
            }
            return Err(error.context(format!(
                "write maintenance history; rollback state also failed: {move_error}"
            )));
        }
        return Err(error.context("write maintenance history; automatic target was rolled back"));
    }

    // Finalized is the commit point. Candidate cleanup is replayable through
    // `finalize`; a transient log or filesystem problem must not make a
    // successfully applied edit look unapplied.
    let _ = settle_candidates(cfg, &proposal);
    Ok(proposal)
}

pub fn pause(cfg: &Config, reason: &str) -> anyhow::Result<()> {
    let reason = reason.trim();
    anyhow::ensure!(!reason.is_empty(), "pause reason is empty");
    anyhow::ensure!(reason.len() <= 4 * 1024, "pause reason is too large");
    if let Some(shape) = secret_shape(reason) {
        anyhow::bail!("pause reason contains a {shape}");
    }
    let _lock = lock(cfg)?;
    fsutil::atomic_write(
        &pause_path(&cfg.brain_root),
        format!(
            "---\nring: 5\ntype: cfetch-maintenance-pause\n---\n\n# Maintenance paused\n\n{reason}\n"
        ),
    )
}

pub fn pause_reason(cfg: &Config) -> Option<String> {
    let text = std::fs::read_to_string(pause_path(&cfg.brain_root)).ok()?;
    text.split_once("# Maintenance paused\n\n")
        .map(|(_, reason)| reason.trim().to_string())
        .filter(|reason| !reason.is_empty())
}

pub fn is_paused(cfg: &Config) -> bool {
    pause_path(&cfg.brain_root).is_file()
}

/// Content revision of the actionable candidate set. Background maintenance
/// uses this as an event signal: polling may be periodic, but model inference
/// happens only after the evidence set changes (or a bounded retry is due).
pub fn candidate_revision(cfg: &Config) -> String {
    let candidates = staging::list(&paths::staging_dir(&cfg.brain_root));
    hash_bytes(serde_json::to_vec(&candidates).unwrap_or_default())
}

pub fn resume(cfg: &Config) -> anyhow::Result<()> {
    let _lock = lock(cfg)?;
    match std::fs::remove_file(pause_path(&cfg.brain_root)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).context("remove maintenance pause marker"),
    }
}

fn record_exception(
    cfg: &Config,
    proposal: Option<&Proposal>,
    candidate_ids: Vec<String>,
    detail: impl Into<String>,
    checks: Vec<Check>,
) -> anyhow::Result<MaintenanceEvent> {
    write_event(
        cfg,
        event_for(
            proposal,
            EventOutcome::Exception,
            None,
            checks,
            detail,
            candidate_ids,
        ),
    )
}

/// Records a worker failure that happened before a proposal existed (for
/// example, endpoint policy or credential resolution). Candidate ids are
/// bounded and the ordinary journal redaction/content-addressing still apply.
pub(crate) fn record_background_exception(
    cfg: &Config,
    detail: impl Into<String>,
) -> anyhow::Result<MaintenanceEvent> {
    let candidate_ids = staging::list(&paths::staging_dir(&cfg.brain_root))
        .into_iter()
        .take(cfg.maintenance.max_candidates)
        .map(|candidate| candidate.id)
        .collect();
    record_exception(cfg, None, candidate_ids, detail, Vec::new())
}

/// Process a bounded batch and continue past per-candidate exceptions. A
/// healthy cycle needs no human input; failures remain visible in history and
/// their source candidates remain available for a later, freshly grounded
/// attempt.
pub fn run_once_with<M: MaintenanceModel>(
    cfg: &Config,
    model: &mut M,
    limit: usize,
) -> anyhow::Result<RunReport> {
    if pause_path(&cfg.brain_root).is_file() {
        return Ok(RunReport {
            paused: true,
            ..RunReport::default()
        });
    }
    let mut report = RunReport::default();
    let candidates = staging::list(&paths::staging_dir(&cfg.brain_root));
    for candidate in candidates.into_iter().take(limit.max(1)) {
        report.examined += 1;
        let mut submitted: Option<Proposal> = None;
        let result = (|| -> anyhow::Result<Proposal> {
            let packet = packet(cfg, &candidate.id)?;
            let input = model.propose(&packet)?;
            anyhow::ensure!(
                input.candidate_ids.len() == 1 && input.candidate_ids[0] == candidate.id,
                "autonomous proposal must name only packet candidate {}; got {:?}",
                candidate.id,
                input.candidate_ids
            );
            let proposal = submit(cfg, input)?.proposal;
            submitted = Some(proposal.clone());
            let review = model.review(&packet, &proposal)?;
            let (recorded, _) = submit_review(cfg, &proposal.id, review)?;
            anyhow::ensure!(
                recorded.verdict == ReviewVerdict::Pass,
                "independent semantic review failed: {}",
                recorded.notes
            );
            automatic_apply(cfg, &proposal.id)
        })();

        match result {
            Ok(proposal) => match proposal.transition {
                Transition::Dismiss => report.dismissed += 1,
                Transition::Noop => report.noops += 1,
                _ => report.applied += 1,
            },
            Err(error) => {
                report.exceptions += 1;
                let detail = error.to_string();
                if let Some(proposal) = submitted.as_ref() {
                    let verification = verify_proposal(cfg, proposal, false, true);
                    let _ = record_exception(
                        cfg,
                        Some(proposal),
                        Vec::new(),
                        detail,
                        verification.checks,
                    );
                    if proposal_path(&cfg.brain_root, PENDING, &proposal.id).is_file() {
                        let _ = reject(cfg, &proposal.id);
                    }
                } else {
                    let _ = record_exception(cfg, None, vec![candidate.id], detail, Vec::new());
                }
            }
        }
    }
    Ok(report)
}

pub fn reject(cfg: &Config, id: &str) -> anyhow::Result<()> {
    let _lock = lock(cfg)?;
    locate(&cfg.brain_root, id, &[PENDING])?;
    // Idempotent: the autonomous loop re-derives deterministic ids, and a
    // reject that finds its target already REJECTED (with the PENDING twin
    // still present) used to bail "already exists" - the swallowed error
    // wedged the twin in PENDING forever, retrying every cycle. An identical
    // terminal copy means the decision was already made: drop the twin.
    let rejected = proposal_path(&cfg.brain_root, REJECTED, id);
    if rejected.exists() {
        let terminal = proposal_path(&cfg.brain_root, PENDING, id);
        if terminal.exists() {
            std::fs::remove_file(&terminal)
                .with_context(|| format!("remove pending twin {}", terminal.display()))?;
        }
        return Ok(());
    }
    move_proposal(&cfg.brain_root, id, PENDING, REJECTED)
}

pub fn revert(cfg: &Config, id: &str) -> anyhow::Result<Proposal> {
    let _lock = lock(cfg)?;
    let (state, _, proposal) = locate(&cfg.brain_root, id, &[APPLIED, FINALIZED])?;
    let verification = verify_proposal(cfg, &proposal, true, true);
    anyhow::ensure!(
        verification.valid,
        "applied proposal no longer matches the tree: {}",
        verification
            .checks
            .iter()
            .filter(|check| !check.ok)
            .map(|check| format!("{}: {}", check.name, check.detail))
            .collect::<Vec<_>>()
            .join("; ")
    );
    if proposal.transition.changes_memory() {
        let path = reject_symlink_path(
            &cfg.brain_root,
            proposal.target.as_deref().unwrap_or_default(),
        )?;
        match proposal.before.as_deref() {
            Some(before) => fsutil::atomic_write(&path, before)?,
            None => {
                std::fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?
            }
        }
        if let Err(error) = move_proposal(&cfg.brain_root, id, &state, REVERTED) {
            fsutil::atomic_write(&path, proposal.after.as_deref().unwrap_or_default())?;
            return Err(error.context("proposal state move failed; applied target was restored"));
        }
    } else {
        move_proposal(&cfg.brain_root, id, &state, REVERTED)?;
    }
    if let Err(error) = write_event(
        cfg,
        event_for(
            Some(&proposal),
            EventOutcome::Reverted,
            verification.review_id,
            verification.checks,
            "exact automatic or manual result was still present and its captured before bytes were restored",
            Vec::new(),
        ),
    ) {
        if proposal.transition.changes_memory() {
            let path = reject_symlink_path(
                &cfg.brain_root,
                proposal.target.as_deref().unwrap_or_default(),
            )?;
            fsutil::atomic_write(&path, proposal.after.as_deref().unwrap_or_default())?;
        }
        if let Err(move_error) = move_proposal(&cfg.brain_root, id, REVERTED, &state) {
            return Err(error.context(format!(
                "write revert history; restoring proposal state also failed: {move_error}"
            )));
        }
        return Err(
            error.context("write revert history; reverted bytes and state were rolled back")
        );
    }
    Ok(proposal)
}

fn git_committed(brain_root: &Path, rel: &str, after: &str) -> anyhow::Result<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(brain_root)
        .output()
        .context("run git rev-parse")?;
    anyhow::ensure!(
        output.status.success(),
        "brain root is not inside a git worktree"
    );
    let top = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    let canonical_top = std::fs::canonicalize(&top)
        .with_context(|| format!("resolve git root {}", top.display()))?;
    let canonical_brain = std::fs::canonicalize(brain_root)
        .with_context(|| format!("resolve brain root {}", brain_root.display()))?;
    let prefix = canonical_brain
        .strip_prefix(&canonical_top)
        .context("brain root is outside its reported git worktree")?;
    let git_rel = if prefix.as_os_str().is_empty() {
        rel.to_string()
    } else {
        index::rel_doc_path(&prefix.join(rel))
    };

    let status = Command::new("git")
        .args(["status", "--porcelain=v1", "--", rel])
        .current_dir(&canonical_brain)
        .output()
        .context("run git status")?;
    anyhow::ensure!(status.status.success(), "git status failed");
    anyhow::ensure!(
        status.stdout.is_empty(),
        "target {rel} still has uncommitted changes"
    );

    let shown = Command::new("git")
        .args(["show", &format!("HEAD:{git_rel}")])
        .current_dir(&canonical_top)
        .output()
        .context("read target from git HEAD")?;
    anyhow::ensure!(
        shown.status.success(),
        "target {rel} is not present in git HEAD"
    );
    anyhow::ensure!(
        shown.stdout == after.as_bytes(),
        "git HEAD does not contain the proposal's exact after bytes for {rel}"
    );
    let head = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .current_dir(&canonical_top)
        .output()
        .context("read git HEAD")?;
    anyhow::ensure!(head.status.success(), "cannot read git HEAD");
    Ok(String::from_utf8_lossy(&head.stdout).trim().to_string())
}

fn settle_candidates(cfg: &Config, proposal: &Proposal) -> anyhow::Result<()> {
    let staging_dir = paths::staging_dir(&cfg.brain_root);
    let ex = exhaust::Exhaust::from_config(cfg);
    for source in &proposal.candidates {
        if !staging::path_of(&staging_dir, &source.id).is_file() {
            continue;
        }
        let action = if matches!(proposal.transition, Transition::Dismiss | Transition::Noop) {
            "dismiss"
        } else {
            "consume"
        };
        ex.record_decision(&source.id, action)?;
        if action == "dismiss" {
            let _ = staging::dismiss(&staging_dir, &source.id)?;
        } else {
            let _ = staging::consume(&staging_dir, &source.id)?;
        }
    }
    Ok(())
}

pub fn finalize(cfg: &Config, id: &str) -> anyhow::Result<FinalizeResult> {
    let _lock = lock(cfg)?;
    let already_finalized = proposal_path(&cfg.brain_root, FINALIZED, id).is_file();
    let (state, _, proposal) = locate(&cfg.brain_root, id, &[APPLIED, FINALIZED])?;
    if state == APPLIED {
        let verification = verify_proposal(cfg, &proposal, true, true);
        anyhow::ensure!(
            verification.valid,
            "applied proposal no longer matches the tree: {}",
            verification
                .checks
                .iter()
                .filter(|check| !check.ok)
                .map(|check| format!("{}: {}", check.name, check.detail))
                .collect::<Vec<_>>()
                .join("; ")
        );
        if proposal.transition.changes_memory() {
            git_committed(
                &cfg.brain_root,
                proposal.target.as_deref().unwrap_or_default(),
                proposal.after.as_deref().unwrap_or_default(),
            )?;
        }
        // The state move is the commit point. Candidate cleanup is replayable:
        // a crash after this line can safely rerun finalize on FINALIZED.
        move_proposal(&cfg.brain_root, id, APPLIED, FINALIZED)?;
    }
    settle_candidates(cfg, &proposal)?;
    Ok(FinalizeResult {
        proposal_id: proposal.id,
        candidate_ids: proposal
            .candidates
            .into_iter()
            .map(|source| source.id)
            .collect(),
        already_finalized,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Fixture {
        brain: tempfile::TempDir,
        cfg: Config,
        candidate: staging::Candidate,
        evidence_ids: Vec<String>,
    }

    impl Fixture {
        fn new() -> Self {
            let brain = tempfile::tempdir().unwrap();
            std::fs::create_dir_all(brain.path().join("knowledge")).unwrap();
            let cfg = Config {
                brain_root: brain.path().to_path_buf(),
                ..Config::default()
            };
            let candidate = staging::Candidate {
                id: staging::id_for("recurring-failure", "cargo test"),
                reason: "recurring-failure".to_string(),
                session: "session-b".to_string(),
                host: "test-host".to_string(),
                ts: 20,
                kind: "bash".to_string(),
                payload: json!({"norm": "cargo test", "command": "cargo test", "sessions": 2}),
            };
            staging::write(&paths::staging_dir(brain.path()), &candidate).unwrap();
            for (ts, session) in [(10, "session-a"), (20, "session-b")] {
                jsonl::append(
                    &paths::logs_dir(brain.path()),
                    exhaust::STREAM,
                    "test-host",
                    1 << 20,
                    json!({
                        "ts": ts,
                        "kind": "bash",
                        "session": session,
                        "payload": {"command": "cargo test", "norm": "cargo test", "failed": true}
                    }),
                )
                .unwrap();
            }
            let records = jsonl::read_all(&paths::logs_dir(brain.path()), exhaust::STREAM).records;
            let evidence_ids = records.iter().map(event_id).collect();
            Fixture {
                brain,
                cfg,
                candidate,
                evidence_ids,
            }
        }

        fn proposal(&self, target: &str, after: &str) -> ProposalInput {
            ProposalInput {
                candidate_ids: vec![self.candidate.id.clone()],
                transition: Transition::Add,
                target: Some(target.to_string()),
                after: Some(after.to_string()),
                authority: Authority::Attested,
                valid_until: None,
                rationale: "Two independent sessions observed the same failure.".to_string(),
                evidence: self.evidence_ids.clone(),
                related_citations: Vec::new(),
            }
        }

        fn pass_review(&self, proposal_id: &str) -> Review {
            submit_review(
                &self.cfg,
                proposal_id,
                ReviewInput {
                    verdict: ReviewVerdict::Pass,
                    method: ReviewMethod::IndependentAgent,
                    evidence_coverage: true,
                    factual_faithfulness: true,
                    preservation: true,
                    authority_fit: true,
                    target_fit: true,
                    contradiction_checked: true,
                    notes: "The proposed bytes are supported by the cited events and do not conflict with current memory.".to_string(),
                },
            )
            .unwrap()
            .0
        }
    }

    #[test]
    fn raw_evidence_is_content_addressed_and_covers_the_candidate() {
        let fixture = Fixture::new();
        let all = jsonl::read_all(&paths::logs_dir(fixture.brain.path()), exhaust::STREAM);
        let events = evidence_events(&fixture.candidate, &all.records);
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|event| event.id.starts_with("e6-")));
        assert_eq!(events[0].id, fixture.evidence_ids[0]);
        assert_ne!(events[0].id, events[1].id);
    }

    #[test]
    fn packet_combines_raw_evidence_with_current_cited_context() {
        let fixture = Fixture::new();
        std::fs::write(
            fixture.brain.path().join("knowledge/current.md"),
            "# Build notes\n\nThe cargo test failure is already documented here.\n",
        )
        .unwrap();
        let state = tempfile::tempdir().unwrap();
        let native = tempfile::tempdir().unwrap();
        let packet = packet_at(
            &fixture.cfg,
            &fixture.candidate.id,
            state.path(),
            native.path(),
        )
        .unwrap();
        assert_eq!(
            packet.candidate.evidence_id,
            candidate_evidence_id(&fixture.candidate)
        );
        assert_eq!(packet.events.len(), 2);
        assert!(
            packet
                .events
                .iter()
                .all(|event| event.id.starts_with("e6-"))
        );
        assert!(
            packet
                .relevant_statements
                .iter()
                .any(|statement| statement.path == "knowledge/current.md"
                    && statement.cite.starts_with("r3-")),
            "current statements are part of the review packet: {:?}",
            packet.relevant_statements
        );
        assert_eq!(
            packet.proposal_contract["transition"],
            "add | fold | supersede | revalidate | dismiss | noop"
        );
    }

    #[test]
    fn proposal_is_quarantined_verified_applied_and_reversible() {
        let fixture = Fixture::new();
        let submitted = submit(
            &fixture.cfg,
            fixture.proposal(
                "knowledge/failures.md",
                "---\nring: 3\n---\n\n# Known failure\n",
            ),
        )
        .unwrap();
        let pending = proposal_path(fixture.brain.path(), PENDING, &submitted.proposal.id);
        assert!(pending.is_file());
        assert!(pending.starts_with(paths::staging_dir(fixture.brain.path())));
        assert!(!fixture.brain.path().join("knowledge/failures.md").exists());

        let before_review = verify(&fixture.cfg, &submitted.proposal.id).unwrap();
        assert!(
            !before_review.valid,
            "deterministic checks alone cannot authorize promotion"
        );
        fixture.pass_review(&submitted.proposal.id);
        let verification = verify(&fixture.cfg, &submitted.proposal.id).unwrap();
        assert!(verification.valid, "{:?}", verification.checks);
        assert!(
            verification
                .diff
                .as_deref()
                .unwrap()
                .contains("+# Known failure")
        );
        apply(
            &fixture.cfg,
            &submitted.proposal.id,
            verification.approval_token.as_deref().unwrap(),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(fixture.brain.path().join("knowledge/failures.md")).unwrap(),
            "---\nring: 3\n---\n\n# Known failure\n"
        );
        assert!(proposal_path(fixture.brain.path(), APPLIED, &submitted.proposal.id).is_file());
        let applied = inspect_verification(&fixture.cfg, &submitted.proposal.id)
            .unwrap()
            .expect("an applied proposal still has live target-boundary checks");
        assert!(applied.valid, "{:?}", applied.checks);
        revert(&fixture.cfg, &submitted.proposal.id).unwrap();
        assert!(!fixture.brain.path().join("knowledge/failures.md").exists());
        assert!(proposal_path(fixture.brain.path(), REVERTED, &submitted.proposal.id).is_file());
        assert!(
            inspect_verification(&fixture.cfg, &submitted.proposal.id)
                .unwrap()
                .is_none(),
            "terminal lifecycle records remain inspectable without looking actionable"
        );
        assert!(
            staging::path_of(
                &paths::staging_dir(fixture.brain.path()),
                &fixture.candidate.id
            )
            .is_file()
        );
    }

    #[test]
    fn stale_target_invalidates_the_approval_token() {
        let fixture = Fixture::new();
        let target = fixture.brain.path().join("knowledge/existing.md");
        std::fs::write(&target, "old\n").unwrap();
        let mut input = fixture.proposal("knowledge/existing.md", "new\n");
        input.transition = Transition::Fold;
        let proposal = submit(&fixture.cfg, input).unwrap().proposal;
        fixture.pass_review(&proposal.id);
        let token = verify(&fixture.cfg, &proposal.id)
            .unwrap()
            .approval_token
            .unwrap();
        std::fs::write(&target, "concurrent edit\n").unwrap();
        let report = verify(&fixture.cfg, &proposal.id).unwrap();
        assert!(!report.valid);
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.name == "target_boundary" && !check.ok)
        );
        assert!(apply(&fixture.cfg, &proposal.id, &token).is_err());
        assert_eq!(
            std::fs::read_to_string(target).unwrap(),
            "concurrent edit\n"
        );
    }

    #[test]
    fn failed_semantic_review_is_immutable_and_blocks_approval() {
        let fixture = Fixture::new();
        let proposal = submit(
            &fixture.cfg,
            fixture.proposal("knowledge/unsupported.md", "# Unsupported claim\n"),
        )
        .unwrap()
        .proposal;
        let failed = ReviewInput {
            verdict: ReviewVerdict::Fail,
            method: ReviewMethod::IndependentAgent,
            evidence_coverage: true,
            factual_faithfulness: false,
            preservation: true,
            authority_fit: true,
            target_fit: true,
            contradiction_checked: true,
            notes: "The evidence shows a failure, but not the causal claim in the proposed text."
                .to_string(),
        };
        submit_review(&fixture.cfg, &proposal.id, failed).unwrap();
        let report = verify(&fixture.cfg, &proposal.id).unwrap();
        assert!(!report.valid);
        assert!(report.approval_token.is_none());
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.name == "semantic_review" && !check.ok)
        );

        let replacement = ReviewInput {
            verdict: ReviewVerdict::Pass,
            method: ReviewMethod::Human,
            evidence_coverage: true,
            factual_faithfulness: true,
            preservation: true,
            authority_fit: true,
            target_fit: true,
            contradiction_checked: true,
            notes: "Override attempt.".to_string(),
        };
        assert!(
            submit_review(&fixture.cfg, &proposal.id, replacement)
                .unwrap_err()
                .to_string()
                .contains("immutable review")
        );
    }

    #[test]
    fn authority_and_ring_are_independent_gates() {
        let fixture = Fixture::new();
        let mut ring_two = fixture.proposal("knowledge/behaviours/rule.md", "# Rule\n");
        assert!(
            submit(&fixture.cfg, ring_two.clone())
                .unwrap_err()
                .to_string()
                .contains("ring 2 requires authorized")
        );
        ring_two.authority = Authority::Authorized;
        assert!(submit(&fixture.cfg, ring_two).is_ok());

        let ring_one = fixture.proposal("AGENT.md", "# Policy\n");
        assert!(
            submit(&fixture.cfg, ring_one)
                .unwrap_err()
                .to_string()
                .contains("only rings 2-4")
        );

        let mut unendorsed = fixture.proposal("knowledge/inference.md", "# Guess\n");
        unendorsed.authority = Authority::Unendorsed;
        assert!(
            submit(&fixture.cfg, unendorsed)
                .unwrap_err()
                .to_string()
                .contains("unendorsed")
        );
    }

    #[test]
    fn traversal_secrets_and_uncovered_evidence_are_refused_before_storage() {
        let fixture = Fixture::new();
        let traversal = fixture.proposal("../outside.md", "no\n");
        assert!(submit(&fixture.cfg, traversal).is_err());

        let secret = fixture.proposal(
            "knowledge/token.md",
            "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz\n",
        );
        assert!(
            submit(&fixture.cfg, secret)
                .unwrap_err()
                .to_string()
                .contains("contains a")
        );

        let mut uncovered = fixture.proposal("knowledge/no-evidence.md", "# Claim\n");
        uncovered.evidence.clear();
        assert!(
            submit(&fixture.cfg, uncovered)
                .unwrap_err()
                .to_string()
                .contains("at least two sessions")
        );

        let mut partial = fixture.proposal("knowledge/partial-evidence.md", "# Claim\n");
        partial.evidence.truncate(1);
        assert!(
            submit(&fixture.cfg, partial)
                .unwrap_err()
                .to_string()
                .contains("at least two sessions")
        );
        assert_eq!(pending_count(&fixture.cfg), 0);
    }

    fn git(brain: &Path, args: &[&str]) {
        let status = Command::new("git")
            .args(args)
            .current_dir(brain)
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?}");
    }

    #[test]
    fn finalization_waits_for_the_exact_git_commit_then_consumes_evidence() {
        let fixture = Fixture::new();
        git(fixture.brain.path(), &["init", "-q"]);
        git(
            fixture.brain.path(),
            &["config", "user.email", "test@example.invalid"],
        );
        git(
            fixture.brain.path(),
            &["config", "user.name", "cfetch test"],
        );
        std::fs::write(fixture.brain.path().join("README.md"), "# Brain\n").unwrap();
        git(fixture.brain.path(), &["add", "README.md"]);
        git(fixture.brain.path(), &["commit", "-qm", "initial"]);

        let proposal = submit(
            &fixture.cfg,
            fixture.proposal("knowledge/learned.md", "---\nring: 3\n---\n\n# Learned\n"),
        )
        .unwrap()
        .proposal;
        fixture.pass_review(&proposal.id);
        let token = verify(&fixture.cfg, &proposal.id)
            .unwrap()
            .approval_token
            .unwrap();
        apply(&fixture.cfg, &proposal.id, &token).unwrap();
        assert!(
            finalize(&fixture.cfg, &proposal.id)
                .unwrap_err()
                .to_string()
                .contains("uncommitted")
        );
        assert!(
            staging::path_of(
                &paths::staging_dir(fixture.brain.path()),
                &fixture.candidate.id
            )
            .is_file()
        );

        git(fixture.brain.path(), &["add", "knowledge/learned.md"]);
        git(
            fixture.brain.path(),
            &["commit", "-qm", "record learned failure"],
        );
        let done = finalize(&fixture.cfg, &proposal.id).unwrap();
        assert!(!done.already_finalized);
        assert!(
            !staging::path_of(
                &paths::staging_dir(fixture.brain.path()),
                &fixture.candidate.id
            )
            .exists()
        );
        assert!(proposal_path(fixture.brain.path(), FINALIZED, &proposal.id).is_file());
        let replay = finalize(&fixture.cfg, &proposal.id).unwrap();
        assert!(replay.already_finalized, "finalize cleanup is replay-safe");
    }

    #[test]
    fn decision_transition_never_writes_memory_and_preserves_dismissed_evidence() {
        let fixture = Fixture::new();
        let input = ProposalInput {
            candidate_ids: vec![fixture.candidate.id.clone()],
            transition: Transition::Dismiss,
            target: None,
            after: None,
            authority: Authority::Unendorsed,
            valid_until: None,
            rationale: "The captured failure is environmental, not durable knowledge.".to_string(),
            evidence: fixture.evidence_ids.clone(),
            related_citations: Vec::new(),
        };
        let proposal = submit(&fixture.cfg, input).unwrap().proposal;
        fixture.pass_review(&proposal.id);
        let token = verify(&fixture.cfg, &proposal.id)
            .unwrap()
            .approval_token
            .unwrap();
        apply(&fixture.cfg, &proposal.id, &token).unwrap();
        finalize(&fixture.cfg, &proposal.id).unwrap();
        assert!(
            staging::dismissed_path(
                &paths::staging_dir(fixture.brain.path()),
                &fixture.candidate.id
            )
            .is_file()
        );
    }

    #[derive(Clone)]
    struct FakeModel {
        proposal: ProposalInput,
        review: ReviewInput,
        edit_before_review: Option<(PathBuf, String)>,
    }

    impl MaintenanceModel for FakeModel {
        fn propose(&mut self, _packet: &EvidencePacket) -> anyhow::Result<ProposalInput> {
            Ok(self.proposal.clone())
        }

        fn review(
            &mut self,
            _packet: &EvidencePacket,
            _proposal: &Proposal,
        ) -> anyhow::Result<ReviewInput> {
            if let Some((path, bytes)) = self.edit_before_review.take() {
                std::fs::write(path, bytes)?;
            }
            Ok(self.review.clone())
        }
    }

    fn passing_review() -> ReviewInput {
        ReviewInput {
            verdict: ReviewVerdict::Pass,
            method: ReviewMethod::IndependentAgent,
            evidence_coverage: true,
            factual_faithfulness: true,
            preservation: true,
            authority_fit: true,
            target_fit: true,
            contradiction_checked: true,
            notes: "Independent review found the proposed bytes faithful to the evidence.".into(),
        }
    }

    #[test]
    fn automatic_cycle_applies_reviews_finalizes_and_journals_without_human_token_or_git() {
        let fixture = Fixture::new();
        let mut model = FakeModel {
            proposal: fixture.proposal(
                "knowledge/automatic.md",
                "---\nring: 3\n---\n\n# Automatically maintained\n",
            ),
            review: passing_review(),
            edit_before_review: None,
        };

        let report = run_once_with(&fixture.cfg, &mut model, 8).unwrap();

        assert_eq!(report.examined, 1);
        assert_eq!(report.applied, 1);
        assert_eq!(report.exceptions, 0);
        assert_eq!(
            std::fs::read_to_string(fixture.brain.path().join("knowledge/automatic.md")).unwrap(),
            "---\nring: 3\n---\n\n# Automatically maintained\n"
        );
        assert_eq!(pending_count(&fixture.cfg), 0);
        assert_eq!(history(&fixture.cfg).len(), 1);
        let event = &history(&fixture.cfg)[0];
        assert_eq!(event.outcome, EventOutcome::Applied);
        assert_eq!(event.target.as_deref(), Some("knowledge/automatic.md"));
        let expected_hash = hash_bytes("---\nring: 3\n---\n\n# Automatically maintained\n");
        assert_eq!(event.after_sha256.as_deref(), Some(expected_hash.as_str()));
        assert!(event.review_id.is_some());
        assert!(event.checks.iter().all(|check| check.ok));
        assert!(
            !staging::path_of(
                &paths::staging_dir(fixture.brain.path()),
                &fixture.candidate.id
            )
            .exists()
        );
    }

    #[test]
    fn autonomous_review_cannot_cover_a_candidate_absent_from_its_packet() {
        let fixture = Fixture::new();
        let mut proposal = fixture.proposal("knowledge/wrong-packet.md", "# Wrong packet\n");
        proposal.candidate_ids = vec!["another-pending-candidate".into()];
        let mut model = FakeModel {
            proposal,
            review: passing_review(),
            edit_before_review: None,
        };

        let report = run_once_with(&fixture.cfg, &mut model, 8).unwrap();

        assert_eq!(report.examined, 1);
        assert_eq!(report.exceptions, 1);
        assert!(
            !fixture
                .brain
                .path()
                .join("knowledge/wrong-packet.md")
                .exists()
        );
        assert!(
            staging::path_of(
                &paths::staging_dir(fixture.brain.path()),
                &fixture.candidate.id
            )
            .is_file()
        );
        assert!(
            history(&fixture.cfg)[0]
                .detail
                .contains("must name only packet candidate")
        );
    }

    #[test]
    fn external_obsidian_edit_wins_and_becomes_a_visible_exception() {
        let fixture = Fixture::new();
        let target = fixture.brain.path().join("knowledge/existing.md");
        std::fs::write(&target, "old\n").unwrap();
        let mut proposal = fixture.proposal("knowledge/existing.md", "automatic\n");
        proposal.transition = Transition::Fold;
        let mut model = FakeModel {
            proposal,
            review: passing_review(),
            edit_before_review: Some((target.clone(), "edited in obsidian\n".into())),
        };

        let report = run_once_with(&fixture.cfg, &mut model, 8).unwrap();

        assert_eq!(report.applied, 0);
        assert_eq!(report.exceptions, 1);
        assert_eq!(
            std::fs::read_to_string(target).unwrap(),
            "edited in obsidian\n"
        );
        let events = history(&fixture.cfg);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].outcome, EventOutcome::Exception);
        assert!(
            events[0].detail.contains("target_boundary"),
            "{}",
            events[0].detail
        );
        assert!(
            staging::path_of(
                &paths::staging_dir(fixture.brain.path()),
                &fixture.candidate.id
            )
            .is_file()
        );
    }

    #[test]
    fn automatic_cycle_skips_a_paused_target() {
        let fixture = Fixture::new();
        let target = fixture.brain.path().join("knowledge/paused.md");
        std::fs::write(
            &target,
            "---\nring: 3\ncfetch-maintenance: paused\n---\n\n# Hands off\n",
        )
        .unwrap();
        let mut proposal = fixture.proposal(
            "knowledge/paused.md",
            "---\nring: 3\ncfetch-maintenance: paused\n---\n\n# Changed\n",
        );
        proposal.transition = Transition::Fold;
        let mut model = FakeModel {
            proposal,
            review: passing_review(),
            edit_before_review: None,
        };

        let report = run_once_with(&fixture.cfg, &mut model, 8).unwrap();

        assert_eq!(report.applied, 0);
        assert_eq!(report.exceptions, 1);
        assert_eq!(
            std::fs::read_to_string(target).unwrap(),
            "---\nring: 3\ncfetch-maintenance: paused\n---\n\n# Hands off\n"
        );
        assert!(
            history(&fixture.cfg)[0]
                .detail
                .contains("paused by frontmatter")
        );
    }

    #[test]
    fn global_pause_prevents_model_calls_and_is_readable_state() {
        let fixture = Fixture::new();
        pause(&fixture.cfg, "debugging an unexpected edit").unwrap();
        let mut model = FakeModel {
            proposal: fixture.proposal("knowledge/never.md", "# Never\n"),
            review: passing_review(),
            edit_before_review: None,
        };

        let report = run_once_with(&fixture.cfg, &mut model, 8).unwrap();

        assert!(report.paused);
        assert_eq!(report.examined, 0);
        assert_eq!(
            pause_reason(&fixture.cfg).as_deref(),
            Some("debugging an unexpected edit")
        );
        resume(&fixture.cfg).unwrap();
        assert!(pause_reason(&fixture.cfg).is_none());
    }

    #[test]
    fn automatic_ring_two_requires_direct_user_evidence_while_ring_four_is_allowed() {
        let fixture = Fixture::new();
        let mut ring_two = fixture.proposal("knowledge/behaviours/automatic.md", "# Behavior\n");
        ring_two.authority = Authority::Authorized;
        let proposal = submit(&fixture.cfg, ring_two).unwrap().proposal;
        fixture.pass_review(&proposal.id);
        assert!(
            automatic_apply(&fixture.cfg, &proposal.id)
                .unwrap_err()
                .to_string()
                .contains("direct user evidence")
        );

        let fixture = Fixture::new();
        let ring_four =
            fixture.proposal("todo/current.md", "---\nring: 4\n---\n\n# Current work\n");
        assert!(submit(&fixture.cfg, ring_four).is_ok());
    }

    #[test]
    fn finalized_automatic_change_is_reversible_only_while_its_bytes_still_match() {
        let fixture = Fixture::new();
        let proposal = submit(
            &fixture.cfg,
            fixture.proposal("knowledge/reversible.md", "# Maintained\n"),
        )
        .unwrap()
        .proposal;
        fixture.pass_review(&proposal.id);
        automatic_apply(&fixture.cfg, &proposal.id).unwrap();
        revert(&fixture.cfg, &proposal.id).unwrap();
        assert!(
            !fixture
                .brain
                .path()
                .join("knowledge/reversible.md")
                .exists()
        );

        let fixture = Fixture::new();
        let proposal = submit(
            &fixture.cfg,
            fixture.proposal("knowledge/reversible.md", "# Maintained\n"),
        )
        .unwrap()
        .proposal;
        fixture.pass_review(&proposal.id);
        automatic_apply(&fixture.cfg, &proposal.id).unwrap();
        std::fs::write(
            fixture.brain.path().join("knowledge/reversible.md"),
            "# Human edit\n",
        )
        .unwrap();
        assert!(revert(&fixture.cfg, &proposal.id).is_err());
        assert_eq!(
            std::fs::read_to_string(fixture.brain.path().join("knowledge/reversible.md")).unwrap(),
            "# Human edit\n"
        );
    }

    #[test]
    fn journal_details_are_secret_safe_and_bounded() {
        let redacted = journal_text("endpoint returned sk-123456789012345678901234567890");
        assert!(redacted.contains("redacted"), "{redacted}");
        assert!(!redacted.contains("sk-"), "{redacted}");

        let large = "é".repeat(MAX_JOURNAL_TEXT_BYTES);
        let bounded = journal_text(&large);
        assert!(bounded.len() <= MAX_JOURNAL_TEXT_BYTES + 64);
        assert!(bounded.ends_with("[journal detail truncated]"));
    }

    #[test]
    fn edited_history_is_rejected_and_reported_for_doctor() {
        let fixture = Fixture::new();
        let event = write_event(
            &fixture.cfg,
            event_for(
                None,
                EventOutcome::Exception,
                None,
                Vec::new(),
                "original exception",
                vec![fixture.candidate.id.clone()],
            ),
        )
        .unwrap();
        let path = history_path(fixture.brain.path(), &event.id);
        let edited = std::fs::read_to_string(&path)
            .unwrap()
            .replace("original exception", "edited exception");
        std::fs::write(path, edited).unwrap();

        assert!(history(&fixture.cfg).is_empty());
        let issues = history_issues(&fixture.cfg);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].contains("content address mismatch"), "{issues:?}");
    }

    #[test]
    fn preflight_failure_is_visible_without_a_proposal() {
        let fixture = Fixture::new();

        let event = record_background_exception(
            &fixture.cfg,
            "maintenance.api_key=sk-123456789012345678901234567890",
        )
        .unwrap();

        assert_eq!(event.outcome, EventOutcome::Exception);
        assert!(event.proposal_id.is_none());
        assert_eq!(event.candidate_ids, vec![fixture.candidate.id.clone()]);
        assert!(event.detail.contains("redacted"));
        assert!(!event.detail.contains("sk-"));
    }
}
