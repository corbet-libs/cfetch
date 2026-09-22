//! Read-only view model for the dashboard's maintenance activity timeline.
//!
//! Automatic outcomes and exceptions are the primary story. Candidates and
//! proposals remain visible so a person can inspect or debug the engine, but
//! normal maintenance does not wait for interaction in this screen.

use crate::config::Config;
use crate::{maintenance, paths, staging};

const MAX_PREVIEW_BYTES: usize = 128 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    Normal,
    Muted,
    Good,
    Warning,
    Error,
    Accent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetailLine {
    pub text: String,
    pub tone: Tone,
}

impl DetailLine {
    fn new(text: impl Into<String>, tone: Tone) -> Self {
        Self {
            text: text.into(),
            tone,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetailDocument {
    pub title: String,
    pub lines: Vec<DetailLine>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InboxRow {
    pub id: String,
    pub badge: String,
    pub summary: String,
    pub tone: Tone,
}

#[derive(Debug, Clone)]
enum Record {
    Event(maintenance::MaintenanceEvent),
    Candidate(staging::Candidate),
    Proposal(maintenance::ProposalSummary),
}

impl Record {
    fn id(&self) -> &str {
        match self {
            Self::Event(event) => &event.id,
            Self::Candidate(candidate) => &candidate.id,
            Self::Proposal(proposal) => &proposal.id,
        }
    }

    fn timestamp(&self) -> i64 {
        match self {
            Self::Event(event) => event.created_at,
            Self::Candidate(candidate) => candidate.ts,
            Self::Proposal(proposal) => proposal.created_at,
        }
    }

    fn row(&self) -> InboxRow {
        match self {
            Self::Event(event) => InboxRow {
                id: event.id.clone(),
                badge: enum_name(event.outcome),
                summary: event
                    .target
                    .as_deref()
                    .map(|target| format!("{} · {target}", event.detail))
                    .unwrap_or_else(|| event.detail.clone()),
                tone: outcome_tone(event.outcome),
            },
            Self::Candidate(candidate) => InboxRow {
                id: candidate.id.clone(),
                badge: "candidate".to_string(),
                summary: format!("{} · {}", candidate.reason, candidate.kind),
                tone: Tone::Warning,
            },
            Self::Proposal(proposal) => InboxRow {
                id: proposal.id.clone(),
                badge: proposal.state.clone(),
                summary: format!(
                    "{} · {}",
                    enum_name(proposal.transition),
                    proposal.target.as_deref().unwrap_or("no memory write")
                ),
                tone: state_tone(&proposal.state),
            },
        }
    }
}

#[derive(Debug)]
pub struct Inbox {
    records: Vec<Record>,
    selected: usize,
    pub detail_scroll: u16,
    pub detail: DetailDocument,
    pub status: String,
}

impl Inbox {
    pub fn load(cfg: &Config, can_verify_locally: bool) -> Self {
        let mut inbox = Self {
            records: Vec::new(),
            selected: 0,
            detail_scroll: 0,
            detail: empty_detail(),
            status: String::new(),
        };
        inbox.refresh(cfg, can_verify_locally);
        inbox
    }

    pub fn refresh(&mut self, cfg: &Config, can_verify_locally: bool) {
        let selected_id = self
            .records
            .get(self.selected)
            .map(Record::id)
            .map(str::to_string);
        let candidate_dir = paths::staging_dir(&cfg.brain_root);
        let mut records: Vec<Record> = maintenance::history(cfg)
            .into_iter()
            .map(Record::Event)
            .chain(staging::list(&candidate_dir).into_iter().map(Record::Candidate))
            .chain(maintenance::list(cfg).into_iter().map(Record::Proposal))
            .collect();
        records.sort_by(|a, b| {
            b.timestamp()
                .cmp(&a.timestamp())
                .then_with(|| a.id().cmp(b.id()))
        });
        self.records = records;
        self.selected = selected_id
            .as_deref()
            .and_then(|id| self.records.iter().position(|record| record.id() == id))
            .unwrap_or(0)
            .min(self.records.len().saturating_sub(1));
        self.detail_scroll = 0;
        let events = self
            .records
            .iter()
            .filter(|record| matches!(record, Record::Event(_)))
            .count();
        let candidates = self
            .records
            .iter()
            .filter(|record| matches!(record, Record::Candidate(_)))
            .count();
        let mode = if !cfg.maintenance.enabled {
            "off"
        } else if maintenance::is_paused(cfg) {
            "paused"
        } else if cfg.maintenance.configured() {
            match crate::runtime_status::endpoint_route(&cfg.maintenance.endpoint) {
                crate::runtime_status::InferenceRoute::Local => "automatic · local model",
                crate::runtime_status::InferenceRoute::Remote => "automatic · remote model",
            }
        } else {
            "setup needed"
        };
        self.status = format!(
            "{events} event(s) · {candidates} staged · {mode}"
        );
        self.load_selected(cfg, can_verify_locally);
    }

    pub fn rows(&self) -> Vec<InboxRow> {
        self.records.iter().map(Record::row).collect()
    }

    pub fn selected(&self) -> Option<usize> {
        (!self.records.is_empty()).then_some(self.selected)
    }

    pub fn select_next(&mut self, cfg: &Config, can_verify_locally: bool) {
        if self.selected + 1 < self.records.len() {
            self.selected += 1;
            self.detail_scroll = 0;
            self.load_selected(cfg, can_verify_locally);
        }
    }

    pub fn select_previous(&mut self, cfg: &Config, can_verify_locally: bool) {
        if self.selected > 0 {
            self.selected -= 1;
            self.detail_scroll = 0;
            self.load_selected(cfg, can_verify_locally);
        }
    }

    pub fn select_first(&mut self, cfg: &Config, can_verify_locally: bool) {
        if !self.records.is_empty() {
            self.selected = 0;
            self.detail_scroll = 0;
            self.load_selected(cfg, can_verify_locally);
        }
    }

    pub fn select_last(&mut self, cfg: &Config, can_verify_locally: bool) {
        if !self.records.is_empty() {
            self.selected = self.records.len() - 1;
            self.detail_scroll = 0;
            self.load_selected(cfg, can_verify_locally);
        }
    }

    pub fn scroll_down(&mut self, lines: u16) {
        self.detail_scroll = self.detail_scroll.saturating_add(lines);
    }

    pub fn scroll_up(&mut self, lines: u16) {
        self.detail_scroll = self.detail_scroll.saturating_sub(lines);
    }

    fn load_selected(&mut self, cfg: &Config, can_verify_locally: bool) {
        self.detail = match self.records.get(self.selected) {
            Some(Record::Event(event)) => event_detail(event),
            Some(Record::Candidate(candidate)) => candidate_detail(candidate),
            Some(Record::Proposal(summary)) => proposal_detail(cfg, summary, can_verify_locally)
                .unwrap_or_else(|error| {
                    error_detail(
                        &summary.id,
                        format!("could not inspect proposal: {error:#}"),
                    )
                }),
            None => empty_detail(),
        };
    }
}

fn enum_name(value: impl std::fmt::Debug) -> String {
    format!("{value:?}").to_ascii_lowercase()
}

fn state_tone(state: &str) -> Tone {
    match state {
        "pending" | "applied" => Tone::Warning,
        "finalized" => Tone::Good,
        "rejected" | "reverted" => Tone::Muted,
        _ => Tone::Error,
    }
}

fn outcome_tone(outcome: maintenance::EventOutcome) -> Tone {
    match outcome {
        maintenance::EventOutcome::Applied => Tone::Good,
        maintenance::EventOutcome::Dismissed
        | maintenance::EventOutcome::Noop
        | maintenance::EventOutcome::Reverted => Tone::Muted,
        maintenance::EventOutcome::Exception => Tone::Error,
    }
}

fn empty_detail() -> DetailDocument {
    DetailDocument {
        title: " maintenance activity ".to_string(),
        lines: vec![
            DetailLine::new(
                "No maintenance activity has been recorded yet.",
                Tone::Muted,
            ),
            DetailLine::new(
                "Captured evidence, automatic changes, and exceptions will appear here.",
                Tone::Muted,
            ),
        ],
    }
}

fn error_detail(id: &str, error: String) -> DetailDocument {
    DetailDocument {
        title: format!(" {id} "),
        lines: vec![DetailLine::new(error, Tone::Error)],
    }
}

fn heading(lines: &mut Vec<DetailLine>, text: &str) {
    if !lines.is_empty() {
        lines.push(DetailLine::new("", Tone::Normal));
    }
    lines.push(DetailLine::new(text, Tone::Accent));
}

fn fields(lines: &mut Vec<DetailLine>, values: impl IntoIterator<Item = (String, String)>) {
    lines.extend(
        values
            .into_iter()
            .map(|(name, value)| DetailLine::new(format!("{name}: {value}"), Tone::Normal)),
    );
}

fn lifecycle(state: &str) -> String {
    match state {
        "candidate" => "[CAPTURED] → propose → review → apply / settle".to_string(),
        "pending" => "captured → [PROPOSED] → review → apply / settle".to_string(),
        "applied" => "captured → proposed → reviewed → [LEGACY APPLY]".to_string(),
        "finalized" => "captured → proposed → reviewed → [SETTLED]".to_string(),
        "rejected" => "captured → proposed → [DISMISSED]".to_string(),
        "reverted" => "captured → proposed → applied → [REVERTED]".to_string(),
        other => format!("candidate → [{other}]"),
    }
}

fn event_detail(event: &maintenance::MaintenanceEvent) -> DetailDocument {
    let mut lines = Vec::new();
    let outcome = enum_name(event.outcome);
    lines.push(DetailLine::new(
        format!("AUTONOMOUS OUTCOME: {}", outcome.to_ascii_uppercase()),
        outcome_tone(event.outcome),
    ));
    heading(&mut lines, "Transaction");
    fields(
        &mut lines,
        [
            ("Outcome".to_string(), outcome.clone()),
            (
                "Target".to_string(),
                event
                    .target
                    .clone()
                    .unwrap_or_else(|| "no Markdown write".to_string()),
            ),
            (
                "Recorded".to_string(),
                format!("Unix timestamp {}", event.created_at),
            ),
            ("Origin".to_string(), event.created_by_host.clone()),
        ],
    );
    lines.push(DetailLine::new(event.detail.clone(), Tone::Normal));

    heading(&mut lines, "Evidence and provenance");
    if event.candidate_ids.is_empty() {
        lines.push(DetailLine::new("No candidate ids recorded.", Tone::Muted));
    } else {
        lines.extend(event.candidate_ids.iter().map(|id| {
            DetailLine::new(format!("• candidate {id}"), Tone::Normal)
        }));
    }
    if let Some(proposal) = &event.proposal_id {
        lines.push(DetailLine::new(format!("• proposal {proposal}"), Tone::Normal));
    }
    if let Some(review) = &event.review_id {
        lines.push(DetailLine::new(format!("• review {review}"), Tone::Normal));
    }
    if let Some(before) = &event.before_sha256 {
        lines.push(DetailLine::new(
            format!("• before sha256:{}", short_hash(before)),
            Tone::Muted,
        ));
    }
    if let Some(after) = &event.after_sha256 {
        lines.push(DetailLine::new(
            format!("• after  sha256:{}", short_hash(after)),
            Tone::Muted,
        ));
    }

    heading(&mut lines, "Checks");
    if event.checks.is_empty() {
        lines.push(DetailLine::new("No deterministic checks recorded.", Tone::Muted));
    } else {
        lines.extend(event.checks.iter().map(|check| {
            DetailLine::new(
                format!("{} {} — {}", mark(check.ok), check.name, check.detail),
                bool_tone(check.ok),
            )
        }));
    }

    heading(&mut lines, "Debug controls");
    lines.push(DetailLine::new(
        "cfetch maintain history --json",
        Tone::Accent,
    ));
    if let Some(proposal) = &event.proposal_id {
        lines.push(DetailLine::new(
            format!("cfetch maintain show {proposal} --json"),
            Tone::Accent,
        ));
        if matches!(event.outcome, maintenance::EventOutcome::Applied) {
            lines.push(DetailLine::new(
                format!("cfetch maintain revert {proposal}"),
                Tone::Warning,
            ));
        }
    }
    lines.push(DetailLine::new(
        "Direct edits in the Markdown tree remain authoritative; cfetch will never overwrite changed bytes with this old transaction.",
        Tone::Muted,
    ));

    DetailDocument {
        title: format!(" {outcome} · {} ", event.id),
        lines,
    }
}

fn candidate_detail(candidate: &staging::Candidate) -> DetailDocument {
    let mut lines = Vec::new();
    lines.push(DetailLine::new(lifecycle("candidate"), Tone::Warning));
    heading(&mut lines, "Captured candidate");
    fields(
        &mut lines,
        [
            ("State".to_string(), "CANDIDATE".to_string()),
            ("Transition".to_string(), "not proposed".to_string()),
            ("Target".to_string(), "not selected".to_string()),
            ("Trap".to_string(), candidate.reason.clone()),
            ("Event kind".to_string(), candidate.kind.clone()),
            (
                "Recorded".to_string(),
                format!("Unix timestamp {}", candidate.ts),
            ),
            ("Origin".to_string(), candidate.host.clone()),
            ("Session".to_string(), candidate.session.clone()),
        ],
    );
    heading(&mut lines, "Captured evidence preview");
    let payload = serde_json::to_string_pretty(&candidate.payload)
        .unwrap_or_else(|_| "<unreadable payload>".to_string());
    lines.extend(
        payload
            .lines()
            .map(|line| DetailLine::new(line, Tone::Normal)),
    );
    heading(&mut lines, "Automatic path");
    lines.push(DetailLine::new(
        "The daemon proposes, independently reviews, verifies, and either applies or safely settles this evidence.",
        Tone::Good,
    ));
    lines.push(DetailLine::new(
        "A direct Obsidian or Markdown edit always wins if the target changes before apply.",
        Tone::Muted,
    ));
    heading(&mut lines, "Debug controls");
    lines.push(DetailLine::new(
        format!("cfetch maintain packet {} --json", candidate.id),
        Tone::Accent,
    ));
    lines.push(DetailLine::new(
        "Use `cfetch maintain run` to request an immediate bounded cycle; pause only when investigating behavior.",
        Tone::Muted,
    ));
    DetailDocument {
        title: format!(" candidate · {} ", candidate.id),
        lines,
    }
}

fn proposal_detail(
    cfg: &Config,
    summary: &maintenance::ProposalSummary,
    can_verify_locally: bool,
) -> anyhow::Result<DetailDocument> {
    let (state, proposal) = maintenance::get(cfg, &summary.id)?;
    let review = maintenance::get_review(cfg, &summary.id)?;
    let verification = if can_verify_locally {
        maintenance::inspect_verification(cfg, &summary.id)?
    } else {
        None
    };

    let mut lines = Vec::new();
    lines.push(DetailLine::new(lifecycle(&state), state_tone(&state)));
    heading(&mut lines, "Proposal");
    fields(
        &mut lines,
        [
            ("State".to_string(), state.to_ascii_uppercase()),
            ("Transition".to_string(), enum_name(proposal.transition)),
            ("Authority".to_string(), enum_name(proposal.authority)),
            (
                "Target".to_string(),
                proposal
                    .target
                    .clone()
                    .unwrap_or_else(|| "no memory write".to_string()),
            ),
            (
                "Created".to_string(),
                format!("Unix timestamp {}", proposal.created_at),
            ),
        ],
    );
    heading(&mut lines, "Candidate evidence");
    if proposal.candidates.is_empty() {
        lines.push(DetailLine::new("No candidates recorded.", Tone::Error));
    } else {
        lines.extend(proposal.candidates.iter().map(|candidate| {
            DetailLine::new(
                format!(
                    "• {}  sha256:{}",
                    candidate.id,
                    short_hash(&candidate.sha256)
                ),
                Tone::Normal,
            )
        }));
    }
    for evidence in &proposal.evidence {
        lines.push(DetailLine::new(
            format!("• evidence {evidence}"),
            Tone::Normal,
        ));
    }
    for citation in &proposal.related_citations {
        lines.push(DetailLine::new(
            format!(
                "• cite {}  sha256:{}",
                citation.cite,
                short_hash(&citation.sha256)
            ),
            Tone::Normal,
        ));
    }
    heading(&mut lines, "Rationale");
    lines.extend(
        proposal
            .rationale
            .lines()
            .map(|line| DetailLine::new(line, Tone::Normal)),
    );

    review_section(&mut lines, review.as_ref());
    verification_section(
        &mut lines,
        &state,
        verification.as_ref(),
        can_verify_locally,
    );
    diff_section(&mut lines, &proposal, verification.as_ref());
    next_step_section(
        &mut lines,
        &state,
        &proposal,
        review.as_ref(),
        verification.as_ref(),
    );

    Ok(DetailDocument {
        title: format!(" {} · {} ", state, proposal.id),
        lines,
    })
}

fn review_section(lines: &mut Vec<DetailLine>, review: Option<&maintenance::Review>) {
    heading(lines, "Independent semantic review");
    let Some(review) = review else {
        lines.push(DetailLine::new("○ not recorded", Tone::Warning));
        for gate in [
            "evidence coverage",
            "factual faithfulness",
            "preservation",
            "authority fit",
            "target fit",
            "contradiction check",
        ] {
            lines.push(DetailLine::new(format!("○ {gate}"), Tone::Muted));
        }
        return;
    };
    let passed = review.verdict == maintenance::ReviewVerdict::Pass;
    lines.push(DetailLine::new(
        format!(
            "{} verdict {} via {} · {}",
            mark(passed),
            enum_name(review.verdict),
            enum_name(review.method),
            review.id
        ),
        bool_tone(passed),
    ));
    for (name, ok) in [
        ("evidence coverage", review.evidence_coverage),
        ("factual faithfulness", review.factual_faithfulness),
        ("preservation", review.preservation),
        ("authority fit", review.authority_fit),
        ("target fit", review.target_fit),
        ("contradiction check", review.contradiction_checked),
    ] {
        lines.push(DetailLine::new(
            format!("{} {name}", mark(ok)),
            bool_tone(ok),
        ));
    }
    lines.push(DetailLine::new(
        format!("Notes: {}", review.notes),
        Tone::Muted,
    ));
}

fn verification_section(
    lines: &mut Vec<DetailLine>,
    state: &str,
    verification: Option<&maintenance::Verification>,
    can_verify_locally: bool,
) {
    heading(lines, "Deterministic verification");
    if !can_verify_locally && matches!(state, "pending" | "applied") {
        lines.push(DetailLine::new(
            "Not run here: this host delegates queries and the dashboard will not create a second local index.",
            Tone::Warning,
        ));
        lines.push(DetailLine::new(
            "Run verification on a storage host that holds the tree and its current derived state.",
            Tone::Muted,
        ));
        return;
    }
    let Some(verification) = verification else {
        lines.push(DetailLine::new(
            format!("Not rerun: {state} is a terminal lifecycle record."),
            Tone::Muted,
        ));
        return;
    };
    lines.push(DetailLine::new(
        if verification.valid {
            "READY: every current gate passes"
        } else {
            "BLOCKED: one or more gates fail"
        },
        bool_tone(verification.valid),
    ));
    for check in &verification.checks {
        lines.push(DetailLine::new(
            format!("{} {} — {}", mark(check.ok), check.name, check.detail),
            bool_tone(check.ok),
        ));
    }
}

fn diff_section(
    lines: &mut Vec<DetailLine>,
    proposal: &maintenance::Proposal,
    verification: Option<&maintenance::Verification>,
) {
    let diff = verification
        .and_then(|report| report.diff.clone())
        .or_else(|| maintenance::exact_diff(proposal));
    let Some(diff) = diff else {
        heading(lines, "Memory write");
        lines.push(DetailLine::new(
            "None. This decision-only transition cannot alter trusted memory.",
            Tone::Good,
        ));
        return;
    };
    heading(lines, "Exact proposed diff");
    let (preview, omitted) = bounded_preview(&diff, MAX_PREVIEW_BYTES);
    lines.extend(preview.lines().map(|line| {
        let tone = if line.starts_with('+') && !line.starts_with("+++") {
            Tone::Good
        } else if line.starts_with('-') && !line.starts_with("---") {
            Tone::Error
        } else {
            Tone::Normal
        };
        DetailLine::new(line, tone)
    }));
    if omitted > 0 {
        lines.push(DetailLine::new(
            format!(
                "… {omitted} diff bytes omitted from this preview; `cfetch maintain show {} --json` retains the exact complete before/after bytes.",
                proposal.id
            ),
            Tone::Warning,
        ));
    }
}

fn next_step_section(
    lines: &mut Vec<DetailLine>,
    state: &str,
    proposal: &maintenance::Proposal,
    review: Option<&maintenance::Review>,
    verification: Option<&maintenance::Verification>,
) {
    heading(lines, "Automatic disposition and debug controls");
    match state {
        "pending" if review.is_none() => {
            lines.push(DetailLine::new(
                "Awaiting semantic review. The autonomous runner normally performs this without interaction.",
                Tone::Warning,
            ));
            lines.push(DetailLine::new(
                format!("Debug manually: cfetch maintain review {} --file review.json", proposal.id),
                Tone::Accent,
            ));
        }
        "pending"
            if review.is_some_and(|review| {
                review.verdict == maintenance::ReviewVerdict::Fail
                    || !review.evidence_coverage
                    || !review.factual_faithfulness
                    || !review.preservation
                    || !review.authority_fit
                    || !review.target_fit
                    || !review.contradiction_checked
            }) =>
        {
            lines.push(DetailLine::new(
                "The failed review is immutable. The autonomous runner settles this evidence as an exception or dismissal.",
                Tone::Error,
            ));
            lines.push(DetailLine::new(
                format!("cfetch maintain reject {}", proposal.id),
                Tone::Accent,
            ));
        }
        "pending" => {
            if verification.is_some_and(|report| report.valid) {
                lines.push(DetailLine::new(
                    "All current gates pass. The autonomous runner applies the exact revision without routine approval.",
                    Tone::Good,
                ));
                lines.push(DetailLine::new(
                    format!("Run now: cfetch maintain auto-apply {}", proposal.id),
                    Tone::Accent,
                ));
            } else {
                lines.push(DetailLine::new(
                    format!("Debug gates: cfetch maintain verify {}", proposal.id),
                    Tone::Accent,
                ));
            }
        }
        "applied" => {
            lines.push(DetailLine::new(
                "Legacy/manual apply state. Automatic transactions settle directly into immutable history.",
                Tone::Warning,
            ));
            lines.push(DetailLine::new(
                format!("Finish legacy transaction: cfetch maintain finalize {}", proposal.id),
                Tone::Accent,
            ));
            lines.push(DetailLine::new(
                format!(
                    "Undo before finalization: cfetch maintain revert {}",
                    proposal.id
                ),
                Tone::Warning,
            ));
        }
        "finalized" => lines.push(DetailLine::new(
            "Settled. Exact target bytes and the immutable event record retain the provenance; no interaction is required.",
            Tone::Good,
        )),
        "rejected" => lines.push(DetailLine::new(
            "Closed without applying the proposal.",
            Tone::Muted,
        )),
        "reverted" => lines.push(DetailLine::new(
            "Closed after restoring the captured before bytes.",
            Tone::Muted,
        )),
        _ => lines.push(DetailLine::new(
            "No action is available for this state.",
            Tone::Muted,
        )),
    }
}

fn mark(ok: bool) -> &'static str {
    if ok { "✓" } else { "✗" }
}

fn bool_tone(ok: bool) -> Tone {
    if ok { Tone::Good } else { Tone::Error }
}

fn short_hash(hash: &str) -> &str {
    hash.get(..12).unwrap_or(hash)
}

fn bounded_preview(text: &str, limit: usize) -> (String, usize) {
    if text.len() <= limit {
        return (text.to_string(), 0);
    }
    let head_target = limit * 3 / 4;
    let tail_target = limit - head_target;
    let mut head = head_target.min(text.len());
    while !text.is_char_boundary(head) {
        head -= 1;
    }
    let mut tail = text.len().saturating_sub(tail_target);
    while !text.is_char_boundary(tail) {
        tail += 1;
    }
    let omitted = tail.saturating_sub(head);
    (
        format!("{}\n… preview gap …\n{}", &text[..head], &text[tail..]),
        omitted,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn candidate(id: &str, ts: i64) -> staging::Candidate {
        staging::Candidate {
            id: id.to_string(),
            reason: "fix-discovered".to_string(),
            session: "session-a".to_string(),
            host: "example-host".to_string(),
            ts,
            kind: "bash".to_string(),
            payload: json!({"command": "cargo test", "recovered": true}),
        }
    }

    fn text(document: &DetailDocument) -> String {
        document
            .lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn candidate_detail_exposes_evidence_and_autonomous_path() {
        let document = candidate_detail(&candidate("fix-discovered-a1b2c3d4", 42));
        let rendered = text(&document);
        assert!(rendered.contains("[CAPTURED] → propose → review → apply / settle"));
        assert!(
            rendered.contains("cargo test"),
            "captured payload must be visible"
        );
        assert!(rendered.contains("cfetch maintain packet fix-discovered-a1b2c3d4 --json"));
        assert!(rendered.contains("daemon proposes, independently reviews, verifies"));
        assert!(rendered.contains("direct Obsidian or Markdown edit always wins"));
    }

    #[test]
    fn event_detail_exposes_outcome_provenance_checks_and_revert() {
        let event = maintenance::MaintenanceEvent {
            schema_version: maintenance::SCHEMA_VERSION,
            id: "event-0011223344556677".to_string(),
            created_at: 44,
            created_by_host: "example-host".to_string(),
            proposal_id: Some("maintenance-001122334455".to_string()),
            candidate_ids: vec!["fix-discovered-a1b2c3d4".to_string()],
            outcome: maintenance::EventOutcome::Applied,
            target: Some("mind/facts.md".to_string()),
            before_sha256: Some("00".repeat(32)),
            after_sha256: Some("11".repeat(32)),
            review_id: Some("review-example".to_string()),
            checks: vec![maintenance::Check {
                name: "target unchanged".to_string(),
                ok: true,
                detail: "exact before bytes still present".to_string(),
            }],
            detail: "applied exact reviewed bytes".to_string(),
        };
        let rendered = text(&event_detail(&event));
        assert!(rendered.contains("AUTONOMOUS OUTCOME: APPLIED"));
        assert!(rendered.contains("candidate fix-discovered-a1b2c3d4"));
        assert!(rendered.contains("✓ target unchanged"));
        assert!(rendered.contains("cfetch maintain revert maintenance-001122334455"));
        assert!(rendered.contains("Markdown tree remain authoritative"));
    }

    #[test]
    fn inbox_combines_candidates_and_all_proposal_states_newest_first() {
        let root = tempfile::tempdir().unwrap();
        let cfg = Config {
            brain_root: root.path().to_path_buf(),
            ..Config::default()
        };
        let staging_dir = paths::staging_dir(root.path());
        staging::write(&staging_dir, &candidate("fix-discovered-a1b2c3d4", 42)).unwrap();
        let finalized = root
            .path()
            .join("scratch/cfetch-staging/maintenance/finalized/maintenance-000000000000.md");
        std::fs::create_dir_all(finalized.parent().unwrap()).unwrap();
        let proposal = maintenance::Proposal {
            schema_version: maintenance::SCHEMA_VERSION,
            id: "maintenance-000000000000".to_string(),
            created_at: 41,
            created_by_host: "example-host".to_string(),
            candidates: Vec::new(),
            transition: maintenance::Transition::Noop,
            target: None,
            authority: maintenance::Authority::Unendorsed,
            valid_until: None,
            rationale: "Nothing durable to record.".to_string(),
            evidence: Vec::new(),
            related_citations: Vec::new(),
            before_sha256: None,
            before: None,
            after: None,
        };
        let json = serde_json::to_string_pretty(&proposal).unwrap();
        std::fs::write(
            finalized,
            format!("---\nring: 5\n---\n\n```json\n{json}\n```\n"),
        )
        .unwrap();

        let inbox = Inbox::load(&cfg, true);
        let rows = inbox.rows();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].badge, "candidate");
        assert_eq!(rows[1].badge, "finalized");
        assert!(text(&inbox.detail).contains("Captured evidence preview"));
    }

    #[test]
    fn review_section_names_every_semantic_gate_and_immutable_failure() {
        let review = maintenance::Review {
            schema_version: maintenance::SCHEMA_VERSION,
            id: "review-example".to_string(),
            proposal_id: "maintenance-example".to_string(),
            proposal_sha256: "00".repeat(32),
            created_at: 1,
            created_by_host: "example-host".to_string(),
            verdict: maintenance::ReviewVerdict::Fail,
            method: maintenance::ReviewMethod::IndependentAgent,
            evidence_coverage: true,
            factual_faithfulness: false,
            preservation: true,
            authority_fit: true,
            target_fit: false,
            contradiction_checked: true,
            notes: "The target overstates the evidence.".to_string(),
        };
        let mut lines = Vec::new();
        review_section(&mut lines, Some(&review));
        let rendered = lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for gate in [
            "evidence coverage",
            "factual faithfulness",
            "preservation",
            "authority fit",
            "target fit",
            "contradiction check",
        ] {
            assert!(rendered.contains(gate), "missing {gate}: {rendered}");
        }
        assert!(rendered.contains("✗ verdict fail"));
    }

    #[test]
    fn remote_verification_is_explicitly_skipped() {
        let mut lines = Vec::new();
        verification_section(&mut lines, "pending", None, false);
        let rendered = lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("delegates queries"));
        assert!(rendered.contains("will not create a second local index"));
        assert!(rendered.contains("storage host"));
        assert!(!rendered.contains("READY"));
    }

    #[test]
    fn valid_pending_proposal_exposes_the_automatic_apply_path() {
        let proposal = maintenance::Proposal {
            schema_version: maintenance::SCHEMA_VERSION,
            id: "maintenance-001122334455".to_string(),
            created_at: 1,
            created_by_host: "example-host".to_string(),
            candidates: Vec::new(),
            transition: maintenance::Transition::Noop,
            target: None,
            authority: maintenance::Authority::Unendorsed,
            valid_until: None,
            rationale: "Nothing durable to record.".to_string(),
            evidence: Vec::new(),
            related_citations: Vec::new(),
            before_sha256: None,
            before: None,
            after: None,
        };
        let review = maintenance::Review {
            schema_version: maintenance::SCHEMA_VERSION,
            id: "review-example".to_string(),
            proposal_id: proposal.id.clone(),
            proposal_sha256: "00".repeat(32),
            created_at: 1,
            created_by_host: "example-host".to_string(),
            verdict: maintenance::ReviewVerdict::Pass,
            method: maintenance::ReviewMethod::IndependentAgent,
            evidence_coverage: true,
            factual_faithfulness: true,
            preservation: true,
            authority_fit: true,
            target_fit: true,
            contradiction_checked: true,
            notes: "All gates pass.".to_string(),
        };
        let verification = maintenance::Verification {
            proposal_id: proposal.id.clone(),
            valid: true,
            checks: Vec::new(),
            review_id: Some(review.id.clone()),
            approval_token: Some("approve-revision001122".to_string()),
            diff: None,
        };
        let mut lines = Vec::new();
        next_step_section(
            &mut lines,
            "pending",
            &proposal,
            Some(&review),
            Some(&verification),
        );
        let rendered = lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("without routine approval"));
        assert!(rendered.contains(
            "cfetch maintain auto-apply maintenance-001122334455"
        ));
        assert!(!rendered.contains("--approval-token"));
        assert!(!rendered.contains("maintain verify"));
    }

    #[test]
    fn large_diff_preview_is_bounded_and_keeps_both_ends() {
        let input = format!("START{}END", "x".repeat(400));
        let (preview, omitted) = bounded_preview(&input, 100);
        assert!(preview.starts_with("START"));
        assert!(preview.ends_with("END"));
        assert!(preview.contains("preview gap"));
        assert!(omitted > 0);
    }
}
