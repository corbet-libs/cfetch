//! One privacy-bounded runtime truth, rendered for the CLI, TUI, hooks, and
//! MCP. The line renderer is cache-only: it never dials a serving host or an
//! inference endpoint and never causes spend.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::{fsutil, index, maintenance, paths};

pub const SCHEMA_VERSION: u32 = 1;
const SNAPSHOT_FILE: &str = "runtime-status-v1.json";
const UPDATE_LOCK_WAIT_MS: u64 = 50;
const MAX_FAILURES: usize = 8;
pub const MCP_MAX_BYTES: usize = 2 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceState {
    Ready,
    Degraded,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRoute {
    Local,
    Serving,
    Remote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FreshnessState {
    Fresh,
    Stale,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalMode {
    Lexical,
    Semantic,
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorCoverageState {
    Complete,
    Partial,
    None,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceMode {
    Unknown,
    Disabled,
    Local,
    Endpoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceRoute {
    Local,
    Remote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub state: ServiceState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LastAnswerStatus {
    pub state: FreshnessState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_at: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRouteStatus {
    pub mode: MemoryRoute,
    pub origin_label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<u64>,
    pub last_answer: LastAnswerStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalStatus {
    pub mode: RetrievalMode,
    pub vector_coverage: VectorCoverageState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedded: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendSelection {
    pub backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route: Option<InferenceRoute>,
    pub selected_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceAttempt {
    pub route: InferenceRoute,
    pub backend: String,
    pub success: bool,
    pub observed_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceStatus {
    pub configured: InferenceMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configured_route: Option<InferenceRoute>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected: Option<BackendSelection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_used: Option<InferenceAttempt>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaintenanceStatus {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub configured: bool,
    #[serde(default)]
    pub route: Option<InferenceRoute>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub last_model_activity: Option<String>,
    #[serde(default)]
    pub last_model_success: Option<bool>,
    #[serde(default)]
    pub last_model_at: Option<u64>,
    #[serde(default)]
    pub paused: bool,
    #[serde(default)]
    pub candidates: u64,
    /// Quarantined proposals awaiting an automatic or debugging action.
    #[serde(default)]
    pub pending: u64,
    /// Legacy manual proposals written but not git-finalized.
    #[serde(default)]
    pub applied: u64,
    #[serde(default)]
    pub history_events: u64,
    #[serde(default)]
    pub exceptions: u64,
    #[serde(default)]
    pub last_event_at: Option<u64>,
    #[serde(default)]
    pub last_outcome: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeFailure {
    pub code: String,
    pub severity: FailureSeverity,
    pub action: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStatusV1 {
    pub schema_version: u32,
    pub observed_at: u64,
    pub service: ServiceStatus,
    pub memory_route: MemoryRouteStatus,
    pub retrieval: RetrievalStatus,
    pub inference: InferenceStatus,
    pub maintenance: MaintenanceStatus,
    #[serde(default)]
    pub failures: Vec<RuntimeFailure>,
}

impl Default for RuntimeStatusV1 {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            observed_at: now(),
            service: ServiceStatus {
                state: ServiceState::Ready,
            },
            memory_route: MemoryRouteStatus {
                mode: MemoryRoute::Local,
                origin_label: "local".to_string(),
                generation: None,
                last_answer: LastAnswerStatus {
                    state: FreshnessState::Unknown,
                    observed_at: None,
                },
            },
            retrieval: RetrievalStatus {
                mode: RetrievalMode::Lexical,
                vector_coverage: VectorCoverageState::Unknown,
                embedded: None,
                total: None,
            },
            inference: InferenceStatus {
                configured: InferenceMode::Disabled,
                configured_route: None,
                selected: None,
                last_used: None,
            },
            maintenance: MaintenanceStatus::default(),
            failures: Vec::new(),
        }
    }
}

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn snapshot_path_in(state_dir: &Path) -> PathBuf {
    state_dir.join(SNAPSHOT_FILE)
}

fn route_for(_cfg: &Config) -> MemoryRoute {
    MemoryRoute::Local
}

fn origin_label(route: MemoryRoute) -> String {
    match route {
        MemoryRoute::Local => "local".to_string(),
        MemoryRoute::Serving => "this-host".to_string(),
        MemoryRoute::Remote => "serving-host".to_string(),
    }
}

/// Missing status is not permission to read source configuration. This path
/// is shared by cached UI/hooks and query telemetry; either must stay local.
fn unavailable_cached_status() -> RuntimeStatusV1 {
    let mut status = RuntimeStatusV1::default();
    status.service.state = ServiceState::Unavailable;
    status.inference.configured = InferenceMode::Unknown;
    upsert_failure(
        &mut status,
        "runtime_status_unavailable",
        FailureSeverity::Critical,
        "run cfetch status to refresh runtime metadata",
    );
    status
}

fn apply_config(status: &mut RuntimeStatusV1, cfg: &Config) {
    let route = route_for(cfg);
    if status.memory_route.mode != route {
        status.memory_route.mode = route;
        status.memory_route.origin_label = origin_label(route);
        status.memory_route.generation = None;
        status.memory_route.last_answer = LastAnswerStatus {
            state: FreshnessState::Unknown,
            observed_at: None,
        };
        status.retrieval.vector_coverage = VectorCoverageState::Unknown;
        status.retrieval.embedded = None;
        status.retrieval.total = None;
        remove_failure(status, "vector_coverage_partial");
        remove_failure(status, "vector_coverage_none");
        status.service.state = if route == MemoryRoute::Remote {
            ServiceState::Degraded
        } else {
            ServiceState::Ready
        };
    }
    let local_plan = crate::local_inference::selected_local_package_plan();
    let local_plan_invalid = cfg.embeddings.enabled
        && cfg.embeddings.local_model.is_none()
        && cfg.embeddings.endpoint.is_empty()
        && local_plan.is_err();
    let packaged_local_embedding = cfg.embeddings.enabled
        && cfg.embeddings.endpoint.is_empty()
        && (cfg.embeddings.local_model.is_some()
            || local_plan.as_ref().is_ok_and(|plan| plan.is_some()));
    let configured = if packaged_local_embedding && !cfg.rerank.enabled {
        InferenceMode::Local
    } else if cfg.embeddings.enabled || cfg.rerank.enabled {
        InferenceMode::Endpoint
    } else {
        InferenceMode::Disabled
    };
    let embedding_route = if packaged_local_embedding {
        Some(InferenceRoute::Local)
    } else {
        (cfg.embeddings.enabled && !cfg.embeddings.endpoint.is_empty())
            .then(|| endpoint_route(&cfg.embeddings.endpoint))
    };
    let rerank_route = (cfg.rerank.enabled && !cfg.rerank.endpoint.is_empty())
        .then(|| endpoint_route(&cfg.rerank.endpoint));
    let configured_route = match (embedding_route, rerank_route) {
        (Some(embedding), Some(rerank)) if embedding != rerank => None,
        (Some(route), _) | (_, Some(route)) => Some(route),
        (None, None) => None,
    };
    if status.inference.configured != configured
        || status.inference.configured_route != configured_route
    {
        status.inference.configured = configured;
        status.inference.configured_route = configured_route;
        status.inference.selected = None;
        status.inference.last_used = None;
    }
    if configured == InferenceMode::Disabled {
        status.inference.configured_route = None;
        status.retrieval.mode = RetrievalMode::Lexical;
        status.retrieval.vector_coverage = VectorCoverageState::Unknown;
        status.retrieval.embedded = None;
        status.retrieval.total = None;
        remove_failure(status, "vector_coverage_partial");
        remove_failure(status, "vector_coverage_none");
        remove_failure(status, "retrieval_degraded");
        remove_failure(status, "inference_unavailable");
        remove_failure(status, "inference_initialization_failed");
    }
    remove_failure(status, "inference_misconfigured");
    if local_plan_invalid {
        upsert_failure(
            status,
            "inference_initialization_failed",
            FailureSeverity::Warning,
            "install an intact target-local inference package or configure an explicit endpoint",
        );
    }
    if (cfg.embeddings.enabled
        && ((!packaged_local_embedding
            && !local_plan_invalid
            && cfg.embeddings.endpoint.is_empty())
            || cfg.embeddings.model.is_empty()))
        || (cfg.rerank.enabled && (cfg.rerank.endpoint.is_empty() || cfg.rerank.model.is_empty()))
    {
        upsert_failure(
            status,
            "inference_misconfigured",
            FailureSeverity::Warning,
            "complete the enabled inference configuration",
        );
    }
    remove_failure(status, "config_unusable");
    remove_failure(status, "runtime_status_unavailable");
    recover_if_clean(status);
}

pub fn load_cached() -> RuntimeStatusV1 {
    load_cached_in(&paths::state_dir())
}

pub(crate) fn load_cached_in(state_dir: &Path) -> RuntimeStatusV1 {
    let mut status = std::fs::read_to_string(snapshot_path_in(state_dir))
        .ok()
        .and_then(|raw| serde_json::from_str::<RuntimeStatusV1>(&raw).ok())
        .filter(|status| status.schema_version == SCHEMA_VERSION)
        .unwrap_or_else(unavailable_cached_status);
    normalize(&mut status);
    status
}

fn store_in(state_dir: &Path, status: &RuntimeStatusV1) -> anyhow::Result<()> {
    let path = snapshot_path_in(state_dir);
    #[cfg(unix)]
    if path.exists() {
        use std::os::unix::fs::PermissionsExt as _;
        // Set the target private before atomic_write asks which mode its
        // replacement should preserve, so there is no world-readable rename
        // window even if someone accidentally widened the old snapshot.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }
    fsutil::atomic_write(&path, serde_json::to_vec(status)?)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn update_in(
    state_dir: &Path,
    mutate: impl FnOnce(&mut RuntimeStatusV1),
) -> Option<RuntimeStatusV1> {
    let locks = state_dir.join("locks");
    std::fs::create_dir_all(&locks).ok()?;
    let _lock =
        crate::lockfile::acquire(&locks.join("runtime-status.lock"), UPDATE_LOCK_WAIT_MS, 0)?;
    let old = load_cached_in(state_dir);
    let mut next = old.clone();
    mutate(&mut next);
    next.schema_version = SCHEMA_VERSION;
    normalize(&mut next);
    let mut comparable_old = old.clone();
    let mut comparable_next = next.clone();
    comparable_old.observed_at = 0;
    comparable_next.observed_at = 0;
    let observed = now();
    if comparable_old == comparable_next && old.observed_at == observed {
        return Some(old);
    }
    next.observed_at = observed;
    store_in(state_dir, &next).ok()?;
    Some(next)
}

fn update(mutate: impl FnOnce(&mut RuntimeStatusV1)) -> Option<RuntimeStatusV1> {
    // Unit tests exercise endpoint clients in parallel. Unless a test opts
    // into an isolated state directory, telemetry must not touch the real
    // user's snapshot as a side effect of a test request.
    #[cfg(test)]
    std::env::var_os("CFETCH_STATE_DIR")?;
    update_in(&paths::state_dir(), mutate)
}

fn normalize(status: &mut RuntimeStatusV1) {
    status.memory_route.origin_label = origin_label(status.memory_route.mode);
    status.maintenance.model = status
        .maintenance
        .model
        .as_deref()
        .and_then(safe_model_label);
    status.maintenance.last_model_activity = status
        .maintenance
        .last_model_activity
        .as_deref()
        .and_then(safe_maintenance_activity);
    if !status.maintenance.configured {
        status.maintenance.route = None;
        status.maintenance.model = None;
        status.maintenance.last_model_activity = None;
        status.maintenance.last_model_success = None;
        status.maintenance.last_model_at = None;
    }
    if let Some(selected) = &mut status.inference.selected {
        selected.backend = safe_backend_label(&selected.backend);
        selected.device_class = selected.device_class.as_deref().and_then(safe_device_label);
    }
    if let Some(last) = &mut status.inference.last_used {
        last.backend = safe_backend_label(&last.backend);
    }
    status.failures = status
        .failures
        .iter()
        .filter_map(|failure| canonical_failure(&failure.code))
        .collect();
    status.failures.sort_by(|a, b| a.code.cmp(&b.code));
    status.failures.dedup_by(|a, b| a.code == b.code);
    status.failures.truncate(MAX_FAILURES);
    if status
        .failures
        .iter()
        .any(|failure| failure.severity == FailureSeverity::Critical)
    {
        status.service.state = ServiceState::Unavailable;
    } else if status
        .failures
        .iter()
        .any(|failure| failure.severity == FailureSeverity::Warning)
    {
        status.service.state = ServiceState::Degraded;
    }
}

fn canonical_failure(code: &str) -> Option<RuntimeFailure> {
    let (severity, action) = match code {
        "runtime_status_unavailable" => (
            FailureSeverity::Critical,
            "run cfetch status to refresh runtime metadata",
        ),
        "config_unusable" => (FailureSeverity::Critical, "run cfetch selfcheck"),
        "daemon_unavailable" => (
            FailureSeverity::Warning,
            "run cfetch daemon start or cfetch selfcheck",
        ),
        "remote_unavailable" | "memory_unavailable" => (
            FailureSeverity::Critical,
            "check cfetch status and serving connectivity",
        ),
        "memory_stale" => (
            FailureSeverity::Warning,
            "wait for serving drain or check cfetch status",
        ),
        "retrieval_degraded" => (
            FailureSeverity::Warning,
            "use lexical recall or restore semantic inference",
        ),
        "vector_coverage_partial" | "vector_coverage_none" => {
            (FailureSeverity::Warning, "run cfetch embed-index")
        }
        "inference_unavailable" => (
            FailureSeverity::Warning,
            "check inference configuration or use lexical recall",
        ),
        "inference_misconfigured" => (
            FailureSeverity::Warning,
            "complete the enabled inference configuration",
        ),
        "inference_initialization_failed" => (
            FailureSeverity::Warning,
            "check inference credentials and endpoint policy",
        ),
        "maintenance_unconfigured" => (
            FailureSeverity::Warning,
            "configure maintenance.endpoint and maintenance.model or run cfetch maintain pause",
        ),
        "maintenance_exception" => (
            FailureSeverity::Warning,
            "run cfetch maintain history and inspect the latest exception",
        ),
        "maintenance_model_unavailable" => (
            FailureSeverity::Warning,
            "check maintenance model configuration and endpoint policy",
        ),
        _ => return None,
    };
    Some(RuntimeFailure {
        code: code.to_string(),
        severity,
        action: action.to_string(),
    })
}

fn recover_if_clean(status: &mut RuntimeStatusV1) {
    if status.failures.is_empty()
        && (status.memory_route.mode != MemoryRoute::Remote
            || status.memory_route.last_answer.observed_at.is_some())
        && status.memory_route.last_answer.state != FreshnessState::Stale
    {
        status.service.state = ServiceState::Ready;
    }
}

fn upsert_failure(
    status: &mut RuntimeStatusV1,
    code: &str,
    severity: FailureSeverity,
    action: &str,
) {
    remove_failure(status, code);
    status.failures.push(RuntimeFailure {
        code: code.to_string(),
        severity,
        action: action.to_string(),
    });
}

fn remove_failure(status: &mut RuntimeStatusV1, code: &str) {
    status.failures.retain(|failure| failure.code != code);
}

/// Refreshes only local, non-spending facts. It may read config, the local
/// SQLite snapshot, and maintenance directories; it never opens a socket or
/// creates an inference client.
pub fn refresh_static() -> anyhow::Result<RuntimeStatusV1> {
    let cfg = Config::load();
    let status = update(|status| match &cfg {
        Ok(cfg) => {
            apply_config(status, cfg);
            let events = maintenance::history(cfg);
            status.maintenance.enabled = cfg.maintenance.enabled;
            status.maintenance.configured = cfg.maintenance.configured();
            status.maintenance.route = status
                .maintenance
                .configured
                .then(|| endpoint_route(&cfg.maintenance.endpoint));
            status.maintenance.model = if status.maintenance.configured {
                safe_model_label(&cfg.maintenance.model)
            } else {
                None
            };
            status.maintenance.paused = maintenance::is_paused(cfg);
            status.maintenance.candidates =
                crate::staging::pending_count(&paths::staging_dir(&cfg.brain_root)) as u64;
            status.maintenance.pending = maintenance::pending_count(cfg) as u64;
            status.maintenance.applied = maintenance::applied_count(cfg) as u64;
            status.maintenance.history_events = events.len() as u64;
            status.maintenance.exceptions = events
                .iter()
                .filter(|event| event.outcome == maintenance::EventOutcome::Exception)
                .count() as u64;
            status.maintenance.last_event_at =
                events.first().map(|event| event.created_at.max(0) as u64);
            status.maintenance.last_outcome = events
                .first()
                .map(|event| format!("{:?}", event.outcome).to_ascii_lowercase());
            remove_failure(status, "maintenance_unconfigured");
            remove_failure(status, "maintenance_exception");
            if !status.maintenance.enabled || !status.maintenance.configured {
                remove_failure(status, "maintenance_model_unavailable");
            }
            if status.maintenance.enabled
                && !status.maintenance.configured
                && status.maintenance.candidates > 0
                && !status.maintenance.paused
            {
                upsert_failure(
                    status,
                    "maintenance_unconfigured",
                    FailureSeverity::Warning,
                    "configure maintenance.endpoint and maintenance.model or pause maintenance",
                );
            }
            if status.maintenance.candidates > 0
                && status.maintenance.last_outcome.as_deref() == Some("exception")
            {
                upsert_failure(
                    status,
                    "maintenance_exception",
                    FailureSeverity::Warning,
                    "inspect cfetch maintain history",
                );
            }
            {
                status.memory_route.generation = None;
                if cfg.embeddings.enabled {
                    status.retrieval.vector_coverage = VectorCoverageState::Unknown;
                    status.retrieval.embedded = None;
                    status.retrieval.total = None;
                    remove_failure(status, "vector_coverage_partial");
                    remove_failure(status, "vector_coverage_none");
                }
                if let Ok(conn) = index::open_ro(&paths::state_dir()) {
                    status.memory_route.generation = Some(index::generation(&conn));
                    if cfg.embeddings.enabled {
                        match index::vector_coverage(&conn, &cfg.embeddings.spec()) {
                            Ok((embedded, total)) => {
                                set_vector_coverage(status, embedded as u64, total as u64)
                            }
                            Err(_) => {
                                status.retrieval.vector_coverage = VectorCoverageState::Unknown;
                                status.retrieval.embedded = None;
                                status.retrieval.total = None;
                            }
                        }
                    }
                }
            }
        }
        Err(_) => {
            status.service.state = ServiceState::Unavailable;
            upsert_failure(
                status,
                "config_unusable",
                FailureSeverity::Critical,
                "run cfetch selfcheck",
            );
        }
    })
    .unwrap_or_else(load_cached);
    Ok(status)
}

fn set_vector_coverage(status: &mut RuntimeStatusV1, embedded: u64, total: u64) {
    status.retrieval.embedded = Some(embedded);
    status.retrieval.total = Some(total);
    status.retrieval.vector_coverage =
        if (total == 0 && embedded == 0) || (total > 0 && embedded >= total) {
            VectorCoverageState::Complete
        } else if embedded > 0 {
            VectorCoverageState::Partial
        } else {
            VectorCoverageState::None
        };
    remove_failure(status, "vector_coverage_partial");
    remove_failure(status, "vector_coverage_none");
    match status.retrieval.vector_coverage {
        VectorCoverageState::Partial => upsert_failure(
            status,
            "vector_coverage_partial",
            FailureSeverity::Warning,
            "run cfetch embed-index",
        ),
        VectorCoverageState::None if total > 0 => upsert_failure(
            status,
            "vector_coverage_none",
            FailureSeverity::Warning,
            "run cfetch embed-index",
        ),
        _ => {}
    }
    recover_if_clean(status);
}

pub fn record_service(state: ServiceState, failure_code: Option<&str>) {
    let _ = update(|status| {
        status.service.state = state;
        remove_failure(status, "daemon_unavailable");
        if let Some(code) = failure_code {
            upsert_failure(
                status,
                code,
                FailureSeverity::Warning,
                "run cfetch daemon start or cfetch selfcheck",
            );
        }
    });
}

/// Applies a live daemon probe to a snapshot for immediate display without
/// persisting it. Cached surfaces remain observation-only and daemon lifecycle
/// events continue to own the durable service state — with one deliberate
/// exception: `cfetch daemon status` records its observation durably, because
/// it is the only repair path for a SIGKILL'd daemon (the shutdown handler
/// never ran, no lifecycle event will ever arrive). This is a feature, not
/// a bug; the observation is authoritative until the next lifecycle event
/// overwrites it.
pub fn apply_daemon_observation(status: &mut RuntimeStatusV1, running: bool) {
    remove_failure(status, "daemon_unavailable");
    if running {
        recover_if_clean(status);
    } else {
        upsert_failure(
            status,
            "daemon_unavailable",
            FailureSeverity::Warning,
            "run cfetch daemon start or cfetch selfcheck",
        );
    }
    normalize(status);
}

pub fn record_generation(route: MemoryRoute, generation: u64) {
    let _ = update(|status| {
        status.memory_route.mode = route;
        status.memory_route.origin_label = origin_label(route);
        status.memory_route.generation = Some(generation);
    });
}

pub fn record_memory_answer(
    route: MemoryRoute,
    generation: Option<u64>,
    fresh: Option<bool>,
    success: bool,
) {
    let _ = update(|status| apply_memory_answer(status, route, generation, fresh, success));
}

fn apply_memory_answer(
    status: &mut RuntimeStatusV1,
    route: MemoryRoute,
    generation: Option<u64>,
    fresh: Option<bool>,
    success: bool,
) {
    status.memory_route.mode = route;
    status.memory_route.origin_label = origin_label(route);
    remove_failure(status, "remote_unavailable");
    remove_failure(status, "memory_unavailable");
    remove_failure(status, "memory_stale");
    if success {
        if let Some(generation) = generation {
            status.memory_route.generation = Some(generation);
        }
        status.memory_route.last_answer = LastAnswerStatus {
            state: match fresh {
                Some(true) => FreshnessState::Fresh,
                Some(false) => FreshnessState::Stale,
                None => FreshnessState::Unknown,
            },
            observed_at: Some(now()),
        };
        status.service.state = if fresh == Some(false) {
            ServiceState::Degraded
        } else {
            ServiceState::Ready
        };
        if fresh == Some(false) {
            upsert_failure(
                status,
                "memory_stale",
                FailureSeverity::Warning,
                "wait for serving drain or check cfetch status",
            );
        }
    } else {
        status.service.state = ServiceState::Unavailable;
        upsert_failure(
            status,
            if route == MemoryRoute::Remote {
                "remote_unavailable"
            } else {
                "memory_unavailable"
            },
            FailureSeverity::Critical,
            "check cfetch status and serving connectivity",
        );
    }
}

pub fn record_retrieval(mode: RetrievalMode, degraded: bool) {
    let _ = update(|status| {
        status.retrieval.mode = mode;
        remove_failure(status, "retrieval_degraded");
        if degraded {
            upsert_failure(
                status,
                "retrieval_degraded",
                FailureSeverity::Warning,
                "use lexical recall or restore semantic inference",
            );
        } else {
            recover_if_clean(status);
        }
    });
}

pub fn retrieval_note_is_degraded(note: Option<&str>) -> bool {
    note.is_some_and(|note| {
        note.split(';').any(|part| {
            let part = part.trim_start();
            part.starts_with("semantic:")
                || part.starts_with("rerank unavailable")
                || part.starts_with("rerank misconfigured")
        })
    })
}

fn safe_backend_label(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "endpoint" => "endpoint",
        "rerank-endpoint" => "rerank-endpoint",
        "openvino" => "openvino",
        "coreml" => "coreml",
        "qnn" => "qnn",
        "tensorrt" => "tensorrt",
        "cuda" => "cuda",
        "rocm" => "rocm",
        "vulkan" => "vulkan",
        "metal" => "metal",
        "onnxruntime" => "onnxruntime",
        "ort" => "ort",
        "ryzen-ai" => "ryzen-ai",
        "cpu" => "cpu",
        _ => "other",
    }
    .to_string()
}

fn safe_device_label(value: &str) -> Option<String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cpu" => Some("cpu"),
        "gpu" => Some("gpu"),
        "npu" => Some("npu"),
        "ane" => Some("ane"),
        "tpu" => Some("tpu"),
        "dsp" => Some("dsp"),
        _ => None,
    }
    .map(str::to_string)
}

fn safe_model_label(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()
        && value.len() <= 80
        && !value.contains("://")
        && !value.starts_with('/')
        && !value.contains("..")
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "-._/".contains(character)))
    .then(|| value.to_string())
}

fn safe_maintenance_activity(value: &str) -> Option<String> {
    matches!(value, "proposal" | "review").then(|| value.to_string())
}

/// Record an actual backend outcome. A typed input refusal happens before
/// execution and cannot establish that the selected backend failed or recovered.
/// Its error and incomplete vector coverage remain visible to the caller.
pub fn record_inference_result<T>(
    configured: InferenceMode,
    route: InferenceRoute,
    backend: &str,
    device_class: Option<&str>,
    result: &anyhow::Result<T>,
) {
    let Some(success) = inference_execution_outcome(result) else {
        return;
    };
    let backend = safe_backend_label(backend);
    let device_class = device_class.and_then(safe_device_label);
    let _ = update(|status| {
        apply_inference_attempt(status, configured, route, backend, device_class, success)
    });
}

fn inference_execution_outcome<T>(result: &anyhow::Result<T>) -> Option<bool> {
    match result {
        Err(error) if error.downcast_ref::<crate::embed::InputRefusal>().is_some() => None,
        outcome => Some(outcome.is_ok()),
    }
}

pub fn record_maintenance_attempt(route: InferenceRoute, activity: &str, success: bool) {
    let activity = safe_maintenance_activity(activity);
    let _ = update(|status| apply_maintenance_attempt(status, route, activity, success));
}

fn apply_maintenance_attempt(
    status: &mut RuntimeStatusV1,
    route: InferenceRoute,
    activity: Option<String>,
    success: bool,
) {
    status.maintenance.enabled = true;
    status.maintenance.configured = true;
    status.maintenance.route = Some(route);
    status.maintenance.last_model_activity = activity;
    status.maintenance.last_model_success = Some(success);
    status.maintenance.last_model_at = Some(now());
    remove_failure(status, "maintenance_model_unavailable");
    if success {
        recover_if_clean(status);
    } else {
        upsert_failure(
            status,
            "maintenance_model_unavailable",
            FailureSeverity::Warning,
            "check maintenance model configuration and endpoint policy",
        );
    }
}

fn apply_inference_attempt(
    status: &mut RuntimeStatusV1,
    configured: InferenceMode,
    route: InferenceRoute,
    backend: String,
    device_class: Option<String>,
    success: bool,
) {
    status.inference.configured = configured;
    status.inference.last_used = Some(InferenceAttempt {
        route,
        backend: backend.clone(),
        success,
        observed_at: now(),
    });
    remove_failure(status, "inference_unavailable");
    remove_failure(status, "inference_initialization_failed");
    if success {
        status.inference.selected = Some(BackendSelection {
            backend,
            device_class,
            route: Some(route),
            selected_at: now(),
        });
        recover_if_clean(status);
    } else {
        upsert_failure(
            status,
            "inference_unavailable",
            FailureSeverity::Warning,
            "check inference configuration or use lexical recall",
        );
    }
}

pub fn record_inference_initialization_failure() {
    let _ = update(|status| {
        upsert_failure(
            status,
            "inference_initialization_failed",
            FailureSeverity::Warning,
            "check inference credentials and endpoint policy",
        );
    });
}

pub fn endpoint_route(endpoint: &str) -> InferenceRoute {
    let authority = endpoint
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(endpoint)
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("");
    let host = if let Some(bracketed) = authority.strip_prefix('[') {
        bracketed
            .split_once(']')
            .map(|(host, _)| host)
            .unwrap_or(bracketed)
    } else {
        authority
            .rsplit_once(':')
            .map(|(host, _)| host)
            .unwrap_or(authority)
    };
    if host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
    {
        InferenceRoute::Local
    } else {
        InferenceRoute::Remote
    }
}

fn age(at: u64) -> String {
    let seconds = now().saturating_sub(at);
    match seconds {
        0..=59 => format!("{seconds}s ago"),
        60..=3599 => format!("{}m ago", seconds / 60),
        3600..=86399 => format!("{}h ago", seconds / 3600),
        _ => format!("{}d ago", seconds / 86400),
    }
}

fn coverage_label(status: &RuntimeStatusV1) -> Option<String> {
    if status.inference.configured == InferenceMode::Disabled {
        return None;
    }
    Some(
        match (
            status.retrieval.vector_coverage,
            status.retrieval.embedded,
            status.retrieval.total,
        ) {
            (VectorCoverageState::Complete, Some(_), Some(total)) if total > 0 => {
                "vectors 100%".to_string()
            }
            (VectorCoverageState::Complete, Some(0), Some(0)) => "vectors n/a".to_string(),
            (VectorCoverageState::Partial, Some(embedded), Some(total)) if total > 0 => {
                format!("vectors {}%", embedded.saturating_mul(100) / total)
            }
            (VectorCoverageState::None, _, _) => "vectors 0%".to_string(),
            _ => "vectors ?".to_string(),
        },
    )
}

fn inference_label(status: &RuntimeStatusV1) -> String {
    match status.inference.configured {
        InferenceMode::Unknown => "embed:unknown".to_string(),
        InferenceMode::Disabled => "embed:off".to_string(),
        InferenceMode::Local | InferenceMode::Endpoint => {
            if let Some(last) = &status.inference.last_used {
                let selected = status.inference.selected.as_ref().filter(|selected| {
                    selected.backend == last.backend && selected.route == Some(last.route)
                });
                let device = selected
                    .and_then(|selected| selected.device_class.as_deref())
                    .and_then(safe_device_label);
                let where_ = device.as_deref().unwrap_or(match last.route {
                    InferenceRoute::Local => "local",
                    InferenceRoute::Remote => "remote",
                });
                let result = if last.success { "used" } else { "failed" };
                let selected = if selected.is_some() { " selected," } else { "" };
                format!(
                    "embed:{where_}{selected} {result} {}",
                    age(last.observed_at)
                )
            } else if let Some(selected) = &status.inference.selected {
                match &selected.device_class {
                    Some(device) => format!(
                        "embed:{} selected",
                        safe_device_label(device).as_deref().unwrap_or("local")
                    ),
                    None => format!("embed:{} selected", safe_backend_label(&selected.backend)),
                }
            } else {
                match (
                    status.inference.configured,
                    status.inference.configured_route,
                ) {
                    (InferenceMode::Endpoint, Some(InferenceRoute::Local)) => {
                        "embed:local configured".to_string()
                    }
                    (InferenceMode::Endpoint, Some(InferenceRoute::Remote)) => {
                        "embed:remote configured".to_string()
                    }
                    (InferenceMode::Local, _) => "embed:local configured".to_string(),
                    (InferenceMode::Endpoint, None) => "embed:endpoint configured".to_string(),
                    (InferenceMode::Unknown | InferenceMode::Disabled, _) => unreachable!(),
                }
            }
        }
    }
}

fn truncate_to_width(value: String, width: usize) -> String {
    let width = width.max(1);
    if value.chars().count() <= width {
        return value;
    }
    let mut out: String = value.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

pub fn render_line_with_width(status: &RuntimeStatusV1, width: Option<usize>) -> String {
    if status
        .failures
        .iter()
        .any(|failure| failure.code == "runtime_status_unavailable")
    {
        return truncate_to_width(
            "cfetch ? cached status unavailable · configuration unverified".to_string(),
            width.unwrap_or(usize::MAX),
        );
    }
    let glyph = match status.service.state {
        ServiceState::Ready => "●",
        ServiceState::Degraded => "!",
        ServiceState::Unavailable => "×",
    };
    let mut memory = format!(
        "memory:{}",
        match status.memory_route.mode {
            MemoryRoute::Local => "local",
            MemoryRoute::Serving => "serving",
            MemoryRoute::Remote => "remote",
        }
    );
    if let Some(generation) = status.memory_route.generation {
        memory.push_str(&format!(" g{generation}"));
    }
    if let Some(at) = status.memory_route.last_answer.observed_at {
        let freshness = match status.memory_route.last_answer.state {
            FreshnessState::Fresh => "fresh",
            FreshnessState::Stale => "stale",
            FreshnessState::Unknown => "answer",
        };
        memory.push_str(&format!(" last {freshness} {}", age(at)));
    }
    let retrieval = match status.retrieval.mode {
        RetrievalMode::Lexical => "retrieve:lexical",
        RetrievalMode::Semantic => "retrieve:semantic",
        RetrievalMode::Hybrid => "retrieve:hybrid",
    };
    let mut parts = vec![format!("cfetch {glyph} {memory}"), retrieval.to_string()];
    if let Some(coverage) = coverage_label(status) {
        parts.push(coverage);
    }
    parts.push(inference_label(status));
    let maintenance = if !status.maintenance.enabled {
        "maint:off".to_string()
    } else if status.maintenance.paused {
        "maint:paused".to_string()
    } else if status.maintenance.exceptions > 0
        && status.maintenance.last_outcome.as_deref() == Some("exception")
    {
        format!("maint:exception {}", status.maintenance.exceptions)
    } else if status.maintenance.last_model_success == Some(false) {
        "maint:model-failed".to_string()
    } else if status.maintenance.configured {
        let route = match status.maintenance.route {
            Some(InferenceRoute::Local) => " local",
            Some(InferenceRoute::Remote) => " remote",
            None => "",
        };
        if status.maintenance.candidates == 0 {
            format!("maint:auto{route} idle")
        } else {
            format!("maint:auto{route} {}", status.maintenance.candidates)
        }
    } else if status.maintenance.candidates > 0 {
        format!("maint:setup {}", status.maintenance.candidates)
    } else {
        "maint:setup".to_string()
    };
    parts.push(maintenance);
    let line = parts.join(" · ");
    truncate_to_width(line, width.unwrap_or(usize::MAX))
}

pub fn render_line(status: &RuntimeStatusV1) -> String {
    let width = std::env::var("COLUMNS")
        .ok()
        .and_then(|raw| raw.parse().ok());
    render_line_with_width(status, width)
}

pub fn mcp_json() -> anyhow::Result<String> {
    let mut status = load_cached();
    loop {
        let json = serde_json::to_string(&status)?;
        if json.len() <= MCP_MAX_BYTES {
            return Ok(json);
        }
        if status.failures.pop().is_none() {
            anyhow::bail!("RuntimeStatusV1 exceeds its {MCP_MAX_BYTES}-byte MCP bound");
        }
    }
}

/// Fingerprint of model-relevant transitions. Generation, timestamps,
/// maintenance counts, and successful-use recency are intentionally absent,
/// so normal catalog churn cannot spam a coding session. Maintenance mode
/// changes are included because they change whether staged evidence will be
/// folded into Markdown without intervention.
pub fn transition_fingerprint(status: &RuntimeStatusV1) -> String {
    let failures = status
        .failures
        .iter()
        .map(|failure| format!("{}:{:?}", failure.code, failure.severity))
        .collect::<Vec<_>>()
        .join(",");
    let selected = status
        .inference
        .selected
        .as_ref()
        .map(|selected| {
            format!(
                "{}:{}:{:?}",
                selected.backend,
                selected.device_class.as_deref().unwrap_or(""),
                selected.route
            )
        })
        .unwrap_or_default();
    let last_route = status
        .inference
        .last_used
        .as_ref()
        .map(|last| format!("{:?}", last.route))
        .unwrap_or_default();
    format!(
        "{:?}|{:?}|{:?}|{:?}|{}|{}|{}:{}:{:?}:{}|{}",
        status.service.state,
        status.memory_route.mode,
        status.inference.configured,
        status.inference.configured_route,
        selected,
        last_route,
        status.maintenance.enabled,
        status.maintenance.configured,
        status.maintenance.route,
        status.maintenance.paused,
        failures,
    )
}

pub fn adaptation_context(status: &RuntimeStatusV1) -> Option<String> {
    let codes: Vec<&str> = status
        .failures
        .iter()
        .map(|failure| failure.code.as_str())
        .collect();
    let text = if codes.contains(&"config_unusable") {
        "[cfetch degraded: memory configuration is unusable; do not assume recall or capture is active. Run `cfetch selfcheck`.]"
    } else if codes.contains(&"runtime_status_unavailable") {
        "[cfetch cached status unavailable: service and inference configuration are unverified. This does not establish that recall is down. Run `cfetch status` to refresh runtime metadata.]"
    } else if codes.contains(&"remote_unavailable") || codes.contains(&"memory_unavailable") {
        "[cfetch degraded: the configured memory route is unavailable; do not claim memory-backed results until it recovers.]"
    } else if codes.contains(&"memory_stale") {
        "[cfetch degraded: the last served memory answer was stale; verify freshness before relying on recent changes.]"
    } else if codes.contains(&"maintenance_exception") {
        "[cfetch maintenance exception: captured evidence remains safe in staging and no stale Markdown was overwritten. Use `cfetch maintain history` when debugging matters.]"
    } else if codes.contains(&"maintenance_unconfigured") {
        "[cfetch maintenance is not configured: recall and capture still work, but staged evidence is not being folded into Markdown automatically.]"
    } else if codes.contains(&"maintenance_model_unavailable") {
        "[cfetch maintenance model is unavailable: captured evidence remains staged and Markdown is unchanged; recall continues normally.]"
    } else if codes.contains(&"retrieval_degraded")
        || codes.contains(&"vector_coverage_none")
        || codes.contains(&"vector_coverage_partial")
        || codes.contains(&"inference_unavailable")
        || codes.contains(&"inference_misconfigured")
        || codes.contains(&"inference_initialization_failed")
    {
        "[cfetch degraded: semantic retrieval is incomplete or unavailable; use lexical recall and state that limitation when it matters.]"
    } else {
        return None;
    };
    Some(text.chars().take(240).collect())
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct HookNotice {
    pub system_message: Option<String>,
    pub additional_context: Option<String>,
}

pub fn codex_hook_notice(
    state: &mut crate::session_state::SessionState,
    status: &RuntimeStatusV1,
    session_start: bool,
    observed_at: u64,
) -> HookNotice {
    let fingerprint = transition_fingerprint(status);
    let changed = state.runtime_status_fingerprint.as_deref() != Some(&fingerprint);
    let degraded = adaptation_context(status);
    let has_degradation = degraded.is_some();
    let repeat_due = has_degradation
        && state
            .runtime_status_last_notice
            .is_some_and(|last| observed_at.saturating_sub(last) >= 300);
    let show = session_start || changed || repeat_due;
    let notice = HookNotice {
        system_message: show.then(|| render_line_with_width(status, Some(180))),
        // Model context is only for a new adaptation. A recurring UI warning
        // must not repeatedly spend context tokens.
        additional_context: (session_start || changed).then_some(degraded).flatten(),
    };
    state.runtime_status_fingerprint = Some(fingerprint);
    if show && has_degradation {
        state.runtime_status_last_notice = Some(observed_at);
    }
    notice
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_embeddings_are_neutral_lexical_status() {
        let status = RuntimeStatusV1::default();
        let line = render_line_with_width(&status, Some(200));
        assert!(line.contains("retrieve:lexical"), "{line}");
        assert!(line.contains("embed:off"), "{line}");
        assert!(
            !line.contains("vectors 0%"),
            "disabled is not failed coverage: {line}"
        );
        assert!(status.failures.is_empty());
    }

    #[test]
    fn cached_freshness_is_always_past_tense() {
        let mut status = RuntimeStatusV1::default();
        status.memory_route.last_answer = LastAnswerStatus {
            state: FreshnessState::Fresh,
            observed_at: Some(now().saturating_sub(8)),
        };
        let line = render_line_with_width(&status, Some(200));
        assert!(line.contains("last fresh"), "{line}");
        assert!(
            !line.contains(" fresh ·"),
            "cached truth must not claim present freshness: {line}"
        );
    }

    #[test]
    fn partial_coverage_is_explicitly_degraded() {
        let mut status = RuntimeStatusV1::default();
        status.inference.configured = InferenceMode::Endpoint;
        set_vector_coverage(&mut status, 25, 100);
        normalize(&mut status);
        assert_eq!(status.service.state, ServiceState::Degraded);
        assert_eq!(
            status.retrieval.vector_coverage,
            VectorCoverageState::Partial
        );
        assert!(
            status
                .failures
                .iter()
                .any(|failure| failure.code == "vector_coverage_partial")
        );
        assert!(render_line_with_width(&status, Some(200)).contains("vectors 25%"));
    }

    #[test]
    fn warning_after_critical_recovery_is_degraded_not_unavailable() {
        let mut status = RuntimeStatusV1::default();
        status.service.state = ServiceState::Unavailable;
        upsert_failure(
            &mut status,
            "vector_coverage_partial",
            FailureSeverity::Warning,
            "ignored",
        );
        normalize(&mut status);
        assert_eq!(status.service.state, ServiceState::Degraded);
    }

    #[test]
    fn live_daemon_observation_updates_display_without_masking_critical_failures() {
        let mut status = RuntimeStatusV1::default();
        apply_daemon_observation(&mut status, false);
        assert_eq!(status.service.state, ServiceState::Degraded);
        assert!(
            status
                .failures
                .iter()
                .any(|failure| failure.code == "daemon_unavailable")
        );

        upsert_failure(
            &mut status,
            "memory_unavailable",
            FailureSeverity::Critical,
            "ignored",
        );
        apply_daemon_observation(&mut status, true);
        assert_eq!(status.service.state, ServiceState::Unavailable);
        assert!(
            !status
                .failures
                .iter()
                .any(|failure| failure.code == "daemon_unavailable")
        );
    }

    #[test]
    fn empty_catalog_has_no_missing_vector_work() {
        let mut status = RuntimeStatusV1::default();
        status.inference.configured = InferenceMode::Endpoint;
        set_vector_coverage(&mut status, 0, 0);
        normalize(&mut status);
        assert_eq!(
            status.retrieval.vector_coverage,
            VectorCoverageState::Complete
        );
        assert!(status.failures.is_empty());
        assert!(render_line_with_width(&status, Some(200)).contains("vectors n/a"));
    }

    #[test]
    fn intentional_precision_filtering_is_not_runtime_degradation() {
        assert!(!retrieval_note_is_degraded(Some(
            "precision gate dropped 2 hit(s) carrying under 2 terms"
        )));
        assert!(retrieval_note_is_degraded(Some(
            "semantic: query embedding failed — answering lexically"
        )));
        assert!(retrieval_note_is_degraded(Some(
            "rerank unavailable (timeout) — answering in retrieval order"
        )));
    }

    #[test]
    fn configured_selected_and_last_used_are_distinct_truths() {
        let mut status = RuntimeStatusV1::default();
        status.inference.configured = InferenceMode::Endpoint;
        assert!(status.inference.selected.is_none());
        assert!(status.inference.last_used.is_none());
        status.inference.last_used = Some(InferenceAttempt {
            route: InferenceRoute::Remote,
            backend: "endpoint".into(),
            success: false,
            observed_at: 1,
        });
        assert!(
            status.inference.selected.is_none(),
            "a failed attempt never selects a backend"
        );
        status.inference.selected = Some(BackendSelection {
            backend: "vulkan".into(),
            device_class: Some("gpu".into()),
            route: Some(InferenceRoute::Local),
            selected_at: 2,
        });
        assert_eq!(
            status
                .inference
                .selected
                .as_ref()
                .unwrap()
                .device_class
                .as_deref(),
            Some("gpu")
        );
    }

    #[test]
    fn older_v1_selection_without_route_still_deserializes() {
        let mut json = serde_json::to_value(RuntimeStatusV1::default()).unwrap();
        json["inference"]["selected"] = serde_json::json!({
            "backend": "openvino",
            "device_class": "npu",
            "selected_at": 2
        });

        let status: RuntimeStatusV1 = serde_json::from_value(json).unwrap();
        assert_eq!(status.inference.selected.unwrap().route, None);
    }

    #[test]
    fn older_v1_maintenance_status_still_deserializes() {
        let mut json = serde_json::to_value(RuntimeStatusV1::default()).unwrap();
        json["maintenance"] = serde_json::json!({
            "pending": 2,
            "applied": 1
        });

        let status: RuntimeStatusV1 = serde_json::from_value(json).unwrap();
        assert_eq!(status.maintenance.pending, 2);
        assert_eq!(status.maintenance.applied, 1);
        assert!(!status.maintenance.enabled);
        assert!(!status.maintenance.configured);
        assert!(status.maintenance.route.is_none());
        assert!(status.maintenance.model.is_none());
        assert!(status.maintenance.last_model_activity.is_none());
        assert!(status.maintenance.last_model_success.is_none());
        assert!(status.maintenance.last_outcome.is_none());
    }

    #[test]
    fn maintenance_line_distinguishes_off_setup_auto_pause_and_exception() {
        let mut status = RuntimeStatusV1::default();
        let line = |status: &RuntimeStatusV1| render_line_with_width(status, Some(240));
        assert!(line(&status).contains("maint:off"));

        status.maintenance.enabled = true;
        status.maintenance.candidates = 2;
        assert!(line(&status).contains("maint:setup 2"));

        status.maintenance.configured = true;
        status.maintenance.route = Some(InferenceRoute::Remote);
        status.maintenance.candidates = 0;
        assert!(line(&status).contains("maint:auto remote idle"));

        status.maintenance.paused = true;
        assert!(line(&status).contains("maint:paused"));

        status.maintenance.paused = false;
        status.maintenance.exceptions = 3;
        status.maintenance.last_outcome = Some("exception".into());
        assert!(line(&status).contains("maint:exception 3"));

        status.maintenance.exceptions = 0;
        status.maintenance.last_outcome = Some("applied".into());
        status.maintenance.last_model_success = Some(false);
        assert!(line(&status).contains("maint:model-failed"));
    }

    #[test]
    fn maintenance_model_attempt_does_not_rewrite_embedding_truth() {
        let mut status = RuntimeStatusV1::default();
        apply_maintenance_attempt(
            &mut status,
            InferenceRoute::Remote,
            Some("proposal".into()),
            false,
        );
        normalize(&mut status);

        assert_eq!(status.inference.configured, InferenceMode::Disabled);
        assert_eq!(
            status.maintenance.last_model_activity.as_deref(),
            Some("proposal")
        );
        assert_eq!(status.maintenance.last_model_success, Some(false));
        assert!(
            status
                .failures
                .iter()
                .any(|failure| failure.code == "maintenance_model_unavailable")
        );
        assert!(
            adaptation_context(&status)
                .unwrap()
                .contains("Markdown is unchanged")
        );

        apply_maintenance_attempt(
            &mut status,
            InferenceRoute::Remote,
            Some("review".into()),
            true,
        );
        normalize(&mut status);
        assert_eq!(
            status.maintenance.last_model_activity.as_deref(),
            Some("review")
        );
        assert_eq!(status.maintenance.last_model_success, Some(true));
        assert!(
            !status
                .failures
                .iter()
                .any(|failure| failure.code == "maintenance_model_unavailable")
        );
    }

    #[test]
    fn failed_memory_attempt_preserves_last_successful_answer() {
        let mut status = RuntimeStatusV1::default();
        status.memory_route.generation = Some(41);
        status.memory_route.last_answer = LastAnswerStatus {
            state: FreshnessState::Fresh,
            observed_at: Some(123),
        };

        apply_memory_answer(&mut status, MemoryRoute::Remote, Some(42), None, false);
        normalize(&mut status);

        assert_eq!(status.memory_route.generation, Some(41));
        assert_eq!(status.memory_route.last_answer.observed_at, Some(123));
        assert_eq!(status.memory_route.last_answer.state, FreshnessState::Fresh);
        assert_eq!(status.service.state, ServiceState::Unavailable);
    }

    #[test]
    fn input_refusal_preserves_the_last_actual_backend_outcome() {
        for prior_success in [true, false] {
            let mut status = RuntimeStatusV1::default();
            apply_inference_attempt(
                &mut status,
                InferenceMode::Local,
                InferenceRoute::Local,
                "other".into(),
                Some("cpu".into()),
                prior_success,
            );
            let before = serde_json::to_value(&status).unwrap();
            let refusal: anyhow::Result<()> = Err(anyhow::Error::new(crate::embed::InputRefusal {
                token_count: crate::embedding_profile::MAX_TOKENS + 1,
            })
            .context("background document embedding"));
            if let Some(success) = inference_execution_outcome(&refusal) {
                apply_inference_attempt(
                    &mut status,
                    InferenceMode::Local,
                    InferenceRoute::Local,
                    "other".into(),
                    Some("cpu".into()),
                    success,
                );
            }
            assert_eq!(serde_json::to_value(&status).unwrap(), before);
        }
    }

    #[test]
    fn only_typed_input_refusal_is_excluded_from_execution_telemetry() {
        assert_eq!(
            inference_execution_outcome(&Ok::<_, anyhow::Error>(())),
            Some(true)
        );
        let refusal_text = crate::embed::InputRefusal {
            token_count: crate::embedding_profile::MAX_TOKENS + 1,
        }
        .to_string();
        for message in ["model execution failed", refusal_text.as_str()] {
            let failure: anyhow::Result<()> = Err(anyhow::anyhow!("{message}"));
            assert_eq!(inference_execution_outcome(&failure), Some(false));
        }
    }

    #[test]
    fn actual_inference_route_does_not_rewrite_mixed_configured_intent() {
        let mut status = RuntimeStatusV1::default();
        status.inference.configured = InferenceMode::Endpoint;
        status.inference.configured_route = None;

        apply_inference_attempt(
            &mut status,
            InferenceMode::Endpoint,
            InferenceRoute::Remote,
            "endpoint".into(),
            None,
            true,
        );

        assert_eq!(status.inference.configured_route, None);
        assert_eq!(
            status.inference.last_used.as_ref().map(|last| last.route),
            Some(InferenceRoute::Remote)
        );
    }

    #[test]
    fn detection_strings_and_endpoints_cannot_leak_into_labels() {
        assert_eq!(safe_backend_label("https://private.invalid"), "other");
        assert_eq!(safe_backend_label("10.20.30.40"), "other");
        assert_eq!(safe_device_label("NPU"), Some("npu".into()));
        assert_eq!(safe_device_label("NPU /sys/private/device"), None);
        assert_eq!(
            endpoint_route("http://127.0.0.1:8080/v1"),
            InferenceRoute::Local
        );
        assert_eq!(
            endpoint_route("https://example.invalid/v1"),
            InferenceRoute::Remote
        );
        let json = serde_json::to_string(&RuntimeStatusV1::default()).unwrap();
        assert!(!json.contains("://"));
        assert!(!json.contains("token_file"));
    }

    #[test]
    fn line_is_terminal_bounded() {
        let status = RuntimeStatusV1::default();
        let line = render_line_with_width(&status, Some(48));
        assert!(line.chars().count() <= 48, "{line}");
        assert!(line.ends_with('…'));
        assert_eq!(render_line_with_width(&status, Some(1)), "…");
    }

    #[test]
    fn local_accelerator_is_named_only_after_a_successful_selection() {
        let mut status = RuntimeStatusV1::default();
        status.inference.configured = InferenceMode::Local;
        status.inference.configured_route = Some(InferenceRoute::Local);
        status.inference.selected = Some(BackendSelection {
            backend: "openvino".into(),
            device_class: Some("npu".into()),
            route: Some(InferenceRoute::Local),
            selected_at: now(),
        });
        status.inference.last_used = Some(InferenceAttempt {
            route: InferenceRoute::Local,
            backend: "openvino".into(),
            success: true,
            observed_at: now(),
        });
        let line = render_line_with_width(&status, Some(200));
        assert!(line.contains("embed:npu selected, used"), "{line}");
    }

    #[test]
    fn prior_local_selection_is_not_misattributed_to_remote_use() {
        let mut status = RuntimeStatusV1::default();
        status.inference.configured = InferenceMode::Endpoint;
        status.inference.selected = Some(BackendSelection {
            backend: "openvino".into(),
            device_class: Some("npu".into()),
            route: Some(InferenceRoute::Local),
            selected_at: 1,
        });
        status.inference.last_used = Some(InferenceAttempt {
            route: InferenceRoute::Remote,
            backend: "endpoint".into(),
            success: true,
            observed_at: now(),
        });

        let line = render_line_with_width(&status, Some(200));
        assert!(line.contains("embed:remote used"), "{line}");
        assert!(!line.contains("npu"), "{line}");
    }

    #[test]
    fn incomplete_endpoint_intent_is_configured_but_not_selected() {
        let mut cfg = Config::default();
        cfg.embeddings.enabled = true;
        let mut status = RuntimeStatusV1::default();
        apply_config(&mut status, &cfg);
        normalize(&mut status);
        assert_eq!(status.inference.configured, InferenceMode::Endpoint);
        assert_eq!(status.inference.configured_route, None);
        assert!(status.inference.selected.is_none());
        assert!(
            status
                .failures
                .iter()
                .any(|failure| failure.code == "inference_misconfigured")
        );
    }

    #[test]
    fn mixed_endpoint_routes_do_not_claim_one_configured_location() {
        let mut cfg = Config::default();
        cfg.embeddings.enabled = true;
        cfg.embeddings.endpoint = "http://127.0.0.1:8080".into();
        cfg.rerank.enabled = true;
        cfg.rerank.endpoint = "https://example.invalid".into();
        cfg.rerank.model = "reranker".into();
        let mut status = RuntimeStatusV1::default();
        apply_config(&mut status, &cfg);
        assert_eq!(status.inference.configured, InferenceMode::Endpoint);
        assert_eq!(status.inference.configured_route, None);
        assert!(render_line_with_width(&status, Some(200)).contains("embed:endpoint configured"));
    }

    #[test]
    fn maintenance_model_route_does_not_masquerade_as_embeddings() {
        let mut cfg = Config::default();
        cfg.maintenance.endpoint = "https://example.invalid/v1".into();
        cfg.maintenance.model = "memory-maintainer-v1".into();
        let mut status = RuntimeStatusV1::default();
        apply_config(&mut status, &cfg);
        status.maintenance.enabled = cfg.maintenance.enabled;
        status.maintenance.configured = cfg.maintenance.configured();
        status.maintenance.route = Some(endpoint_route(&cfg.maintenance.endpoint));
        status.maintenance.model = safe_model_label(&cfg.maintenance.model);
        normalize(&mut status);

        assert_eq!(status.inference.configured, InferenceMode::Disabled);
        assert_eq!(status.maintenance.route, Some(InferenceRoute::Remote));
        let line = render_line_with_width(&status, Some(240));
        assert!(line.contains("embed:off"), "{line}");
        assert!(line.contains("maint:auto remote idle"), "{line}");
        assert!(!line.contains("vectors"), "{line}");
    }

    #[test]
    fn disabling_inference_clears_old_inference_degradation() {
        let mut status = RuntimeStatusV1::default();
        status.service.state = ServiceState::Degraded;
        upsert_failure(
            &mut status,
            "retrieval_degraded",
            FailureSeverity::Warning,
            "ignored",
        );
        upsert_failure(
            &mut status,
            "inference_initialization_failed",
            FailureSeverity::Warning,
            "ignored",
        );
        apply_config(&mut status, &Config::default());
        normalize(&mut status);
        assert_eq!(status.service.state, ServiceState::Ready);
        assert!(status.failures.is_empty());
    }

    #[test]
    fn stale_answer_has_an_actionable_degraded_context() {
        let mut status = RuntimeStatusV1::default();
        status.memory_route.last_answer = LastAnswerStatus {
            state: FreshnessState::Stale,
            observed_at: Some(now()),
        };
        status.service.state = ServiceState::Degraded;
        upsert_failure(
            &mut status,
            "memory_stale",
            FailureSeverity::Warning,
            "ignored",
        );
        normalize(&mut status);
        assert!(adaptation_context(&status).unwrap().contains("was stale"));
    }

    #[test]
    fn maintenance_context_preserves_safe_staging_and_explains_setup() {
        let mut status = RuntimeStatusV1::default();
        upsert_failure(
            &mut status,
            "maintenance_exception",
            FailureSeverity::Warning,
            "ignored",
        );
        assert!(
            adaptation_context(&status)
                .unwrap()
                .contains("no stale Markdown was overwritten")
        );

        remove_failure(&mut status, "maintenance_exception");
        upsert_failure(
            &mut status,
            "maintenance_unconfigured",
            FailureSeverity::Warning,
            "ignored",
        );
        assert!(
            adaptation_context(&status)
                .unwrap()
                .contains("not being folded into Markdown automatically")
        );
    }

    #[test]
    fn missing_corrupt_and_wrong_schema_cache_are_explicitly_unavailable() {
        for contents in [None, Some("{broken"), Some(r#"{"schema_version":999}"#)] {
            let dir = tempfile::tempdir().unwrap();
            let path = snapshot_path_in(dir.path());
            if let Some(contents) = contents {
                std::fs::write(&path, contents).unwrap();
            }
            let status = load_cached_in(dir.path());
            assert_eq!(status.service.state, ServiceState::Unavailable);
            assert_eq!(status.inference.configured, InferenceMode::Unknown);
            assert_eq!(
                status.memory_route.last_answer.state,
                FreshnessState::Unknown
            );
            assert_eq!(status.memory_route.generation, None);
            assert_eq!(status.failures[0].code, "runtime_status_unavailable");
            assert!(render_line_with_width(&status, None).contains("configuration unverified"));
            assert!(
                adaptation_context(&status)
                    .unwrap()
                    .contains("does not establish that recall is down")
            );
            assert_eq!(
                std::fs::read_to_string(path).ok().as_deref(),
                contents,
                "a cache read must not create or repair the snapshot"
            );
        }
    }

    #[test]
    fn telemetry_does_not_invent_configuration_and_explicit_refresh_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let status = update_in(dir.path(), |status| {
            status.service.state = ServiceState::Ready;
            status.memory_route.generation = Some(7);
        })
        .unwrap();
        assert_eq!(status.service.state, ServiceState::Unavailable);
        assert_eq!(status.inference.configured, InferenceMode::Unknown);
        assert_eq!(status.memory_route.generation, Some(7));
        let mut refreshed = load_cached_in(dir.path());
        apply_config(&mut refreshed, &Config::default());
        normalize(&mut refreshed);
        assert!(refreshed.failures.is_empty());
        assert_eq!(refreshed.inference.configured, InferenceMode::Disabled);
        assert_eq!(refreshed.service.state, ServiceState::Ready);
    }

    #[test]
    fn cached_line_read_path_stays_under_the_status_line_budget() {
        let dir = tempfile::tempdir().unwrap();
        store_in(dir.path(), &RuntimeStatusV1::default()).unwrap();
        let mut samples = (0..40)
            .map(|_| {
                let start = std::time::Instant::now();
                let status = load_cached_in(dir.path());
                let _ = render_line_with_width(&status, Some(120));
                start.elapsed()
            })
            .collect::<Vec<_>>();
        samples.sort();
        let p95 = samples[(samples.len() * 95 / 100).saturating_sub(1)];
        assert!(
            p95 < std::time::Duration::from_millis(25),
            "p95 was {p95:?}"
        );
    }

    #[test]
    fn corrupt_cached_strings_are_not_rendered_or_returned_over_mcp() {
        let mut status = RuntimeStatusV1::default();
        status.memory_route.origin_label = "https://private.invalid".into();
        status.inference.selected = Some(BackendSelection {
            backend: "https://private.invalid/token".into(),
            device_class: Some("/sys/secret/device".into()),
            route: None,
            selected_at: 1,
        });
        status.failures.push(RuntimeFailure {
            code: "https://private.invalid".into(),
            severity: FailureSeverity::Critical,
            action: "read /secret/token".into(),
        });
        status.maintenance.configured = true;
        status.maintenance.model = Some("https://private.invalid/model".into());
        normalize(&mut status);
        let json = serde_json::to_string(&status).unwrap();
        assert!(!json.contains("://"), "{json}");
        assert!(!json.contains("/sys/"), "{json}");
        assert!(!json.contains("/secret/"), "{json}");
        assert!(status.maintenance.model.is_none());
        assert!(
            status.failures.is_empty(),
            "unknown failure codes are refused"
        );
    }

    #[test]
    fn fingerprint_ignores_generation_and_observation_churn() {
        let a = RuntimeStatusV1::default();
        let mut b = a.clone();
        b.observed_at += 100;
        b.memory_route.generation = Some(999);
        b.memory_route.last_answer.observed_at = Some(999);
        b.maintenance.pending = 7;
        b.maintenance.candidates = 11;
        b.maintenance.history_events = 19;
        b.maintenance.exceptions = 3;
        b.maintenance.last_event_at = Some(999);
        b.maintenance.last_outcome = Some("exception".into());
        b.retrieval.mode = RetrievalMode::Hybrid;
        b.retrieval.vector_coverage = VectorCoverageState::Complete;
        assert_eq!(transition_fingerprint(&a), transition_fingerprint(&b));
        b.service.state = ServiceState::Degraded;
        assert_ne!(transition_fingerprint(&a), transition_fingerprint(&b));
    }

    #[test]
    fn fingerprint_notices_maintenance_mode_changes_but_not_counts() {
        let a = RuntimeStatusV1::default();
        let mut enabled = a.clone();
        enabled.maintenance.enabled = true;
        assert_ne!(transition_fingerprint(&a), transition_fingerprint(&enabled));

        let mut configured = enabled.clone();
        configured.maintenance.configured = true;
        assert_ne!(
            transition_fingerprint(&enabled),
            transition_fingerprint(&configured)
        );

        let configured_without_route = transition_fingerprint(&configured);
        configured.maintenance.route = Some(InferenceRoute::Remote);
        assert_ne!(
            configured_without_route,
            transition_fingerprint(&configured)
        );
        let mut paused = configured.clone();
        paused.maintenance.paused = true;
        assert_ne!(
            transition_fingerprint(&configured),
            transition_fingerprint(&paused)
        );

        let paused_mode = transition_fingerprint(&paused);
        paused.maintenance.candidates = 12;
        paused.maintenance.history_events = 40;
        assert_eq!(paused_mode, transition_fingerprint(&paused));
    }

    #[test]
    fn fingerprint_notices_a_successful_inference_route_change() {
        let mut local = RuntimeStatusV1::default();
        local.inference.configured = InferenceMode::Endpoint;
        local.inference.last_used = Some(InferenceAttempt {
            route: InferenceRoute::Local,
            backend: "endpoint".into(),
            success: true,
            observed_at: 10,
        });
        let mut remote = local.clone();
        remote.inference.last_used.as_mut().unwrap().route = InferenceRoute::Remote;
        remote.inference.last_used.as_mut().unwrap().observed_at = 11;
        assert_ne!(
            transition_fingerprint(&local),
            transition_fingerprint(&remote)
        );
    }

    #[test]
    fn codex_healthy_notice_has_no_model_context_and_failure_is_transition_bounded() {
        let mut session = crate::session_state::SessionState::default();
        let healthy = RuntimeStatusV1::default();
        let start = codex_hook_notice(&mut session, &healthy, true, 100);
        assert!(start.system_message.is_some());
        assert!(start.additional_context.is_none());
        let quiet = codex_hook_notice(&mut session, &healthy, false, 101);
        assert_eq!(quiet, HookNotice::default());

        let mut degraded = healthy.clone();
        degraded.service.state = ServiceState::Degraded;
        upsert_failure(
            &mut degraded,
            "retrieval_degraded",
            FailureSeverity::Warning,
            "use lexical recall",
        );
        let transition = codex_hook_notice(&mut session, &degraded, false, 102);
        assert!(transition.system_message.is_some());
        assert!(
            transition
                .additional_context
                .as_ref()
                .is_some_and(|text| text.len() <= 240)
        );
        assert_eq!(
            codex_hook_notice(&mut session, &degraded, false, 200),
            HookNotice::default()
        );
        let repeat = codex_hook_notice(&mut session, &degraded, false, 402);
        assert!(repeat.system_message.is_some());
        assert!(repeat.additional_context.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn snapshot_is_private_and_atomic() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempfile::tempdir().unwrap();
        let status = RuntimeStatusV1::default();
        store_in(dir.path(), &status).unwrap();
        let path = snapshot_path_in(dir.path());
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        store_in(dir.path(), &status).unwrap();
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}
