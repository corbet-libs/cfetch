//! Read-only, evidence-first diagnostics for people and support tooling.
//!
//! `status` is the compact operational contract consumed by hooks and MCP.
//! `doctor` is intentionally wider: it explains how that state was reached,
//! which hardware was detected, what this binary can bind, which model
//! contract is configured, and whether remembered peers answer right now.
//! It never treats discovery as selection or a past selection as live device
//! utilization. The normal report does not call a model. `doctor --deep` is
//! the explicit exception: it runs the configured retrieval models over safe
//! temporary data and includes their actual outputs.

use std::path::Path;
use std::time::Duration;

use serde::Serialize;

use crate::config::Config;
use crate::{
    daemon, embed, embedding_profile, grant, hardware, heartbeat, index, maintenance,
    maintenance_model, net, paths, rerank, retrieval_fixture, runtime_status, variant, vectors,
};

pub const SCHEMA_VERSION: u32 = 1;
const PEER_PROBE_TIMEOUT: Duration = Duration::from_millis(1_500);
const MAX_PROBED_PEERS: usize = 16;

#[derive(Debug, Clone, Serialize)]
pub struct ReportV1 {
    pub schema_version: u32,
    pub observed_at: u64,
    pub build: BuildDiagnostic,
    pub platform: PlatformDiagnostic,
    pub config: ConfigDiagnostic,
    pub catalog: CatalogDiagnostic,
    pub daemon: DaemonDiagnostic,
    pub memory: MemoryDiagnostic,
    pub inference: InferenceDiagnostic,
    pub hardware: Vec<HardwareDiagnostic>,
    pub topology: TopologyDiagnostic,
    pub integrations: IntegrationDiagnostic,
    pub runtime: runtime_status::RuntimeStatusV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval_probe: Option<retrieval_fixture::RetrievalFixtureReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval_probe_error: Option<String>,
    pub findings: Vec<Finding>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuildDiagnostic {
    pub version: String,
    pub variant: Option<String>,
    pub recommended_release_variant: Option<String>,
    pub inference_backend: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlatformDiagnostic {
    pub os: String,
    pub arch: String,
    pub x86_64_level: Option<String>,
    pub hardware_detection: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigDiagnostic {
    pub loaded: bool,
    pub brain_root: Option<String>,
    pub error: Option<String>,
}

/// The derived catalogue's state — a measurement, not an assumption: an
/// installation that never scanned answers every recall from nothing, and
/// one lagging the tree answers from a version of it that no longer exists.
/// Doctor reported neither until this field existed.
#[derive(Debug, Clone, Serialize)]
pub struct CatalogDiagnostic {
    /// `current`, `stale`, `never_scanned`, `unavailable`, or `remote`
    /// (a none-tier client holds no local catalog by design).
    pub state: String,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DaemonDiagnostic {
    pub state: DaemonState,
    pub version: Option<String>,
    pub version_matches_cli: Option<bool>,
    pub endpoint_id: Option<String>,
    /// A running daemon owns the one persistent iroh endpoint. This says the
    /// endpoint is bound, not that any particular remote peer is connected.
    pub network_endpoint_bound: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonState {
    Running,
    Stopped,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryDiagnostic {
    pub route: String,
    pub origin: String,
    pub generation: Option<u64>,
    pub vector_coverage: CoverageDiagnostic,
    pub shared_vector_artifacts: Option<usize>,
    pub peer_artifacts: PeerArtifactDiagnostic,
}

#[derive(Debug, Clone, Serialize)]
pub struct PeerArtifactDiagnostic {
    pub transport: String,
    pub state: String,
    pub authorized_routes: usize,
    pub route_order: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CoverageDiagnostic {
    pub state: String,
    pub embedded: Option<u64>,
    pub total: Option<u64>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InferenceDiagnostic {
    pub build_backend: String,
    pub embeddings: ModelDiagnostic,
    pub reranker: RerankerDiagnostic,
    pub maintenance: MaintenanceModelDiagnostic,
    pub selected: Option<runtime_status::BackendSelection>,
    pub last_used: Option<runtime_status::InferenceAttempt>,
    pub utilization: UtilizationDiagnostic,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelDiagnostic {
    pub enabled: bool,
    pub backend: String,
    pub route: Option<String>,
    pub model: String,
    pub model_revision: String,
    pub artifact_policy: String,
    pub profile_id: String,
    pub dimensions: usize,
    pub vector_encoding: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankerDiagnostic {
    pub enabled: bool,
    pub backend: String,
    pub route: Option<String>,
    pub model: Option<String>,
    pub candidates: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MaintenanceModelDiagnostic {
    pub enabled: bool,
    pub configured: bool,
    pub state: String,
    pub route: Option<String>,
    pub proposal_model: Option<String>,
    pub review_model: Option<String>,
    pub candidates: u64,
    pub history_events: u64,
    pub unreadable_history: u64,
    pub exceptions: u64,
    pub last_outcome: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UtilizationDiagnostic {
    pub state: UtilizationState,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UtilizationState {
    NotReported,
}

#[derive(Debug, Clone, Serialize)]
pub struct HardwareDiagnostic {
    pub device: String,
    pub token: String,
    pub class: String,
    pub evidence: String,
    pub architecturally_usable: bool,
    pub unusable_reason: Option<String>,
    pub caveat: Option<String>,
    pub binding: BindingState,
    pub selected: bool,
    pub utilization: DeviceUtilizationState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingState {
    NotSupportedByBuild,
    AvailableNotSelected,
    Selected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceUtilizationState {
    NotSelected,
    NotReported,
}

#[derive(Debug, Clone, Serialize)]
pub struct TopologyDiagnostic {
    pub local_endpoint_id: Option<String>,
    pub joined_origins: Vec<JoinedOriginDiagnostic>,
    pub granted_peers: Vec<GrantedPeerDiagnostic>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JoinedOriginDiagnostic {
    pub endpoint_id: String,
    pub slice: String,
    pub mode: String,
    pub joined_at: u64,
    pub reachability: Reachability,
    pub generation: Option<u64>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Reachability {
    Reachable,
    Unreachable,
    NotProbed,
}

#[derive(Debug, Clone, Serialize)]
pub struct GrantedPeerDiagnostic {
    pub endpoint_id: Option<String>,
    pub slice: String,
    pub mode: String,
    pub state: String,
    pub expires_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct IntegrationDiagnostic {
    pub hooks: Vec<HookDiagnostic>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct HookDiagnostic {
    pub name: String,
    pub registered: bool,
    pub state: String,
    pub last_ok: Option<u64>,
    pub consecutive_failures: u32,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub code: String,
    pub severity: FindingSeverity,
    pub summary: String,
    pub action: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingSeverity {
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayTone {
    Heading,
    Normal,
    Muted,
    Good,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayLine {
    pub tone: DisplayTone,
    pub text: String,
}

impl DisplayLine {
    fn new(tone: DisplayTone, text: impl Into<String>) -> Self {
        Self {
            tone,
            text: text.into(),
        }
    }
}

/// Collects a current diagnostic report. Network probes are bounded and may
/// be disabled for an offline bug report or a low-latency render.
pub fn gather(probe_network: bool) -> ReportV1 {
    let state_dir = paths::state_dir();
    match Config::load() {
        Ok(cfg) => gather_with(&cfg, &state_dir, None, probe_network),
        Err(error) => gather_without_config(&state_dir, error, probe_network),
    }
}

/// Runs the real embedding and optional reranking routes against a disposable
/// fixture, then collects the wider report so the recorded backend selection
/// reflects the probe that just ran.
pub fn gather_deep(probe_network: bool, show_vectors: bool) -> ReportV1 {
    let state_dir = paths::state_dir();
    match Config::load() {
        Ok(cfg) => {
            let retrieval_probe =
                retrieval_fixture::gather_with_config(&cfg, show_vectors);
            let mut report = gather_with(&cfg, &state_dir, None, probe_network);
            match retrieval_probe {
                Ok(probe) => report.retrieval_probe = Some(probe),
                Err(error) => {
                    let error = short_error(&format!("{error:#}"));
                    report.retrieval_probe_error = Some(error.clone());
                    report.findings.push(Finding {
                        code: "deep_retrieval_check_failed".into(),
                        severity: FindingSeverity::Warning,
                        summary: error,
                        action: Some(
                            "inspect the error and retry cfetch doctor --deep".into(),
                        ),
                    });
                }
            }
            report
        }
        Err(error) => {
            let detail = short_error(&error.to_string());
            let mut report = gather_without_config(&state_dir, error, probe_network);
            report.retrieval_probe_error = Some(format!(
                "configuration unavailable, so retrieval models could not be tested: {detail}"
            ));
            report
        }
    }
}

/// Dashboard entry point: reuse its already-completed daemon probe so one
/// screen refresh does not ask the local control channel the same question
/// twice.
pub fn gather_with(
    cfg: &Config,
    state_dir: &Path,
    daemon_probe: Option<&daemon::Response>,
    probe_network: bool,
) -> ReportV1 {
    gather_inner(Some(cfg), None, state_dir, daemon_probe, probe_network)
}

fn gather_without_config(state_dir: &Path, error: anyhow::Error, probe_network: bool) -> ReportV1 {
    gather_inner(
        None,
        Some(short_error(&error.to_string())),
        state_dir,
        None,
        probe_network,
    )
}

fn gather_inner(
    cfg: Option<&Config>,
    config_error: Option<String>,
    state_dir: &Path,
    supplied_daemon_probe: Option<&daemon::Response>,
    probe_network: bool,
) -> ReportV1 {
    let owned_probe;
    let daemon_probe = match supplied_daemon_probe {
        Some(probe) => Some(probe),
        None => {
            owned_probe = daemon::call("ping", Duration::from_millis(300));
            owned_probe.as_ref()
        }
    };
    let daemon_running = daemon_probe.is_some_and(|probe| probe.ok);
    let daemon_version = daemon_probe.and_then(|probe| probe.version.clone());
    let mut findings = Vec::new();
    let local_endpoint = match net::existing_endpoint_id(state_dir) {
        Ok(endpoint) => endpoint.map(|id| id.to_string()),
        Err(error) => {
            findings.push(Finding {
                code: "network_identity_unreadable".into(),
                severity: FindingSeverity::Critical,
                summary: short_error(&error.to_string()),
                action: Some(
                    "repair the identity key deliberately; replacing it changes this host's identity"
                        .into(),
                ),
            });
            None
        }
    };

    let build_backend = compiled_backend();
    let mut runtime = runtime_status::load_cached_in(state_dir);
    runtime_status::apply_daemon_observation(&mut runtime, daemon_running);

    let config = match cfg {
        Some(cfg) => ConfigDiagnostic {
            loaded: true,
            brain_root: Some(cfg.brain_root.to_string_lossy().into_owned()),
            error: None,
        },
        None => {
            findings.push(Finding {
                code: "config_unusable".into(),
                severity: FindingSeverity::Critical,
                summary: config_error
                    .clone()
                    .unwrap_or_else(|| "configuration did not load".into()),
                action: Some("run cfetch selfcheck and repair the configuration".into()),
            });
            ConfigDiagnostic {
                loaded: false,
                brain_root: None,
                error: config_error,
            }
        }
    };

    if !daemon_running {
        findings.push(Finding {
            code: "daemon_stopped".into(),
            severity: FindingSeverity::Warning,
            summary: "the warm daemon is not answering; hooks use slower direct fallbacks".into(),
            action: Some("run cfetch daemon start".into()),
        });
    }
    if let Some(version) = &daemon_version
        && version != env!("CARGO_PKG_VERSION")
    {
        findings.push(Finding {
            code: "daemon_version_mismatch".into(),
            severity: FindingSeverity::Warning,
            summary: format!(
                "daemon v{version} does not match CLI v{}",
                env!("CARGO_PKG_VERSION")
            ),
            action: Some(
                "restart the daemon with cfetch daemon stop && cfetch daemon start".into(),
            ),
        });
    }

    // The brain root and the derived catalogue are measurements doctor used
    // not to take: selfcheck detected a missing root, an unopenable or stale
    // index, and doctor reported `0 critical` through all of it.
    let catalog = match cfg {
        Some(cfg) => catalog_diagnostic(cfg, state_dir, &mut findings),
        None => CatalogDiagnostic { state: "unavailable".into(), detail: Some("config did not load".into()) },
    };

    let memory = memory_diagnostic(cfg, state_dir, &runtime, daemon_running, &mut findings);
    let inference = inference_diagnostic(cfg, &runtime, &build_backend, &mut findings);
    let hardware = hardware_diagnostics(&runtime, &build_backend);
    let hardware_detection = if cfg!(target_os = "windows") {
        findings.push(Finding {
            code: "hardware_detection_limited".into(),
            severity: FindingSeverity::Warning,
            summary: "this Windows build currently reports the CPU floor but does not enumerate accelerators"
                .into(),
            action: Some("use OS device tools to confirm accelerators until native enumeration lands".into()),
        });
        "cpu_only"
    } else {
        "platform_native"
    };
    let topology = topology_diagnostic(
        cfg,
        state_dir,
        local_endpoint.clone(),
        daemon_running,
        probe_network,
        &mut findings,
    );
    let integrations = integration_diagnostic(state_dir, &mut findings);

    for failure in &runtime.failures {
        if matches!(
            failure.code.as_str(),
            "daemon_unavailable" | "config_unusable"
        ) {
            continue;
        }
        let severity = match failure.severity {
            runtime_status::FailureSeverity::Critical => FindingSeverity::Critical,
            runtime_status::FailureSeverity::Info | runtime_status::FailureSeverity::Warning => {
                FindingSeverity::Warning
            }
        };
        if !findings.iter().any(|finding| finding.code == failure.code) {
            findings.push(Finding {
                code: failure.code.clone(),
                severity,
                summary: failure.code.replace('_', " "),
                action: Some(failure.action.clone()),
            });
        }
    }

    ReportV1 {
        schema_version: SCHEMA_VERSION,
        observed_at: runtime_status::now(),
        build: BuildDiagnostic {
            version: env!("CARGO_PKG_VERSION").into(),
            variant: variant::build_id().map(str::to_string),
            recommended_release_variant: variant::recommended_release().map(|v| v.id.clone()),
            inference_backend: build_backend,
        },
        platform: PlatformDiagnostic {
            os: variant::os_token().into(),
            arch: variant::arch_token().into(),
            x86_64_level: hardware::x86_64_level().map(str::to_string),
            hardware_detection: hardware_detection.into(),
        },
        config,
        catalog,
        daemon: DaemonDiagnostic {
            state: if daemon_running {
                DaemonState::Running
            } else {
                DaemonState::Stopped
            },
            version: daemon_version.clone(),
            version_matches_cli: daemon_version
                .as_deref()
                .map(|version| version == env!("CARGO_PKG_VERSION")),
            endpoint_id: local_endpoint,
            network_endpoint_bound: daemon_running,
        },
        memory,
        inference,
        hardware,
        topology,
        integrations,
        runtime,
        retrieval_probe: None,
        retrieval_probe_error: None,
        findings,
    }
}

fn compiled_backend() -> String {
    if let Some(id) = variant::build_id()
        && let Some(release) = variant::catalog()
            .variants
            .iter()
            .find(|release| release.id == id)
    {
        return release.backend.clone();
    }
    "endpoint".into()
}

/// The derived catalogue's state, with findings for everything a recall
/// would silently get wrong. Mirrors the liveness verdict selfcheck prints,
/// so the two diagnostics can never disagree about the same index.
fn catalog_diagnostic(
    cfg: &crate::config::Config,
    state_dir: &std::path::Path,
    findings: &mut Vec<Finding>,
) -> CatalogDiagnostic {
    if !cfg.brain_root.is_dir() {
        findings.push(Finding {
            code: "brain_root_missing".into(),
            severity: FindingSeverity::Critical,
            summary: format!("brain root {} does not exist", cfg.brain_root.display()),
            action: Some("create the tree (cfetch init) or fix the brain root".into()),
        });
        return CatalogDiagnostic { state: "unavailable".into(), detail: Some("brain root missing".into()) };
    }
    if let Some(cs) = &cfg.client.serving {
        // A none-tier client holds no local catalog by design; opening one
        // here would build the second, silently stale truth it exists to
        // avoid.
        return CatalogDiagnostic { state: "remote".into(), detail: Some(format!("served by {}", cs.addr)) };
    }
    // A diagnostic must not be the thing that first writes an index: the
    // existence check comes first, and the open is READ-ONLY (no
    // delete-and-rebuild on a corrupt file — that recovery belongs to the
    // writer, with the operator's eyes on it).
    if !index::db_exists(state_dir) {
        findings.push(Finding {
            code: "index_never_scanned".into(),
            severity: FindingSeverity::Warning,
            summary: "no catalog has ever been committed; every recall answers from nothing".into(),
            action: Some("run cfetch scan".into()),
        });
        return CatalogDiagnostic { state: "never_scanned".into(), detail: None };
    }
    let conn = match index::open_read_only(state_dir) {
        Ok(conn) => conn,
        Err(error) => {
            let detail = short_error(&error.to_string());
            findings.push(Finding {
                code: "index_unopenable".into(),
                severity: FindingSeverity::Critical,
                summary: format!("the index database does not open: {detail}"),
                action: Some("delete the state index and run cfetch scan to rebuild".into()),
            });
            return CatalogDiagnostic { state: "unavailable".into(), detail: Some(detail) };
        }
    };
    let tree =
        index::tree_fingerprint(&cfg.brain_root, Some(&crate::paths::native_projects_root()), &cfg.rings());
    let stored = match index::stored_fingerprint(&conn) {
        Some(fp) => fp,
        None => {
            // A corrupt or schemaless file: present, unreadable, and not
            // ours to repair from here.
            findings.push(Finding {
                code: "index_unopenable".into(),
                severity: FindingSeverity::Critical,
                summary: "the index database is present but unreadable (empty or not a cfetch catalog)".into(),
                action: Some("delete the state index and run cfetch scan to rebuild".into()),
            });
            return CatalogDiagnostic { state: "unavailable".into(), detail: Some("unreadable catalog".into()) };
        }
    };
    let verdict = heartbeat::observe_index_in(state_dir, Some(stored.as_str()), &tree);
    match verdict {
        heartbeat::IndexLiveness::Current => CatalogDiagnostic {
            state: "current".into(),
            detail: Some(format!("generation {}", index::generation(&conn))),
        },
        heartbeat::IndexLiveness::NeverScanned => {
            findings.push(Finding {
                code: "index_never_scanned".into(),
                severity: FindingSeverity::Warning,
                summary: "no catalog has ever been committed; every recall answers from nothing".into(),
                action: Some("run cfetch scan".into()),
            });
            CatalogDiagnostic { state: "never_scanned".into(), detail: Some(verdict.describe()) }
        }
        heartbeat::IndexLiveness::Stale { .. } => {
            findings.push(Finding {
                code: "index_stale".into(),
                severity: FindingSeverity::Warning,
                summary: format!("index: {}", verdict.describe()),
                action: Some("run cfetch scan".into()),
            });
            CatalogDiagnostic { state: "stale".into(), detail: Some(verdict.describe()) }
        }
    }
}

fn memory_diagnostic(
    cfg: Option<&Config>,
    state_dir: &Path,
        runtime: &runtime_status::RuntimeStatusV1,
        daemon_running: bool,
        findings: &mut Vec<Finding>,
    ) -> MemoryDiagnostic {
    let Some(cfg) = cfg else {
        return MemoryDiagnostic {
            route: "unknown".into(),
            origin: "unknown".into(),
            generation: runtime.memory_route.generation,
            vector_coverage: CoverageDiagnostic {
                state: "unknown".into(),
                embedded: None,
                total: None,
                detail: Some("configuration unavailable".into()),
            },
            shared_vector_artifacts: None,
            peer_artifacts: PeerArtifactDiagnostic {
                transport: "iroh-blobs".into(),
                state: "configuration_unavailable".into(),
                authorized_routes: 0,
                route_order: "shared_store_then_authorized_peers_then_configured_endpoint".into(),
            },
        };
    };

    let (route, origin) = if let Some(serving) = &cfg.client.serving {
        ("remote", serving.addr.clone())
    } else if cfg.serve.enabled {
        ("serving", crate::serve::origin_of(cfg))
    } else {
        ("local", "this host".into())
    };

    let shared_vector_artifacts =
        match vectors::VectorStore::open(&cfg.brain_root, &cfg.embeddings.spec()) {
            Ok(store) => Some(store.len()),
            Err(error) => {
                findings.push(Finding {
                    code: "vector_store_unreadable".into(),
                    severity: FindingSeverity::Warning,
                    summary: short_error(&error.to_string()),
                    action: Some("inspect the shared vector store before embedding again".into()),
                });
                None
            }
        };

    let production_unavailable = embedding_profile::production_availability()
        .err()
        .map(|error| short_error(&error.to_string()));

    let vector_coverage = if !cfg.embeddings.enabled {
        CoverageDiagnostic {
            state: "disabled".into(),
            embedded: None,
            total: None,
            detail: Some("semantic retrieval is disabled".into()),
        }
    } else if let Some(error) = production_unavailable.as_ref() {
        CoverageDiagnostic {
            state: "profile_inactive".into(),
            embedded: None,
            total: None,
            detail: Some(error.clone()),
        }
    } else if cfg.client.serving.is_some() {
        CoverageDiagnostic {
            state: "reported_by_serving_host_on_query".into(),
            embedded: runtime.retrieval.embedded,
            total: runtime.retrieval.total,
            detail: Some("this none-tier host intentionally has no local index".into()),
        }
    } else {
        match index::open_ro(state_dir)
            .and_then(|conn| index::vector_coverage(&conn, &cfg.embeddings.spec()))
        {
            Ok((embedded, total)) => CoverageDiagnostic {
                state: if total == 0 || embedded >= total {
                    "complete"
                } else if embedded == 0 {
                    "none"
                } else {
                    "partial"
                }
                .into(),
                embedded: Some(embedded as u64),
                total: Some(total as u64),
                detail: None,
            },
            Err(error) => CoverageDiagnostic {
                state: "unknown".into(),
                embedded: None,
                total: None,
                detail: Some(short_error(&error.to_string())),
            },
        }
    };

    let authorized_routes = grant::memberships(state_dir)
        .map(|memberships| {
            memberships
                .into_iter()
                .filter(|membership| {
                    membership.network_major == embedding_profile::NETWORK_MAJOR
                })
                .count()
        })
        .unwrap_or(0);
    let peer_artifact_state = if production_unavailable.is_some() {
        "profile_inactive"
    } else if authorized_routes == 0 {
        "no_joined_routes"
    } else if daemon_running {
        "ready"
    } else {
        "daemon_stopped"
    };

    MemoryDiagnostic {
        route: route.into(),
        origin,
        generation: runtime.memory_route.generation,
        vector_coverage,
        shared_vector_artifacts,
        peer_artifacts: PeerArtifactDiagnostic {
            transport: "iroh-blobs".into(),
            state: peer_artifact_state.into(),
            authorized_routes,
            route_order: "shared_store_then_authorized_peers_then_configured_endpoint".into(),
        },
    }
}

fn route_name(route: runtime_status::InferenceRoute) -> String {
    match route {
        runtime_status::InferenceRoute::Local => "local".into(),
        runtime_status::InferenceRoute::Remote => "remote".into(),
    }
}

fn inference_diagnostic(
    cfg: Option<&Config>,
    runtime: &runtime_status::RuntimeStatusV1,
    build_backend: &str,
    findings: &mut Vec<Finding>,
) -> InferenceDiagnostic {
    let profile = embedding_profile::manifest();
    let admission_policy = embedding_profile::admission_policy();
    let (embeddings, reranker, maintenance) = match cfg {
        Some(cfg) => {
            let maintenance_history_issues = maintenance::history_issues(cfg);
            if let Some(issue) = maintenance_history_issues.first() {
                findings.push(Finding {
                    code: "maintenance_history_unreadable".into(),
                    severity: FindingSeverity::Critical,
                    summary: short_error(issue),
                    action: Some(
                        "inspect scratch/cfetch-staging/maintenance/history and restore the immutable record from version control"
                            .into(),
                    ),
                });
            }
            if cfg.embeddings.enabled {
                if let Err(error) = cfg.embeddings.validate_profile() {
                    findings.push(Finding {
                        code: "embedding_profile_mismatch".into(),
                        severity: FindingSeverity::Critical,
                        summary: short_error(&error.to_string()),
                        action: Some(
                            "restore the frozen embedding profile or migrate the network major"
                                .into(),
                        ),
                    });
                } else if let Err(error) = embedding_profile::production_availability() {
                    findings.push(Finding {
                        code: "embedding_profile_not_active".into(),
                        severity: FindingSeverity::Critical,
                        summary: short_error(&error.to_string()),
                        action: Some(
                            "install a release with an active profile and at least one admitted backend"
                                .into(),
                        ),
                    });
                } else if let Err(error) = embed::EmbedClient::new(&cfg.embeddings) {
                    findings.push(Finding {
                        code: "embedding_endpoint_unusable".into(),
                        severity: FindingSeverity::Critical,
                        summary: short_error(&error.to_string()),
                        action: Some(
                            "repair the embedding endpoint, host policy, or credential environment"
                                .into(),
                        ),
                    });
                }
            }
            if cfg.rerank.enabled
                && let Err(error) = rerank::RerankClient::new(&cfg.rerank)
            {
                findings.push(Finding {
                    code: "rerank_endpoint_unusable".into(),
                    severity: FindingSeverity::Warning,
                    summary: short_error(&error.to_string()),
                    action: Some(
                        "repair the rerank endpoint, model, host policy, or credential environment"
                            .into(),
                    ),
                });
            }
            if cfg.maintenance.enabled
                && cfg.maintenance.configured()
                && let Err(error) = maintenance_model::MaintenanceClient::new(&cfg.maintenance)
            {
                findings.push(Finding {
                    code: "maintenance_model_unusable".into(),
                    severity: FindingSeverity::Warning,
                    summary: short_error(&error.to_string()),
                    action: Some(
                        "repair the maintenance model, host policy, or credential environment"
                            .into(),
                    ),
                });
            }
            let embeddings = ModelDiagnostic {
                enabled: cfg.embeddings.enabled,
                backend: if cfg.embeddings.enabled {
                    build_backend.into()
                } else {
                    "disabled".into()
                },
                route: cfg
                    .embeddings
                    .enabled
                    .then(|| {
                        if cfg.embeddings.endpoint.is_empty() && build_backend == "local" {
                            "local".into()
                        } else {
                            route_name(runtime_status::endpoint_route(&cfg.embeddings.endpoint))
                        }
                    }),
                model: cfg.embeddings.model.clone(),
                model_revision: profile.model_revision.into(),
                artifact_policy: admission_policy.artifact_policy.into(),
                profile_id: profile.profile_id.into(),
                dimensions: cfg.embeddings.dimensions,
                vector_encoding: cfg.embeddings.spec().vector_encoding(),
            };
            let reranker = RerankerDiagnostic {
                enabled: cfg.rerank.enabled,
                backend: if cfg.rerank.enabled {
                    "endpoint".into()
                } else {
                    "disabled".into()
                },
                route: cfg
                    .rerank
                    .enabled
                    .then(|| route_name(runtime_status::endpoint_route(&cfg.rerank.endpoint))),
                model: cfg.rerank.enabled.then(|| cfg.rerank.model.clone()),
                candidates: cfg.rerank.enabled.then_some(cfg.rerank.candidates),
            };
            let maintenance = MaintenanceModelDiagnostic {
                enabled: cfg.maintenance.enabled,
                configured: cfg.maintenance.configured(),
                state: if !cfg.maintenance.enabled {
                    "disabled"
                } else if runtime.maintenance.paused {
                    "paused"
                } else if !cfg.maintenance.configured() {
                    "setup_needed"
                } else if runtime.maintenance.last_outcome.as_deref() == Some("exception") {
                    "exception"
                } else if runtime.maintenance.last_model_success == Some(false) {
                    "model_unavailable"
                } else if runtime.maintenance.candidates > 0 {
                    "processing"
                } else {
                    "idle"
                }
                .into(),
                route: cfg.maintenance.configured().then(|| {
                    route_name(runtime_status::endpoint_route(&cfg.maintenance.endpoint))
                }),
                proposal_model: cfg
                    .maintenance
                    .configured()
                    .then(|| cfg.maintenance.model.clone()),
                review_model: cfg.maintenance.configured().then(|| {
                    cfg.maintenance
                        .review_model
                        .clone()
                        .unwrap_or_else(|| cfg.maintenance.model.clone())
                }),
                candidates: runtime.maintenance.candidates,
                history_events: runtime.maintenance.history_events,
                unreadable_history: maintenance_history_issues.len() as u64,
                exceptions: runtime.maintenance.exceptions,
                last_outcome: runtime.maintenance.last_outcome.clone(),
            };
            (embeddings, reranker, maintenance)
        }
        None => (
            ModelDiagnostic {
                enabled: false,
                backend: "unknown".into(),
                route: None,
                model: profile.model.into(),
                model_revision: profile.model_revision.into(),
                artifact_policy: admission_policy.artifact_policy.into(),
                profile_id: profile.profile_id.into(),
                dimensions: profile.dimensions,
                vector_encoding: profile.vector_encoding.into(),
            },
            RerankerDiagnostic {
                enabled: false,
                backend: "unknown".into(),
                route: None,
                model: None,
                candidates: None,
            },
            MaintenanceModelDiagnostic {
                enabled: false,
                configured: false,
                state: "unknown".into(),
                route: None,
                proposal_model: None,
                review_model: None,
                candidates: runtime.maintenance.candidates,
                history_events: runtime.maintenance.history_events,
                unreadable_history: 0,
                exceptions: runtime.maintenance.exceptions,
                last_outcome: runtime.maintenance.last_outcome.clone(),
            },
        ),
    };

    InferenceDiagnostic {
        build_backend: build_backend.into(),
        embeddings,
        reranker,
        maintenance,
        selected: runtime.inference.selected.clone(),
        last_used: runtime.inference.last_used.clone(),
        utilization: UtilizationDiagnostic {
            state: UtilizationState::NotReported,
            detail: "this backend reports selection and completed attempts, but no live device utilization counter"
                .into(),
        },
    }
}

fn hardware_diagnostics(
    runtime: &runtime_status::RuntimeStatusV1,
    build_backend: &str,
) -> Vec<HardwareDiagnostic> {
    hardware::detect()
        .into_iter()
        .map(|found| {
            let class = format!("{:?}", found.device.class()).to_lowercase();
            let caveat = found.caveat().or_else(|| {
                (found.device.class() != hardware::Class::Cpu).then(|| {
                    "device discovery is not backend admission; require accelerated placement and mixed query/document retrieval evidence"
                        .into()
                })
            });
            let selected = runtime
                .inference
                .selected
                .as_ref()
                .is_some_and(|selection| {
                    selection.route == Some(runtime_status::InferenceRoute::Local)
                        && selection.device_class.as_deref() == Some(class.as_str())
                });
            // Discovery does not decide whether a native artifact is usable.
            // Initialization, accelerated placement, repeatability, and the
            // mixed-backend retrieval matrix are the evidence boundary.
            let supported = match build_backend {
                "cpu" => found.device == hardware::Device::Cpu,
                "qnn" => found.device == hardware::Device::QualcommNpu,
                "vitis" => found.device == hardware::Device::AmdNpu,
                "openvino" => matches!(
                    found.device,
                    hardware::Device::IntelNpu
                        | hardware::Device::IntelGpu
                        | hardware::Device::Cpu
                ),
                "coreml" => matches!(
                    found.device,
                    hardware::Device::AppleNeuralEngine
                        | hardware::Device::AppleGpu
                        | hardware::Device::Cpu
                ),
                "cuda" | "tensorrt" => found.device == hardware::Device::NvidiaGpu,
                "migraphx" | "rocm" => found.device == hardware::Device::AmdGpu,
                "directml" | "webgpu" => found.device.class() == hardware::Class::Gpu,
                _ => false,
            };
            let binding = if selected {
                BindingState::Selected
            } else if supported {
                BindingState::AvailableNotSelected
            } else {
                BindingState::NotSupportedByBuild
            };
            HardwareDiagnostic {
                device: found.device.describe().into(),
                token: found.device.token().into(),
                class,
                evidence: found.evidence,
                architecturally_usable: true,
                unusable_reason: None,
                caveat,
                binding,
                selected,
                utilization: if selected {
                    DeviceUtilizationState::NotReported
                } else {
                    DeviceUtilizationState::NotSelected
                },
            }
        })
        .collect()
}

fn topology_diagnostic(
    cfg: Option<&Config>,
    state_dir: &Path,
    local_endpoint_id: Option<String>,
    daemon_running: bool,
    probe_network: bool,
    findings: &mut Vec<Finding>,
) -> TopologyDiagnostic {
    let memberships = match grant::memberships(state_dir) {
        Ok(memberships) => memberships,
        Err(error) => {
            findings.push(Finding {
                code: "memberships_unreadable".into(),
                severity: FindingSeverity::Critical,
                summary: short_error(&error.to_string()),
                action: Some("repair the per-host memberships document".into()),
            });
            Vec::new()
        }
    };

    let mut joined_origins: Vec<JoinedOriginDiagnostic> = memberships
        .iter()
        .map(|membership| JoinedOriginDiagnostic {
            endpoint_id: membership.origin.id.to_string(),
            slice: membership.slice.clone(),
            mode: membership.mode.as_str().into(),
            joined_at: membership.joined_at,
            reachability: Reachability::NotProbed,
            generation: None,
            detail: Some(if !probe_network {
                "network probes disabled".into()
            } else if !daemon_running {
                "local daemon is stopped".into()
            } else if membership.network_major != embedding_profile::NETWORK_MAJOR {
                format!(
                    "membership is network major {}, this build requires {}",
                    membership.network_major,
                    embedding_profile::NETWORK_MAJOR
                )
            } else {
                "probe not run".into()
            }),
        })
        .collect();

    for membership in &memberships {
        if membership.network_major != embedding_profile::NETWORK_MAJOR {
            findings.push(Finding {
                code: "membership_network_incompatible".into(),
                severity: FindingSeverity::Critical,
                summary: format!(
                    "joined slice {:?} belongs to network major {}, this build requires {}",
                    membership.slice,
                    membership.network_major,
                    embedding_profile::NETWORK_MAJOR
                ),
                action: Some("rejoin the slice through a compatible origin".into()),
            });
        }
    }

    if probe_network && daemon_running {
        let mut probes = Vec::new();
        for (index, membership) in memberships
            .iter()
            .take(MAX_PROBED_PEERS)
            .cloned()
            .enumerate()
        {
            if membership.network_major != embedding_profile::NETWORK_MAJOR {
                continue;
            }
            probes.push((
                index,
                std::thread::spawn(move || {
                    daemon::probe_iroh(&membership.origin, &membership.slice, PEER_PROBE_TIMEOUT)
                        .map_err(|error| short_error(&error.to_string()))
                }),
            ));
        }
        for (index, probe) in probes {
            match probe.join() {
                Ok(Ok(response)) if response.iroh_connected == Some(true) && response.ok => {
                    joined_origins[index].reachability = Reachability::Reachable;
                    joined_origins[index].generation = response.generation;
                    joined_origins[index].detail = response.fresh.map(|fresh| {
                        if fresh {
                            "serving path is fresh"
                        } else {
                            "serving path answered stale"
                        }
                        .into()
                    });
                }
                Ok(Ok(response)) if response.iroh_connected == Some(true) => {
                    let error = response
                        .error
                        .as_deref()
                        .map(short_error)
                        .unwrap_or_else(|| "serving path refused without a reason".into());
                    joined_origins[index].reachability = Reachability::Reachable;
                    joined_origins[index].detail = Some(format!(
                        "transport connected, but the authorized serving path is unavailable: {error}"
                    ));
                    findings.push(Finding {
                        code: "joined_origin_unusable".into(),
                        severity: FindingSeverity::Warning,
                        summary: format!(
                            "joined origin {} for slice {:?} connected but could not serve: {error}",
                            short_id(&joined_origins[index].endpoint_id),
                            joined_origins[index].slice
                        ),
                        action: Some("check serving mode, network major, and the slice grant".into()),
                    });
                }
                Ok(Ok(response)) => {
                    let error = response
                        .error
                        .as_deref()
                        .map(short_error)
                        .unwrap_or_else(|| "transport did not complete".into());
                    joined_origins[index].reachability = Reachability::Unreachable;
                    joined_origins[index].detail = Some(error.clone());
                    findings.push(Finding {
                        code: "joined_origin_unreachable".into(),
                        severity: FindingSeverity::Warning,
                        summary: format!(
                            "joined origin {} for slice {:?} did not answer: {error}",
                            short_id(&joined_origins[index].endpoint_id),
                            joined_origins[index].slice
                        ),
                        action: Some("check both daemons and network routes".into()),
                    });
                }
                Ok(Err(error)) => {
                    joined_origins[index].reachability = Reachability::Unreachable;
                    joined_origins[index].detail = Some(error.clone());
                    findings.push(Finding {
                        code: "joined_origin_unreachable".into(),
                        severity: FindingSeverity::Warning,
                        summary: format!(
                            "joined origin {} for slice {:?} did not answer: {error}",
                            short_id(&joined_origins[index].endpoint_id),
                            joined_origins[index].slice
                        ),
                        action: Some(
                            "check both daemons, network routes, and the slice grant".into(),
                        ),
                    });
                }
                Err(_) => {
                    joined_origins[index].reachability = Reachability::Unreachable;
                    joined_origins[index].detail = Some("diagnostic probe worker failed".into());
                }
            }
        }
        if memberships.len() > MAX_PROBED_PEERS {
            for peer in joined_origins.iter_mut().skip(MAX_PROBED_PEERS) {
                peer.detail = Some(format!(
                    "not probed: one report is capped at {MAX_PROBED_PEERS} peers"
                ));
            }
        }
    }

    let now = runtime_status::now();
    let mut granted_peers = Vec::new();
    if let Some(cfg) = cfg {
        match cfg.slice_model() {
            Ok(model) => {
                for slice in model.names() {
                    match grant::read(&cfg.brain_root, slice) {
                        Ok(grants) => {
                            granted_peers.extend(grants.into_iter().map(|grant| {
                                let state =
                                    if grant.expires_at.is_some_and(|expires| now >= expires) {
                                        "expired"
                                    } else if grant.pending() {
                                        "pending_invite"
                                    } else {
                                        "authorized"
                                    };
                                GrantedPeerDiagnostic {
                                    endpoint_id: grant.peer,
                                    slice: grant.slice,
                                    mode: grant.mode.as_str().into(),
                                    state: state.into(),
                                    expires_at: grant.expires_at,
                                }
                            }));
                        }
                        Err(error) => findings.push(Finding {
                            code: format!("grants_unreadable_{slice}"),
                            severity: FindingSeverity::Critical,
                            summary: short_error(&error.to_string()),
                            action: Some(format!("repair the grants record for slice {slice:?}")),
                        }),
                    }
                }
            }
            Err(error) => findings.push(Finding {
                code: "slice_model_unusable".into(),
                severity: FindingSeverity::Critical,
                summary: short_error(&error.to_string()),
                action: Some("repair the configured slice rules".into()),
            }),
        }
    }
    granted_peers.sort_by(|a, b| {
        a.slice
            .cmp(&b.slice)
            .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
    });

    TopologyDiagnostic {
        local_endpoint_id,
        joined_origins,
        granted_peers,
    }
}

fn integration_diagnostic(state_dir: &Path, findings: &mut Vec<Finding>) -> IntegrationDiagnostic {
    let liveness = heartbeat::liveness_in(state_dir);
    let summary = liveness.summary();
    match liveness.severity() {
        heartbeat::Severity::Healthy => {}
        heartbeat::Severity::Unobserved => findings.push(Finding {
            code: "hooks_unobserved".into(),
            severity: FindingSeverity::Warning,
            summary: "one or more registered hooks have never reported on this host".into(),
            action: Some(
                "start an agent session, exercise it, then run cfetch doctor again".into(),
            ),
        }),
        heartbeat::Severity::Failing => findings.push(Finding {
            code: "hooks_failing".into(),
            severity: FindingSeverity::Warning,
            summary: "one or more hooks are repeatedly failing".into(),
            action: Some("inspect the hook rows below and run cfetch selfcheck".into()),
        }),
    }
    let hooks = liveness
        .hooks
        .into_iter()
        .map(|hook| {
            let (state, last_ok, consecutive_failures, last_error) = match hook.state {
                heartbeat::HookState::Unobserved => ("unobserved", None, 0, None),
                heartbeat::HookState::Healthy { last_ok } => ("healthy", Some(last_ok), 0, None),
                heartbeat::HookState::Failing {
                    consecutive,
                    last_error,
                    last_ok,
                } => ("failing", last_ok, consecutive, last_error),
            };
            HookDiagnostic {
                name: hook.name,
                registered: hook.registered,
                state: state.into(),
                last_ok,
                consecutive_failures,
                last_error,
            }
        })
        .collect();
    IntegrationDiagnostic { hooks, summary }
}

fn short_error(error: &str) -> String {
    let mut out: String = error.chars().take(300).collect();
    if error.chars().count() > 300 {
        out.push('…');
    }
    out
}

fn short_id(id: &str) -> &str {
    id.get(..12).unwrap_or(id)
}

fn binding_name(binding: BindingState) -> &'static str {
    match binding {
        BindingState::NotSupportedByBuild => "not supported by this build",
        BindingState::AvailableNotSelected => "available, not selected",
        BindingState::Selected => "selected",
    }
}

/// Human renderer shared by the one-shot CLI and the terminal dashboard.
pub fn display_lines(report: &ReportV1) -> Vec<DisplayLine> {
    let critical = report
        .findings
        .iter()
        .filter(|finding| finding.severity == FindingSeverity::Critical)
        .count();
    let warnings = report.findings.len() - critical;
    let mut lines = vec![DisplayLine::new(
        if critical > 0 {
            DisplayTone::Error
        } else if warnings > 0 {
            DisplayTone::Warning
        } else {
            DisplayTone::Good
        },
        format!(
            "cfetch doctor v{} — {critical} critical, {warnings} warning(s)",
            report.build.version
        ),
    )];

    lines.push(DisplayLine::new(DisplayTone::Heading, "System"));
    lines.push(DisplayLine::new(
        if report.config.loaded {
            DisplayTone::Good
        } else {
            DisplayTone::Error
        },
        match (&report.config.brain_root, &report.config.error) {
            (Some(root), _) => format!("  config loaded — brain {root}"),
            (_, Some(error)) => format!("  config unavailable — {error}"),
            _ => "  config unavailable".into(),
        },
    ));
    lines.push(DisplayLine::new(
        if report.daemon.state == DaemonState::Running {
            DisplayTone::Good
        } else {
            DisplayTone::Warning
        },
        match &report.daemon.version {
            Some(version) => format!(
                "  daemon running v{version} — network endpoint {}",
                if report.daemon.network_endpoint_bound {
                    "bound"
                } else {
                    "not bound"
                }
            ),
            None => "  daemon stopped — direct fallbacks remain available".into(),
        },
    ));
    lines.push(DisplayLine::new(
        DisplayTone::Normal,
        format!(
            "  build {} · backend {} · {} / {}{}",
            report
                .build
                .variant
                .as_deref()
                .unwrap_or("unidentified source build"),
            report.build.inference_backend,
            report.platform.os,
            report.platform.arch,
            report
                .platform
                .x86_64_level
                .as_deref()
                .map(|level| format!(" / x86-64-{level}"))
                .unwrap_or_default(),
        ),
    ));
    if report.platform.hardware_detection != "platform_native" {
        lines.push(DisplayLine::new(
            DisplayTone::Warning,
            format!(
                "  hardware detection: {}",
                report.platform.hardware_detection
            ),
        ));
    }

    lines.push(DisplayLine::new(
        DisplayTone::Heading,
        "Memory and artifacts",
    ));
    lines.push(DisplayLine::new(
        DisplayTone::Normal,
        format!(
            "  route {} — {}{}",
            report.memory.route,
            report.memory.origin,
            report
                .memory
                .generation
                .map(|generation| format!(" · generation {generation}"))
                .unwrap_or_default()
        ),
    ));
    lines.push(DisplayLine::new(
        match report.memory.vector_coverage.state.as_str() {
            "complete" | "disabled" | "reported_by_serving_host_on_query" => DisplayTone::Good,
            "partial" | "none" => DisplayTone::Warning,
            _ => DisplayTone::Muted,
        },
        format!(
            "  vectors {}{} · shared artifacts {}",
            report.memory.vector_coverage.state,
            match (
                report.memory.vector_coverage.embedded,
                report.memory.vector_coverage.total,
            ) {
                (Some(embedded), Some(total)) => format!(" ({embedded}/{total} blocks)"),
                _ => String::new(),
            },
            report
                .memory
                .shared_vector_artifacts
                .map(|count| count.to_string())
                .unwrap_or_else(|| "unknown".into())
        ),
    ));
    if let Some(detail) = &report.memory.vector_coverage.detail {
        lines.push(DisplayLine::new(
            DisplayTone::Muted,
            format!("    {detail}"),
        ));
    }
    lines.push(DisplayLine::new(
        match report.memory.peer_artifacts.state.as_str() {
            "ready" => DisplayTone::Good,
            "daemon_stopped" => DisplayTone::Warning,
            _ => DisplayTone::Muted,
        },
        format!(
            "  peer artifacts {} — {} · {} authorized route(s)",
            report.memory.peer_artifacts.transport,
            report.memory.peer_artifacts.state,
            report.memory.peer_artifacts.authorized_routes,
        ),
    ));

    lines.push(DisplayLine::new(DisplayTone::Heading, "Inference"));
    let embedding_config_error = report.findings.iter().any(|finding| {
        matches!(
            finding.code.as_str(),
            "embedding_profile_mismatch"
                | "embedding_profile_not_active"
                | "embedding_endpoint_unusable"
        )
    });
    let (embedding_tone, embedding_state) = match report.retrieval_probe.as_ref() {
        Some(probe) if retrieval_fixture::vector_active(probe) => {
            (DisplayTone::Good, "ACTIVE — deep check passed")
        }
        Some(_) if report.inference.embeddings.enabled => {
            (DisplayTone::Error, "INACTIVE — deep check failed")
        }
        Some(_) => (DisplayTone::Muted, "disabled"),
        None if !report.inference.embeddings.enabled => (DisplayTone::Muted, "disabled"),
        None if embedding_config_error => (DisplayTone::Error, "unavailable"),
        None => (DisplayTone::Normal, "configured, not tested"),
    };
    let embedding_route = report
        .inference
        .embeddings
        .route
        .as_deref()
        .map(|route| format!(" ({route})"))
        .unwrap_or_default();
    lines.push(DisplayLine::new(
        embedding_tone,
        format!(
            "  embedding model {embedding_state}{embedding_route} — {} @ {}",
            report.inference.embeddings.model,
            short_id(&report.inference.embeddings.model_revision),
        ),
    ));
    lines.push(DisplayLine::new(
        DisplayTone::Muted,
        "    job: turns queries and notes into vectors for meaning-based search",
    ));
    lines.push(DisplayLine::new(
        DisplayTone::Muted,
        format!(
            "    output: {} dimensions, {}",
            report.inference.embeddings.dimensions,
            report.inference.embeddings.vector_encoding,
        ),
    ));
    lines.push(DisplayLine::new(
        DisplayTone::Muted,
        format!(
            "    profile: {} · artifact rule: {}",
            report.inference.embeddings.profile_id,
            report.inference.embeddings.artifact_policy,
        ),
    ));
    let reranker_config_error = report
        .findings
        .iter()
        .any(|finding| finding.code == "rerank_endpoint_unusable");
    let (reranker_tone, reranker_state) = match report.retrieval_probe.as_ref() {
        Some(probe) => match retrieval_fixture::reranker_status(probe) {
            "active" => (DisplayTone::Good, "ACTIVE — deep check passed"),
            "unavailable" => (DisplayTone::Warning, "INACTIVE — deep check failed"),
            _ => (DisplayTone::Muted, "disabled"),
        },
        None if !report.inference.reranker.enabled => (DisplayTone::Muted, "disabled"),
        None if reranker_config_error => (DisplayTone::Warning, "unavailable"),
        None => (DisplayTone::Normal, "configured, not tested"),
    };
    lines.push(DisplayLine::new(
        reranker_tone,
        if report.inference.reranker.enabled {
            format!(
                "  reranker model {reranker_state} ({}) — {} · top {} candidates",
                report
                    .inference
                    .reranker
                    .route
                    .as_deref()
                    .unwrap_or("route unknown"),
                report
                    .inference
                    .reranker
                    .model
                    .as_deref()
                    .unwrap_or("model not configured"),
                report.inference.reranker.candidates.unwrap_or(0)
            )
        } else {
            "  reranker model disabled".into()
        },
    ));
    lines.push(DisplayLine::new(
        DisplayTone::Muted,
        "    job: reorders the shortlist; it does not find new notes",
    ));
    lines.push(DisplayLine::new(
        match report.inference.maintenance.state.as_str() {
            "idle" => DisplayTone::Good,
            "processing" | "setup_needed" | "model_unavailable" | "exception" => {
                DisplayTone::Warning
            }
            _ => DisplayTone::Muted,
        },
        if report.inference.maintenance.configured {
            format!(
                "  maintenance {} ({}) — propose {} · review {} · {} staged / {} event(s) / {} unreadable / {} exception(s)",
                report.inference.maintenance.state.replace('_', " "),
                report
                    .inference
                    .maintenance
                    .route
                    .as_deref()
                    .unwrap_or("route unknown"),
                report
                    .inference
                    .maintenance
                    .proposal_model
                    .as_deref()
                    .unwrap_or("model not configured"),
                report
                    .inference
                    .maintenance
                    .review_model
                    .as_deref()
                    .unwrap_or("model not configured"),
                report.inference.maintenance.candidates,
                report.inference.maintenance.history_events,
                report.inference.maintenance.unreadable_history,
                report.inference.maintenance.exceptions,
            )
        } else {
            format!(
                "  maintenance {} — {} staged candidate(s)",
                report.inference.maintenance.state.replace('_', " "),
                report.inference.maintenance.candidates,
            )
        },
    ));
    lines.push(DisplayLine::new(
        DisplayTone::Muted,
        "    job: proposes and reviews memory changes; it is not part of recall",
    ));
    lines.push(DisplayLine::new(
        if report.inference.selected.is_some() {
            DisplayTone::Good
        } else {
            DisplayTone::Muted
        },
        match &report.inference.selected {
            Some(selected) => format!(
                "  selected {}{}{} at {}",
                selected.backend,
                selected
                    .device_class
                    .as_deref()
                    .map(|class| format!(" / {class}"))
                    .unwrap_or_default(),
                selected
                    .route
                    .map(|route| format!(" / {}", route_name(route)))
                    .unwrap_or_default(),
                selected.selected_at
            ),
            None => "  selected backend: no successful selection recorded yet".into(),
        },
    ));
    lines.push(DisplayLine::new(
        match &report.inference.last_used {
            Some(attempt) if attempt.success => DisplayTone::Good,
            Some(_) => DisplayTone::Warning,
            None => DisplayTone::Muted,
        },
        match &report.inference.last_used {
            Some(attempt) => format!(
                "  last used {} / {} at {} — {}",
                route_name(attempt.route),
                attempt.backend,
                attempt.observed_at,
                if attempt.success {
                    "succeeded"
                } else {
                    "failed"
                }
            ),
            None => "  last used: no inference attempt recorded yet".into(),
        },
    ));
    lines.push(DisplayLine::new(
        DisplayTone::Muted,
        format!(
            "  live utilization: not reported — {}",
            report.inference.utilization.detail
        ),
    ));
    if report.retrieval_probe.is_none() && report.retrieval_probe_error.is_none() {
        lines.push(DisplayLine::new(
            DisplayTone::Muted,
            "  normal doctor does not call models; run cfetch doctor --deep to prove enabled retrieval models answer",
        ));
    }

    lines.push(DisplayLine::new(DisplayTone::Heading, "Detected hardware"));
    for device in &report.hardware {
        lines.push(DisplayLine::new(
            if device.selected {
                DisplayTone::Good
            } else {
                DisplayTone::Normal
            },
            format!(
                "  {} [{}] — {} · utilization {}",
                device.device,
                device.class,
                binding_name(device.binding),
                match device.utilization {
                    DeviceUtilizationState::NotSelected => "not selected",
                    DeviceUtilizationState::NotReported => "not reported",
                }
            ),
        ));
        lines.push(DisplayLine::new(
            DisplayTone::Muted,
            format!("    evidence: {}", device.evidence),
        ));
        if let Some(reason) = &device.unusable_reason {
            lines.push(DisplayLine::new(
                DisplayTone::Warning,
                format!("    unusable: {reason}"),
            ));
        }
        if let Some(caveat) = &device.caveat {
            lines.push(DisplayLine::new(
                DisplayTone::Warning,
                format!("    note: {caveat}"),
            ));
        }
    }

    if let Some(probe) = &report.retrieval_probe {
        lines.push(DisplayLine::new(
            DisplayTone::Heading,
            "Deep retrieval check",
        ));
        for text in retrieval_fixture::display_lines(probe) {
            let tone = if text.starts_with("embedding model: ACTIVE")
                || text.starts_with("reranker model: ACTIVE")
                || text.starts_with("graph expansion: ACTIVE")
                || text.starts_with("production retrieval gate: PASS")
                || text.trim_start().starts_with("PASS ")
            {
                DisplayTone::Good
            } else if text.starts_with("embedding model: INACTIVE")
                || text.starts_with("reranker model: UNAVAILABLE")
                || text.starts_with("graph expansion: INACTIVE")
                || text.starts_with("production retrieval gate: BLOCKED")
                || text.trim_start().starts_with("FAIL ")
            {
                DisplayTone::Warning
            } else if text.starts_with("temporary retrieval test")
                || text.trim_start().starts_with("NOT RUN ")
            {
                DisplayTone::Muted
            } else {
                DisplayTone::Normal
            };
            lines.push(DisplayLine::new(
                tone,
                if text.is_empty() {
                    text
                } else {
                    format!("  {text}")
                },
            ));
        }
    } else if let Some(error) = &report.retrieval_probe_error {
        lines.push(DisplayLine::new(
            DisplayTone::Heading,
            "Deep retrieval check",
        ));
        lines.push(DisplayLine::new(
            DisplayTone::Warning,
            format!("  NOT RUN — {error}"),
        ));
    }

    lines.push(DisplayLine::new(DisplayTone::Heading, "Peers and grants"));
    lines.push(DisplayLine::new(
        if report.topology.local_endpoint_id.is_some() {
            DisplayTone::Good
        } else {
            DisplayTone::Muted
        },
        match &report.topology.local_endpoint_id {
            Some(id) => format!("  this host endpoint {id}"),
            None => "  this host has no network identity yet (no network operation has needed one)"
                .into(),
        },
    ));
    if report.topology.joined_origins.is_empty() {
        lines.push(DisplayLine::new(
            DisplayTone::Muted,
            "  no remote origins joined",
        ));
    }
    for peer in &report.topology.joined_origins {
        let (tone, state) = match peer.reachability {
            Reachability::Reachable => (DisplayTone::Good, "reachable"),
            Reachability::Unreachable => (DisplayTone::Warning, "unreachable"),
            Reachability::NotProbed => (DisplayTone::Muted, "not probed"),
        };
        lines.push(DisplayLine::new(
            tone,
            format!(
                "  joined {} · slice {} ({}) — {}{}",
                short_id(&peer.endpoint_id),
                peer.slice,
                peer.mode,
                state,
                peer.generation
                    .map(|generation| format!(" · generation {generation}"))
                    .unwrap_or_default()
            ),
        ));
        if let Some(detail) = &peer.detail {
            lines.push(DisplayLine::new(
                DisplayTone::Muted,
                format!("    {detail}"),
            ));
        }
    }
    if report.topology.granted_peers.is_empty() {
        lines.push(DisplayLine::new(
            DisplayTone::Muted,
            "  no outbound slice grants",
        ));
    }
    for peer in &report.topology.granted_peers {
        lines.push(DisplayLine::new(
            DisplayTone::Normal,
            format!(
                "  grant {} ({}) -> {} — {}",
                peer.slice,
                peer.mode,
                peer.endpoint_id
                    .as_deref()
                    .map(short_id)
                    .unwrap_or("unused invite"),
                peer.state
            ),
        ));
    }

    lines.push(DisplayLine::new(DisplayTone::Heading, "Agent integrations"));
    lines.push(DisplayLine::new(
        if report
            .integrations
            .hooks
            .iter()
            .all(|hook| hook.state == "healthy")
        {
            DisplayTone::Good
        } else {
            DisplayTone::Warning
        },
        format!("  {}", report.integrations.summary),
    ));
    for hook in &report.integrations.hooks {
        lines.push(DisplayLine::new(
            match hook.state.as_str() {
                "healthy" => DisplayTone::Good,
                "failing" => DisplayTone::Error,
                _ => DisplayTone::Muted,
            },
            format!(
                "    {:<14} {}{}",
                hook.name,
                hook.state,
                if hook.consecutive_failures > 0 {
                    format!(" ({} consecutive)", hook.consecutive_failures)
                } else {
                    String::new()
                }
            ),
        ));
    }

    lines.push(DisplayLine::new(DisplayTone::Heading, "Findings"));
    if report.findings.is_empty() {
        lines.push(DisplayLine::new(
            DisplayTone::Good,
            "  no actionable findings",
        ));
    }
    for finding in &report.findings {
        lines.push(DisplayLine::new(
            if finding.severity == FindingSeverity::Critical {
                DisplayTone::Error
            } else {
                DisplayTone::Warning
            },
            format!(
                "  {} {} — {}{}",
                if finding.severity == FindingSeverity::Critical {
                    "FAIL"
                } else {
                    "WARN"
                },
                finding.code,
                finding.summary,
                finding
                    .action
                    .as_deref()
                    .map(|action| format!(" · {action}"))
                    .unwrap_or_default()
            ),
        ));
    }
    lines
}

pub fn render_text(report: &ReportV1) -> String {
    display_lines(report)
        .into_iter()
        .map(|line| line.text)
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ReportV1 {
        let runtime = runtime_status::RuntimeStatusV1::default();
        ReportV1 {
            schema_version: 1,
            observed_at: 1,
            build: BuildDiagnostic {
                version: "0.9.9".into(),
                variant: Some("linux-cfetch-remote-x86_64".into()),
                recommended_release_variant: Some("linux-cfetch-remote-x86_64".into()),
                inference_backend: "endpoint".into(),
            },
            platform: PlatformDiagnostic {
                os: "linux".into(),
                arch: "x86_64".into(),
                x86_64_level: Some("v3".into()),
                hardware_detection: "platform_native".into(),
            },
            config: ConfigDiagnostic {
                loaded: true,
                brain_root: Some("/brain".into()),
                error: None,
            },
            catalog: CatalogDiagnostic { state: "current".into(), detail: Some("generation 1".into()) },
            daemon: DaemonDiagnostic {
                state: DaemonState::Running,
                version: Some("0.9.9".into()),
                version_matches_cli: Some(true),
                endpoint_id: Some("abcdefghijklmnopqrst".into()),
                network_endpoint_bound: true,
            },
            memory: MemoryDiagnostic {
                route: "local".into(),
                origin: "this host".into(),
                generation: Some(7),
                vector_coverage: CoverageDiagnostic {
                    state: "partial".into(),
                    embedded: Some(3),
                    total: Some(5),
                    detail: None,
                },
                shared_vector_artifacts: Some(3),
                peer_artifacts: PeerArtifactDiagnostic {
                    transport: "iroh-blobs".into(),
                    state: "ready".into(),
                    authorized_routes: 1,
                    route_order: "shared_store_then_authorized_peers_then_configured_endpoint"
                        .into(),
                },
            },
            inference: InferenceDiagnostic {
                build_backend: "endpoint".into(),
                embeddings: ModelDiagnostic {
                    enabled: true,
                    backend: "endpoint".into(),
                    route: Some("remote".into()),
                    model: "embedding-model".into(),
                    model_revision: "revision".into(),
                    artifact_policy: "backend-native".into(),
                    profile_id: "profile".into(),
                    dimensions: 768,
                    vector_encoding: "signed-int8x768".into(),
                },
                reranker: RerankerDiagnostic {
                    enabled: false,
                    backend: "disabled".into(),
                    route: None,
                    model: None,
                    candidates: None,
                },
                maintenance: MaintenanceModelDiagnostic {
                    enabled: true,
                    configured: true,
                    state: "idle".into(),
                    route: Some("remote".into()),
                    proposal_model: Some("maintenance-model".into()),
                    review_model: Some("review-model".into()),
                    candidates: 0,
                    history_events: 12,
                    unreadable_history: 0,
                    exceptions: 1,
                    last_outcome: Some("applied".into()),
                },
                selected: None,
                last_used: None,
                utilization: UtilizationDiagnostic {
                    state: UtilizationState::NotReported,
                    detail: "backend has no counter".into(),
                },
            },
            hardware: vec![HardwareDiagnostic {
                device: "NVIDIA GPU".into(),
                token: "nvidia".into(),
                class: "gpu".into(),
                evidence: "/sys/class/drm/card0".into(),
                architecturally_usable: true,
                unusable_reason: None,
                caveat: None,
                binding: BindingState::NotSupportedByBuild,
                selected: false,
                utilization: DeviceUtilizationState::NotSelected,
            }],
            topology: TopologyDiagnostic {
                local_endpoint_id: Some("abcdefghijklmnopqrst".into()),
                joined_origins: vec![JoinedOriginDiagnostic {
                    endpoint_id: "zyxwvutsrqponmlkjihg".into(),
                    slice: "shared".into(),
                    mode: "ro".into(),
                    joined_at: 1,
                    reachability: Reachability::Reachable,
                    generation: Some(9),
                    detail: Some("serving path is fresh".into()),
                }],
                granted_peers: Vec::new(),
            },
            integrations: IntegrationDiagnostic {
                hooks: Vec::new(),
                summary: "hooks: all registered hooks reporting, healthy".into(),
            },
            runtime,
            retrieval_probe: None,
            retrieval_probe_error: None,
            findings: Vec::new(),
        }
    }

    #[test]
    fn human_report_never_confuses_discovery_selection_and_utilization() {
        let text = render_text(&sample());
        assert!(
            text.contains("NVIDIA GPU [gpu] — not supported by this build"),
            "{text}"
        );
        assert!(
            text.contains("selected backend: no successful selection recorded yet"),
            "{text}"
        );
        assert!(text.contains("live utilization: not reported"), "{text}");
        assert!(
            text.contains("joined zyxwvutsrqpo · slice shared (ro) — reachable"),
            "{text}"
        );
    }

    #[test]
    fn json_is_structured_and_contains_no_grant_secret_field() {
        let json = serde_json::to_value(sample()).unwrap();
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["hardware"][0]["binding"], "not_supported_by_build");
        assert_eq!(json["inference"]["utilization"]["state"], "not_reported");
        let text = serde_json::to_string(&json).unwrap();
        assert!(
            !text.contains("secret"),
            "diagnostics must not expose invite material: {text}"
        );
    }

    #[test]
    fn enabled_embeddings_report_inactive_profile_before_endpoint_health() {
        let brain = tempfile::tempdir().unwrap();
        let mut cfg = Config {
            brain_root: brain.path().to_path_buf(),
            ..Config::default()
        };
        cfg.embeddings.enabled = true;
        cfg.embeddings.endpoint.clear();
        let mut findings = Vec::new();
        inference_diagnostic(
            Some(&cfg),
            &runtime_status::RuntimeStatusV1::default(),
            "endpoint",
            &mut findings,
        );
        let finding = findings
            .iter()
            .find(|finding| finding.code == "embedding_profile_not_active")
            .expect("unavailable embedding profile must be critical and explicit");
        assert_eq!(finding.severity, FindingSeverity::Critical);
        // The profile is active but has no admitted backends (the local
        // build's registry); the finding fires either way — the profile
        // is unavailable for the package-local path.
        assert!(
            finding.summary.contains("not active")
                || finding.summary.contains("no admitted"),
            "{}",
            finding.summary
        );
        assert!(
            findings
                .iter()
                .all(|finding| finding.code != "embedding_endpoint_unusable"),
            "profile lifecycle must be diagnosed before endpoint construction"
        );
    }
}
