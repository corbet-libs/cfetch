//! Embeddings client for semantic recall — an attested OpenAI-compatible
//! `POST {endpoint}/embeddings` with `{"model", "input": [texts]}`. The
//! transport can be a packaged target-native adapter on loopback or an
//! explicitly configured loopback deployment; runtime status keeps those routes
//! distinct.
//!
//! NEVER called from hook entrypoints. Hooks sit on the interactive path and
//! must not spend network time; embedding happens only from the CLI
//! (`cfetch embed-index`, `cfetch recall --semantic/--hybrid`) or the daemon.
//!
//! The endpoint URL comes from the config file — a file agents write — so it
//! is SSRF-guarded at use: https or loopback only, private/link-local/
//! metadata ranges refused, redirects disabled (a 3xx must never be able to
//! walk a request, and its Authorization header, somewhere else).
//!
//! Auth: `embeddings.api_key_env` names an environment VARIABLE holding the
//! key (never the key itself in config) -> `Authorization: Bearer` header.
//! Timeouts: `embeddings.timeout_secs` (default 10 s) bounds one interactive
//! request — a backend that slow is down, not busy. The embed-index batch
//! path scales that bound per batched item (`batch_timeout`): a 64-block
//! batch on a CPU backend is busy, not down, and timing it out only to
//! resend the identical batch is a livelock.

use anyhow::Context as _;
use rusqlite::Connection;

use crate::config::{Config, EmbeddingsConfig};
use crate::index;
use crate::vectors;

/// (scheme, host) of a URL, lowercased, without pulling in a URL crate.
/// Userinfo is refused outright: `https://safe.example@evil.host/` parses two
/// different ways in two different libraries — the classic SSRF confusion.
fn split_url(url: &str) -> anyhow::Result<(String, String)> {
    let (scheme, rest) = url
        .split_once("://")
        .with_context(|| format!("endpoint {url:?} is not a scheme://host URL"))?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    anyhow::ensure!(
        !authority.contains('@'),
        "endpoint URL must not contain userinfo"
    );
    let host = if let Some(bracketed) = authority.strip_prefix('[') {
        // [ipv6]:port
        bracketed
            .split_once(']')
            .context("endpoint URL has an unclosed IPv6 bracket")?
            .0
            .to_string()
    } else {
        authority
            .rsplit_once(':')
            .map_or(authority, |(h, _)| h)
            .to_string()
    };
    anyhow::ensure!(!host.is_empty(), "endpoint URL has no host");
    Ok((scheme.to_ascii_lowercase(), host.to_ascii_lowercase()))
}

fn is_loopback_host(host: &str) -> bool {
    host == "localhost"
        || host
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

/// All inference stays on this device. Legacy host allowlists cannot enable
/// remote computation. Redirects are disabled by the HTTP client.
pub fn check_endpoint(url: &str, _allow_hosts: &[String]) -> anyhow::Result<()> {
    let (scheme, host) = split_url(url)?;
    anyhow::ensure!(
        scheme == "http" || scheme == "https",
        "endpoint must use http or https"
    );
    anyhow::ensure!(
        is_loopback_host(&host),
        "inference endpoint must be on loopback; remote computing was removed"
    );
    Ok(())
}

/// Extra timeout allowance per batched input on the embed-index path.
/// Interactive recall (one input) keeps the tight base bound.
const PER_ITEM: std::time::Duration = std::time::Duration::from_secs(2);

/// The embed-index batch bound: base + per-item.
fn batch_timeout(base: std::time::Duration, items: usize) -> std::time::Duration {
    base + PER_ITEM * items as u32
}

/// Resolves `api_key_env` (an environment variable NAME) to a ready
/// `Bearer …` header value. Empty config = no auth. A value that cannot be
/// an env var name is refused loudly — it is almost certainly a pasted key,
/// and a key in the config file is exactly what this indirection prevents.
pub(crate) fn resolve_auth(api_key_env: &str, field: &str) -> anyhow::Result<Option<String>> {
    resolve_auth_with(api_key_env, field, |name| std::env::var(name).ok())
}

fn resolve_auth_with(
    api_key_env: &str,
    field: &str,
    lookup: impl FnOnce(&str) -> Option<String>,
) -> anyhow::Result<Option<String>> {
    let name = api_key_env.trim();
    if name.is_empty() {
        return Ok(None);
    }
    let valid = !name.starts_with(|c: char| c.is_ascii_digit())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    anyhow::ensure!(
        valid,
        "{field}.api_key_env {name:?} is not an environment variable NAME — \
         configure the variable's name, never the key itself"
    );
    let key = lookup(name).ok_or_else(|| {
        anyhow::anyhow!("{field}.api_key_env: environment variable {name} is not set")
    })?;
    anyhow::ensure!(
        !key.trim().is_empty(),
        "{field}.api_key_env: environment variable {name} is empty"
    );
    Ok(Some(format!("Bearer {}", key.trim())))
}

enum EmbedBackend {
    #[cfg(feature = "embedded-embeddings")]
    Cpu(std::sync::Arc<std::sync::Mutex<crate::local_embedding::Model>>),
    Endpoint {
        agent: ureq::Agent,
        /// Full `./embeddings` URL, endpoint trailing slashes normalized away.
        url: String,
        /// Ready `Bearer .` header value, resolved from `api_key_env` at
        /// construction (a missing variable fails fast, not mid-batch).
        auth: Option<String>,
    },
    Local {
        agent: ureq::Agent,
        state: std::sync::Arc<std::sync::Mutex<LocalBackendState>>,
    },
}

struct LocalBackendState {
    supervisor: crate::local_adapter::AdapterSupervisor,
    ordered_scope_ids: Vec<String>,
    selected_scope: Option<usize>,
    /// Failed scopes stay disabled even if the dispatcher process restarts.
    unavailable_scopes: std::collections::BTreeSet<usize>,
}

type SharedLocalBackendState = std::sync::Arc<std::sync::Mutex<LocalBackendState>>;

/// One package-local dispatcher and successful-scope cache per cfetch
/// process. The serving daemon constructs an `EmbedClient` per query, so a
/// client-owned supervisor would otherwise restart the native runtime and
/// repeat NPU/GPU/CPU discovery for every recall.
static LOCAL_BACKEND_STATE: std::sync::OnceLock<Result<SharedLocalBackendState, String>> =
    std::sync::OnceLock::new();

/// Builds the package-local backend (admitted NPU/GPU/CPU plan).
fn build_package_local_backend(
    cfg: &crate::config::EmbeddingsConfig,
    agent: ureq::Agent,
) -> anyhow::Result<EmbedBackend> {
    anyhow::ensure!(
        cfg.model == crate::embedding_profile::MODEL,
        "an unmanaged embedding model requires an explicit endpoint"
    );
    anyhow::ensure!(
        cfg.api_key_env.is_empty() && cfg.allow_hosts.is_empty(),
        "embeddings endpoint credentials/host exemptions cannot configure a package-local adapter"
    );
    let plan = crate::local_inference::selected_local_package_plan()?
        .context(
            "no admitted local inference package; install a compatible local variant or configure an admitted endpoint",
        )?;
    let executable = std::env::current_exe().context("resolve the running cfetch binary")?;
    let package_directory = executable
        .parent()
        .context("the running cfetch binary has no package directory")?;
    let sibling = package_directory.join(&plan.dispatcher.binary);
    let state = cached_local_backend_state(
        &LOCAL_BACKEND_STATE,
        crate::local_adapter::AdapterLaunch {
            binary: sibling,
            sha256: plan.dispatcher.sha256,
            package_manifest: package_directory.join("package-manifest.json"),
            package_manifest_sha256: plan.package_manifest_sha256,
            ordered_scope_ids: plan.ordered_scope_ids,
        },
    )?;
    Ok(EmbedBackend::Local { agent, state })
}

fn cached_local_backend_state(
    cache: &std::sync::OnceLock<Result<SharedLocalBackendState, String>>,
    launch: crate::local_adapter::AdapterLaunch,
) -> anyhow::Result<SharedLocalBackendState> {
    // Only SUCCESS is cached: a transient launch failure (binary mid-replace,
    // AV lock on Windows, ENOMEM) poisoned the OnceLock for the process
    // lifetime in a long-running daemon — every later query replayed the
    // same stale error even though a retry would succeed. The supervisor's
    // own one-restart logic is the pattern; the first launch deserves the
    // same treatment.
    if let Some(state) = cache.get() {
        return match state {
            Ok(state) => Ok(std::sync::Arc::clone(state)),
            Err(error) => anyhow::bail!("initialize package-local adapter: {error}"),
        };
    }
    let outcome = crate::local_adapter::AdapterSupervisor::new(launch)
        .map(|supervisor| {
            std::sync::Arc::new(std::sync::Mutex::new(LocalBackendState {
                ordered_scope_ids: supervisor.ordered_scope_ids().to_vec(),
                supervisor,
                selected_scope: None,
                unavailable_scopes: std::collections::BTreeSet::new(),
            }))
        })
        .map_err(|error| format!("{error:#}"));
    match outcome {
        Ok(state) => {
            let _ = cache.set(Ok(std::sync::Arc::clone(&state)));
            Ok(state)
        }
        Err(error) => {
            // Do NOT cache the failure; the next query retries the launch.
            anyhow::bail!("initialize package-local adapter: {error}")
        }
    }
}

#[derive(Debug)]
struct AdapterTransportError(String);

impl std::fmt::Display for AdapterTransportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for AdapterTransportError {}

fn response_body_error(
    error: ureq::Error,
    url: &str,
    requested_scope_id: Option<&str>,
) -> anyhow::Error {
    if requested_scope_id.is_some() {
        AdapterTransportError(format!("read response from {url}: {error}")).into()
    } else {
        anyhow::Error::new(error).context(format!("read response from {url}"))
    }
}

#[derive(Debug)]
struct ScopeUnavailableError(String);

impl std::fmt::Display for ScopeUnavailableError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ScopeUnavailableError {}

/// A deterministic refusal of the input itself. Runtime/model failures never
/// acquire this type: only the adapter's exact overlength envelope qualifies.
#[derive(Debug)]
pub(crate) struct InputRefusal {
    pub(crate) token_count: usize,
}

impl std::fmt::Display for InputRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "prefixed input contains {} tokens; the profile limit is {} and truncation is forbidden",
            self.token_count,
            crate::embedding_profile::MAX_TOKENS
        )
    }
}

impl std::error::Error for InputRefusal {}

/// The packaged adapter currently sends `{ "error": "..." }` for every
/// HTTP 400. Match the complete known overlength message, not a status code,
/// substring, or arbitrary request error that row retries cannot repair.
fn input_refusal(status: u16, body: &str) -> Option<InputRefusal> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Envelope {
        error: String,
    }

    if status != 400 {
        return None;
    }
    let envelope = serde_json::from_str::<Envelope>(body).ok()?;
    let suffix = format!(
        " tokens; the profile limit is {} and truncation is forbidden",
        crate::embedding_profile::MAX_TOKENS
    );
    let count = envelope
        .error
        .strip_prefix("prefixed input contains ")?
        .strip_suffix(&suffix)?;
    let token_count = count.parse::<usize>().ok()?;
    if token_count <= crate::embedding_profile::MAX_TOKENS || count != token_count.to_string() {
        return None;
    }
    Some(InputRefusal { token_count })
}

pub struct EmbedClient {
    backend: EmbedBackend,
    model: String,
    /// The name actually sent to the endpoint — `endpoint_model` from the
    /// config when the serving layer renames the model (LM Studio prefixes
    /// "text-embedding-", Ollama uses short names), otherwise `model`.
    /// Profile validation and response parsing use the canonical `model`;
    /// only the wire request carries this alias.
    wire_model: String,
    /// One interactive request's bound; the batch path scales it.
    base_timeout: std::time::Duration,
    /// Stored/queried vector width. Sent to the endpoint as `dimensions` and
    /// required exactly in the response. v1 never truncates or pads.
    dimensions: usize,
    /// Instruction prepended to a query, never to a document (see
    /// [`crate::config::EmbeddingsConfig::query_prefix`]).
    query_prefix: String,
    /// Instruction prepended to every DOCUMENT, never to a query. Part of the
    /// artifact identity (see [`crate::config::VectorSpec`]).
    doc_prefix: String,
}

#[derive(Debug, serde::Deserialize)]
struct WireExecutionScope {
    scope_id: String,
    transport: crate::embedding_profile::ExecutionTransport,
    backend: String,
    runtime: String,
    compiler: String,
    package_target: String,
    artifact_source: String,
    device_class: String,
    device: String,
    artifact_sha256: String,
    internal_precision: String,
    placement_evidence_sha256: String,
    supported_max_tokens: usize,
    supported_sequence_buckets: BoundedVec<usize, MAX_SEQUENCE_BUCKETS>,
    supported_max_batch_size: usize,
    sequence_capability_evidence_sha256: String,
    performance_evidence_sha256: String,
    compatibility_report_sha256: String,
    accelerated_placement: bool,
}

#[derive(Debug)]
struct ExecutionScope {
    scope_id: String,
    backend: String,
    device_class: String,
    attestation_public_key: String,
}

struct EmbeddedBatch {
    vectors: Vec<Vec<f32>>,
    execution: Option<ExecutionScope>,
}

/// A JSON array whose deserializer never reserves or materializes more than
/// `MAX` elements. The response byte limit bounds raw input; this bound also
/// prevents a compact nested array from amplifying into unbounded typed
/// allocations before shape validation runs.
#[derive(Debug)]
struct BoundedVec<T, const MAX: usize>(Vec<T>);

impl<T, const MAX: usize> BoundedVec<T, MAX> {
    fn as_slice(&self) -> &[T] {
        &self.0
    }

    fn into_vec(self) -> Vec<T> {
        self.0
    }
}

impl<'de, T, const MAX: usize> serde::Deserialize<'de> for BoundedVec<T, MAX>
where
    T: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct BoundedVecVisitor<T, const MAX: usize>(std::marker::PhantomData<T>);

        impl<'de, T, const MAX: usize> serde::de::Visitor<'de> for BoundedVecVisitor<T, MAX>
        where
            T: serde::Deserialize<'de>,
        {
            type Value = BoundedVec<T, MAX>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "an array with at most {MAX} elements")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                if sequence.size_hint().is_some_and(|hint| hint > MAX) {
                    return Err(serde::de::Error::custom(format_args!(
                        "array exceeds the hard limit of {MAX} elements"
                    )));
                }
                let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(MAX));
                while values.len() < MAX {
                    let Some(value) = sequence.next_element()? else {
                        return Ok(BoundedVec(values));
                    };
                    values.push(value);
                }
                if sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {
                    return Err(serde::de::Error::custom(format_args!(
                        "array exceeds the hard limit of {MAX} elements"
                    )));
                }
                Ok(BoundedVec(values))
            }
        }

        deserializer.deserialize_seq(BoundedVecVisitor::<T, MAX>(std::marker::PhantomData))
    }
}

const MAX_EMBEDDING_REQUEST_BYTES: usize = 8 * 1024 * 1024;
const MAX_EMBEDDING_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const MAX_NONCANONICAL_EMBEDDING_COMPONENTS: usize = 16 * 1024;
const MAX_SEQUENCE_BUCKETS: usize = 16;

#[derive(serde::Deserialize)]
struct WireEmbeddingResponse<const MAX_COMPONENTS: usize> {
    data: BoundedVec<
        WireEmbeddingRow<MAX_COMPONENTS>,
        { crate::embedding_profile::MAX_WIRE_BATCH_SIZE },
    >,
    #[serde(default)]
    model: String,
    #[serde(default)]
    cfetch_profile: String,
    #[serde(default)]
    cfetch_profile_manifest_sha256: String,
    #[serde(default)]
    cfetch_admission_policy_sha256: String,
    #[serde(default)]
    cfetch_model_revision: String,
    #[serde(default)]
    cfetch_execution: Option<WireExecutionScope>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLocalErrorEnvelope {
    error: WireLocalError,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLocalError {
    code: String,
    scope_id: String,
    message: String,
}

#[derive(serde::Deserialize)]
struct WireEmbeddingRow<const MAX_COMPONENTS: usize> {
    index: Option<usize>,
    embedding: BoundedVec<f32, MAX_COMPONENTS>,
    #[serde(default)]
    token_count: Option<usize>,
    #[serde(default)]
    sequence_bucket: Option<usize>,
    #[serde(default)]
    truncated: Option<bool>,
    #[serde(default)]
    cfetch_scope_id: Option<String>,
}

#[derive(Debug)]
struct ParsedEmbeddingResponse {
    data: Vec<ParsedEmbeddingRow>,
    model: String,
    cfetch_profile: String,
    cfetch_profile_manifest_sha256: String,
    cfetch_admission_policy_sha256: String,
    cfetch_model_revision: String,
    cfetch_execution: Option<WireExecutionScope>,
}

#[derive(Debug)]
struct ParsedEmbeddingRow {
    index: Option<usize>,
    embedding: Vec<f32>,
    token_count: Option<usize>,
    sequence_bucket: Option<usize>,
    truncated: Option<bool>,
    cfetch_scope_id: Option<String>,
}

impl<const MAX_COMPONENTS: usize> From<WireEmbeddingResponse<MAX_COMPONENTS>>
    for ParsedEmbeddingResponse
{
    fn from(response: WireEmbeddingResponse<MAX_COMPONENTS>) -> Self {
        Self {
            data: response
                .data
                .into_vec()
                .into_iter()
                .map(|row| ParsedEmbeddingRow {
                    index: row.index,
                    embedding: row.embedding.into_vec(),
                    token_count: row.token_count,
                    sequence_bucket: row.sequence_bucket,
                    truncated: row.truncated,
                    cfetch_scope_id: row.cfetch_scope_id,
                })
                .collect(),
            model: response.model,
            cfetch_profile: response.cfetch_profile,
            cfetch_profile_manifest_sha256: response.cfetch_profile_manifest_sha256,
            cfetch_admission_policy_sha256: response.cfetch_admission_policy_sha256,
            cfetch_model_revision: response.cfetch_model_revision,
            cfetch_execution: response.cfetch_execution,
        }
    }
}

fn parse_embedding_response(
    text: &str,
    canonical_profile: bool,
) -> serde_json::Result<ParsedEmbeddingResponse> {
    if canonical_profile {
        serde_json::from_str::<WireEmbeddingResponse<{ crate::embedding_profile::DIMENSIONS }>>(
            text,
        )
        .map(Into::into)
    } else {
        serde_json::from_str::<WireEmbeddingResponse<MAX_NONCANONICAL_EMBEDDING_COMPONENTS>>(text)
            .map(Into::into)
    }
}

#[derive(serde::Serialize)]
struct WireEmbeddingRequest<'model, 'slice, 'text> {
    model: &'model str,
    input: &'slice [&'text str],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    /// Present only for a packaged local plan. An explicit remote endpoint
    /// chooses its own admitted execution scope, while local fallback must
    /// prove that an NPU attempt did not silently answer on a GPU or CPU.
    #[serde(skip_serializing_if = "Option::is_none")]
    cfetch_requested_scope_id: Option<&'model str>,
}

struct BoundedJsonBody {
    bytes: Vec<u8>,
    overflowed: bool,
}

impl std::io::Write for BoundedJsonBody {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|length| length > MAX_EMBEDDING_REQUEST_BYTES)
        {
            self.overflowed = true;
            return Err(std::io::Error::other(
                "embedding request exceeds its byte limit",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn serialize_embedding_request(
    model: &str,
    texts: &[&str],
    dimensions: usize,
    requested_scope_id: Option<&str>,
) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        !texts.is_empty() && texts.len() <= crate::embedding_profile::MAX_WIRE_BATCH_SIZE,
        "embedding requests must contain 1..={} items",
        crate::embedding_profile::MAX_WIRE_BATCH_SIZE
    );
    let request = WireEmbeddingRequest {
        model,
        input: texts,
        dimensions: (dimensions > 0).then_some(dimensions),
        cfetch_requested_scope_id: requested_scope_id,
    };
    let mut writer = BoundedJsonBody {
        bytes: Vec::with_capacity(16 * 1024),
        overflowed: false,
    };
    if let Err(error) = serde_json::to_writer(&mut writer, &request) {
        if writer.overflowed {
            anyhow::bail!(
                "serialized embedding request exceeds the hard limit of {MAX_EMBEDDING_REQUEST_BYTES} bytes"
            );
        }
        return Err(error).context("serialize embedding request");
    }
    Ok(writer.bytes)
}

fn validate_execution_scope(
    scope: Option<WireExecutionScope>,
    requested_scope_id: Option<&str>,
) -> anyhow::Result<ExecutionScope> {
    let scope =
        scope.context("embedding producer omitted its exact cfetch_execution scope attestation")?;
    anyhow::ensure!(
        valid_scope_id(&scope.scope_id),
        "embedding producer scope_id must be at most 128 lowercase slug characters"
    );
    if let Some(requested_scope_id) = requested_scope_id {
        anyhow::ensure!(
            scope.scope_id == requested_scope_id,
            "embedding producer answered with scope {:?}, but the local package plan requested {:?}",
            scope.scope_id,
            requested_scope_id
        );
    }
    let expected_transport = if requested_scope_id.is_some() {
        crate::embedding_profile::ExecutionTransport::SupervisedLocal
    } else {
        crate::embedding_profile::ExecutionTransport::RemoteAttested
    };
    anyhow::ensure!(
        scope.transport == expected_transport,
        "embedding producer transport {:?} does not match the required {:?} route",
        scope.transport.as_str(),
        expected_transport.as_str()
    );
    anyhow::ensure!(
        matches!(scope.device_class.as_str(), "npu" | "gpu" | "cpu"),
        "embedding producer attested invalid device class {:?}",
        scope.device_class
    );
    anyhow::ensure!(
        scope.accelerated_placement,
        "embedding producer did not attest accelerated placement"
    );
    anyhow::ensure!(
        [
            scope.scope_id.as_str(),
            scope.backend.as_str(),
            scope.runtime.as_str(),
            scope.compiler.as_str(),
            scope.package_target.as_str(),
            scope.artifact_source.as_str(),
            scope.internal_precision.as_str(),
            scope.device.as_str(),
        ]
        .into_iter()
        .all(|value| !value.trim().is_empty()),
        "embedding producer execution provenance contains an empty field"
    );
    anyhow::ensure!(
        scope.artifact_sha256.len() == 64
            && scope
                .artifact_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "embedding producer artifact_sha256 must be 64 lowercase hexadecimal characters"
    );
    for (field, digest) in [
        (
            "placement_evidence_sha256",
            scope.placement_evidence_sha256.as_str(),
        ),
        (
            "sequence_capability_evidence_sha256",
            scope.sequence_capability_evidence_sha256.as_str(),
        ),
        (
            "performance_evidence_sha256",
            scope.performance_evidence_sha256.as_str(),
        ),
        (
            "compatibility_report_sha256",
            scope.compatibility_report_sha256.as_str(),
        ),
    ] {
        anyhow::ensure!(
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "embedding producer {field} must be 64 lowercase hexadecimal characters"
        );
    }
    anyhow::ensure!(
        scope.supported_max_tokens == crate::embedding_profile::MAX_TOKENS
            && scope.supported_sequence_buckets.as_slice()
                == crate::embedding_profile::SEQUENCE_BUCKETS
            && scope.supported_max_batch_size == crate::embedding_profile::MAX_WIRE_BATCH_SIZE,
        "embedding producer does not cover the complete cfetch sequence and batch contract"
    );
    let attestation = crate::embedding_profile::BackendScopeAttestation {
        scope_id: &scope.scope_id,
        transport: scope.transport,
        backend: &scope.backend,
        runtime: &scope.runtime,
        compiler: &scope.compiler,
        package_target: &scope.package_target,
        artifact_source: &scope.artifact_source,
        artifact_sha256: &scope.artifact_sha256,
        internal_precision: &scope.internal_precision,
        device: &scope.device,
        device_class: &scope.device_class,
        placement_evidence_sha256: &scope.placement_evidence_sha256,
        supported_max_tokens: scope.supported_max_tokens,
        supported_sequence_buckets: scope.supported_sequence_buckets.as_slice(),
        supported_max_batch_size: scope.supported_max_batch_size,
        sequence_capability_evidence_sha256: &scope.sequence_capability_evidence_sha256,
        performance_evidence_sha256: &scope.performance_evidence_sha256,
        compatibility_report_sha256: &scope.compatibility_report_sha256,
        accelerated_placement: scope.accelerated_placement,
    };
    let attestation_public_key =
        crate::embedding_profile::admitted_backend_attestation_public_key(&attestation);
    anyhow::ensure!(
        attestation_public_key.is_some(),
        "embedding producer scope {:?} is not an admitted backend in this cfetch release",
        scope.scope_id
    );
    Ok(ExecutionScope {
        scope_id: scope.scope_id,
        backend: scope.backend,
        device_class: scope.device_class,
        attestation_public_key: attestation_public_key.expect("checked above"),
    })
}

fn valid_scope_id(value: &str) -> bool {
    if value.is_empty() || value.len() > 128 {
        return false;
    }
    let mut previous_was_separator = true;
    for byte in value.bytes() {
        let separator = matches!(byte, b'.' | b'_' | b'-');
        if !(byte.is_ascii_lowercase() || byte.is_ascii_digit() || separator)
            || (separator && previous_was_separator)
        {
            return false;
        }
        previous_was_separator = separator;
    }
    !previous_was_separator
}

const ATTESTATION_NONCE_HEADER: &str = "x-cfetch-attestation-nonce";
const ATTESTATION_SIGNATURE_HEADER: &str = "x-cfetch-attestation-signature";
const ATTESTATION_DOMAIN: &[u8] = b"cfetch-embedding-response-attestation-v1\0";
fn lowercase_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn decode_lowercase_hex<const N: usize>(value: &str, field: &str) -> anyhow::Result<[u8; N]> {
    anyhow::ensure!(
        value.len() == N * 2
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{field} must be {} lowercase hexadecimal characters",
        N * 2
    );
    let mut decoded = [0u8; N];
    for (index, pair) in value.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        let nibble = |byte: u8| match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            _ => unreachable!("validated lowercase hexadecimal input"),
        };
        decoded[index] = (nibble(pair[0]) << 4) | nibble(pair[1]);
    }
    Ok(decoded)
}

fn attestation_message(nonce: &[u8; 32], request_body: &[u8], response_body: &[u8]) -> Vec<u8> {
    use sha2::Digest as _;

    let request_sha256 = sha2::Sha256::digest(request_body);
    let response_sha256 = sha2::Sha256::digest(response_body);
    let mut message = Vec::with_capacity(ATTESTATION_DOMAIN.len() + 32 + 32 + 32);
    message.extend_from_slice(ATTESTATION_DOMAIN);
    message.extend_from_slice(nonce);
    message.extend_from_slice(&request_sha256);
    message.extend_from_slice(&response_sha256);
    message
}

fn verify_execution_signature(
    public_key_hex: &str,
    signature_hex: &str,
    nonce: &[u8; 32],
    request_body: &[u8],
    response_body: &[u8],
) -> anyhow::Result<()> {
    let public_key_bytes =
        decode_lowercase_hex::<32>(public_key_hex, "admitted attestation public key")?;
    let signature_bytes = decode_lowercase_hex::<64>(signature_hex, ATTESTATION_SIGNATURE_HEADER)?;
    let public_key = ed25519_dalek::VerifyingKey::from_bytes(&public_key_bytes)
        .context("admitted attestation public key is not a valid Ed25519 key")?;
    let signature = ed25519_dalek::Signature::from_bytes(&signature_bytes);
    public_key
        .verify_strict(
            &attestation_message(nonce, request_body, response_body),
            &signature,
        )
        .context("embedding response failed its admitted scope-key signature")
}

fn validate_profile_row_metadata(
    index: usize,
    token_count: Option<usize>,
    sequence_bucket: Option<usize>,
    truncated: Option<bool>,
    row_scope_id: Option<&str>,
    execution_scope_id: &str,
) -> anyhow::Result<()> {
    let token_count = token_count
        .with_context(|| format!("embeddings response index {index} omitted token_count"))?;
    let expected_bucket = crate::embedding_profile::sequence_bucket_for_token_count(token_count)
        .with_context(|| {
            format!(
                "embeddings response index {index} token_count {token_count} is outside the 1..={} profile limit",
                crate::embedding_profile::MAX_TOKENS
            )
        })?;
    anyhow::ensure!(
        sequence_bucket == Some(expected_bucket),
        "embeddings response index {index} used sequence bucket {sequence_bucket:?}, expected smallest fitting bucket {expected_bucket}"
    );
    anyhow::ensure!(
        truncated == Some(false),
        "embeddings response index {index} was truncated or omitted truncation metadata"
    );
    anyhow::ensure!(
        row_scope_id == Some(execution_scope_id),
        "embeddings response index {index} was not produced by the response's single admitted execution scope"
    );
    Ok(())
}

impl std::fmt::Debug for EmbedClient {
    // Manual impl: ureq::Agent's Debug is not part of our contract, and the
    // interesting identity is (url, model) anyway.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let backend = match &self.backend {
            #[cfg(feature = "embedded-embeddings")]
            EmbedBackend::Cpu(_) => "local-cpu",
            EmbedBackend::Endpoint { url, .. } => url.as_str(),
            EmbedBackend::Local { .. } => "package-local",
        };
        f.debug_struct("EmbedClient")
            .field("backend", &backend)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl EmbedClient {
    /// Refuses to construct when embeddings are disabled or unconfigured —
    /// the ONE place that gates every semantic path, so the CLI error is a
    /// single clear line.
    pub fn new(cfg: &EmbeddingsConfig) -> anyhow::Result<EmbedClient> {
        anyhow::ensure!(
            cfg.enabled,
            "embeddings disabled (set embeddings.enabled=true in config)"
        );
        anyhow::ensure!(!cfg.model.is_empty(), "embeddings.model is required");
        if cfg.local_model.is_none() && cfg.model == crate::embedding_profile::MODEL {
            // Every producer of canonical vectors must be admitted before it
            // can write into the shared vector space, regardless of whether
            // the transport is package-local or a configured endpoint.
            crate::embedding_profile::production_availability()?;
        }
        let base_timeout = std::time::Duration::from_secs(cfg.timeout_secs.max(1));
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .max_redirects(0) // with max_redirects_will_error (default true): any 3xx is an Err
            .timeout_global(Some(base_timeout))
            .http_status_as_error(false) // status checked explicitly below
            .build()
            .new_agent();
        let backend = if let Some(dir) = &cfg.local_model {
            #[cfg(feature = "embedded-embeddings")]
            {
                EmbedBackend::Cpu(crate::local_embedding::load(dir, true)?)
            }
            #[cfg(not(feature = "embedded-embeddings"))]
            {
                let _ = dir;
                anyhow::bail!("local vectors require the embedded-embeddings build");
            }
        } else if cfg.endpoint.is_empty() {
            build_package_local_backend(cfg, agent)?
        } else {
            // A configured endpoint is an explicit route and never a hidden
            // fourth step after package-local NPU/GPU/CPU fallback.
            check_endpoint(&cfg.endpoint, &cfg.allow_hosts)?;
            let auth = resolve_auth(&cfg.api_key_env, "embeddings")?;
            EmbedBackend::Endpoint {
                agent,
                url: format!("{}/embeddings", cfg.endpoint.trim_end_matches('/')),
                auth,
            }
        };
        Ok(EmbedClient {
            backend,
            model: cfg.model.clone(),
            wire_model: cfg
                .endpoint_model
                .clone()
                .unwrap_or_else(|| cfg.model.clone()),
            base_timeout,
            dimensions: cfg.dimensions,
            query_prefix: cfg.query_prefix.clone(),
            doc_prefix: cfg.document_prefix.clone(),
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Enforces the profile width. Truncating, padding, or accepting a native
    /// width would create a second vector space inside the same major.
    /// Wrong width and degenerate output are systemic execution failures,
    /// not evidence that retrying each input separately will repair the model.
    fn fit(&self, v: Vec<f32>) -> anyhow::Result<Vec<f32>> {
        if self.dimensions == 0 {
            return Ok(v);
        }
        anyhow::ensure!(
            v.len() == self.dimensions,
            "model {} returned {}-d vectors but cfetch network major {} requires exactly {}; \
             refusing truncation or padding because that would create incompatible vectors",
            self.model,
            v.len(),
            crate::embedding_profile::NETWORK_MAJOR,
            self.dimensions
        );
        if let Some(reason) = vectors::degenerate(&v) {
            anyhow::bail!(
                "model {} returned a degenerate embedding: {reason}",
                self.model
            );
        }
        Ok(v)
    }

    /// Interactive path (recall): one request under the tight base bound.
    /// Embeds DOCUMENTS — text destined for the shared store — applying the
    /// configured document prefix. The bound scales per batched item, because
    /// a large batch on a slow backend is busy, not down.
    ///
    /// Applies `doc_prefix` to every input, then embeds. Kept in ONE place so
    /// a document can never reach the endpoint half-prefixed, and a query can
    /// never reach it prefixed as a document.
    fn embed_prefixed(
        &self,
        texts: &[&str],
        timeout: std::time::Duration,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if self.doc_prefix.is_empty() {
            return self.embed_with_timeout(texts, timeout);
        }
        let owned: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.doc_prefix))
            .collect();
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        self.embed_with_timeout(&refs, timeout)
    }

    /// Batch path (embed-index): the bound scales base + per-item, because a
    /// large batch on a slow backend is busy, not down.
    pub fn embed_documents_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        anyhow::ensure!(
            !texts.is_empty() && texts.len() <= crate::embedding_profile::MAX_WIRE_BATCH_SIZE,
            "embedding requests must contain 1..={} items; split larger batches without changing per-item semantics",
            crate::embedding_profile::MAX_WIRE_BATCH_SIZE
        );
        self.embed_prefixed(texts, batch_timeout(self.base_timeout, texts.len()))
    }

    /// Embeds ONE query, with the configured instruction prefix applied.
    ///
    /// This is the only path that prefixes anything. `embed`/`embed_batch`
    /// stay raw, because what they produce is stored and shared, and a
    /// document embedded with an instruction is not the same artifact.
    pub fn embed_query(&self, query: &str) -> anyhow::Result<Vec<f32>> {
        let text = if self.query_prefix.is_empty() {
            query.to_string()
        } else {
            format!("{}{query}", self.query_prefix)
        };
        self.embed_with_timeout(&[text.as_str()], self.base_timeout)?
            .into_iter()
            .next()
            .context("endpoint returned no vector for the query")
    }

    /// Embeds a batch of texts; returns one vector per input, in input order
    /// (the response's `index` field is honored, not the array order).
    fn embed_with_timeout(
        &self,
        texts: &[&str],
        timeout: std::time::Duration,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        #[cfg(feature = "embedded-embeddings")]
        if let EmbedBackend::Cpu(model) = &self.backend {
            let result = model
                .lock()
                .map_err(|_| anyhow::anyhow!("local model poisoned"))?
                .embed(texts);
            crate::runtime_status::record_inference_attempt(
                crate::runtime_status::InferenceMode::Local,
                crate::runtime_status::InferenceRoute::Local,
                "fastembed-ort-cpu",
                Some("cpu"),
                result.is_ok(),
            );
            return result;
        }
        let (mode, route, result) = match &self.backend {
            #[cfg(feature = "embedded-embeddings")]
            EmbedBackend::Cpu(_) => unreachable!("CPU path returned above"),
            EmbedBackend::Endpoint { agent, url, auth } => (
                crate::runtime_status::InferenceMode::Endpoint,
                crate::runtime_status::endpoint_route(url),
                self.embed_request(agent, url, auth.as_deref(), texts, timeout, None),
            ),
            EmbedBackend::Local { agent, state } => (
                crate::runtime_status::InferenceMode::Local,
                crate::runtime_status::InferenceRoute::Local,
                self.embed_local(agent, state, texts, timeout),
            ),
        };
        match &result {
            Ok(batch) => {
                let (backend, device_class) = match &batch.execution {
                    Some(execution) => (
                        execution.backend.as_str(),
                        Some(execution.device_class.as_str()),
                    ),
                    None => ("endpoint", None),
                };
                crate::runtime_status::record_inference_attempt(
                    mode,
                    route,
                    backend,
                    device_class,
                    true,
                );
            }
            Err(_) => crate::runtime_status::record_inference_attempt(
                mode,
                route,
                if self.model == crate::embedding_profile::MODEL {
                    "profile-producer"
                } else {
                    "endpoint"
                },
                None,
                false,
            ),
        }
        result.map(|batch| batch.vectors)
    }

    fn embed_local(
        &self,
        agent: &ureq::Agent,
        state: &std::sync::Mutex<LocalBackendState>,
        texts: &[&str],
        timeout: std::time::Duration,
    ) -> anyhow::Result<EmbeddedBatch> {
        let mut state = state
            .lock()
            .map_err(|_| anyhow::anyhow!("package-local adapter supervisor lock was poisoned"))?;
        let selected = state.selected_scope;
        let mut attempts = Vec::new();
        if let Some(index) = selected.filter(|index| !state.unavailable_scopes.contains(index)) {
            attempts.push(index);
        }
        attempts.extend(
            (0..state.ordered_scope_ids.len()).filter(|index| {
                Some(*index) != selected && !state.unavailable_scopes.contains(index)
            }),
        );

        let mut unavailable: Vec<String> = state
            .unavailable_scopes
            .iter()
            .map(|index| state.ordered_scope_ids[*index].clone())
            .collect();
        for index in attempts {
            let scope_id = state.ordered_scope_ids[index].clone();
            let endpoint = state.supervisor.endpoint()?;
            let url = format!("{}/embeddings", endpoint.base_url.trim_end_matches('/'));
            let result = self.embed_request(
                agent,
                &url,
                Some(&endpoint.authorization),
                texts,
                timeout,
                Some(&scope_id),
            );
            if result
                .as_ref()
                .err()
                .is_some_and(|error| error.downcast_ref::<AdapterTransportError>().is_some())
            {
                state.unavailable_scopes.insert(index);
                state.selected_scope = None;
                unavailable.push(scope_id);
                // Confirm cleanup before another scope can execute. A live
                // hung dispatcher latches the supervisor off; a confirmed
                // crash can consume its one restart for the next scope.
                state.supervisor.restart_after_transport_failure()?;
                continue;
            }
            match result {
                Ok(batch) => {
                    state.selected_scope = Some(index);
                    return Ok(batch);
                }
                Err(error) if error.downcast_ref::<ScopeUnavailableError>().is_some() => {
                    state.selected_scope = None;
                    state.unavailable_scopes.insert(index);
                    unavailable.push(scope_id);
                }
                Err(error) => return Err(error),
            }
        }
        anyhow::bail!(
            "every admitted package-local execution scope was unavailable in NPU, GPU, accelerated CPU order: {}",
            unavailable.join(", ")
        )
    }

    fn embed_request(
        &self,
        agent: &ureq::Agent,
        url: &str,
        auth: Option<&str>,
        texts: &[&str],
        timeout: std::time::Duration,
        requested_scope_id: Option<&str>,
    ) -> anyhow::Result<EmbeddedBatch> {
        // Serialize directly into a bounded writer: even a config or caller
        // bug cannot materialize and send an arbitrarily large HTTP body.
        let body = serialize_embedding_request(
            &self.wire_model,
            texts,
            self.dimensions,
            requested_scope_id,
        )?;
        // A fresh challenge binds these exact request and response bytes to
        // the admitted scope key. On supervised-local it checks response
        // consistency inside the separately hashed and supervised package
        // boundary; on remote-attested it authenticates the admitted producer.
        let attestation_nonce = rand::random::<[u8; 32]>();
        let mut req = agent
            .post(url)
            .config()
            .timeout_global(Some(timeout)) // per-request override of the agent bound
            .build()
            .header("content-type", "application/json")
            .header(ATTESTATION_NONCE_HEADER, lowercase_hex(&attestation_nonce));
        if let Some(auth) = auth {
            req = req.header("authorization", auth);
        }
        let mut resp = req
            .send(body.as_slice())
            .map_err(|error| AdapterTransportError(format!("POST {url}: {error}")))?;
        let status = resp.status();
        let response_signature = resp
            .headers()
            .get(ATTESTATION_SIGNATURE_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let text = resp
            .body_mut()
            .with_config()
            .limit(MAX_EMBEDDING_RESPONSE_BYTES as u64)
            .read_to_string()
            .map_err(|error| response_body_error(error, url, requested_scope_id))?;
        anyhow::ensure!(
            text.len() <= MAX_EMBEDDING_RESPONSE_BYTES,
            "embeddings endpoint response exceeds {MAX_EMBEDDING_RESPONSE_BYTES} bytes"
        );
        if !status.is_success()
            && requested_scope_id.is_some()
            && status.as_u16() == 503
            && let Ok(unavailable) = serde_json::from_str::<WireLocalErrorEnvelope>(&text)
            && unavailable.error.code == "scope_unavailable"
            && Some(unavailable.error.scope_id.as_str()) == requested_scope_id
            && unavailable.error.message
                == "requested admitted scope could not initialize or execute"
        {
            return Err(ScopeUnavailableError(format!(
                "package-local scope {} is unavailable",
                unavailable.error.scope_id
            ))
            .into());
        }
        if let Some(refusal) = input_refusal(status.as_u16(), &text) {
            return Err(refusal.into());
        }
        anyhow::ensure!(
            status.is_success(),
            "embeddings endpoint returned {status}: {}",
            snippet(&text)
        );
        // Every transport that claims the canonical model must prove the same
        // profile and admitted execution scope. Noncanonical endpoint models
        // remain separate, explicitly configured vector spaces.
        let is_canonical = self.model == crate::embedding_profile::MODEL;
        let parsed = parse_embedding_response(&text, is_canonical)
            .with_context(|| format!("unparseable embeddings response: {}", snippet(&text)))?;
        if is_canonical {
            anyhow::ensure!(
                parsed.cfetch_profile == crate::embedding_profile::PROFILE_ID
                    && parsed.cfetch_profile_manifest_sha256
                        == crate::embedding_profile::manifest_sha256()
                    && parsed.cfetch_model_revision == crate::embedding_profile::MODEL_REVISION,
                "embeddings endpoint is not profile-attested for {} (vector-space profile/revision mismatch)",
                crate::embedding_profile::PROFILE_ID
            );
        }
        let execution = if is_canonical {
            anyhow::ensure!(
                parsed.cfetch_admission_policy_sha256
                    == crate::embedding_profile::admission_policy_sha256(),
                "embeddings endpoint admission-policy digest does not match this cfetch release"
            );
            let execution = validate_execution_scope(parsed.cfetch_execution, requested_scope_id)?;
            let signature = response_signature
                .context("admitted embedding producer omitted its response-signature header")?;
            verify_execution_signature(
                &execution.attestation_public_key,
                &signature,
                &attestation_nonce,
                &body,
                text.as_bytes(),
            )?;
            Some(execution)
        } else {
            None
        };
        anyhow::ensure!(
            parsed.model == self.wire_model,
            "embeddings endpoint answered with model {:?}, requested {:?}",
            parsed.model,
            self.wire_model
        );
        anyhow::ensure!(
            parsed.data.len() == texts.len(),
            "embeddings endpoint returned {} vector(s) for {} input(s)",
            parsed.data.len(),
            texts.len()
        );
        // Every index is explicit, unique and in range. Merely sorting rows is
        // insufficient: duplicates can otherwise attach one semantic meaning
        // to two different blocks while still returning the expected count.
        let mut ordered = vec![None; texts.len()];
        for row in parsed.data {
            let index = row.index.context("embeddings response row has no index")?;
            anyhow::ensure!(
                index < texts.len(),
                "embeddings endpoint returned out-of-range index {index} for {} input(s)",
                texts.len()
            );
            anyhow::ensure!(
                ordered[index].is_none(),
                "embeddings endpoint returned duplicate index {index}"
            );
            if is_canonical {
                let execution_scope_id = execution
                    .as_ref()
                    .expect("canonical profile validated one execution scope")
                    .scope_id
                    .as_str();
                validate_profile_row_metadata(
                    index,
                    row.token_count,
                    row.sequence_bucket,
                    row.truncated,
                    row.cfetch_scope_id.as_deref(),
                    execution_scope_id,
                )?;
            }
            ordered[index] = Some(self.fit(row.embedding)?);
        }
        let vectors = ordered
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                row.with_context(|| format!("embeddings endpoint omitted index {index}"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(EmbeddedBatch { vectors, execution })
    }
}

/// First ~120 chars of an error body — enough to diagnose, never a dump.
pub(crate) fn snippet(text: &str) -> String {
    text.chars().take(120).collect()
}

pub struct EmbedIndexReport {
    /// Vectors derived from the endpoint in this run.
    pub embedded: usize,
    /// Vectors taken from the shared store instead of being re-derived.
    pub imported: usize,
    pub total_blocks: usize,
}

/// Derives the vectors this storage group is still missing.
///
/// COMPUTE-ONCE: the shared store in the tree is the record. The run first
/// takes everything it already holds (`hydrate`), then embeds only content
/// hashes no host has derived yet — writing each batch to the SHARED store
/// before caching it locally, so an interrupted run leaves its work where the
/// next run (on any host) will find it. A missing vector is the whole
/// resumability contract; nothing else is remembered between runs.
pub fn run(
    conn: &mut Connection,
    client: &EmbedClient,
    batch: usize,
    store: &mut vectors::VectorStore,
) -> anyhow::Result<EmbedIndexReport> {
    anyhow::ensure!(
        (1..=crate::embedding_profile::MAX_WIRE_BATCH_SIZE).contains(&batch),
        "embed-index batch must be between 1 and {}",
        crate::embedding_profile::MAX_WIRE_BATCH_SIZE
    );
    let spec = store.spec().clone();
    anyhow::ensure!(
        spec.model == client.model() && spec.dim == client.dimensions(),
        "the shared store is {} at {} dimensions, the client is {} at {} — one artifact, one spec",
        spec.model,
        spec.dim,
        client.model(),
        client.dimensions()
    );
    if index::ensure_vector_spec(conn, &spec)? {
        println!(
            "embedding spec changed -> local vector cache dropped, re-filling for {}",
            spec.model
        );
    }
    let imported = vectors::hydrate(conn, store)?;
    if imported > 0 {
        println!(
            "imported {imported} vector(s) from the shared store (already derived by this group)"
        );
    }
    let mut embedded = 0usize;
    // Refusals stay missing in the durable store, but must not be selected
    // again during this run or a poisoned head would starve later inputs.
    let mut refused = std::collections::HashSet::new();
    let mut pending = index::hashes_without_vectors(conn, &spec, batch)?;
    if !pending.is_empty() {
        // The write lock is taken only when there IS something to derive: a
        // host that only reads never needs the store to be writable.
        let mut writer = store.begin_write()?;
        loop {
            let refused_before = refused.len();
            let texts: Vec<&str> = pending.iter().map(|(_, t)| t.as_str()).collect();
            // Only a proven input refusal justifies isolating the rows. A
            // transport, auth, scope, protocol, or model failure stops at once;
            // turning those into singleton requests multiplies broken work.
            // An empty vector marks a refused row, preserving input alignment.
            let vectors = match client.embed_documents_batch(&texts) {
                Ok(v) => v,
                Err(first) if first.downcast_ref::<InputRefusal>().is_some() => {
                    if texts.len() == 1 {
                        eprintln!("cfetch embed-index: skipping one block this run: {first:#}");
                        refused.insert(pending[0].0.clone());
                        vec![Vec::new()]
                    } else {
                        eprintln!("cfetch embed-index: input refused ({first:#}); isolating rows");
                        let mut single: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
                        for ((hash, _), t) in pending.iter().zip(&texts) {
                            match client.embed_documents_batch(std::slice::from_ref(t)) {
                                Ok(mut v) => single.append(&mut v),
                                Err(row) if row.downcast_ref::<InputRefusal>().is_some() => {
                                    eprintln!("cfetch embed-index: skipping one block this run: {row:#}");
                                    refused.insert(hash.clone());
                                    single.push(Vec::new());
                                }
                                Err(error) => return Err(error)
                                    .context("embedding stopped during input isolation; earlier committed batches are kept"),
                            }
                        }
                        single
                    }
                }
                Err(error) => {
                    return Err(error)
                        .context("embedding stopped; earlier committed batches are kept");
                }
            };
            // Record first, cache second: the shared artifact is what the
            // group keeps, the local row is a convenience. Pairs, not two
            // zipped vectors: skipped rows must not shift the alignment.
            let mut cached: Vec<(&String, Vec<f32>)> = Vec::new();
            for ((hash, _), vector) in pending.iter().zip(&vectors) {
                if vector.is_empty() {
                    continue;
                }
                if writer.put(hash, vector)? {
                    cached.push((hash, vector.clone()));
                } else {
                    let retained = writer.get_retained(hash)?.with_context(|| {
                        format!(
                            "derive-once store reported existing content {hash} but retained no record"
                        )
                    })?;
                    cached.push((hash, retained));
                }
            }
            let succeeded = cached.len();
            anyhow::ensure!(
                succeeded > 0 || refused.len() > refused_before,
                "embedding made no progress: no cacheable vectors or newly refused inputs; earlier committed work is kept"
            );
            if succeeded > 0 {
                writer.flush()?;
                let tx = conn.transaction()?;
                for (hash, vector) in &cached {
                    index::insert_vector(&tx, hash, &spec, vector)?;
                }
                tx.commit()?;
                embedded += succeeded;
                let (done, total) = index::vector_coverage(conn, &spec)?;
                println!("embedded {done}/{total} blocks");
            }
            // Widen the window by this run's refused keys, then remove them.
            // Even an entirely refused batch advances to remaining work.
            pending =
                index::hashes_without_vectors(conn, &spec, batch.saturating_add(refused.len()))?
                    .into_iter()
                    .filter(|(hash, _)| !refused.contains(hash))
                    .take(batch)
                    .collect();
            if pending.is_empty() {
                break;
            }
        }
    }
    anyhow::ensure!(
        refused.is_empty(),
        "embedding incomplete: {} input(s) exceeded the profile token limit; \
         {embedded} newly derived vector(s) and earlier committed work are kept",
        refused.len()
    );
    let (_, total_blocks) = index::vector_coverage(conn, &spec)?;
    Ok(EmbedIndexReport {
        embedded,
        imported,
        total_blocks,
    })
}

/// Brings the local vector cache and shared content-addressed store up to the
/// current Markdown generation. Both the CLI and the daemon's change-driven
/// worker use this exact path.
pub fn sync_configured(cfg: &Config, batch: usize) -> anyhow::Result<(EmbedIndexReport, usize)> {
    // Hydration and peer ingress can publish canonical artifacts without
    // constructing an EmbedClient, so they enforce admission independently.
    cfg.embeddings.available()?;
    let spec = cfg.embeddings.spec();
    let mut store = vectors::VectorStore::open(&cfg.brain_root, &spec)?;
    let mut conn = index::ensure_fresh(
        &crate::paths::state_dir(),
        &cfg.brain_root,
        None,
        &cfg.rings(),
    )?;
    if index::ensure_vector_spec(&conn, &spec)? {
        println!(
            "embedding spec changed -> local vector cache dropped, re-filling for {}",
            spec.model
        );
    }
    let shared_imported = vectors::hydrate(&conn, &store)?;
    if shared_imported > 0 {
        println!(
            "imported {shared_imported} vector(s) from the shared store (already derived by this group)"
        );
    }
    let pending = index::hashes_without_vectors(&conn, &spec, 1)?;
    let report = if pending.is_empty() {
        let (_, total_blocks) = index::vector_coverage(&conn, &spec)?;
        EmbedIndexReport {
            embedded: 0,
            imported: shared_imported,
            total_blocks,
        }
    } else {
        let client = EmbedClient::new(&cfg.embeddings)?;
        let mut report = run(&mut conn, &client, batch, &mut store)?;
        report.imported += shared_imported;
        report
    };
    Ok((report, store.len()))
}

/// CLI entry for `cfetch embed-index`.
pub fn embed_index_cmd(batch: usize) -> anyhow::Result<()> {
    let cfg = Config::load()?;
    let spec = cfg.embeddings.spec();
    let (report, shared_artifacts) = sync_configured(&cfg, batch)?;
    println!(
        "embed-index complete: {} embedded this run, {} imported from existing artifacts, {} block(s) total",
        report.embedded, report.imported, report.total_blocks
    );
    println!(
        "shared vector store: {} ({} artifact(s), {} at {} dimensions)",
        crate::paths::shared_vector_dir(&cfg.brain_root).display(),
        shared_artifacts,
        spec.precision.as_str(),
        spec.dim
    );
    Ok(())
}

/// The `cfetch status` line for semantic recall. Coverage leads, because a
/// half-embedded index is exactly the state that would otherwise degrade
/// every hybrid query without anyone noticing.
pub fn coverage_status_line(
    spec: &crate::config::VectorSpec,
    embedded: usize,
    total: usize,
    shared: usize,
) -> String {
    let health = if total == 0 {
        "index is empty".to_string()
    } else if embedded >= total {
        "complete".to_string()
    } else {
        format!(
            "run cfetch embed-index for the remaining {}",
            total - embedded
        )
    };
    format!(
        "semantic: {embedded}/{total} blocks embedded — {health}\n  \
         {} at {} dims ({}), shared store holds {shared} artifact(s)",
        spec.model,
        spec.dim,
        spec.precision.as_str()
    )
}

/// A semantic/hybrid answer, plus what was WRONG with it. `note` is the
/// project's anti-silent-degradation contract in one field: a memory system
/// that quietly returns worse answers is the failure this exists to prevent,
/// so partial or absent vector coverage travels with the result and every
/// caller surfaces it.
#[derive(Debug)]
pub struct SemanticRecall {
    pub hits: Vec<index::Hit>,
    pub note: Option<String>,
}

/// The coverage line, or None when nothing is degraded.
fn coverage_note(embedded: usize, total: usize) -> Option<String> {
    if total == 0 || embedded >= total {
        return None;
    }
    Some(if embedded == 0 {
        format!(
            "semantic: 0/{total} blocks embedded — answering lexically only; run cfetch embed-index"
        )
    } else {
        format!(
            "semantic: {embedded}/{total} blocks embedded — {} block(s) cannot be reached \
             semantically; run cfetch embed-index",
            total - embedded
        )
    })
}

fn prepare_query_vector(vector: &mut [f32], precision: crate::config::Precision) {
    if precision != crate::config::Precision::I8 {
        index::l2_normalize(vector);
    }
}

/// Semantic (`--semantic`) or hybrid (`--hybrid`) recall: embeds the query,
/// then ranks by cosine alone or fuses with the BM25 list via RRF.
pub fn semantic_hits(
    cfg: &Config,
    conn: &Connection,
    query: &str,
    limit: usize,
    hybrid: bool,
    prefixes: &[String],
) -> anyhow::Result<SemanticRecall> {
    let client = EmbedClient::new(&cfg.embeddings).map_err(|e| {
        crate::runtime_status::record_inference_initialization_failure();
        anyhow::anyhow!("semantic recall unavailable: {e}")
    })?;
    let spec = cfg.embeddings.spec();
    // The shared store is the record: take whatever this group already
    // derived before judging our own coverage.
    let store = vectors::VectorStore::open(&cfg.brain_root, &spec)?;
    vectors::hydrate(conn, &store)?;
    let (embedded, total) = index::vector_coverage(conn, &spec)?;
    let note = coverage_note(embedded, total);
    if embedded == 0 {
        // Nothing to rank against. Answer lexically — but say so; a silently
        // lexical "hybrid" is the degradation this project bans.
        return Ok(SemanticRecall {
            hits: index::recall_in(conn, query, limit, prefixes)?,
            note,
        });
    }
    let embedded_query = client.embed_query(query);
    let mut qv = match embedded_query {
        Ok(qv) => qv,
        Err(e) => {
            // Configured but not answering: the vectors are here, the thing
            // that would place the QUERY among them is not. Degrade to
            // lexical — and say exactly that, on one line, with the reason.
            // (An UNCONFIGURED endpoint is a different thing and still
            // errors above: you cannot degrade a feature you never enabled.)
            let reason = format!("semantic: query embedding failed ({e:#}) — answering lexically");
            let reason = reason.replace('\n', " ");
            return Ok(SemanticRecall {
                hits: index::recall_in(conn, query, limit, prefixes)?,
                note: Some(match note {
                    Some(coverage) => format!("{coverage}; {reason}"),
                    None => reason,
                }),
            });
        }
    };
    // Canonical INT8 uses the same direct f32 -> maxabs/RNE codec for queries
    // and documents. An extra f32 L2 pass is mathematically redundant but can
    // move a component across an INT8 rounding boundary. Legacy floating-point
    // specs retain their normalized query representation.
    prepare_query_vector(&mut qv, spec.precision);
    let hits = if hybrid {
        index::hybrid_recall(conn, &spec, query, &qv, limit, cfg.recall.rrf_k, prefixes)?
    } else {
        index::semantic_recall(conn, &spec, &qv, limit, prefixes)?
    };
    Ok(SemanticRecall { hits, note })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testhttp::{http_response, spawn_server};

    #[test]
    fn canonical_int8_query_skips_the_extra_float_normalization_pass() {
        let original = vec![-1.0_f32, 0.765_432_1, 0.888_888_9];
        let mut int8 = original.clone();
        prepare_query_vector(&mut int8, crate::config::Precision::I8);
        assert_eq!(int8, original);

        let mut legacy = original.clone();
        prepare_query_vector(&mut legacy, crate::config::Precision::F32);
        assert_ne!(legacy, original);
    }

    #[test]
    fn package_key_signature_binds_nonce_request_and_exact_response_bytes() {
        use ed25519_dalek::Signer as _;
        let key = ed25519_dalek::SigningKey::from_bytes(&rand::random::<[u8; 32]>());
        let nonce = [9; 32];
        let request = br#"{"model":"m","input":["one"]}"#;
        let response = br#"{"data":[{"index":0}]}"#;
        let signature = key.sign(&attestation_message(&nonce, request, response));
        let public_key = lowercase_hex(key.verifying_key().as_bytes());
        let signature = lowercase_hex(&signature.to_bytes());

        verify_execution_signature(&public_key, &signature, &nonce, request, response).unwrap();
        assert!(
            verify_execution_signature(
                &public_key,
                &signature,
                &nonce,
                request,
                br#"{"data":[{"index":1}]}"#,
            )
            .is_err()
        );
        let mut another_nonce = nonce;
        another_nonce[0] ^= 1;
        assert!(
            verify_execution_signature(&public_key, &signature, &another_nonce, request, response,)
                .is_err()
        );
    }

    #[test]
    fn canonical_rows_freeze_token_bucket_truncation_and_single_scope() {
        validate_profile_row_metadata(
            0,
            Some(33),
            Some(64),
            Some(false),
            Some("scope-a"),
            "scope-a",
        )
        .unwrap();
        for (token_count, bucket, truncated, scope, expected) in [
            (Some(0), Some(32), Some(false), Some("scope-a"), "outside"),
            (
                Some(33),
                Some(128),
                Some(false),
                Some("scope-a"),
                "smallest fitting",
            ),
            (Some(33), Some(64), Some(true), Some("scope-a"), "truncated"),
            (
                Some(33),
                Some(64),
                Some(false),
                Some("scope-b"),
                "single admitted",
            ),
        ] {
            let error =
                validate_profile_row_metadata(0, token_count, bucket, truncated, scope, "scope-a")
                    .unwrap_err()
                    .to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?}, got {error:?}"
            );
        }
        let error = validate_profile_row_metadata(
            0,
            Some(crate::embedding_profile::MAX_TOKENS + 1),
            Some(crate::embedding_profile::MAX_TOKENS),
            Some(false),
            Some("scope-a"),
            "scope-a",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("outside"), "{error}");
    }

    #[test]
    fn serialized_requests_are_bounded_before_send() {
        let oversized = "x".repeat(MAX_EMBEDDING_REQUEST_BYTES);
        let error = serialize_embedding_request("test-model", &[oversized.as_str()], 2, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("hard limit"), "{error}");

        let too_many = vec!["x"; crate::embedding_profile::MAX_WIRE_BATCH_SIZE + 1];
        let error = serialize_embedding_request("test-model", &too_many, 2, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("1..=64"), "{error}");
    }

    #[test]
    fn only_the_exact_adapter_overlength_envelope_is_an_input_refusal() {
        let message = |count: &str| {
            format!(
                "prefixed input contains {count} tokens; the profile limit is {} and truncation is forbidden",
                crate::embedding_profile::MAX_TOKENS
            )
        };
        let body = serde_json::json!({"error": message("2049")}).to_string();
        assert_eq!(input_refusal(400, &body).unwrap().token_count, 2049);
        for status in [200, 401, 413, 429, 500, 503] {
            assert!(
                input_refusal(status, &body).is_none(),
                "HTTP {status} must stop"
            );
        }
        for count in [
            "2048",
            "0",
            "-2049",
            "+2049",
            "02049",
            "2049.0",
            "2049 ",
            "99999999999999999999999999999999999999",
        ] {
            let body = serde_json::json!({"error": message(count)}).to_string();
            assert!(
                input_refusal(400, &body).is_none(),
                "noncanonical/impossible count {count}"
            );
        }
        for body in [
            "not json".to_string(),
            serde_json::json!({"error": "cfetch_requested_scope_id must name an exact scope in this target package"}).to_string(),
            serde_json::json!({"error": {"message": message("2049")}}).to_string(),
            serde_json::json!({"error": message("2049"), "unexpected": true}).to_string(),
            serde_json::json!({"error": format!("{} extra", message("2049"))}).to_string(),
            serde_json::json!({"error": message("2049").replace("2048", "4096")}).to_string(),
            format!(r#"{{"error":{},"error":{}}}"#, serde_json::json!(message("2049")), serde_json::json!(message("2049"))),
        ] {
            assert!(input_refusal(400, &body).is_none(), "arbitrary 400 must stop: {body}");
        }
    }

    #[test]
    fn only_supervised_response_body_failures_trigger_adapter_restart() {
        let local = response_body_error(
            ureq::Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "truncated",
            )),
            "http://127.0.0.1:1/v1/embeddings",
            Some("local-scope"),
        );
        assert!(local.downcast_ref::<AdapterTransportError>().is_some());

        let configured_endpoint = response_body_error(
            ureq::Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "truncated",
            )),
            "https://example.invalid/embeddings",
            None,
        );
        assert!(
            configured_endpoint
                .downcast_ref::<AdapterTransportError>()
                .is_none()
        );
    }

    #[test]
    fn local_request_binds_the_exact_scope_inside_the_signed_body() {
        let body = serialize_embedding_request(
            crate::embedding_profile::MODEL,
            &["already prefixed"],
            crate::embedding_profile::DIMENSIONS,
            Some("intel-lunar-lake-npu-openvino-v1"),
        )
        .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            value["cfetch_requested_scope_id"],
            "intel-lunar-lake-npu-openvino-v1"
        );

        let remote = serialize_embedding_request("test-model", &["text"], 2, None).unwrap();
        let remote: serde_json::Value = serde_json::from_slice(&remote).unwrap();
        assert!(remote.get("cfetch_requested_scope_id").is_none());
    }

    #[test]
    fn response_deserialization_refuses_more_than_64_rows() {
        let rows = std::iter::repeat_n(
            r#"{"index":0,"embedding":[]}"#,
            crate::embedding_profile::MAX_WIRE_BATCH_SIZE + 1,
        )
        .collect::<Vec<_>>()
        .join(",");
        let body = format!(r#"{{"data":[{rows}]}}"#);
        let error = parse_embedding_response(&body, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("hard limit of 64"), "{error}");
    }

    #[test]
    fn response_deserialization_bounds_canonical_and_noncanonical_widths() {
        let canonical_components = vec!["0"; crate::embedding_profile::DIMENSIONS + 1].join(",");
        let canonical =
            format!(r#"{{"data":[{{"index":0,"embedding":[{canonical_components}]}}]}}"#);
        let error = parse_embedding_response(&canonical, true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("hard limit of 768"), "{error}");

        let global_components = vec!["0"; MAX_NONCANONICAL_EMBEDDING_COMPONENTS + 1].join(",");
        let noncanonical =
            format!(r#"{{"data":[{{"index":0,"embedding":[{global_components}]}}]}}"#);
        let error = parse_embedding_response(&noncanonical, false)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!(
                "hard limit of {MAX_NONCANONICAL_EMBEDDING_COMPONENTS}"
            )),
            "{error}"
        );
    }

    /// OpenAI-shaped response with one deterministic 2-d vector per input:
    /// input i (0-based, per request) -> [seed + i, 1.0]. Data rows are
    /// emitted in REVERSED order to prove the client honors `index`.
    fn canned_embeddings(body: &str, seed: f32) -> String {
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        let n = v["input"].as_array().unwrap().len();
        let rows: Vec<String> = (0..n)
            .rev()
            .map(|i| {
                format!(
                    r#"{{"object":"embedding","index":{i},"embedding":[{},1.0]}}"#,
                    seed + i as f32
                )
            })
            .collect();
        http_response(
            200,
            &format!(
                r#"{{"object":"list","model":{},"cfetch_profile":"{}","cfetch_profile_manifest_sha256":"{}","cfetch_model_revision":"{}","data":[{}]}}"#,
                v["model"],
                crate::embedding_profile::PROFILE_ID,
                crate::embedding_profile::manifest_sha256(),
                crate::embedding_profile::MODEL_REVISION,
                rows.join(",")
            ),
        )
    }

    fn client_for(url: &str) -> EmbedClient {
        EmbedClient::new(&EmbeddingsConfig {
            enabled: true,
            endpoint: url.to_string(),
            model: "test-model".to_string(),
            dimensions: 2,
            ..EmbeddingsConfig::default()
        })
        .unwrap()
    }

    fn client_for_auth(url: &str, auth: Option<String>) -> EmbedClient {
        let mut client = client_for(url);
        let EmbedBackend::Endpoint {
            auth: backend_auth, ..
        } = &mut client.backend
        else {
            unreachable!("test client always uses an endpoint")
        };
        *backend_auth = auth;
        client
    }

    fn attested_rows(model: &str, rows: &str) -> String {
        http_response(
            200,
            &format!(
                r#"{{"object":"list","model":{model:?},"cfetch_profile":"{}","cfetch_profile_manifest_sha256":"{}","cfetch_model_revision":"{}","data":[{rows}]}}"#,
                crate::embedding_profile::PROFILE_ID,
                crate::embedding_profile::manifest_sha256(),
                crate::embedding_profile::MODEL_REVISION,
            ),
        )
    }

    #[cfg(unix)]
    #[test]
    fn packaged_local_selection_falls_back_npu_gpu_cpu_then_caches_success() {
        use sha2::Digest as _;
        use std::os::unix::fs::PermissionsExt as _;

        let requested = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = requested.clone();
        let (server_url, _, _) = spawn_server(move |_, body| {
            let request: serde_json::Value = serde_json::from_str(body).unwrap();
            let scope = request["cfetch_requested_scope_id"]
                .as_str()
                .unwrap()
                .to_string();
            observed.lock().unwrap().push(scope.clone());
            if scope != "cpu-scope" {
                return http_response(
                    503,
                    &format!(
                        r#"{{"error":{{"code":"scope_unavailable","scope_id":{scope:?},"message":"requested admitted scope could not initialize or execute"}}}}"#
                    ),
                );
            }
            canned_embeddings(body, 1.0)
        });

        let directory = tempfile::tempdir().unwrap();
        let adapter = directory.path().join("fake-local-adapter");
        let package_manifest = directory.path().join("package-manifest.json");
        let ready_url = format!("{server_url}/v1");
        std::fs::write(
            &adapter,
            format!(
                "#!/bin/sh\nIFS= read -r secret\nprintf '%s\\n' '{{\"schema_version\":1,\"url\":\"{ready_url}\",\"scope_ids\":[\"npu-scope\",\"gpu-scope\",\"cpu-scope\"]}}'\ncat >/dev/null\n"
            ),
        )
        .unwrap();
        std::fs::set_permissions(&adapter, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::write(&package_manifest, "{\"schema_version\":1}\n").unwrap();
        let digest =
            crate::hashing::hex_lower(sha2::Sha256::digest(std::fs::read(&adapter).unwrap()));
        let package_manifest_sha256 = crate::hashing::hex_lower(sha2::Sha256::digest(
            std::fs::read(&package_manifest).unwrap(),
        ));
        let launch = crate::local_adapter::AdapterLaunch {
            binary: adapter,
            sha256: digest,
            package_manifest,
            package_manifest_sha256,
            ordered_scope_ids: vec!["npu-scope".into(), "gpu-scope".into(), "cpu-scope".into()],
        };
        let cache = std::sync::OnceLock::new();
        let first_state = cached_local_backend_state(&cache, launch.clone()).unwrap();
        let second_state = cached_local_backend_state(&cache, launch).unwrap();
        assert!(std::sync::Arc::ptr_eq(&first_state, &second_state));

        let client = |state| EmbedClient {
            backend: EmbedBackend::Local {
                agent: ureq::Agent::config_builder()
                    .max_redirects(0)
                    .http_status_as_error(false)
                    .build()
                    .new_agent(),
                state,
            },
            model: "test-model".into(),
            wire_model: "test-model".into(),
            base_timeout: std::time::Duration::from_secs(2),
            dimensions: 2,
            query_prefix: String::new(),
            doc_prefix: String::new(),
        };
        client(first_state)
            .embed_documents_batch(&["first"])
            .unwrap();
        client(second_state)
            .embed_documents_batch(&["second"])
            .unwrap();
        assert_eq!(
            *requested.lock().unwrap(),
            ["npu-scope", "gpu-scope", "cpu-scope", "cpu-scope"]
        );
    }

    /// Response whose vectors are `width` long regardless of the requested
    /// `dimensions` — the endpoint that ignores the parameter.
    fn canned_width(body: &str, width: usize) -> String {
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        let n = v["input"].as_array().unwrap().len();
        let rows: Vec<String> = (0..n)
            .map(|i| {
                let comps: Vec<String> = (0..width)
                    .map(|d| format!("{}", (i + d + 1) as f32))
                    .collect();
                format!(
                    r#"{{"object":"embedding","index":{i},"embedding":[{}]}}"#,
                    comps.join(",")
                )
            })
            .collect();
        http_response(
            200,
            &format!(
                r#"{{"object":"list","model":{},"cfetch_profile":"{}","cfetch_profile_manifest_sha256":"{}","cfetch_model_revision":"{}","data":[{}]}}"#,
                v["model"],
                crate::embedding_profile::PROFILE_ID,
                crate::embedding_profile::manifest_sha256(),
                crate::embedding_profile::MODEL_REVISION,
                rows.join(",")
            ),
        )
    }

    /// Response honoring the requested `dimensions` (Matryoshka-style
    /// truncation at the endpoint), falling back to `native` when absent.
    fn canned_honoring_dimensions(body: &str, native: usize) -> String {
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        let width = v["dimensions"]
            .as_u64()
            .map(|d| d as usize)
            .unwrap_or(native);
        canned_width(body, width.min(native))
    }

    fn spec_for(dim: usize) -> crate::config::VectorSpec {
        EmbeddingsConfig {
            model: "test-model".into(),
            dimensions: dim,
            ..EmbeddingsConfig::default()
        }
        .spec()
    }

    // ---- SSRF guard ----

    #[test]
    fn inference_is_local_even_with_a_legacy_allowlist() {
        for url in [
            "http://127.0.0.1:8080",
            "http://localhost:1234/v1",
            "http://[::1]:8080",
            "https://127.0.0.1/v1",
        ] {
            assert!(check_endpoint(url, &[]).is_ok(), "{url}");
        }
        for url in [
            "https://api.example.com/v1",
            "http://10.0.0.5:11434",
            "https://100.64.0.7",
            "https://169.254.169.254",
            "https://[fd00::1]",
            "https://0x7f000001",
            "file:///etc/passwd",
            "not a url",
            "https://",
            "https://user:pass@localhost/v1",
        ] {
            assert!(
                check_endpoint(url, &["api.example.com".into(), "100.64.0.7".into()]).is_err(),
                "{url}"
            );
        }
    }

    #[test]
    fn disabled_or_unconfigured_client_is_refused_with_one_line() {
        let err = EmbedClient::new(&EmbeddingsConfig::default()).unwrap_err();
        assert!(err.to_string().contains("disabled"), "got: {err}");
        assert!(!err.to_string().contains('\n'), "one-line error contract");
        let err = EmbedClient::new(&EmbeddingsConfig {
            enabled: true,
            endpoint: String::new(),
            model: "m".into(),
            ..EmbeddingsConfig::default()
        })
        .unwrap_err();
        assert!(!err.to_string().is_empty());
        let err = EmbedClient::new(&EmbeddingsConfig {
            enabled: true,
            endpoint: "http://127.0.0.1:1".into(),
            model: String::new(),
            ..EmbeddingsConfig::default()
        })
        .unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn inactive_canonical_profile_is_refused_before_any_endpoint_can_be_used() {
        let error = EmbedClient::new(&EmbeddingsConfig {
            enabled: true,
            endpoint: "http://127.0.0.1:1".into(),
            ..EmbeddingsConfig::default()
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("not active"), "{error}");
    }

    #[test]
    fn semantic_recall_unavailable_without_config_is_one_line() {
        let state = tempfile::tempdir().unwrap();
        let conn = index::open(state.path()).unwrap();
        let err = semantic_hits(&Config::default(), &conn, "query", 5, false, &[]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("semantic recall unavailable"), "got: {msg}");
        assert!(!msg.contains('\n'));
        let err = semantic_hits(&Config::default(), &conn, "query", 5, true, &[]).unwrap_err();
        assert!(
            err.to_string().contains("semantic recall unavailable"),
            "--hybrid gated too"
        );
    }

    // ---- client wire behavior ----

    #[test]
    fn the_instruction_prefixes_queries_and_never_documents() {
        // Asymmetric retrieval: the instruction belongs on the query side.
        // A document embedded with it would not be the same artifact as the
        // one every other host derived, so `embed`/`embed_batch` stay raw.
        let (url, bodies, _) = spawn_server(|_, body| canned_embeddings(body, 0.0));
        let mut cfg = EmbeddingsConfig {
            enabled: true,
            endpoint: url.clone(),
            model: "test-model".into(),
            dimensions: 2,
            ..EmbeddingsConfig::default()
        };
        cfg.query_prefix = "Instruct: find it\nQuery: ".into();
        cfg.document_prefix.clear();
        let client = EmbedClient::new(&cfg).unwrap();

        client.embed_query("what is it").unwrap();
        client.embed_documents_batch(&["a stored block"]).unwrap();
        let sent = bodies.lock().unwrap();
        let q: serde_json::Value = serde_json::from_str(&sent[0]).unwrap();
        assert_eq!(
            q["input"],
            serde_json::json!(["Instruct: find it\nQuery: what is it"])
        );
        let d: serde_json::Value = serde_json::from_str(&sent[1]).unwrap();
        assert_eq!(
            d["input"],
            serde_json::json!(["a stored block"]),
            "documents stay raw"
        );
    }

    #[test]
    fn each_side_gets_its_own_prefix_and_never_the_others() {
        // The whole point of asymmetric retrieval: an instruction on the
        // query, a different one on the document, and neither leaking into
        // the other. A document embedded with the query instruction is not
        // the artifact every other host derived.
        let (url, bodies, _) = spawn_server(|_, body| canned_embeddings(body, 0.0));
        let mut cfg = EmbeddingsConfig {
            enabled: true,
            endpoint: url.clone(),
            model: "test-model".into(),
            dimensions: 2,
            ..EmbeddingsConfig::default()
        };
        cfg.query_prefix = "Q: ".into();
        cfg.document_prefix = "D: ".into();
        let client = EmbedClient::new(&cfg).unwrap();

        client.embed_query("who").unwrap();
        client.embed_documents_batch(&["what"]).unwrap();
        client.embed_documents_batch(&["one", "two"]).unwrap();

        let sent = bodies.lock().unwrap();
        let q: serde_json::Value = serde_json::from_str(&sent[0]).unwrap();
        assert_eq!(
            q["input"],
            serde_json::json!(["Q: who"]),
            "query takes the query prefix only"
        );
        let d: serde_json::Value = serde_json::from_str(&sent[1]).unwrap();
        assert_eq!(d["input"], serde_json::json!(["D: what"]));
        let b: serde_json::Value = serde_json::from_str(&sent[2]).unwrap();
        assert_eq!(
            b["input"],
            serde_json::json!(["D: one", "D: two"]),
            "every batched document"
        );
    }

    #[test]
    fn an_empty_prefix_leaves_the_query_untouched() {
        let (url, bodies, _) = spawn_server(|_, body| canned_embeddings(body, 0.0));
        let client = EmbedClient::new(&EmbeddingsConfig {
            enabled: true,
            endpoint: url,
            model: "test-model".into(),
            dimensions: 2,
            query_prefix: String::new(),
            document_prefix: String::new(),
            ..EmbeddingsConfig::default()
        })
        .unwrap();
        client.embed_query("plain").unwrap();
        let q: serde_json::Value = serde_json::from_str(&bodies.lock().unwrap()[0]).unwrap();
        assert_eq!(q["input"], serde_json::json!(["plain"]));
    }

    #[test]
    fn embed_posts_openai_shape_and_orders_by_index() {
        let (url, bodies, _) = spawn_server(|_, body| canned_embeddings(body, 10.0));
        let client = client_for(&url);
        let out = client.embed_documents_batch(&["alpha", "beta"]).unwrap();
        // response rows arrive reversed; `index` must restore input order
        assert_eq!(out, vec![vec![10.0, 1.0], vec![11.0, 1.0]]);
        let sent: serde_json::Value = serde_json::from_str(&bodies.lock().unwrap()[0]).unwrap();
        assert_eq!(sent["model"], "test-model");
        assert_eq!(
            sent["input"],
            serde_json::json!(["title: none | text: alpha", "title: none | text: beta"])
        );
    }

    #[test]
    fn redirects_are_refused() {
        let (url, _, _) = spawn_server(|_, _| {
            "HTTP/1.1 302 Found\r\nlocation: http://127.0.0.1:9/elsewhere\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                .to_string()
        });
        let client = client_for(&url);
        assert!(
            client.embed_documents_batch(&["x"]).is_err(),
            "a 3xx must be an error, never followed"
        );
    }

    #[test]
    fn non_2xx_status_is_an_error() {
        let (url, _, _) = spawn_server(|_, _| http_response(500, r#"{"error":"boom"}"#));
        let client = client_for(&url);
        let err = client.embed_documents_batch(&["x"]).unwrap_err();
        assert!(err.to_string().contains("500"), "status surfaced: {err}");
    }

    #[test]
    fn short_response_is_an_error() {
        // 1 vector for 2 inputs must not silently mis-align block ids
        let (url, _, _) = spawn_server(|_, body| {
            let _ = body;
            http_response(
                200,
                r#"{"object":"list","data":[{"index":0,"embedding":[1.0,0.0]}]}"#,
            )
        });
        let client = client_for(&url);
        assert!(client.embed_documents_batch(&["a", "b"]).is_err());
    }

    #[test]
    fn duplicate_missing_and_out_of_range_indices_are_refused() {
        let cases = [
            (
                r#"{"index":0,"embedding":[1.0,0.0]},{"index":0,"embedding":[0.0,1.0]}"#,
                "duplicate index",
            ),
            (
                r#"{"index":0,"embedding":[1.0,0.0]},{"index":2,"embedding":[0.0,1.0]}"#,
                "out-of-range index",
            ),
            (
                r#"{"index":0,"embedding":[1.0,0.0]},{"embedding":[0.0,1.0]}"#,
                "has no index",
            ),
        ];
        for (rows, expected) in cases {
            let rows = rows.to_string();
            let (url, _, _) = spawn_server(move |_, _| attested_rows("test-model", &rows));
            let error = client_for(&url)
                .embed_documents_batch(&["a", "b"])
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?}, got {error:?}"
            );
        }
    }

    #[test]
    fn an_unattested_noncanonical_endpoint_uses_its_own_vector_space() {
        let (url, _, _) = spawn_server(|_, _| {
            http_response(
                200,
                r#"{"object":"list","model":"test-model","data":[{"index":0,"embedding":[1.0,0.0]}]}"#,
            )
        });
        // A deliberately noncanonical endpoint does not claim compatibility
        // with the admitted shared profile, so its standard response remains
        // valid in its separately named vector space.
        let result = client_for(&url).embed_documents_batch(&["a"]);
        assert!(
            result.is_ok(),
            "endpoint-configured client accepts standard response: {:?}",
            result.err().map(|e| e.to_string())
        );
    }

    #[test]
    fn profile_strings_cannot_self_admit_an_execution_scope() {
        let error = validate_execution_scope(
            Some(WireExecutionScope {
                scope_id: "unreviewed-scope".into(),
                transport: crate::embedding_profile::ExecutionTransport::RemoteAttested,
                backend: "candidate-runtime".into(),
                runtime: "runtime-version".into(),
                compiler: "compiler-version".into(),
                package_target: "test-target".into(),
                artifact_source: "source@revision/file".into(),
                device_class: "npu".into(),
                device: "test-npu".into(),
                artifact_sha256: "1".repeat(64),
                internal_precision: "target-native".into(),
                placement_evidence_sha256: "2".repeat(64),
                supported_max_tokens: crate::embedding_profile::MAX_TOKENS,
                supported_sequence_buckets: BoundedVec(
                    crate::embedding_profile::SEQUENCE_BUCKETS.to_vec(),
                ),
                supported_max_batch_size: crate::embedding_profile::MAX_WIRE_BATCH_SIZE,
                sequence_capability_evidence_sha256: "3".repeat(64),
                performance_evidence_sha256: "4".repeat(64),
                compatibility_report_sha256: "5".repeat(64),
                accelerated_placement: true,
            }),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("not an admitted backend"), "{error}");

        let error = validate_execution_scope(
            Some(WireExecutionScope {
                scope_id: "intel-lunar-lake-cpu-openvino-v1".into(),
                transport: crate::embedding_profile::ExecutionTransport::SupervisedLocal,
                backend: "openvino".into(),
                runtime: "openvino-2026.3.1".into(),
                compiler: "openvino-ir".into(),
                package_target: "x86_64-unknown-linux-gnu".into(),
                artifact_source: "pinned-source@revision".into(),
                device_class: "cpu".into(),
                device: "intel-lunar-lake".into(),
                artifact_sha256: "1".repeat(64),
                internal_precision: "target-native".into(),
                placement_evidence_sha256: "2".repeat(64),
                supported_max_tokens: crate::embedding_profile::MAX_TOKENS,
                supported_sequence_buckets: BoundedVec(
                    crate::embedding_profile::SEQUENCE_BUCKETS.to_vec(),
                ),
                supported_max_batch_size: crate::embedding_profile::MAX_WIRE_BATCH_SIZE,
                sequence_capability_evidence_sha256: "3".repeat(64),
                performance_evidence_sha256: "4".repeat(64),
                compatibility_report_sha256: "5".repeat(64),
                accelerated_placement: true,
            }),
            Some("intel-lunar-lake-npu-openvino-v1"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("local package plan requested"), "{error}");

        let error = validate_execution_scope(
            Some(WireExecutionScope {
                scope_id: "unaccelerated".into(),
                transport: crate::embedding_profile::ExecutionTransport::RemoteAttested,
                backend: "candidate-runtime".into(),
                runtime: "runtime-version".into(),
                compiler: "compiler-version".into(),
                package_target: "test-target".into(),
                artifact_source: "source@revision/file".into(),
                device_class: "cpu".into(),
                device: "test-cpu".into(),
                artifact_sha256: "2".repeat(64),
                internal_precision: "target-native".into(),
                placement_evidence_sha256: "3".repeat(64),
                supported_max_tokens: crate::embedding_profile::MAX_TOKENS,
                supported_sequence_buckets: BoundedVec(
                    crate::embedding_profile::SEQUENCE_BUCKETS.to_vec(),
                ),
                supported_max_batch_size: crate::embedding_profile::MAX_WIRE_BATCH_SIZE,
                sequence_capability_evidence_sha256: "4".repeat(64),
                performance_evidence_sha256: "5".repeat(64),
                compatibility_report_sha256: "6".repeat(64),
                accelerated_placement: false,
            }),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("accelerated placement"), "{error}");
    }

    #[test]
    fn a_degenerate_query_vector_is_refused_before_ranking() {
        let (url, _, _) = spawn_server(|_, _| {
            attested_rows("test-model", r#"{"index":0,"embedding":[0.0,0.0]}"#)
        });
        let err = client_for(&url).embed_query("x").unwrap_err().to_string();
        assert!(err.contains("degenerate embedding"), "{err}");
        assert!(err.contains("norm"), "{err}");
    }

    // ---- auth header ----

    #[test]
    fn api_key_env_sets_bearer_header() {
        let (url, _, headers) = spawn_server(|_, body| canned_embeddings(body, 0.0));
        let auth = resolve_auth_with("CFETCH_TEST_EMBED_KEY", "embeddings", |name| {
            (name == "CFETCH_TEST_EMBED_KEY").then(|| "sk-cfetch-test".to_string())
        })
        .unwrap();
        let client = client_for_auth(&url, auth);
        client.embed_documents_batch(&["x"]).unwrap();
        let sent = headers.lock().unwrap()[0].to_ascii_lowercase();
        assert!(
            sent.contains("authorization: bearer sk-cfetch-test"),
            "got headers:\n{sent}"
        );
    }

    #[test]
    fn no_api_key_env_means_no_auth_header() {
        let (url, _, headers) = spawn_server(|_, body| canned_embeddings(body, 0.0));
        let client = client_for(&url);
        client.embed_documents_batch(&["x"]).unwrap();
        let sent = headers.lock().unwrap()[0].to_ascii_lowercase();
        assert!(
            !sent.contains("authorization:"),
            "no auth configured, none sent:\n{sent}"
        );
    }

    #[test]
    fn unset_or_literal_api_key_env_is_refused() {
        let base = EmbeddingsConfig {
            enabled: true,
            endpoint: "http://127.0.0.1:1".into(),
            model: "m".into(),
            ..EmbeddingsConfig::default()
        };
        // configured NAME whose variable is absent from the environment
        let err = EmbedClient::new(&EmbeddingsConfig {
            api_key_env: "CFETCH_TEST_DEFINITELY_UNSET_VAR".into(),
            ..base.clone()
        })
        .unwrap_err();
        assert!(err.to_string().contains("not set"), "got: {err}");
        // a literal key pasted where the NAME belongs must be refused loudly
        let err = EmbedClient::new(&EmbeddingsConfig {
            api_key_env: "sk-abc123.secret-key".into(),
            ..base
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("never the key itself"),
            "got: {err}"
        );
    }

    // ---- timeouts ----

    #[test]
    fn batch_timeout_scales_base_plus_per_item() {
        let base = std::time::Duration::from_secs(10);
        assert_eq!(batch_timeout(base, 0), base);
        assert_eq!(batch_timeout(base, 1), base + PER_ITEM);
        assert_eq!(batch_timeout(base, 64), base + PER_ITEM * 64);
    }

    #[test]
    fn interactive_timeout_stays_tight() {
        // Server answers after 2 s; the interactive bound is 1 s.
        let (url, _, _) = spawn_server(|_, body| {
            std::thread::sleep(std::time::Duration::from_secs(2));
            canned_embeddings(body, 0.0)
        });
        let client = EmbedClient::new(&EmbeddingsConfig {
            enabled: true,
            endpoint: url,
            model: "test-model".into(),
            timeout_secs: 1,
            dimensions: 2,
            ..EmbeddingsConfig::default()
        })
        .unwrap();
        // The interactive path is the QUERY embed — the batch path
        // deliberately scales its bound per item, so asserting the tight
        // bound there would be asserting the wrong thing.
        assert!(
            client.embed_query("x").is_err(),
            "recall must not wait for a slow backend"
        );
    }

    #[test]
    fn batch_timeout_outlives_a_backend_too_slow_for_interactive() {
        // Same 2 s server, same 1 s base — but the batch path's bound is
        // base + per-item (1 + 2 = 3 s for one input), so it succeeds.
        let (url, _, _) = spawn_server(|_, body| {
            std::thread::sleep(std::time::Duration::from_secs(2));
            canned_embeddings(body, 0.0)
        });
        let client = EmbedClient::new(&EmbeddingsConfig {
            enabled: true,
            endpoint: url,
            model: "test-model".into(),
            timeout_secs: 1,
            dimensions: 2,
            ..EmbeddingsConfig::default()
        })
        .unwrap();
        let out = client.embed_documents_batch(&["x"]).unwrap();
        assert_eq!(out.len(), 1);
    }

    // ---- embed-index over a real (temp) index ----

    fn five_block_index() -> (tempfile::TempDir, tempfile::TempDir, Connection) {
        let brain = tempfile::tempdir().unwrap();
        let p = brain.path().join("knowledge/a.md");
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, "- one\n- two\n- three\n- four\n- five\n").unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut conn = index::open(state.path()).unwrap();
        index::scan(
            &mut conn,
            brain.path(),
            None,
            &crate::config::RingRules::default(),
        )
        .unwrap();
        (brain, state, conn)
    }

    /// A shared store over the given brain tree, at the 2-d test spec.
    fn store_for(brain: &std::path::Path) -> crate::vectors::VectorStore {
        crate::vectors::VectorStore::open(brain, &spec_for(2)).unwrap()
    }

    #[test]
    fn embed_index_embeds_all_blocks_in_batches() {
        let (brain, _state, mut conn) = five_block_index();
        let (url, bodies, _) = spawn_server(|_, body| canned_embeddings(body, 0.0));
        let client = client_for(&url);
        let mut store = store_for(brain.path());
        let report = run(&mut conn, &client, 2, &mut store).unwrap();
        assert_eq!(report.embedded, 5);
        assert_eq!(report.imported, 0);
        assert_eq!(report.total_blocks, 5);
        assert_eq!(index::vector_coverage(&conn, &spec_for(2)).unwrap(), (5, 5));
        assert_eq!(
            store.len(),
            5,
            "the shared tree is the record, not index.db"
        );
        let sizes: Vec<usize> = bodies
            .lock()
            .unwrap()
            .iter()
            .map(|b| {
                serde_json::from_str::<serde_json::Value>(b).unwrap()["input"]
                    .as_array()
                    .unwrap()
                    .len()
            })
            .collect();
        assert_eq!(sizes, vec![2, 2, 1], "batched requests");
        // meta recorded for future model/dim gating
        let model: String = conn
            .query_row("SELECT value FROM meta WHERE key='embed_model'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(model, "test-model");
        let dim: String = conn
            .query_row("SELECT value FROM meta WHERE key='embed_dim'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(dim, "2");
    }

    #[test]
    fn embed_index_rejects_batch_outside_the_wire_contract_before_work() {
        let (brain, _state, mut conn) = five_block_index();
        let client = client_for("http://127.0.0.1:1");
        let mut store = store_for(brain.path());
        for batch in [0, crate::embedding_profile::MAX_WIRE_BATCH_SIZE + 1] {
            let error = match run(&mut conn, &client, batch, &mut store) {
                Ok(_) => panic!("batch {batch} must be rejected"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains("between 1 and 64"), "{error}");
        }
        assert!(
            store.is_empty(),
            "invalid batch must not touch the shared store"
        );
    }

    fn overlength_response() -> String {
        http_response(400, &serde_json::json!({
            "error": format!(
                "prefixed input contains {} tokens; the profile limit is {} and truncation is forbidden",
                crate::embedding_profile::MAX_TOKENS + 1,
                crate::embedding_profile::MAX_TOKENS
            )
        }).to_string())
    }

    #[test]
    fn embed_index_systemic_failures_make_one_request_without_row_retries() {
        for case in [
            "503",
            "transport",
            "400",
            "auth",
            "parse",
            "model",
            "width",
            "degenerate",
        ] {
            let (brain, _state, mut conn) = five_block_index();
            let (url, bodies, _) = spawn_server(move |_, body| match case {
                "503" => http_response(503, r#"{"error":"resource governor unavailable"}"#),
                "transport" => String::new(), // Close after receiving the request.
                "400" => http_response(400, r#"{"error":"invalid requested scope"}"#),
                "auth" => http_response(401, r#"{"error":"unauthorized"}"#),
                "parse" => http_response(200, "not json"),
                "model" => canned_embeddings(body, 0.0).replace("test-model", "wrong-model"),
                "width" => canned_width(body, 3),
                "degenerate" => {
                    let request: serde_json::Value = serde_json::from_str(body).unwrap();
                    let rows = (0..request["input"].as_array().unwrap().len())
                        .map(|index| format!(r#"{{"index":{index},"embedding":[0.0,0.0]}}"#))
                        .collect::<Vec<_>>()
                        .join(",");
                    attested_rows("test-model", &rows)
                }
                _ => unreachable!(),
            });
            let mut store = store_for(brain.path());
            let error = match run(&mut conn, &client_for(&url), 5, &mut store) {
                Ok(_) => panic!("systemic failure must stop the run"),
                Err(error) => error,
            };
            assert!(
                error.downcast_ref::<InputRefusal>().is_none(),
                "{case}: {error:#}"
            );
            assert_eq!(
                bodies.lock().unwrap().len(),
                1,
                "{case} must not become five singleton retries"
            );
            assert!(store.is_empty(), "{case} must not admit any failed batch");
            assert_eq!(index::vector_coverage(&conn, &spec_for(2)).unwrap(), (0, 5));
        }
    }

    #[test]
    fn embed_index_stops_isolation_at_the_first_systemic_failure() {
        let (brain, _state, mut conn) = five_block_index();
        let (url, bodies, _) = spawn_server(|n, body| match n {
            0 => overlength_response(),
            1 => canned_embeddings(body, 0.0),
            _ => http_response(503, r#"{"error":"all execution scopes exhausted"}"#),
        });
        let mut store = store_for(brain.path());
        let error = match run(&mut conn, &client_for(&url), 5, &mut store) {
            Ok(_) => panic!("a systemic error during row isolation must stop"),
            Err(error) => error,
        };
        assert!(
            format!("{error:#}").contains("503"),
            "original failure must survive: {error:#}"
        );
        assert_eq!(
            bodies.lock().unwrap().len(),
            3,
            "batch, first row, failing second row; nothing after failure"
        );
        assert!(store.is_empty(), "the interrupted batch was never admitted");
        assert_eq!(index::vector_coverage(&conn, &spec_for(2)).unwrap(), (0, 5));
    }

    #[test]
    fn embed_index_skips_overlength_heads_once_and_preserves_later_progress() {
        for (batch, poison_count) in [(1, 1), (1, 2), (2, 1), (2, 2)] {
            let (brain, _state, mut conn) = five_block_index();
            let (url, bodies, _) = spawn_server(move |_, body| {
                let request: serde_json::Value = serde_json::from_str(body).unwrap();
                let too_long = request["input"].as_array().unwrap().iter().any(|text| {
                    let text = text.as_str().unwrap();
                    text.ends_with("- one") || (poison_count == 2 && text.ends_with("- two"))
                });
                if too_long {
                    overlength_response()
                } else {
                    canned_embeddings(body, 0.0)
                }
            });
            let mut store = store_for(brain.path());
            let error = match run(&mut conn, &client_for(&url), batch, &mut store) {
                Ok(_) => panic!("refused inputs cannot report a complete run"),
                Err(error) => error,
            };
            assert!(
                error
                    .to_string()
                    .contains(&format!("{poison_count} input(s) exceeded")),
                "{error:#}"
            );
            assert_eq!(
                store.len(),
                5 - poison_count,
                "later valid inputs are durably stored"
            );
            assert_eq!(
                index::vector_coverage(&conn, &spec_for(2)).unwrap(),
                (5 - poison_count, 5)
            );
            assert_eq!(
                index::hashes_without_vectors(&conn, &spec_for(2), 10)
                    .unwrap()
                    .len(),
                poison_count
            );
            let requests = bodies.lock().unwrap();
            assert_eq!(
                requests.len(),
                5,
                "refused heads are not reselected; singleton refusals are not retried"
            );
            for suffix in ["- one", "- two"].into_iter().take(poison_count) {
                let occurrences = requests
                    .iter()
                    .map(|body| {
                        let request: serde_json::Value = serde_json::from_str(body).unwrap();
                        request["input"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter(|text| text.as_str().unwrap().ends_with(suffix))
                            .count()
                    })
                    .sum::<usize>();
                assert_eq!(
                    occurrences, batch,
                    "one initial batch plus isolation only when needed"
                );
            }
        }
    }

    #[test]
    fn embed_index_all_overlength_inputs_finish_with_one_bounded_isolation_pass() {
        let (brain, _state, mut conn) = five_block_index();
        let (url, bodies, _) = spawn_server(|_, _| overlength_response());
        let mut store = store_for(brain.path());
        let error = match run(&mut conn, &client_for(&url), 5, &mut store) {
            Ok(_) => panic!("all-refused run must fail without looping"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("5 input(s) exceeded"),
            "{error:#}"
        );
        assert_eq!(
            bodies.lock().unwrap().len(),
            6,
            "one batch and one isolation attempt per input"
        );
        assert!(store.is_empty());
        assert_eq!(index::vector_coverage(&conn, &spec_for(2)).unwrap(), (0, 5));
    }

    #[test]
    fn embed_index_is_resumable_after_midway_failure() {
        let (brain, _state, mut conn) = five_block_index();
        // First server: batch 1 succeeds, batch 2 fails -> run() errors, but
        // the first batch's vectors are already committed.
        let (url_a, bodies_a, _) = spawn_server(|n, body| {
            if n == 0 {
                canned_embeddings(body, 0.0)
            } else {
                http_response(500, "{}")
            }
        });
        let client_a = client_for(&url_a);
        let mut store = store_for(brain.path());
        assert!(run(&mut conn, &client_a, 2, &mut store).is_err());
        assert_eq!(
            bodies_a.lock().unwrap().len(),
            2,
            "failure stops without row retries"
        );
        assert_eq!(
            index::vector_coverage(&conn, &spec_for(2)).unwrap().0,
            2,
            "committed batch survives the failure"
        );
        assert_eq!(
            store.len(),
            2,
            "and it survives in the SHARED store, not only locally"
        );

        // Second server: healthy. Only the 3 missing blocks get embedded.
        let (url_b, bodies_b, _) = spawn_server(|_, body| canned_embeddings(body, 100.0));
        let client_b = client_for(&url_b);
        let report = run(&mut conn, &client_b, 2, &mut store).unwrap();
        assert_eq!(report.embedded, 3, "resume embeds only what is missing");
        assert_eq!(index::vector_coverage(&conn, &spec_for(2)).unwrap(), (5, 5));
        let inputs_b: Vec<String> = bodies_b
            .lock()
            .unwrap()
            .iter()
            .flat_map(|b| {
                serde_json::from_str::<serde_json::Value>(b).unwrap()["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(
            inputs_b,
            vec![
                "title: none | text: - three",
                "title: none | text: - four",
                "title: none | text: - five",
            ],
            "already-embedded blocks not re-sent"
        );
    }

    #[test]
    fn embed_index_after_model_change_re_embeds_everything() {
        let (brain, _state, mut conn) = five_block_index();
        let (url, bodies, _) = spawn_server(|_, body| canned_embeddings(body, 0.0));
        let client = client_for(&url);
        let mut store = store_for(brain.path());
        run(&mut conn, &client, 8, &mut store).unwrap();
        assert_eq!(index::vector_coverage(&conn, &spec_for(2)).unwrap(), (5, 5));
        // Same endpoint, different model name -> a different artifact: full
        // drop of the old cache, a NEW shared store, everything re-embedded.
        let cfg2 = EmbeddingsConfig {
            enabled: true,
            endpoint: url.clone(),
            model: "other-model".to_string(),
            dimensions: 2,
            ..EmbeddingsConfig::default()
        };
        let client2 = EmbedClient::new(&cfg2).unwrap();
        let mut store2 = crate::vectors::VectorStore::open(brain.path(), &cfg2.spec()).unwrap();
        let report = run(&mut conn, &client2, 8, &mut store2).unwrap();
        assert_eq!(
            report.embedded, 5,
            "model change drops vectors; all re-embedded"
        );
        assert_eq!(bodies.lock().unwrap().len(), 2);
        assert_eq!(
            store.len(),
            5,
            "the old model's artifacts are untouched, not destroyed"
        );
        assert_eq!(store2.len(), 5);
    }

    // ---- dimensions and width ----

    #[test]
    fn an_endpoint_ignoring_the_profile_width_is_rejected() {
        let (url, bodies, _) = spawn_server(|_, body| canned_width(body, 8));
        let client = EmbedClient::new(&EmbeddingsConfig {
            enabled: true,
            endpoint: url,
            model: "test-model".into(),
            dimensions: 4,
            ..EmbeddingsConfig::default()
        })
        .unwrap();
        let err = client
            .embed_documents_batch(&["alpha"])
            .unwrap_err()
            .to_string();
        let sent: serde_json::Value = serde_json::from_str(&bodies.lock().unwrap()[0]).unwrap();
        assert_eq!(
            sent["dimensions"], 4,
            "the request asks the endpoint for the width"
        );
        assert!(
            err.contains("returned 8-d") && err.contains("exactly 4"),
            "got: {err}"
        );
    }

    #[test]
    fn an_endpoint_honoring_dimensions_is_taken_at_its_word() {
        let (url, _, _) = spawn_server(|_, body| canned_honoring_dimensions(body, 8));
        let client = EmbedClient::new(&EmbeddingsConfig {
            enabled: true,
            endpoint: url,
            model: "test-model".into(),
            dimensions: 3,
            ..EmbeddingsConfig::default()
        })
        .unwrap();
        assert_eq!(
            client.embed_documents_batch(&["alpha"]).unwrap()[0].len(),
            3
        );
    }

    #[test]
    fn a_model_narrower_than_the_configured_width_is_loud() {
        // Never pad, never silently store a shorter vector: the operator
        // asked for a width this model cannot produce, and must hear it.
        let (url, _, _) = spawn_server(|_, body| canned_width(body, 2));
        let client = EmbedClient::new(&EmbeddingsConfig {
            enabled: true,
            endpoint: url,
            model: "test-model".into(),
            dimensions: 16,
            ..EmbeddingsConfig::default()
        })
        .unwrap();
        let err = client
            .embed_documents_batch(&["alpha"])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("16") && err.contains('2'),
            "both widths named: {err}"
        );
        assert!(
            err.contains("refusing truncation or padding"),
            "the compatibility reason is named: {err}"
        );
    }

    #[test]
    fn dimensions_are_honored_end_to_end_into_the_shared_store() {
        let (brain, _state, mut conn) = five_block_index();
        let (url, _, _) = spawn_server(|_, body| canned_honoring_dimensions(body, 8));
        let cfg = EmbeddingsConfig {
            enabled: true,
            endpoint: url,
            model: "test-model".into(),
            dimensions: 4,
            precision: crate::config::Precision::I8,
            ..EmbeddingsConfig::default()
        };
        let client = EmbedClient::new(&cfg).unwrap();
        let mut store = crate::vectors::VectorStore::open(brain.path(), &cfg.spec()).unwrap();
        run(&mut conn, &client, 8, &mut store).unwrap();
        let stored = store
            .get(&crate::embedding_input::hash("- one"))
            .unwrap()
            .unwrap();
        assert_eq!(stored.len(), 4, "the artifact carries the configured width");
        let dim: i64 = conn
            .query_row("SELECT dim FROM vectors LIMIT 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(dim, 4);
        let blob_len: i64 = conn
            .query_row("SELECT length(embedding) FROM vectors LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(blob_len, 4, "INT8: 4 components in 4 bytes");
        let dir = crate::paths::shared_vector_dir(brain.path());
        let files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("bin"))
            .collect();
        assert_eq!(files.len(), 1);
        assert_eq!(
            std::fs::metadata(&files[0]).unwrap().len(),
            5 * 4,
            "5 blocks x 4 INT8 components"
        );
    }

    // ---- compute-once across hosts ----

    #[test]
    fn a_second_host_reads_the_shared_store_and_embeds_nothing() {
        let (brain, _state, mut conn) = five_block_index();
        let (url, _, _) = spawn_server(|_, body| canned_embeddings(body, 0.0));
        let mut store = store_for(brain.path());
        run(&mut conn, &client_for(&url), 8, &mut store).unwrap();
        assert_eq!(store.len(), 5);

        // Host B: same tree, its own empty state dir, and an endpoint that
        // fails every request. Derived-once means it must never be called.
        let state_b = tempfile::tempdir().unwrap();
        let mut conn_b = index::open(state_b.path()).unwrap();
        index::scan(
            &mut conn_b,
            brain.path(),
            None,
            &crate::config::RingRules::default(),
        )
        .unwrap();
        let (url_b, bodies_b, _) = spawn_server(|_, _| http_response(500, r#"{"error":"no"}"#));
        let mut store_b = store_for(brain.path());
        let report = run(&mut conn_b, &client_for(&url_b), 8, &mut store_b).unwrap();
        assert_eq!(
            report.embedded, 0,
            "nothing to derive: the group already derived it"
        );
        assert_eq!(
            report.imported, 5,
            "the shared artifacts are read, not recomputed"
        );
        assert!(
            bodies_b.lock().unwrap().is_empty(),
            "the endpoint was never called"
        );
        assert_eq!(
            index::vector_coverage(&conn_b, &spec_for(2)).unwrap(),
            (5, 5)
        );
    }

    // ---- no silent degradation ----

    fn semantic_config(brain: &std::path::Path, url: &str) -> Config {
        Config {
            brain_root: brain.to_path_buf(),
            embeddings: EmbeddingsConfig {
                enabled: true,
                endpoint: url.to_string(),
                model: "test-model".into(),
                dimensions: 2,
                ..EmbeddingsConfig::default()
            },
            ..Config::default()
        }
    }

    #[test]
    fn the_status_line_states_coverage_and_the_artifact_spec() {
        let spec = spec_for(1024);
        let line = coverage_status_line(&spec, 0, 19478, 0);
        assert!(line.contains("0/19478 blocks embedded"), "got: {line}");
        assert!(line.contains("cfetch embed-index"), "got: {line}");
        let line = coverage_status_line(&spec, 19478, 19478, 19478);
        assert!(line.contains("complete"), "got: {line}");
        assert!(
            line.contains("1024 dims") && line.contains("i8"),
            "got: {line}"
        );
        assert!(coverage_status_line(&spec, 0, 0, 0).contains("index is empty"));
    }

    #[test]
    fn zero_coverage_warns_with_the_numbers_and_answers_lexically() {
        let (brain, _state, conn) = five_block_index();
        let (url, bodies, _) = spawn_server(|_, body| canned_embeddings(body, 0.0));
        let cfg = semantic_config(brain.path(), &url);
        let out = semantic_hits(&cfg, &conn, "three", 5, true, &[]).unwrap();
        let note = out
            .note
            .expect("zero coverage must be reported, never hidden");
        assert!(
            note.contains("0/5"),
            "the numbers are in the warning: {note}"
        );
        assert!(
            note.contains("cfetch embed-index"),
            "the fix is named: {note}"
        );
        assert_eq!(out.hits.len(), 1, "the lexical answer is still delivered");
        assert!(out.hits[0].snippet.contains("three"));
        assert!(
            bodies.lock().unwrap().is_empty(),
            "no point embedding a query nothing can match"
        );
        // --semantic degrades the same way: an answer plus the truth about it.
        let out = semantic_hits(&cfg, &conn, "three", 5, false, &[]).unwrap();
        assert!(out.note.is_some());
        assert_eq!(out.hits.len(), 1);
    }

    #[test]
    fn partial_coverage_warns_with_the_numbers_and_still_ranks() {
        let (brain, _state, conn) = five_block_index();
        let spec = spec_for(2);
        index::ensure_vector_spec(&conn, &spec).unwrap();
        for (hash, _) in index::hashes_without_vectors(&conn, &spec, 2).unwrap() {
            index::insert_vector(&conn, &hash, &spec, &[1.0, 0.0]).unwrap();
        }
        let (url, _, _) = spawn_server(|_, body| canned_embeddings(body, 0.0));
        let cfg = semantic_config(brain.path(), &url);
        let out = semantic_hits(&cfg, &conn, "one", 5, true, &[]).unwrap();
        let note = out
            .note
            .expect("partial coverage is degradation, and must be said");
        assert!(note.contains("2/5"), "got: {note}");
        assert!(!out.hits.is_empty());
    }

    #[test]
    fn an_unreachable_endpoint_degrades_to_labeled_lexical_never_to_nothing() {
        // Vectors are all there; the endpoint that would embed the QUERY is
        // down (a laptop off the VPN, a GPU box asleep). An agent must still
        // get its lexical answer — and must be told what it did not get.
        let (brain, _state, mut conn) = five_block_index();
        let (url, _, _) = spawn_server(|_, body| canned_embeddings(body, 0.0));
        let cfg = semantic_config(brain.path(), &url);
        let mut store = store_for(brain.path());
        run(
            &mut conn,
            &EmbedClient::new(&cfg.embeddings).unwrap(),
            8,
            &mut store,
        )
        .unwrap();

        // Same config, an endpoint nothing listens on.
        let dead = Config {
            embeddings: EmbeddingsConfig {
                endpoint: "http://127.0.0.1:1".into(),
                ..cfg.embeddings
            },
            ..cfg
        };
        let out = semantic_hits(&dead, &conn, "three", 5, true, &[]).unwrap();
        let note = out.note.expect("a degraded answer must be labeled");
        assert!(note.contains("query embedding failed"), "got: {note}");
        assert!(note.contains("lexical"), "got: {note}");
        assert!(!note.contains('\n'), "one-line note contract: {note}");
        assert_eq!(out.hits.len(), 1, "the lexical answer still arrives");
    }

    #[test]
    fn full_coverage_is_silent() {
        let (brain, _state, mut conn) = five_block_index();
        let (url, _, _) = spawn_server(|_, body| canned_embeddings(body, 0.0));
        let cfg = semantic_config(brain.path(), &url);
        let mut store = store_for(brain.path());
        run(
            &mut conn,
            &EmbedClient::new(&cfg.embeddings).unwrap(),
            8,
            &mut store,
        )
        .unwrap();
        let out = semantic_hits(&cfg, &conn, "one", 5, true, &[]).unwrap();
        assert!(
            out.note.is_none(),
            "no warning when nothing is degraded: {:?}",
            out.note
        );
        assert!(!out.hits.is_empty());
    }

    #[test]
    fn a_query_host_hydrates_from_the_shared_store_before_answering() {
        // The none-embed host: vectors exist in the tree, its index.db has
        // none. It must answer semantically without an embed run of its own.
        let (brain, _state, mut conn) = five_block_index();
        let (url, _, _) = spawn_server(|_, body| canned_embeddings(body, 0.0));
        let cfg = semantic_config(brain.path(), &url);
        let mut store = store_for(brain.path());
        run(
            &mut conn,
            &EmbedClient::new(&cfg.embeddings).unwrap(),
            8,
            &mut store,
        )
        .unwrap();

        let state_b = tempfile::tempdir().unwrap();
        let mut conn_b = index::open(state_b.path()).unwrap();
        index::scan(
            &mut conn_b,
            brain.path(),
            None,
            &crate::config::RingRules::default(),
        )
        .unwrap();
        assert_eq!(
            index::vector_coverage(&conn_b, &spec_for(2)).unwrap(),
            (0, 5)
        );
        let out = semantic_hits(&cfg, &conn_b, "one", 5, false, &[]).unwrap();
        assert!(out.note.is_none(), "the tree covered it: {:?}", out.note);
        assert!(!out.hits.is_empty());
        assert_eq!(
            index::vector_coverage(&conn_b, &spec_for(2)).unwrap(),
            (5, 5)
        );
    }
}
