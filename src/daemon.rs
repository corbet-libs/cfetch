//! The warm per-host daemon. Hooks are thin clients: they must never open
//! databases or scan trees themselves (they sit on the interactive path), so
//! anything heavier than a file read lives behind the LOCAL control channel
//! ([`crate::ipc`]: a unix socket on unix, loopback TCP on Windows).
//!
//! Protocol: one JSON object per line in, one per line out, then the server
//! closes the connection. Ops: ping, resident, health, scan-code,
//! scan-status, serve-status, shutdown — plus, when serving mode is enabled
//! (config `serve.enabled`), the barrier-gated query ops recall, expand,
//! find, map, code-path, code-impact, code-context, code-symbol, graph, slices, generation
//! and checksum.
//!
//! A serving daemon also keeps its OWN code index current: once the tree
//! watches are registered it kicks the single-flight background code scan and
//! repeats it on the fingerprint cadence. Nothing outside has to send
//! `scan-code` for `find`/`map` to answer on a freshly started host.
//!
//! With `serve.bind` set, the SAME protocol is additionally served over TCP,
//! gated by a bearer token (`token` field in every request; token sourced
//! from `serve.token_file`, 0600). Shutdown is refused on that listener.
//!
//! The daemon also owns this host's single iroh endpoint. Invite redemption
//! and slice-scoped read queries use the same line-JSON request/response
//! objects over one QUIC bidirectional stream. QUIC authenticates the peer's
//! endpoint id; the origin then checks that id against its grant record.
//!
//! Three channels, one gate. [`Channel`] is the whole policy surface: whether
//! a connection must present a token, and whether it may shut the daemon
//! down. A unix socket is access-controlled by its file mode and needs no
//! token; loopback TCP is not, so the Windows LOCAL channel presents the
//! daemon's own token — checked by the same `serve::token_eq` comparison the
//! serving listener uses, never a second implementation.

use std::io::{BufRead, BufReader, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use iroh::protocol::ProtocolHandler as _;
use serde::{Deserialize, Serialize};

use crate::config::{Config, VectorSpec};
use crate::{
    grant, heartbeat, hooks, index, ipc, maintenance_worker, net, paths, resident, serve,
    vector_worker, vectors,
};

#[derive(Debug, Default, Serialize, Deserialize)]
struct Request {
    op: String,
    /// Required on every remote transport. Local control calls omit it.
    #[serde(default)]
    network_major: Option<u32>,
    #[serde(default)]
    token: Option<String>,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    cite: Option<String>,
    #[serde(default)]
    path: Option<String>,
    /// code-path op endpoints. Kept separate from `path`, which is used by
    /// several single-target operations.
    #[serde(default)]
    from_path: Option<String>,
    #[serde(default)]
    to_path: Option<String>,
    /// code-path/code-impact/code-context traversal bound.
    #[serde(default)]
    depth: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
    /// map op: personalize the ranking toward this term.
    #[serde(default)]
    focus: Option<String>,
    /// map op: token budget the rendered map must fit.
    #[serde(default)]
    budget_tokens: Option<u64>,
    /// recall op: rank by vectors (`semantic`) or fuse with BM25 (`hybrid`).
    /// A none-tier host cannot embed its own query — the host holding the
    /// index does it, which is the whole point of serving.
    #[serde(default)]
    semantic: Option<bool>,
    #[serde(default)]
    hybrid: Option<bool>,
    /// recall op: restrict to one slice. Resolved on the SERVING host, which
    /// is where the slice model lives.
    #[serde(default)]
    slice: Option<String>,
    /// SessionStart's working directory, for the resident op: the daemon
    /// shares the host with its caller but not the cwd, so the repo half of
    /// the injection scope has to travel with the request.
    #[serde(default)]
    cwd: Option<String>,
    /// Invite redemption fields. The remote endpoint id is NEVER accepted
    /// from this payload; it comes from the authenticated iroh connection.
    #[serde(default)]
    secret: Option<String>,
    #[serde(default)]
    mode: Option<grant::Mode>,
    /// Local-only forwarding envelope: target plus the ordinary line-JSON
    /// request to send through this daemon's persistent endpoint.
    #[serde(default)]
    target: Option<iroh::EndpointAddr>,
    #[serde(default)]
    request: Option<Box<Request>>,
    /// Local forwarding envelope only: bound a diagnostic probe without
    /// weakening the ordinary query deadline. Never trusted from a remote
    /// request and never copied into its nested payload.
    #[serde(default)]
    timeout_ms: Option<u64>,
    /// Peer-artifact negotiation. The ordinary slice grant is still the
    /// authority; these fields only describe which exact derived artifacts a
    /// receiver already knows it needs.
    #[serde(default)]
    vector_hashes: Option<Vec<String>>,
    #[serde(default)]
    vector_spec: Option<VectorSpec>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Response {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degraded_hooks: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scan: Option<ScanStatus>,
    // ---- serving-mode fields ----
    /// Host id of the answering serving host.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// Catalog generation the answer was served from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<u64>,
    /// True when the drain barrier fully applied before answering; false =
    /// bounded-barrier expiry, see `stale_note`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fresh: Option<bool>,
    /// Why the answer may be stale (only when `fresh` is false).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stale_note: Option<String>,
    /// Time this query spent in the drain barrier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub barrier_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
    /// How the ranking degraded, if it did — absent vector coverage, an
    /// unreachable reranker. It travels with the answer because the client
    /// that asked cannot see the endpoint the serving host talked to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hits: Option<Vec<serve::WireHit>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocks: Option<Vec<serve::WireBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_hits: Option<Vec<serve::WireFindHit>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub map: Option<serve::WireMap>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependency_path: Option<crate::graph::DependencyPath>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependency_impact: Option<crate::graph::DependencyImpact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependency_context: Option<crate::graph::DependencyContext>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_context: Option<crate::graph::SymbolContext>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub knowledge_graph: Option<crate::knowledge_graph::KnowledgeGraph>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slices: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serve: Option<ServeInfo>,
    /// Current address of this daemon's authenticated iroh endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iroh_addr: Option<iroh::EndpointAddr>,
    /// Present only on successful invite redemption.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grant: Option<WireGrant>,
    /// Local forwarding diagnostics only. `Some(true)` means QUIC completed
    /// and returned a valid cfetch response, even when that response refused
    /// the slice or operation. Absent means no transport claim can be made.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iroh_connected: Option<bool>,
    /// Authenticated vector capabilities offered by a serving peer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_artifacts: Option<Vec<WireVectorArtifact>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_spec: Option<VectorSpec>,
    /// Local-only result after the daemon fetched, verified and durably
    /// appended peer artifacts to this host's shared vector store.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vectors_imported: Option<usize>,
}

impl Response {
    fn err(msg: impl Into<String>) -> Response {
        Response { ok: false, error: Some(msg.into()), ..Response::default() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireGrant {
    pub slice: String,
    pub mode: grant::Mode,
    pub peer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireVectorArtifact {
    pub content_hash: String,
    pub blob_hash: String,
    pub bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServeInfo {
    pub enabled: bool,
    pub origin: String,
    pub generation: u64,
    pub last_barrier_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bind: Option<String>,
    /// Which drain-barrier coverage proof this host's watcher backend can
    /// support — `ordered (sentinel)` or `unordered (fingerprint)`. An
    /// operator must be able to see whether their platform is on the fast or
    /// the sound-and-slower path (see `serve::BarrierMode`). Defaulted for
    /// wire compatibility with a serving host older than the split.
    #[serde(default)]
    pub barrier_mode: String,
}

/// Counts from the most recent SUCCESSFUL code scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanCounts {
    pub files: usize,
    pub symbols: usize,
    pub edges: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanStatus {
    pub running: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_finished: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_counts: Option<ScanCounts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

/// Single-flight gate for the background code scan. A scan over NFS code
/// roots takes minutes, so the daemon runs it on a detached thread; a second
/// request while one runs is REFUSED, not queued — the second scan would only
/// re-read the same tree.
pub struct ScanCoordinator {
    inner: Mutex<ScanStatus>,
}

impl ScanCoordinator {
    pub const fn new() -> Self {
        ScanCoordinator {
            inner: Mutex::new(ScanStatus {
                running: false,
                last_finished: None,
                last_counts: None,
                last_error: None,
            }),
        }
    }

    /// A panicked scan thread must not wedge status reporting.
    fn lock(&self) -> std::sync::MutexGuard<'_, ScanStatus> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Claims the single scan slot; false while another scan runs.
    pub fn try_begin(&self) -> bool {
        let mut s = self.lock();
        if s.running {
            return false;
        }
        s.running = true;
        true
    }

    /// Releases the slot. An error records `last_error` without erasing the
    /// last good counts; a success clears the error.
    pub fn complete(&self, result: Result<ScanCounts, String>) {
        let mut s = self.lock();
        s.running = false;
        s.last_finished = Some(now_secs());
        match result {
            Ok(c) => {
                s.last_counts = Some(c);
                s.last_error = None;
            }
            Err(e) => s.last_error = Some(e),
        }
    }

    pub fn status(&self) -> ScanStatus {
        self.lock().clone()
    }
}

impl Default for ScanCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

static SCAN: ScanCoordinator = ScanCoordinator::new();

/// How often the scan cadence checks whether it may start.
const SCAN_READY_POLL: Duration = Duration::from_millis(200);
/// How long the first code scan waits for the markdown index to settle before
/// starting regardless. SQLite has ONE writer: a cold code scan holds the write
/// lock for as long as it walks, so letting the tree index (which the drain
/// barrier depends on) finish first keeps a fresh host's answers fresh. The
/// bound is there so a tree index that never settles cannot mean a code index
/// that never exists.
const FIRST_SCAN_SETTLE_WAIT: Duration = Duration::from_secs(300);

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn run_code_scan() -> Result<ScanCounts, String> {
    let cfg = Config::load().map_err(|e| e.to_string())?;
    let mut conn = crate::index::open(&paths::state_dir()).map_err(|e| e.to_string())?;
    // Background work with no interactive deadline: waiting out the tree
    // index's write transaction beats failing on a locked database.
    let _ = conn.busy_timeout(Duration::from_secs(30));
    let r = crate::code::scan_code(&mut conn, &cfg.effective_code_roots()).map_err(|e| e.to_string())?;
    Ok(ScanCounts { files: r.files, symbols: r.symbols, edges: r.edges })
}

/// Starts the background code scan unless one is already running. Returns
/// whether THIS call started it (single flight: a refusal is not an error,
/// the running scan re-reads the same tree anyway).
fn begin_code_scan() -> bool {
    if !SCAN.try_begin() {
        return false;
    }
    // Detached worker: the daemon keeps answering while the scan (minutes on
    // a cold network tree) runs; a panic still releases the slot.
    std::thread::spawn(|| {
        let result = std::panic::catch_unwind(run_code_scan)
            .unwrap_or_else(|_| Err("code scan thread panicked".to_string()));
        SCAN.complete(result);
    });
    true
}

/// The daemon's own code-scan cadence, on its own thread.
///
/// The deployed defect this fixes: a fresh serving host answered `find` and
/// `map` with "no hits" until somebody sent `scan-code` by hand. A serving
/// host owns its index lifecycle for the markdown tree already; the code index
/// is the same obligation.
///
/// Waits for the tree watches first (registering them and walking the code
/// roots at once would only contend for the same io), then kicks the
/// single-flight scan and repeats it on the fingerprint cadence — the
/// incremental scan is nearly free when nothing changed. Neither the listener
/// nor the drain barrier ever waits on this: the scan runs detached and the
/// barrier tracks the markdown watcher, not the code index.
fn scan_cadence(
    watches_ready: impl Fn() -> bool,
    kick: impl Fn() -> bool,
    poll: Duration,
    cadence: Duration,
    rounds: Option<usize>,
) {
    while !watches_ready() {
        std::thread::sleep(poll);
    }
    let mut done = 0usize;
    loop {
        kick();
        done += 1;
        if rounds.is_some_and(|r| done >= r) {
            return;
        }
        std::thread::sleep(cadence);
    }
}

/// Whether the first code scan may start: the tree watches are registered and
/// the tree index has settled — or the bounded wait for that has expired.
fn first_scan_ready(state: &serve::ServeState, deadline: Instant) -> bool {
    state.is_settled() || (state.watches_ready() && Instant::now() >= deadline)
}

/// The one line `cfetch status` leads with: which side of the serving topology
/// this host is on. Burying it is how a none-tier host reads as "empty" when
/// it is merely remote.
/// What a serving host reports as its barrier mode. A serving host older
/// than the mode split sends nothing; say so rather than imply the fast path.
fn barrier_mode_of(i: &ServeInfo) -> &str {
    if i.barrier_mode.is_empty() { "mode not reported" } else { &i.barrier_mode }
}

pub fn mode_line(cfg: &Config, info: Option<&ServeInfo>) -> String {
    if let Some(cs) = &cfg.client.serving {
        return format!(
            "mode: none-tier — no local index; recall/find/expand/map served by {}",
            cs.addr
        );
    }
    if cfg.serve.enabled {
        return match info.filter(|i| i.enabled) {
            Some(i) => format!(
                "mode: serving host {} (generation {}, {}, barrier {})",
                i.origin,
                i.generation,
                i.bind.clone().map_or_else(|| "unix socket only".to_string(), |b| format!("tcp {b}")),
                barrier_mode_of(i),
            ),
            None => format!(
                "mode: serving host {} (daemon down — generation unknown, nothing is being served)",
                serve::origin_of(cfg)
            ),
        };
    }
    "mode: local index only (not serving, no serving host configured)".to_string()
}

/// Client call with a hard deadline. The hook path budget is ~250ms total; on
/// any failure the caller falls back or stays silent.
pub fn call(op: &str, timeout: Duration) -> Option<Response> {
    call_req(&serde_json::json!({ "op": op }), timeout)
}

/// Structured client call over the local control channel.
pub fn call_req(body: &serde_json::Value, timeout: Duration) -> Option<Response> {
    let mut stream = ipc::connect(timeout)?;
    // A no-op where the transport is access-controlled by the operating
    // system: on unix the request goes out byte-for-byte as built here.
    let body = ipc::authenticate(body);
    writeln!(stream, "{body}").ok()?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line).ok()?;
    serde_json::from_str(&line).ok()
}

/// Current dialable address of the running daemon's endpoint. Invites require
/// a live daemon so the ticket cannot contain a plausible id with no process
/// actually listening behind it.
pub fn iroh_addr() -> anyhow::Result<iroh::EndpointAddr> {
    let resp = call("iroh-addr", IROH_ONLINE_WAIT + Duration::from_secs(2))
        .context("daemon is not running — start it before minting an invite")?;
    anyhow::ensure!(resp.ok, "{}", resp.error.unwrap_or_else(|| "iroh endpoint unavailable".into()));
    resp.iroh_addr.context("daemon returned no iroh address")
}

/// Sends one ordinary daemon protocol request through the local daemon's
/// persistent iroh endpoint. The caller never opens a competing endpoint with
/// the same host key.
pub fn call_iroh(
    target: &iroh::EndpointAddr,
    body: serde_json::Value,
) -> anyhow::Result<Response> {
    call_iroh_with_timeout(target, body, None, true)
}

/// A short, read-only reachability check for a previously joined slice.
/// This is deliberately stronger than a transport-only ping: a peer counts as
/// usable only when the authenticated endpoint still accepts this host's
/// grant and has an active serving path. It does not drain the watcher or open
/// the remote index, so diagnostics remain observation-only.
pub fn probe_iroh(
    target: &iroh::EndpointAddr,
    slice: &str,
    timeout: Duration,
) -> anyhow::Result<Response> {
    anyhow::ensure!(
        timeout >= Duration::from_millis(100) && timeout <= Duration::from_secs(5),
        "diagnostic probe timeout must be between 100 ms and 5 s"
    );
    call_iroh_with_timeout(
        target,
        serde_json::json!({"op": "diagnose", "slice": slice}),
        Some(timeout),
        false,
    )
}

fn call_iroh_with_timeout(
    target: &iroh::EndpointAddr,
    body: serde_json::Value,
    timeout: Option<Duration>,
    require_ok: bool,
) -> anyhow::Result<Response> {
    // Validate the nested request here so malformed locally constructed JSON
    // never reaches the network.
    let mut request: Request = serde_json::from_value(body).context("invalid iroh request")?;
    request.network_major = Some(crate::embedding_profile::NETWORK_MAJOR);
    let envelope = serde_json::to_value(Request {
        op: "iroh-forward".to_string(),
        target: Some(target.clone()),
        request: Some(Box::new(request)),
        timeout_ms: timeout.map(|value| value.as_millis().min(u64::MAX as u128) as u64),
        ..Request::default()
    })?;
    let local_timeout = timeout.unwrap_or(IROH_REQUEST_TIMEOUT) + Duration::from_secs(2);
    let resp = call_req(&envelope, local_timeout)
        .context("local daemon is not running — start it before using a remote slice")?;
    if require_ok {
        anyhow::ensure!(
            resp.ok,
            "{}",
            resp.error
                .clone()
                .unwrap_or_else(|| "iroh request failed".into())
        );
    }
    Ok(resp)
}

pub fn redeem_iroh(ticket: &grant::Ticket) -> anyhow::Result<WireGrant> {
    let resp = call_iroh(
        &ticket.origin,
        serde_json::json!({
            "op": "redeem",
            "slice": ticket.slice,
            "secret": ticket.secret,
            "mode": ticket.mode,
        }),
    )?;
    resp.grant.context("origin accepted redemption without returning the grant")
}

/// Asks this host's daemon to import missing canonical vectors from one
/// joined origin. The CLI never opens a second endpoint with the same host
/// identity; negotiation and iroh-blobs transfer both travel through the
/// daemon-owned endpoint.
pub fn sync_peer_vectors(
    target: &iroh::EndpointAddr,
    slice: &str,
    hashes: &[String],
    spec: &VectorSpec,
) -> anyhow::Result<usize> {
    if hashes.is_empty() {
        return Ok(0);
    }
    anyhow::ensure!(
        hashes.len() <= vectors::MAX_PEER_ARTIFACTS,
        "one peer vector request may name at most {} hashes",
        vectors::MAX_PEER_ARTIFACTS
    );
    let resp = call_req(
        &serde_json::to_value(Request {
            op: "vector-sync".to_string(),
            target: Some(target.clone()),
            slice: Some(slice.to_string()),
            vector_hashes: Some(hashes.to_vec()),
            vector_spec: Some(spec.clone()),
            ..Request::default()
        })?,
        IROH_REQUEST_TIMEOUT + Duration::from_secs(5),
    )
    .context("local daemon is not running — peer vectors were not checked")?;
    anyhow::ensure!(
        resp.ok,
        "{}",
        resp.error.unwrap_or_else(|| "peer vector synchronization failed".into())
    );
    Ok(resp.vectors_imported.unwrap_or(0))
}

/// Shared state of one daemon process across its connection threads.
struct Ctx {
    /// The daemon's own configuration. A serving host ranks ON BEHALF OF
    /// clients that hold nothing, so the endpoints it may reach — embeddings,
    /// reranking — are its own, never the caller's.
    cfg: Arc<crate::config::Config>,
    serve: Option<Arc<serve::ServeState>>,
    /// Bearer token required on the serving TCP listener (None = no TCP
    /// serving).
    tcp_token: Option<String>,
    /// Bearer token required on the LOCAL channel — `Some` only where the
    /// local transport is not access-controlled by the operating system.
    local_token: Option<String>,
    /// Set exactly once after the endpoint has bound. The endpoint and runtime
    /// handle are cloneable, so synchronous local-channel workers can ask the
    /// async iroh runtime to make an outbound call without minting a second
    /// endpoint with the same host identity.
    iroh: Mutex<Option<IrohClient>>,
    /// Per-process capability key. It salts peer artifact blobs so their
    /// BLAKE3 hashes cannot be guessed from public vector bytes; stable within
    /// the daemon so repeated requests deduplicate in the blob store.
    artifact_key: [u8; 32],
    shutdown: AtomicBool,
}

#[derive(Clone)]
struct IrohClient {
    endpoint: iroh::Endpoint,
    runtime: tokio::runtime::Handle,
    download_blobs: iroh_blobs::api::Store,
    peer_blobs: PeerBlobStores,
}

/// Provider storage is partitioned by the QUIC-authenticated endpoint id.
/// A blob hash disclosed to peer A is therefore not servable to peer B even
/// if A leaks it. The durable vectors stay in cfetch's packed store; these are
/// bounded transfer views populated only after slice authorization.
#[derive(Clone, Default)]
struct PeerBlobStores {
    inner: Arc<Mutex<std::collections::HashMap<String, iroh_blobs::api::Store>>>,
}

impl std::fmt::Debug for PeerBlobStores {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let peers = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        f.debug_struct("PeerBlobStores").field("peers", &peers).finish()
    }
}

impl PeerBlobStores {
    fn get(&self, peer: &str) -> Option<iroh_blobs::api::Store> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(peer)
            .cloned()
    }

    fn get_or_create(&self, peer: &str) -> iroh_blobs::api::Store {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(peer.to_string())
            .or_insert_with(|| iroh_blobs::store::mem::MemStore::new().into())
            .clone()
    }
}

#[derive(Clone, Debug)]
struct PeerBlobsProtocol {
    stores: PeerBlobStores,
}

impl iroh::protocol::ProtocolHandler for PeerBlobsProtocol {
    async fn accept(
        &self,
        conn: iroh::endpoint::Connection,
    ) -> Result<(), iroh::protocol::AcceptError> {
        let peer = conn.remote_id().to_string();
        let Some(store) = self.stores.get(&peer) else {
            conn.close(1u32.into(), b"no authorized artifact offer");
            return Ok(());
        };
        iroh_blobs::BlobsProtocol::new(&store, None).accept(conn).await
    }
}

const IROH_ALPN: &[u8] = b"cfetch/network/1/line-json";
const IROH_MAX_REQUEST: usize = 64 * 1024;
const IROH_MAX_RESPONSE: usize = 16 * 1024 * 1024;
const IROH_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const IROH_REQUEST_TIMEOUT: Duration = Duration::from_secs(45);
const IROH_ONLINE_WAIT: Duration = Duration::from_secs(5);
const MAX_LINE_REQUEST: u64 = 64 * 1024;

/// Which connection a request arrived on, and therefore what it may do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    /// Local channel the operating system already gates (unix socket file
    /// mode). No credential; every op, shutdown included.
    LocalTrusted,
    /// Local channel any process on this machine can reach (loopback TCP).
    /// Token required; still local, so shutdown stays allowed.
    LocalToken,
    /// The serving listener: token required, shutdown refused.
    Remote,
}

/// The policy this platform's LOCAL channel runs under.
const LOCAL_CHANNEL: Channel =
    if ipc::LOCAL_REQUIRES_TOKEN { Channel::LocalToken } else { Channel::LocalTrusted };

impl Channel {
    /// The credential this channel demands, if any.
    fn expected_token(self, ctx: &Ctx) -> Option<Option<&String>> {
        match self {
            Channel::LocalTrusted => None,
            Channel::LocalToken => Some(ctx.local_token.as_ref()),
            Channel::Remote => Some(ctx.tcp_token.as_ref()),
        }
    }

    /// Shutdown is a LOCAL op: a serving peer must never be able to stop the
    /// daemon that answers it.
    fn allows_shutdown(self) -> bool {
        !matches!(self, Channel::Remote)
    }

    fn allows_local_only(self) -> bool {
        !matches!(self, Channel::Remote)
    }

    fn allows_op(self, op: &str) -> bool {
        !matches!(self, Channel::Remote)
            || matches!(
                op,
                "ping"
                    | "serve-status"
                    | "recall"
                    | "expand"
                    | "find"
                    | "map"
                    | "code-path"
                    | "code-impact"
                    | "code-context"
                    | "code-symbol"
                    | "graph"
                    | "slices"
                    | "generation"
                    | "checksum"
            )
    }
}

/// Bearer check for a channel that demands one. A missing expectation and a
/// missing presentation both refuse — an unconfigured token is never an open
/// door.
fn authorized(expected: Option<&String>, presented: Option<&String>) -> bool {
    match (expected, presented) {
        (Some(expected), Some(got)) => serve::token_eq(expected, got),
        _ => false,
    }
}

fn iroh_client(ctx: &Ctx) -> anyhow::Result<IrohClient> {
    ctx.iroh
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .context("iroh endpoint is not ready")
}

fn dialable_iroh_addr(client: &IrohClient) -> iroh::EndpointAddr {
    let addr = client.endpoint.addr();
    if addr.relay_urls().next().is_some() {
        return addr;
    }
    // A freshly bound endpoint has direct routes immediately but may need a
    // moment to select its relay. Waiting here (when minting an invite), not
    // at daemon boot, preserves offline local operation while making WAN
    // tickets self-contained whenever a relay is reachable.
    client.runtime.block_on(async {
        let _ = tokio::time::timeout(IROH_ONLINE_WAIT, client.endpoint.online()).await;
    });
    client.endpoint.addr()
}

async fn iroh_exchange(
    endpoint: &iroh::Endpoint,
    target: iroh::EndpointAddr,
    req: &Request,
    connect_timeout: Duration,
    response_timeout: Duration,
) -> anyhow::Result<Response> {
    let mut body = serde_json::to_vec(req)?;
    body.push(b'\n');
    anyhow::ensure!(body.len() <= IROH_MAX_REQUEST, "iroh request is too large");

    let conn = tokio::time::timeout(
        connect_timeout,
        endpoint.connect(target, IROH_ALPN),
    )
    .await
    .context("timed out connecting to the invite origin")??;
    let (mut send, mut recv) = tokio::time::timeout(connect_timeout, conn.open_bi())
        .await
        .context("timed out opening an iroh request stream")??;
    send.write_all(&body).await.context("send iroh request")?;
    send.finish().context("finish iroh request")?;
    let raw = tokio::time::timeout(response_timeout, recv.read_to_end(IROH_MAX_RESPONSE))
        .await
        .context("timed out reading the iroh response")??;
    conn.close(0u32.into(), b"request complete");
    serde_json::from_slice(&raw).context("origin returned an invalid response")
}

fn forward_iroh(
    ctx: &Ctx,
    target: iroh::EndpointAddr,
    req: &Request,
    diagnostic_timeout: Option<Duration>,
) -> anyhow::Result<Response> {
    let client = iroh_client(ctx)?;
    let connect_timeout = diagnostic_timeout.unwrap_or(IROH_CONNECT_TIMEOUT);
    let response_timeout = diagnostic_timeout.unwrap_or(IROH_REQUEST_TIMEOUT);
    client.runtime.block_on(iroh_exchange(
        &client.endpoint,
        target,
        req,
        connect_timeout,
        response_timeout,
    ))
}

fn artifact_nonce(ctx: &Ctx, peer: &str, slice: &str, content_hash: &str) -> [u8; 32] {
    use sha2::Digest as _;
    let mut hasher = sha2::Sha256::new();
    hasher.update(ctx.artifact_key);
    hasher.update(b"cfetch-peer-vector-v1\0");
    for field in [peer, slice, content_hash] {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field.as_bytes());
    }
    // Keying both ends avoids treating the construction as a generic
    // secret-prefix digest. The output is a capability salt, not a MAC.
    hasher.update(ctx.artifact_key);
    hasher.finalize().into()
}

/// Builds iroh-blobs capabilities only for vectors whose source block falls
/// inside the authenticated slice. The blob hash is not itself the grant: it
/// is disclosed only after the grant check above and is salted per peer and
/// slice so public vector bytes do not make it guessable.
fn offer_peer_vectors(req: &Request, peer: &str, slice: &str, ctx: &Ctx) -> Response {
    if let Err(error) = crate::embedding_profile::production_availability() {
        return Response::err(format!(
            "canonical peer vector serving is unavailable: {error}"
        ));
    }
    let Some(wanted_spec) = req.vector_spec.as_ref() else {
        return Response::err("vector artifact request names no embedding profile");
    };
    let local_spec = ctx.cfg.embeddings.spec();
    if wanted_spec != &local_spec {
        return Response::err(format!(
            "vector artifact profile mismatch: peer requested {}, this host serves {}",
            wanted_spec.profile_id, local_spec.profile_id
        ));
    }
    let Some(hashes) = req.vector_hashes.as_ref() else {
        return Response::err("vector artifact request names no content hashes");
    };
    if hashes.is_empty() || hashes.len() > vectors::MAX_PEER_ARTIFACTS {
        return Response::err(format!(
            "vector artifact request must name 1..={} hashes",
            vectors::MAX_PEER_ARTIFACTS
        ));
    }
    if hashes.iter().any(|hash| {
        hash.len() != 64
            || !hash.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    }) {
        return Response::err("vector artifact request contains a non-canonical content hash");
    }
    let model = match ctx.cfg.slice_model() {
        Ok(model) => model,
        Err(e) => return Response::err(e.to_string()),
    };
    let store = match vectors::VectorStore::open(&ctx.cfg.brain_root, &local_spec) {
        Ok(store) => store,
        Err(e) => return Response::err(e.to_string()),
    };
    let client = match iroh_client(ctx) {
        Ok(client) => client,
        Err(e) => return Response::err(e.to_string()),
    };
    serve_query(ctx, |conn| {
        let mut path_stmt = conn.prepare(
            "SELECT DISTINCT d.path FROM blocks b JOIN docs d ON d.id=b.doc_id WHERE b.embedding_hash=?1",
        )?;
        let mut unique = std::collections::HashSet::with_capacity(hashes.len());
        let mut artifacts = Vec::new();
        for hash in hashes {
            if !unique.insert(hash.as_str()) {
                continue;
            }
            let paths = path_stmt.query_map([hash], |row| row.get::<_, String>(0))?;
            let in_slice = paths.filter_map(Result::ok).any(|path| model.contains(slice, &path));
            if !in_slice {
                continue;
            }
            let Some(record) = store.get_blob(hash)? else {
                continue;
            };
            let raw = vectors::encode_peer_artifact(
                hash,
                &record,
                artifact_nonce(ctx, peer, slice, hash),
            )?;
            let bytes = raw.len();
            let peer_store = client.peer_blobs.get_or_create(peer);
            let blob_hash = client.runtime.block_on(async {
                let mut tag = peer_store.add_slice(&raw).temp_tag().await?;
                let hash = tag.hash();
                // MemStore has no GC task, but leaking the temporary tag makes
                // the serving lifetime explicit and deduplicates by hash.
                tag.leak();
                Ok::<_, anyhow::Error>(hash)
            })?;
            artifacts.push(WireVectorArtifact {
                content_hash: hash.clone(),
                blob_hash: blob_hash.to_string(),
                bytes,
            });
        }
        Ok(Response {
            ok: true,
            vector_artifacts: Some(artifacts),
            vector_spec: Some(local_spec.clone()),
            ..Response::default()
        })
    })
}

fn ensure_exact_peer_artifact_len(actual: usize, expected: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        actual == expected,
        "peer vector artifact has {actual} bytes; this VectorSpec requires exactly {expected}"
    );
    Ok(())
}

fn ensure_peer_fetch_progress(
    existing: u64,
    downloaded: u64,
    expected: usize,
) -> anyhow::Result<()> {
    let expected = u64::try_from(expected).context("peer vector artifact length exceeds u64")?;
    let total = existing
        .checked_add(downloaded)
        .context("peer vector artifact download byte count overflowed u64")?;
    anyhow::ensure!(
        total <= expected,
        "peer vector artifact download reached {total} bytes and exceeded its {expected}-byte profile limit"
    );
    Ok(())
}

async fn fetch_peer_blob_bounded(
    store: &iroh_blobs::api::Store,
    conn: iroh::endpoint::Connection,
    hash: iroh_blobs::Hash,
    expected: usize,
) -> anyhow::Result<()> {
    use n0_future::StreamExt as _;

    // `Progress` counts only bytes downloaded by this request. Include any
    // verified partial blob retained from an earlier interrupted attempt so a
    // retry cannot reset the profile-derived memory budget.
    let existing = store
        .remote()
        .local(hash)
        .await
        .context("inspect partial peer vector artifact")?
        .local_bytes();
    ensure_peer_fetch_progress(existing, 0, expected)?;
    let mut progress = store.remote().fetch(conn, hash).stream();
    while let Some(item) = progress.next().await {
        match item {
            iroh_blobs::api::remote::GetProgressItem::Progress(downloaded) => {
                // Dropping the stream on failure drops the in-flight fetch
                // future. At most the current verified transfer chunk can
                // reach the in-memory scratch store beyond this boundary.
                ensure_peer_fetch_progress(existing, downloaded, expected)?;
            }
            iroh_blobs::api::remote::GetProgressItem::Done(_) => return Ok(()),
            iroh_blobs::api::remote::GetProgressItem::Error(error) => {
                anyhow::bail!("fetch peer vector artifact: {error:?}")
            }
        }
    }
    anyhow::bail!("peer vector artifact fetch ended without completion")
}

fn fetch_peer_vectors(req: &Request, ctx: &Ctx) -> anyhow::Result<usize> {
    crate::embedding_profile::production_availability()
        .context("canonical peer vector synchronization is unavailable")?;
    let target = req
        .target
        .clone()
        .context("peer vector synchronization names no target")?;
    let slice = req
        .slice
        .as_deref()
        .context("peer vector synchronization names no slice")?;
    let hashes = req
        .vector_hashes
        .as_ref()
        .context("peer vector synchronization names no content hashes")?;
    let spec = req
        .vector_spec
        .as_ref()
        .context("peer vector synchronization names no embedding profile")?;
    anyhow::ensure!(spec == &ctx.cfg.embeddings.spec(), "local embedding profile changed");
    anyhow::ensure!(
        !hashes.is_empty() && hashes.len() <= vectors::MAX_PEER_ARTIFACTS,
        "peer vector synchronization must name 1..={} hashes",
        vectors::MAX_PEER_ARTIFACTS
    );
    let requested: std::collections::HashSet<&str> = hashes.iter().map(String::as_str).collect();
    anyhow::ensure!(requested.len() == hashes.len(), "peer vector synchronization repeats a hash");

    let remote = Request {
        op: "vector-artifacts".to_string(),
        network_major: Some(crate::embedding_profile::NETWORK_MAJOR),
        slice: Some(slice.to_string()),
        vector_hashes: Some(hashes.clone()),
        vector_spec: Some(spec.clone()),
        ..Request::default()
    };
    let offered = forward_iroh(ctx, target.clone(), &remote, None)?;
    anyhow::ensure!(
        offered.ok,
        "{}",
        offered.error.unwrap_or_else(|| "peer refused vector artifacts".into())
    );
    anyhow::ensure!(
        offered.vector_spec.as_ref() == Some(spec),
        "peer answered with a different embedding profile"
    );
    let artifacts = offered.vector_artifacts.unwrap_or_default();
    anyhow::ensure!(
        artifacts.len() <= hashes.len(),
        "peer offered more artifacts than requested"
    );
    let expected_artifact_len = vectors::peer_artifact_len(spec)?;
    let mut offered_hashes = std::collections::HashSet::with_capacity(artifacts.len());
    for artifact in &artifacts {
        anyhow::ensure!(
            requested.contains(artifact.content_hash.as_str()),
            "peer offered unrequested vector {}",
            artifact.content_hash
        );
        anyhow::ensure!(
            offered_hashes.insert(artifact.content_hash.as_str()),
            "peer offered vector {} twice",
            artifact.content_hash
        );
        // Validate the negotiated size before connecting to the blob
        // transport. The content hash authenticates bytes, not whether those
        // bytes have the profile's one legal shape.
        ensure_exact_peer_artifact_len(artifact.bytes, expected_artifact_len)?;
    }
    if artifacts.is_empty() {
        return Ok(0);
    }

    let client = iroh_client(ctx)?;
    let fetched = client.runtime.block_on(async {
        tokio::time::timeout(IROH_REQUEST_TIMEOUT, async {
            let conn = client
                .endpoint
                .connect(target, iroh_blobs::ALPN)
                .await
                .context("connect peer artifact transport")?;
            let mut records = Vec::with_capacity(artifacts.len());
            for artifact in &artifacts {
                let blob_hash: iroh_blobs::Hash = artifact
                    .blob_hash
                    .parse()
                    .context("peer returned an invalid artifact hash")?;
                fetch_peer_blob_bounded(
                    &client.download_blobs,
                    conn.clone(),
                    blob_hash,
                    expected_artifact_len,
                )
                .await?;
                let raw = client.download_blobs.get_bytes(blob_hash).await?;
                ensure_exact_peer_artifact_len(raw.len(), expected_artifact_len)
                    .context("peer artifact size changed in transit")?;
                let record = vectors::decode_peer_artifact(&raw, spec, &artifact.content_hash)?;
                records.push((artifact.content_hash.clone(), record));
            }
            conn.close(0u32.into(), b"artifacts complete");
            Ok::<_, anyhow::Error>(records)
        })
        .await
        .context("timed out fetching peer vector artifacts")?
    })?;

    let mut store = vectors::VectorStore::open(&ctx.cfg.brain_root, spec)?;
    let mut writer = store.begin_write()?;
    let mut imported = 0usize;
    for (hash, record) in fetched {
        imported += usize::from(writer.put_encoded(&hash, &record)?);
    }
    writer.flush()?;
    Ok(imported)
}

/// Handles a request after QUIC has authenticated `peer` as the remote
/// endpoint id. A ticket secret may create a grant; every subsequent data
/// request must name a slice that this exact peer holds.
fn handle_iroh(req: &Request, peer: &str, ctx: &Ctx) -> Response {
    if req.network_major != Some(crate::embedding_profile::NETWORK_MAJOR) {
        return Response::err(format!(
            "incompatible cfetch network major: peer sent {:?}, this host requires {}",
            req.network_major,
            crate::embedding_profile::NETWORK_MAJOR
        ));
    }
    if req.op == "redeem" {
        let Some(slice) = req.slice.as_deref() else {
            return Response::err("redeem request names no slice");
        };
        let Some(secret) = req.secret.as_deref() else {
            return Response::err("redeem request carries no secret");
        };
        let Some(mode) = req.mode else {
            return Response::err("redeem request names no mode");
        };
        return match grant::redeem(&ctx.cfg.brain_root, slice, secret, mode, peer, now_secs()) {
            Ok(g) => Response {
                ok: true,
                grant: Some(WireGrant { slice: g.slice, mode: g.mode, peer: peer.to_string() }),
                ..Response::default()
            },
            Err(e) => Response::err(e.to_string()),
        };
    }

    // Connection authentication alone says WHO is asking, not what it may
    // read. Even harmless-looking catalog responses are gated so a grant is
    // the single authorization boundary for the remote protocol.
    let Some(slice) = req.slice.as_deref() else {
        return Response::err("iroh requests must name the granted slice");
    };
    match grant::access(&ctx.cfg.brain_root, slice, peer, now_secs()) {
        Ok(Some(_)) => {}
        Ok(None) => return Response::err("this endpoint has no active grant for that slice"),
        Err(e) => return Response::err(e.to_string()),
    }

    match req.op.as_str() {
        "diagnose" => match &ctx.serve {
            Some(state) => Response {
                ok: true,
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
                origin: Some(state.origin.clone()),
                generation: Some(state.generation.load(Ordering::Relaxed)),
                ..Response::default()
            },
            None => Response::err(
                "authenticated endpoint is reachable, but serving is not enabled on this daemon",
            ),
        },
        "vector-artifacts" => offer_peer_vectors(req, peer, slice, ctx),
        "recall" | "generation" => handle(req, ctx).0,
        "checksum" => {
            let model = match ctx.cfg.slice_model() {
                Ok(model) => model,
                Err(e) => return Response::err(e.to_string()),
            };
            serve_query(ctx, |conn| {
                Ok(Response {
                    checksum: Some(index::catalog_checksum_matching(conn, |path| {
                        model.contains(slice, path)
                    })?),
                    ..Response::default()
                })
            })
        }
        "expand" => {
            if ctx.serve.is_none() {
                return Response::err("serving is not enabled on this daemon (config serve.enabled)");
            }
            let cite = req.cite.clone().unwrap_or_default();
            let model = match ctx.cfg.slice_model() {
                Ok(model) => model,
                Err(e) => return Response::err(e.to_string()),
            };
            serve_query(ctx, |conn| {
                let blocks = index::expand(conn, &cite)?
                    .into_iter()
                    .filter(|block| model.contains(slice, &block.path))
                    .map(Into::into)
                    .collect();
                Ok(Response { blocks: Some(blocks), ..Response::default() })
            })
        }
        "graph" => {
            if ctx.serve.is_none() {
                return Response::err("serving is not enabled on this daemon (config serve.enabled)");
            }
            let model = match ctx.cfg.slice_model() {
                Ok(model) => model,
                Err(e) => return Response::err(e.to_string()),
            };
            let focus = req.focus.as_deref();
            let limit = req.limit.unwrap_or(40);
            serve_query(ctx, |conn| {
                let graph = crate::knowledge_graph::build_matching(
                    conn,
                    focus,
                    limit,
                    |path| model.contains(slice, path),
                )?;
                Ok(Response {
                    knowledge_graph: Some(graph),
                    ..Response::default()
                })
            })
        }
        // These operations either mutate the host or are not slice-scoped.
        // Returning them through a slice grant would widen that grant to code
        // roots, daemon control, or resident private context.
        other => Response::err(format!("op {other:?} is not available through a slice grant")),
    }
}

async fn serve_iroh_connection(
    incoming: iroh::endpoint::Incoming,
    ctx: Arc<Ctx>,
    blobs: PeerBlobsProtocol,
) -> anyhow::Result<()> {
    let conn = tokio::time::timeout(IROH_CONNECT_TIMEOUT, incoming)
        .await
        .context("iroh handshake timed out")??;
    if conn.alpn() == iroh_blobs::ALPN {
        return blobs
            .accept(conn)
            .await
            .map_err(|e| anyhow::anyhow!("iroh-blobs provider failed: {e:?}"));
    }
    anyhow::ensure!(conn.alpn() == IROH_ALPN, "unsupported iroh protocol");
    let peer = conn.remote_id().to_string();
    let (mut send, mut recv) = tokio::time::timeout(IROH_CONNECT_TIMEOUT, conn.accept_bi())
        .await
        .context("iroh peer did not open a request stream")??;
    let raw = tokio::time::timeout(
        IROH_CONNECT_TIMEOUT,
        recv.read_to_end(IROH_MAX_REQUEST),
    )
    .await
    .context("iroh request read timed out")??;
    let resp = match serde_json::from_slice::<Request>(&raw) {
        Ok(req) => {
            // Barrier waits, SQLite and grant lockfiles are intentionally
            // synchronous. Keep them off Tokio's networking workers so a few
            // slow queries cannot starve QUIC progress for every peer.
            let ctx = ctx.clone();
            match tokio::task::spawn_blocking(move || handle_iroh(&req, &peer, &ctx)).await {
                Ok(resp) => resp,
                Err(e) => Response::err(format!("request worker failed: {e}")),
            }
        }
        Err(e) => Response::err(format!("bad request: {e}")),
    };
    let mut raw = serde_json::to_vec(&resp)?;
    raw.push(b'\n');
    anyhow::ensure!(raw.len() <= IROH_MAX_RESPONSE, "iroh response is too large");
    send.write_all(&raw).await.context("send iroh response")?;
    send.finish().context("finish iroh response")?;
    // `finish` queues the FIN; dropping the last connection handle
    // immediately can tear down QUIC before the peer has read the response.
    // The client closes after `read_to_end`, so this normally returns at once.
    let _ = tokio::time::timeout(Duration::from_secs(2), conn.closed()).await;
    Ok(())
}

/// Starts the one persistent endpoint for this host and waits until its UDP
/// sockets are bound. WAN discovery/relay selection continues asynchronously;
/// the current direct routes are already enough for a same-LAN invite.
fn start_iroh(ctx: Arc<Ctx>) -> anyhow::Result<()> {
    let secret = net::load_or_create(&paths::state_dir())?;
    let (ready_tx, ready_rx) =
        std::sync::mpsc::sync_channel::<Result<iroh::EndpointAddr, String>>(1);
    std::thread::Builder::new()
        .name("cfetch-iroh".to_string())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("cfetch-iroh-rt")
                .build()
            {
                Ok(runtime) => runtime,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("create iroh runtime: {e}")));
                    return;
                }
            };
            let handle = runtime.handle().clone();
            runtime.block_on(async move {
                let download_store = iroh_blobs::store::mem::MemStore::new();
                let download_blobs: iroh_blobs::api::Store = download_store.into();
                let peer_blobs = PeerBlobStores::default();
                let blob_protocol = PeerBlobsProtocol {
                    stores: peer_blobs.clone(),
                };
                let endpoint = match iroh::Endpoint::builder(iroh::endpoint::presets::N0)
                    .secret_key(secret)
                    .alpns(vec![IROH_ALPN.to_vec(), iroh_blobs::ALPN.to_vec()])
                    .bind()
                    .await
                {
                    Ok(endpoint) => endpoint,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("bind iroh endpoint: {e}")));
                        return;
                    }
                };
                *ctx.iroh
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(IrohClient {
                    endpoint: endpoint.clone(),
                    runtime: handle,
                    download_blobs,
                    peer_blobs,
                });
                let _ = ready_tx.send(Ok(endpoint.addr()));
                let permits = Arc::new(tokio::sync::Semaphore::new(64));
                while let Some(incoming) = endpoint.accept().await {
                    let Ok(permit) = permits.clone().try_acquire_owned() else {
                        // Bound unauthenticated connection work. Dropping the
                        // handshake refuses overload without growing a task
                        // queue an attacker can hold open.
                        drop(incoming);
                        continue;
                    };
                    let ctx = ctx.clone();
                    let blobs = blob_protocol.clone();
                    tokio::spawn(async move {
                        let _permit = permit;
                        if let Err(e) = serve_iroh_connection(incoming, ctx, blobs).await {
                            eprintln!("cfetch iroh: {e:#}");
                        }
                    });
                }
            });
        })
        .context("start iroh runtime thread")?;

    match ready_rx.recv_timeout(Duration::from_secs(10)) {
        Ok(Ok(addr)) => {
            eprintln!("cfetch daemon serving iroh as {}", addr.id);
            Ok(())
        }
        Ok(Err(e)) => anyhow::bail!(e),
        Err(_) => anyhow::bail!("timed out starting the iroh endpoint"),
    }
}

/// Runs a barrier-gated query op against the committed index snapshot and
/// stamps the coherence labels (origin, generation, fresh) on the response.
/// The generation is read from the SAME connection that answers, so the
/// label always matches the snapshot.
fn serve_query(
    ctx: &Ctx,
    f: impl FnOnce(&rusqlite::Connection) -> anyhow::Result<Response>,
) -> Response {
    let Some(state) = &ctx.serve else {
        return Response::err("serving is not enabled on this daemon (config serve.enabled)");
    };
    let outcome = state.barrier(serve::BARRIER_TIMEOUT);
    let conn = match index::open_ro(state.state_dir()) {
        Ok(c) => c,
        Err(e) => return Response::err(format!("open index: {e}")),
    };
    let mut resp = match f(&conn) {
        Ok(r) => r,
        Err(e) => return Response::err(e.to_string()),
    };
    resp.ok = true;
    resp.origin = Some(state.origin.clone());
    resp.generation = Some(index::generation(&conn));
    resp.fresh = Some(outcome.fresh);
    resp.stale_note = outcome.note;
    resp.barrier_ms = Some(outcome.waited_ms);
    crate::runtime_status::record_memory_answer(
        crate::runtime_status::MemoryRoute::Serving,
        resp.generation,
        resp.fresh,
        true,
    );
    resp
}

fn handle(req: &Request, ctx: &Ctx) -> (Response, bool) {
    match req.op.as_str() {
        "ping" => (
            Response {
                ok: true,
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
                ..Response::default()
            },
            false,
        ),
        "iroh-addr" => match iroh_client(ctx) {
            Ok(client) => (
                Response {
                    ok: true,
                    iroh_addr: Some(dialable_iroh_addr(&client)),
                    ..Response::default()
                },
                false,
            ),
            Err(e) => (Response::err(e.to_string()), false),
        },
        "iroh-forward" => {
            let Some(target) = req.target.clone() else {
                return (Response::err("iroh forward request names no target"), false);
            };
            let Some(remote) = req.request.as_deref() else {
                return (Response::err("iroh forward request has no payload"), false);
            };
            let diagnostic_timeout = req
                .timeout_ms
                .map(|ms| Duration::from_millis(ms.clamp(100, 5_000)));
            match forward_iroh(ctx, target, remote, diagnostic_timeout) {
                Ok(mut resp) => {
                    resp.iroh_connected = Some(true);
                    (resp, false)
                }
                Err(e) => (Response::err(e.to_string()), false),
            }
        }
        "vector-sync" => match fetch_peer_vectors(req, ctx) {
            Ok(imported) => (
                Response {
                    ok: true,
                    vectors_imported: Some(imported),
                    ..Response::default()
                },
                false,
            ),
            Err(e) => (Response::err(e.to_string()), false),
        },
        "resident" => {
            // Config is reloaded per request: a startup snapshot would make
            // the warm path silently diverge from the daemon-less fallback
            // after a config edit.
            match Config::load() {
                Ok(cfg) => {
                    let scope = resident::SessionScope::from_cwd(req.cwd.as_deref());
                    let d = resident::build(&cfg, &scope);
                    (Response { ok: true, digest: Some(d.text), ..Response::default() }, false)
                }
                Err(e) => (Response::err(e.to_string()), false),
            }
        }
        "health" => {
            let degraded = heartbeat::degraded().into_iter().map(|(n, _)| n).collect();
            (Response { ok: true, degraded_hooks: Some(degraded), ..Response::default() }, false)
        }
        "scan-code" => {
            if begin_code_scan() {
                (Response { ok: true, scan: Some(SCAN.status()), ..Response::default() }, false)
            } else {
                (
                    Response {
                        ok: false,
                        error: Some("a code scan is already running".to_string()),
                        scan: Some(SCAN.status()),
                        ..Response::default()
                    },
                    false,
                )
            }
        }
        "scan-status" => (Response { ok: true, scan: Some(SCAN.status()), ..Response::default() }, false),
        "serve-status" => {
            let info = match &ctx.serve {
                Some(s) => ServeInfo {
                    enabled: true,
                    origin: s.origin.clone(),
                    generation: s.generation.load(Ordering::Relaxed),
                    last_barrier_ms: s.last_barrier_ms.load(Ordering::Relaxed),
                    bind: s.bind_addr.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone(),
                    barrier_mode: s.mode().label().to_string(),
                },
                None => ServeInfo {
                    enabled: false,
                    origin: String::new(),
                    generation: 0,
                    last_barrier_ms: 0,
                    bind: None,
                    barrier_mode: String::new(),
                },
            };
            (Response { ok: true, serve: Some(info), ..Response::default() }, false)
        }
        "recall" => {
            let query = req.query.clone().unwrap_or_default();
            let limit = req.limit.unwrap_or(8);
            let semantic = req.semantic.unwrap_or(false);
            let hybrid = req.hybrid.unwrap_or(false);
            let slice = req.slice.clone();
            let cfg = ctx.cfg.clone();
            (
                serve_query(ctx, |conn| {
                    let r = crate::pipeline::ranked(
                        &cfg,
                        conn,
                        &query,
                        limit,
                        semantic,
                        hybrid,
                        slice.as_deref(),
                    )?;
                    Ok(Response {
                        hits: Some(r.hits.into_iter().map(Into::into).collect()),
                        note: r.note,
                        ..Response::default()
                    })
                }),
                false,
            )
        }
        "expand" => {
            let cite = req.cite.clone().unwrap_or_default();
            (
                serve_query(ctx, |conn| {
                    let blocks = index::expand(conn, &cite)?;
                    Ok(Response {
                        blocks: Some(blocks.into_iter().map(Into::into).collect()),
                        ..Response::default()
                    })
                }),
                false,
            )
        }
        "find" => {
            let query = req.query.clone().unwrap_or_default();
            let limit = req.limit.unwrap_or(10);
            (
                serve_query(ctx, |conn| {
                    let hits = crate::code::find(conn, &query, limit)?;
                    Ok(Response {
                        code_hits: Some(hits.into_iter().map(Into::into).collect()),
                        ..Response::default()
                    })
                }),
                false,
            )
        }
        "map" => {
            // The repo map is a pure read over the committed catalog, so it
            // serves exactly like find: same barrier, same coherence labels.
            // (scan and embed-index stay local to the storage host — they
            // WRITE.) The code roots come from the serving host's config,
            // reloaded per request like every other config read here.
            let focus = req.focus.clone();
            let budget = req.budget_tokens.unwrap_or(crate::graph::DEFAULT_MAP_BUDGET_TOKENS);
            (
                serve_query(ctx, |conn| {
                    let cfg = Config::load()?;
                    let m = crate::graph::map(
                        conn,
                        &cfg.effective_code_roots(),
                        focus.as_deref(),
                        budget,
                    )?;
                    Ok(Response { map: Some(m.into()), ..Response::default() })
                }),
                false,
            )
        }
        "code-path" => {
            let from = req.from_path.clone().unwrap_or_default();
            let to = req.to_path.clone().unwrap_or_default();
            let depth = req.depth.unwrap_or(crate::graph::DEFAULT_PATH_DEPTH);
            (
                serve_query(ctx, |conn| {
                    let cfg = Config::load()?;
                    let path = crate::graph::dependency_path(
                        conn,
                        &cfg.effective_code_roots(),
                        &from,
                        &to,
                        depth,
                    )?;
                    Ok(Response {
                        dependency_path: Some(path),
                        ..Response::default()
                    })
                }),
                false,
            )
        }
        "code-impact" => {
            let target = req.path.clone().unwrap_or_default();
            let depth = req.depth.unwrap_or(crate::graph::DEFAULT_IMPACT_DEPTH);
            let limit = req.limit.unwrap_or(crate::graph::DEFAULT_IMPACT_LIMIT);
            (
                serve_query(ctx, |conn| {
                    let cfg = Config::load()?;
                    let impact = crate::graph::dependency_impact(
                        conn,
                        &cfg.effective_code_roots(),
                        &target,
                        depth,
                        limit,
                    )?;
                    Ok(Response {
                        dependency_impact: Some(impact),
                        ..Response::default()
                    })
                }),
                false,
            )
        }
        "code-context" => {
            let target = req.path.clone().unwrap_or_default();
            let depth = req.depth.unwrap_or(crate::graph::DEFAULT_CONTEXT_DEPTH);
            let limit = req.limit.unwrap_or(crate::graph::DEFAULT_CONTEXT_LIMIT);
            (
                serve_query(ctx, |conn| {
                    let cfg = Config::load()?;
                    let context = crate::graph::dependency_context(
                        conn,
                        &cfg.effective_code_roots(),
                        &target,
                        depth,
                        limit,
                    )?;
                    Ok(Response {
                        dependency_context: Some(context),
                        ..Response::default()
                    })
                }),
                false,
            )
        }
        "code-symbol" => {
            let query = req.query.clone().unwrap_or_default();
            let limit = req.limit.unwrap_or(crate::graph::DEFAULT_SYMBOL_LIMIT);
            (
                serve_query(ctx, |conn| {
                    let cfg = Config::load()?;
                    let context = crate::graph::symbol_context(
                        conn,
                        &cfg.effective_code_roots(),
                        &query,
                        limit,
                    )?;
                    Ok(Response {
                        symbol_context: Some(context),
                        ..Response::default()
                    })
                }),
                false,
            )
        }
        "graph" => {
            let focus = req.focus.as_deref();
            let limit = req.limit.unwrap_or(40);
            let slice = req.slice.clone();
            (
                serve_query(ctx, |conn| {
                    let graph = if let Some(slice) = slice.as_deref() {
                        let model = ctx.cfg.slice_model()?;
                        anyhow::ensure!(
                            slice == crate::config::ROOT_SLICE
                                || model.names().any(|name| name == slice),
                            "unknown slice {slice:?}"
                        );
                        crate::knowledge_graph::build_matching(conn, focus, limit, |path| {
                            model.contains(slice, path)
                        })?
                    } else {
                        crate::knowledge_graph::build(conn, focus, limit)?
                    };
                    Ok(Response {
                        knowledge_graph: Some(graph),
                        ..Response::default()
                    })
                }),
                false,
            )
        }
        "slices" => {
            // Hook advisory path: read-only, deliberately WITHOUT the
            // barrier — a hint must never cost seconds on the interactive
            // path, and a slightly stale line range is a missed optimization,
            // not wrong knowledge.
            if ctx.serve.is_none() {
                return (
                    Response::err("serving is not enabled on this daemon (config serve.enabled)"),
                    false,
                );
            }
            let path = req.path.clone().unwrap_or_default();
            let limit = req.limit.unwrap_or(5);
            let slices = hooks::symbol_slices(&paths::state_dir().join("index.db"), &path, limit);
            (Response { ok: true, slices: Some(slices), ..Response::default() }, false)
        }
        "generation" => (serve_query(ctx, |_conn| Ok(Response::default())), false),
        "checksum" => (
            serve_query(ctx, |conn| {
                Ok(Response { checksum: Some(index::catalog_checksum(conn)?), ..Response::default() })
            }),
            false,
        ),
        "shutdown" => (Response { ok: true, ..Response::default() }, true),
        other => (Response::err(format!("unknown op: {other}")), false),
    }
}

/// Serves one connection: one request line, one response line. Returns true
/// when this connection requested shutdown (local channels only).
fn serve_conn<S: Read + Write>(stream: S, ctx: &Ctx, chan: Channel) -> bool {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let read = {
        let mut limited = std::io::Read::take(&mut reader, MAX_LINE_REQUEST + 1);
        std::io::BufRead::read_line(&mut limited, &mut line)
    };
    if read.is_err() {
        return false;
    }
    let (resp, shutdown) = if line.len() as u64 > MAX_LINE_REQUEST {
        (Response::err("request exceeds the 64 KiB protocol limit"), false)
    } else {
        match serde_json::from_str::<Request>(&line) {
            Ok(req) => {
                // Bearer gate wherever the transport is not access-controlled by
                // the operating system: every op requires the token.
                if matches!(chan, Channel::Remote)
                    && req.network_major != Some(crate::embedding_profile::NETWORK_MAJOR)
                {
                    (
                        Response::err(format!(
                            "incompatible cfetch network major: client sent {:?}, server requires {}",
                            req.network_major,
                            crate::embedding_profile::NETWORK_MAJOR
                        )),
                        false,
                    )
                } else if let Some(expected) = chan.expected_token(ctx)
                    && !authorized(expected, req.token.as_ref())
                {
                    (Response::err("unauthorized"), false)
                } else if matches!(
                    req.op.as_str(),
                    "shutdown" | "iroh-addr" | "iroh-forward" | "vector-sync"
                )
                    && !chan.allows_local_only()
                {
                    (Response::err(format!("{} is local-only", req.op)), false)
                } else if !chan.allows_op(&req.op) {
                    (
                        Response::err(format!(
                            "{} is not available on the remote serving channel",
                            req.op
                        )),
                        false,
                    )
                } else {
                    let (r, shutdown) = handle(&req, ctx);
                    (r, shutdown && chan.allows_shutdown())
                }
            }
            Err(e) => (Response::err(format!("bad request: {e}")), false),
        }
    };
    let mut stream = reader.into_inner();
    if let Ok(s) = serde_json::to_string(&resp) {
        if s.len() > IROH_MAX_RESPONSE {
            // The iroh path refuses to build a response past its size bound;
            // the TCP path must not become the bypass. A client-sized
            // `limit: usize::MAX` gets the same refusal instead of a
            // whole-catalog serialization and an equally huge write.
            let refused = Response::err("response exceeds the 16 MiB serving limit");
            if let Ok(e) = serde_json::to_string(&refused) {
                let _ = writeln!(stream, "{e}");
            }
        } else {
            let _ = writeln!(stream, "{s}");
        }
    }
    shutdown
}

/// Foreground run loop (systemd/`daemon start` both end up here).
pub fn run() -> anyhow::Result<()> {
    // Serving mode needs the config at boot; a corrupt config fails LOUDLY
    // here (hooks then use their daemon-less fallbacks) instead of silently
    // starting a daemon that cannot serve.
    let cfg = Config::load()?;
    let _ = crate::runtime_status::refresh_static();
    // Every storage host keeps the disposable index synchronized with direct
    // Obsidian edits. Serving is only one consumer of this watcher; local
    // recall, graph, and autonomous vector upkeep need the same freshness.
    let index_handle = if cfg.client.serving.is_none() {
        Some(serve::start(&cfg)?)
    } else {
        None
    };
    let serving_state = if cfg.serve.enabled {
        index_handle.as_ref().map(|handle| handle.state.clone())
    } else {
        None
    };
    let tcp_token = match (&cfg.serve.bind, &cfg.serve.token_file) {
        (Some(_), Some(tf)) => Some(serve::read_token(tf, true)?),
        _ => None,
    };
    // Minted before anything binds so the request context can carry it
    // without reordering the boot sequence; `None` on unix.
    let local_token = ipc::new_local_token();
    let ctx = Arc::new(Ctx {
        cfg: Arc::new(cfg.clone()),
        serve: serving_state,
        tcp_token: tcp_token.clone(),
        local_token: local_token.clone(),
        iroh: Mutex::new(None),
        artifact_key: iroh::SecretKey::generate().to_bytes(),
        shutdown: AtomicBool::new(false),
    });
    start_iroh(ctx.clone())?;

    if cfg.maintenance.enabled && cfg.maintenance.configured() {
        let maintenance_cfg = cfg.clone();
        let maintenance_ctx = ctx.clone();
        std::thread::Builder::new()
            .name("cfetch-maintenance".into())
            .spawn(move || {
                maintenance_worker::run(maintenance_cfg, || {
                    maintenance_ctx.shutdown.load(Ordering::SeqCst)
                });
            })
            .context("start maintenance worker")?;
    }

    if cfg.client.serving.is_none() {
        let vector_cfg = cfg.clone();
        let vector_ctx = ctx.clone();
        std::thread::Builder::new()
            .name("cfetch-vectors".into())
            .spawn(move || {
                vector_worker::run(vector_cfg, || {
                    vector_ctx.shutdown.load(Ordering::SeqCst)
                });
            })
            .context("start vector maintenance worker")?;
    }

    if let (Some(bind), Some(_)) = (&cfg.serve.bind, &tcp_token) {
        let listener = std::net::TcpListener::bind(bind)?;
        let local = listener.local_addr()?;
        // Record the actual bound address (resolves ":0" configs) for
        // status/selfcheck and the torture harness.
        std::fs::write(paths::state_dir().join("serve.addr"), local.to_string())?;
        if let Some(h) = &index_handle {
            *h.state.bind_addr.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
                Some(local.to_string());
        }
        eprintln!("cfetch daemon serving TCP on {local}");
        let ctx = ctx.clone();
        // Bound pre-auth connection work, mirroring the iroh accept loop's
        // permit cap: the token is only known AFTER a full line arrives, so
        // an unbounded thread-per-connection lets anyone who can open
        // sockets hold one 2 MiB-stack thread each without knowing the
        // token. Excess connections are dropped before any read.
        const MAX_TCP_CONNECTIONS: usize = 64;
        let live = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        std::thread::spawn(move || {
            for conn in listener.incoming().flatten() {
                if live.fetch_add(1, std::sync::atomic::Ordering::AcqRel) >= MAX_TCP_CONNECTIONS {
                    live.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                    drop(conn);
                    continue;
                }
                let _ = conn.set_read_timeout(Some(Duration::from_secs(5)));
                let _ = conn.set_write_timeout(Some(Duration::from_secs(5)));
                let ctx = ctx.clone();
                let live = live.clone();
                std::thread::spawn(move || {
                    let _ = serve_conn(conn, &ctx, Channel::Remote);
                    live.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                });
            }
        });
    }

    // The daemon's own code-scan cadence. Detached: the listener bind below
    // must not wait for a code scan, and neither must any query.
    if let Some(h) = &index_handle {
        let state = h.state.clone();
        let deadline = Instant::now() + FIRST_SCAN_SETTLE_WAIT;
        std::thread::spawn(move || {
            scan_cadence(
                || first_scan_ready(&state, deadline),
                begin_code_scan,
                SCAN_READY_POLL,
                serve::FINGERPRINT_INTERVAL,
                None,
            );
        });
    }

    // A stale endpoint from a dead daemon is cleared; a live one is never
    // stolen (see `ipc::listen`).
    let listener = ipc::listen(local_token)?;
    crate::runtime_status::record_service(crate::runtime_status::ServiceState::Ready, None);
    eprintln!("cfetch daemon listening on {}", listener.describe());
    for conn in listener.incoming() {
        if ctx.shutdown.load(Ordering::SeqCst) {
            break;
        }
        let Ok(conn) = conn else { continue };
        let _ = conn.set_read_timeout(Some(Duration::from_secs(5)));
        let _ = conn.set_write_timeout(Some(Duration::from_secs(5)));
        // Thread per connection: a query blocked in the drain barrier (up to
        // 5s) must not starve other clients — hooks poll this channel on the
        // interactive path.
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            if serve_conn(conn, &ctx, LOCAL_CHANNEL) {
                ctx.shutdown.store(true, Ordering::SeqCst);
                // Wake the accept loop so it observes the flag.
                ipc::wake();
            }
        });
    }
    if ctx.shutdown.load(Ordering::SeqCst) {
        // A daemon that acknowledged shutdown must not become a zombie.
        // The graceful tail below (iroh endpoint close, listener cleanup)
        // can block on network drains indefinitely; the watchdog bounds
        // the whole shutdown to five seconds — measured on a real host,
        // the acknowledged-but-alive process outlived its operator's
        // patience by minutes.
        std::thread::Builder::new()
            .name("cfetch-shutdown-watchdog".into())
            .spawn(|| {
                std::thread::sleep(Duration::from_secs(5));
                std::process::exit(0);
            })
            .ok();
    }
    if let Ok(client) = iroh_client(&ctx) {
        client.runtime.block_on(client.endpoint.close());
    }
    listener.cleanup();
    crate::runtime_status::record_service(
        crate::runtime_status::ServiceState::Degraded,
        Some("daemon_unavailable"),
    );
    Ok(())
}

/// Detached start: re-executes this binary with `daemon run`.
pub fn start() -> anyhow::Result<()> {
    if call("ping", Duration::from_millis(300)).is_some() {
        println!("daemon already running");
        return Ok(());
    }
    let exe = std::env::current_exe()?;
    // The daemon's stderr goes to a file, not /dev/null: when a start fails,
    // "did not answer" alone sends the operator nowhere. The file lives in
    // the state directory and is truncated on every start — it describes
    // the LAST start attempt, not an ever-growing log.
    let stderr_path = crate::paths::state_dir().join("daemon-start.stderr.log");
    let stderr_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&stderr_path);
    let mut cmd = std::process::Command::new(exe);
    cmd.args(["daemon", "run"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null());
    match &stderr_file {
        Ok(file) => {
            cmd.stderr(std::process::Stdio::from(file.try_clone()?));
        }
        Err(_) => {
            // The state dir may be unwritable — starting is still worth the
            // attempt, only the diagnosis gets poorer.
            cmd.stderr(std::process::Stdio::null());
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: the daemon outlives
        // the console that started it and never receives its Ctrl-C. Closing
        // stdio is enough on unix; on Windows the console is inherited
        // separately from the handles.
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
    }
    cmd.spawn()?;
    for _ in 0..20 {
        std::thread::sleep(Duration::from_millis(50));
        if call("ping", Duration::from_millis(200)).is_some() {
            println!("daemon started on {}", ipc::describe());
            return Ok(());
        }
    }
    // Name the actual reason when we captured one: the last few lines of
    // the failed start's stderr say more than "did not answer" ever will.
    let mut reason = String::new();
    if stderr_file.is_ok()
        && let Ok(tail) = std::fs::read_to_string(&stderr_path)
    {
        let lines: Vec<&str> = {
            let all: Vec<&str> = tail.lines().collect();
            all.split_at(all.len().saturating_sub(5)).1.to_vec()
        };
        if !lines.is_empty() {
            reason = format!("\ndaemon stderr ({}):\n  {}", stderr_path.display(), lines.join("\n  "));
        }
    }
    anyhow::bail!("daemon did not answer after start{reason}")
}

pub fn stop() -> anyhow::Result<()> {
    match call("shutdown", Duration::from_millis(500)) {
        Some(r) if r.ok => {
            // An acknowledged shutdown is a promise, not a fact: the daemon
            // may still be draining. Poll until it stops ANSWERING — only
            // then is "stopped" a measurement — and refuse to claim success
            // while the process demonstrably lives on.
            for _ in 0..50 {
                std::thread::sleep(Duration::from_millis(100));
                if call("ping", Duration::from_millis(200)).is_none() {
                    println!("daemon stopped");
                    return Ok(());
                }
            }
            anyhow::bail!(
                "daemon acknowledged shutdown but is still answering after 5s — it may be blocked; check the process"
            );
        }
        _ => {
            println!("daemon not running");
            Ok(())
        }
    }
}

pub fn status() -> anyhow::Result<()> {
    // The mode LEADS: which side of the serving topology this host is on is
    // the first thing an operator needs, not a footnote under the ledger.
    let cfg = Config::load().ok();
    let info = call("serve-status", Duration::from_millis(300)).and_then(|r| r.serve);
    match &cfg {
        Some(c) => println!("{}", mode_line(c, info.as_ref())),
        None => println!("mode: unknown (config does not load)"),
    }
    match call("ping", Duration::from_millis(300)) {
        Some(r) => {
            crate::runtime_status::record_service(crate::runtime_status::ServiceState::Ready, None);
            println!("daemon: running (v{})", r.version.unwrap_or_default());
            if let Some(info) = info.filter(|i| i.enabled) {
                println!(
                    "serving: origin {}, generation {}, drain barrier {}, last barrier {} ms, {}",
                    info.origin,
                    info.generation,
                    barrier_mode_of(&info),
                    info.last_barrier_ms,
                    info.bind
                        .clone()
                        .map_or_else(|| "local channel only".to_string(), |b| format!("tcp {b}"))
                );
            }
            if let Some(s) = call("scan-status", Duration::from_millis(300)).and_then(|r| r.scan) {
                if s.running {
                    println!("code scan: running in the background");
                } else if let Some(t) = s.last_finished {
                    let ago = now_secs().saturating_sub(t);
                    match &s.last_counts {
                        Some(c) => println!(
                            "code scan: finished {ago}s ago ({} files, {} symbols, {} import edges)",
                            c.files, c.symbols, c.edges
                        ),
                        None => println!("code scan: finished {ago}s ago"),
                    }
                }
                if let Some(e) = &s.last_error {
                    println!("code scan: last error: {e}");
                }
            }
        }
        None => {
            crate::runtime_status::record_service(
                crate::runtime_status::ServiceState::Degraded,
                Some("daemon_unavailable"),
            );
            println!("daemon: not running ({})", ipc::describe());
        }
    }
    // "healthy" used to mean "nothing has failed", which is also what a host
    // where no hook has ever run reports. The liveness picture compares the
    // registered set against what actually reported, so silence reads as
    // silence.
    let liveness = heartbeat::liveness();
    println!("{}", liveness.summary());
    for h in liveness.degraded() {
        if let heartbeat::HookState::Failing { consecutive, last_error, .. } = &h.state {
            println!(
                "hooks: {} failing ({consecutive} consecutive; last: {})",
                h.name,
                last_error.as_deref().unwrap_or_default()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_request_carries_the_session_cwd() {
        // The daemon shares the host with its caller but not the working
        // directory; without this field every scoped entry would be decided
        // against the daemon's own cwd.
        let req: Request =
            serde_json::from_str(r#"{"op":"resident","cwd":"/srv/work/widget"}"#).unwrap();
        assert_eq!(req.cwd.as_deref(), Some("/srv/work/widget"));
        let scope = resident::SessionScope::from_cwd(req.cwd.as_deref());
        assert_eq!(scope.repo.as_deref(), Some("widget"));

        // An older client sends no cwd at all: host-scoped entries still
        // resolve, repo-scoped ones simply do not match.
        let old: Request = serde_json::from_str(r#"{"op":"resident"}"#).unwrap();
        assert!(old.cwd.is_none());
        assert!(resident::SessionScope::from_cwd(None).repo.is_none());
    }

    #[test]
    fn scan_coordinator_is_single_flight() {
        let c = ScanCoordinator::new();
        assert!(c.try_begin());
        assert!(!c.try_begin(), "a second scan must be refused while one runs");
        assert!(c.status().running);
        c.complete(Ok(ScanCounts { files: 3, symbols: 5, edges: 2 }));
        let s = c.status();
        assert!(!s.running);
        assert!(s.last_finished.is_some());
        assert_eq!(s.last_counts.as_ref().map(|c| c.files), Some(3));
        assert!(c.try_begin(), "a finished scan frees the slot");
    }

    #[test]
    fn scan_coordinator_error_keeps_last_good_counts() {
        let c = ScanCoordinator::new();
        assert!(c.try_begin());
        c.complete(Ok(ScanCounts { files: 7, symbols: 9, edges: 1 }));
        assert!(c.try_begin());
        c.complete(Err("boom".to_string()));
        let s = c.status();
        assert!(!s.running);
        assert_eq!(s.last_error.as_deref(), Some("boom"));
        assert_eq!(s.last_counts.as_ref().map(|c| c.files), Some(7), "an error must not erase the last good counts");
        assert!(c.try_begin());
        c.complete(Ok(ScanCounts { files: 8, symbols: 9, edges: 1 }));
        assert!(c.status().last_error.is_none(), "success clears the error");
    }

    fn no_serve_ctx() -> Ctx {
        Ctx {
            cfg: Arc::new(crate::config::Config::default()),
            serve: None,
            tcp_token: None,
            local_token: None,
            iroh: Mutex::new(None),
            artifact_key: [0; 32],
            shutdown: AtomicBool::new(false),
        }
    }

    #[test]
    fn artifact_capabilities_are_stable_but_bound_to_peer_and_slice() {
        let ctx = no_serve_ctx();
        let hash = "ab".repeat(32);
        let a = artifact_nonce(&ctx, "peer-a", "shared", &hash);
        assert_eq!(a, artifact_nonce(&ctx, "peer-a", "shared", &hash));
        assert_ne!(a, artifact_nonce(&ctx, "peer-b", "shared", &hash));
        assert_ne!(a, artifact_nonce(&ctx, "peer-a", "private", &hash));

        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let _entered = runtime.enter();
        let stores = PeerBlobStores::default();
        stores.get_or_create("peer-a");
        assert!(stores.get("peer-a").is_some());
        assert!(stores.get("peer-b").is_none(), "another endpoint cannot see peer-a's blobs");
    }

    #[test]
    fn peer_artifact_offer_and_fetch_progress_are_exactly_bounded() {
        let expected = 872;
        ensure_exact_peer_artifact_len(expected, expected).unwrap();
        assert!(ensure_exact_peer_artifact_len(expected - 1, expected).is_err());
        assert!(ensure_exact_peer_artifact_len(expected + 1, expected).is_err());

        ensure_peer_fetch_progress(0, expected as u64, expected).unwrap();
        ensure_peer_fetch_progress(320, 552, expected).unwrap();
        let error = ensure_peer_fetch_progress(320, 553, expected)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("exceeded") && error.contains("872"),
            "{error}"
        );
        assert!(ensure_peer_fetch_progress(expected as u64 + 1, 0, expected).is_err());
        let overflow = ensure_peer_fetch_progress(u64::MAX, 1, usize::MAX)
            .unwrap_err()
            .to_string();
        assert!(overflow.contains("overflowed"), "{overflow}");
    }

    #[test]
    fn daemon_start_kicks_exactly_one_code_scan() {
        // A fabricated daemon start: watch registration takes two polls, then
        // the cadence runs two rounds. Round one begins THE scan (nobody sent
        // `scan-code` — the daemon kicks itself); round two is refused because
        // that scan is still running. Single flight, one scan.
        use std::sync::atomic::AtomicUsize;
        let polls = AtomicUsize::new(0);
        let coord = ScanCoordinator::new();
        let kicked = AtomicUsize::new(0);
        let begun = AtomicUsize::new(0);
        scan_cadence(
            || polls.fetch_add(1, Ordering::SeqCst) >= 2,
            || {
                kicked.fetch_add(1, Ordering::SeqCst);
                let started = coord.try_begin();
                if started {
                    begun.fetch_add(1, Ordering::SeqCst);
                }
                started
            },
            Duration::from_millis(1),
            Duration::from_millis(1),
            Some(2),
        );
        assert!(
            polls.load(Ordering::SeqCst) >= 3,
            "the cadence must wait for watch registration before scanning"
        );
        assert_eq!(kicked.load(Ordering::SeqCst), 2, "the cadence repeats on the fingerprint tick");
        assert_eq!(
            begun.load(Ordering::SeqCst),
            1,
            "single flight: a second scan must not start while the first runs"
        );
        assert!(coord.status().running);
    }

    #[test]
    fn the_first_code_scan_waits_for_the_tree_index_to_settle() {
        let dir = tempfile::tempdir().unwrap();
        let state = serve::ServeState::new(
            "o".to_string(),
            dir.path().to_path_buf(),
            dir.path().join("barrier"),
        );
        let far = Instant::now() + Duration::from_secs(300);
        assert!(!first_scan_ready(&state, far), "nothing is ready at startup");
        state.mark_watches_ready();
        assert!(
            !first_scan_ready(&state, far),
            "watches alone must not start a cold code scan against the tree index's writer"
        );
        state.mark_applied(0, 1);
        assert!(first_scan_ready(&state, far), "a settled tree index releases the code scan");

        // A tree index that never settles must not mean a code index that
        // never exists: the bounded wait releases the scan anyway.
        let stuck = serve::ServeState::new(
            "o".to_string(),
            dir.path().to_path_buf(),
            dir.path().join("barrier"),
        );
        let past = Instant::now() - Duration::from_secs(1);
        assert!(!first_scan_ready(&stuck, past), "watches are still the floor");
        stuck.mark_watches_ready();
        assert!(first_scan_ready(&stuck, past));
    }

    #[test]
    fn status_states_the_serving_mode_in_one_line() {
        let mut cfg = Config::default();
        cfg.serve.enabled = true;
        cfg.serve.origin = Some("storage-1".to_string());
        let info = ServeInfo {
            enabled: true,
            origin: "storage-1".to_string(),
            generation: 42,
            last_barrier_ms: 3,
            bind: Some("198.51.100.7:9737".to_string()),
            barrier_mode: serve::BarrierMode::Unordered.label().to_string(),
        };
        let line = mode_line(&cfg, Some(&info));
        assert!(line.starts_with("mode: serving host storage-1"), "{line}");
        assert!(line.contains("generation 42"), "{line}");
        assert!(line.contains("198.51.100.7:9737"), "{line}");
        // The operator must be able to READ which coherence path is in force.
        assert!(line.contains("barrier unordered (fingerprint)"), "{line}");
        // A serving host too old to report one must not read as the fast path.
        let unreported = ServeInfo { barrier_mode: String::new(), ..info.clone() };
        assert!(
            mode_line(&cfg, Some(&unreported)).contains("barrier mode not reported"),
            "an unreported mode must never imply a guarantee"
        );
        assert_eq!(line.lines().count(), 1, "the mode must be ONE line: {line}");
        // Daemon down: still obviously a serving host, generation unknown.
        let down = mode_line(&cfg, None);
        assert!(down.starts_with("mode: serving host storage-1"), "{down}");
        assert!(down.contains("daemon"), "{down}");
    }

    #[test]
    fn status_states_the_none_tier_mode_in_one_line() {
        let mut cfg = Config::default();
        cfg.client.serving = Some(crate::config::ClientServingConfig {
            addr: "198.51.100.7:9737".to_string(),
            token_file: std::path::PathBuf::from("/var/empty/token"),
        });
        let line = mode_line(&cfg, None);
        assert!(line.starts_with("mode: none-tier"), "{line}");
        assert!(line.contains("198.51.100.7:9737"), "the serving address must be on the line: {line}");
        assert_eq!(line.lines().count(), 1, "the mode must be ONE line: {line}");
        // A host with neither role still says which mode it is in.
        let plain = mode_line(&Config::default(), None);
        assert!(plain.starts_with("mode: local"), "{plain}");
    }

    #[test]
    fn query_ops_refuse_when_serving_disabled() {
        let ctx = no_serve_ctx();
        for op in [
            "recall",
            "expand",
            "find",
            "map",
            "code-path",
            "code-impact",
            "code-context",
            "code-symbol",
            "graph",
            "slices",
            "generation",
            "checksum",
        ] {
            let (resp, shutdown) =
                handle(&Request { op: op.to_string(), ..Request::default() }, &ctx);
            assert!(!resp.ok, "{op} must refuse without serve.enabled");
            assert!(resp.error.unwrap().contains("serve.enabled"), "{op} error must name the fix");
            assert!(!shutdown);
        }
    }

    #[test]
    fn serve_status_reports_disabled_without_serving() {
        let ctx = no_serve_ctx();
        let (resp, _) = handle(&Request { op: "serve-status".into(), ..Request::default() }, &ctx);
        assert!(resp.ok);
        assert!(!resp.serve.unwrap().enabled);
    }

    /// In-memory duplex "stream" for serve_conn tests.
    struct Duplex {
        input: std::io::Cursor<Vec<u8>>,
        output: Vec<u8>,
    }

    impl Read for Duplex {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.input.read(buf)
        }
    }
    impl Write for Duplex {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.output.write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn roundtrip(ctx: &Ctx, chan: Channel, mut body: serde_json::Value) -> Response {
        if matches!(chan, Channel::Remote) && body.get("network_major").is_none() {
            body["network_major"] = serde_json::json!(crate::embedding_profile::NETWORK_MAJOR);
        }
        let mut stream = Duplex { input: std::io::Cursor::new(format!("{body}\n").into_bytes()), output: Vec::new() };
        serve_conn(&mut stream, ctx, chan);
        serde_json::from_slice(&stream.output).unwrap()
    }

    /// Did this connection ask for shutdown, and was it granted?
    fn roundtrip_shutdown(ctx: &Ctx, chan: Channel, mut body: serde_json::Value) -> bool {
        if matches!(chan, Channel::Remote) && body.get("network_major").is_none() {
            body["network_major"] = serde_json::json!(crate::embedding_profile::NETWORK_MAJOR);
        }
        let mut stream = Duplex { input: std::io::Cursor::new(format!("{body}\n").into_bytes()), output: Vec::new() };
        serve_conn(&mut stream, ctx, chan)
    }

    #[test]
    fn tcp_requires_the_bearer_token_for_every_op() {
        let ctx = Ctx {
            cfg: Arc::new(crate::config::Config::default()),
            serve: None,
            tcp_token: Some("right-token".to_string()),
            local_token: None,
            iroh: Mutex::new(None),
            artifact_key: [0; 32],
            shutdown: AtomicBool::new(false),
        };
        let r = roundtrip(&ctx, Channel::Remote, serde_json::json!({"op": "ping"}));
        assert!(!r.ok, "missing token must be refused");
        assert_eq!(r.error.as_deref(), Some("unauthorized"));
        let r = roundtrip(&ctx, Channel::Remote, serde_json::json!({"op": "ping", "token": "wrong-token"}));
        assert_eq!(r.error.as_deref(), Some("unauthorized"));
        let r = roundtrip(&ctx, Channel::Remote, serde_json::json!({"op": "ping", "token": "right-token"}));
        assert!(r.ok);
    }

    #[test]
    fn remote_network_majors_fail_closed_before_serving_data() {
        let ctx = Ctx {
            cfg: Arc::new(crate::config::Config::default()),
            serve: None,
            tcp_token: Some("right-token".to_string()),
            local_token: None,
            iroh: Mutex::new(None),
            artifact_key: [0; 32],
            shutdown: AtomicBool::new(false),
        };
        let r = roundtrip(
            &ctx,
            Channel::Remote,
            serde_json::json!({"op": "ping", "token": "right-token", "network_major": 2}),
        );
        assert!(!r.ok);
        assert!(r.error.unwrap().contains("incompatible cfetch network major"));
    }

    #[test]
    fn tcp_token_never_widens_the_serving_port_into_host_control() {
        let ctx = Ctx {
            cfg: Arc::new(crate::config::Config::default()),
            serve: None,
            tcp_token: Some("right-token".to_string()),
            local_token: None,
            iroh: Mutex::new(None),
            artifact_key: [0; 32],
            shutdown: AtomicBool::new(false),
        };
        for op in [
            "resident",
            "health",
            "scan-code",
            "scan-status",
            "iroh-forward",
            "vector-sync",
        ] {
            let r = roundtrip(
                &ctx,
                Channel::Remote,
                serde_json::json!({"op": op, "token": "right-token"}),
            );
            assert!(!r.ok, "remote {op} must be refused even with the bearer token");
        }
    }

    #[test]
    fn line_protocol_rejects_oversized_requests_before_json_parsing() {
        let body = format!(
            "{{\"op\":\"ping\",\"padding\":\"{}\"}}\n",
            "x".repeat(MAX_LINE_REQUEST as usize)
        );
        let mut stream = Duplex {
            input: std::io::Cursor::new(body.into_bytes()),
            output: Vec::new(),
        };
        serve_conn(&mut stream, &no_serve_ctx(), Channel::LocalTrusted);
        let response: Response = serde_json::from_slice(&stream.output).unwrap();
        assert_eq!(response.error.as_deref(), Some("request exceeds the 64 KiB protocol limit"));
    }

    #[test]
    fn iroh_redeem_uses_the_authenticated_connection_peer() {
        let brain = tempfile::tempdir().unwrap();
        let origin = iroh::SecretKey::from_bytes(&[1; 32]).public().into();
        let ticket = grant::invite(brain.path(), &origin, "shared", grant::Mode::Ro, 1, None)
            .unwrap();
        let cfg = crate::config::Config {
            brain_root: brain.path().to_path_buf(),
            ..crate::config::Config::default()
        };
        let ctx = Ctx {
            cfg: Arc::new(cfg),
            serve: None,
            tcp_token: None,
            local_token: None,
            iroh: Mutex::new(None),
            artifact_key: [0; 32],
            shutdown: AtomicBool::new(false),
        };
        let peer = iroh::SecretKey::from_bytes(&[2; 32]).public().to_string();
        let response = handle_iroh(
            &Request {
                op: "redeem".into(),
                network_major: Some(crate::embedding_profile::NETWORK_MAJOR),
                slice: Some(ticket.slice),
                secret: Some(ticket.secret),
                mode: Some(ticket.mode),
                ..Request::default()
            },
            &peer,
            &ctx,
        );
        assert!(response.ok, "{response:?}");
        assert_eq!(response.grant.unwrap().peer, peer);
        assert_eq!(grant::read(brain.path(), "shared").unwrap()[0].peer.as_deref(), Some(peer.as_str()));
    }

    #[test]
    fn the_ranking_mode_and_its_note_cross_the_wire_in_both_directions() {
        // A client asks the SERVING host to rank semantically, because a
        // none-tier host has no vectors and no endpoint of its own.
        let r: Request =
            serde_json::from_str(r#"{"op":"recall","query":"q","limit":3,"semantic":true}"#).unwrap();
        assert_eq!(r.semantic, Some(true));
        assert_eq!(r.hybrid, None);
        // An older client that never learned the flags still parses, and asks
        // for exactly what it always asked for.
        let r: Request = serde_json::from_str(r#"{"op":"recall","query":"q"}"#).unwrap();
        assert_eq!((r.semantic, r.hybrid), (None, None));
        // And an older SERVING host's answer, which carries no note, is not a
        // parse failure on a newer client.
        let resp: Response = serde_json::from_str(r#"{"ok":true}"#).unwrap();
        assert!(resp.note.is_none());
    }

    #[test]
    fn shutdown_is_local_only() {
        let ctx = Ctx {
            cfg: Arc::new(crate::config::Config::default()),
            serve: None,
            tcp_token: Some("t".to_string()),
            local_token: None,
            iroh: Mutex::new(None),
            artifact_key: [0; 32],
            shutdown: AtomicBool::new(false),
        };
        let r = roundtrip(&ctx, Channel::Remote, serde_json::json!({"op": "shutdown", "token": "t"}));
        assert!(!r.ok);
        assert!(r.error.unwrap().contains("local-only"));
        assert!(
            !roundtrip_shutdown(&ctx, Channel::Remote, serde_json::json!({"op": "shutdown", "token": "t"})),
            "a refused shutdown must not stop the daemon either"
        );
    }

    // ---- local channel policy, per transport ----
    //
    // Both local policies are exercised on every platform: the Windows local
    // channel is loopback TCP, and its gate must be provable on the runner
    // that can actually run these tests.

    #[test]
    fn an_os_gated_local_channel_needs_no_token_and_may_shut_down() {
        // The unix socket: its file mode IS the access control.
        let ctx = no_serve_ctx();
        let r = roundtrip(&ctx, Channel::LocalTrusted, serde_json::json!({"op": "ping"}));
        assert!(r.ok, "a unix-socket client presents no credential: {r:?}");
        assert!(roundtrip_shutdown(&ctx, Channel::LocalTrusted, serde_json::json!({"op": "shutdown"})));
    }

    #[test]
    fn a_loopback_local_channel_needs_the_token_and_may_still_shut_down() {
        // Loopback TCP (Windows): any local process can connect, so the
        // daemon's own token gates it — but it is still a LOCAL channel, so
        // `daemon stop` must keep working.
        let ctx = Ctx {
            cfg: Arc::new(crate::config::Config::default()),
            serve: None,
            tcp_token: None,
            local_token: Some("local-token".to_string()),
            iroh: Mutex::new(None),
            artifact_key: [0; 32],
            shutdown: AtomicBool::new(false),
        };
        let r = roundtrip(&ctx, Channel::LocalToken, serde_json::json!({"op": "ping"}));
        assert_eq!(r.error.as_deref(), Some("unauthorized"), "no token: refused");
        let r = roundtrip(&ctx, Channel::LocalToken, serde_json::json!({"op": "ping", "token": "guess"}));
        assert_eq!(r.error.as_deref(), Some("unauthorized"), "wrong token: refused");
        let r = roundtrip(&ctx, Channel::LocalToken, serde_json::json!({"op": "ping", "token": "local-token"}));
        assert!(r.ok, "the published token opens the local channel: {r:?}");
        assert!(
            roundtrip_shutdown(&ctx, Channel::LocalToken, serde_json::json!({"op": "shutdown", "token": "local-token"})),
            "shutdown is a local op on every transport"
        );
    }

    #[test]
    fn a_token_gated_channel_without_a_configured_token_refuses_everything() {
        // An unconfigured credential is never an open door.
        let ctx = no_serve_ctx();
        for chan in [Channel::LocalToken, Channel::Remote] {
            let r = roundtrip(&ctx, chan, serde_json::json!({"op": "ping", "token": "anything"}));
            assert_eq!(r.error.as_deref(), Some("unauthorized"), "{chan:?}");
            let r = roundtrip(&ctx, chan, serde_json::json!({"op": "ping"}));
            assert_eq!(r.error.as_deref(), Some("unauthorized"), "{chan:?}");
        }
    }

    #[test]
    fn this_platforms_local_channel_matches_its_transport() {
        assert_eq!(
            LOCAL_CHANNEL,
            if cfg!(windows) { Channel::LocalToken } else { Channel::LocalTrusted }
        );
        assert!(LOCAL_CHANNEL.allows_shutdown(), "the local channel always accepts shutdown");
    }
}
