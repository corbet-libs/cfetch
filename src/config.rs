//! Configuration. Deep-merged over defaults so a partial file never erases the
//! rest; unknown fields are ignored so old binaries read new configs.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::paths;

/// The outermost legal ring. 0-6 is the whole trust scale; a rule naming
/// anything beyond it is a typo, not a policy, and is refused at load.
pub const MAX_RING: u8 = 6;

/// Ring a path gets when NO rule matches it. It is the curated-knowledge
/// ring: an unclassified brain file is knowledge, never policy and never
/// quarantine. A list that wants a different answer says so with a catch-all
/// rule (empty prefix) of its own.
pub const UNMATCHED_RING: u8 = 3;

/// Path prefixes that can never be indexed, whatever the config says:
/// credentials, session exhaust, and git internals. This is the boundary the
/// whole trust model rests on, so it is compiled in rather than configured —
/// `exclude_prefixes` can only ADD to it.
const HARD_EXCLUDE_PREFIXES: &[&str] = &["mind/secrets/", "logs/"];

/// One entry of the path -> ring taxonomy.
///
/// Matching is deliberately ORDERED and first-match-wins rather than
/// longest-prefix: the file reads top to bottom in the order it fires, so a
/// specific rule is simply placed above a general one and no one has to
/// count characters to predict the outcome.
///
/// A prefix ending in `/` matches the whole subtree. A prefix without a
/// trailing slash matches that exact path only, so `AGENT.md` never captures
/// `AGENT.md.bak`. The empty prefix matches everything — that is how a list
/// declares its own catch-all.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingRule {
    pub prefix: String,
    pub ring: u8,
}

impl RingRule {
    fn matches(&self, rel: &str) -> bool {
        if self.prefix.is_empty() {
            true
        } else if self.prefix.ends_with('/') {
            rel.starts_with(&self.prefix)
        } else {
            rel == self.prefix
        }
    }
}

/// The complete path taxonomy one scan works from: the ordered ring rules and
/// the operator's own exclusions. Everything that needs to know "what ring is
/// this path, and may it be indexed at all" takes one of these, so the
/// mapping exists in exactly one place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingRules {
    pub rules: Vec<RingRule>,
    pub exclude_prefixes: Vec<String>,
}

impl Default for RingRules {
    fn default() -> Self {
        RingRules {
            rules: default_ring_rules(),
            exclude_prefixes: default_exclude_prefixes(),
        }
    }
}

/// The name of the slice every document falls into when no configured slice
/// claims it. Reserved: a configured slice may not take it, because then a
/// path would have two truthful answers to "which slice is this?".
pub const ROOT_SLICE: &str = "root";

/// A named set of brain-relative prefixes.
///
/// Slices are the unit of composition and sharing: any set of markdown files,
/// nestable, down to a single file. Nesting is IMPLICIT — a slice whose
/// prefixes all sit under another's is inside it — so there is no parent
/// field to keep in step with the paths.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SliceRule {
    pub name: String,
    pub prefixes: Vec<String>,
}

impl SliceRule {
    /// The longest prefix of this slice that claims `rel`, if any. Length is
    /// how "innermost" is decided, so it is returned rather than a bool.
    fn claim(&self, rel: &str) -> Option<usize> {
        self.prefixes
            .iter()
            .filter(|p| under_prefix(rel, p))
            .map(|p| p.trim_end_matches('/').len())
            .max()
    }

}

/// The configured slices, validated once so every later lookup is total.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Slices {
    rules: Vec<SliceRule>,
}

impl Slices {
    /// Rejects the shapes that would make a later answer ambiguous rather
    /// than carrying them into the index.
    pub fn new(rules: Vec<SliceRule>) -> anyhow::Result<Slices> {
        let mut seen = std::collections::HashSet::new();
        for r in &rules {
            let name = r.name.trim();
            anyhow::ensure!(name == r.name, "slice names may not have surrounding whitespace");
            crate::grant::validate_slice_name(name)?;
            anyhow::ensure!(
                name != ROOT_SLICE,
                "slice name {ROOT_SLICE:?} is reserved for documents no slice claims"
            );
            anyhow::ensure!(seen.insert(name.to_string()), "two slices are both named {name:?}");
            anyhow::ensure!(!r.prefixes.is_empty(), "slice {name:?} claims no prefixes");
            for p in &r.prefixes {
                anyhow::ensure!(
                    !p.trim().trim_end_matches('/').is_empty(),
                    "slice {name:?} has an empty prefix, which would claim the whole tree"
                );
            }
        }
        Ok(Slices { rules })
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.rules.iter().map(|r| r.name.as_str())
    }

    /// The INNERMOST slice claiming `rel`: the longest matching prefix wins,
    /// and configuration order breaks a tie so the answer never depends on
    /// iteration luck.
    pub fn slice_for(&self, rel: &str) -> &str {
        self.rules
            .iter()
            .filter_map(|r| r.claim(rel).map(|len| (len, r)))
            // `reduce` keeping the incumbent on a tie, NOT `max_by_key`:
            // max_by_key returns the LAST maximum, which would hand an
            // equal-depth tie to the last declaration instead of the first
            // and make the answer depend on where a slice was appended.
            .reduce(|best, next| if next.0 > best.0 { next } else { best })
            .map(|(_, r)| r.name.as_str())
            .unwrap_or(ROOT_SLICE)
    }

    /// The path prefixes a query for `name` should match.
    ///
    /// `None` means "no filter": the root slice is the whole tree. A nested
    /// slice's prefixes sit under its parent's by construction, so filtering
    /// on the named slice's own prefixes already includes everything nested
    /// inside it — there is no ancestry to walk. An unknown name yields an
    /// empty set, so a typo returns nothing rather than everything.
    pub fn prefixes_of(&self, name: &str) -> Option<&[String]> {
        if name == ROOT_SLICE {
            return None;
        }
        Some(
            self.rules
                .iter()
                .find(|r| r.name == name)
                .map(|r| r.prefixes.as_slice())
                .unwrap_or(&[]),
        )
    }

    /// Whether `rel` belongs to the named slice or anything nested beneath
    /// it. This is the response-side guard for operations such as citation
    /// expansion, where the lookup starts from an id instead of a path filter.
    pub fn contains(&self, name: &str, rel: &str) -> bool {
        match self.prefixes_of(name) {
            None => name == ROOT_SLICE,
            Some(prefixes) => prefixes.iter().any(|prefix| under_prefix(rel, prefix)),
        }
    }
}

/// True when `rel` is `prefix` itself or anything beneath it. Written against
/// path components, not raw strings, so `drafts` never swallows `draftsman`;
/// a trailing slash on the prefix is accepted and meaningless.
fn under_prefix(rel: &str, prefix: &str) -> bool {
    let prefix = prefix.trim_end_matches('/');
    if prefix.is_empty() {
        return false; // an empty exclusion would exclude the world by accident
    }
    rel == prefix || rel.strip_prefix(prefix).is_some_and(|r| r.starts_with('/'))
}

/// Git internals ANYWHERE in the tree — a brain may hold repo clones, each
/// with its own `.git`.
fn in_git_dir(rel: &str) -> bool {
    rel == ".git" || rel.starts_with(".git/") || rel.ends_with("/.git") || rel.contains("/.git/")
}

impl RingRules {
    /// Ring for a brain-root-relative path: the first matching rule, else
    /// [`UNMATCHED_RING`]. Pure — no filesystem, no frontmatter (a `ring: N`
    /// key in the file overrides this afterwards, at scan time).
    pub fn ring_for(&self, rel: &str) -> u8 {
        self.rules.iter().find(|r| r.matches(rel)).map(|r| r.ring).unwrap_or(UNMATCHED_RING)
    }

    /// Whether a path must never enter any index or capture record. The hard
    /// boundary first, the operator's own exclusions after it.
    pub fn excluded(&self, rel: &str) -> bool {
        let rel = rel.trim_end_matches('/');
        in_git_dir(rel)
            || HARD_EXCLUDE_PREFIXES.iter().any(|p| under_prefix(rel, p))
            || self.exclude_prefixes.iter().any(|p| under_prefix(rel, p))
    }

    /// Directory form: true when nothing under `rel` can ever be indexed, so
    /// a watcher can skip the whole subtree. It is the SAME predicate as
    /// [`RingRules::excluded`] — the two used to be hand-maintained copies of
    /// one list, which is exactly how a watcher drifts from its indexer.
    pub fn excluded_dir(&self, rel: &str) -> bool {
        self.excluded(rel)
    }
}

/// The shipped taxonomy. Opinionated (it describes a conventional agent brain
/// tree) but not baked in: replacing `ring_rules` in the config replaces it
/// whole.
fn default_ring_rules() -> Vec<RingRule> {
    vec![
        // The tree's own entry points: what every agent reads first.
        RingRule { prefix: "AGENT.md".into(), ring: 1 },
        RingRule { prefix: "README.md".into(), ring: 1 },
        // Exactly the memory index — a topic file named e.g. OLD-MEMORY.md
        // must not inherit ring 1 by suffix accident.
        RingRule { prefix: "mind/memories/MEMORY.md".into(), ring: 1 },
        // Distilled behavioral memories.
        RingRule { prefix: "mind/memories/".into(), ring: 2 },
        RingRule { prefix: "scratch/cfetch-staging/".into(), ring: 5 },
        // Legacy ring-5 candidates. This MUST precede the `todo/` rule:
        // `ring_for` takes the first match, so the general lane would
        // otherwise claim the quarantined one and make candidates recallable.
        RingRule { prefix: "todo/staging/".into(), ring: 5 },
        // Where staging lived before the tree was standardised. Kept as a
        // rule, not merely migrated: without it an unmigrated tree matches
        // nothing on upgrade, falls through to the unmatched ring, and
        // silently promotes quarantined candidates into recall.
        RingRule { prefix: "staging/".into(), ring: 5 },
        // Working state: queues and task notes.
        RingRule { prefix: "todo/".into(), ring: 4 },
        // Ring-5 staging candidates. The LOCATION decides, so a candidate
        // whose frontmatter is stripped or hand-mangled is still never
        // recallable — this exclusion cannot be edited away file by file.
        // Everything else is curated knowledge; see UNMATCHED_RING.
    ]
}

/// Shipped exclusions ON TOP of the hard boundary: repo clones belong to the
/// code index, and the archive is retired knowledge nobody should recall by
/// accident. Both are conventions, so both are configurable.
fn default_exclude_prefixes() -> Vec<String> {
    vec![
        "projects/".into(),
        "knowledge/archive/".into(),
        // Disposable working material. Without this a scratch lane drowns the
        // ring it shares: a real tree measured 12,276 scratch files against 27
        // files of live task state, and every query paid the ratio.
        "scratch/".into(),
        "todo/scratch/".into(),
    ]
}

/// Where a resident entry may be injected. An entry with no scope at all is
/// injected everywhere — which is what every config written before scopes
/// existed keeps meaning.
///
/// `hosts` and `repos` are ORed: an entry listing both lands on any listed
/// host AND in any listed repo, rather than only where the two coincide.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scope {
    /// Machine names this entry belongs to.
    #[serde(default)]
    pub hosts: Vec<String>,
    /// Working-directory names (the repo, as the agent sees it) this entry
    /// belongs to.
    #[serde(default)]
    pub repos: Vec<String>,
    /// Inject regardless of host and repo — an explicit "everywhere" that
    /// survives someone later adding one host to the list.
    ///
    /// It answers "how much", not only "whether": the digest reduces an entry
    /// it cannot carry whole to one index line, and an `always` entry is
    /// exempt — it arrives in full, clipped at the budget rather than
    /// summarized. That is the same claim in both directions ("this content
    /// belongs in every session, entire"), which is why it is this flag and
    /// not a second one; rings 0-1 may make it, ring 2 may not (see the load
    /// check), because unconditional behavior memory is what scoping exists
    /// to prevent.
    #[serde(default)]
    pub always: bool,
}

impl Scope {
    /// No condition named at all.
    pub fn is_unscoped(&self) -> bool {
        self.hosts.is_empty() && self.repos.is_empty()
    }

    /// Whether this entry belongs in a session on `host`, working in `repo`.
    pub fn matches(&self, host: &str, repo: Option<&str>) -> bool {
        if self.always || self.is_unscoped() {
            return true;
        }
        if self.hosts.iter().any(|h| host_matches(host, h)) {
            return true;
        }
        repo.is_some_and(|r| self.repos.iter().any(|want| want == r))
    }
}

/// A `hosts` entry matches the machine's node name exactly, or its first
/// dot-label — so `build-box` still matches a `build-box.example.net` that
/// reports its FQDN.
fn host_matches(host: &str, want: &str) -> bool {
    host == want || host.split('.').next() == Some(want)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidentEntry {
    pub path: PathBuf,
    /// Privilege ring of the file's statements (0 = invariants, 1 = policy,
    /// 2 = distilled behavior). Rings 0-1 may be injected everywhere; ring 2
    /// only under a scope, because behavior memories are SELECTIVE by
    /// definition — injecting the lot of them unconditionally is the
    /// "resident set" this design replaced. Rings 3+ are recall-only and are
    /// refused at load time.
    pub ring: u8,
    /// When this entry reaches a session. Absent = every session.
    #[serde(default)]
    pub scope: Scope,
    /// Share of the digest budget this entry competes for, relative to the
    /// other entries that reached this session. Absent = derived from the
    /// ring, because the ring already states how load-bearing the content is:
    /// an invariant that gets clipped is a worse outcome than a behavior note
    /// that does, and an equal split cannot express that.
    #[serde(default)]
    pub weight: Option<f32>,
}

impl ResidentEntry {
    /// Non-finite or non-positive weights are refused rather than propagated:
    /// a NaN would poison the whole allocation, and a zero would silently
    /// delete an entry the operator asked to inject.
    pub fn budget_weight(&self) -> f32 {
        match self.weight {
            Some(w) if w.is_finite() && w > 0.0 => w,
            _ => match self.ring {
                0 => 4.0,
                1 => 2.0,
                _ => 1.0,
            },
        }
    }
}

/// Ring-6 exhaust capture (PostToolUse/Stop -> the tree's exhaust stream).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    /// On by default: exhaust is the raw material every later promotion is
    /// distilled from; an empty ring 6 starves rings 5 and 2.
    #[serde(default = "default_capture_enabled")]
    pub enabled: bool,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        CaptureConfig { enabled: default_capture_enabled() }
    }
}

fn default_capture_enabled() -> bool {
    true
}

/// Stored width of one vector component.
///
/// `I8` is the only precision admitted by the v1 profile. The legacy enum
/// values remain parseable solely so an upgraded binary can reject an old
/// configuration with a precise major-version error instead of a serde typo.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    F16,
    F32,
    /// Canonical signed 8-bit vector components, with no float trailer.
    #[default]
    I8,
}

impl Precision {
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::F16 => "f16",
            Precision::F32 => "f32",
            Precision::I8 => "i8",
        }
    }

    /// Bytes one component occupies on disk.
    pub fn width(self) -> usize {
        match self {
            Precision::F16 => 2,
            Precision::F32 => 4,
            Precision::I8 => 1,
        }
    }

    /// Bytes each record carries beyond its components. The canonical INT8
    /// codec derives its max-absolute scale per vector and cosine ignores
    /// absolute magnitude, so the scale is deliberately not serialized.
    pub fn trailer(self) -> usize {
        0
    }

    /// Bytes one whole record occupies: components plus any trailer.
    pub fn record_bytes(self, dim: usize) -> usize {
        dim * self.width() + self.trailer()
    }
}

/// The identity of a vector artifact. Every vector
/// in the store and in the local cache belongs to exactly one of these, and a
/// query only ever scores vectors of ITS spec — mixing models, widths or
/// dimensions produces numbers that look like similarity and are not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorSpec {
    pub network_major: u32,
    pub profile_id: String,
    pub model: String,
    pub dim: usize,
    pub precision: Precision,
    /// Text prepended to every DOCUMENT before embedding it. Part of the
    /// artifact identity because it changes the stored vectors: two hosts
    /// configured differently would otherwise write incompatible vectors into
    /// the same shared file, and nothing would ever notice.
    pub doc_prefix: String,
}

impl VectorSpec {
    pub fn vector_encoding(&self) -> String {
        match self.precision {
            Precision::I8 => format!("signed-int8x{}", self.dim),
            Precision::F16 => format!("f16x{}", self.dim),
            Precision::F32 => format!("f32x{}", self.dim),
        }
    }
}

/// Embeddings configuration. The current release transport is an attested
/// OpenAI-compatible endpoint: either a packaged target-native adapter on
/// loopback or an explicitly configured remote deployment. Native artifacts
/// and the NPU-first local selection order are package policy, never
/// user-selectable vector-space parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Explicit remote/base URL; `/embeddings` is appended. Empty selects the
    /// exact admitted package-local plan when this release contains one.
    /// A configured URL always wins and is never a fallback after local CPU.
    /// It must be https or loopback because config is a file agents write.
    #[serde(default)]
    pub endpoint: String,
    #[serde(default = "default_embed_model")]
    pub model: String,
    /// Model name SENT to the endpoint, when the serving layer renames the
    /// model (LM Studio prefixes "text-embedding-", Ollama uses short names).
    /// The `model` field above remains the canonical profile identity and
    /// is what the vector-space consistency check validates against; this
    /// field is purely the wire-level name the endpoint expects.
    #[serde(default)]
    pub endpoint_model: Option<String>,
    /// Exact hostnames/IPs the operator exempts from the private-range
    /// refusal (mesh overlays, lab networks). Never exempts from https.
    #[serde(default)]
    pub allow_hosts: Vec<String>,
    /// NAME of the environment variable holding the API key (e.g.
    /// "OPENAI_API_KEY") — never the key itself: config files travel in
    /// dotfiles and backups, keys must not. Empty = no Authorization header.
    #[serde(default)]
    pub api_key_env: String,
    /// Timeout for ONE interactive embeddings request, in seconds. This is
    /// the tight bound recall rides on; the embed-index batch path scales it
    /// per batched item (see embed::batch_timeout).
    #[serde(default = "default_embed_timeout_secs")]
    pub timeout_secs: u64,
    /// Vector width carried on the wire and in the store. This field remains
    /// serializable so old configuration fails with a precise migration
    /// error; network major 1 admits exactly EmbeddingGemma's full 768.
    #[serde(default = "default_embed_dimensions")]
    pub dimensions: usize,
    /// Text prepended to every DOCUMENT before embedding it.
    ///
    /// It is frozen to the v1 EmbeddingGemma retrieval prompt. The field is
    /// retained for migration diagnostics, but changing it requires a new
    /// network major and full re-embedding.
    #[serde(default = "default_document_prefix")]
    pub document_prefix: String,
    /// Text prepended to a QUERY before embedding it, and to nothing else.
    ///
    /// It is frozen to the v1 EmbeddingGemma retrieval prompt. The field is
    /// retained for migration diagnostics, but changing it requires a new
    /// network major and full re-embedding.
    #[serde(default = "default_query_prefix")]
    pub query_prefix: String,
    /// Stored component width (see [`Precision`]).
    #[serde(default)]
    pub precision: Precision,
}

impl EmbeddingsConfig {
    /// The artifact identity this configuration asks for.
    pub fn spec(&self) -> VectorSpec {
        // Production configuration is validated against the canonical profile
        // before use. Keeping deliberately non-canonical specs out of network
        // major 1 also lets migration/unit callers exercise the generic store
        // without accidentally claiming shareable v1 artifacts.
        let canonical = self.model == crate::embedding_profile::MODEL;
        VectorSpec {
            network_major: if canonical {
                crate::embedding_profile::NETWORK_MAJOR
            } else {
                0
            },
            profile_id: if canonical {
                crate::embedding_profile::PROFILE_ID.to_string()
            } else {
                "unmanaged-embedding-profile".to_string()
            },
            model: self.model.clone(),
            dim: self.dimensions,
            precision: self.precision,
            doc_prefix: self.document_prefix.clone(),
        }
    }

    pub fn validate_profile(&self) -> anyhow::Result<()> {
        crate::embedding_profile::validate(self)
    }
}

/// Cross-encoder reranking of an already-retrieved candidate list.
///
/// Retrieval (BM25, vectors, their fusion) scores a query against a document
/// it has never seen beside it. A cross-encoder reads the two TOGETHER and is
/// far better at it — but it costs a forward pass per candidate, so it can
/// only ever run over a shortlist, never over the corpus. That is exactly the
/// shape here: recall proposes, rerank reorders.
///
/// Off by default. It is a second endpoint and a second dependency, and the
/// lexical answer must never require one.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RerankConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Base URL; `/rerank` is appended. Same SSRF guard as embeddings.
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub model: String,
    /// Exact hostnames/IPs exempted from the private-range refusal.
    #[serde(default)]
    pub allow_hosts: Vec<String>,
    /// NAME of the environment variable holding the API key, never the key.
    #[serde(default)]
    pub api_key_env: String,
    #[serde(default = "default_rerank_timeout_secs")]
    pub timeout_secs: u64,
    /// How many top hits are sent to the cross-encoder. Everything past this
    /// keeps its retrieval order and follows the reranked head, so raising it
    /// buys quality at a linear cost in endpoint work.
    #[serde(default = "default_rerank_candidates")]
    pub candidates: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        RerankConfig {
            enabled: false,
            endpoint: String::new(),
            model: String::new(),
            allow_hosts: Vec::new(),
            api_key_env: String::new(),
            timeout_secs: default_rerank_timeout_secs(),
            candidates: default_rerank_candidates(),
        }
    }
}

impl Default for EmbeddingsConfig {
    fn default() -> Self {
        EmbeddingsConfig {
            enabled: false,
            endpoint: String::new(),
            model: crate::embedding_profile::MODEL.to_string(),
            endpoint_model: None,
            allow_hosts: Vec::new(),
            api_key_env: String::new(),
            timeout_secs: default_embed_timeout_secs(),
            dimensions: default_embed_dimensions(),
            document_prefix: crate::embedding_profile::DOCUMENT_PREFIX.to_string(),
            query_prefix: crate::embedding_profile::QUERY_PREFIX.to_string(),
            precision: Precision::default(),
        }
    }
}

/// Continuous second-brain maintenance. The daemon only starts inference when
/// an endpoint and model are both configured; `enabled` defaults on so a
/// completed configuration becomes seamless without another feature switch.
/// Missing endpoint/model remains an explicit "not configured" state, never a
/// silent network fallback.
pub const MAX_MAINTENANCE_CANDIDATES: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaintenanceConfig {
    #[serde(default = "default_maintenance_enabled")]
    pub enabled: bool,
    /// OpenAI-compatible base URL; `/chat/completions` is appended. Subject to
    /// the same redirect and SSRF policy as embedding/reranking endpoints.
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub model: String,
    /// Optional distinct reviewer model on the same endpoint. Absent means a
    /// fresh, isolated pass through `model`; it never means self-approval in
    /// the proposal call.
    #[serde(default)]
    pub review_model: Option<String>,
    #[serde(default)]
    pub allow_hosts: Vec<String>,
    /// Environment variable NAME holding the bearer key, never the key.
    #[serde(default)]
    pub api_key_env: String,
    #[serde(default = "default_maintenance_timeout_secs")]
    pub timeout_secs: u64,
    /// Quiet period after a candidate/tree change before background work.
    #[serde(default = "default_maintenance_debounce_secs")]
    pub debounce_secs: u64,
    /// Per-cycle bound; exceptions do not prevent later candidates in the
    /// same cycle from being processed.
    #[serde(default = "default_maintenance_max_candidates")]
    pub max_candidates: usize,
}

impl MaintenanceConfig {
    pub fn configured(&self) -> bool {
        !self.endpoint.trim().is_empty() && !self.model.trim().is_empty()
    }
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            enabled: default_maintenance_enabled(),
            endpoint: String::new(),
            model: String::new(),
            review_model: None,
            allow_hosts: Vec::new(),
            api_key_env: String::new(),
            timeout_secs: default_maintenance_timeout_secs(),
            debounce_secs: default_maintenance_debounce_secs(),
            max_candidates: default_maintenance_max_candidates(),
        }
    }
}

fn default_maintenance_enabled() -> bool {
    true
}

fn default_maintenance_timeout_secs() -> u64 {
    120
}

fn default_maintenance_debounce_secs() -> u64 {
    30
}

fn default_maintenance_max_candidates() -> usize {
    4
}

fn default_embed_timeout_secs() -> u64 {
    10
}

fn default_embed_model() -> String {
    crate::embedding_profile::MODEL.to_string()
}

fn default_embed_dimensions() -> usize {
    crate::embedding_profile::DIMENSIONS
}

fn default_document_prefix() -> String {
    crate::embedding_profile::DOCUMENT_PREFIX.to_string()
}

fn default_query_prefix() -> String {
    crate::embedding_profile::QUERY_PREFIX.to_string()
}

fn default_rerank_timeout_secs() -> u64 {
    20
}

fn default_rerank_candidates() -> usize {
    40
}

/// The governance loop: Stop-side reminder producers plus instruction-decay
/// cadence re-injection, all delivered at the next UserPromptSubmit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceConfig {
    #[serde(default = "default_governance_enabled")]
    pub enabled: bool,
    /// Token ceiling for ONE brain file before the write site says so. A file
    /// over it is not an error — it is a recurring bill, paid by every session
    /// that loads it — so the response is one factual line naming the size and
    /// the remedy, never a block. 0 disables the check entirely.
    #[serde(default = "default_state_file_budget_tokens")]
    pub state_file_budget_tokens: u64,
    /// Every N-th post-tool event re-queues the top ring-0 rules (compliance
    /// decays measurably with generated output; see DESIGN's instruction-decay
    /// study). Zero switches the cadence off while keeping the Stop producers.
    #[serde(default = "default_reinject_every")]
    pub reinject_every: u32,
}

impl Default for GovernanceConfig {
    fn default() -> Self {
        GovernanceConfig {
            state_file_budget_tokens: default_state_file_budget_tokens(),
            enabled: default_governance_enabled(),
            reinject_every: default_reinject_every(),
        }
    }
}

fn default_governance_enabled() -> bool {
    true
}

fn default_reinject_every() -> u32 {
    25
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallConfig {
    /// Reciprocal-rank-fusion constant for `recall --hybrid`. Small K weights
    /// top ranks heavily; K=2 measurably beat the textbook K=60 on a 22k-unit
    /// corpus (MRR 0.845 vs 0.627), so 2 is the default.
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f64,
    /// The precision gate over a purely lexical answer.
    #[serde(default)]
    pub gate: GateConfig,
}

impl Default for RecallConfig {
    fn default() -> Self {
        RecallConfig { rrf_k: default_rrf_k(), gate: GateConfig::default() }
    }
}

fn default_rrf_k() -> f64 {
    2.0
}

/// The precision gate: an admission test for hits that no other stage judges.
///
/// Retrieval OR-joins the query's terms on purpose, so a six-word question
/// still finds the block that phrased it differently. The price is that a
/// block sharing ONE ordinary word with the query is retrieved, and with no
/// semantic stage configured it is also returned. This is the knob that says
/// no to it.
///
/// Off by default. A gate that drops a hit the operator expected is worse
/// than the weak hit it was meant to suppress, so switching it on is a
/// deliberate act with a corpus in front of you.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateConfig {
    /// How many DISTINCT non-stopword query terms a hit must carry to be
    /// admitted. Clamped to the number of such terms the query actually has,
    /// so a one-word query is never gated against itself; 0 or 1 therefore
    /// means no gate at all, which is the default.
    #[serde(default = "default_gate_min_terms")]
    pub min_terms: usize,
}

impl Default for GateConfig {
    fn default() -> Self {
        GateConfig { min_terms: default_gate_min_terms() }
    }
}

fn default_gate_min_terms() -> usize {
    1
}

/// Serving mode: this daemon owns its index lifecycle (watcher, generations,
/// drain barrier) and answers recall/find/expand for local clients — and for
/// remote ones when `bind` is set.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServeConfig {
    #[serde(default)]
    pub enabled: bool,
    /// TCP listen address (e.g. "0.0.0.0:9737"). Absent = the local control
    /// channel only (see `crate::ipc`).
    #[serde(default)]
    pub bind: Option<String>,
    /// Host id stamped on every response. Defaults to the machine hostname.
    #[serde(default)]
    pub origin: Option<String>,
    /// Bearer token file gating the TCP listener; must be mode 0600.
    /// Required when `bind` is set.
    #[serde(default)]
    pub token_file: Option<PathBuf>,
}

/// Remote serving host a none-tier client queries instead of any local index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientServingConfig {
    /// "host:port" of the serving daemon's TCP listener.
    pub addr: String,
    /// File holding the bearer token for that listener.
    pub token_file: PathBuf,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientConfig {
    /// When set, recall/find/expand route to this serving host and the host
    /// opens NO local index at all (none-tier by config). Unreachable is an
    /// explicit error — never a silent fallback to stale local data.
    #[serde(default)]
    pub serving: Option<ClientServingConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "paths::default_brain_root")]
    pub brain_root: PathBuf,
    /// Files injected verbatim (budget-clipped) at session start, in order.
    /// Paths are relative to brain_root unless absolute.
    #[serde(default)]
    pub resident: Vec<ResidentEntry>,
    /// Roots for the code index (`cfetch find`). Empty means the default:
    /// `<brain_root>/projects/github` — where the house repos live.
    #[serde(default)]
    pub code_roots: Vec<PathBuf>,
    /// Hard cap on the injected digest, in characters.
    #[serde(default = "default_budget_chars")]
    pub budget_chars: usize,
    /// Writer-side byte cap on THIS host's exhaust stream before it rotates
    /// (`<brain_root>/logs/cfetch/exhaust-<host>.jsonl`). Two rotated
    /// generations are kept, so the footprint per host is at most 3x this.
    #[serde(default = "default_exhaust_max_bytes")]
    pub exhaust_max_bytes: u64,
    /// Same, for this host's ledger stream.
    #[serde(default = "default_ledger_max_bytes")]
    pub ledger_max_bytes: u64,
    /// Ring-6 exhaust capture.
    #[serde(default)]
    pub capture: CaptureConfig,
    #[serde(default)]
    pub embeddings: EmbeddingsConfig,
    #[serde(default)]
    pub rerank: RerankConfig,
    /// Automatic, evidence-grounded Markdown maintenance.
    #[serde(default)]
    pub maintenance: MaintenanceConfig,
    /// Named prefix sets, the unit of composition and sharing. Empty (the
    /// default) means one implicit slice covering everything, which is
    /// exactly today's behavior.
    #[serde(default)]
    pub slices: Vec<SliceRule>,
    #[serde(default)]
    pub recall: RecallConfig,
    /// Governance loop (reminder queue + cadence rule refresh).
    #[serde(default)]
    pub governance: GovernanceConfig,
    /// Serving mode (storage host answering queries).
    #[serde(default)]
    pub serve: ServeConfig,
    /// Client routing (none-tier host querying a serving host).
    #[serde(default)]
    pub client: ClientConfig,
    /// The path -> ring taxonomy, in order; the FIRST matching rule wins.
    /// Replacing this list replaces the shipped one whole.
    #[serde(default = "default_ring_rules")]
    pub ring_rules: Vec<RingRule>,
    /// Extra path prefixes to keep out of the index, ON TOP of the hard
    /// boundary (secrets, logs, `.git`), which no config can lift.
    #[serde(default = "default_exclude_prefixes")]
    pub exclude_prefixes: Vec<String>,
}

pub fn default_budget_chars() -> usize {
    6000
}

/// 4000 tokens (~14k chars). Deliberately well above `default_budget_chars`,
/// which caps the WHOLE injected digest: this is not "too big to inject" but
/// "big enough that loading it is a decision", which is the point at which
/// splitting starts to pay.
fn default_state_file_budget_tokens() -> u64 {
    4000
}

/// 32 MiB of ring-6 exhaust per host before rotation — months of capture at
/// typical line sizes, and small next to any brain tree.
fn default_exhaust_max_bytes() -> u64 {
    32 * 1024 * 1024
}

/// The ledger books a few lines per turn, so a quarter of the exhaust cap
/// covers far more history than the audit window ever asks for.
fn default_ledger_max_bytes() -> u64 {
    8 * 1024 * 1024
}

impl Default for Config {
    fn default() -> Self {
        Config {
            brain_root: paths::default_brain_root(),
            resident: vec![ResidentEntry {
                path: PathBuf::from("AGENT.md"),
                ring: 1,
                scope: Scope::default(),
                weight: None,
            }],
            code_roots: Vec::new(),
            budget_chars: default_budget_chars(),
            exhaust_max_bytes: default_exhaust_max_bytes(),
            ledger_max_bytes: default_ledger_max_bytes(),
            capture: CaptureConfig::default(),
            embeddings: EmbeddingsConfig::default(),
            rerank: RerankConfig::default(),
            maintenance: MaintenanceConfig::default(),
            slices: Vec::new(),
            recall: RecallConfig::default(),
            governance: GovernanceConfig::default(),
            serve: ServeConfig::default(),
            client: ClientConfig::default(),
            ring_rules: default_ring_rules(),
            exclude_prefixes: default_exclude_prefixes(),
        }
    }
}

/// Config keys that only the machine layer (`CFETCH_CONFIG` or the
/// machine-local file) may set. The tree's `.cfetch/config.json` lives inside
/// the brain tree — the thing agents write and other machines clone — so it
/// describes CONTENT (rings, slices, resident files, budgets); it must never
/// choose where the brain itself lives, where requests egress, or how this
/// machine serves.
const MACHINE_OWNED_KEYS: &[&str] =
    &["brain_root", "embeddings", "rerank", "maintenance", "serve", "client"];

impl Config {
    /// Loads the configuration in two layers.
    ///
    /// Search order: an explicit `CFETCH_CONFIG` (fully trusted, operator
    /// hand-written), otherwise the TREE's `.cfetch/config.json` for content
    /// keys overlaid by the MACHINE-LOCAL file.
    ///
    /// The tree still comes first for what it owns — ring rules, slices and
    /// resident entries describe content every host sees identically — but a
    /// key in [`MACHINE_OWNED_KEYS`] found in the tree is a hard error, not a
    /// silently honored value: a cloned or agent-edited brain must not get to
    /// pick this machine's embedding endpoint (and thereby which environment
    /// variable becomes a bearer token), its serving topology, or the root.
    ///
    /// `brain_root` itself never comes from any file: it is always
    /// `CFETCH_BRAIN` or the default location, which is also what makes
    /// finding the tree before reading its config non-circular.
    pub fn load() -> anyhow::Result<Config> {
        if let Some(explicit) = std::env::var_os("CFETCH_CONFIG") {
            return Config::load_from(std::path::Path::new(&explicit));
        }
        let tree_path = paths::tree_config_path(&paths::default_brain_root());
        let machine_path = paths::config_path();
        let tree = Config::read_raw(&tree_path)?;
        let machine = Config::read_raw(&machine_path)?;
        Config::load_layered(
            tree.as_ref().map(|raw| (tree_path.as_path(), raw.as_str())),
            machine.as_ref().map(|raw| (machine_path.as_path(), raw.as_str())),
        )
    }

    fn read_raw(path: &std::path::Path) -> anyhow::Result<Option<String>> {
        match std::fs::read_to_string(path) {
            Ok(s) => Ok(Some(s)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(anyhow::anyhow!("read {}: {e}", path.display())),
        }
    }

    /// Two-layer load. Each argument is `(label, raw json)`; either may be
    /// absent. Tree keys land first, machine keys overlay them; machine-owned
    /// keys are refused outright in the tree layer.
    fn load_layered(
        tree: Option<(&std::path::Path, &str)>,
        machine: Option<(&std::path::Path, &str)>,
    ) -> anyhow::Result<Config> {
        let mut merged = serde_json::json!({});
        if let Some((tpath, raw)) = tree {
            let v: serde_json::Value = serde_json::from_str(raw)
                .map_err(|e| anyhow::anyhow!("parse {}: {e}", tpath.display()))?;
            for key in MACHINE_OWNED_KEYS {
                if v.get(*key).map(|x| !x.is_null()).unwrap_or(false) {
                    anyhow::bail!(
                        "config {} sets \"{key}\": the tree config lives inside the brain, which \
                         agents write and other machines clone — endpoints, serving topology and \
                         the root belong in the machine-local config ({}) or in CFETCH_CONFIG",
                        tpath.display(),
                        paths::config_path().display(),
                    );
                }
            }
            Config::reject_absolute_tree_paths(&v, tpath)?;
            merged = v;
        }
        if let Some((mpath, raw)) = machine {
            let m: serde_json::Value = serde_json::from_str(raw)
                .map_err(|e| anyhow::anyhow!("parse {}: {e}", mpath.display()))?;
            if let Some(map) = m.as_object() {
                for (k, val) in map {
                    if k == "brain_root" {
                        eprintln!(
                            "[cfetch] ignoring brain_root set in {} — the root comes from \
                             CFETCH_BRAIN or the default location, never from a config file",
                            mpath.display()
                        );
                        continue;
                    }
                    merged[k.as_str()] = val.clone();
                }
            }
        }
        Config::parse_and_validate(merged).map(|mut cfg| {
            cfg.brain_root = paths::default_brain_root();
            cfg
        })
    }

    /// Tree-layer paths may only point INSIDE the tree: a resident entry or
    /// code root that is absolute (or carries `..`) would let a cloned brain
    /// aim this host's reads and injections at files outside it. The machine
    /// layer keeps the absolute-path escape hatch; it is operator state.
    fn reject_absolute_tree_paths(
        v: &serde_json::Value,
        label: &std::path::Path,
    ) -> anyhow::Result<()> {
        // `Path::is_absolute` alone is not enough: on Windows a leading `/`
        // has no drive and parses as relative, and `C:` drive-relative forms
        // carry a colon no tree-relative path ever needs.
        fn escapes_tree(p: &str) -> bool {
            let path = std::path::Path::new(p);
            path.is_absolute()
                || p.starts_with('/')
                || p.starts_with('\\')
                || p.contains(':')
                || path.components().any(|c| matches!(c, std::path::Component::ParentDir))
        }
        if let Some(entries) = v.get("resident").and_then(|r| r.as_array()) {
            for entry in entries {
                let Some(path) = entry.get("path").and_then(|p| p.as_str()) else { continue };
                if escapes_tree(path) {
                    anyhow::bail!(
                        "resident entry {path:?} in {} is not tree-relative: tree-layer paths \
                         must stay inside the brain (absolute paths belong in the machine-local \
                         config)",
                        label.display(),
                    );
                }
            }
        }
        if let Some(roots) = v.get("code_roots").and_then(|r| r.as_array()) {
            for root in roots {
                let Some(path) = root.as_str() else { continue };
                if escapes_tree(path) {
                    anyhow::bail!(
                        "code root {path:?} in {} is not tree-relative: tree-layer paths must \
                         stay inside the brain (absolute paths belong in the machine-local \
                         config)",
                        label.display(),
                    );
                }
            }
        }
        Ok(())
    }

    fn parse_and_validate(v: serde_json::Value) -> anyhow::Result<Config> {
        let cfg: Config = serde_json::from_value(v)
            .map_err(|e| anyhow::anyhow!("config: {e}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Loads one fully trusted config file (`CFETCH_CONFIG`, tests, tooling).
    /// Machine-owned keys are honored here; `brain_root` is still never taken
    /// from the file — the invariant [`Config::load`] documents holds on this
    /// path too, and a file that tries is warned about, not obeyed.
    pub fn load_from(path: &std::path::Path) -> anyhow::Result<Config> {
        let raw = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Config::default());
            }
            Err(e) => return Err(anyhow::anyhow!("read {}: {e}", path.display())),
        };
        let mut v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("parse {}: {e}", path.display()))?;
        let claimed_root = v.as_object_mut().and_then(|obj| obj.remove("brain_root"));
        if claimed_root.is_some() {
            eprintln!(
                "[cfetch] ignoring brain_root set in {} — the root comes from CFETCH_BRAIN \
                 or the default location, never from a config file",
                path.display()
            );
        }
        Config::parse_and_validate(v).map(|mut cfg| {
            cfg.brain_root = paths::default_brain_root();
            cfg
        })
    }

    /// The shared post-parse checks; every load path funnels through here.
    fn validate(&self) -> anyhow::Result<()> {
        // An explicitly empty `resident` list means "inject nothing" — the
        // default (AGENT.md) applies only when no config file exists at all.
        // On hosts where the harness already auto-loads the ring files,
        // injecting them again would double-pay the context budget.
        for r in &self.resident {
            // The bar the whole trust model rests on — "never indexed,
            // recalled or injected, at ANY configuration" — must hold at
            // this layer too. A resident entry is the one mechanism that
            // injects WITHOUT recall being asked, and the tree config it
            // lives in is written by agents and cloned across machines:
            // an entry pointing into the hard-excluded prefixes (or the
            // operator's own exclusions) would smuggle exactly what those
            // boundaries exist to keep out into every session. Refused at
            // load, loudly, whatever the layer it arrived from.
            let rel = r.path.to_string_lossy().replace('\\', "/");
            if HARD_EXCLUDE_PREFIXES.iter().any(|p| under_prefix(&rel, p)) {
                anyhow::bail!(
                    "resident entry {} is under a hard-excluded prefix: the secrets/exhaust bar — never indexed, recalled or injected, at ANY configuration — holds for resident entries too",
                    r.path.display()
                );
            }
            if self.exclude_prefixes.iter().any(|p| under_prefix(&rel, p.as_str())) {
                anyhow::bail!(
                    "resident entry {} is under exclude_prefixes: a file cannot be excluded from the index and injected into every session at once",
                    r.path.display()
                );
            }
            match r.ring {
                0 | 1 => {}
                // Ring 2 is injectable, but only as policy: a scope, and not
                // an `always` that would smuggle back the unconditional set.
                2 if !r.scope.is_unscoped() && !r.scope.always => {}
                2 => anyhow::bail!(
                    "resident entry {} is ring 2: behavior memories are injected selectively — give it a scope (hosts/repos), or leave it to recall",
                    r.path.display()
                ),
                n => anyhow::bail!(
                    "resident entry {} has ring {n}; rings 0-1 may be resident anywhere, ring 2 only under a scope, rings 3+ never",
                    r.path.display()
                ),
            }
        }
        if self.serve.enabled && self.client.serving.is_some() {
            anyhow::bail!(
                "serve.enabled and client.serving are mutually exclusive: serving needs a local \
                 index, a none-tier client must open none"
            );
        }
        for r in &self.ring_rules {
            if r.ring > MAX_RING {
                anyhow::bail!(
                    "ring rule {:?} names ring {}; rings run 0-{MAX_RING}",
                    r.prefix,
                    r.ring
                );
            }
        }
        if self.serve.bind.is_some() && self.serve.token_file.is_none() {
            anyhow::bail!(
                "serve.bind requires serve.token_file: the TCP listener is bearer-token gated, \
                 an open listener is unconfigurable"
            );
        }
        if self.embeddings.enabled {
            self.embeddings.validate_profile()?;
        }
        let maintenance_endpoint = self.maintenance.endpoint.trim();
        let maintenance_model = self.maintenance.model.trim();
        anyhow::ensure!(
            maintenance_endpoint.is_empty() == maintenance_model.is_empty(),
            "maintenance.endpoint and maintenance.model must be configured together"
        );
        if let Some(review_model) = self.maintenance.review_model.as_deref() {
            anyhow::ensure!(!review_model.trim().is_empty(), "maintenance.review_model may not be empty");
        }
        anyhow::ensure!(self.maintenance.timeout_secs > 0, "maintenance.timeout_secs must be at least 1");
        anyhow::ensure!(self.maintenance.debounce_secs > 0, "maintenance.debounce_secs must be at least 1");
        anyhow::ensure!(
            (1..=MAX_MAINTENANCE_CANDIDATES).contains(&self.maintenance.max_candidates),
            "maintenance.max_candidates must be between 1 and {MAX_MAINTENANCE_CANDIDATES}"
        );
        Ok(())
    }

    /// The taxonomy this config describes, as one value to hand the indexer.
    pub fn rings(&self) -> RingRules {
        RingRules {
            rules: self.ring_rules.clone(),
            exclude_prefixes: self.exclude_prefixes.clone(),
        }
    }

    /// The validated slice model. An invalid one is an error at the point of
    /// use rather than a silently ignored config block.
    pub fn slice_model(&self) -> anyhow::Result<Slices> {
        Slices::new(self.slices.clone())
    }

    pub fn resolve(&self, p: &std::path::Path) -> PathBuf {
        if p.is_absolute() { p.to_path_buf() } else { self.brain_root.join(p) }
    }

    pub fn effective_code_roots(&self) -> Vec<PathBuf> {
        if self.code_roots.is_empty() {
            vec![self.brain_root.join("projects/github")]
        } else {
            self.code_roots.iter().map(|p| self.resolve(p)).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_yields_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::load_from(&dir.path().join("absent.json")).unwrap();
        assert_eq!(cfg.resident.len(), 1);
        assert_eq!(cfg.resident[0].path, PathBuf::from("AGENT.md"));
    }

    #[test]
    fn explicit_empty_resident_stays_empty() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(&p, r#"{"resident": []}"#).unwrap();
        let cfg = Config::load_from(&p).unwrap();
        assert!(cfg.resident.is_empty(), "explicit [] must mean inject nothing");
    }

    #[test]
    fn resident_ring_above_two_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(&p, r#"{"resident": [{"path": "x.md", "ring": 3}]}"#).unwrap();
        assert!(Config::load_from(&p).is_err());
        std::fs::write(&p, r#"{"resident": [{"path": "x.md", "ring": 5}]}"#).unwrap();
        assert!(Config::load_from(&p).is_err());
    }

    #[test]
    fn ring_two_is_resident_only_under_a_scope() {
        // The selective-injection contract, enforced at load: a behavior
        // memory reaches the sessions it is FOR, or it stays in recall.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");

        std::fs::write(
            &p,
            r#"{"resident": [{"path": "b.md", "ring": 2, "scope": {"repos": ["widget"]}}]}"#,
        )
        .unwrap();
        assert!(Config::load_from(&p).is_ok(), "a scoped ring-2 entry is the point");

        std::fs::write(&p, r#"{"resident": [{"path": "b.md", "ring": 2}]}"#).unwrap();
        let err = Config::load_from(&p).unwrap_err().to_string();
        assert!(err.contains("selectively"), "the message must say why: {err}");

        std::fs::write(
            &p,
            r#"{"resident": [{"path": "b.md", "ring": 2, "scope": {"always": true}}]}"#,
        )
        .unwrap();
        assert!(
            Config::load_from(&p).is_err(),
            "`always` must not smuggle an unconditional ring-2 set back in"
        );
    }

    #[test]
    fn stream_caps_default_and_parse() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::load_from(&dir.path().join("absent.json")).unwrap();
        assert_eq!(cfg.exhaust_max_bytes, 32 * 1024 * 1024);
        assert_eq!(cfg.ledger_max_bytes, 8 * 1024 * 1024);
        let p = dir.path().join("config.json");
        std::fs::write(&p, r#"{"exhaust_max_bytes": 1048576}"#).unwrap();
        let cfg = Config::load_from(&p).unwrap();
        assert_eq!(cfg.exhaust_max_bytes, 1048576);
        assert_eq!(cfg.ledger_max_bytes, 8 * 1024 * 1024, "a partial file keeps the other default");
        // A config written for the SQLite era still loads: the retired
        // ledger_max_sessions key is simply ignored.
        std::fs::write(&p, r#"{"ledger_max_sessions": 200, "resident": []}"#).unwrap();
        assert!(Config::load_from(&p).is_ok());
    }

    #[test]
    fn capture_defaults_on_and_can_be_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        assert!(Config::load_from(&p).unwrap().capture.enabled, "default: capture on");
        std::fs::write(&p, r#"{"resident": []}"#).unwrap();
        assert!(Config::load_from(&p).unwrap().capture.enabled, "partial file: capture on");
        std::fs::write(&p, r#"{"capture": {"enabled": false}}"#).unwrap();
        assert!(!Config::load_from(&p).unwrap().capture.enabled);
    }

    #[test]
    fn maintenance_defaults_ready_for_configuration_but_never_invents_a_model_route() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        let cfg = Config::load_from(&p).unwrap();
        assert!(cfg.maintenance.enabled);
        assert!(!cfg.maintenance.configured());
        assert_eq!(cfg.maintenance.timeout_secs, 120);
        assert_eq!(cfg.maintenance.debounce_secs, 30);
        assert_eq!(cfg.maintenance.max_candidates, 4);

        std::fs::write(
            &p,
            r#"{"maintenance":{"endpoint":"http://127.0.0.1:8080/v1","model":"maintainer","review_model":"reviewer","max_candidates":7}}"#,
        )
        .unwrap();
        let cfg = Config::load_from(&p).unwrap();
        assert!(cfg.maintenance.configured());
        assert_eq!(cfg.maintenance.review_model.as_deref(), Some("reviewer"));
        assert_eq!(cfg.maintenance.max_candidates, 7);
    }

    #[test]
    fn partial_or_unbounded_maintenance_configuration_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        for raw in [
            r#"{"maintenance":{"endpoint":"https://example.invalid/v1"}}"#,
            r#"{"maintenance":{"model":"maintainer"}}"#,
            r#"{"maintenance":{"timeout_secs":0}}"#,
            r#"{"maintenance":{"debounce_secs":0}}"#,
            r#"{"maintenance":{"max_candidates":0}}"#,
            r#"{"maintenance":{"max_candidates":33}}"#,
            r#"{"maintenance":{"review_model":""}}"#,
        ] {
            std::fs::write(&p, raw).unwrap();
            assert!(Config::load_from(&p).is_err(), "must refuse {raw}");
        }
    }

    #[test]
    fn embeddings_default_disabled_and_rrf_k_default_2() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::load_from(&dir.path().join("absent.json")).unwrap();
        assert!(!cfg.embeddings.enabled);
        assert!(cfg.embeddings.endpoint.is_empty());
        assert_eq!(cfg.embeddings.model, crate::embedding_profile::MODEL);
        assert_eq!(cfg.recall.rrf_k, 2.0);
    }

    #[test]
    fn embeddings_api_key_env_and_timeout_parse_with_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::load_from(&dir.path().join("absent.json")).unwrap();
        assert!(cfg.embeddings.api_key_env.is_empty(), "default: no auth");
        assert_eq!(cfg.embeddings.timeout_secs, 10, "default: tight interactive bound");
        assert_eq!(EmbeddingsConfig::default().timeout_secs, 10, "Default impl must agree with serde");
        let p = dir.path().join("config.json");
        std::fs::write(
            &p,
            r#"{"embeddings": {"api_key_env": "MY_EMBED_KEY", "timeout_secs": 30}}"#,
        )
        .unwrap();
        let cfg = Config::load_from(&p).unwrap();
        assert_eq!(cfg.embeddings.api_key_env, "MY_EMBED_KEY");
        assert_eq!(cfg.embeddings.timeout_secs, 30);
        // partial block keeps the timeout default
        std::fs::write(&p, r#"{"embeddings": {"enabled": true}}"#).unwrap();
        assert_eq!(Config::load_from(&p).unwrap().embeddings.timeout_secs, 10);
    }

    #[test]
    fn embeddings_dimensions_and_precision_defaults_and_parse() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::load_from(&dir.path().join("absent.json")).unwrap();
        assert_eq!(cfg.embeddings.dimensions, 768, "v1 keeps EmbeddingGemma's full width");
        assert_eq!(cfg.embeddings.precision, Precision::I8, "v1 stores signed INT8 only");
        assert_eq!(EmbeddingsConfig::default().dimensions, 768, "Default impl must agree with serde");
        assert_eq!(EmbeddingsConfig::default().precision, Precision::I8);

        let p = dir.path().join("config.json");
        std::fs::write(
            &p,
            r#"{"embeddings": {"enabled": true, "dimensions": 512, "precision": "f32"}}"#,
        )
        .unwrap();
        let err = Config::load_from(&p).unwrap_err().to_string();
        assert!(err.contains("network major") && err.contains("re-embedding"), "{err}");

        // An unknown width is a typo, not a policy: refused at load.
        std::fs::write(&p, r#"{"embeddings": {"precision": "bfloat16"}}"#).unwrap();
        assert!(Config::load_from(&p).is_err(), "unknown precision must be loud");
    }

    #[test]
    fn disabled_stale_embedding_profile_does_not_block_config_load() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        let stale = r#"{"embeddings":{"enabled":false,"model":"retired/model","dimensions":512,"precision":"f32","document_prefix":"old document","query_prefix":"old query"}}"#;
        std::fs::write(&p, stale).unwrap();
        let cfg = Config::load_from(&p).unwrap();
        assert!(!cfg.embeddings.enabled);
        assert_eq!(cfg.embeddings.model, "retired/model");

        let enabled = stale.replacen("\"enabled\":false", "\"enabled\":true", 1);
        std::fs::write(&p, enabled).unwrap();
        let error = Config::load_from(&p).unwrap_err().to_string();
        assert!(
            error.contains("network major") && error.contains("re-embedding"),
            "{error}"
        );
    }

    #[test]
    fn vector_spec_carries_model_dim_and_precision() {
        let spec = EmbeddingsConfig {
            model: "nomic".into(),
            dimensions: 256,
            precision: Precision::F32,
            ..EmbeddingsConfig::default()
        }
        .spec();
        assert_eq!(spec.model, "nomic");
        assert_eq!(spec.network_major, 0);
        assert_eq!(spec.profile_id, "unmanaged-embedding-profile");
        assert_eq!(spec.dim, 256);
        assert_eq!(spec.precision, Precision::F32);
        assert_eq!(Precision::F16.width(), 2);
        assert_eq!(Precision::F32.width(), 4);
        assert_eq!(Precision::F16.as_str(), "f16");
        assert_eq!(Precision::F32.as_str(), "f32");
    }

    #[test]
    fn embeddings_and_recall_blocks_parse() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(
            &p,
            r#"{"embeddings": {"enabled": true, "endpoint": "https://llm.example/v1"},
                "recall": {"rrf_k": 60}}"#,
        )
        .unwrap();
        let cfg = Config::load_from(&p).unwrap();
        assert!(cfg.embeddings.enabled);
        assert_eq!(cfg.embeddings.endpoint, "https://llm.example/v1");
        assert_eq!(cfg.embeddings.model, crate::embedding_profile::MODEL);
        assert_eq!(cfg.recall.rrf_k, 60.0);
    }

    #[test]
    fn governance_defaults_on_with_cadence_25() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::load_from(&dir.path().join("absent.json")).unwrap();
        assert!(cfg.governance.enabled, "default: governance on");
        assert_eq!(cfg.governance.reinject_every, 25);
        let p = dir.path().join("config.json");
        std::fs::write(&p, r#"{"resident": []}"#).unwrap();
        let cfg = Config::load_from(&p).unwrap();
        assert!(cfg.governance.enabled, "partial file: governance on");
        assert_eq!(cfg.governance.reinject_every, 25);
    }

    #[test]
    fn governance_block_parses_and_gates() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(&p, r#"{"governance": {"enabled": false}}"#).unwrap();
        let cfg = Config::load_from(&p).unwrap();
        assert!(!cfg.governance.enabled);
        assert_eq!(cfg.governance.reinject_every, 25, "partial block keeps the default cadence");
        std::fs::write(&p, r#"{"governance": {"reinject_every": 7}}"#).unwrap();
        let cfg = Config::load_from(&p).unwrap();
        assert!(cfg.governance.enabled);
        assert_eq!(cfg.governance.reinject_every, 7);
    }

    #[test]
    fn serve_and_client_default_off() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::load_from(&dir.path().join("absent.json")).unwrap();
        assert!(!cfg.serve.enabled);
        assert!(cfg.serve.bind.is_none());
        assert!(cfg.serve.origin.is_none());
        assert!(cfg.serve.token_file.is_none());
        assert!(cfg.client.serving.is_none());
    }

    #[test]
    fn serve_and_client_blocks_parse() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(
            &p,
            r#"{"serve": {"enabled": true, "bind": "0.0.0.0:9737",
                          "origin": "storage-1", "token_file": "/tmp/t"}}"#,
        )
        .unwrap();
        let cfg = Config::load_from(&p).unwrap();
        assert!(cfg.serve.enabled);
        assert_eq!(cfg.serve.bind.as_deref(), Some("0.0.0.0:9737"));
        assert_eq!(cfg.serve.origin.as_deref(), Some("storage-1"));
        assert_eq!(cfg.serve.token_file, Some(PathBuf::from("/tmp/t")));

        std::fs::write(
            &p,
            r#"{"client": {"serving": {"addr": "storage-1.example:9737", "token_file": "/tmp/t"}}}"#,
        )
        .unwrap();
        let cfg = Config::load_from(&p).unwrap();
        let cs = cfg.client.serving.as_ref().unwrap();
        assert_eq!(cs.addr, "storage-1.example:9737");
        assert_eq!(cs.token_file, PathBuf::from("/tmp/t"));
    }

    #[test]
    fn serving_host_and_none_tier_client_are_mutually_exclusive() {
        // serve.enabled needs a local index; client.serving forbids one.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(
            &p,
            r#"{"serve": {"enabled": true},
                "client": {"serving": {"addr": "h:1", "token_file": "/tmp/t"}}}"#,
        )
        .unwrap();
        assert!(Config::load_from(&p).is_err());
    }

    #[test]
    fn serve_bind_requires_token_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(&p, r#"{"serve": {"enabled": true, "bind": "127.0.0.1:0"}}"#).unwrap();
        assert!(Config::load_from(&p).is_err(), "an open unauthenticated TCP listener must be unconfigurable");
    }

    #[test]
    fn shipped_ring_rules_reproduce_the_historical_mapping() {
        // Regression lock: the shipped default must assign exactly what the
        // hardcoded taxonomy assigned before it became configuration.
        let r = RingRules::default();
        assert_eq!(r.ring_for("AGENT.md"), 1);
        assert_eq!(r.ring_for("README.md"), 1);
        assert_eq!(r.ring_for("mind/memories/MEMORY.md"), 1);
        assert_eq!(r.ring_for("mind/memories/feedback_x.md"), 2);
        assert_eq!(r.ring_for("todo/active/task/STATUS.md"), 4);
        assert_eq!(r.ring_for("knowledge/hosts/example/storage.md"), 3);
        // A non-slash prefix is an EXACT path, never a string prefix.
        assert_eq!(r.ring_for("AGENT.md.bak"), 3);
        assert_eq!(r.ring_for("docs/AGENT.md"), 3);
    }

    #[test]
    fn first_matching_rule_wins_in_list_order() {
        let rules = RingRules {
            rules: vec![
                RingRule { prefix: "notes/pinned/".into(), ring: 1 },
                RingRule { prefix: "notes/".into(), ring: 4 },
                RingRule { prefix: String::new(), ring: 2 },
            ],
            exclude_prefixes: Vec::new(),
        };
        assert_eq!(rules.ring_for("notes/pinned/a.md"), 1, "the specific rule stands first");
        assert_eq!(rules.ring_for("notes/b.md"), 4);
        assert_eq!(rules.ring_for("anything/else.md"), 2, "empty prefix is the catch-all");
    }

    #[test]
    fn unmatched_paths_land_on_the_documented_fallback_ring() {
        let rules = RingRules { rules: Vec::new(), exclude_prefixes: Vec::new() };
        assert_eq!(rules.ring_for("whatever.md"), UNMATCHED_RING);
        assert_eq!(UNMATCHED_RING, 3);
    }

    #[test]
    fn custom_ring_rules_remap_a_fabricated_tree() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(
            &p,
            r#"{"ring_rules": [
                 {"prefix": "laws.md", "ring": 0},
                 {"prefix": "handbook/", "ring": 1},
                 {"prefix": "scratch/", "ring": 5},
                 {"prefix": "", "ring": 4}
               ]}"#,
        )
        .unwrap();
        let r = Config::load_from(&p).unwrap().rings();
        assert_eq!(r.ring_for("laws.md"), 0);
        assert_eq!(r.ring_for("handbook/style.md"), 1);
        assert_eq!(r.ring_for("scratch/dump.md"), 5);
        assert_eq!(r.ring_for("anything.md"), 4);
        // The shipped tree names carry no special meaning any more.
        assert_eq!(r.ring_for("AGENT.md"), 4);
        assert_eq!(r.ring_for("todo/x.md"), 4);
    }

    #[test]
    fn ring_above_six_is_refused_at_load() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(&p, r#"{"ring_rules": [{"prefix": "x/", "ring": 7}]}"#).unwrap();
        let err = Config::load_from(&p).unwrap_err().to_string();
        assert!(err.contains("ring 7"), "the message must name the bad ring: {err}");
        std::fs::write(&p, r#"{"ring_rules": [{"prefix": "x/", "ring": 6}]}"#).unwrap();
        assert!(Config::load_from(&p).is_ok(), "6 is the outermost legal ring");
    }

    #[test]
    fn hard_exclusions_survive_an_override_attempt() {
        // An operator may ADD exclusions; the credential/exhaust/git boundary
        // is not theirs to remove.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(
            &p,
            r#"{"exclude_prefixes": [], "ring_rules": [
                 {"prefix": "mind/secrets/", "ring": 0},
                 {"prefix": "logs/", "ring": 0},
                 {"prefix": ".git/", "ring": 0}
               ]}"#,
        )
        .unwrap();
        let r = Config::load_from(&p).unwrap().rings();
        assert!(r.excluded("mind/secrets/tokens.yml"));
        assert!(r.excluded("logs/session.log"));
        assert!(r.excluded(".git/config"));
        assert!(r.excluded("nested/repo/.git/config"), "a nested .git is git internals too");
        assert!(r.excluded_dir("mind/secrets"));
        assert!(r.excluded_dir("logs"));
        // What IS configurable actually became configurable.
        assert!(!r.excluded("projects/repo/notes.md"), "an emptied list stops excluding projects/");
        assert!(!r.excluded("knowledge/archive/old.md"));
    }

    #[test]
    fn shipped_exclusions_keep_projects_and_archive_out() {
        let r = RingRules::default();
        assert!(r.excluded("projects/repo/README.md"));
        assert!(r.excluded("knowledge/archive/old.md"));
        assert!(r.excluded_dir("projects"));
        assert!(r.excluded_dir("knowledge/archive"));
        assert!(!r.excluded("knowledge/live.md"));
        assert!(!r.excluded(".gitignore"), "a dotfile is not the .git directory");
    }

    #[test]
    fn operator_exclusions_add_to_the_hard_ones() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(&p, r#"{"exclude_prefixes": ["drafts"]}"#).unwrap();
        let r = Config::load_from(&p).unwrap().rings();
        assert!(r.excluded("drafts/x.md"), "a slashless prefix still means the subtree");
        assert!(r.excluded_dir("drafts"));
        assert!(!r.excluded("draftsman.md"), "prefix matching stops at the path separator");
        assert!(r.excluded("mind/secrets/x.yml"), "hard exclusions never depend on the list");
    }

    #[test]
    fn scope_defaults_to_everywhere_and_matches_host_or_repo() {
        let unscoped = Scope::default();
        assert!(unscoped.matches("any-host", Some("any-repo")));
        assert!(unscoped.matches("any-host", None));

        let by_host = Scope { hosts: vec!["build-box".into()], ..Scope::default() };
        assert!(by_host.matches("build-box", None));
        assert!(by_host.matches("build-box.example.net", None), "the first label matches too");
        assert!(!by_host.matches("laptop", Some("build-box")), "a host rule is not a repo rule");

        let by_repo = Scope { repos: vec!["widget".into()], ..Scope::default() };
        assert!(by_repo.matches("laptop", Some("widget")));
        assert!(!by_repo.matches("laptop", Some("gadget")));
        assert!(!by_repo.matches("laptop", None), "no cwd, no repo match");

        let both = Scope { hosts: vec!["build-box".into()], repos: vec!["widget".into()], always: false };
        assert!(both.matches("build-box", Some("gadget")), "hosts and repos are ORed");
        assert!(both.matches("laptop", Some("widget")));
        assert!(!both.matches("laptop", Some("gadget")));

        let always = Scope { hosts: vec!["build-box".into()], repos: Vec::new(), always: true };
        assert!(always.matches("laptop", None), "always wins over a narrower list");
    }

    #[test]
    fn resident_entry_scope_parses_and_defaults_empty() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(
            &p,
            r#"{"resident": [
                 {"path": "a.md", "ring": 1},
                 {"path": "b.md", "ring": 1, "scope": {"repos": ["widget"]}},
                 {"path": "c.md", "ring": 0, "scope": {"hosts": ["build-box"], "always": true}}
               ]}"#,
        )
        .unwrap();
        let cfg = Config::load_from(&p).unwrap();
        assert!(cfg.resident[0].scope.matches("any", None), "absent scope = everywhere");
        assert_eq!(cfg.resident[1].scope.repos, vec!["widget".to_string()]);
        assert!(cfg.resident[2].scope.always);
    }

    #[test]
    fn corrupt_config_is_an_error_not_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(&p, "{ nope").unwrap();
        assert!(Config::load_from(&p).is_err());
    }

    // ---- slices

    fn slice(name: &str, prefixes: &[&str]) -> SliceRule {
        SliceRule {
            name: name.into(),
            prefixes: prefixes.iter().map(|p| p.to_string()).collect(),
        }
    }

    #[test]
    fn a_document_resolves_to_its_innermost_slice() {
        let m = Slices::new(vec![
            slice("work", &["knowledge"]),
            slice("hosts", &["knowledge/hosts"]),
        ])
        .unwrap();
        // Longest matching prefix wins, whatever order they are declared in.
        assert_eq!(m.slice_for("knowledge/hosts/server/zfs.md"), "hosts");
        assert_eq!(m.slice_for("knowledge/world/cloudflare.md"), "work");
        assert_eq!(m.slice_for("mind/memories/x.md"), ROOT_SLICE);
    }

    #[test]
    fn a_prefix_matches_path_components_never_substrings() {
        let m = Slices::new(vec![slice("d", &["drafts"])]).unwrap();
        assert_eq!(m.slice_for("drafts/a.md"), "d");
        assert_eq!(m.slice_for("drafts"), "d");
        assert_eq!(m.slice_for("draftsman/a.md"), ROOT_SLICE, "no substring claims");
    }

    #[test]
    fn equal_length_claims_are_broken_by_configuration_order() {
        // Two slices claiming the same depth must not depend on iteration
        // luck: the earlier declaration wins, every time.
        let m = Slices::new(vec![slice("first", &["a/b"]), slice("second", &["a/b"])]).unwrap();
        assert_eq!(m.slice_for("a/b/c.md"), "first");
    }

    #[test]
    fn a_query_for_a_slice_matches_everything_nested_inside_it() {
        // `hosts` sits under `work`, so restricting to `work` must still
        // reach it — that is the whole matryoshka.
        let m = Slices::new(vec![
            slice("work", &["knowledge"]),
            slice("hosts", &["knowledge/hosts"]),
        ])
        .unwrap();
        assert_eq!(m.prefixes_of("work"), Some(["knowledge".to_string()].as_slice()));
        assert_eq!(m.prefixes_of(ROOT_SLICE), None, "the root slice restricts nothing");
        assert_eq!(m.prefixes_of("typo"), Some([].as_slice()), "an unknown name matches nothing");
    }

    #[test]
    fn ambiguous_or_world_claiming_slice_configurations_are_refused() {
        assert!(Slices::new(vec![slice("", &["a"])]).is_err(), "unnamed");
        assert!(Slices::new(vec![slice(ROOT_SLICE, &["a"])]).is_err(), "reserved name");
        assert!(
            Slices::new(vec![slice("x", &["a"]), slice("x", &["b"])]).is_err(),
            "two slices with one name have no innermost answer"
        );
        assert!(Slices::new(vec![slice("x", &[])]).is_err(), "claims nothing");
        assert!(Slices::new(vec![slice("x", &[""])]).is_err(), "an empty prefix claims the tree");
        assert!(Slices::new(vec![slice("x", &["/"])]).is_err());
        for bad in ["../escape", r"..\escape", "with space", " leading"] {
            assert!(Slices::new(vec![slice(bad, &["a"])]).is_err(), "name {bad:?}");
        }
    }

    #[test]
    fn no_slices_configured_is_one_implicit_slice() {
        let m = Slices::new(Vec::new()).unwrap();
        assert!(m.is_empty());
        assert_eq!(m.slice_for("anything/at/all.md"), ROOT_SLICE);
        assert_eq!(m.prefixes_of(ROOT_SLICE), None);
    }

    #[test]
    fn tree_config_cannot_set_machine_owned_keys() {
        // The tree config is agent-writable and cloned across machines; any
        // machine-owned key found there must be a hard error naming the key.
        for key in ["embeddings", "rerank", "maintenance", "serve", "client", "brain_root"] {
            let tree = format!("{{\"{key}\": {{}}}}");
            let err = Config::load_layered(
                Some((std::path::Path::new("/tree/.cfetch/config.json"), tree.as_str())),
                None,
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains(key), "error must name {key}: {err}");
        }
    }

    #[test]
    fn tree_config_paths_must_stay_inside_the_tree() {
        let cases = [
            r#"{"resident": [{"path": "/etc/passwd", "ring": 1}]}"#,
            r#"{"resident": [{"path": "../outside.md", "ring": 1}]}"#,
            r#"{"resident": [{"path": "C:/windows/system32/config", "ring": 1}]}"#,
            r#"{"code_roots": ["/home"]}"#,
            r#"{"code_roots": ["../.."]}"#,
        ];
        for tree in cases {
            let err = Config::load_layered(
                Some((std::path::Path::new("/tree/.cfetch/config.json"), tree)),
                None,
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("tree-relative"), "case {tree}: {err}");
        }
        // Relative paths remain fine, and the machine layer may still use
        // absolute ones (operator state; `resolve` keeps them as given, with
        // the usual platform drive-root normalization on Windows).
        let cfg = Config::load_layered(
            Some((std::path::Path::new("/t"), r#"{"code_roots": ["projects/local"]}"#)),
            Some((std::path::Path::new("/m"), r#"{"code_roots": ["/opt/repos"]}"#)),
        )
        .unwrap();
        assert_eq!(
            cfg.effective_code_roots(),
            vec![cfg.resolve(std::path::Path::new("/opt/repos"))]
        );
    }

    #[test]
    fn machine_layer_overlays_tree_content() {
        let cfg = Config::load_layered(
            Some((std::path::Path::new("/t"), r#"{"budget_chars": 100, "exclude_prefixes": ["z/"]}"#)),
            Some((std::path::Path::new("/m"), r#"{"budget_chars": 4321}"#)),
        )
        .unwrap();
        assert_eq!(cfg.budget_chars, 4321);
        assert!(cfg.exclude_prefixes.iter().any(|p| p == "z/"), "tree content keys survive the overlay");
    }

    #[test]
    fn resident_entries_cannot_break_the_secrets_bar() {
        // The one mechanism that injects without recall being asked must
        // honor the strongest guarantee the tool makes — whatever layer
        // the entry arrived from, including an agent-written, cloned tree
        // config.
        for path in ["mind/secrets/token.md", "mind/secrets/deep/nested.key", "logs/exhaust.txt"] {
            let err = Config::load_layered(
                Some((
                    std::path::Path::new("/t"),
                    &format!(r#"{{"resident":[{{"path":"{path}","ring":0}}]}}"#),
                )),
                None,
            )
            .unwrap_err();
            assert!(
                err.to_string().contains("hard-excluded"),
                "resident {path} must be refused: {err}"
            );
        }
        // The tree config is exactly the layer agents write and machines
        // clone; a machine-layer entry is refused just the same.
        let err = Config::load_layered(
            None,
            Some((
                std::path::Path::new("/m"),
                r#"{"resident":[{"path":"mind/secrets/api_key","ring":1}]}"#,
            )),
        )
        .unwrap_err();
        assert!(err.to_string().contains("hard-excluded"), "machine layer too: {err}");
    }

    #[test]
    fn resident_entries_under_operator_exclusions_are_refused() {
        let err = Config::load_layered(
            Some((
                std::path::Path::new("/t"),
                r#"{"exclude_prefixes":["knowledge/drafts/"],"resident":[{"path":"knowledge/drafts/wip.md","ring":1}]}"#,
            )),
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("exclude_prefixes"),
            "excluded and injected at once must be refused: {err}"
        );
        // Sanity: the same entry outside the exclusion loads fine.
        Config::load_layered(
            Some((
                std::path::Path::new("/t"),
                r#"{"exclude_prefixes":["knowledge/drafts/"],"resident":[{"path":"knowledge/notes.md","ring":1}]}"#,
            )),
            None,
        )
        .unwrap();
    }

    #[test]
    fn machine_layer_owns_the_embeddings_endpoint() {
        let cfg = Config::load_layered(
            Some((std::path::Path::new("/t"), r#"{"resident": []}"#)),
            Some((
                std::path::Path::new("/m"),
                r#"{"embeddings": {"endpoint": "https://embed.example/v1"}}"#,
            )),
        )
        .unwrap();
        assert_eq!(cfg.embeddings.endpoint, "https://embed.example/v1");
        assert!(!cfg.embeddings.enabled);
    }

    #[test]
    fn brain_root_never_comes_from_a_file() {
        let cfg = Config::load_layered(
            None,
            Some((std::path::Path::new("/m"), r#"{"brain_root": "/elsewhere"}"#)),
        )
        .unwrap();
        assert_eq!(cfg.brain_root, paths::default_brain_root());

        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(&p, r#"{"brain_root": "/also/elsewhere"}"#).unwrap();
        let cfg = Config::load_from(&p).unwrap();
        assert_eq!(cfg.brain_root, paths::default_brain_root());
    }
}
