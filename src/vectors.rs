//! The SHARED vector artifact store.
//!
//! Vectors are profile-bound derived artifacts of shared CONTENT, not of a
//! host. An admitted backend emits the frozen profile's canonical output
//! codec, but another admitted backend is not required to emit identical
//! bytes. The first derived record for a content hash is retained and shared;
//! compatibility comes from global ordered-pair and adversarial mixed-store
//! retrieval tests, never from treating that producer as a numerical
//! reference. Records live in the tree at
//! `<brain_root>/state/cfetch/vectors/`, where every host that can reach the
//! tree reads them. Only a host with the embed capability writes. The per-host
//! index.db keeps a cache of the same vectors for query speed; the cache is
//! never the record, and losing it costs a re-read, never an embedding run.
//!
//! Layout, per network-major embedding profile:
//!
//! - `<slug>.bin` — records back to back, no per-record header. Network-major
//!   v1 records use the signed `INT8×768` output/storage codec and are exactly
//!   768 bytes. The store machinery remains generic for non-network test and
//!   migration profiles; every record is `dim * precision.width()` bytes, so
//!   record i lives at offset `i * stride` and the offset table is implicit.
//! - `<slug>.idx` — a text header (magic, exact model, dim, precision) then
//!   one content hash per line, line i naming record i.
//!
//! One packed file plus one index beats one file per hash: this corpus has
//! ~20k blocks, and 20k files of ~2 KB would burn an inode each, turn a
//! hydrate into 20k network round trips on an NFS-mounted tree, and drop a
//! 20k-entry directory into a tree people open in Obsidian and rsync. The
//! packed form reads whole in one sequential pass.
//!
//! The store is APPEND-ONLY. A vector whose text left this host's tree is not
//! deleted here: the same content may still live in a slice another host
//! holds, and this file is the group's record, not one host's cache. (The
//! local cache in index.db IS pruned on every scan — that is what keeps
//! coverage numbers honest.) Compaction, when it comes, has to be a
//! group-wide operation that knows every holder; a local delete never is.
//! `embed-index` and `status` print artifact count and block coverage side by
//! side, so the gap between them is visible rather than mysterious.
//!
//! Crash consistency: the record is appended first, its index line second, so
//! a torn tail can only ever leave an ORPHAN RECORD — never an index line
//! pointing at bytes that are not there. Readers use `min(index lines,
//! records)`; the next writer truncates the orphan away under the lock.

use std::collections::HashSet;
use std::io::{Read as _, Seek as _, Write as _};
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use rusqlite::Connection;

use crate::config::{Precision, VectorSpec};
use crate::index;

/// First line of every profile-bound `.idx`. Earlier formats were configurable
/// and may contain FP trailers, so they are intentionally not read as v1
/// network artifacts.
const MAGIC: &str = "cfetch-vectors v3";
const HEADER_LINES: usize = 8;
const PEER_ARTIFACT_MAGIC: &[u8] = b"cfetch-vector-artifact-v1\0";
const PEER_ARTIFACT_HEADER_BYTES: usize = PEER_ARTIFACT_MAGIC.len() + 32 + 64;
pub(crate) const MAX_PEER_ARTIFACTS: usize = 256;

/// Short, stable, filename-safe digest of a document prefix.
fn prefix_tag(prefix: &str) -> String {
    use sha2::Digest as _;
    let mut h = sha2::Sha256::new();
    h.update(prefix.as_bytes());
    h.finalize().iter().take(4).map(|b| format!("{b:02x}")).collect()
}

/// Up to this many missing vectors, a hydrate seeks per record instead of
/// streaming the whole artifact file.
const SEEK_UNTIL: usize = 64;

/// Enforce the canonical codec once a store identifies the frozen v1 model.
/// Full semantic configuration is rejected by `embedding_profile::validate`
/// before runtime selection; this boundary independently prevents an exact-v1
/// producer from persisting a backend-native float format by mistake.
fn is_canonical_v1_profile(spec: &VectorSpec) -> bool {
    spec.network_major == crate::embedding_profile::NETWORK_MAJOR
        && spec.profile_id == crate::embedding_profile::PROFILE_ID
}

fn validate_storage_spec(spec: &VectorSpec) -> anyhow::Result<()> {
    if is_canonical_v1_profile(spec) {
        anyhow::ensure!(
            spec.model == crate::embedding_profile::MODEL,
            "embedding profile {} requires model {:?}",
            spec.profile_id,
            crate::embedding_profile::MODEL
        );
        anyhow::ensure!(
            spec.dim == crate::embedding_profile::DIMENSIONS,
            "embedding profile {} requires signed INT8×{} output records",
            spec.profile_id,
            crate::embedding_profile::DIMENSIONS
        );
        anyhow::ensure!(
            spec.precision == Precision::I8,
            "embedding profile {} requires the canonical signed INT8×{} output/storage codec; backend-internal precision remains native",
            spec.profile_id,
            crate::embedding_profile::DIMENSIONS
        );
        anyhow::ensure!(
            spec.doc_prefix == crate::embedding_profile::DOCUMENT_PREFIX,
            "embedding profile {} requires document prefix {:?}",
            spec.profile_id,
            crate::embedding_profile::DOCUMENT_PREFIX
        );
    }
    Ok(())
}

/// Validates bytes at every boundary that can introduce or consume a vector
/// record. The v1 codec is not merely "some non-zero i8 values": `vec_to_blob`
/// can emit neither -128 nor a non-zero record without at least one component
/// saturated to +/-127. Accepting a wider byte language would give the same
/// vector two encodings and undermine byte-exact repeatability within a scope.
fn validate_record_bytes(spec: &VectorSpec, record: &[u8]) -> anyhow::Result<()> {
    anyhow::ensure!(
        record.len() == spec.precision.record_bytes(spec.dim),
        "canonical vector record has {} bytes, profile {} requires {}",
        record.len(),
        spec.profile_id,
        spec.precision.record_bytes(spec.dim)
    );
    if is_canonical_v1_profile(spec) {
        anyhow::ensure!(
            !record.contains(&0x80),
            "canonical signed INT8 vector contains forbidden -128"
        );
        anyhow::ensure!(
            record
                .iter()
                .any(|byte| ((*byte as i8) as i16).unsigned_abs() == 127),
            "canonical signed INT8 vector has no component saturated to +/-127"
        );
    }
    degenerate(&index::blob_to_vec(record, spec.precision)).map_or(Ok(()), |why| {
        Err(anyhow::anyhow!(
            "canonical vector record is degenerate or invalid: {why}"
        ))
    })
}

/// Why a vector cannot be a real embedding, or `None` when it looks sane.
///
/// Deliberately cheap and total: this runs once per stored vector, and its
/// job is to catch a backend producing garbage, not to judge embedding
/// quality. Every check below is a thing a WORKING embedder cannot produce.
pub(crate) fn degenerate(v: &[f32]) -> Option<String> {
    if v.is_empty() {
        return Some("it has no components".into());
    }
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Some(format!("component {i} is {}", v[i]));
    }
    // Embeddings are L2-normalized before storage, so a healthy norm is ~1.0.
    // An all-zero vector is the specific shape a half-supported accelerator
    // graph returns, and it carries no information at all: every cosine
    // against it is 0, so it would rank identically against every query.
    let norm = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if norm < 1e-6 {
        return Some(format!("its L2 norm is {norm:e} — the vector is all zeros"));
    }
    None
}

/// Filename-safe form of a model id. Model names carry slashes and colons
/// (`sentence-transformers/all-MiniLM-L6-v2`), which are not filenames; the
/// mapping is lossy on purpose (two models CAN collide here), and the exact
/// model string in the `.idx` header is what actually gates a match.
fn slug(spec: &VectorSpec) -> String {
    let model: String = spec
        .model
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let profile: String = spec
        .profile_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let base = format!("network{}-{profile}-{model}-{}-{}", spec.network_major, spec.dim, spec.precision.as_str());
    // A document prefix changes every vector in the file, so it changes the
    // FILE — otherwise two hosts with different prefixes would append
    // incompatible records to one artifact and the header check would only
    // catch it on the unlucky one that opened it second.
    if spec.doc_prefix.is_empty() {
        base
    } else {
        format!("{base}-{}", prefix_tag(&spec.doc_prefix))
    }
}

/// A `(model, dim, precision)` artifact set on disk, with its hash list
/// loaded. Reading needs nothing else; writing goes through [`VectorWriter`].
#[derive(Debug)]
pub struct VectorStore {
    dir: PathBuf,
    spec: VectorSpec,
    /// Content hash of record i, in record order.
    hashes: Vec<String>,
    present: HashSet<String>,
}

impl VectorStore {
    /// Opens (never creates) the store for `spec` under a brain root. A store
    /// that does not exist yet reads as empty — a host with no artifacts is a
    /// host with zero coverage, not an error.
    pub fn open(brain_root: &Path, spec: &VectorSpec) -> anyhow::Result<VectorStore> {
        validate_storage_spec(spec)?;
        anyhow::ensure!(
            !spec.model.contains(['\n', '\r']),
            "embeddings.model must not contain newlines"
        );
        anyhow::ensure!(
            !spec.doc_prefix.contains(['\n', '\r']),
            "embeddings.document_prefix must not contain newlines — it is stored on one \
             header line, and a newline there would be read as the end of the header"
        );
        anyhow::ensure!(spec.dim > 0, "embeddings.dimensions must be at least 1");
        let dir = crate::paths::shared_vector_dir(brain_root);
        let mut store =
            VectorStore { dir, spec: spec.clone(), hashes: Vec::new(), present: HashSet::new() };
        store.reload()?;
        Ok(store)
    }

    fn idx_path(&self) -> PathBuf {
        self.dir.join(format!("{}.idx", slug(&self.spec)))
    }

    fn bin_path(&self) -> PathBuf {
        self.dir.join(format!("{}.bin", slug(&self.spec)))
    }

    /// Rereads the index file. Only records that are BOTH listed and present
    /// in the `.bin` count — a torn tail is invisible to readers.
    fn reload(&mut self) -> anyhow::Result<()> {
        self.hashes.clear();
        self.present.clear();
        let idx = self.idx_path();
        let raw = match std::fs::read_to_string(&idx) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(anyhow::anyhow!("read {}: {e}", idx.display())),
        };
        let mut lines = raw.lines();
        let magic = lines.next().unwrap_or_default();
        anyhow::ensure!(
            magic == MAGIC,
            "{} uses legacy vector format {magic:?}; cfetch network major {} requires {} and a coordinated re-embedding",
            idx.display(),
            self.spec.network_major,
            MAGIC
        );
        let mut header = std::collections::HashMap::new();
        for _ in 0..HEADER_LINES - 1 {
            let line = lines.next().context("truncated vector index header")?;
            let (k, v) = line.split_once(' ').context("malformed vector index header")?;
            header.insert(k.to_string(), v.to_string());
        }
        anyhow::ensure!(
            header.get("network_major").and_then(|v| v.parse().ok()) == Some(self.spec.network_major),
            "{} belongs to another cfetch network major",
            idx.display()
        );
        anyhow::ensure!(
            header.get("profile_id").map(String::as_str) == Some(self.spec.profile_id.as_str()),
            "{} belongs to another embedding profile",
            idx.display()
        );
        // The slug is lossy (a slash becomes an underscore), so the exact
        // model string is what actually decides whether these vectors are
        // ours. A mismatch is loud: two models' vectors in one ranking are
        // numbers that only LOOK like similarity.
        let stored_model = header.get("model").map(String::as_str).unwrap_or_default();
        anyhow::ensure!(
            stored_model == self.spec.model,
            "{} holds vectors of model {stored_model:?}, not {:?}",
            idx.display(),
            self.spec.model
        );
        anyhow::ensure!(
            header.get("vector_encoding").map(String::as_str)
                == Some(self.spec.vector_encoding().as_str()),
            "{} holds another vector encoding",
            idx.display()
        );
        anyhow::ensure!(
            header.get("dim").map(String::as_str) == Some(self.spec.dim.to_string().as_str()),
            "{} holds a different dimension than embeddings.dimensions={}",
            idx.display(),
            self.spec.dim
        );
        anyhow::ensure!(
            header.get("precision").map(String::as_str) == Some(self.spec.precision.as_str()),
            "{} holds a different precision than embeddings.precision={}",
            idx.display(),
            self.spec.precision.as_str()
        );
        // A v1 file carries no doc_prefix line, which MEANS the empty prefix —
        // documents were embedded raw. Comparing it explicitly is what stops a
        // host that has since configured a prefix from appending vectors of a
        // different shape to the same file.
        let stored_prefix = header.get("doc_prefix").map(String::as_str).unwrap_or("");
        anyhow::ensure!(
            stored_prefix == self.spec.doc_prefix,
            "{} holds vectors embedded with document prefix {stored_prefix:?}, not {:?} — \
             these are different artifacts and must not share a file",
            idx.display(),
            self.spec.doc_prefix
        );
        let listed_raw: Vec<&str> = lines.filter(|l| !l.is_empty()).collect();
        let stored_records = std::fs::metadata(self.bin_path()).map(|m| m.len()).unwrap_or(0) as usize
            / self.stride();
        // Torn-tail repair: `writeln!` on an unbuffered File is two write(2)
        // calls (payload, then `\n`), so a crash between them leaves the
        // final line WITHOUT its newline. `lines()` accepted it as a valid
        // entry, and the next append's `writeln!` concatenated onto the
        // partial hash — silently corrupting two records. If the raw text
        // does not end with a newline, the last line is torn: drop it here
        // (and the bin matches by the truncate below), so the next append
        // starts on a clean boundary.
        let torn = !raw.ends_with('\n') && !listed_raw.is_empty();
        let listed: Vec<String> = if torn {
            listed_raw[..listed_raw.len() - 1].iter().map(|s| s.to_string()).collect()
        } else {
            listed_raw.iter().map(|s| s.to_string()).collect()
        };
        self.hashes = listed;
        self.hashes.truncate(stored_records);
        self.present = self.hashes.iter().cloned().collect();
        Ok(())
    }

    pub fn spec(&self) -> &VectorSpec {
        &self.spec
    }

    /// Refreshes the read view after another process (normally the daemon's
    /// peer-artifact receiver) appended records under the store lock.
    pub fn refresh(&mut self) -> anyhow::Result<()> {
        self.reload()
    }

    /// Bytes one vector occupies.
    fn stride(&self) -> usize {
        self.spec.precision.record_bytes(self.spec.dim)
    }

    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    pub fn contains(&self, hash: &str) -> bool {
        self.present.contains(hash)
    }

    /// One vector by content hash, widened to f32.
    pub fn get(&self, hash: &str) -> anyhow::Result<Option<Vec<f32>>> {
        Ok(self
            .get_blob(hash)?
            .map(|blob| index::blob_to_vec(&blob, self.spec.precision)))
    }

    /// One canonical output/storage record by content hash, without decoding
    /// or rewriting. Canonical describes its codec and retained identity, not
    /// backend-internal arithmetic or cross-backend byte equality.
    pub(crate) fn get_blob(&self, hash: &str) -> anyhow::Result<Option<Vec<u8>>> {
        let Some(record) = self.hashes.iter().position(|h| h == hash) else {
            return Ok(None);
        };
        let path = self.bin_path();
        let mut file = std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
        file.seek(std::io::SeekFrom::Start((record * self.stride()) as u64))?;
        let mut buf = vec![0u8; self.stride()];
        file.read_exact(&mut buf)
            .with_context(|| format!("read record {record} of {}", path.display()))?;
        validate_record_bytes(&self.spec, &buf)
            .with_context(|| format!("validate record {record} of {}", path.display()))?;
        Ok(Some(buf))
    }

    /// Streams the whole store in record order — one sequential read, the
    /// shape a hydrate wants.
    pub fn for_each(
        &self,
        mut f: impl FnMut(&str, Vec<f32>) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        if self.hashes.is_empty() {
            return Ok(());
        }
        let path = self.bin_path();
        let file = std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mut reader = std::io::BufReader::new(file);
        let mut buf = vec![0u8; self.stride()];
        for hash in &self.hashes {
            reader
                .read_exact(&mut buf)
                .with_context(|| format!("read {} (short of its index)", path.display()))?;
            validate_record_bytes(&self.spec, &buf)
                .with_context(|| format!("validate vector for {hash} in {}", path.display()))?;
            f(hash, index::blob_to_vec(&buf, self.spec.precision))?;
        }
        Ok(())
    }

    /// Takes the store's write lock and repairs any torn tail. Only a host
    /// with the embed capability ever calls this; every other host reads.
    pub fn begin_write(&mut self) -> anyhow::Result<VectorWriter<'_>> {
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("create {}", self.dir.display()))?;
        // Derived bytes belong in the tree, never in the tree's git history.
        let ignore = self.dir.join(".gitignore");
        if !ignore.exists() {
            std::fs::write(&ignore, "# Derived vector artifacts: shared as files, never as commits.\n*\n!.gitignore\n")
                .with_context(|| format!("write {}", ignore.display()))?;
        }
        let lock = crate::lockfile::acquire(&self.dir.join("store.lock"), 5_000, 0).context(
            "another embed run holds the shared vector store (derive-once: one writer per group)",
        )?;
        // Another writer may have appended since we opened: reload UNDER the
        // lock, then repair whatever a crash left behind.
        self.reload()?;
        let bin = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.bin_path())?;
        // `reload` already dropped unlisted records from the view; make the
        // file agree, so the next append lands where its index line says.
        bin.set_len((self.hashes.len() * self.stride()) as u64)?;
        let mut idx = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.idx_path())?;
        if idx.metadata()?.len() == 0 {
            write!(
                idx,
                "{}\nnetwork_major {}\nprofile_id {}\nmodel {}\ndim {}\nprecision {}\nvector_encoding {}\n",
                MAGIC,
                self.spec.network_major,
                self.spec.profile_id,
                self.spec.model,
                self.spec.dim,
                self.spec.precision.as_str(),
                self.spec.vector_encoding(),
            )?;
            writeln!(idx, "doc_prefix {}", self.spec.doc_prefix)?;
        } else {
            // Rewrite the hash list whenever the view is shorter than the
            // file (an index line whose record never landed).
            let raw = std::fs::read_to_string(self.idx_path())?;
            let listed = raw.lines().skip(HEADER_LINES).filter(|l| !l.is_empty()).count();
            if listed != self.hashes.len() {
                let header: String = raw.lines().take(HEADER_LINES).map(|l| format!("{l}\n")).collect();
                let body: String = self.hashes.iter().map(|h| format!("{h}\n")).collect();
                idx.set_len(0)?;
                idx.seek(std::io::SeekFrom::Start(0))?;
                idx.write_all(header.as_bytes())?;
                idx.write_all(body.as_bytes())?;
            }
        }
        idx.seek(std::io::SeekFrom::End(0))?;
        let mut bin = bin;
        bin.seek(std::io::SeekFrom::End(0))?;
        Ok(VectorWriter { store: self, _lock: lock, bin, idx })
    }
}

/// Exclusive writer: holds the store lock for its whole lifetime, so two
/// embed runs (on one host or two) can never interleave appends.
pub struct VectorWriter<'a> {
    store: &'a mut VectorStore,
    _lock: crate::lockfile::Lock,
    bin: std::fs::File,
    idx: std::fs::File,
}

impl VectorWriter<'_> {
    /// Returns the record currently retained by the derive-once store. Callers
    /// use this after losing a `put` race so their disposable local cache is
    /// populated from the shared winner rather than from the vector they just
    /// computed.
    pub(crate) fn get_retained(&self, hash: &str) -> anyhow::Result<Option<Vec<f32>>> {
        self.store.get(hash)
    }

    /// Appends one vector. Returns false when the hash is already stored (the
    /// derive-once contract: an artifact that exists is never recomputed).
    pub fn put(&mut self, hash: &str, vector: &[f32]) -> anyhow::Result<bool> {
        anyhow::ensure!(
            !hash.contains(['\n', '\r']) && !hash.is_empty(),
            "content hash {hash:?} is not a hash"
        );
        anyhow::ensure!(
            vector.len() == self.store.spec.dim,
            "vector has {} components, the store holds {}",
            vector.len(),
            self.store.spec.dim
        );
        // A degenerate vector must never reach the store. An accelerator whose
        // graph is only partly supported can return NaN or all zeros WITHOUT
        // raising anything — measured: a CoreML provider covering 20% of an
        // EmbeddingGemma graph produced norm-0.0 output and no error. Persist
        // one of those and it is indistinguishable from a healthy vector
        // afterwards; the corpus is quietly poisoned and only a full re-derive
        // finds it. This is the cheapest place in the system to make that
        // impossible, because everything funnels through here.
        degenerate(vector).map_or(Ok(()), |why| {
            Err(anyhow::anyhow!(
                "refusing to store a degenerate vector for {hash}: {why}. \
                 This usually means the inference backend cannot actually run this \
                 model — a partly-supported graph can return zeros or NaN without \
                 reporting an error."
            ))
        })?;
        let encoded = index::vec_to_blob(vector, self.store.spec.precision);
        self.put_encoded(hash, &encoded)
    }

    /// Appends one already-canonical output/storage record received from a peer.
    /// The byte representation is preserved exactly: decoding and re-
    /// quantizing an INT8 artifact could otherwise manufacture drift. A
    /// different admitted backend may legitimately have produced different
    /// bytes; it cannot replace the derive-once record already selected for
    /// this content hash.
    pub(crate) fn put_encoded(&mut self, hash: &str, encoded: &[u8]) -> anyhow::Result<bool> {
        anyhow::ensure!(
            !hash.contains(['\n', '\r']) && !hash.is_empty(),
            "content hash {hash:?} is not a hash"
        );
        validate_record_bytes(&self.store.spec, encoded)
            .with_context(|| format!("refusing peer vector for {hash}"))?;
        if self.store.present.contains(hash) {
            if !is_canonical_v1_profile(&self.store.spec) {
                let existing = self
                    .store
                    .get_blob(hash)?
                    .context("vector index named an existing record that could not be read")?;
                anyhow::ensure!(
                    existing == encoded,
                    "received different canonical bytes for existing content {hash} under profile {} — the derive-once store keeps its first record",
                    self.store.spec.profile_id
                );
            }
            // Under the admitted v1 profile, the first record is the
            // derive-once winner. Another backend may legitimately emit
            // different canonical bytes for the same text; retaining the
            // existing record is not a conflict and must not turn
            // cross-backend byte equality into an accidental runtime gate.
            return Ok(false);
        }
        // Record FIRST, index line second: a crash between them leaves an
        // orphan record (invisible, truncated by the next writer), never an
        // index line pointing at bytes that are not there.
        self.bin.write_all(encoded)?;
        writeln!(self.idx, "{hash}")?;
        self.store.hashes.push(hash.to_string());
        self.store.present.insert(hash.to_string());
        Ok(true)
    }

    /// Durably lands everything written so far — called per batch, so an
    /// interrupted run leaves committed work behind for the next one.
    pub fn flush(&mut self) -> anyhow::Result<()> {
        self.bin.flush()?;
        self.bin.sync_all()?;
        self.idx.flush()?;
        self.idx.sync_all()?;
        Ok(())
    }
}

/// Encodes one canonical vector record for iroh-blobs.
///
/// `nonce` is derived from a daemon-private key plus peer, slice and content
/// hash. It makes the resulting BLAKE3 hash an unguessable, stable bearer
/// capability: repeated requests deduplicate in memory, while another peer
/// or slice gets a different capability for the same vector.
pub(crate) fn encode_peer_artifact(
    hash: &str,
    record: &[u8],
    nonce: [u8; 32],
) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "content hash {hash:?} is not canonical lowercase SHA-256"
    );
    anyhow::ensure!(!record.is_empty(), "peer vector artifact has no vector bytes");
    let mut out = Vec::with_capacity(PEER_ARTIFACT_MAGIC.len() + 32 + 64 + record.len());
    out.extend_from_slice(PEER_ARTIFACT_MAGIC);
    out.extend_from_slice(&nonce);
    out.extend_from_slice(hash.as_bytes());
    out.extend_from_slice(record);
    Ok(out)
}

/// Exact byte length of one peer artifact for this vector-space profile.
/// Negotiation checks this before opening the blob transport so a peer cannot
/// use an honest content hash with a dishonest, much larger payload.
pub(crate) fn peer_artifact_len(spec: &VectorSpec) -> anyhow::Result<usize> {
    validate_storage_spec(spec)?;
    anyhow::ensure!(spec.dim > 0, "peer vector artifact has zero dimensions");
    let record = spec
        .dim
        .checked_mul(spec.precision.width())
        .and_then(|bytes| bytes.checked_add(spec.precision.trailer()))
        .context("peer vector artifact record length overflows usize")?;
    PEER_ARTIFACT_HEADER_BYTES
        .checked_add(record)
        .context("peer vector artifact length overflows usize")
}

/// Decodes and structurally validates one peer artifact. The expected hash is
/// supplied by the receiver, so a faulty peer cannot smuggle an unrelated
/// record into the local artifact store.
pub(crate) fn decode_peer_artifact(
    raw: &[u8],
    spec: &VectorSpec,
    expected_hash: &str,
) -> anyhow::Result<Vec<u8>> {
    let expected_len = peer_artifact_len(spec)?;
    anyhow::ensure!(
        raw.len() == expected_len,
        "peer vector artifact length is inconsistent"
    );
    anyhow::ensure!(
        raw.starts_with(PEER_ARTIFACT_MAGIC),
        "peer vector artifact has unknown format"
    );
    let hash_start = PEER_ARTIFACT_MAGIC.len() + 32;
    let hash = std::str::from_utf8(&raw[hash_start..hash_start + 64])?;
    anyhow::ensure!(
        hash == expected_hash,
        "peer returned vector {hash} for requested {expected_hash}"
    );
    let record = raw[PEER_ARTIFACT_HEADER_BYTES..].to_vec();
    validate_record_bytes(spec, &record)
        .with_context(|| format!("peer returned an invalid vector for {hash}"))?;
    Ok(record)
}

/// Fills the local index cache from the shared store: every block hash that
/// has no cached vector but IS in the store. Returns how many were imported.
/// This is what lets a second host answer semantically without ever holding
/// an embeddings key.
pub fn hydrate(conn: &Connection, store: &VectorStore) -> anyhow::Result<usize> {
    let spec = store.spec();
    // The cache and the meta rows that describe it move together: a cache
    // left over from another model/width/precision is unusable ballast, and
    // leaving meta disagreeing with the rows is how the NEXT run decides to
    // drop work it did not need to.
    index::ensure_vector_spec(conn, spec)?;
    if store.is_empty() {
        return Ok(0);
    }
    let missing = index::hashes_without_vectors(conn, spec, usize::MAX)?;
    let wanted: HashSet<String> = missing
        .into_iter()
        .map(|(hash, _)| hash)
        .filter(|hash| store.contains(hash))
        .collect();
    if wanted.is_empty() {
        return Ok(0);
    }
    let tx = conn.unchecked_transaction()?;
    let mut imported = 0usize;
    // One record whose bytes fail validation (a flipped byte on disk, or a
    // record written by an older build under looser rules) must not brick
    // semantic recall for every host: the read-side skip keeps the failure
    // to that one hash - it stays "missing" locally, the next derive
    // re-embeds it - instead of aborting the whole hydrate. Write-side
    // validation stays strict.
    let mut skipped = 0usize;
    if wanted.len() <= SEEK_UNTIL {
        // The everyday case after an edit: a handful of hashes. Seeking to
        // each record beats streaming a 40 MB artifact file to find five.
        for hash in &wanted {
            match store.get(hash) {
                Ok(Some(vector)) => {
                    index::insert_vector(&tx, hash, spec, &vector)?;
                    imported += 1;
                }
                Ok(None) => {}
                Err(error) => {
                    eprintln!("cfetch: skipping corrupt vector record {hash}: {error:#}");
                    skipped += 1;
                }
            }
        }
    } else {
        // A first hydrate wants everything: one sequential pass, not 20k
        // seeks over what may be an NFS-mounted tree.
        store.for_each(|hash, vector| {
            if wanted.contains(hash) {
                index::insert_vector(&tx, hash, spec, &vector)?;
                imported += 1;
            }
            Ok(())
        })?;
    }
    if skipped > 0 {
        eprintln!(
            "cfetch: {skipped} corrupt vector record(s) skipped; re-embedding will replace them"
        );
    }
    tx.commit()?;
    Ok(imported)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(dim: usize, precision: Precision) -> VectorSpec {
        VectorSpec {
            network_major: 1,
            profile_id: "test-profile".into(),
            model: "test-model".into(),
            dim,
            precision,
            doc_prefix: String::new(),
        }
    }

    fn canonical_spec() -> VectorSpec {
        VectorSpec {
            network_major: crate::embedding_profile::NETWORK_MAJOR,
            profile_id: crate::embedding_profile::PROFILE_ID.into(),
            model: crate::embedding_profile::MODEL.into(),
            dim: crate::embedding_profile::DIMENSIONS,
            precision: Precision::I8,
            doc_prefix: crate::embedding_profile::DOCUMENT_PREFIX.into(),
        }
    }

    #[test]
    fn v1_model_store_accepts_only_the_signed_int8x768_codec() {
        let brain = tempfile::tempdir().unwrap();
        VectorStore::open(brain.path(), &canonical_spec()).unwrap();

        let mut wrong = canonical_spec();
        wrong.precision = Precision::F16;
        let error = VectorStore::open(brain.path(), &wrong)
            .unwrap_err()
            .to_string();
        assert!(error.contains("output/storage codec"), "{error}");
        assert!(
            error.contains("backend-internal precision remains native"),
            "{error}"
        );

        let mut wrong = canonical_spec();
        wrong.model = "wrong/model".into();
        assert!(VectorStore::open(brain.path(), &wrong).is_err());

        let mut wrong = canonical_spec();
        wrong.dim = 256;
        assert!(VectorStore::open(brain.path(), &wrong).is_err());

        let mut wrong = canonical_spec();
        wrong.doc_prefix.clear();
        assert!(VectorStore::open(brain.path(), &wrong).is_err());
    }

    #[test]
    fn v1_codec_is_scale_free_signed_int8x768_with_ties_to_even() {
        let brain = tempfile::tempdir().unwrap();
        let s = canonical_spec();
        let mut vector = vec![0.0; crate::embedding_profile::DIMENSIONS];
        vector[..6].copy_from_slice(&[
            1.0,
            -1.0,
            0.5 / 127.0,
            1.5 / 127.0,
            -0.5 / 127.0,
            -1.5 / 127.0,
        ]);
        let scaled: Vec<f32> = vector.iter().map(|component| component * 0.25).collect();

        let mut store = VectorStore::open(brain.path(), &s).unwrap();
        {
            let mut writer = store.begin_write().unwrap();
            assert!(writer.put("native-output-a", &vector).unwrap());
            assert!(writer.put("native-output-b", &scaled).unwrap());
        }

        let first = store.get_blob("native-output-a").unwrap().unwrap();
        let second = store.get_blob("native-output-b").unwrap().unwrap();
        assert_eq!(first.len(), 768);
        assert_eq!(
            first, second,
            "the codec discards positive per-vector scale"
        );
        assert_eq!(first[0] as i8, 127);
        assert_eq!(first[1] as i8, -127);
        assert_eq!(first[2] as i8, 0, "0.5 rounds to the even integer 0");
        assert_eq!(first[3] as i8, 2, "1.5 rounds to the even integer 2");
        assert_eq!(first[4] as i8, 0, "-0.5 rounds to the even integer 0");
        assert_eq!(first[5] as i8, -2, "-1.5 rounds to the even integer -2");
    }

    #[test]
    fn v1_rejects_noncanonical_int8_bytes_at_write_peer_and_read_boundaries() {
        let brain = tempfile::tempdir().unwrap();
        let s = canonical_spec();
        let hash = "ab".repeat(32);
        let mut store = VectorStore::open(brain.path(), &s).unwrap();
        {
            let mut writer = store.begin_write().unwrap();
            let unsaturated = vec![1; crate::embedding_profile::DIMENSIONS];
            let error = format!("{:#}", writer.put_encoded(&hash, &unsaturated).unwrap_err());
            assert!(error.contains("no component saturated"), "{error}");

            let mut minus_128 = vec![0; crate::embedding_profile::DIMENSIONS];
            minus_128[0] = 0x80;
            minus_128[1] = 127;
            let error = format!("{:#}", writer.put_encoded(&hash, &minus_128).unwrap_err());
            assert!(error.contains("forbidden -128"), "{error}");
        }

        let mut peer_record = vec![0; crate::embedding_profile::DIMENSIONS];
        peer_record[0] = 0x81; // canonical -127
        peer_record[1] = 0x80; // forbidden -128
        let raw = encode_peer_artifact(&hash, &peer_record, [7; 32]).unwrap();
        let error = format!("{:#}", decode_peer_artifact(&raw, &s, &hash).unwrap_err());
        assert!(error.contains("forbidden -128"), "{error}");

        let mut valid = vec![0.0; crate::embedding_profile::DIMENSIONS];
        valid[0] = 1.0;
        {
            let mut writer = store.begin_write().unwrap();
            assert!(writer.put(&hash, &valid).unwrap());
            writer.flush().unwrap();
        }
        let path = store.bin_path();
        std::fs::write(&path, vec![1; crate::embedding_profile::DIMENSIONS]).unwrap();
        let error = format!("{:#}", store.get_blob(&hash).unwrap_err());
        assert!(error.contains("no component saturated"), "{error}");
    }

    #[test]
    fn derive_once_retains_the_first_record_without_demanding_byte_identity() {
        let brain = tempfile::tempdir().unwrap();
        let s = canonical_spec();
        let mut first = vec![0.0; crate::embedding_profile::DIMENSIONS];
        first[0] = 1.0;
        let mut another_admitted_output = first.clone();
        another_admitted_output[1] = 0.01;

        let mut store = VectorStore::open(brain.path(), &s).unwrap();
        let mut writer = store.begin_write().unwrap();
        assert!(writer.put("same-content", &first).unwrap());
        assert!(
            !writer
                .put("same-content", &another_admitted_output)
                .unwrap()
        );
        let retained = writer.get_retained("same-content").unwrap().unwrap();
        let retained = crate::index::vec_to_blob(&retained, Precision::I8);
        assert_eq!(retained, crate::index::vec_to_blob(&first, Precision::I8));
        assert_ne!(
            retained,
            crate::index::vec_to_blob(&another_admitted_output, Precision::I8)
        );
    }

    #[test]
    fn missing_store_reads_as_empty_never_as_an_error() {
        let brain = tempfile::tempdir().unwrap();
        let store = VectorStore::open(brain.path(), &spec(4, Precision::F16)).unwrap();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(!store.contains("deadbeef"));
        assert!(store.get("deadbeef").unwrap().is_none());
    }

    #[test]
    fn peer_artifact_round_trip_preserves_canonical_bytes() {
        let s = spec(4, Precision::I8);
        let hash = "ab".repeat(32);
        let record = vec![1, 2, 3, 4];
        let raw = encode_peer_artifact(&hash, &record, [7; 32]).unwrap();
        assert_eq!(decode_peer_artifact(&raw, &s, &hash).unwrap(), record);
        assert!(
            decode_peer_artifact(&raw, &s, &"cd".repeat(32))
                .unwrap_err()
                .to_string()
                .contains("requested")
        );
    }

    #[test]
    fn peer_artifact_length_is_exact_and_overflow_checked() {
        for (precision, width) in [(Precision::I8, 1), (Precision::F16, 2), (Precision::F32, 4)] {
            let s = spec(4, precision);
            assert_eq!(
                peer_artifact_len(&s).unwrap(),
                PEER_ARTIFACT_HEADER_BYTES + 4 * width
            );
        }
        let overflow = spec(usize::MAX, Precision::F32);
        assert!(
            peer_artifact_len(&overflow)
                .unwrap_err()
                .to_string()
                .contains("overflows")
        );
    }

    #[test]
    fn peer_artifact_rejects_wrong_width_and_degenerate_bytes() {
        let s = spec(4, Precision::I8);
        let hash = "ab".repeat(32);
        let short = encode_peer_artifact(&hash, &[1, 2], [7; 32]).unwrap();
        assert!(decode_peer_artifact(&short, &s, &hash).is_err());
        let zero = encode_peer_artifact(&hash, &[0, 0, 0, 0], [7; 32]).unwrap();
        assert!(
            format!("{:#}", decode_peer_artifact(&zero, &s, &hash).unwrap_err())
                .contains("degenerate")
        );
    }

    #[test]
    fn round_trips_vectors_through_the_shared_tree() {
        let brain = tempfile::tempdir().unwrap();
        let s = spec(4, Precision::F16);
        let mut store = VectorStore::open(brain.path(), &s).unwrap();
        {
            let mut w = store.begin_write().unwrap();
            assert!(w.put("aa", &[1.0, 0.0, 0.0, 0.0]).unwrap());
            assert!(w.put("bb", &[0.0, 1.0, 0.0, 0.0]).unwrap());
            assert!(!w.put("aa", &[1.0, 0.0, 0.0, 0.0]).unwrap(), "same bytes: already stored");
            let drift = w.put("aa", &[9.0, 9.0, 9.0, 9.0]).unwrap_err().to_string();
            assert!(drift.contains("derive-once store"), "{drift}");
            w.flush().unwrap();
        }
        assert_eq!(store.len(), 2);

        // A SECOND host: nothing but the tree, no endpoint, no key.
        let reopened = VectorStore::open(brain.path(), &s).unwrap();
        assert_eq!(reopened.len(), 2);
        assert!(reopened.contains("aa") && reopened.contains("bb"));
        assert_eq!(reopened.get("aa").unwrap().unwrap(), vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(reopened.get("bb").unwrap().unwrap(), vec![0.0, 1.0, 0.0, 0.0]);
        let mut seen = Vec::new();
        reopened
            .for_each(|hash, v| {
                seen.push((hash.to_string(), v));
                Ok(())
            })
            .unwrap();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].0, "aa", "record order is index order");
    }

    #[test]
    fn store_files_are_named_by_model_dim_and_precision() {
        let brain = tempfile::tempdir().unwrap();
        let s = VectorSpec {
            network_major: 1,
            profile_id: "test-profile".into(),
            model: "vendor/embed-8b".into(),
            dim: 8,
            precision: Precision::F32,
            doc_prefix: String::new(),
        };
        let mut store = VectorStore::open(brain.path(), &s).unwrap();
        store.begin_write().unwrap().put("aa", &[1.0; 8]).unwrap();
        let dir = crate::paths::shared_vector_dir(brain.path());
        let stem = "network1-test-profile-vendor_embed-8b-8-f32";
        assert!(dir.join(format!("{stem}.bin")).is_file(), "slug carries network-profile-model-dim-precision");
        assert!(dir.join(format!("{stem}.idx")).is_file());
        // Derived bytes must never ride into the operator's git history.
        assert!(dir.join(".gitignore").is_file(), "the store ignores itself in git");
    }

    #[test]
    fn a_different_spec_is_a_different_store() {
        let brain = tempfile::tempdir().unwrap();
        let mut a = VectorStore::open(brain.path(), &spec(4, Precision::F16)).unwrap();
        a.begin_write().unwrap().put("aa", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let b = VectorStore::open(brain.path(), &spec(8, Precision::F16)).unwrap();
        assert!(b.is_empty(), "another dimension is another artifact, never a partial read");
        let c = VectorStore::open(brain.path(), &spec(4, Precision::F32)).unwrap();
        assert!(c.is_empty(), "another precision is another artifact");
    }

    #[test]
    fn a_foreign_model_behind_the_same_slug_is_refused() {
        // Slugging is lossy: "a/b" and "a_b" collide. The exact model string
        // in the header is the real gate — a mismatch is loud, never a silent
        // mix of two models' vectors.
        let brain = tempfile::tempdir().unwrap();
        let mut a = VectorStore::open(
            brain.path(),
            &VectorSpec { model: "a/b".into(), ..spec(4, Precision::F16) },
        )
        .unwrap();
        a.begin_write().unwrap().put("aa", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let err = VectorStore::open(
            brain.path(),
            &VectorSpec { model: "a_b".into(), ..spec(4, Precision::F16) },
        )
        .unwrap_err();
        assert!(err.to_string().contains("a/b"), "the stored model is named: {err}");
    }

    #[test]
    fn an_orphan_record_from_a_torn_write_is_truncated_not_misread() {
        let brain = tempfile::tempdir().unwrap();
        let s = spec(4, Precision::F16);
        let mut store = VectorStore::open(brain.path(), &s).unwrap();
        {
            let mut w = store.begin_write().unwrap();
            w.put("aa", &[1.0, 0.0, 0.0, 0.0]).unwrap();
            w.flush().unwrap();
        }
        // Simulate a crash between the record append and the index line.
        let dir = crate::paths::shared_vector_dir(brain.path());
        let bin = dir.join(format!("{}.bin", slug(&s)));
        let mut f = std::fs::OpenOptions::new().append(true).open(&bin).unwrap();
        f.write_all(&[0xffu8; 8]).unwrap();
        drop(f);

        let torn = VectorStore::open(brain.path(), &s).unwrap();
        assert_eq!(torn.len(), 1, "readers see only paired records");

        let mut repaired = VectorStore::open(brain.path(), &s).unwrap();
        {
            let mut w = repaired.begin_write().unwrap();
            w.put("bb", &[0.0, 1.0, 0.0, 0.0]).unwrap();
            w.flush().unwrap();
        }
        let after = VectorStore::open(brain.path(), &s).unwrap();
        assert_eq!(after.len(), 2);
        assert_eq!(
            after.get("bb").unwrap().unwrap(),
            vec![0.0, 1.0, 0.0, 0.0],
            "the orphan was truncated, so bb's line names bb's bytes"
        );
    }

    #[test]
    fn hydrate_drops_a_cache_left_over_from_another_spec() {
        let brain = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(brain.path().join("knowledge")).unwrap();
        std::fs::write(brain.path().join("knowledge/a.md"), "- one\n").unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut conn = index::open(state.path()).unwrap();
        index::scan(&mut conn, brain.path(), None, &crate::config::RingRules::default()).unwrap();

        let old = VectorSpec { model: "old-model".into(), ..spec(2, Precision::F16) };
        index::ensure_vector_spec(&conn, &old).unwrap();
        let hash = index::content_hash("- one");
        index::insert_vector(&conn, &hash, &old, &[1.0, 0.0]).unwrap();

        let s = spec(2, Precision::F16);
        let mut store = VectorStore::open(brain.path(), &s).unwrap();
        store.begin_write().unwrap().put(&hash, &[0.0, 1.0]).unwrap();
        assert_eq!(hydrate(&conn, &store).unwrap(), 1);
        assert_eq!(index::stored_vector_spec(&conn).as_ref(), Some(&s), "meta follows the cache");
        assert_eq!(index::vector_coverage(&conn, &s).unwrap(), (1, 1));
        assert_eq!(index::vector_coverage(&conn, &old).unwrap(), (0, 1), "the old spec is gone");
    }

    #[test]
    fn hydrate_fills_the_local_cache_from_the_tree() {
        let brain = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(brain.path().join("knowledge")).unwrap();
        std::fs::write(brain.path().join("knowledge/a.md"), "- one\n- two\n").unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut conn = index::open(state.path()).unwrap();
        index::scan(&mut conn, brain.path(), None, &crate::config::RingRules::default()).unwrap();

        let s = spec(2, Precision::F16);
        let mut store = VectorStore::open(brain.path(), &s).unwrap();
        let missing = index::hashes_without_vectors(&conn, &s, 10).unwrap();
        assert_eq!(missing.len(), 2);
        {
            let mut w = store.begin_write().unwrap();
            for (hash, _) in &missing {
                w.put(hash, &[1.0, 0.0]).unwrap();
            }
            w.flush().unwrap();
        }
        assert_eq!(index::vector_coverage(&conn, &s).unwrap(), (0, 2), "cache is not the record");
        assert_eq!(hydrate(&conn, &store).unwrap(), 2);
        assert_eq!(index::vector_coverage(&conn, &s).unwrap(), (2, 2));
        assert_eq!(hydrate(&conn, &store).unwrap(), 0, "a second hydrate imports nothing");
    }

    #[test]
    fn legacy_normalized_record_cannot_hydrate_an_exact_body_key() {
        use sha2::Digest as _;

        let brain = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(brain.path().join("knowledge")).unwrap();
        std::fs::write(brain.path().join("knowledge/a.md"), "- us\n").unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut conn = index::open(state.path()).unwrap();
        index::scan(&mut conn, brain.path(), None, &crate::config::RingRules::default()).unwrap();

        let s = spec(2, Precision::F16);
        // The legacy key for "- US" was SHA256("- us"), even though the
        // model saw the uppercase body. Plain SHA256(body) would reuse it.
        let legacy_hash = crate::hashing::hex_lower(sha2::Sha256::digest(b"- us"));
        let current_hash = index::content_hash("- us");
        let mut store = VectorStore::open(brain.path(), &s).unwrap();
        {
            let mut writer = store.begin_write().unwrap();
            writer.put(&legacy_hash, &[1.0, 0.0]).unwrap();
            writer.flush().unwrap();
        }
        let legacy_record = store.get_blob(&legacy_hash).unwrap().unwrap();
        assert_eq!(hydrate(&conn, &store).unwrap(), 0);
        assert_eq!(index::vector_coverage(&conn, &s).unwrap(), (0, 1));
        assert_eq!(
            index::hashes_without_vectors(&conn, &s, 10).unwrap(),
            vec![(current_hash.clone(), "- us".to_string())]
        );
        {
            let mut writer = store.begin_write().unwrap();
            writer.put(&current_hash, &[0.0, 1.0]).unwrap();
            writer.flush().unwrap();
        }
        let reopened = VectorStore::open(brain.path(), &s).unwrap();
        assert_eq!(hydrate(&conn, &reopened).unwrap(), 1);
        assert_eq!(index::vector_coverage(&conn, &s).unwrap(), (1, 1));
        assert_eq!(reopened.len(), 2, "legacy artifacts remain in the append-only store");
        assert_eq!(reopened.get_blob(&legacy_hash).unwrap().unwrap(), legacy_record);
        let cached: Vec<u8> = conn
            .query_row("SELECT embedding FROM vectors", [], |row| row.get(0))
            .unwrap();
        assert_eq!(cached, reopened.get_blob(&current_hash).unwrap().unwrap());
        assert_ne!(cached, legacy_record);
    }

    // ---- document prefix as artifact identity

    fn spec_pfx(dim: usize, prefix: &str) -> VectorSpec {
        VectorSpec {
            network_major: 1,
            profile_id: "test-profile".into(),
            model: "test-model".into(),
            dim,
            precision: Precision::F16,
            doc_prefix: prefix.to_string(),
        }
    }

    #[test]
    fn a_document_prefix_makes_a_different_artifact_file() {
        // Same model, same width, different prefix: the vectors are not
        // comparable, so they must not land in one file.
        let raw = slug(&spec_pfx(4, ""));
        let pfx = slug(&spec_pfx(4, "passage: "));
        assert_ne!(raw, pfx);
        assert!(pfx.starts_with(&raw), "the prefixed name extends the base: {pfx}");
        // Stable across runs, and different prefixes never collide.
        assert_eq!(pfx, slug(&spec_pfx(4, "passage: ")));
        assert_ne!(pfx, slug(&spec_pfx(4, "query: ")));
    }

    #[test]
    fn a_store_refuses_vectors_embedded_under_another_prefix() {
        let brain = tempfile::tempdir().unwrap();
        let a = spec_pfx(2, "passage: ");
        {
            let mut store = VectorStore::open(brain.path(), &a).unwrap();
            let mut w = store.begin_write().unwrap();
            w.put("hash-one", &[1.0, 0.0]).unwrap();
        }
        // Reading it back under the SAME prefix works.
        assert_eq!(VectorStore::open(brain.path(), &a).unwrap().len(), 1);

        // Now force a different prefix onto the same filename and confirm the
        // header check catches it, since the filename tag is only a hint.
        let idx = VectorStore::open(brain.path(), &a).unwrap().idx_path();
        let raw = std::fs::read_to_string(&idx).unwrap();
        std::fs::write(&idx, raw.replace("doc_prefix passage: ", "doc_prefix other: ")).unwrap();
        let e = VectorStore::open(brain.path(), &a).unwrap_err().to_string();
        assert!(e.contains("document prefix"), "{e}");
    }

    #[test]
    fn every_store_header_names_the_complete_profile() {
        let brain = tempfile::tempdir().unwrap();
        let raw_spec = spec_pfx(2, "");
        {
            let mut store = VectorStore::open(brain.path(), &raw_spec).unwrap();
            let mut w = store.begin_write().unwrap();
            w.put("hash-one", &[1.0, 0.0]).unwrap();
        }
        let idx = VectorStore::open(brain.path(), &raw_spec).unwrap().idx_path();
        let head = std::fs::read_to_string(&idx).unwrap();
        assert!(head.starts_with(MAGIC), "profile store carries the current format: {head}");
        assert!(head.contains("network_major 1\n"));
        assert!(head.contains("profile_id test-profile\n"));
        assert!(head.contains("vector_encoding f16x2\n"));
        assert!(head.contains("doc_prefix \n"), "an empty prompt is still explicit");
        assert_eq!(VectorStore::open(brain.path(), &raw_spec).unwrap().len(), 1);
    }

    #[test]
    fn a_prefix_containing_a_newline_is_refused_before_it_corrupts_a_header() {
        let brain = tempfile::tempdir().unwrap();
        let e = VectorStore::open(brain.path(), &spec_pfx(2, "a\nb")).unwrap_err().to_string();
        assert!(e.contains("must not contain newlines"), "{e}");
    }

    // ---- the degenerate-vector guard

    #[test]
    fn a_healthy_normalized_vector_passes() {
        assert!(degenerate(&[0.6, 0.8]).is_none());
        assert!(degenerate(&[1.0, 0.0, 0.0, 0.0]).is_none());
        // Un-normalized but real vectors are not this guard's business.
        assert!(degenerate(&[3.0, 4.0]).is_none());
    }

    #[test]
    fn the_shapes_a_broken_accelerator_produces_are_caught() {
        // All zeros: what a partly-supported graph returns. Every cosine
        // against it is 0, so it would rank the same against every query.
        assert!(degenerate(&[0.0, 0.0, 0.0]).unwrap().contains("all zeros"));
        assert!(degenerate(&[f32::NAN, 1.0]).unwrap().contains("component 0"));
        assert!(degenerate(&[1.0, f32::INFINITY]).unwrap().contains("component 1"));
        assert!(degenerate(&[1.0, f32::NEG_INFINITY]).unwrap().contains("component 1"));
        assert!(degenerate(&[]).unwrap().contains("no components"));
    }

    #[test]
    fn the_store_refuses_a_degenerate_vector_rather_than_persisting_it() {
        // The whole point: once written, a zero vector looks exactly like a
        // healthy one. It has to be stopped at the door.
        let brain = tempfile::tempdir().unwrap();
        let sp = spec(2, Precision::F16);
        let mut store = VectorStore::open(brain.path(), &sp).unwrap();
        {
            let mut w = store.begin_write().unwrap();
            let e = w.put("hash-zero", &[0.0, 0.0]).unwrap_err().to_string();
            assert!(e.contains("degenerate"), "{e}");
            assert!(e.contains("cannot actually run this model"), "the cause is named: {e}");
            assert!(w.put("hash-nan", &[f32::NAN, 1.0]).is_err());
            // A good vector still goes in, so the guard is not a blanket refusal.
            assert!(w.put("hash-good", &[0.6, 0.8]).unwrap());
        }
        let reopened = VectorStore::open(brain.path(), &sp).unwrap();
        assert_eq!(reopened.len(), 1, "only the healthy vector was stored");
    }
}
