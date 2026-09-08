# Embedding profile v1

Status: candidate. No target-native local backend has been admitted yet.

cfetch network major 1 defines one shared semantic profile and one canonical
signed `INT8x768` output/index codec. It does not define one executable graph
or one internal numeric format. Each device package may use the artifact and
internal precision its native accelerator supports, provided its outputs pass
the same retrieval contract.

The decision priority is fixed: **compatibility first, then retrieval quality,
then efficiency**. NPU-first selection serves performance and energy use only
after a package belongs to the shared space; it never overrides compatibility.

Run `cfetch embedding-profile --json` to print the machine-readable contract.

## Frozen semantic and output contract

| Field | Value |
|---|---|
| Profile | `cfetch-embedding-v1` |
| Candidate source model | `google/embeddinggemma-300m` |
| Candidate source revision | `57c266a740f537b4dc058e1b0cda161fd15afa75` |
| Dimensions | 768 |
| Maximum context | 2,048 tokens |
| Token count | already-prefixed input, including every special token |
| Bucket selection | smallest supported bucket greater than or equal to token count |
| Padding | right padding; attention mask excludes padding |
| Truncation | disabled; reject counts above 2,048 |
| Query prompt | `task: search result \| query: ` |
| Document prompt | `title: none \| text: ` |
| Pooling | attention-mask-weighted mean, including the prompt |
| Normalization | L2 before output encoding |
| Output/index codec | signed `INT8x768`, per-vector max-absolute quantization with round-to-nearest-even |

The upstream Git revision binds the complete source repository. The executable
manifest additionally records SHA-256 for the backbone, both dense projection
weights, both tokenizer formats, and selected top-level configuration files.
Changing the admitted source revision or weights, tokenizer, prompts, pooling,
normalization, dimensions, or output/index codec creates a different vector
space and requires re-embedding.

Lifecycle and admission governance are deliberately outside that semantic
digest. A second, versioned admission-policy digest covers NPU/GPU/CPU class
requirements, the pinned evaluation dataset, absolute floors, exact signed-INT8
ranking and tie semantics, ordered all-pairs, adversarial mixed-document
selection, per-bucket semantic conformance, and wire-batch invariance. Changing
that policy forces backend recertification but does not reinterpret or
re-embed vectors whose semantic profile is unchanged.

The canonical codec is applied at the cfetch adapter boundary. A native
backend returns one finite, non-zero, 768-dimensional result. At runtime,
cfetch directly validates only observable shape and health, then requires the
complete exact execution scope admitted by this release and encodes the result
as the canonical 768-byte signed record. Pooling and source-normalization
conformance are certified for the pinned artifact/package scope during
admission; neither can be reconstructed or independently proven from one
output vector.

Every response row reports its post-prefix token count, selected sequence
bucket, `truncated: false`, and the one execution scope that produced it.
Production accepts batches of 1 through 64 items. Admission evidence proves
that the same 64 ordered inputs have identical canonical bytes under every
grouping size from 1 through 64, so scheduling cannot create another space.

## Target-native execution

These properties are deliberately outside the shared vector-space identity:

- internal weight, activation, accumulator, and output precision;
- runtime graph serialization and container format;
- compiler/runtime version and graph cache;
- kernel fusion, scheduling, packing, and device-specific layout;
- provider-specific execution and placement controls.

An Apple package may, for example, use Core ML with FP16 or mixed-precision
internals while an Android package uses a LiteRT model compiled for its NPU.
Both write the same canonical output codec. Neither is compatible merely
because it loads or emits 768 numbers; compatibility is earned by the global
retrieval gate.

Each admitted package pins its native artifact, runtime, compiler settings,
device family, and placement evidence. Those values make that package
reproducible without forcing a different device family to consume the same
file.

Each admitted registry entry also repeats the current semantic-profile and
admission-policy digests and points to the byte-hashed, passing exact-cohort
report. Updating a global header cannot make stale evidence current. Runtime
acceptance additionally binds a fresh nonce and the exact request/response
bytes with an Ed25519 signature from the scope key. For `supervised-local`,
that key is distributed with the package and is therefore a consistency check
inside cfetch's hash-pinned launcher, manifest, bearer, and child-process
boundary—not independent producer identity. For `remote-attested`, the key is
operator-held and copying public registry fields into another endpoint is
insufficient.

## Numeric compatibility

There is no reference NPU, CPU byte oracle, or preferred numerical backend.
Every candidate backend is evaluated as both query producer and document
producer against every candidate and admitted backend, including itself. For
`n` backends the gate contains all `n x n` ordered pairings.

The shared store may contain a different first-writer backend for every
document, so homogeneous pairs are not sufficient. For each query backend the
gate also evaluates a conservative mixed store: every relevant document takes
its minimum score and every irrelevant document its maximum score across all
document backends. That query-dependent adversary is stricter than any real
store, whose producer is fixed per content hash.

SciFact does not exercise every compiled sequence shape, so a separate
profile-pinned semantic fixture runs twice through each 32, 64, 128, 257, 512,
1,024, and 2,048-token bucket. Its query and both documents must tokenize to
the exact limit of the selected bucket under the frozen tokenizer. At every
bucket, the relevant document must rank strictly before the irrelevant
document for every ordered query/document scope pair. The gate also constructs
the derive-once adversary for each query scope and bucket: the minimum relevant
score and maximum irrelevant score may come from different document scopes,
and the former must still rank strictly first. The public report retains the
exact integer dots, norms, checks, and counts needed to replay those decisions;
cache hashes bind the canonical probe arrays without publishing them in the
report.

The 257-token boundary is intentional. Pre-activation physical diagnostics on
the first Lunar Lake NPU cohort found a deterministic semantic failure only at
an exact static length of 256; 255, 257, 258, and the other profile shapes
passed. Since no producer had been activated, the shared profile moved the
published bucket to 257 instead of disguising an internal padding workaround.

Every pairing must independently meet the same fixed absolute SciFact floors:

| Metric | Minimum |
|---|---:|
| NDCG@10 | 0.767907905520953 |
| Recall@100 | 0.970 |
| MRR@10 | 0.7305529100529101 |

No backend's score moves the floor for another backend. Corresponding-vector
byte equality and cosine similarity across backends are diagnostics, not
admission gates. Repeatability within the exact same artifact, runtime, and
device scope remains required.

The evaluator ranks signed records with the same exact integer dot products,
norms, sign handling, and cross multiplication as production Rust—never BLAS
or floating-point approximation. Ties use pinned corpus insertion order,
which is the evaluation database's block-ID order.

The executable gate is
`experiments/embedding-profile/cross_backend_eval.py`.

Admission does not trust the aggregate report alone. Every admitted registry
scope points to an immutable, SHA-256-named cache NPZ and measurement-evidence
ZIP in a cfetch GitHub release. CI downloads the bounded assets, validates the
NPZ/NPY container and all embedded sequence, placement, and performance JSON,
binds their scope metadata to the registry, retains every raw profiler and
benchmark output named by the summaries, and recomputes all retrieval and
per-bucket decisions. With the admitted registry currently empty, this replay
is a clean network-free no-op.

That replay proves byte integrity and policy arithmetic, not physical
placement. Honest raw profiler capture on the named device and trustworthy
runner provenance remain the hardware-attestation boundary.

## Shared-store behavior

Statement identity hashes the exact segmented Markdown body: SHA-256 over
`cfetch-statement-body-v1\0` followed by the body's UTF-8 bytes (`\0` is one
NUL byte). Case, spacing, indentation, and internal line breaks remain part of
the input. Segmentation retains its existing line-ending handling; enclosing
headings and table headers are separate retrieval context. Document embedding
prepends the profile's fixed document prefix to that same body. The citation
uses a prefix of the full statement hash, and identical bodies across files or
rings share one vector.

Schema 8 rebuilds older disposable catalogs from Markdown into this identity
scheme; read-only query and diagnostic opens reject stale schemas. Existing
citations must be refreshed after upgrading. The hash domain prevents reuse of
legacy normalized-body keys, which could name vectors produced from different
case or whitespace variants. Old shared records remain untouched in the
append-only store, but cannot satisfy new-body coverage. This changes statement
identity without changing model inputs, the semantic profile, or vector encoding.

cfetch stores one canonical record per profile and content hash. The first
record derived by an admitted backend is retained and distributed; another
producer does not silently overwrite it. This is the derive-once storage rule,
not a claim that every backend would emit the same bytes.

The byte language is strict: `-128` is forbidden and every non-zero record
contains at least one `-127` or `+127`, exactly as the max-absolute codec
emits. Those invariants are checked on local writes, peer ingress, cache-gate
loads, and reads.

The ordered-pair and adversarial mixed-document gates make that safe: a
query computed by any admitted backend must retrieve correctly against both
homogeneous producer sets and every possible per-document first-writer mix.

Records intentionally do not carry producer identity; content and semantic
profile are their durable identity. Consequently, revoking a scope for a
semantic defect requires rebuilding affected v1 artifacts (or advancing the
network major), not selectively trusting old records whose producer is no
longer knowable.

## Candidate packages, not support claims

EmbeddingGemma has useful existing packaging in both directions:

- Apple Neural Engine: Core ML is the package direction. The available
  community Core ML artifact declares the pinned standard EmbeddingGemma model
  family and carries matching tokenizer/prompt metadata. Exact encoder-weight
  and conversion lineage to the target source revision remains unproven. Its
  fixed 256-token shape also does not implement the profile's 2,048-token
  maximum, so it is an upstream input to rebuild and test rather than an
  admitted cfetch artifact or proof of actual ANE placement.
- Android NPUs: the official LiteRT community package provides static 256,
  512, 1,024, and 2,048-token variants, including compiled variants for Google
  Tensor, MediaTek, and Qualcomm families. Each concrete runtime/artifact/
  device scope still needs cfetch profile attestation, native placement
  evidence, and the full ordered-pair plus adversarial mixed-store gates.
- GPU and accelerated CPU: the ggml-org Q8_0 GGUF declares the standard
  EmbeddingGemma base family and is a ready cross-platform candidate for
  llama.cpp Metal, CUDA, ROCm/Vulkan, SYCL, and SIMD CPU packages. Its exact
  conversion lineage to the pinned source revision remains unproven. It still
  needs the same package evidence and global retrieval admission; being portable does
  not make it a numerical anchor.

`release/inference-backends.json` deliberately keeps the admitted set empty
until that evidence exists. The immediate integration boundary is a local
target-native adapter that serves cfetch's already attested embedding protocol
over loopback; it is local inference, not a remote-provider fallback. See
[Local embedding adapters](local-embedding-adapter.md) and
[Local accelerated inference](local-inference.md).

The production path checks the exact execution scope against that admitted
set before accepting freshly produced query or document vectors. Peer import
and serving are also disabled until the profile and registry are active. Once
active, a peer record crosses an authenticated, authorized storage-group trust
boundary and is rechecked for exact profile, content hash, length, and canonical
codec; the compact record intentionally carries no per-vector producer receipt.
Because the admitted set is currently empty, semantic production and peer
vector transfer on main fail closed; candidate probes and the compatibility
exporter remain available without contaminating the shared derive-once store.

## Pre-activation correction

The former candidate treated one static W8A8 ORT graph and its CPU bytes as
the network identity. It failed the retrieval-quality budget and could not
provide native coverage across accelerator families. It is not an active
cfetch vector-space artifact.

No tagged release enabled a local network-major-1 producer, and the public
variant catalog remains endpoint-only. The v1 candidate can therefore adopt
target-native internals and the global retrieval gates before activation without
reinterpreting a released shared store.
