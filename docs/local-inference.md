# Local accelerated inference

cfetch's packaging goal is accelerated local embedding on every supported
install. Runtime selection is fixed:

1. NPU
2. GPU
3. accelerated CPU

This is an execution and energy preference, not a numerical hierarchy. No NPU
defines the shared vector space. A target uses its NPU when an admitted native
artifact initializes and proves the required placement; otherwise it tries an
admitted GPU route and then its accelerated SIMD CPU route.

Remote inference is an explicit deployment choice. It is not inserted into
the local fallback chain and is not a substitute for a missing local package.

## The local adapter boundary

cfetch already accepts an attested OpenAI-compatible embedding endpoint and
applies its canonical signed `INT8x768` output/index encoding. The shortest
target-native integration is therefore a small local adapter:

    cfetch -> authenticated loopback -> native Core ML/LiteRT/OpenVINO runner
           -> 768-dimensional semantic output -> cfetch INT8x768 codec

The adapter process and model execute on the same device as cfetch. Loopback
is only the process boundary; this is local inference, not a remote provider.
It lets each platform use its mature native runtime without coupling the Rust
binary to every vendor SDK in the first package. See
[Local embedding adapters](local-embedding-adapter.md) for the protocol and
attestation contract.

## FastEmbed diagnostics

Build with `--features embedded-embeddings` to inspect and exercise
FastEmbed's model catalogue independently of the shared semantic profile:

```console
cfetch embed-model list
cfetch embed-model --model EmbeddingGemma300M status
cfetch embed-model --model EmbeddingGemma300M download
cfetch embed-model --model EmbeddingGemma300M test "A diagnostic input"
```

`--model` accepts the exact variant names printed by `list`; it applies only
to that command. The default remains `MultilingualE5Base`. `status` inspects
the selected repository's cache directory without downloading files or
initializing a model. An existing directory does not prove complete files or
loadability. `download` and `test` explicitly load the selected model and may
fetch missing files. `HF_HOME` overrides the model cache for both inspection
and loading.

These commands run the catalogue model's own tokenizer, sequence, and pooling
behavior. Their vectors are never indexed or shared. A model name, matching
dimension count, or successful diagnostic cannot admit a producer into the
shared space; the package and global compatibility gates below determine
that. There is no automatic shared-store model switch or re-embedding command
in this diagnostic interface. Query-local reranking remains independent.

## Backend plan

Different hardware families require different native artifacts and may use
different internal precision. FP16, mixed precision, weight-only integer, or
fully integer internals are all permitted. They are implementation details;
the frozen tokenizer, prompts, pooling, normalization, dimensions, and final
signed `INT8x768` codec are shared.

| Device | Primary runtime direction | Candidate artifact direction |
|---|---|---|
| Apple Neural Engine | Core ML | fixed-shape `mlpackage` / compiled model |
| Android Google Tensor NPU | LiteRT | device-compiled LiteRT model |
| Android MediaTek NPU | LiteRT / NeuroPilot | device-compiled LiteRT model |
| Android Qualcomm NPU | LiteRT Qualcomm delegate or QNN | LiteRT model / QNN context binary |
| Intel NPU | OpenVINO | OpenVINO IR / compiled blob |
| AMD NPU | Ryzen AI / Vitis AI | vendor-compiled graph |
| NVIDIA GPU | TensorRT | serialized native engine |
| AMD GPU | MIGraphX / ROCm | compiled ONNX graph |
| Intel GPU | OpenVINO | OpenVINO IR / compiled cache |
| Apple GPU | Core ML / Metal | compiled Core ML graph |
| Android GPU | LiteRT GPU delegate / Vulkan | delegate-compatible LiteRT model or GGUF |
| Linux ARM GPU floor | Vulkan / vendor OpenCL | GGUF or runtime-native graph for Mali, Adreno, or PowerVR |
| Windows GPU floor | DirectML | compiled ONNX graph |
| Windows Qualcomm NPU | QNN / Windows ML | QNN context or runtime-native graph |
| CPU floor | XNNPACK, oneDNN, or equivalent SIMD runtime | runtime-native graph |

The table is a packaging direction, not a support claim. Current admission
state is recorded in `release/inference-backends.json`; its admitted list is
empty until hardware evidence and retrieval results exist.

EmbeddingGemma is the current model candidate. Community packages are useful
conversion references, but none of them is a shared-space authority:

- The available Apple Core ML community package is an ANE-targeted route and
  declares the pinned standard EmbeddingGemma family, but its current
  artifact is fixed to 256 tokens. It must be rebuilt for the remaining profile
  buckets rather than silently truncating 2,048-token inputs.
- The official LiteRT community package offers 256, 512, 1,024, and 2,048-token
  artifacts, with device variants for Google Tensor, MediaTek, and Qualcomm.
  Those upstream artifacts are candidates; they are not admitted cfetch
  backends until their exact lineage, placement, and retrieval evidence pass.
- The ggml-org Q8_0 GGUF declares the standard EmbeddingGemma base family, but
  its exact conversion lineage to the pinned source revision is unproven and
  it cannot supply an NPU route. It remains a diagnostic candidate, not the
  basis of the first cohort.
- The first implementation package uses one exact-source OpenVINO IR on an
  Intel Lunar Lake NPU, Arc GPU, and accelerated CPU. OpenVINO is only that
  Intel package's native runtime; it is not cfetch's universal runtime or a
  numerical anchor. Apple, Android, AMD, Qualcomm, and other packages join by
  implementing the same adapter and admission contract.

## What admission proves

Every local backend scope must provide:

- the frozen semantic profile and output-codec attestation;
- a pinned, reproducible native artifact and runtime manifest;
- recorded internal precision rather than an assumed universal dtype;
- profiler-backed accelerator execution on the claimed NPU, GPU, or
  accelerated CPU for every supported sequence bucket, with complete fallback
  disclosure and no unexpected or silent fallback;
- finite, non-zero 768-dimensional output before canonical encoding;
- byte repeatability after encoding on the same artifact/runtime/device;
- the same fixed absolute SciFact floor for every ordered query/document
  pairing with every admitted and candidate backend;
- strict relevant-before-irrelevant ranking for the profile-pinned semantic
  fixture at every compiled sequence bucket, both for every ordered scope pair
  and for the relevant-minimum/irrelevant-maximum mixed-scope adversary;
- latency distribution and peak-memory measurements for every supported
  sequence bucket, plus energy measurements where the platform exposes them
  or an explicit `not_measured` reason;
- correct token counting, smallest-bucket selection, no truncation, and
  canonical-output invariance for the same 64 ordered inputs under every wire
  grouping size from 1 through 64.

There is no anchor backend. If there are `n` concrete backend scopes, release
evaluation runs all `n x n` ordered query-backend x document-backend pairings,
including every self-pair. Each pair must independently meet the absolute
NDCG@10, Recall@100, and MRR@10 floors in the embedding profile. Exact bytes
and corresponding-vector cosine across scopes remain diagnostics.

The derive-once store can mix document producers per content hash. Therefore
each query backend must also pass the adversarial mixed-document test: relevant
documents use their minimum score and irrelevant documents their maximum score
across all document backends. This is stricter than any real fixed first-writer
mixture.

The same derive-once construction is applied separately at every compiled
sequence bucket to a profile-pinned semantic fixture. This closes the gap where
short SciFact inputs pass but a deterministic long-shape graph is semantically
wrong. Each of the three inputs reaches the bucket's exact token limit under
the frozen tokenizer, and each bucket is executed twice for byte
repeatability. The global gate then checks every ordered scope pair and
requires the lowest relevant score across document scopes to remain strictly
above the highest irrelevant score.

Admission is scoped to the exact artifact, runtime, device family, compiler
settings, and placement evidence in its manifest. A result from one Apple,
Qualcomm, or other device family does not certify another family.

## Selection and failure

A released local package will expose one supervised loopback dispatcher, not
several manually chosen endpoints. Its embedded package plan supplies admitted
scope IDs in NPU, GPU, accelerated CPU order. The dispatcher attempts that
exact order, puts `cfetch_requested_scope_id` in each signed request body,
requires `cfetch_execution.scope_id` to match it, and caches only a successful
selection. Each packaged scope and response also binds
`transport: supervised-local`; explicitly configured endpoints bind
`transport: remote-attested`. Runtime initialization and signed response evidence—not a device
name—prove which candidate was used. A valid CPU reply therefore cannot pass
as completion of an NPU attempt.

Failure remains local and ordered:

    admitted NPU fails -> admitted GPU -> admitted accelerated CPU

Remote use happens only when the operator explicitly configures it.

The generic typed package-plan validator, supervised adapter process, exact
requested-scope binding, ordered fallback, and successful-scope cache are
implemented. They deliberately select nothing today because `local_packages`
and `admitted_backends` are both empty. Actual support begins only when the
first physical cohort and immutable package payload enter those arrays.
The serving daemon keeps one supervised dispatcher and successful-scope cache
for its process lifetime; rebuilding a query client does not restart the native
runtime or repeat failed accelerator discovery.

## Current state

The public release catalog is endpoint-only. Main admits no producer scope yet,
so semantic production fails closed rather than allowing an untested endpoint
to seed the shared derive-once store. The rejected local ORT/W8A8
candidate and its CPU-byte certification route are not released support.
Target-native local packages return only after their actual artifacts pass
placement, repeatability, performance, global ordered-pair, and adversarial
mixed-store retrieval gates. Their machine-readable OS/architecture/device to
ordered-scope composition lives in `release/inference-backends.json`.

The immediate admission sequence is:

1. build the pinned full EmbeddingGemma pipeline as seven static-shape
   OpenVINO graphs and execute the exact package on the available Lunar Lake
   NPU, Arc GPU, and accelerated CPU;
2. verify exact profile lineage and export canonical SciFact caches plus
   byte-bound per-bucket placement, repeated semantic-probe, sequence, and
   performance evidence;
3. assemble an initial cohort containing at least one NPU, one GPU, and one
   accelerated CPU scope—no NPU-only cohort can pass the policy;
4. run every ordered cohort/self query-document pairing and every adversarial
   mixed-document query backend against the fixed absolute floors, then repeat
   the ordered and adversarial strict-ranking checks at every sequence bucket;
5. run final conformance against the immutable staged package bytes, then
   atomically admit that package and cohort; repeat the global process for
   further device families without changing the shared vector identity.

For admission, maintainers rehost each exact cache as the cfetch release asset
`<sha256>.npz` and every raw profiler/benchmark output in a separately hashed
measurement ZIP. Registry entries bind both immutable HTTPS locators and
digests. CI replays all vector-space gates from those cache bytes and verifies
the measurement bundle against the embedded evidence summaries. This checks
integrity and reproducibility; the reviewed physical runner and profiler
provenance remain the trust boundary for claims about actual NPU/GPU/CPU
placement.
