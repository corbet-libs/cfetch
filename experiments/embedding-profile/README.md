# Global embedding compatibility admission

cross_backend_eval.py is the vector-compatibility gate for the shared INT8
space. Package admission additionally requires independently verified
placement, sequence coverage, latency, memory, and provenance evidence.
No device or runtime supplies the numerical reference. For `n` submitted
backend scopes, the evaluator runs all `n x n` ordered
query-backend/document-backend pairings, including every self-pair. Every
pairing must independently meet the same fixed absolute SciFact floors:

- NDCG@10: 0.767907905520953
- Recall@100: 0.970
- MRR@10: 0.7305529100529101

It also tests the derive-once store's real mixed-producer case. For each query
backend, relevant documents take their minimum score and irrelevant documents
their maximum score across every document backend. This adversarial matrix is
stricter than any real store, where each document's first-writer producer is
fixed.

Each backend runner writes an .npz file containing:

- metadata: one JSON string with schema_version, profile_id,
  profile_manifest_sha256, admission_policy_sha256, model, model_revision,
  vector_encoding,
  supported_max_tokens, supported_sequence_buckets,
  supported_max_batch_size, sequence_semantic_fixture_id,
  sequence_semantic_fixture_sha256,
  sequence_capability_evidence, sequence_capability_evidence_sha256, dataset,
  dataset_revision, scope_id, backend, runtime,
  compiler, package_target, artifact_source, artifact_sha256,
  attestation_public_key, internal_precision, device, device_class, placement_evidence,
  placement_evidence_sha256, performance_evidence,
  performance_evidence_sha256, and accelerated_placement: true;
- queries and documents: the canonical signed INT8x768 vectors in pinned
  SciFact order;
- queries_repeat and documents_repeat: a second run on the same
  runtime/artifact/device;
- sequence_probe_queries, sequence_probe_relevant_documents, and
  sequence_probe_irrelevant_documents: one pinned semantic triple for every
  sequence bucket, in `32..2048` bucket order;
- the same three sequence-probe arrays with `_repeat` suffix: a second actual
  adapter run at every graph shape;
- sequence_capability_evidence_bytes, placement_evidence_bytes, and
  performance_evidence_bytes: the exact raw evidence files whose SHA-256
  values appear in metadata;
- wire_batch_outputs: the retained canonical output for every grouping size 1
  through 64, stored as exact signed INT8 with shape `(64, 64, 768)`.

Every vector-bearing array, including the rank-three wire-batch output array,
must be stored with the exact signed 8-bit integer dtype; the evaluator rejects
values merely convertible to INT8, rejects `-128`, and requires every row to
contain `-127` or `+127` as the canonical max-absolute codec does. The repeat
arrays must
be byte-identical to their first run in the same backend/runtime/artifact/device
scope. Arrays from different scopes do not have to be byte-identical.
Cross-backend equality and cosine may be computed out of band for debugging,
but are not serialized in the authoritative report and never determine
admission. The sequence-shape semantic fixture is the narrow exception: for
every query-producer x document-producer pairing and every bucket, all three
fixture inputs must tokenize to that bucket's exact token limit and its
relevant document must rank strictly before its irrelevant document under
production's exact INT8 comparator. The conversion smoke rechecks those token
counts with the frozen tokenizer, so tokenizer or prefix drift aborts the
package build. For each query producer and bucket, the gate also takes the
relevant minimum and irrelevant maximum across all document producers and
requires that adversarial mixed-scope ordering to pass.

## Collect physical package evidence

`physical_evidence.py` owns and measures one exact package scope. It accepts
only an OpenVINO package whose top-level `package_state` is `physical-probe`;
all three physical-evidence digests and the compatibility-report digest must
be explicit JSON `null`. This is the only non-circular bootstrap package. A
probe package cannot be exported, admitted, selected by cfetch, or released.

The collector verifies the dispatcher and probe-package manifest hashes before
and after every child process. It launches the sibling dispatcher directly,
sends a random bearer through stdin without persisting it, retains the stdin
pipe as the parent-lifetime boundary, and accepts only the one bounded
loopback readiness line. Every embedding request uses a globally unique
32-byte nonce and the package-bound Ed25519 response signature.

For each of the seven sequence buckets the collector starts a fresh dispatcher
so RSS from previously compiled buckets cannot masquerade as this bucket's
memory. It runs the pinned semantic triple twice, warmups, and at least 20
measured signed loopback requests. It then starts a separate dispatcher and
executes the same 64 pinned SciFact inputs under every grouping size from 1
through 64. Latency is the end-to-end signed loopback request/response time.
Peak memory is the highest observed Linux process-tree `VmRSS`, sampled every
2 ms across initialization and measured requests. Energy is recorded only
when a future physical meter implementation measures it; the current command
requires a nonempty reason and writes `energy_measurement: not_measured`.

Placement does not trust `cfetch_execution`. Every normal signed response must
also carry live provider evidence. For OpenVINO this is the exact compiled
model `EXECUTION_DEVICES`, exact `core.get_property` values, and exact
platform/kernel/driver-file hashes. Expected and actual values, source APIs,
compile configuration, dispatcher/runtime/probe-manifest identities, and the
raw signed transaction digest remain in the placement summary. Missing,
echoed, mismatched, aggregate, or fallback device claims fail collection.

Run one scope at a time with a 64-input JSON manifest produced from the pinned
SciFact revision and selection named in `WIRE_BATCH_INPUT_SELECTION`:

    python experiments/embedding-profile/scifact_contract.py \
      --wire-inputs-output results/wire-probe-inputs.json

    python experiments/embedding-profile/physical_evidence.py \
      --dispatcher results/openvino-probe/cfetch-openvino-adapter \
      --dispatcher-sha256 "$DISPATCHER_SHA256" \
      --package-manifest results/openvino-probe/package-manifest.json \
      --package-manifest-sha256 "$PROBE_PACKAGE_MANIFEST_SHA256" \
      --scope-id intel-lunar-lake-npu \
      --wire-inputs results/wire-probe-inputs.json \
      --checkpoint-directory results/intel-lunar-lake-npu-checkpoint \
      --energy-not-measured-reason "No device-scoped physical meter was available" \
      --output-directory results/intel-lunar-lake-npu

Physical checkpoints are optional, exclusive working journals. Their identity
binds the exact package, dispatcher, runtime, scope and host policy, input plan,
collector implementation and measurement settings. Each request nonce is
reserved durably before HTTP; each authenticated transaction is persisted
before collection continues. Resume revalidates all retained signed responses
and live evidence before starting a dispatcher. Corrupt or mismatched history
is refused. Completed bucket trials and complete wire groupings are reused only
after regenerating and comparing their summaries. An interrupted bucket starts
a fresh process, warmups and samples; its previous transactions and nonces remain
in the journal. Old warmups are never combined with new cold-process samples.
Reuse recovers the same trial and cannot count as an independent repeat.

The checkpoint and final output directories must be separate. After failure,
resume with the same checkpoint and an unused output path; successful output
publication also refuses overwrite. The journal is retained on success and
failure. A governor's unfinished native operation still requires explicit
external recovery; resuming the collector does not clear safety state or lift
a device quarantine. The 64 wire groupings involve 4,096 actual input
inferences with this singleton native engine, so budget actual calls rather
than HTTP requests. See the [OpenVINO governor contract](../../packages/openvino/README.md).

The output directory contains the exact sequence, placement, and performance
JSON summaries plus only their referenced content-addressed raw signed wire
transactions, profiler records, and benchmark records. Reassemble
`package_state: candidate` with the three
generated summary digests before running `export_adapter_cache.py`. Admission
reconstructs the canonical probe manifest from that candidate by resetting
the state, three evidence bindings, and report binding; its SHA-256 must equal
`placement.provider_binding.probe_package_manifest_sha256`. Therefore the
measured probe and exported candidate may differ only in those bindings.
Admission also verifies the candidate package inventory and native launcher,
projects the reconstructed probe manifest into that verified tree, and requires
the projected probe-launcher SHA-256 to equal the dispatcher measured by the
collector. This binds every other package file across the lifecycle, not just
the parsed manifest fields. The later `release` state adds only the global
report binding, rebinds the inventory and launcher transactionally, and must
pass final package conformance.

## Reproduce the runtime KAT on one host

`kat_host_runner.py` replays the bundle's recorded session contract on any
host and compares the canonical `INT8x768` outputs against the schema-2
known answers. It is the harness behind third-party hardware reports on the
certification issue: passing, mismatching, and blocked results are all
useful evidence.

    python kat_host_runner.py <bundle-dir> <provider> --require-provider

The runner downloads nothing and hides nothing:

- `--require-provider` refuses to compare bytes when the requested provider
  is not ACTIVE — a GPU factory failure otherwise leaves the session silently
  on CPU and measures the wrong route.
- `--strict` sets `session.disable_cpu_ep_fallback`; any CPU remainder then
  fails session creation instead of quietly joining the computation.
- `--no-u8s8` applies the `session.qdqisint8allowed=0` lowering control.
- `--baseline-schema1` compares against the superseded schema-1 known
  answers embedded in `model.onnx.build.json`. This baseline is diagnostic
  only: a host that reports 11/11 there and 0/11 against schema 2 is
  deterministically reproducing the old saturating-kernel bytes and is a
  rejected producer, not a passing one.

Report the exact runtime version(s), the requested and active providers,
per-case byte-diff counts, and whether repeats were byte-identical. On
Windows with ORT 1.28 CUDA, also state how the CUDA 13 dependencies were
sourced; the PyPI `nvidia-cublas-cu13` package is a placeholder and CUDA 12
DLLs make the CUDA EP fail closed into a CPU-only session.

## Export one adapter scope

`export_adapter_cache.py` is the backend-neutral bridge between a target-native
local adapter and the global retrieval gate. It loads only the revision-pinned
SciFact rows, applies the frozen query and document prompts before sending any
text, and calls an attested OpenAI-shaped loopback `/embeddings` endpoint in
batches. The endpoint must return one explicitly indexed, finite, non-zero
768-component float vector per input. The exporter applies the canonical
max-absolute, round-ties-even signed INT8 codec, runs the complete query and
document scope twice, and refuses non-repeatable output.
Every request body carries `cfetch_requested_scope_id`; a response from any
other execution scope is rejected even when its signature is otherwise valid.
Transport is also scope identity: package-owned adapters use
`supervised-local`, while an explicitly configured endpoint is admitted as
`remote-attested`; one cannot impersonate the other by reusing provenance.

An offline host without `datasets` may pass `--scifact-snapshot` naming a
directory with `corpus.jsonl`, `queries.jsonl`, and `qrels/test.jsonl`. The
loader accepts only the frozen raw bytes (SHA-256 `f0d32db0d156b526d75921ed7a76f2cb912902631c87248c5c97c617bad0b60c`,
`9db7df096f7414435d52bafcbacf814c30cea50eb565c1e8fa6d11440759bba8`,
and `da33fc0edc7447d43908ea92bf11de010e361ac43228295f09bdcd33ce14730c`,
respectively) and then enforces the same schemas, counts, IDs, order, and
semantics as the pinned `datasets` path. Local snapshot paths are never stored
in an admission cache.

For example:

    python experiments/embedding-profile/export_adapter_cache.py \
      --endpoint http://127.0.0.1:8080/embeddings \
      --scope-id apple-coreml-ane-example \
      --transport supervised-local \
      --backend apple-coreml-ane \
      --runtime "Core ML 8.0" \
      --compiler "coremltools <version>" \
      --package-target aarch64-apple-darwin \
      --artifact-source "<repo>@<revision>/<file>" \
      --artifact-sha256 "$ARTIFACT_SHA256" \
      --attestation-public-key "$ATTESTATION_PUBLIC_KEY" \
      --internal-precision "int8 weights, fp16 activations/output" \
      --supported-max-tokens 2048 \
      --supported-sequence-bucket 32 \
      --supported-sequence-bucket 64 \
      --supported-sequence-bucket 128 \
      --supported-sequence-bucket 257 \
      --supported-sequence-bucket 512 \
      --supported-sequence-bucket 1024 \
      --supported-sequence-bucket 2048 \
      --sequence-capability-evidence results/apple-coreml-sequences.json \
      --sequence-capability-evidence-sha256 "$SEQUENCE_EVIDENCE_SHA256" \
      --device "Apple Neural Engine" \
      --device-class npu \
      --placement-evidence results/apple-coreml-placement.json \
      --placement-evidence-sha256 "$PLACEMENT_EVIDENCE_SHA256" \
      --performance-evidence results/apple-coreml-performance.json \
      --performance-evidence-sha256 "$PERFORMANCE_EVIDENCE_SHA256" \
      --accelerated-placement \
      --batch-size 32 \
      --output results/apple-coreml-ane.npz

`--accelerated-placement` is an explicit evidence assertion, not automatic
device detection. Supply it only when the native adapter's placement report
proves the named accelerator. An adapter requiring loopback authentication can
read its bearer token through `--bearer-token-env NAME`; the token is not
written to the cache. The admission exporter requires the proposed package's
64-character lowercase hexadecimal Ed25519 public key. It sends a fresh nonce,
verifies the signature over the exact request and response bytes, and records
the public key in cache metadata. One shared nonce register covers the entire
export, so a repeated challenge fails before another request is sent. Unsigned
candidate smoke probes are not admission caches. The exporter refuses
non-loopback hosts, redirects, implicit output paths, and overwriting an
existing cache.

All three evidence files are UTF-8 JSON, are embedded byte-for-byte in the
cache, and must identify the same scope ID, backend, artifact source/digest,
runtime, compiler, package target, internal precision, device, and device
class as the CLI. Sequence evidence contains one result per declared bucket
with equal requested/tokenized/executed lengths, 768-dimensional finite
non-zero output, and `truncated: false`. It also covers every batch size 1
through 64 over the same ordered inputs—the first 32 pinned queries followed by
the first 32 pinned documents—with the exact request count and input/output
digests for each grouping. The exporter reruns all 64 groupings and requires
identical canonical output bytes. It persists every grouping's complete
canonical output in `wire_batch_outputs`; registry replay recomputes each
digest from those bytes, checks the pinned ordered-input digest, and proves all
64 retained outputs are byte-identical. Placement evidence
confirms accelerator execution for every bucket, records all fallback work
explicitly, rejects unexpected fallback, and binds profiler output digests.
Its provider binding retains normalized expected and live host bindings,
compile configuration, dispatcher and runtime identities, and the probe
package manifest digest. Every bucket retains expected and actual execution
devices and device properties plus the exact live API source. The admission
transaction can therefore compare the final manifest to measured facts instead
of trusting only a placement-summary digest.
Performance evidence records sample count, p50/p95 latency, peak memory, and
measured energy—or an explicit reason it was not measurable—separately for
every bucket. Each placement bucket includes `profiler_output_sha256`; each
performance bucket includes `benchmark_output_sha256`.

Every bucket also runs the profile-pinned
`cfetch-sequence-semantic-v1-cat-vs-music` query/relevant/irrelevant triple
twice through the actual adapter. Response token counts must select that exact
bucket, both canonical runs must be byte-identical, and their input/output
digests must match the sequence evidence. The exporter requires the relevant
document to beat the irrelevant document in the self-scope; the global gate
requires the same strict ordering for every ordered query/document scope pair
and for the derive-once relevant-minimum/irrelevant-maximum combination, whose
two records may come from different document scopes.

CLI evidence paths are host-local and are not published. The three validated
JSON summaries are embedded byte-for-byte in the cache under stable `npz:...`
logical locators. Raw profiler and benchmark outputs are retained separately
in the measurement bundle described below, together with every physical-probe
wire grouping's signed request/response transactions.

The exporter never invents sequence capability from the profile. The package
must explicitly record its supported maximum and every supported bucket plus
the evidence location. The global gate loader requires the complete profile set
`32, 64, 128, 257, 512, 1024, 2048`; a fixed-seq256 artifact can produce a
diagnostic cache but cannot pass the profile gate by succeeding on short
SciFact text alone.

Run the compatibility gate with every already-admitted scope, every new scope
named for this admission attempt, and at least one cache from every device
class. Cache labels must equal their embedded scope IDs:

    python experiments/embedding-profile/cross_backend_eval.py \
      --candidate-scope intel-npu-candidate \
      --candidate-scope nvidia-gpu-candidate \
      --candidate-scope xnnpack-cpu-candidate \
      --backend intel-npu-candidate=results/intel-npu.npz \
      --backend nvidia-gpu-candidate=results/nvidia-gpu.npz \
      --backend xnnpack-cpu-candidate=results/xnnpack-cpu.npz \
      --output results/cross-backend-report.json

The evaluator automatically adds every scope already present in
`release/inference-backends.json` to the required set and refuses missing,
unexpected, or identity-mismatched admitted scopes. `passed` means vector
compatibility for that complete set; it is not by itself a package-admission
or hardware-placement verdict. Hash the report's exact bytes and commit it at
exactly `release/admission/<sha256>.json`; a cohort name, mutable `current`
path, symlink, or filename that disagrees with the bytes is rejected. Every
admission updates every admitted registry entry—old and new—to that newest
full-cohort report path and digest.

Reports form a bounded, content-addressed lineage inside the repository. The
genesis report has both `parent_report` and `parent_report_sha256` set to JSON
`null`, has no `already_admitted_scopes`, and names the complete nonempty
initial cohort as `candidate_scopes`. A successor names its parent's exact
`release/admission/<sha256>.json` path and digest. Its
`already_admitted_scopes` must equal the parent's complete cohort exactly, and
its nonempty `candidate_scopes` must be exactly the new current scopes absent
from that parent. Retained backend metadata and cache digests cannot change
across an edge. CI walks newest-to-genesis with depth and total-byte bounds;
missing, rewritten, cyclic, split, shrinking, or internally inconsistent
history fails closed.

That update is an atomic release boundary. Every later cohort requires every
previously published target package to be rebuilt and republished so its
signed production response carries the new global report digest. The new
packages and the registry that binds all entries to that one path/digest pair
must roll out together. A package that still attests an earlier cohort report
is stale and must no longer be accepted as current; mixed old/new package
generations are not a valid admitted cohort.

## Durable admission replay

An admitted scope has two immutable cfetch release assets:

- `<admission_cache_sha256>.npz` is the exact exporter cache;
- `<measurement_evidence_sha256>.zip` retains every raw signed wire transaction,
  profiler output, and benchmark output referenced by the embedded summaries.

Both registry URLs must be credential-free
`https://github.com/corbet-libs/cfetch/releases/download/<tag>/...` locators.
The measurement ZIP contains only `measurement-manifest.json` and
`raw/<sha256>.bin` members. Its manifest has schema version 1, the scope ID, the
sequence, placement, and performance summary digests, and one
`{path, sha256, roles}` entry per raw file. Roles are
`wire-signed-transactions`, `placement-profiler`, and/or
`performance-benchmark`. Every referenced wire-grouping and per-bucket digest
must be present; unreferenced files and duplicate JSON keys are rejected.

CI runs:

    python experiments/embedding-profile/cross_backend_eval.py \
      --verify-release-registry

With no admitted scopes this is a network-free successful no-op. Otherwise it
downloads every exact cache and measurement bundle with per-file and cohort
byte bounds and a 64-scope ceiling, verifies their SHA-256 values, rejects
unsafe ZIP/NPY members before NumPy allocation, validates all three embedded
JSON schemas and registry bindings, and walks every committed report from the
newest digest-named file to genesis. Each lineage node is rebuilt from its
exact cache subset: the full `n x n`, adversarial mixed-store, per-bucket,
persisted wire-batch, and combined gate decisions are replayed. Integer
results, identities, and verdicts compare exactly. Only the three recomputed
floating quality values permit a `1e-12` absolute replay tolerance for
platform-library last-bit drift. Optional cross-backend byte/cosine
observations remain local debugging data outside the report and admission.

This makes report arithmetic and retained bytes independently replayable. It
does not turn CI into silicon attestation: whether the raw profiler output was
captured honestly on the named physical device, or whether a vendor tool's
summary correctly describes that device, remains the reviewed runner,
profiler, and measurement-provenance trust boundary. Generic CI validates the
summary schema, bindings, retained raw-output digests, and report arithmetic;
it cannot vendor-neutrally derive placement, latency, memory, or energy claims
from arbitrary profiler formats.

The exporter cannot include `compatibility_report_sha256` in its challenged
responses because that report is created only after all cohort caches exist.
Candidate nonces and signatures are generated and verified live by the
exporter, but the nonce, raw request/response bodies, and signatures are not
persisted in the cache and therefore are not replayed by registry CI. The
cache persists the verified public key and exact semantic outputs, not a claim
that a past challenge can authenticate a future package.
After admission, packaging injects the final report digest into the production
adapter response/manifest without changing the evaluated artifact, runtime,
compiler, package target, public key, or scope identity. The packaged adapter
then runs the bounded final conformance replay before release:

    python experiments/embedding-profile/final_package_conformance.py \
      --endpoint http://127.0.0.1:8080/embeddings \
      --cache results/<scope>.npz \
      --cache-sha256 "$CACHE_SHA256" \
      --compatibility-report-sha256 "$COMPATIBILITY_REPORT_SHA256"

The command validates the exact cache container and digest before allocation,
requires the cached package key, exact requested scope, full execution
identity, and injected compatibility-report digest on every response, and
issues a distinct signed nonce challenge for every adapter call. It replays
the same 64 inputs under grouping sizes 1 through 64 and each of the seven
sequence-bucket fixtures twice, then byte-compares every canonical INT8 record
with the retained cache arrays. The endpoint form above is candidate debugging
only and cannot emit an activation receipt. Receipt mode, documented below,
extracts and launches the exact staged package itself.

The admission implementation is byte-bound separately from the semantic and
policy identities. Its domain-separated digest covers the exact paths and
bytes of the evaluator, evidence validator, exporter, measurement builder,
stage/activation transaction, final-package conformance launcher, dataset
contract, explicit activation source promoter, direct requirements, and fully
hashed transitive lock. That digest
is pinned outside the bundle, recorded in the registry, and embedded in every
report. Run:

    python experiments/embedding-profile/cross_backend_eval.py \
      --verify-implementation-bundle

The command recomputes the framed bundle hash and checks both pins. This makes
the exporter/evaluator implementation used for admission an exact-byte
identity rather than an unverified script name or mutable environment. It is
not the target adapter package identity: the cache separately binds that
scope's native artifact digest and detailed runtime/compiler/device identity,
and final package conformance remains required.

The committed report contains only an allowlisted projection of public
registry-bound metadata, never local evidence paths or unknown backend extras.
Production adapters must attest the final compatibility-report digest.

Admission installs `requirements-lock.txt` with `--require-hashes`; its direct
roots remain documented in `requirements-test.txt`. The shared
dataset loader checks the revision-pinned `mteb/scifact` contract has 300 test
queries, 339 positive qrels, and 5,183 ordered documents before export or
evaluation.
Ranking uses the same exact signed-INT8 dot products, norm cross
multiplication, sign ordering, and insertion-order tie break as production
Rust; floating-point BLAS scores are not part of admission.
Model execution and hardware placement remain the responsibility of each local
target-native adapter using the artifacts listed in
release/inference-backends.json; the exporter deliberately contains no vendor
runtime code.

## Hermetic stage and activation

`admission_transaction.py` is the only supported bridge from candidate caches
to a nonempty proposed registry. Its JSON manifest names the exact base
registry digest, a fixed GitHub release tag, the complete retained-plus-new
cohort, and every target-native package:

Create the transaction-only receipt key once; the command refuses an existing
path and prints the raw public-key hex for the manifest:

    python experiments/embedding-profile/admission_transaction.py keygen \
      --output results/receipt-attestation.key

    {
      "schema_version": 1,
      "base_registry": "../../release/inference-backends.json",
      "base_registry_sha256": "<exact digest>",
      "base_variants": "../../release/variants.json",
      "base_variants_sha256": "<exact digest>",
      "release_tag": "admission-v0.9.10-intel-lnl-1",
      "receipt_attestation_public_key": "<raw Ed25519 public key hex>",
      "scifact_snapshot": "inputs/scifact",
      "candidate_scopes": ["intel-lnl-npu", "intel-lnl-gpu", "intel-lnl-cpu"],
      "scopes": [
        {
          "scope_id": "intel-lnl-npu",
          "admission_cache": "results/intel-lnl-npu.npz",
          "admission_cache_sha256": "<exact digest>",
          "raw_measurements": "results/intel-lnl-npu-raw"
        }
      ],
      "packages": [
        {
          "package_id": "linux-openvino-lunar-lake-x86_64",
          "release_variant_id": "linux-cfetch-local-intel-lunar-lake-x86_64",
          "os": "linux",
          "arch": "x86_64",
          "device_families": [
            "intel-lunar-lake-npu",
            "intel-arc-140v",
            "x86-64-avx2"
          ],
          "ordered_scope_ids": ["intel-lnl-npu", "intel-lnl-gpu", "intel-lnl-cpu"],
          "package_directory": "packages/linux-openvino-lunar-lake-x86_64",
          "package_manifest": "package-manifest.json",
          "package_format": "zip",
          "dispatcher": {
            "binary": "cfetch-openvino-adapter",
            "sha256": "<exact digest>"
          }
        }
      ]
    }

The abbreviated arrays above must contain the full cohort in a real manifest.
All paths are normalized relative paths below the manifest directory. Inputs
and package trees containing symlinks, special files, missing digests, path
escapes, unbounded containers, stale cache identities, missing hardware
classes, or non-NPU/GPU/CPU package order are rejected. The dispatcher must be
an executable regular non-symlink file at the package root, named by a plain
basename, and its exact digest is rechecked inside the final package archive.
Each package must also bind one unique `backend: local` entry from the exact
hash-pinned release-variant catalog with matching OS, architecture, and a
different cfetch binary basename.

For the actual admission boundary, run the complete local transaction as one
command on the physical package host:

    python experiments/embedding-profile/admission_transaction.py run \
      --manifest results/admission-transaction.json \
      --receipt-attestation-private-key results/receipt-attestation.key \
      --output results/complete-admission

`run` first replays the current release registry. Existing admitted assets are
downloaded only from their exact credential-free GitHub HTTPS locators; every
redirect remains on GitHub, and byte bounds plus SHA-256 are checked before
parsing. The required `scifact_snapshot` input is resolved below the manifest
directory, rejected if it or a member is a symlink, and checked against the
three frozen raw-file digests before staging. `run` then stages the complete
local cohort, launches every exact staged package/scope pair for final
conformance using that snapshot, creates the complete activation bundle, and
emits a digest-named `*.publication.json` plan. The plan binds the fixed release
tag, every content-addressed upload and URL, the repository changes, and their
required order. The command is atomic with respect to its new output directory.
It neither uploads nor edits the checkout, and no host snapshot path enters the
publication output.

The lower-level commands below remain useful for inspecting a failed stage.
Run the offline stage into a new path:

    python experiments/embedding-profile/admission_transaction.py stage \
      --manifest results/admission-transaction.json \
      --output results/staged-admission

Staging replays the full gate, builds each deterministic raw-measurement ZIP,
writes the digest-named report, injects that report digest into a copy of each
package manifest, builds digest-named target-package ZIPs, and emits a proposed
registry plus `stage-plan.json`. It does not edit the checked-out registry,
upload an asset, create a tag, or publish a release. Reusing an output path is
an error.

Exercise every package/scope pair from the exact staged package bytes and ask
the final conformance command for a content-addressed receipt:

    python experiments/embedding-profile/final_package_conformance.py \
      --cache "results/staged-admission/assets/$CACHE_SHA256.npz" \
      --cache-sha256 "$CACHE_SHA256" \
      --compatibility-report-sha256 "$REPORT_SHA256" \
      --stage-id "$STAGE_ID" \
      --stage-plan results/staged-admission/stage-plan.json \
      --package-id linux-openvino-lunar-lake-x86_64 \
      --package-asset "results/staged-admission/assets/$PACKAGE_SHA256.zip" \
      --package-sha256 "$PACKAGE_SHA256" \
      --dispatcher cfetch-openvino-adapter \
      --receipt-attestation-private-key results/receipt-attestation.key \
      --receipt-directory results/receipts

Receipt mode forbids a caller-supplied endpoint. It safely extracts the exact
staged ZIP, re-hashes and launches its stage-pinned dispatcher, accepts only its
strict loopback readiness record and scope order, and stops and re-hashes the
package after the live replay. The receipt binds the stage, cache, newest
report, package bytes, implementation bundle, all 64 wire groupings, all seven
buckets, and all 351 fresh signed requests. It is signed by the separate raw
Ed25519 transaction key whose public half is pinned in `stage-plan.json`; the
private key is never placed in the target package or activation bundle.
Activation requires exactly one valid signed receipt for every pair declared
by the stage:

    python experiments/embedding-profile/admission_transaction.py activate \
      --stage-plan results/staged-admission/stage-plan.json \
      --receipt results/receipts/<sha256>.json \
      --output results/release-ready-admission

The command still performs no external mutation. It verifies every staged byte
again and creates a release-ready directory containing `assets/`, `release/`,
the receipts, and a digest-named activation manifest. That manifest binds the
current registry and variant-catalog digests plus the exact one-line
`src/embedding_profile.rs` candidate-to-active transformation. A publisher can
upload the assets to the already-selected fixed GitHub tag before applying the
source promotion. This order removes the old tag/CI deadlock: the nonempty
registry never points at release assets that do not exist yet.

After those immutable assets exist at the bound tag, apply the complete local
promotion explicitly from the repository root:

    python scripts/apply_admission_activation.py \
      --activation-manifest results/release-ready-admission/<sha256>.activation.json \
      --repository .

The dependency-light command verifies the activation filename and complete
bundle inventory, every content hash, the checkout's still-current base
registry and variant catalog, the nonempty active registry/report binding, and
the exact candidate source digest before writing. Before any checkout mutation,
it also downloads every planned release asset from the fixed tag through
credential-free GitHub HTTPS, constrains every redirect to GitHub, and verifies
the exact byte count and SHA-256. It then copies the report and registry and
changes only the hash-bound `PROFILE_STATUS` constant. Any schema, remote
asset, byte, or checkout drift is a hard error with a nonzero exit; success
prints a machine-readable summary. It does not upload assets or publish a
release.

## Activation blockers

The two-phase transaction, fully hashed dependency lock, and receipt-producing
final conformance command are implemented and covered by synthetic fail-closed
tests. They are machinery, not evidence. Before the first nonempty cohort:

- Capture the complete physical NPU, GPU, and accelerated-CPU caches plus raw
  placement/performance outputs, then run the stage command successfully under
  the hash-locked environment.
- Run the one-command transaction on the physical package host so every exact
  staged package/scope pair produces its final-conformance receipt and the
  immutable activation/publication bundle.
- Publish the content-addressed assets from that plan at the fixed tag, run the
  bound activation-application command (which downloads and verifies them
  again), and pass the nonempty registry replay in normal CI.

`admitted_backends` and `local_packages` remain empty. No target scope or local
producer is admitted until both boundaries and a real complete cohort pass.

The complete first-cohort operator sequence, including the frozen-runtime
host-preflight boundary, is in `OPENVINO_OPERATIONS.md`. Do not substitute
guessed host properties or a diagnostic run for physical input.
