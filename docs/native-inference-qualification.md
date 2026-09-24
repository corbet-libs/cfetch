# Native inference qualification

The `native-openvino` Linux feature provides an owned OpenVINO child and a
persistent host governor. It is a qualification interface, not an admitted
production backend. Enabling the feature does not select a model, change the
working CPU backend, write vectors, or populate `release/inference-backends.json`.

## Execution contract

`cfetch native-probe PLAN EVIDENCE --policy-sha256 SHA256` consumes one finite
JSON plan: schema version 1, one compile command followed by 1–64 inference
commands. The plan envelope remains schema 1; each command and response uses
internal protocol schema 3 with structured failure kinds, exact device properties,
and the full runtime/plugin/host closure. Older protocol
commands/checkpoints are rejected. Every command has the same model, pipeline, output, dimensions,
runtime build, and requested device. A plan has one static token bucket and
batch size one. The hidden `native-worker` command is its internal child entry
point; callers use the parent probe so the governor and checkpoint contract
cover every native operation.

The parent validates bounded regular-file input before acquiring a lease. It
requires inherited cgroup-v2 limits of no more than two CPUs and 8 GiB memory.
The child starts with host-thread limits of two, arms a parent-death kill, and
checks parent identity. Starting the child or reaching EOF loads no native
library. Runtime loading, model reading, compilation, and synchronous inference
begin only after the parent has durably charged a governor lease and sent the
validated operation. Responses identify the exact request and actual execution
device. Protocol replies use a private close-on-exec descriptor; vendor stdout
is redirected to stderr before loading native code. Another device, malformed
identity, nonfinite output, wrong dimensions,
or invalid normalization fails closed.

Before starting a probe, the operator must provision the root-owned
`/var/lib/cfetch/inference` namespace explicitly. Schema-2 policy uses namespace
`cfetch-host-inference-v2`, an explicit device allowlist, finite compile/inference
operation and bucket budgets, bounded operation deadlines, and cooldown at
least as long as active work. Its SHA-256 is passed to the probe. Root owns the
namespace and policy; only the intended runner may update the provisioned
lock, state, and intent files. The runtime never creates or resets this policy.
A new boot, changed epoch or policy, exhausted budget, or pending intent requires
operator inspection. A CPU/GPU test policy must omit NPU.

Each operation retains the permanent host lock across the request and bounded
child supervision. Request intent is fsynced before native work. The parent
fsyncs an exact response checkpoint and its directory before committing usage
and clearing intent. `NativeFailure::Controlled` requires an explicit
`scope_unavailable` classification, exact identity, durable error checkpoint,
confirmed owned-child death and successful lease completion. Its payload cannot
be constructed by callers. The current worker has no proven typed vendor absence
signal, so all actual native/runtime/model errors are `hard_stop`; error text
never authorizes fallback. The reserved controlled path is exercised only by
injected tests. An already exited/reaped worker, unknown or missing error kind,
timeout, crash, protocol mismatch, checkpoint failure, failed completion, or
uncertain child termination leaves intent pending and blocks another scope. Fallback cannot
clear that evidence. A restarted finite plan can reuse a successful inference
checkpoint only for byte-identical request identity; its new worker still pays
a compile lease. Completed error checkpoints are not retried automatically.

## Artifact and runtime prerequisites

Every compile request specifies an absolute runtime-library path and digest;
the safe vendor loader uses that exact library instead of ambient discovery.
The command also pins model and optional named weights bytes. Runtime build and
requested precision are checked or recorded separately; a build string or
precision hint does not prove the identity or arithmetic of every plugin.

Before physical execution, separately audit all external model-data references,
tokenizer files, prompts, pooling, and the complete runtime/plugin dependency
closure. Retain exact hashes and immutable paths in the qualification record.
The named-weights check alone does not validate arbitrary ONNX external-data
references. The Python artifact audit in `experiments/embedding-candidates`
checks that closure for registered candidates without loading an inference
runtime. It is tooling for artifact inspection, not the production runner.

CPU/GPU diagnostics do not authorize NPU execution. Host-specific quarantine and
physical-readiness requirements remain effective after unit tests or artifact
audits pass. Model support in vendor documentation is not admission on an
untested device. EmbeddingGemma requires float32 or bfloat16 activations; do not
request float16 merely because the generic worker can represent that hint.

## Production admission

Keep the existing profile and working CPU path until a replacement is selected
and passes end-to-end qualification. Model selection must freeze model revision,
tokenizer, query/document instructions, sequence limit, pooling, normalization,
dimensions, and index encoding. Changing dimensions or vector meaning requires
a distinct profile and deliberate index migration.

Admission requires numerical/retrieval comparisons on representative multilingual
text and code, physical device placement, bounded compile and inference latency,
peak memory and energy evidence, crash/fallback tests, and endurance under the
same governor. Device order is qualified NPU, qualified GPU, then qualified CPU;
all routes must produce the same profile. Linux Intel OpenVINO code does not
establish AMD, Apple, ARM, mobile, or Windows support. Those platforms need their
own native export/runtime package and evidence before registry admission.

Focused Crow selectors are `native-governor` and `native-worker`. They exercise
pure validation, persistent state, owned-child supervision, worker framing and
pooling, and startup without native libraries. They perform no physical model
inference and do not lift a host quarantine.

## Rust package-local serving integration

`native_serving.rs`, `native_manifest.rs`, and `native_http.rs` implement the
existing supervised local producer contract in Rust. The final client validates
its compiled package plan before spawning the independently built native
sibling; a strict private startup permit binds the exact manifest and ordered
scopes. The sibling revalidates that closure, serves authenticated loopback HTTP,
and signs responses with the bound scope key. See
[native package schema 2](native-package-v2.md) for the acyclic build contract.

The candidate registry remains empty. This code does not admit a backend, switch
the active model/index, provision a governor, or qualify any hardware. Native
schema 3 probes require explicit new plans; old checkpoints are not replayed
through compatibility defaults. All present vendor errors are hard failures;
controlled unavailability remains reserved for a future proven typed origin.

The request core checks exact Gemma tokenizer bytes and BOS/EOS semantics,
prevalidates the whole batch, selects the smallest of all seven declared sequence
buckets, and rejects overlength input without truncation. Rows are grouped by
bucket with original output positions preserved. One worker caches one exact
scope/runtime/compile recipe/bucket; changing it requires confirmed child death.
Every compile and inference acquires the host governor. An uncertain operation,
invalid vector/identity, transport failure, or unexpected child exit latches all
later scopes, including CPU. Only a settled controlled failure can advance the
existing NPU → GPU → CPU selection policy.

Fixture tests cover signed response identity/order, tokenizer and bucket rules,
whole-batch refusal, bounded compile counts, cleanup/retained intent, strict
package inventories, policy budgets, supervised HTTP limits/EOF/deadlines, and
outer transport failures without CPU fallback. They do not prove a real model's
numerics, physical placement, all-bucket performance, all-platform support, or
the global ordered-pair/mixed-store admission cohort. Those remain mandatory
before a release package can enter the compiled registry.
