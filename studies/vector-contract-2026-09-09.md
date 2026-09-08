# Canonical vector contract and contextual retrieval — 2026-09-09

The native exporter had a long-input attention defect. Correcting it brought
OpenVINO CPU embeddings into agreement with an independently assembled upstream
reference, with a maximum component difference below 2e-7 in the six measured
inputs. Contextual document payloads also recovered missing retrieval context.
These results establish a useful CPU candidate; the profile remains unadmitted.

The [machine-readable evidence](vector-contract-2026-09-09.json) records source,
artifact, implementation, executable, input, output and report hashes, both
failed and corrected audits, and all sixteen retrieval comparisons. Model
conversion and inference ran in an isolated Linux CPU container. No accelerator
was mounted and no device safety policy was installed.

## Encoder defect and independent check

The pinned Google source revision is
`57c266a740f537b4dc058e1b0cda161fd15afa75`. All thirteen required source files
were verified against the package's canonical hashes. The raw configuration
contains `sliding_window: 512`. In the pinned Transformers implementation,
`Gemma3TextConfig.__post_init__` changes this to `512 // 2 + 1 = 257` for
bidirectional attention. The exporter incorrectly used raw 512 as the exclusive
attention radius instead of the effective 257, allowing attention beyond the
upstream 256-token radius.

Short inputs could not reveal this: the first affected real length is 258.
The old smoke reference reused the exporter's patched masks and therefore
shared the error. On a 1,946-token input, an independent upstream reference
exposed cosine similarity 0.986876805 and maximum component error 0.0189059.
The failed artifact and audit were retained. No acceptance threshold was relaxed.

The converter now requires both the canonical raw value and the effective
loaded configuration, and builds masks with effective 257. A stale pair of
rotary MatMul names was also updated for the locked exporter, retaining exact
node names, dimensions, types and match counts; unrelated nodes remain intact.

The independent reference loads a fresh upstream model with its ordinary 2D
attention mask, then independently applies masked mean pooling, the two pinned
dense projections, and L2 normalization. It does not use the exporter's mask or
pooling patches. The audit checked all 26 selected real-length/padding cases
across buckets 32, 64, 128, 257, 512, 1024 and 2048. Six numerical probes passed,
including the 1,946-token input: minimum upstream/native cosine was
0.9999999999988822 and maximum absolute component difference was
1.9744038581848145e-7. The package smoke gate now uses this independent reference
and includes actual 258- and 1,946-token inputs, in addition to existing probes.

The corrected artifact manifest hash is
`1c79b0e67ec4e0b4ac93e66f05cdf7a4c54647b2bf87e235be3d1b4d6ab6775c`.
The rejected artifact hash is
`ba6f18c3e766f6012a0edc5d7a55da4f551c736c7c194e3a32d45a58e3f56bbe`.
The sequence bucket 257 and this effective attention parameter have different
roles. This correction does not explain or clear the previously observed
device failure at sequence shape 256.

## Context and producer comparisons

Production documents now render full heading/table context, two newlines, and
the exact atomic body under the unchanged `title: none | text: ` prefix.
The versioned payload hash identifies that context-sensitive input; citations
continue to identify the body. Identical bodies under different headings or
table headers no longer share vectors or collapse into one mirror result.
Context changes invalidate the affected vector, while identical complete
payloads still deduplicate. The semantic encoder profile and INT8 codec remain
unchanged.

The original synthetic fixture has 32 queries. A separate 36-query fixture was
authored before this measurement, with exact identifiers, multilingual cases,
current-policy decoys and long atomic paragraphs. Four paragraphs were
soft-wrapped to satisfy the production generated-line limit before export;
their content and labels were unchanged. Both fixtures ran through production
segmentation, citation handling, INT8 storage, vector ranking and RRF.

Each fixture used all four ordered query/document producer pairs plus two
fixed alternating document mixtures with each query producer. The first
producer was corrected canonical OpenVINO CPU f32; the second was the pinned
community ONNX FP32 CPU artifact. Results below show the pure native pair;
the evidence records every comparison.

| Fixture | Mode | Recall@5 | nDCG@10 |
| --- | --- | ---: | ---: |
| Original, 32 queries | BM25 | 0.65625 | 0.50823 |
| Original, 32 queries | Vector | 1.00000 | 0.95387 |
| Original, 32 queries | Hybrid | 1.00000 | 0.84232 |
| Holdout, 36 queries | BM25 | 0.83333 | 0.54781 |
| Holdout, 36 queries | Vector | 0.97222 | 0.89021 |
| Holdout, 36 queries | Hybrid | 0.97222 | 0.85118 |

All eight producer cases per fixture retained these Recall@5 values. The one
holdout miss was `long-courtyard-end`: its document requires 2,144 prefixed
tokens and was explicitly refused above the frozen 2,048-token limit. No text
was silently truncated. Native/native holdout ranking improved over the
community/community pair, whose vector nDCG@10 was 0.84556.

Critical policy ordering remains unresolved. BM25 placed all six current-policy
answers first in each fixture; vector and hybrid placed five of six first in
the original fixture and four of six in the holdout. Remaining failed query IDs
are `policy-delete-next-batch`, `policy-pass-reuse`, and
`policy-cancellation-store-credit`. High top-five recall does not establish
trust-aware first-result correctness.

Canonical and community tokenizer inputs were equivalent for all 130 holdout
inputs despite different tokenizer JSON hashes. However, the three runnable
long document vectors had cross-producer cosines only 0.90066–0.92733. Short
inputs exceeded 0.99996. The community artifact therefore has not established
canonical encoder equivalence. Retrieval scores do not admit it, and these
observations alone do not identify its internal graph defect.

## Runtime prerequisites and remaining gates

The OpenVINO adapter now requires a provisioned host-wide governor before
operational native startup. Separate leases govern every actual compilation
and inference. Durable intent, finite operation and bucket budgets, cooldowns,
and kernel deadline termination survive process interruption without silently
resetting the budget. The exact installed policy is bound through the existing
host-file evidence. Uncertain cleanup and systemic transport failures stop or
advance to a different scope without repeating the failed scope. See the
[package instructions](../packages/openvino/README.md).

These are software controls, not measured accelerator safety limits. A process
alarm cannot recover a wedged SoC. No local policy or certified endurance limit
was invented, and the quarantined device remains quarantined.

Rust tests also exercise offline holders deriving different valid vector bytes,
context changes, peer-artifact wire record round trips, first-record retention,
and catalog reunion. Those tests use synthetic canonical vectors and do not
constitute a live peer or physical backend qualification.

Remaining acceptance work includes independent repeatability under each exact
physical scope, all SciFact producer pairs and adversarial mixtures, sequence
and grouping checks, measured governor endurance, final package evidence,
critical-policy ordering, and a decision on oversized atomic documents. The
admitted-backend and local-package registries remain empty. Neither these
synthetic fixtures nor the corrected CPU audit permits profile activation.
