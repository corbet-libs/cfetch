# Synthetic memory retrieval: exact inputs, context, and mixed producers

The measured candidate benefits strongly from heading/table context. The
body-only representation loses information needed to distinguish short
statements. Context restores top-five retrieval on this fixture, while
current-versus-obsolete policy ranking still needs work. These results support
continuing representation qualification before backend activation.

## Measurement

The [fixture and reproduction commands](../experiments/memory-retrieval/README.md)
use 32 invented documents, 105 indexed blocks, and 32 authored queries. There
are six paraphrase, eight context, six critical policy, eight multilingual,
and four exact queries. These labels were designed with the fixture; they are
not held-out judgments or representative real-user relevance evidence.

Rust exports the actual segmenter's full prefixed inputs, then imports
candidate vectors into an isolated catalog. Ranking uses cfetch's real signed
INT8 codec and rankers, default RRF `k=2`, and full denominators including
refusals. The corpus's 84 distinct body inputs become 105 distinct document
inputs with context. Both representations share 32 query inputs.

CPU measurements use the pinned community EmbeddingGemma ONNX FP32 and Q4
artifacts recorded in the runner, one input per actual inference, smallest
bucket, and no truncation. The experiment does not establish those artifacts'
lineage to cfetch's canonical source checkpoint. It is not NPU/GPU evidence.

For each representation, four ordered query/document producer pairs and four
mixture cases were replayed: alternating sorted document-input producers and
their complement, each with both query producers. No case refused an input.
The [machine-readable results](memory-retrieval-2026-09-08.json) retain all
category scores, critical query failures, exact manifest/source bundle/report
hashes, runtime provenance, and the frozen cfetch executable digest. Source
`first` means FP32 and `second` means Q4. The comparison script checks its
executable identity before and after every replay.

## Results after the lexical-fusion correction

The following rows use the same artifact for query and document production.

| Representation / artifact | Vector nDCG@10 | Vector Recall@5 | Hybrid nDCG@10 | Hybrid Recall@5 |
|---|---:|---:|---:|---:|
| Body / FP32 | 0.7942 | 0.8125 | 0.7258 | 0.9062 |
| Body / Q4 | 0.7903 | 0.8125 | 0.7268 | 0.9062 |
| Heading context / FP32 | 0.9654 | 1.0000 | 0.8486 | 1.0000 |
| Heading context / Q4 | 0.9654 | 1.0000 | 0.8527 | 1.0000 |

Standalone lexical recall is identical across representations: nDCG@10
0.5082 and Recall@5 0.6562. For the eight context queries, FP32 vector Recall@5
rises from 0.25 to 1.0. Paraphrase and multilingual retrieval already benefit
from body-only vectors; the principal new gain is contextual disambiguation.

Across all eight producer/pair/mixture cases, heading context retains vector
and hybrid Recall@5 of 1.0. Vector nDCG@10 ranges from 0.9308 to 0.9654;
hybrid nDCG@10 ranges from 0.8214 to 0.8527. These fixed mixtures are diagnostic,
not the admission evaluator's query-specific adversarial document selection.

Policy rank one remains a failure: lexical recall puts the labeled current
policy first in all six critical queries, while context vector/hybrid ranking
does so in only four or five queries depending on producer combination. The
correct policy remains in the top five. Retrieval rank must not be interpreted
as resolving a contradiction against the cited ring's authority.

## Defects exposed and corrected

The original statement key normalized case and whitespace even though the
model consumed raw body text. Different inputs could therefore reuse one
vector. Exact, domain-separated body hashing now distinguishes them. Schema 8
rebuilds disposable catalogs; legacy normalized keys cannot hydrate new vectors.
Tests cover case, spacing, line breaks, incremental/full invalidation, retained
old shared records, and deduplication of exact copies across rings.

Hybrid also consumed raw BM25 candidate order while standalone lexical recall
reserved its first slot for a top-trust hit. This could put an obsolete policy
first even when the displayed lexical and semantic rankings both preferred
the current policy. Both lexical paths now share the existing reservation.
On context/FP32, critical hybrid rank-one hits improve from two to five of six;
the remaining failure is not removed by that correction. The result table and
machine-readable file use the corrected ranker. Pure semantic ranking and the
production semantic profile are unchanged.

## What the evidence permits next

Use heading/table context as the next representation candidate and validate it
on a larger, independently labeled memory set. Production adoption needs
explicit identity and invalidation rules for context-only edits; this experiment
does not silently attach contextual vectors to production body-only keys.
Citation identity and rendered model-input identity should remain deliberate
contracts rather than accidental consequences of one database column.

Keep the current model candidate while testing that representation. This small
fixture does not justify a model replacement or relaxed admission floors.
Policy ordering, overlength statements, edit/reunion behavior, and realistic
latency remain separate measurements. Fixed producer mixtures do not prove
network convergence or the complete adversarial gate.

Canonical-source/native artifacts still need their full ordered-pair SciFact,
sequence, repeatability, package, placement, performance, and governed endurance
evidence. Resumable exports and bounded dispatcher cleanup are useful pieces;
they do not provide a host-wide per-actual-inference governor or make a wedged
accelerator recoverable. No backend or production profile was activated.
