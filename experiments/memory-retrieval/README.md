# Memory retrieval qualification

This diagnostic measures labeled Markdown through cfetch's production
segmenter, citation handling, signed INT8 codec, lexical ranking, semantic
ranking, and reciprocal rank fusion. It opens a temporary catalog and never
loads the configured brain, calls its endpoint, or writes shared vectors.

`corpus.json` contains 32 synthetic documents and 32 labeled queries: six
paraphrases, eight context-dependent queries, six current-policy queries with
obsolete decoys, eight English/German/French queries, and four exact lookups.
Repeated short statements and table rows deliberately need their surrounding
headings or table headers. These invented examples reveal failure mechanisms;
they are not a representative sample of a real brain or a held-out benchmark.

## Export exact inputs

Run a locally built CLI in an isolated compute environment:

```sh
cfetch retrieval-eval --corpus experiments/memory-retrieval/corpus.json \
  --representation body --export body-manifest.json > body-lexical.json
cfetch retrieval-eval --corpus experiments/memory-retrieval/corpus.json \
  --representation heading-context --export context-manifest.json > context-lexical.json
cfetch retrieval-eval --corpus experiments/memory-retrieval/holdout.json \
  --representation context-payload --export payload-manifest.json > payload-lexical.json
```

`body` uses the candidate's existing fixed document prefix and exact statement
body. `heading-context` is an experimental representation: the segmenter's
heading and table-header context fills the `title` field. The manifest records
every full prefixed input, original citation, body identity, context, token
buckets, and the production RRF setting. Context changes affect experimental
vector identity while preserving the original citation. Production profile
semantics and citations are not changed by selecting this diagnostic mode.

`context-payload` uses the production document renderer: full heading/table
context, two newlines, then the exact statement body, under the unchanged
`title: none | text: ` prefix. The renderer has its own payload-hash domain;
the encoder profile and body citations keep their separate identities.
Table context uses the full source header, including text beyond the preview.
`holdout.json` adds 36 queries authored before measurement, including long
atomic paragraphs, exact identifiers, and unrelated ring-0 decoys. Paragraphs
are soft-wrapped to remain prose under the production generated-line limit.
These synthetic labels are not representative real-user judgments.

Labels must identify exactly one segmented block. Path aliases and duplicate
labels for one mirror class are rejected. Corpus size, document count, query
count, and indexed block count are bounded. Export refuses to overwrite files.

## Optional CPU candidate measurements

`embed_cpu.py` requires Linux and Python 3.12+. The tested environment uses
NumPy 2.5.2, ONNX Runtime 1.28.0, and tokenizers 0.22.2; installed versions are
recorded in provenance rather than enforced. Use an isolated environment. It
accepts existing local files from the pinned community model revision recorded
in the script and verifies every graph, external weight, tokenizer, and model
configuration digest it consumes. It does not download or substitute a model.

```sh
python experiments/memory-retrieval/embed_cpu.py \
  --manifest body-manifest.json --model-root /path/to/model \
  --artifact fp32 --checkpoints body-fp32-work --output body-fp32-vectors.json
cfetch retrieval-eval --corpus experiments/memory-retrieval/corpus.json --representation body \
  --vectors body-fp32-vectors.json > body-fp32-report.json
```

Repeat with a separate output and checkpoint directory for each representation
and `fp32`/`q4` artifact. The runner uses CPU execution only, one actual input
per call, four compute threads, no spinning, and a pause between calls. Each
input uses its smallest frozen bucket; overlength input is explicitly refused.
These CPU diagnostic limits are not certified accelerator safety limits.

Interrupted work resumes under an exclusive directory lock. Identity binds
the exact input manifest, runner bytes, runtime versions/options, and artifact
digests. Each completed record also binds that experiment identity. Completed
files are published atomically without replacement; orphaned records cannot be
adopted by another model run. Resume is working-state recovery, not an
independent repeatability trial.

The Rust importer checks complete input accounting, exact manifest identity,
token bucket selection, refusals, dimensions, finite values, and normalization.
It applies the real INT8 codec once. Refused queries count as zero in the full
denominator; a corpus without document vectors cannot report lexical fallback
as successful hybrid measurement. Reports include nDCG@10, Recall@5, MRR@10,
per-query rankings, coverage, and source bundle hashes.

## Compare producers without new inference

```sh
python experiments/memory-retrieval/compare.py --cfetch /path/to/cfetch \
  --corpus experiments/memory-retrieval/corpus.json \
  --representation body --manifest body-manifest.json \
  --first body-fp32-vectors.json --second body-q4-vectors.json \
  --output-directory body-comparison
```

This replays all four ordered query/document producer pairs and two fixed
alternating document mixtures with both query producers. Saved source hashes
and per-input assignments make every mixed bundle reviewable. It does not
simulate peer transport, claim offline convergence, or replace the adversarial
mixed-document admission gate.

## Interpretation

The community ONNX artifacts have not established lineage to cfetch's canonical
source checkpoint. Their CPU output is candidate evidence only. No result from
this fixture admits a backend, activates a profile, validates NPU/GPU placement,
or certifies cross-backend repeatability. The context-payload renderer preserves
the frozen encoder function and versions its document input independently.
Backend activation still requires complete admission.

`audit_native.py` checks the canonical tokenizer, upstream attention masks at
all seven buckets, and independently assembled upstream embeddings against
the converted OpenVINO CPU artifact. `embed_native.py` then produces compatible
resumable vector bundles from verified canonical source/artifact bytes, using
the OpenVINO build lock. Both require an isolated Linux Python 3.12 environment
and existing local files; neither downloads models or activates a backend.

```sh
python experiments/memory-retrieval/audit_native.py \
  --manifest payload-manifest.json --source-dir /path/to/canonical-source \
  --community-model-root /path/to/community-model --artifact-dir /path/to/native-artifact \
  --sample-count 6 --output native-audit.json
python experiments/memory-retrieval/embed_native.py \
  --manifest payload-manifest.json --source-dir /path/to/canonical-source \
  --artifact-dir /path/to/native-artifact --checkpoints native-work \
  --output native-vectors.json
```

See the [native vector contract study](../../studies/vector-contract-2026-09-09.md)
for the independently detected sliding-window defect and corrected results.

The frozen SciFact all-pairs, adversarial mixture, sequence, package, and
physical evidence gates remain separate and unchanged. This fixture also omits
reranking, graph expansion, network reunion, and real-user relevance judgments.

Checkpoint regressions run without model dependencies or inference:

```sh
python -m unittest discover -s experiments/memory-retrieval -v
```
