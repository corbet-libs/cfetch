# Recovered embedding candidates

This experiment makes existing sweep candidates reproducible without
changing the production model. It does not admit any candidate or populate
the release backend registry. Python is isolated experiment tooling and must
not ship as a product dependency.

| Candidate | Width | Token limit | Pooling | Query / document prefixes |
| --- | --- | --- | --- | --- |
| `lightonai/modernbert-embed-large` | 1024 | 8192 | Masked mean, then L2 | `search_query: ` / `search_document: ` |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 512 | CLS, then L2 | `Represent this sentence for searching relevant passages: ` / empty |
| `BAAI/bge-m3` | 1024 | 8192 | CLS, then L2 | empty / empty |

Sources: the pinned [ModernBERT model card](https://huggingface.co/lightonai/modernbert-embed-large/blob/95a19bff4963b66d3c14fd4a20d147ebb4aaccfc/README.md),
[mxbai model card](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/blob/b33106f585b9ce46904ad7443a3b52b7a63e231c/README.md)
and their tokenizer, model and pooling configurations. `models.json` binds
the exact revisions and upstream ONNX LFS hashes; Git blob identities verify
the cached tokenizers and are supplemented by SHA-256 in every audit result.
ModernBERT and mxbai are English-oriented; BGE-M3 is multilingual. Actual
multilingual retrieval remains a quality gate. Width reduction and silent
truncation are not part of this experiment.

## Artifact audit

Use an existing worker environment with NumPy and ONNX. Supply a read-only
model root containing each candidate's files from `models.json`, including
external weights where declared. No downloads,
conversion, model compilation or inference occur. The result records verified
bytes, graph inputs/outputs and operator domains; it proves artifact identity,
not model quality or hardware compatibility. External weight tensors, including
nested graph attributes, must reference pinned local files and byte ranges inside
those verified artifacts. BGE-M3 uses the pinned upstream
[model and ONNX export](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181).

```sh
python experiments/embedding-candidates/candidates.py /models/sweep /evidence/audit.json
```

The shared CI check is `embedding-candidates`; `CFETCH_CANDIDATE_ROOT` and
`CFETCH_CANDIDATE_REPORT` enable the artifact audit after the contract tests.
`CFETCH_CANDIDATE_PYTHONPATH` may point to existing isolated worker dependencies.
The check never invokes a native inference backend.

## Direct OpenVINO comparison

`probe.py` bypasses ORT's bundled OpenVINO provider and uses the explicitly
versioned native OpenVINO installation. A provisioned, healthy host governor
and its reviewed policy digest are mandatory even for CPU. The runner does
not create policy/state, reset budgets, lift quarantine or establish operator
presence. Host-specific supervision and hardware approval remain external.

Each invocation compiles exactly one `[1,64]` graph, then makes three serial
inference calls: a query, a related document and an unrelated document. Every
native compile/inference has its own persisted governor lease and kernel
deadline. There are no retries or background loops. Device selection is
explicit `NPU`, `GPU` or `CPU`, with the execution device checked afterward;
AUTO/HETERO/MULTI are not used. Full padded tensors consume the governor budget.

```sh
python experiments/embedding-candidates/probe.py /models/sweep ModernBertEmbedLarge \
  /evidence/cpu.json --device CPU --runtime-version 2026.3.1 --policy-sha256 "$POLICY_SHA256"
python experiments/embedding-candidates/probe.py /models/sweep ModernBertEmbedLarge \
  /evidence/npu.json --device NPU --runtime-version 2026.3.1 --policy-sha256 "$POLICY_SHA256" \
  --reference /evidence/cpu.json
```

Reuse the same CPU result for the GPU comparison. Accelerator runs require
identical verified artifacts, prefixes and input tensors; changing any of
these refuses the reference before device compilation. Pooling and L2 happen
explicitly after native execution. The runner rejects unexpected output shapes
and nonfinite/zero vectors. Device parity below 0.9999 or failed semantic
ordering returns failure and retains the measurement. Output files are never
overwritten.

This short comparison cannot establish canonical Transformers parity,
long-input correctness, retrieval quality, endurance, restart/fallback or
production admission. In particular, a ModernBERT 8192-token declaration does
not prove an NPU can execute that shape. Production still needs a Rust adapter,
host-bound native runtime artifacts and the existing supervised NPU -> GPU ->
accelerated CPU dispatcher. Different models must never share a vector profile.
