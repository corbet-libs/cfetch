# Pinned GGUF candidates

## Canonical F32 conversion and Vulkan runtime

`convert_canonical.py` converts the thirteen verified files from the canonical
Google revision with the pinned llama.cpp converter. It explicitly enables
`--sentence-transformers-dense-modules`; the upstream default omits the two
projection heads. The wrapper verifies the source inventory, converter bytes,
installed dependencies, F32 tensors, both projection shapes, pooling, special
tokens and attention metadata, then records path-free conversion lineage.
It requires a fresh output directory and preserves failed conversion evidence.

Run conversion in an isolated CPU worker with the pinned converter's
`requirements/requirements-convert_hf_to_gguf.txt` installed:

```bash
python3 experiments/embedding-profile/gguf/convert_canonical.py \
  --source-dir "$canonical_source" \
  --llama-source "$llama_source" \
  --output-directory "$candidate_output" \
  --timeout-seconds 600
```

The GGUF attention window must remain **512**. This runtime interprets it as
the symmetric distance of at most 256 positions; the pinned Transformers
reference expresses the same boundary as an exclusive radius of 257.
The execution bucket named 257 is a separate property.

`build-pinned-vulkan.sh SOURCE WORK OUTPUT` verifies the official b10516
Ubuntu Vulkan archive and compiles only the missing upstream embedding client
against its matching libraries. It requires Ubuntu 24.04 x86_64, glibc 2.39
and a C++17 compiler. Its provenance distinguishes the prebuilt libraries'
compiler from the client compiler and retains dynamic backends, dependencies
and licenses. The manual **GGUF Vulkan candidate preparation** workflow runs
this helper on a standard hosted CPU runner; it performs no model inference.

`native_probe.cpp` supplies a narrow C ABI for bounded qualification. It
separates initialization, explicit device selection, verified-descriptor model
loading, context allocation, one decode and cleanup so a caller can govern
each native operation. Outputs are 768 projected floats requiring caller L2
normalization. Actual context capacity rounds up to a multiple of 256 while
batch and microbatch capacities retain the requested bucket. Full GPU-layer
offload is a request; placement needs execution evidence.

Conversion lineage and a prepared runtime do not establish tokenizer parity,
long-input numerical parity, physical reliability or backend admission. The
community Q8 candidate below remains a separate historical diagnostic.

## Community Q8 SIMD CPU diagnostic

This directory reproduces one deliberately narrow result: the pinned standard
EmbeddingGemma Q8 GGUF can execute the exact cfetch query and document prompts
through a pinned, SIMD-enabled llama.cpp CPU build and produce finite,
non-zero, immediately repeatable canonical signed INT8x768 records.

This is a candidate smoke, not admission into the shared vector space. It does
not run SciFact, the global ordered-pair and adversarial mixed-store gates, a
sequence-bucket latency suite, or a peak-memory measurement. It proves nothing
about NPU or GPU
placement, and one successful hosted x86 CPU does not establish portability to
other CPU families.

## Pinned inputs

| Input | Pin |
|---|---|
| llama.cpp | tag `b10516`, revision `b95502ba9aa0eb73a2f4fc8878d7fbe6a847a0b9` |
| GGUF repository | `ggml-org/embeddinggemma-300M-GGUF` |
| GGUF revision | `0f741b5a6585bd53aeb15cd1372c56f2a0f65e12` |
| GGUF file | `embeddinggemma-300M-Q8_0.gguf` |
| GGUF SHA-256 | `b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63` |

The upstream GGUF declares `google/embeddinggemma-300m` as its base model but
does not prove the exact source revision used for conversion. That missing
lineage remains an admission blocker even when this smoke is green.

Q8_0 describes the GGUF's internal weight representation, not the shared
vector record. llama.cpp returns a normalized floating-point vector. The probe
then applies cfetch's float32 max-absolute, round-to-nearest-even signed INT8
codec. Cross-backend byte equality is not required; each backend must instead
pass the complete ordered all-pairs and adversarial mixed-store retrieval
gates.

## Local reproduction

The scripts require Linux, Git, curl, CMake, a C/C++ compiler with OpenMP, and
Python 3.10 or newer. The Python probe itself uses only the standard library.
Large source and model artifacts belong outside the repository:

```bash
probe_root="${TMPDIR:-/tmp}/cfetch-gguf-cpu-candidate"
mkdir -p "$probe_root"

bash experiments/embedding-profile/gguf/download-pinned-artifact.sh \
  "$probe_root/embeddinggemma-300M-Q8_0.gguf"

bash experiments/embedding-profile/gguf/build-pinned-llama.sh \
  "$probe_root/llama.cpp-b10516" \
  "$probe_root/llama.cpp-build"

python3 experiments/embedding-profile/gguf/probe.py \
  --llama-embedding "$probe_root/llama.cpp-build/bin/llama-embedding" \
  --model "$probe_root/embeddinggemma-300M-Q8_0.gguf"
```

The build enables the native CPU backend, OpenMP, CPU tensor repacking, and
host-native compiler optimization while explicitly disabling GPU backends.
The probe fails unless llama.cpp reports an enabled SIMD instruction family at
runtime. It hashes the model, verifies the pinned llama.cpp revision, runs one
prefixed query and one prefixed document twice with identical settings,
validates both 768-dimensional outputs, and requires the final canonical INT8
bytes to repeat exactly. Its JSON always says `candidate_only: true` and
`global_all_pairs_admitted: false`.

The **GGUF SIMD CPU candidate** GitHub Actions workflow performs the same check
on the then-current hosted Ubuntu runner. Because the runner image, system
CMake, and compiler are not a pinned release toolchain, a green run proves only
that run's exact-artifact immediate repeatability and SIMD discovery. It is not
portable reproducibility, full package placement evidence, or admission and
must never be copied into `release/inference-backends.json` as an admitted
backend.
