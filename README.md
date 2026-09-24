# cfetch

Local memory for agents: cited Markdown search, CPU vectors, Obsidian links,
code navigation and ordinary Git synchronization. One Rust binary; your files
remain the source of truth.

This is the current `main` architecture. Older releases and historical design
notes describe the removed remote-serving and peer-sharing system.

## Installation

Use a native archive from [Releases](https://github.com/corbet-libs/cfetch/releases),
or `nix run github:corbet-libs/cfetch` (`.#cfetch` when building this checkout).
Linux, Apple Silicon and Windows CPU packages include the embedding engine.
Intel Mac packages currently provide lexical and graph retrieval: upstream
ONNX Runtime does not publish that target. Model weights remain a separate,
qualified input configured with `embeddings.local_model`; installation never
starts an accelerator or silently downloads a model.

## Storage

```text
agents/
├── knowledge/                    # shared, human-readable repositories
│   ├── rules/                    # ring 0: musts
│   ├── behaviours/               # ring 2: tools, skills, practices
│   └── <topic>/                  # ring 3: independently shareable topics
├── mind/
│   ├── <selected-mind>/          # one Git repository per mind
│   │   ├── guidance/            # ring 1: shoulds
│   │   ├── identity/
│   │   ├── policy/
│   │   └── memories/            # ring 5: observations awaiting promotion
│   └── models/                   # ignored model files
├── todo/                         # ring 4: shared task repository
│   ├── backlog/
│   ├── active/
│   ├── blocked/
│   └── done/
├── logs/                         # ring 6: local activity
├── scratch/                      # disposable working files
└── projects/                     # code checkouts
```

`CFETCH_BRAIN` selects the root; `CFETCH_MIND` selects the mind independently
of the hostname. The default mind is the hostname. `CFETCH_STATE_DIR` selects
the local derived index and daemon state. Model files, secrets, other minds,
scratch and logs stay outside knowledge retrieval.

Nested repositories are ordinary independent checkouts. Their parent ignores
them; cfetch still indexes their visible Markdown. Repository access controls
what can be shared. cfetch reads a filesystem and does not manage its mounts.

## Use

```sh
cfetch init
cfetch recall "deployment decision"
cfetch recall "deployment decision" --hybrid
cfetch recall --id r3-<citation>
cfetch recall "deployment decision" --fresh
cfetch graph --focus "deployment"
cfetch graph-path "Deployment" "Backups" --depth 6
cfetch find SymbolName
cfetch code-graph path path/to/source.rs path/to/dependency.rs
cfetch repos ~/agents/knowledge ~/agents/todo --sync
cfetch memories list
cfetch daemon run
cfetch mcp
```

Recall combines lexical and semantic candidates with citations. Vector coverage
and fallback are reported. The graph follows actual Markdown links/backlinks;
missing and ambiguous notes remain explicit. Code dependency edges carry source
evidence. Graph traversal is bounded, never an LLM-generated claim.

The daemon keeps indexes warm and fingerprints source files before answering.
Edits, deletions and Git updates invalidate derived content. Markdown remains
authoritative. `status` reads cached health; `doctor` explains configuration.

## Local vectors

Build with `--features embedded-embeddings`. Set `embeddings.local_model` to a
qualified offline EmbeddingGemma pack. The pack binds graph, weights, tokenizer
and pipeline by content digest. `cfetch qualify-model PACK REFERENCES` checks
short/long canonical vectors and semantic ordering before enabling it. Model
preparation lives in `scripts/prepare-local-model.py` and runs on a build worker.
Ordinary recall, citation expansion and MCP memory queries return a committed cached snapshot with its generation and an explicit freshness note, and request one coalesced background refresh. `recall --fresh` (MCP `freshness: "strict"`) waits at most five seconds for a background scan begun after the request; it returns a freshness error if that proof is unavailable. When the daemon is unavailable, memory queries return a bounded error instead of scanning the source tree. Semantic ranking has its own bounded inference wait.

No model is downloaded during recall. Inputs exceeding the model's context are
reported as uncovered, never silently truncated.

```json
{
  "resident": [],
  "embeddings": {
    "enabled": true,
    "local_model": "/path/to/qualified/model-pack"
  },
  "git": {
    "enabled": true,
    "roots": ["/path/to/agents/knowledge", "/path/to/agents/todo"],
    "interval_secs": 60
  }
}
```

Configuration belongs in the machine's cfetch config, outside shared Markdown.
Model packs are immutable inputs; indexes and vector caches are derived. Remote
inference endpoints are refused. Optional model services must use loopback.

## Git behavior

Sync discovers nested repositories and serializes operations using a lock in
Git's common directory. It fetches and integrates committed history, then pushes.
Dirty worktrees, conflicting histories and repositories without an upstream are
reported individually. It never stashes, resets, force-pushes or silently commits
arbitrary edits. Commit intended changes explicitly; the next sync publishes them.
No Git LFS service or hosted memory account is required.

## License

See [LICENSE.md](LICENSE.md) and [third-party notices](THIRD-PARTY-LICENSES.txt).
