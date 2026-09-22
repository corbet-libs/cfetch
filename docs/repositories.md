# Repository composition and graph navigation

A brain can contain independent topic, skill and project repositories. Clone
each at its chosen path and exclude the child checkout in its parent's
`.gitignore`. Each checkout keeps its own branch, history and access permissions;
no submodule pins are required. cfetch neither creates mounts nor assumes that
each machine has separate copies of files.

`cfetch repos knowledge todo --sync` discovers repositories below those roots,
fetches their configured upstream branches, merges clean committed changes and
pushes ahead commits. Omit `--sync` for a read-only view using cached Git refs.
Use `--json` for per-repository outcomes. Dirty, detached, unborn and untracked
branches are reported without automatic commits. Existing staged changes stay
staged. Conflicting merges are previewed without writing conflict markers into
working documents; resolve them explicitly using Git. An offline repository
does not stop another repository from synchronizing. A failed synchronization
returns a nonzero status, including when some repositories succeeded.

Enable periodic synchronization of enrolled roots in the machine configuration:

```json
{"git":{"enabled":true,"roots":["knowledge","todo"],"interval_secs":60,"timeout_secs":30}}
```

Roots relative to the brain are supported in configuration. Read operations do
not fetch implicitly. Git owns credentials and hosting; there is no provider API.
Automatic synchronization requires Git with `merge-tree --write-tree` support
for divergent branches. cfetch never resets, stashes, rebases or force pushes.
Its atomic lock is `cfetch-repository.lock` in Git's common directory, shared
across cooperating processes and linked worktrees. A crash may leave that lock;
inspect for a live writer before removing it. Locks are never stolen by age.
Other writers should cooperate with this lock or perform their work while the
synchronizer is paused. Repository state is cached locally in `repositories.json`.

Markdown stays authoritative. The graph connects available notes across all
enrolled content, independent of repository boundaries. Missing checkouts and
ambiguous note names do not produce guessed edges.

```sh
cfetch graph --focus architecture
cfetch graph-path architecture deployment --depth 6 --json
```

Paths follow explicit wikilinks and backlinks, with original link directions
retained in JSON. The shortest route is bounded to at most 32 hops. The same
operations are available over MCP as `cfetch_graph` and `cfetch_graph_path`.
