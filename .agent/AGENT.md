# cfetch — agent knowledge

- The governing documents live in the operator's private knowledge tree under
  `todo/active/cfetch/`: **PRD.md** (product requirements — wins on conflict),
  **VOCAB.md** (canonical terms — use these words, not synonyms), DESIGN.md
  (mechanism dossiers + decision history) and STATUS.md (current state).
  Read PRD + VOCAB before implementing anything.
- ARCHITECTURE: Markdown is the storage of record. Independently shared Git
  repositories compose the brain, including nested skills and project repos.
  Filesystem placement is external; cfetch never configures mounts. Minds
  are selected independently of hostnames and may serve several environments.
  Git synchronizes committed changes with ordinary remotes; delays and offline
  work are expected. Preserve edits, isolate conflicts per repository, never
  force push, stash or reset. Queries and computation run locally. Text search,
  vectors, Obsidian link traversal and code dependency graphs are core features.
  Reuse compatible content-addressed vectors; mutable indexes remain local.
  Remove the former peer transport, grants, remote queries and compute routing.
  Authoritative source edits invalidate derived results regardless of watchers.
- INDEXES ARE NEVER A FACT OF RECORD. Deleting any index must lose nothing.
- CLEAN-ROOM RULE (load-bearing): this project implements mechanisms described in
  the private dossiers. Never port, translate, or paraphrase source code from
  `cytostack/openwolf` or `bassprofressor-lab/openwolf-enhanced` (both AGPL) —
  that is what keeps cfetch's own license possible.
- All output, comments, and docs in English.
- Commit directly to `main`, no AI attribution lines.
- PUBLIC GENERAL TOOL (load-bearing): this repo is public and cfetch is a
  general product, not operator-specific tooling. Never commit private
  infrastructure details — no LAN/overlay IPs, hostnames, usernames, service
  names, or paths from the operator's network, not even in tests, fixtures,
  comments, or commit messages. Use RFC 5737/3849 documentation addresses and
  generic slugs in fixtures. Defaults must work for any deployment; anything
  operator-specific belongs in the operator's config, never in code.
