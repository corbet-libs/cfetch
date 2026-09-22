# Autonomous AI memory maintenance

cfetch continuously turns captured evidence into an Obsidian-compatible
Markdown second brain. Normal maintenance is change-driven and unattended: a
configured daemon proposes, independently reviews, verifies, and applies one
bounded transaction after relevant evidence changes. There is no routine
approval queue.

Markdown remains the authority. The model can suggest complete replacement
bytes, but cfetch alone enforces evidence, trust, path, secret, and concurrency
boundaries. A direct edit in Obsidian or any other editor always wins over a
stale maintenance transaction.

## Configure the maintenance models

The endpoint must expose an OpenAI-compatible `/chat/completions` route.
Credentials come from an environment variable, never from the JSON file. The
optional review model makes the second pass use a different model; without it,
cfetch still creates a fresh isolated request to the proposal model.

```json
{
  "maintenance": {
    "enabled": true,
    "endpoint": "http://127.0.0.1:8080/v1",
    "model": "memory-maintainer",
    "review_model": "memory-reviewer",
    "api_key_env": "MAINTENANCE_API_KEY",
    "debounce_secs": 30,
    "max_candidates": 4
  }
}
```

HTTPS and loopback endpoints are accepted by default. Any other host must be
named in `maintenance.allow_hosts`. Requests and responses are bounded, model
labels are sanitized before reaching agent-visible status, and credentials are
never written to the maintenance history.

Start the daemon once:

```console
$ cfetch daemon start
```

## The continuous lifecycle

1. Hooks append redacted activity to ring 6. Deterministic signals turn
   repeated failures, discovered fixes, and unusually hot trusted files into
   ring-5 candidates.
2. After a quiet period, the daemon notices a changed candidate revision and
   builds a bounded evidence packet containing raw events, current cited
   statements, and exact target snapshots.
3. The proposal pass returns one typed transition: `add`, `fold`, `supersede`,
   `revalidate`, `dismiss`, or `noop`. A proposal that writes memory contains
   the complete target Markdown, not a patch.
4. A fresh isolated review pass checks evidence coverage, factual
   faithfulness, preservation, authority, target choice, and contradictions.
5. Under a transaction lock, cfetch re-runs every deterministic gate against
   the current filesystem. No result from either model bypasses these checks.
6. A passing transaction writes the exact proposed bytes, records an immutable
   history event, finalizes the record, and settles its source candidate. A
   dismissal or no-op is recorded without writing trusted memory.
7. The Markdown watcher advances the catalog generation. Lexical search and
   the wikilink graph rebuild from the files; the vector worker hydrates shared
   or peer artifacts and embeds only still-missing content hashes.

The loop is event-driven rather than a model-on-a-timer cron. An unchanged
candidate revision does not trigger another inference call. Failures back off,
retain the candidate, and do not stop later candidates from being considered.

## External edits are authoritative

Every proposal captures the target's exact before bytes and hash. cfetch
checks them again immediately before writing. If a person or another process
edits the note first, the automatic transaction records an exception and
leaves the newer bytes untouched.

Per-file frontmatter can interject without disabling the rest of the system:

```yaml
---
cfetch-maintenance: pause
---
```

The values `manual` and `off` also block automatic writes to that file. Remove
the field to return it to the normal autonomous policy. To pause the whole
maintenance loop while debugging:

```console
$ cfetch maintain pause investigating-index-state
$ cfetch maintain history
$ cfetch maintain resume
```

Pausing does not discard evidence or candidates.

## Trust and authority

Rings describe how trusted and how readily served a statement is. Authority
describes what kind of evidence may support the claim.

| Authority | Meaning | Automatic result |
|---|---|---|
| `authorized` | Direct operator instruction present in the evidence | Ring 2, 3, or 4 |
| `attested` | Independently observable evidence | Ring 3 or 4 |
| `unendorsed` | Third-party text or model inference | Dismiss or no-op only |

Maintenance never writes rings 0 or 1. Ring 2 additionally requires direct
operator evidence; a model returning `authorized` cannot create that authority
itself. Ring 5 holds quarantined workflow records, and ring 6 holds redacted
capture. Neither is implicitly recalled or injected.

## Deterministic gates

Before any write, cfetch verifies that:

- proposal, review, candidate, evidence, and citation content addresses match;
- the immutable semantic review passed every named check for the exact
  proposal revision;
- the proposal is supported by matching captured evidence;
- cited statements still resolve to the bytes the models saw;
- the target is a brain-relative, symlink-free, indexable Markdown path;
- the target's effective ring and evidence authority permit the transition;
- the target still matches the captured before bytes;
- no other active transaction owns the same target;
- optional validity has not expired; and
- secret-shaped content is absent from both the proposed Markdown and the
  bounded journal fields.

These gates are re-evaluated inside the write lock. They are not assertions
made by the model.

## Observe and debug the system

The terminal dashboard is the live observability surface:

- **Activity** shows candidates, proposals, reviews, automatic outcomes,
  exceptions, exact hashes, and safe recovery commands.
- **Graph** shows the highest-connected notes or a focused wikilink
  neighborhood derived from Markdown.
- **System** shows maintenance configuration and route, proposal and review
  models, last attempt, catalog generation, vector coverage, selected
  inference backend, detected hardware, peer reachability, and findings.

The same evidence is available without the TUI:

```console
$ cfetch status --line
$ cfetch status --json
$ cfetch doctor
$ cfetch doctor --json
$ cfetch maintain history --json
$ cfetch graph --focus knowledge/runbooks/deployment.md
```

Status reports distinguish configured, selected, and last-used state. Hardware
that was merely detected is never presented as utilized, and a configured
remote route is never presented as successfully used until an attempt records
that result.

## Manual intervention

`cfetch maintain run` requests one bounded autonomous cycle immediately. The
lower-level commands remain available for reproduction, debugging, and legacy
automation:

```console
$ cfetch maintain packet <candidate-id> --json
$ cfetch maintain submit --file proposal.json
$ cfetch maintain review <proposal-id> --file review.json
$ cfetch maintain verify <proposal-id> --json
$ cfetch maintain auto-apply <proposal-id>
$ cfetch maintain revert <proposal-id>
```

The older `apply --approval-token` and git-aware `finalize` path remains
available for compatibility with manual workflows. It is not part of the
healthy autonomous path. MCP may create quarantined proposals and reviews for
debugging, but it cannot apply or revert trusted-memory bytes.

## Reversal and failure recovery

Every automatic outcome has an append-only Markdown history record under
`scratch/cfetch-staging/maintenance/history/`. Applied events retain the before and after
hashes, provenance, checks, rationale, candidate id, proposal id, and review
id. Journal text is bounded and secret-redacted.

`cfetch maintain revert <proposal-id>` restores the captured before bytes only
while the target still exactly matches the automatic after bytes. A subsequent
Obsidian edit makes the revert fail closed instead of destroying newer work.

Transport errors, malformed or oversized responses, rejected host policy,
missing credentials, failed semantic review, stale targets, and failed
deterministic checks become visible exceptions. The candidate remains
available for a later change or retry, and the daemon continues processing
other bounded work.

## What cfetch does not make authoritative

- A model verdict is not trust, evidence, or authorization.
- SQLite, vector artifacts, and graph edges are not a second fact store; all
  are rebuilt from readable Markdown.
- The dashboard is not a control room that must stay open.
- A browser service is not required for maintenance, observation, or editing.
- Captured activity is not promoted merely because it sounds plausible.

The result is a continuously maintained second brain that remains readable,
portable, inspectable, and directly editable with ordinary Markdown tools.


Generated evidence lives under `scratch/cfetch-staging/`, outside the task
ledger. `todo/` contains only backlog, active, blocked and done task states.
After upgrading every writer, run `cfetch init` to migrate both former paths
(`todo/staging/` and `staging/cfetch/`). Nested reviews and history move with
candidates. Name conflicts preserve both files and are reported for resolution;
empty old directories are removed. Migration also works when scratch is on a
separate filesystem. Existing Markdown and ignore files are not overwritten.
