# Bot review verification (maintainers / agents)

Resolve a GitHub review thread only after the change is on `main`.
`isResolved=true` is not evidence. Historical PRs in this repo are usually
**squash-merged**, so the original review-reply SHA is often *not* an ancestor
of `main`.

## Enumerate threads (GitHub MCP)

Use `github__pull_request_read` method `get_review_comments` on
`rmems/xai-dissect` with `perPage=50`. Paginate with `after` until
`hasNextPage` is false.

In-scope first authors for this audit:

- `macroscopeapp`
- `chatgpt-codex-connector`
- `codacy-production`
- `devin-ai-integration`

Skip unless they left an inline thread with a concrete suggestion:

- `kilo-code-bot` (conversation summaries)
- `codeant-ai` conversation “reviewing / finished”
- `gemini-code-assist`, `copilot-pull-request-reviewer` with no inline comments
- `coderabbitai` conversation-only / rate-limit notes

## Per-thread proof (required before resolve)

**Proof target is always `main`.** For a squash-merged PR, use the merge
commit on `main` (for example PR #37 → `3b31ebf`), not the pre-squash
review-reply SHA.

```bash
# 1) Content on main (required for verified)
git show main:<path>

# 2) Optional: original review SHA still exists as an object
git cat-file -t <reply-sha>

# 3) Only if the PR was *not* squashed (merge commit preserves parents)
git merge-base --is-ancestor <reply-sha> main
```

If `git merge-base --is-ancestor <reply-sha> main` fails after a squash,
that is expected. Do not treat it as a missing fix. Compare `main:<path>`
to the bot concern instead.

Status vocabulary:

| Status | Meaning | Resolve? |
| --- | --- | --- |
| verified | Diff on `main` matches the bot concern | yes, after `git show main:<path>` |
| fixed-now | Gap confirmed; fix committed on `audit/bot-followups` (or this follow-up PR) but **not yet on `main`** | **no** — leave open until the follow-up is merged and `git show main:<path>` proves it |
| deferred-with-rationale | Intentional non-fix; rationale already on the thread or recorded | yes, with the rationale on the thread |

## Worked example (bad vs good)

Bot comment on `src/report/mod.rs:212`:

> This builds the metric label with `format!` inside the per-tensor loop, so it
> allocates once per tensor. Hoist the label table out of the loop.

### Bad — reply-only resolve

```text
Addressed in a1b2c3d: hoisted the label table.
```
…then mark the thread **Resolved**.

Why this fails the gate: nothing here proves `a1b2c3d` touched
`src/report/mod.rs`, and nothing proves the change survived the squash merge
onto `main`. A later agent reads `isResolved=true` and moves on. This is the
exact failure mode that produced the gh-30 backfill audit.

### Good — proof, then resolve

```bash
$ git show main:src/report/mod.rs | sed -n '205,215p'
const METRIC_LABELS: BTreeMap<&'static str, usize> = ...
    for tensor in tensors {
        let label = METRIC_LABELS[tensor.kind.as_str()];
```

The table is now a module-level constant and the loop indexes it — the concern
is satisfied **on `main`**, not merely in a branch. Reply with that evidence:

```text
verified — `git show main:src/report/mod.rs` (lines 205-215) shows METRIC_LABELS
hoisted to a module constant; the loop now indexes it instead of calling format!.
```

…then mark the thread **Resolved**.

If the same check had shown `format!` still inside the loop, the status is
`fixed-now` (fix it on the follow-up branch and **leave the thread open** until
that branch is on `main`) or `deferred-with-rationale` — never `verified`.

## At PR time

[`.github/pull_request_template.md`](../.github/pull_request_template.md)
puts the three-state checklist in front of the author while the threads are
still open, and asks for any `fixed-now` thread to be named in the PR body.
It is a reminder, not a gate — CI cannot see thread state.

The template exists because PR #56 merged with 11 threads unresolved. Every
one was, on inspection, already fixed on `main` by content; the code was fine
and the bookkeeping was not, which is the failure mode this document was
written to prevent in the other direction.

## Anti-patterns

- Reply “Addressed in …” without a commit that touches `<path>`
- Paste a truncated or concatenated SHA
- Resolve at session end without per-thread `git show`
- Trust `isResolved` on a historical PR
