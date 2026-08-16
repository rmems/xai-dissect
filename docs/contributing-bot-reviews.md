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

## Anti-patterns

- Reply “Addressed in …” without a commit that touches `<path>`
- Paste a truncated or concatenated SHA
- Resolve at session end without per-thread `git show`
- Trust `isResolved` on a historical PR
