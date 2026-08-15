# Bot review verification (maintainers / agents)

Resolve a GitHub review thread only after the cited change exists on
`main` (or on the fix PR that will merge to `main`). `isResolved=true`
is not evidence.

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

```bash
# SHA from the "Addressed in <sha>" reply, or the commit that should contain the fix
git merge-base --is-ancestor <sha> main
git show <sha> -- <path>
git show main:<path>
```

Status vocabulary:

| Status | Meaning |
| --- | --- |
| verified | Diff on `main` matches the bot concern |
| fixed-now | Missing on `main`; landed in `audit/bot-followups` |
| deferred-with-rationale | Intentional non-fix; rationale already on the thread or recorded |

## Anti-patterns

- Reply “Addressed in …” without a commit that touches `<path>`
- Paste a truncated or concatenated SHA
- Resolve at session end without per-thread `git show`
- Trust `isResolved` on a historical PR
