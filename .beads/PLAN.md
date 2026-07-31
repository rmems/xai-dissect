# xai-dissect Beads Plan — Bot Review Backfill Audit (gh-30)

> **Handoff doc for agent/model switches.** Read this + run `bd ready` before starting work.
> Last updated: 2026-07-31 (gh-33: Codecov + Qodana + optional Sentry; Aikido and New Relic excluded from CI scope)

## Quick start (any agent)

```bash
# From any clone of this repository root:
cd "$(git rev-parse --show-toplevel)"
bd prime                    # full beads workflow context
bd ready                    # next unblocked bead
bd show xai-dissect-iz3     # epic overview
bd children xai-dissect-iz3 # full tree
```

**Claim before work:** `bd update <id> --claim`  
**Close when done:** `bd close <id> --reason "..."`  
**Never use TodoWrite** — beads is the task tracker for this repo.

---

## GitHub mapping

| GitHub | Role |
|--------|------|
| [#30](https://github.com/rmems/xai-dissect/issues/30) | Parent epic — bot review backfill across merged PRs |
| [#38](https://github.com/rmems/xai-dissect/issues/38) | Sub-issue: PR #37 audit |
| [#39](https://github.com/rmems/xai-dissect/issues/39) | Sub-issue: PR #36/32/20 audit |
| [#40](https://github.com/rmems/xai-dissect/issues/40) | Sub-issue: workflow docs (no pr-review skill refs) |

**Beads epic:** `xai-dissect-iz3` → `external_ref: gh-30`

---

## Problem statement

Agents replied `Addressed in <sha>…` and marked GitHub review threads **Resolved** without always landing the fix on `main`. Future agents treat `isResolved=true` as done.

**Goal:** For each merged PR in the audit queue, verify every resolved inline bot thread with `git show` evidence. Fix gaps in one follow-up PR (`Refs #30`).

---

## Bead tree (sequential blocks chain)

```
xai-dissect-iz3 [EPIC P0] Bot review backfill audit — all previous PRs (gh-30)
├── xai-dissect-iz3.1 [IN PROGRESS] Setup: thread enumeration + git-show workflow
├── xai-dissect-iz3.2         Audit PR #37 (gh-38) — handoff-hardening, ~34 threads
├── xai-dissect-iz3.3         Audit PR #36 — planning/report surfaces (25 comments)
├── xai-dissect-iz3.4         Audit PR #32 — export contract (47 comments)
├── xai-dissect-iz3.5         Audit PR #34 — tensor manifest (45 comments)
├── xai-dissect-iz3.6         Audit PR #20 — explicit bot followups (23 comments)
├── xai-dissect-iz3.7         Audit PR #24 — coverage manifest (26 comments)
├── xai-dissect-iz3.8         Audit PR #35 — CI hardening (CLOSED not merged; skip if N/A)
├── xai-dissect-iz3.9         Consolidated audit table on gh-30
├── xai-dissect-iz3.10        Implement missing fixes (branch: audit/bot-followups)
└── xai-dissect-iz3.11        Open follow-up PR + final summary (Refs #30)
```

**Dependency order:** `.1 → .2 → .3 → .4 → .5 → .6 → .7 → .8 → .9 → .10 → .11`

---

## Per-bead acceptance criteria

### iz3.1 — Setup (CURRENT)
- Document repeatable commands (below) in gh-30 comment or `docs/contributing-bot-reviews.md` stub
- Confirm GraphQL query returns resolved threads for PR #37 as smoke test

### iz3.2–iz3.8 — Per-PR audit (same pattern)
- List all `isResolved=true` inline threads from: macroscopeapp, codacy-production, chatgpt-codex-connector
- Skip kilo-code-bot if informational-only (no inline suggestion)
- Each thread status: **verified** | **fixed-now** | **deferred-with-rationale**
- Evidence: `git merge-base --is-ancestor <sha> main` + `git show <sha> -- <file>` + `git show main:<file>`

### iz3.9 — Audit table
- Single markdown table on gh-30: PR | thread_id | file | bot | status | evidence

### iz3.10 — Fixes

- Only if gaps found in .2–.8
- Branch: `audit/bot-followups`
- `cargo test --locked` + `cargo clippy --all-targets --all-features -- -D warnings` green

### iz3.11 — Closeout

- PR with `Refs #30` (not Fixes until all gaps closed)
- Final summary comment on gh-30 with per-PR counts

### Epic iz3 — Done when

- PRs #37, #36, #32, #34, #20, #24 audited (#35 if applicable)
- Table posted; follow-up PR merged if needed

---

## Verification commands (copy-paste)

```bash
# Per PR N:
gh pr view N --json mergedAt,mergeCommit,title,state

# Resolved threads (GraphQL) — paginate reviewThreads via $after until hasNextPage=false.
# Each thread includes id (for audit table) and paginated comments (first:100 + after).
PR=N
AFTER=""
while true; do
  if [ -n "$AFTER" ]; then AFTER_ARG=(-f after="$AFTER"); else AFTER_ARG=(); fi
  PAGE=$(gh api graphql \
    -f query='
      query($owner:String!,$repo:String!,$pr:Int!,$after:String) {
        repository(owner:$owner,name:$repo) {
          pullRequest(number:$pr) {
            reviewThreads(first:50, after:$after) {
              pageInfo { hasNextPage endCursor }
              nodes {
                id
                isResolved
                path
                line
                comments(first:100) {
                  pageInfo { hasNextPage endCursor }
                  nodes { author { login } body }
                }
              }
            }
          }
        }
      }' \
    -f owner=rmems -f repo=xai-dissect -F pr="$PR" "${AFTER_ARG[@]}")
  echo "$PAGE" | jq -c '.data.repository.pullRequest.reviewThreads.nodes[]'
  HAS=$(echo "$PAGE" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage')
  [ "$HAS" = "true" ] || break
  AFTER=$(echo "$PAGE" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.endCursor')
done

# If a thread's comments.pageInfo.hasNextPage is true, re-query that thread's
# comments connection with after=<endCursor> until complete (max 100 per page).

# Per thread with cited SHA:
git merge-base --is-ancestor <sha> main
git show <sha> -- <file>
git show main:<file>
```

---

## Anti-patterns (do NOT repeat)

- Reply "Addressed in …" without a commit touching the cited file
- Resolve/re-resolve threads before diff proof on the fix branch
- Trust `isResolved` without `git show` verification
- Batch-resolve at session end without per-thread checks

---

## Deferred GitHub issues (do NOT start until iz3 epic closes)

| Issue | Title |
|-------|-------|
| #33 | ~~CI~~ — **started 2026-07-31** (user priority; not blocked by iz3). Full scope: Codecov + Qodana + optional Sentry; **no Aikido/NR**. |
| #41 | docs/codebase-map.md |
| #42 | model-family extension design |
| #43 | split report/mod.rs |
| #44 | isolate families/grok1 |
| #45 | main.rs boilerplate |
| #46 | CLI routing maintainer docs |
| #47 | ModelFamily epic |

**Not on backlog:** GO/NO-GO / Vultr runbook gate (#46 was repurposed away from this).

---

## Repo / tooling notes

- **Beads DB:** `.beads/` in repo root, prefix `xai-dissect-*`
- **Beads MCP:** set `workspace_root` to this repo’s root (`git rev-parse --show-toplevel`); MCP may route to a global DB if context is unset — prefer `bd` CLI in-repo
- **Chroma:** collection `agent-workflows`, doc id `xai-dissect-beads-plan-gh30`
- **Ogham:** tagged `project:xai-dissect`, `beads`, `gh-30`

---

## Agent handoff checklist

When picking up this work:

1. `cd "$(git rev-parse --show-toplevel)" && git pull && bd ready`
2. Read this file + `bd show xai-dissect-iz3`
3. Claim current bead: `bd update <id> --claim`
4. Work only the claimed bead scope
5. Post incremental notes on gh-30 as each PR audit completes
6. `bd close <id>` with reason when acceptance criteria met
7. `bd ready` for next bead

When ending a session mid-bead:

- `bd update <id> --append-notes "..."` with progress + blockers
- Do not close partial work