# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, complete the checklist below. Steps that mutate shared remotes (push, prune, stash drop) require **explicit user authorization** unless the user already granted push/session-close autonomy for this session.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** (only with user authorization for remote mutation):
   ```bash
   git pull --rebase
   git push
   git status  # should show "up to date with origin" after a successful push
   ```
5. **Clean up** (only with user authorization) - Clear stashes, prune remote branches
6. **Verify** - Intended changes committed; remote updated when push was authorized
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Do not leave unfinished local work without handoff notes
- Do not push, force-push, or prune shared remotes without authorization
- If authorized push fails, resolve and retry until it succeeds (or report the blocker)
<!-- END BEADS INTEGRATION -->


## Build & Test

```bash
cargo fmt --check
cargo test --locked
cargo clippy --all-targets --all-features -- -D warnings
cargo run --locked -- --help
cargo run --locked -- quant-plan --help
cargo run --locked -- inventory --help
cargo run --locked -- saaq-readiness --help
```

Optional coverage: `cargo llvm-cov --workspace --locked --lcov --output-path lcov.info`.  
CI details: [docs/ci.md](docs/ci.md).

## Architecture Overview

Read-only Grok-family checkpoint dissector (Rust):

`parser → schema → inventory → {experts, routing, stats, planning} → report/exports`

No inference, no weight mutation, no quant runtime (that lives in `grok-ozempic`).

## Conventions & Patterns

- Prefer CLI + export schema stability over broad in-process API surface
- Track work with `bd` (Beads), not markdown TODO lists
- Quality gate before handoff: fmt, test `--locked`, clippy `-D warnings`
- Session push/cleanup requires explicit user authorization when operating as an agent
