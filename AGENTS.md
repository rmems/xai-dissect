# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

> **Architecture in one line:** Issues live in a local Dolt database
> (`.beads/dolt/`); cross-machine sync uses `bd dolt push/pull` (a
> git-compatible protocol), stored under `refs/dolt/data` on your git
> remote — separate from `refs/heads/*` where your code lives.
> `.beads/issues.jsonl` is a passive export, not the wire protocol.
>
> See [SYNC_CONCEPTS.md](https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md)
> for the one-screen overview and anti-patterns (don't treat JSONL as the
> source of truth; don't `bd import` during normal operation; don't
> reach for third-party Dolt hosting before trying the default).

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

## Non-Interactive Shell Commands

For unattended agent runs, non-interactive flags reduce hangs on y/n prompts.
When a human is at the keyboard, interactive tools are fine.

Shell tools such as `cp`, `mv`, and `rm` may be aliased with `-i` on some hosts.

**Usual forms for unattended agents:**


```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**

- `scp` — use `-o BatchMode=yes` for non-interactive
- `ssh` — use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` — use `-y` flag
- `brew` — use `HOMEBREW_NO_AUTO_UPDATE=1` env var

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

Canonical agent task tracker is **bd**. Prefer `bd` over ad-hoc markdown TODO
lists or host-specific todo tools (TodoWrite / TaskCreate).

### Common bd commands

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Tracking rules

- Route open work through `bd` (create / claim / close)
- Load workflow detail with `bd prime`
- Persist cross-session notes with `bd remember` (avoid separate MEMORY.md files)

Dolt under `.beads/` is the issue source of truth; sync with `bd dolt push` /
`bd dolt pull`. Details: https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md

## Session Completion

After a coding session, run this checklist when the agent is about to stop or
hand off. Authorization rules:

| Operation | Needs explicit user OK? |
|-----------|-------------------------|
| `git push` / `git pull --rebase` to shared remote | Yes, unless user granted **push** autonomy this session |
| `git remote prune` / deleting remote branches | Yes, unless user granted **remote-cleanup** autonomy |
| `git stash drop` / discarding local stashes | Yes, unless user granted **stash-drop** autonomy |

**Workflow:**

1. File remaining work as beads issues
2. If code changed: run quality gates (`cargo fmt --check`, `cargo test --locked`, clippy `-D warnings`)
3. Update issue status (close finished, claim still-open)
4. With push authorization:

   ```bash
   git pull --rebase
   git push
   git status
   ```

5. With cleanup authorization: clear stashes / prune remotes as needed
6. Confirm intended commits exist; remote matches only if push was authorized
7. Hand off context for the next session

**Defaults:** leave a handoff note if work remains; do not push or prune shared
remotes without authorization; if an authorized push fails, fix or report the
blocker.
<!-- END BEADS INTEGRATION -->
