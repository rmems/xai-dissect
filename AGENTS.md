# Agent Instructions

This project tracks work in **GitHub Issues**. Use `gh issue list` / `gh issue view <n>`
to find and review work; open new issues with `gh issue create`.

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

## GitHub Issues

Canonical agent task tracker is **GitHub Issues**. Prefer `gh issue` over ad-hoc
markdown TODO lists or host-specific todo tools (TodoWrite / TaskCreate).

### Common gh commands

```bash
gh issue list                 # Find available work
gh issue view <n>              # View issue details
gh issue edit <n> --add-assignee @me  # Claim work
gh issue close <n>             # Complete work
```

### Tracking rules

- Route open work through GitHub Issues (create / assign / close)

## PR Review Threads

Do **not** resolve a GitHub review thread on the strength of a reply. A thread is
resolvable only when the change is provably on `main`:

```bash
git show main:<path>        # content on main must match the bot concern
git show <sha> -- <path>    # only meaningful if the PR was not squashed
```

The one exception is a thread you are deliberately **not** fixing: mark it
`deferred-with-rationale` and resolve it with that rationale on the thread. If a
gap is real but the fix has not landed on `main` yet, the thread stays open —
`fixed-now` is not a resolvable state. When you cannot prove the content and
cannot justify deferring, leave the thread open and say so.

Most PRs here are squash-merged, so the SHA cited in an `Addressed in <sha>` reply
is usually **not** an ancestor of `main`. That alone is neither proof of a fix nor
proof of a gap — verify by content.

Full gate, status vocabulary (`verified` / `fixed-now` / `deferred-with-rationale`),
and the anti-pattern list: [docs/contributing-bot-reviews.md](docs/contributing-bot-reviews.md).

This gate applies to PR babysitting sessions too — the same proof is required when
closing out threads at the end of a babysit pass as when handling them one at a time.

## Session Completion

After a coding session, run this checklist when the agent is about to stop or
hand off. Authorization rules:

| Operation | Needs explicit user OK? |
|-----------|-------------------------|
| `git push` to shared remote | Yes, unless user granted **push** autonomy this session |
| `git pull --rebase` from shared remote | Yes, unless user granted **push** (or explicit pull) autonomy this session |
| `git remote prune` / deleting remote branches | Yes, unless user granted **remote-cleanup** autonomy |
| `git stash drop` / discarding local stashes | Yes, unless user granted **stash-drop** autonomy |

**Workflow:**

1. File remaining work as GitHub issues
2. If code changed: run quality gates (`cargo fmt --check`, `cargo test --locked`, `cargo clippy --all-targets --all-features -- -D warnings`)
3. Update issue status (close finished, claim still-open)
4. With push authorization:

   ```bash
   git pull --rebase
   git push
   git status
   ```

5. With **remote-cleanup** / **stash-drop** authorization only: clear stashes / prune remotes as needed (do not treat push OK as cleanup OK)
6. Confirm intended commits exist; remote matches only if push was authorized
7. Hand off context for the next session

**Defaults:** leave a handoff note if work remains; do not push or prune shared
remotes without authorization; if an authorized push fails, fix or report the
blocker.
