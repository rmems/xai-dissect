# Agent Instructions

`xai-dissect` is a read-only Grok-family checkpoint dissector. Preserve public
CLI and export-schema compatibility; do not add inference, checkpoint mutation,
or quantization-runtime behavior to this repository.

## Non-Interactive Shell Commands

For unattended agent runs, non-interactive flags reduce hangs on y/n prompts.
When a human is at the keyboard, interactive tools are fine.

Shell tools such as `cp`, `mv`, and `rm` may be aliased with `-i` on some hosts.

**Usual forms for unattended agents:**

```bash
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r directory
```

**Other commands that may prompt:**

- `scp` — use `-o BatchMode=yes` for non-interactive
- `ssh` — use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` — use `-y` flag
- `brew` — use `HOMEBREW_NO_AUTO_UPDATE=1` env var

## Validation

Run these before handing off a Rust-source, Cargo, CI, or behavior change:

```bash
cargo fmt --check
cargo test --locked
cargo clippy --all-targets --all-features -- -D warnings
```

When CLI behavior changes, also run the relevant help smoke command, for
example `cargo run --locked -- --help` or the changed subcommand's `--help`.

## Pull Request Review Threads

Do **not** resolve a GitHub review thread on the strength of a reply. Run both
commands before resolving; a thread is resolvable only when the change is
provably on `main`:

```bash
git show <sha> -- <path>   # cited commit touches the file
git show main:<path>       # content on main; exception: deferred-with-rationale
```

Empty output from the first command means that SHA did not change `<path>`.
Do not use the PR tip as a stand-in for a missing SHA — walk
`git log main..HEAD -- <path>` (or `origin/main` if `main` is not a local
ref). The one exception is a thread you are deliberately **not** fixing: mark it
`deferred-with-rationale` and resolve it with that rationale on the thread. If a
gap is real but the fix has not landed on `main` yet, the thread stays open —
`fixed-now` is not a resolvable state. When you cannot prove the content and
cannot justify deferring, leave the thread open and say so.

Most PRs here are squash-merged, so the SHA cited in an `Addressed in <sha>` reply
is usually **not** an ancestor of `main`. That alone is neither proof of a fix nor
proof of a gap — verify by content.


This gate applies to PR babysitting sessions too — the same proof is required when
closing out threads at the end of a babysit pass as when handling them one at a time.

## Session Completion

Before stopping or handing off, run the relevant validation and report the
current branch, commit, working-tree status, and any remaining review threads.
Do not push, rebase from a shared remote, prune remotes, or discard stashes
without the authorization described below.

| Operation | Needs explicit user OK? |
|-----------|-------------------------|
| `git push` to shared remote | Yes, unless user granted **push** autonomy this session |
| `git pull --rebase` from shared remote | Yes, unless user granted **push** (or explicit pull) autonomy this session |
| `git remote prune` / deleting remote branches | Yes, unless user granted **remote-cleanup** autonomy |
| `git stash drop` / discarding local stashes | Yes, unless user granted **stash-drop** autonomy |
