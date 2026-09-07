<!--
Delete any section that does not apply. This template is a reminder, not
paperwork — the two checklists below are the gates that are already written
down in AGENTS.md and docs/contributing-bot-reviews.md.
-->

## What and why

<!-- What changes, and what problem it solves. Link the issue or bd id. -->

Refs:

## Quality gates

<!-- The three checks CI enforces. Run them locally before pushing. -->

- [ ] `cargo fmt --check`
- [ ] `cargo test --locked`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`

If the change touches **any Rust source**, `Cargo.toml`, `Cargo.lock`, or CI:

- [ ] `cargo +1.88 check --locked --all-targets --all-features` (the declared MSRV)

<!--
Source counts: a stdlib API or language feature newer than the floor breaks
the MSRV from inside `src/` without the manifest changing at all.
-->

## Review threads

`isResolved` is **not** evidence — the proof target is always `main`.
See [docs/contributing-bot-reviews.md](../docs/contributing-bot-reviews.md).

- [ ] There are no review threads on this PR

Otherwise list **every** thread. One line each, so a reader can check the
claim instead of taking it:

| Thread (`path:line`, author) | State | Proof / rationale |
|---|---|---|
|  |  |  |

<!--
State is one of:

  verified                 `git show main:<path>` matches the concern -> resolve
  deferred-with-rationale  intentional non-fix, rationale on the thread -> resolve
  fixed-now                real gap, fix not yet on `main` -> LEAVE OPEN

`fixed-now` is not a resolvable state. For those, say what lands the fix
("fixed on <branch>, resolves once that is on main") so the next reader does
not have to reconstruct it.
-->

## Export contract

<!-- Delete if this PR emits no artifact and changes no schema type. -->

- [ ] No change to an export schema type, **or** `schema_version` bumped and
      `docs/export-contracts.md` + `CHANGELOG.md` updated
- [ ] `tests/fixtures/exports/*.snap` are unchanged, **or** the diff is
      intentional and explained above
