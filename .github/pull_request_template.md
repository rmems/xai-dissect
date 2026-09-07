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

If the change touches `Cargo.toml`, `Cargo.lock`, or CI:

- [ ] `cargo +1.88 check --locked --all-targets --all-features` (the declared MSRV)

## Review threads

Every bot review thread on this PR is in one of three states before merge.
`isResolved` is **not** evidence — the proof target is always `main`.
See [docs/contributing-bot-reviews.md](../docs/contributing-bot-reviews.md).

- [ ] **verified** — `git show main:<path>` matches the concern → resolved
- [ ] **deferred-with-rationale** — intentional non-fix, rationale on the thread → resolved
- [ ] **fixed-now** — real gap, fix not yet on `main` → **left open**, and named below

<!--
`fixed-now` is not a resolvable state. If a thread is in it, say which one and
what lands it, so the next reader does not have to reconstruct that:

  - <thread/path>: fixed on <branch>, resolves once that is on main
-->

Threads left open, and why:

## Export contract

<!-- Delete if this PR emits no artifact and changes no schema type. -->

- [ ] No change to an export schema type, **or** `schema_version` bumped and
      `docs/export-contracts.md` + `CHANGELOG.md` updated
- [ ] `tests/fixtures/exports/*.snap` are unchanged, **or** the diff is
      intentional and explained above
