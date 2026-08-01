# Project Instructions for AI Agents

This file provides Claude-specific project context. Beads/session protocol lives
in [AGENTS.md](AGENTS.md) (single source of truth for tracking and handoff).

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
- Track work with `bd` (Beads); see AGENTS.md
- Quality gate before handoff: fmt, test `--locked`, clippy `-D warnings`
- Session push/cleanup: follow AGENTS.md authorization table
