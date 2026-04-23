# Changelog

All notable changes to `xai-dissect` are documented here.

## 0.1.0 - 2026-04-23

Initial coherent milestone release.

### Added

- `inventory` command for full checkpoint inventory and architecture summary
- `experts` command for MoE expert discovery and expert-atlas generation
- `routing-report` command for routing/gating structure inspection
- `stats` command for offline tensor-statistics profiling
- `saaq-readiness` command for future SAAQ candidate scouting
- unified `reports/`, `exports/`, and `manifests/` output-tree convention
- Markdown and JSON export writers plus focused machine-readable manifests

### Scope

- Grok-1 support is the current release target
- Structural analysis only: parser-driven inventory, expert, routing, and
  offline profiling
- No inference runtime, no quantization runtime, no checkpoint mutation

### Stability Notes

- The stable integration surface is the export schema and CLI artifact layout
- The in-process Rust API remains intentionally small and secondary to the CLI
- Current shard assumptions center on Grok-1-style `f32` and `int8` layouts
- Docs, CLI help, and output conventions are kept aligned for the milestone
- Export bundles are covered by fixture-driven snapshot tests

### Validation

- `cargo test` passes on this milestone
- CLI help and per-command usage examples are documented in `README.md`
- Output conventions are documented in `docs/output-conventions.md`
- Export schema contracts are documented in `docs/export-contracts.md`

### Follow-ons

- Grok-2 support remains deferred until compatible public weights exist
- Future support planning lives in `docs/grok2-future-support.md`
