# CLI Routing

Maintainer map of how `xai-dissect` subcommands reach the library pipeline
and where artifacts land. This is the dispatch layer, not a usage guide.

User-facing walkthroughs for the same commands live in
[`docs/run-examples.md`](run-examples.md). Filenames and directory layout are
in [`docs/output-conventions.md`](output-conventions.md). Schema types behind
the JSON files are in [`docs/export-contracts.md`](export-contracts.md).

`src/main.rs` is the clap entry point. After
[#45](https://github.com/rmems/xai-dissect/issues/45), inventory-backed
`run_*` handlers live in `src/cli/`. `dissect` remains parser-only in
`src/main.rs`.

```
  clap Command (src/main.rs)
        |
        +-- dissect -------------> parser::dissect_shard  -> stdout table
        |
        +-- inventory-backed -----> cli::run_*
              |
              v
        inventory::build_inventory
              |
              +--> experts / routing / stats / planning
              |
              +--> report::write_* / render_*     (--json / --md)
              +--> exports::write_*_bundle        (--output-root)
```

## What this layer does NOT do

- It does **not** execute model inference, mutate checkpoints, or run
  quantization kernels (same boundary as the crate).
- It does **not** replace the export schema. Downstream consumers still key
  off artifacts, not in-process Rust types.
- `dissect` does **not** load an inventory, write reports, or participate in
  the unified output tree.

## Command routing

Every current `Command` variant in `src/main.rs` is listed. Inventory-backed
commands share `cli::run_inventory_command`, which builds
`inventory::InventoryConfig` from `CheckpointScanArgs` plus `--family` and
calls `inventory::build_inventory`.

| Subcommand | Handler | Inventory load | Analyzer | Report / export writers |
|---|---|---|---|---|
| `dissect` | `run_dissect` in `src/main.rs` | none — parser only (`parser::dissect_shard` per shard) | none | stdout `comfy_table` only; no `--json` / `--md` / `--output-root` |
| `inventory` | `cli::run_inventory` | `inventory::build_inventory` | none beyond inventory classification | `report::write_json` / `write_markdown` / `render_markdown`; `exports::write_inventory_bundle` |
| `experts` | `cli::run_experts` | `inventory::build_inventory` | `experts::build_expert_atlas` | `report::write_expert_json` / `write_expert_markdown` / `render_expert_markdown`; `exports::write_expert_bundle` |
| `routing-report` | `cli::run_routing_report` | `inventory::build_inventory` | `routing::build_routing_report` | `report::write_routing_json` / `write_routing_markdown` / `render_routing_markdown`; `exports::write_routing_bundle` |
| `stats` | `cli::run_stats` | `inventory::build_inventory` | `stats::build_stats_report` | `report::write_stats_json` / `write_stats_markdown` / `render_stats_markdown`; `exports::write_stats_bundle` |
| `saaq-readiness` | `cli::run_saaq_readiness` | `inventory::build_inventory` | `stats::build_stats_report` then `stats::build_saaq_readiness_report` | `report::write_saaq_readiness_json` / `write_saaq_readiness_markdown` / `render_saaq_readiness_markdown`; optional `report::write_candidate_manifest_json` (`--manifest`); `exports::write_saaq_bundle` |
| `pilot-plan` | `cli::run_pilot_plan` | `inventory::build_inventory` after complete-scope gate | `planning::build_grok1_pilot_selection_plan` | `report::write_pilot_selection_plan_json` / `write_pilot_selection_plan_markdown` / `render_pilot_selection_plan_markdown`; `exports::write_pilot_plan_bundle` |
| `route-preservation` | `cli::run_route_preservation` | `inventory::build_inventory` after complete-scope gate | `planning::build_grok1_route_preservation_report` | `report::write_route_preservation_report_json` / `write_route_preservation_markdown` / `render_route_preservation_markdown`; `exports::write_route_preservation_bundle` |
| `quant-plan` | `cli::run_quant_plan` | `inventory::build_inventory` after complete-scope gate | `experts::build_expert_atlas`, `routing::build_routing_report`, `stats::build_stats_report`, `stats::build_saaq_readiness_report`, then `planning::build_grok1_planning_artifacts` | `report::write_quant_plan_json` / `write_quant_plan_markdown` / `render_quant_plan_markdown`; optional `report::write_conversion_manifest_json` / `write_conversion_manifest_markdown`; `exports::write_quant_plan_bundle` |

`quant-plan` is the only command that runs the full analyzer stack (experts +
routing + stats + SAAQ readiness) before the planning layer. `pilot-plan` and
`route-preservation` take inventory only.

Dispatch is the `match cli.command` in `main()`. `Command::name()` and
`Command::fields()` feed tracing / Sentry tags (`command`, `limit`, `prefix`,
`family`, `sample_values`). Adding a variant without updating those match arms
fails to compile.

## Shared flags

### Output tree (`OutputTreeArgs` in `src/cli/mod.rs`)

Flattened onto every inventory-backed subcommand. `dissect` does not expose
these flags (`tests/cli_help.rs` asserts that).

| Flag | Behavior |
|---|---|
| `--output-root <dir>` | Enables the unified tree. The command still honors `--json` / `--md` (and command-specific file flags). The bundle is **additive**. |
| `--checkpoint-slug <slug>` | Overrides the inferred slug. clap `requires = "output_root"`, so this flag is rejected unless `--output-root` is also set. |

Slug resolution is `exports::resolve_checkpoint_slug`:

- If the override is set, it is sanitized; an empty result after sanitization
  is an error.
- Otherwise the slug is the last two sanitized path components of the
  checkpoint directory, joined with `__` (for example
  `grok-1-official/ckpt-0` → `grok-1-official__ckpt-0`). A single remaining
  component is used as-is; an empty path falls back to `checkpoint`.

When `--output-root` is set, `cli::write_output_tree` calls the command's
`exports::write_*_bundle` and prints the written tree to stderr:

```text
wrote <label> -> <root>/{reports,exports,manifests}/<slug>/...
```

`pilot-plan` and `route-preservation` print `{reports,manifests}` instead.
Those two bundles, and `quant-plan`, write Markdown under `reports/` and JSON
under `manifests/` only; they do not emit `exports/` JSON. `quant-plan` still
prints `{reports,exports,manifests}` in the stderr line. `prepare_output_layout`
still creates all three directories.

Bundle filenames are listed in [`docs/output-conventions.md`](output-conventions.md).

### Scan, family, sampling, and explicit files

Shared clap groups also live in `src/cli/mod.rs`.

| Flag / group | Default | Who uses it |
|---|---|---|
| `CheckpointScanArgs`: positional `path`, `--prefix`, `--limit` | prefix `"tensor"`; limit unset | every inventory-backed command; `dissect` inlines the same `path` / `--prefix` / `--limit` names |
| `--family` (`ModelFamilyArg` or `PlanningFamilyArg`) | `"grok-1"` | every inventory-backed command; written into inventory / export headers. Only `grok-1` is officially supported |
| `--sample-values` | `65_536` | `stats`, `saaq-readiness` (`SampleValuesArg`); `quant-plan` repeats the same long flag inline |
| `--json <path>` | unset | write the command's primary JSON document; omitted means no explicit JSON file (the output tree may still write one) |
| `--md <path>` | unset | write the Markdown report to a file. **If omitted**, inventory-backed commands print Markdown to stdout after the console summary |
| `--manifest <path>` | unset | `saaq-readiness` only: candidate-target JSON |
| `--conversion-manifest` / `--conversion-manifest-md` | unset | `quant-plan` only: conversion-manifest JSON / Markdown beside the quant-plan documents |

`--json` / `--md` paths are caller-chosen files. They are independent of the
`<root>/{reports,exports,manifests}/<slug>/` layout. Setting both writes both.

## Complete-inventory-scope gate

`pilot-plan`, `route-preservation`, and `quant-plan` pass their command name
into `cli::run_inventory_command`, which calls
`cli::validate_complete_inventory_scope` before `build_inventory`.

The enforced rule:

- `--prefix` must be the default `"tensor"`
- `--limit` must be unset

The runtime error text states the "what". The "why" is not in the source:
these three commands emit Grok-1 planning artifacts that assume a complete
checkpoint (full shard set, default prefix). A partial scan would still
produce JSON, but the plan would be invalid or misleading relative to the
clean-baseline coverage gate those artifacts depend on.

`inventory`, `experts`, `routing-report`, `stats`, and `saaq-readiness`
intentionally allow `--prefix` / `--limit` for subset debugging.

## Add a new subcommand

1. Add a `Command` variant in `src/main.rs` with clap args. Flatten
   `OutputTreeArgs` and `CheckpointScanArgs` unless the command is parser-only
   like `dissect`.
2. Extend `Command::name()` and `Command::fields()` so tracing / Sentry see
   the new command. `command_name_and_fields_cover_every_variant` in
   `src/main.rs` enumerates all nine variants today.
3. Add a `cli::run_*` handler (or `run_dissect`-style parser handler in
   `src/main.rs`) and dispatch it from `main()`.
4. If the command needs a complete Grok-1 inventory, pass
   `Some("command-name")` into `run_inventory_command` so the scope gate runs
   first.
5. If it produces a new export bundle: add `exports::write_*_bundle`, matching
   `report::write_*` / `render_*` functions, and a schema type in
   `src/schema/` when a new report type is introduced.
6. Tests:
   - Add the command name to `top_level_help_lists_current_commands` and, for
     inventory-backed commands, `analysis_commands_expose_output_tree_options`
     in `tests/cli_help.rs`. Parser-only commands follow
     `dissect_help_stays_parser_only` instead of the output-tree array.
     (`dissect` is currently covered by that parser-only test and is not in
     the top-level help name array.)
   - Add a `src/cli/tests.rs` handler test if the new `run_*` writes files.
   - For a new bundle: `sample_*()` in `tests/support/mod.rs`, a snapshot test
     in `tests/export_contracts.rs`, and
     `tests/fixtures/exports/<name>.snap` (regenerate with
     `XAI_DISSECT_WRITE_SNAPSHOTS=1`).
7. Docs: add a row to the table above, update the module-level command map in
   `src/main.rs`, and add a user-facing section in
   [`docs/run-examples.md`](run-examples.md). Artifact names go in
   [`docs/output-conventions.md`](output-conventions.md); new schema types go in
   [`docs/export-contracts.md`](export-contracts.md).

## Related

- [`docs/architecture.md`](architecture.md) — layer responsibilities
- [`docs/run-examples.md`](run-examples.md) — user-facing annotated CLI
- [`docs/output-conventions.md`](output-conventions.md) — bundle paths
- [`docs/export-contracts.md`](export-contracts.md) — schema contracts
- Issue [#46](https://github.com/rmems/xai-dissect/issues/46) — this document
- Issue [#41](https://github.com/rmems/xai-dissect/issues/41) — preferred
  home if `docs/codebase-map.md` lands later
- Issue [#45](https://github.com/rmems/xai-dissect/issues/45) — CLI helper
  extract (`src/cli/`)
