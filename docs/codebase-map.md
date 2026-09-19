# Codebase map

Maintainer map of how the Rust modules connect. Layer *responsibilities*
live in [`architecture.md`](architecture.md). User-facing CLI walkthroughs
live in [`run-examples.md`](run-examples.md). Command dispatch (handler,
flags, test touchpoints) lives in [`cli-routing.md`](cli-routing.md).

This file is the change-index: which crate files to open when adding a
`TensorKind`, CLI subcommand, export artifact, Grok-1 slot spec, or
planning validator. It does not describe Grok-1 the model
([`grok1-architecture.md`](grok1-architecture.md)) and it is not an
in-process API contract ([`export-contracts.md`](export-contracts.md)).

## Pipeline

```text
  raw shard bytes
        |
        v
  [ parser ]          RawTensor  (pickle PROTO 4 scan, no unpickler)
        |
        v
  [ schema ]          TensorInfo and every serializable export type
        |
        v
  [ inventory ]       ModelInventory  (classify, block-map)
        |
        +--> [ experts ]    ExpertAtlas
        +--> [ routing ]    RoutingReport
        +--> [ stats ]      StatsProfileReport / SaaqReadinessReport
        +--> [ planning ]   ConversionManifest / QuantPlan / pilot / route
        |
        +--> [ report ]     Markdown + explicit --json / --md files
        +--> [ exports ]    --output-root tree (reports / exports / manifests)
```

`parser → schema → inventory → {experts, routing, stats, planning} → report/exports`

Every inventory-backed CLI command loads `ModelInventory` first. `dissect`
stops at the parser. `quant-plan` is the only command that runs the full
analyzer stack before planning. The Grok-1 coverage gate
(`inventory::validate_grok1_complete_manifest`) is not part of
`build_inventory`; it runs when an inventory bundle is exported and when
planning validators run.

## Per-module table

Line counts are approximate (`wc -l` on this tree). The issue that asked
for this map required every `src/` module over 100 lines; those rows are
marked **yes**. Smaller files are listed so the crate has no holes.

| File | LOC >100 | Responsibility | Key types in → out | Primary tests |
| --- | --- | --- | --- | --- |
| [`src/lib.rs`](../src/lib.rs) | no (19) | Library crate root. Re-exports analysis modules. Binary-only modules (`cli`, `observability`) are not here. | — → `pub mod` surface | none (wiring only) |
| [`src/main.rs`](../src/main.rs) | **yes** | clap entry. Owns `Command` plus parser-only `dissect`. Inventory-backed variants dispatch to `cli::run_*`. `Command::name()` / `fields()` feed tracing / Sentry. | argv → `cli::run_*` / `parser::dissect_shard` → stdout table | `src/main.rs` (`command_name_and_fields_cover_every_variant`, `dissect_hex_fixture_prints_tensor_table`); [`tests/cli_help.rs`](../tests/cli_help.rs) |
| [`src/observability.rs`](../src/observability.rs) | **yes** | Opt-in Sentry (`XAI_DISSECT_SENTRY` + `SENTRY_DSN`) and tracing init. Default off. Path-scrubs `$HOME` only. | `anyhow::Error` → Sentry event / `error_category` tag | `src/observability.rs` (`categorizes_*`, `sentry_opt_in_ready`, path scrub) |
| [`src/cli/mod.rs`](../src/cli/mod.rs) | **yes** | Inventory-backed handlers. Shared clap groups (`CheckpointScanArgs`, `OutputTreeArgs`, family, sample-values). Complete-scope gate for planning commands. | scan args → `inventory::build_inventory` → analyzer → `report` / `exports` | [`src/cli/tests.rs`](../src/cli/tests.rs); [`tests/cli_orchestration.rs`](../tests/cli_orchestration.rs) |
| [`src/cli/summary.rs`](../src/cli/summary.rs) | **yes** | stderr console summaries after each subcommand. Display only; not an export. | schema report types → stderr | `src/cli/summary.rs` (per-command summary tests) |
| [`src/cli/tests.rs`](../src/cli/tests.rs) | **yes** | Handler tests for `cli::run_*`, complete-scope gate, and `--json` / `--md` / `--output-root` writers. Test-only (`#[cfg(test)] mod tests` from `cli/mod.rs`). | `cli::run_*` + hex fixtures → written files / stdout | `run_inventory_*`, `run_experts_*`, `run_stats_and_saaq_*`, `planning_commands_reject_*` |
| [`src/parser/mod.rs`](../src/parser/mod.rs) | **yes** | PROTO 4 byte-grammar scanner. No Python, no payload decode, no `TensorKind`. | shard path → `Vec<RawTensor>` (`role`, `dtype`, `shape`, `offset`, `nbytes`) | `src/parser/mod.rs` (`qw8_role_assignment_*`); [`tests/parser_inventory.rs`](../tests/parser_inventory.rs) (`parser_fixture_discovers_single_f32_tensor`) |
| [`src/schema/mod.rs`](../src/schema/mod.rs) | **yes** | Canonical serializable types. Every JSON document lives here. Version discipline is documented at the top of the file. | parser/inventory fields → `TensorKind`, `TensorInfo`, `ModelInventory`, `ExpertAtlas`, `RoutingReport`, `StatsProfileReport`, `SaaqReadinessReport`, `Grok1CoverageManifest`, `ConversionManifest`, `QuantPlan`, `PilotSelectionPlan`, `RoutePreservationReport`, manifests | round-trip coverage via [`tests/export_contracts.rs`](../tests/export_contracts.rs) and schema-using unit tests; no dedicated `schema` test module |
| [`src/inventory/mod.rs`](../src/inventory/mod.rs) | **yes** | Walk shards, infer hyperparams, classify `TensorKind`, assign `block_index` / `block_slot`, disambiguate Grok-1 gate/up. | `RawTensor` + `InventoryConfig` → `ModelInventory` | `src/inventory/mod.rs` (`shifted_grok_layout_*`, classification/layout tests); [`tests/parser_inventory.rs`](../tests/parser_inventory.rs) (`inventory_fixture_builds_without_real_weights`) |
| [`src/inventory/grok1_coverage.rs`](../src/inventory/grok1_coverage.rs) | **yes** | Fail-closed `grok1-map-v1-clean` coverage. Owns `GROK1_SLOT_SPECS` (12 slots/block). | `ModelInventory` → `Grok1CoverageManifest` | `src/inventory/grok1_coverage.rs` (`validates_complete_grok1_coverage_manifest` and fail-path tests); [`docs/grok1-coverage-manifest.md`](grok1-coverage-manifest.md) |
| [`src/inventory/test_fixtures.rs`](../src/inventory/test_fixtures.rs) | **yes** | Canonical 770-tensor Grok-1 inventory builder. Test-only (`cfg(test)`). | constants from `grok1_coverage` → `ModelInventory` | consumed by inventory, exports, planning tests (not a test module itself) |
| [`src/experts/mod.rs`](../src/experts/mod.rs) | **yes** | MoE geometry from inventory records. Shape-only. Resolves gate/down/up slices. | `&ModelInventory` → `ExpertAtlas` | `src/experts/mod.rs` (`atlas_discovers_expert_blocks_and_slices` and sibling layout tests) |
| [`src/routing/mod.rs`](../src/routing/mod.rs) | **yes** | Router/gate structure. Identifies `(d_model, n_experts)` tensors and orientation. Does not execute routing. | `&ModelInventory` → `RoutingReport` | `src/routing/mod.rs` (`report_discovers_primary_router_and_orientation` and sibling tests) |
| [`src/stats/mod.rs`](../src/stats/mod.rs) | **yes** | Read-only payload sampling. Tensor stats and SAAQ-readiness scouting. No weight mutation. | `&ModelInventory` + `StatsConfig` → `StatsProfileReport` / `SaaqReadinessReport` | `src/stats/mod.rs` (`stats_report_profiles_tensor_values` and SAAQ scoring tests) |
| [`src/planning/mod.rs`](../src/planning/mod.rs) | **yes** | Grok-1 conversion / quant / pilot / route-preservation artifacts. Validators run before emission. Does not quantize. | inventory + atlas + routing + readiness → `ConversionManifest`, `QuantPlan`, `PilotSelectionPlan`, `RoutePreservationReport` | `src/planning/mod.rs` (`planning_artifacts_use_named_grok1_baseline`, `validate_readiness_groups_*`, conversion-policy tests); [`tests/pilot_route_reports.rs`](../tests/pilot_route_reports.rs) |
| [`src/report/mod.rs`](../src/report/mod.rs) | no (58) | Barrel: re-exports Markdown/JSON writers split by artifact family. | schema types → files / Markdown strings | [`tests/report_render_branches.rs`](../tests/report_render_branches.rs) |
| [`src/report/common.rs`](../src/report/common.rs) | no (99) | Shared pretty-JSON / text writers and findings-summary helper. | `Serialize` / `&str` → path | covered by report/export tests |
| [`src/report/inventory.rs`](../src/report/inventory.rs) | **yes** | `inventory.md` / `.json`, coverage + snapshot manifests. | `ModelInventory` / `Grok1CoverageManifest` → files | [`tests/report_render_branches.rs`](../tests/report_render_branches.rs); [`tests/export_contracts.rs`](../tests/export_contracts.rs) |
| [`src/report/experts.rs`](../src/report/experts.rs) | **yes** | Expert atlas Markdown/JSON. | `ExpertAtlas` → files | same |
| [`src/report/routing.rs`](../src/report/routing.rs) | **yes** | Routing report Markdown/JSON + critical-tensor manifest writer. | `RoutingReport` → files | same |
| [`src/report/stats.rs`](../src/report/stats.rs) | **yes** | Stats Markdown/JSON. | `StatsProfileReport` → files | same |
| [`src/report/saaq.rs`](../src/report/saaq.rs) | **yes** | SAAQ readiness Markdown/JSON + candidate manifest. | `SaaqReadinessReport` → files | same |
| [`src/report/planning.rs`](../src/report/planning.rs) | **yes** | Quant-plan, conversion-manifest, pilot, route-preservation writers. | planning schema types → files | same; [`tests/pilot_route_reports.rs`](../tests/pilot_route_reports.rs) |
| [`src/exports/mod.rs`](../src/exports/mod.rs) | **yes** | Unified `--output-root` layout, slug resolution, bundle writers, findings summaries, inventory snapshot, routing-critical manifest. | schema types + root/slug → `OutputBundle` (written paths) | `src/exports/mod.rs` (slug / findings unit tests); [`tests/export_contracts.rs`](../tests/export_contracts.rs) (`*_bundle_matches_snapshot`) |
| [`src/test_support.rs`](../src/test_support.rs) | no (56) | Hex-checkpoint helper shared by `main.rs` and `cli` tests. Binary-only. | prefix → temp shard dir | used by `src/main.rs` / `src/cli/tests.rs` |

Integration tests under `tests/` that cut across modules:

| Test file | What it locks |
| --- | --- |
| [`tests/cli_help.rs`](../tests/cli_help.rs) | `--help` command list and output-tree flags |
| [`tests/cli_orchestration.rs`](../tests/cli_orchestration.rs) | end-to-end inventory-backed CLI writes |
| [`tests/export_contracts.rs`](../tests/export_contracts.rs) | golden snapshots in `tests/fixtures/exports/*.snap` |
| [`tests/parser_inventory.rs`](../tests/parser_inventory.rs) | synthetic pickle fixture through parser + inventory |
| [`tests/pilot_route_reports.rs`](../tests/pilot_route_reports.rs) | planning-side pilot / route-preservation reports |
| [`tests/report_render_branches.rs`](../tests/report_render_branches.rs) | Markdown render branches |

## Where to change X

Each recipe is the minimum set of files. Skip a later step only when the
change truly does not touch that layer (for example a parser-only command
does not need a coverage slot spec).

### New `TensorKind`

Semantic classification, not the parser-level `TensorRole`.

1. Add the variant (and `short_label`) on `schema::TensorKind` in
   [`src/schema/mod.rs`](../src/schema/mod.rs). This is a tagged serde enum;
   an incompatible JSON shape change bumps `ModelInventory.schema_version`
   and [`docs/export-contracts.md`](export-contracts.md).
2. Teach `inventory::classify_tensor` in [`src/inventory/mod.rs`](../src/inventory/mod.rs)
   the shape/role/dtype rule. If Grok-1 gate/up-style slot disambiguation
   is required, extend `disambiguate_grok1_moe_projection_slots` in the
   same file.
3. If the kind appears in a Grok-1 block, add a `Grok1ExpectedKind` arm and
   a `GROK1_SLOT_SPECS` row in [`src/inventory/grok1_coverage.rs`](../src/inventory/grok1_coverage.rs)
   (see [New Grok-1 slot spec](#new-grok-1-slot-spec)).
4. Update downstream matches that currently enumerate kinds:
   - experts: [`src/experts/mod.rs`](../src/experts/mod.rs)
   - routing: [`src/routing/mod.rs`](../src/routing/mod.rs)
   - stats / SAAQ scoring: [`src/stats/mod.rs`](../src/stats/mod.rs)
   - planning policy: `quant_policy_for_tensor` in [`src/planning/mod.rs`](../src/planning/mod.rs)
   - console summaries: [`src/cli/summary.rs`](../src/cli/summary.rs)
5. Document the rule in [`docs/tensor-schema.md`](tensor-schema.md). Add a
   classification unit test in `src/inventory/mod.rs`. If exports change
   shape, refresh snapshots (`XAI_DISSECT_WRITE_SNAPSHOTS=1`).

### New CLI subcommand

The step-by-step for clap + handlers + tests is in
[`cli-routing.md` → Add a new subcommand](cli-routing.md#add-a-new-subcommand).
In crate terms:

1. `Command` variant, `name()`, `fields()`, and `match` arm in
   [`src/main.rs`](../src/main.rs).
2. `cli::run_*` in [`src/cli/mod.rs`](../src/cli/mod.rs) (or `run_dissect`-style
   in `main.rs` if parser-only). Flatten `CheckpointScanArgs` /
   `OutputTreeArgs` unless the command is parser-only.
3. Analyzer in the matching module (`experts` / `routing` / `stats` /
   `planning`, or a new `src/<layer>/mod.rs` re-exported from `lib.rs`).
4. Writers: `report::write_*` / `render_*` and, for `--output-root`,
   `exports::write_*_bundle`.
5. Tests: `tests/cli_help.rs`, `src/cli/tests.rs`, and a snapshot if a new
   bundle exists.
6. Docs: this table, [`cli-routing.md`](cli-routing.md),
   [`run-examples.md`](run-examples.md),
   [`output-conventions.md`](output-conventions.md).

### New export artifact / snapshot

1. Schema type in [`src/schema/mod.rs`](../src/schema/mod.rs) with
   `schema_version`. Incompatible shape → bump that version.
2. Markdown/JSON writer in the matching [`src/report/`](../src/report/) file
   (or a new family file re-exported from [`src/report/mod.rs`](../src/report/mod.rs)).
3. Bundle writer `exports::write_*_bundle` in [`src/exports/mod.rs`](../src/exports/mod.rs).
   Filenames must follow [`output-conventions.md`](output-conventions.md).
4. Sample builder `sample_*()` in [`tests/support/mod.rs`](../tests/support/mod.rs).
5. Snapshot test in [`tests/export_contracts.rs`](../tests/export_contracts.rs)
   plus `tests/fixtures/exports/<name>.snap`. Regenerate with
   `XAI_DISSECT_WRITE_SNAPSHOTS=1`.
6. If the artifact is on the grok-ozempic ingest path, update
   [`export-contracts.md`](export-contracts.md) and `CHANGELOG.md`. Optional
   companions stay optional; do not silently promote them into the required
   handoff table.

### New Grok-1 slot spec

A slot spec is the per-block occupancy contract: role, dtype, shape, and
expected kind for one of the 12 shards in a transformer block.

1. Confirm the on-disk layout in [`docs/grok1-architecture.md`](grok1-architecture.md)
   (the 12-slot table). Domain docs first; the validator copies that table.
2. Add or edit `Grok1SlotSpec` / `Grok1ExpectedKind` /
   `GROK1_SLOT_SPECS` in [`src/inventory/grok1_coverage.rs`](../src/inventory/grok1_coverage.rs).
   `GROK1_BLOCK_SLOTS` must stay in sync with the array length.
3. If classification cannot see the slot from shape alone, update
   `disambiguate_grok1_moe_projection_slots` / `grok1_moe_projection_for_slot`
   in [`src/inventory/mod.rs`](../src/inventory/mod.rs).
4. Update `canonical_tensors` in
   [`src/inventory/test_fixtures.rs`](../src/inventory/test_fixtures.rs)
   so the 770-tensor fixture still validates.
5. Tests: `src/inventory/grok1_coverage.rs` (`validates_complete_grok1_coverage_manifest`
   plus the fail-path tests for the new mismatch). Coverage counts
   (`GROK1_EXPECTED_*`) change only when the checkpoint cardinality changes.
6. Docs: [`grok1-coverage-manifest.md`](grok1-coverage-manifest.md) and the
   slot table in [`grok1-architecture.md`](grok1-architecture.md).

Changing a slot without updating the fixture will fail the clean-baseline
gate used by `quant-plan` / `pilot-plan` / `route-preservation`.

### New planning validator

Validators run inside `planning::build_grok1_planning_artifacts` *before*
`build_conversion_manifest` / `build_quant_plan`. They must not mutate
weights.

1. Add `validate_*` next to `validate_grok1_clean_baseline`,
   `validate_expert_atlas`, `validate_routing_report`, and
   `validate_readiness_groups` in [`src/planning/mod.rs`](../src/planning/mod.rs).
   Call it from `build_grok1_planning_artifacts`.
2. Keep fail-closed: `bail!` with a message that names the mismatched
   field. Do not warn-and-continue on a clean-baseline invariant.
3. Unit test in `src/planning/mod.rs` following
   `validate_readiness_groups_reports_extra_tensor_keys` (complete inputs
   from `complete_inputs()`, then one broken field).
4. If the validator is also a user-facing command (like
   `route-preservation`), add `cli::run_*`, report/export writers, and a
   row in [`cli-routing.md`](cli-routing.md) / [`run-examples.md`](run-examples.md).
5. Policy lists (`GROK1_KEEP_FP32_ORDER`, `GROK1_PILOT_QUANTIZE_ORDER`,
   `GROK1_DEFER_ORDER`) are separate from validators. Changing *what gets
   quantized* is `quant_policy_for_tensor` / `build_quant_plan`, not a new
   validator.

`pilot-plan` and `route-preservation` call `validate_grok1_clean_baseline`
via their own builders and do **not** currently run the expert/routing/SAAQ
validators. If a new check must apply to those commands too, call it from
`build_grok1_pilot_selection_plan` / `build_grok1_route_preservation_report`
as well, not only from `build_grok1_planning_artifacts`.

## Related docs

| Doc | Role |
| --- | --- |
| [`architecture.md`](architecture.md) | Layer overview and current file tree |
| [`cli-routing.md`](cli-routing.md) | Subcommand → handler → writer map |
| [`run-examples.md`](run-examples.md) | Annotated user-facing CLI |
| [`export-contracts.md`](export-contracts.md) | Stable JSON contracts + grok-ozempic handoff |
| [`output-conventions.md`](output-conventions.md) | Bundle paths and filenames |
| [`tensor-schema.md`](tensor-schema.md) | `TensorKind` classification rules |
| [`grok1-architecture.md`](grok1-architecture.md) | Model / 12-slot layout (not crate layout) |
| [`grok1-coverage-manifest.md`](grok1-coverage-manifest.md) | Coverage gate semantics |
| [`contributing-bot-reviews.md`](contributing-bot-reviews.md) | Review-thread proof before resolve |

Tracked as [GitHub #41](https://github.com/rmems/xai-dissect/issues/41)
and Linear [RM-152](https://linear.app/rpd-34/issue/RM-152).
