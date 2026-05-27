# Export Contracts

`xai-dissect` treats exported artifacts as the primary stable integration
surface for downstream repos. The in-process Rust API is intentionally smaller
and may evolve faster than the CLI artifact layer.

Every top-level JSON document includes a `schema_version` field. Some
documents, such as `grok1-coverage.json`, also carry a second
document-specific version field. Consumers should key compatibility checks off
the documented version fields, not off incidental ordering or internal module
layout.

## Unified Output Tree

When `--output-root <dir>` is provided, artifacts are written under:

```text
<dir>/
  reports/<checkpoint_slug>/
  exports/<checkpoint_slug>/
  manifests/<checkpoint_slug>/
```

`docs/output-conventions.md` defines the directory and filename conventions.
This document defines the schema contracts behind those files.

## grok-ozempic Handoff Contract

Issue `#21` defines a smaller contract profile than the full export surface.
For the `grok-ozempic` handoff, `xai-dissect` must produce a complete Grok-1
bundle for one `<checkpoint_slug>`, and downstream ingest must key that bundle
off the directory slug plus tensor locators, not off the host-local
`checkpoint_path`.

### Required Machine-Ingest Artifacts

| Path | Purpose in `grok-ozempic` | Contract | Version expectation | Checksum expectation |
| --- | --- | --- | --- | --- |
| `exports/<slug>/inventory.json` | Canonical tensor catalog for deciding what exists, where it lives, and how big it is before any packing or quantization step. | `schema::ModelInventory` | `schema_version = 2` | No embedded checksum in v1. Require the sibling `grok1-coverage.json` gate for complete Grok-1 handoff bundles. |
| `exports/<slug>/experts.json` | Canonical MoE mapping for expanding expert families into resolved `gate` / `down` / `up` slices without re-inferring layout from shape alone. | `schema::ExpertAtlas` | `schema_version = 1` | No embedded checksum in v1. Must agree with the same-slug `inventory.json` locators and dimensions. |
| `manifests/<slug>/routing-critical-tensors.json` | Compact routing guardrail list for tensors that downstream compression or packing logic must treat as routing-sensitive. | `schema::RoutingCriticalTensorManifest` | `schema_version = 1` | No embedded checksum in v1. Must agree with the same-slug inventory and coverage facts. |
| `manifests/<slug>/grok1-coverage.json` | Fail-closed completeness and integrity proof that the bundle came from a recognized complete Grok-1 parse. | `schema::Grok1CoverageManifest` | `schema_version = 2`, `coverage_schema_version = 2`, and `baseline_profile = grok1-map-v1-clean` | Embedded `checksum` field is required. Downstream should reject bundles whose coverage validation is not `pass`. |

### Bundle Rules

- All required files must exist under the same `<checkpoint_slug>` across
  `exports/` and `manifests/`.
- `model_family` must be `"grok-1"` in every required JSON document for this
  handoff profile.
- `inventory.json` is the canonical tensor table. Downstream tensor identity is
  `shard_ordinal` plus `in_shard_index`; `kind`, `shape`, `role`,
  `block_index`, and `block_slot` are the stable classification helpers.
- `experts.json` is the canonical expert layout map. For MoE packing plans,
  downstream must use the resolved `projection` values and `source_*` locator
  fields rather than re-deriving projection identity from shape alone.
- `routing-critical-tensors.json` is a guardrail list, not the source of truth
  for the whole checkpoint. `criticality_reason` is explanatory; the stable
  ingest fields are tensor identity, `structural_name`, `orientation`,
  `linked_expert_count`, `block_index`, and `block_slot`.
- `grok1-coverage.json` is a fail-closed gate. Downstream should require
  `validation = "pass"`, `expected == discovered`, and `unknown_slots = []`
  before treating the bundle as complete enough for `grok-ozempic` ingestion.
- `checkpoint_path` is informative and machine-local. `grok-ozempic` should
  not use it as a cache key, artifact identifier, or portability boundary.

### Optional Companion Artifacts

- `manifests/<slug>/checkpoint-inventory-snapshot.json`: compact summary for
  dashboards or quick sanity checks. It is not sufficient by itself for
  ingestion because it omits the full tensor table and expert slice mapping.
- `exports/<slug>/routing-report.json`: richer routing analysis companion for
  review and debugging. It is not required for ingest because the normative
  routing guardrail list lives in `routing-critical-tensors.json`.
- `reports/<slug>/*.md` and `exports/<slug>/*-findings.json`: human-review and
  summary outputs only, not machine-ingest inputs.
- `exports/<slug>/stats.json`, `exports/<slug>/saaq-readiness.json`, and
  `manifests/<slug>/candidate-saaq-targets.json`: exploratory analysis outputs
  intentionally left out of the v1 `grok-ozempic` contract so downstream work
  does not depend on sampling heuristics or SAAQ-oriented scoring.

### Validated Example

This handoff contract was checked against the real output tree under
`out/grok1_run2_after_fixes_20260525T002904Z` with
`<checkpoint_slug> = grok-1-official__ckpt-0`. The coverage manifest for that
run recorded `validation = "pass"`, `expected.tensors = discovered.tensors =
770`, and checksum `fnv1a64:de5a1c978121c62c`. Newly emitted coverage manifests
from this contract carry `coverage_schema_version = 2` and the clean baseline
label `baseline_profile = grok1-map-v1-clean` so downstream parsers can branch
cleanly on the versioned payload shape.

## Inventory

Current schema version: **2**.

- `exports/<slug>/inventory.json`
  Contract: `schema::ModelInventory`
- `reports/<slug>/inventory.md`
  Contract: `report::render_markdown`
- `exports/<slug>/inventory-findings.json`
  Contract: `schema::FindingsSummary` with `analysis = "inventory"`
- `manifests/<slug>/checkpoint-inventory-snapshot.json`
  Contract: `schema::CheckpointInventorySnapshot`
- `manifests/<slug>/grok1-coverage.json`
  Contract: `schema::Grok1CoverageManifest` for complete Grok-1 inventories

## Experts

- `exports/<slug>/experts.json`
  Contract: `schema::ExpertAtlas`
- `reports/<slug>/experts.md`
  Contract: `report::render_expert_markdown`
- `exports/<slug>/experts-findings.json`
  Contract: `schema::FindingsSummary` with `analysis = "experts"`

## Routing Report

- `exports/<slug>/routing-report.json`
  Contract: `schema::RoutingReport`
- `reports/<slug>/routing-report.md`
  Contract: `report::render_routing_markdown`
- `exports/<slug>/routing-report-findings.json`
  Contract: `schema::FindingsSummary` with `analysis = "routing-report"`
- `manifests/<slug>/routing-critical-tensors.json`
  Contract: `schema::RoutingCriticalTensorManifest`

## Stats

- `exports/<slug>/stats.json`
  Contract: `schema::StatsProfileReport`
- `reports/<slug>/stats.md`
  Contract: `report::render_stats_markdown`
- `exports/<slug>/stats-findings.json`
  Contract: `schema::FindingsSummary` with `analysis = "stats"`

## SAAQ Readiness

- `exports/<slug>/saaq-readiness.json`
  Contract: `schema::SaaqReadinessReport`
- `reports/<slug>/saaq-readiness.md`
  Contract: `report::render_saaq_readiness_markdown`
- `exports/<slug>/saaq-readiness-findings.json`
  Contract: `schema::FindingsSummary` with `analysis = "saaq-readiness"`
- `manifests/<slug>/candidate-saaq-targets.json`
  Contract: `schema::CandidateTensorManifest`

## Pilot Planning

- `manifests/<slug>/pilot-selection-plan.json`
  Contract: `schema::PilotSelectionPlan`
- `reports/<slug>/pilot-selection-plan.md`
  Contract: `report::render_pilot_selection_plan_markdown`

## Route Preservation Gate

- `manifests/<slug>/route-preservation-report.json`
  Contract: `schema::RoutePreservationReport`
- `reports/<slug>/route-preservation-report.md`
  Contract: `report::render_route_preservation_markdown`

## Quant Planning

- `manifests/<slug>/conversion-manifest.json`
  Contract: `schema::ConversionManifest`
- `manifests/<slug>/quant-plan.json`
  Contract: `schema::QuantPlan`
- `reports/<slug>/quant-plan.md`
  Contract: `report::render_quant_plan_markdown`

These planning artifacts are downstream of the core Grok-1 handoff contract.
They depend on the clean baseline profile `grok1-map-v1-clean`, the resolved
expert layout, and the grouped readiness buckets. They are meant to drive
policy selection and pilot-scoping work without widening `xai-dissect` into a
runtime or checkpoint-mutation tool.

## Stability Rules

- Adding new top-level artifact files is a contract change and must be called
  out in `CHANGELOG.md`.
- Incompatible JSON shape changes require a `schema_version` bump on the
  affected top-level document type.
- Markdown is human-readable rather than schema-tagged, but section structure
  and filenames are still treated as stable enough for downstream review and
  automation.
- The export bundle path conventions are intentionally more stable than the
  current in-process Rust module layout.

## Test Coverage

The repo includes fixture-driven snapshot tests for representative bundles and
a tiny synthetic parser fixture so the export surface can be exercised without
real Grok weights.
