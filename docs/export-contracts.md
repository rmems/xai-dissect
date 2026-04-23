# Export Contracts

`xai-dissect` treats exported artifacts as the primary stable integration
surface for downstream repos. The in-process Rust API is intentionally smaller
and may evolve faster than the CLI artifact layer.

Every top-level JSON document includes a `schema_version` field. Consumers
should key compatibility checks off that field, not off incidental ordering or
internal module layout.

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

## Inventory

- `exports/<slug>/inventory.json`
  Contract: `schema::ModelInventory`
- `reports/<slug>/inventory.md`
  Contract: `report::render_markdown`
- `exports/<slug>/inventory-findings.json`
  Contract: `schema::FindingsSummary` with `analysis = "inventory"`
- `manifests/<slug>/checkpoint-inventory-snapshot.json`
  Contract: `schema::CheckpointInventorySnapshot`

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
