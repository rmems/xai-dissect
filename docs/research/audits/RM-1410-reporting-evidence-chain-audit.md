# RM-1410: xai-dissect reporting truthfulness and evidence-chain audit

**Date:** 2026-11-09  
**Repo:** [rmems/xai-dissect](https://github.com/rmems/xai-dissect)  
**Baseline commit audited:** `2dd71ec` (`chore: add CodeRabbit config (#80)`) on `main`  
**Method:** architecture/docs review, export-contract snapshot tests, report renderer branch tests, `build_*` symbol trace, `cargo test --locked` / `clippy` on Rust 1.99.1

## Executive summary

The reporting layer is **mostly truthful**: Markdown renderers are projections of typed schema structs, golden snapshots lock JSON+MD bundles, and route-preservation metrics honestly use `status: unknown` until downstream pilots supply observations. The main risks are **companion-artifact gaps** (`*-findings.json` and console summaries omit lossy-parse signals that full reports already carry) and **human gate docs** (`GO_NO_GO.md`) that list runtime thresholds beside structural prerequisites without a single “structural-only ≠ GO” banner. One narrow fix for inventory findings is included on branch `cursor/inventory-findings-skipped-anchors-e498`.

## Evidence chain (claim → source → command)

| Claim surface | Canonical evidence | Executable check |
|---------------|-------------------|------------------|
| Tensor inventory counts/kinds | `exports/<slug>/inventory.json` (`schema::ModelInventory`) | `cargo test --locked export_contracts` |
| MoE / expert geometry | `exports/<slug>/experts.json` | same + `experts.snap` |
| Routing-critical guardrails | `manifests/<slug>/routing-critical-tensors.json` | `routing-report.snap` |
| Completeness gate | `manifests/<slug>/grok1-coverage.json` (`validation = pass`) | `families::grok1::validate_grok1_clean_baseline` |
| SAAQ scouting (heuristic) | `exports/<slug>/saaq-readiness.json` v2 `quantization_candidates` | `stats::build_saaq_readiness_report` + serde alias tests |
| Pilot / quant policy | `manifests/<slug>/quant-plan.json`, `conversion-manifest.json` | `planning::build_grok1_planning_artifacts` |
| Route-preservation **definitions** | `manifests/<slug>/route-preservation-report.json` | `planning::build_grok1_route_preservation_report` |
| Human-readable summaries | `reports/<slug>/*.md` via `report::render_*` | `report_render_branches`, export snapshots |

Pipeline invariant (documented): `docs/architecture.md` — parser → schema → inventory → analyzers → report/exports; no inference path.

## Ranked findings

Severity uses **confidence** (high/medium/low) and **blast radius** (who misreads if unfixed).

### F1 — `inventory-findings.json` omitted skipped parser anchors (fixed this session)

| Field | Value |
|-------|-------|
| Confidence | **High** |
| Blast radius | Automation/dashboards that ingest only `*-findings.json` miss lossy permissive parses |
| Evidence | `inventory.md` exposes `skipped_anchor_count` + table (`src/report/inventory.rs`); snapshot `inventory.json` carries `skipped_anchors`; `build_inventory_findings_summary` (`src/exports/mod.rs`) had no branch for `skipped_anchor_count > 0` until fix |
| Fix | Warn finding `skipped_parser_anchors` + unit test `inventory_findings_warn_when_parser_skipped_anchors` |
| Branch | `cursor/inventory-findings-skipped-anchors-e498` |

### F2 — Route-preservation Markdown omits structural `required_validation` counts present in JSON

| Field | Value |
|-------|-------|
| Confidence | **High** |
| Blast radius | Reviewers reading only `.md` may not see blocks/tensors/routers/expert_families gate embedded in JSON |
| Evidence | `RoutePreservationReport.required_validation` populated in `planning::build_grok1_route_preservation_report`; `render_route_preservation_markdown` (`src/report/planning.rs:252-277`) renders metric tables only — no `render_coverage_counts` unlike `render_quant_plan_markdown` |
| Command | `cargo test --locked pilot_route_reports` (thresholds in MD); compare to `route-preservation.snap` JSON header |
| Follow-up | RM-1410-F2 (below) |

### F3 — `GO_NO_GO.md` mixes structural prerequisites with runtime thresholds

| Field | Value |
|-------|-------|
| Confidence | **High** |
| Blast radius | Sprint owners may treat emitted `route-preservation-report.json` as satisfying router agreement / cosine rows |
| Evidence | `docs/GO_NO_GO.md` lists `>= 99.0%` agreement beside structural counts; JSON defaults all metrics to `MetricStatus::Unknown` (`planning/mod.rs:605-731`); notes explicitly: “Statuses remain unknown until downstream pilot artifacts supply the observed values.” (`route-preservation.snap`) |
| Mitigation today | `docs/metric-definitions.md` “What xai-dissect guarantees vs. what it requires” |
| Follow-up | RM-1410-F3 (below) |

### F4 — Console summaries under-report parser loss and expert check failures

| Field | Value |
|-------|-------|
| Confidence | **Medium** |
| Blast radius | CLI-only operators |
| Evidence | `print_console_summary` (`src/cli/summary.rs:57-85`) warns on `Unknown` kinds but not `skipped_anchor_count`; `print_expert_console_summary` prints count of **passed** naming checks, not failures |
| Follow-up | RM-1410-F4 (optional) |

### F5 — No `route-preservation-report-findings.json` companion

| Field | Value |
|-------|-------|
| Confidence | **High** |
| Blast radius | Low today (artifact is manifest-tier, notes are explicit); increases if CI starts scraping findings only |
| Evidence | `write_route_preservation_bundle` writes JSON+MD only (`src/exports/mod.rs:369-386`); other analyses emit `*-findings.json` |
| Follow-up | RM-1410-F5 (optional) |

### F6 — Schema module comment drift on `SaaqReadinessReport` version

| Field | Value |
|-------|-------|
| Confidence | **High** |
| Blast radius | Maintainer confusion only; runtime emits `schema_version: 2` per `stats::SAAQ_READINESS_SCHEMA_VERSION` and `docs/export-contracts.md` |
| Evidence | `src/schema/mod.rs` header table said v1; corrected to v2 in doc fix on fix branch |
| Follow-up | N/A (doc-only) |

### F7 — SAAQ readiness scores are sampling-heuristic, not calibration evidence

| Field | Value |
|-------|-------|
| Confidence | **High** |
| Blast radius | Downstream over-trust of readiness/opportunity/risk floats |
| Evidence | `docs/export-contracts.md` excludes saaq/stats from grok-ozempic v1 handoff; `HYP-XD-001` falsifiers; builder in `src/stats/mod.rs` |
| Mitigation | Contract + disposition enums; findings headline uses “candidate” language |
| Follow-up | None (documented non-goal) |

### F8 — Findings `schema_version` mirrors source document (intentional but sharp edge)

| Field | Value |
|-------|-------|
| Confidence | **High** |
| Blast radius | Consumers treating findings `schema_version` as a unified contract version |
| Evidence | `src/schema/mod.rs:30-43` explicit warning; `docs/export-contracts.md` classifies findings as non-ingest |
| Follow-up | None |

## Ordering stability and provenance

- **Inventory tensor order:** deterministic shard walk (`inventory::build_inventory`); stats tests assert interleaved shard order preserved (`stats/mod.rs` `stats_report_preserves_interleaved_shard_order_and_values`).
- **SAAQ candidates:** explicit `rank` field; empty-input fallback documented in `stats/mod.rs`.
- **Provenance fields:** tensor identity uses `shard_ordinal` + `in_shard_index` across exports (contract in `export-contracts.md`).
- **Snapshot regression:** `tests/export_contracts.rs` + `tests/fixtures/exports/*.snap` provide reproducible evidence without real weights.

## Verification run (this audit session)

```bash
bash .agents/setup   # Rust stable + fetch
cargo fmt --check
cargo test --locked  # all passed
cargo clippy --all-targets --all-features -- -D warnings
```

## Proposed follow-up issues (non-duplicate)

### RM-1410-F2: Surface `required_validation` in route-preservation Markdown

**Acceptance criteria**

- `reports/<slug>/route-preservation-report.md` includes a “Required validation” table matching JSON `required_validation` counts.
- `tests/fixtures/exports/route-preservation.snap` updated.
- No change to JSON schema version.

### RM-1410-F3: Structural-only banner on GO/NO-GO doc

**Acceptance criteria**

- `docs/GO_NO_GO.md` opens with explicit distinction: structural emission ≠ runtime GO.
- Links to `metric-definitions.md` and route-preservation notes.
- PR checklist block tags runtime rows as “downstream pilot evidence required”.

### RM-1410-F4: CLI summary parity for skipped anchors and failed expert checks

**Acceptance criteria**

- `print_console_summary` emits warn when `skipped_anchor_count > 0`.
- `print_expert_console_summary` reports failed naming check count (or renames field to `naming_checks_passed`).
- Unit tests in `src/cli/summary.rs` or `cli/tests.rs`.

### RM-1410-F5: Optional route-preservation findings companion

**Acceptance criteria**

- `exports/<slug>/route-preservation-findings.json` with headline + warn on any `MetricStatus::Unknown` for thresholded metrics.
- Documented in `export-contracts.md` as optional companion (not grok-ozempic ingest).

## Implemented fix (this session)

- **Branch:** `cursor/inventory-findings-skipped-anchors-e498`
- **Change:** `build_inventory_findings_summary` warns on `skipped_anchor_count > 0`; schema doc table fix for SAAQ v2; unit test added.
