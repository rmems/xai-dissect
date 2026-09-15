// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Markdown and JSON export writers for all xai-dissect analysis outputs.
//!
//! This module is the only layer that produces files on disk. It serializes
//! the stable schema types to JSON (machine-ingest) and renders Markdown
//! (human-review and PR discussion). Each render function is versioned and
//! documented so downstream tooling can predict the section structure.
//!
//! ## Stability guarantees
//! - Section headings in Markdown outputs are treated as stable surface
//!   identifiers (not schema-tagged, but intentionally stable)
//! - Filenames under `exports/` and `manifests/` follow `docs/output-conventions.md`
//! - The JSON export schema is the normative machine-ingest contract;
//!   Markdown is a human-readable companion, not a machine interface
//!
//! ## Render functions
//! | Function | Output type | File path |
//! |----------|-------------|-----------|
//! | `render_markdown` | `inventory.md` | `reports/<slug>/` |
//! | `render_expert_markdown` | `experts.md` | `reports/<slug>/` |
//! | `render_routing_markdown` | `routing-report.md` | `reports/<slug>/` |
//! | `render_stats_markdown` | `stats.md` | `reports/<slug>/` |
//! | `render_saaq_readiness_markdown` | `saaq-readiness.md` | `reports/<slug>/` |
//! | `render_pilot_selection_plan_markdown` | `pilot-selection-plan.md` | `reports/<slug>/` |
//! | `render_route_preservation_markdown` | `route-preservation-report.md` | `reports/<slug>/` |
//! | `render_quant_plan_markdown` | `quant-plan.md` | `reports/<slug>/` |

mod common;
mod experts;
mod inventory;
mod planning;
mod routing;
mod saaq;
mod stats;

pub use experts::{render_expert_markdown, write_expert_json, write_expert_markdown};
pub use inventory::{
    render_markdown, write_findings_summary_json, write_grok1_coverage_manifest_json,
    write_inventory_snapshot_manifest_json, write_json, write_markdown,
};
pub use planning::{
    render_conversion_manifest_markdown, render_pilot_selection_plan_markdown,
    render_quant_plan_markdown, render_route_preservation_markdown, write_conversion_manifest_json,
    write_conversion_manifest_markdown, write_pilot_selection_plan_json,
    write_pilot_selection_plan_markdown, write_quant_plan_json, write_quant_plan_markdown,
    write_route_preservation_markdown, write_route_preservation_report_json,
};
pub use routing::{
    render_routing_markdown, write_routing_critical_manifest_json, write_routing_json,
    write_routing_markdown,
};
pub use saaq::{
    render_saaq_readiness_markdown, write_candidate_manifest_json, write_saaq_readiness_json,
    write_saaq_readiness_markdown,
};
pub use stats::{render_stats_markdown, write_stats_json, write_stats_markdown};
