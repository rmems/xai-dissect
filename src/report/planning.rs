// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Quant-plan, pilot-plan, route-preservation, and conversion-manifest writers.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;

use super::common::{render_bullet_section, render_notes_section, write_pretty_json, write_text};
use crate::schema::{
    ConversionManifest, Grok1CoverageCounts, PilotSelectionPlan, QuantPlan, QuantPolicy,
    RouteMetricStatus, RoutePreservationReport,
};

/// Write the conversion-ready manifest as pretty-printed JSON.
pub fn write_conversion_manifest_json(manifest: &ConversionManifest, out: &Path) -> Result<()> {
    write_pretty_json(manifest, "serialize conversion manifest to json", out)
}

/// Write the deterministic quant plan as pretty-printed JSON.
pub fn write_quant_plan_json(plan: &QuantPlan, out: &Path) -> Result<()> {
    write_pretty_json(plan, "serialize quant plan to json", out)
}

/// Render a Markdown summary for the deterministic Grok-1 quant plan.
pub fn render_quant_plan_markdown(plan: &QuantPlan) -> String {
    let mut md = String::new();
    render_quant_plan_header(&mut md, plan);
    render_coverage_counts(
        &mut md,
        "Required validation",
        &plan.required_validation,
        &plan.discovered_validation,
    );
    render_quant_plan_kind_lists(&mut md, plan);
    render_notes_section(&mut md, &plan.notes, "None.");
    md
}

/// Write the quant-plan Markdown summary to `out`.
pub fn write_quant_plan_markdown(plan: &QuantPlan, out: &Path) -> Result<()> {
    let s = render_quant_plan_markdown(plan);
    write_text(&s, out)
}

fn render_quant_plan_header(md: &mut String, plan: &QuantPlan) {
    let _ = writeln!(md, "# xai-dissect quant plan");
    let _ = writeln!(md);
    let _ = writeln!(md, "- **model_family**: `{}`", plan.model_family);
    let _ = writeln!(md, "- **checkpoint**: `{}`", plan.checkpoint_path.display());
    let _ = writeln!(md, "- **baseline**: `{}`", plan.baseline);
    let _ = writeln!(md, "- **schema_version**: {}", plan.schema_version);
}

fn render_coverage_counts(
    md: &mut String,
    heading: &str,
    required: &Grok1CoverageCounts,
    discovered: &Grok1CoverageCounts,
) {
    let _ = writeln!(md);
    let _ = writeln!(md, "## {heading}");
    let _ = writeln!(md);
    let _ = writeln!(md, "| Metric | Required | Discovered |");
    let _ = writeln!(md, "| ------ | -------: | ---------: |");
    let _ = writeln!(
        md,
        "| blocks | {} | {} |",
        required.blocks, discovered.blocks
    );
    let _ = writeln!(
        md,
        "| tensors | {} | {} |",
        required.tensors, discovered.tensors
    );
    let _ = writeln!(
        md,
        "| routers | {} | {} |",
        required.routers, discovered.routers
    );
    let _ = writeln!(
        md,
        "| expert_families | {} | {} |",
        required.expert_families, discovered.expert_families
    );
    let _ = writeln!(
        md,
        "| unknown_tensors | {} | {} |",
        required.unknown_tensors, discovered.unknown_tensors
    );
}

fn render_quant_plan_kind_lists(md: &mut String, plan: &QuantPlan) {
    let _ = writeln!(md);
    let _ = writeln!(md, "## Keep fp32");
    let _ = writeln!(md);
    for kind in &plan.keep_fp32 {
        let _ = writeln!(md, "- `{kind}`");
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Pilot quantize");
    let _ = writeln!(md);
    for kind in &plan.pilot_quantize {
        let _ = writeln!(md, "- `{kind}`");
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Defer");
    let _ = writeln!(md);
    for kind in &plan.defer {
        let _ = writeln!(md, "- `{kind}`");
    }
}

/// Render a human-readable Markdown summary of a conversion manifest.
pub fn render_conversion_manifest_markdown(manifest: &ConversionManifest) -> String {
    let mut md = String::new();
    render_conversion_header(&mut md, manifest);
    render_coverage_counts(
        &mut md,
        "Validation",
        &manifest.required_validation,
        &manifest.discovered_validation,
    );
    render_conversion_policy_summary(&mut md, manifest);
    render_bullet_section(&mut md, "Warnings", &manifest.warnings, "None.");
    md
}

/// Write the conversion manifest Markdown summary to `out`.
pub fn write_conversion_manifest_markdown(manifest: &ConversionManifest, out: &Path) -> Result<()> {
    let s = render_conversion_manifest_markdown(manifest);
    write_text(&s, out)
}

fn render_conversion_header(md: &mut String, manifest: &ConversionManifest) {
    let _ = writeln!(md, "# xai-dissect conversion manifest");
    let _ = writeln!(md);
    let _ = writeln!(md, "- **model_family**: `{}`", manifest.model_family);
    let _ = writeln!(
        md,
        "- **checkpoint**: `{}`",
        manifest.checkpoint_path.display()
    );
    let _ = writeln!(
        md,
        "- **baseline_profile**: `{}`",
        manifest.baseline_profile
    );
    let _ = writeln!(md, "- **schema_version**: {}", manifest.schema_version);
    if let Some(ref shape) = manifest.router_shape {
        let _ = writeln!(md, "- **router_shape**: `{}`", shape.render());
    }
    if let Some(ref orientation) = manifest.router_orientation {
        let _ = writeln!(md, "- **router_orientation**: `{}`", orientation.label());
    }
}

fn render_conversion_policy_summary(md: &mut String, manifest: &ConversionManifest) {
    let _ = writeln!(md);
    let _ = writeln!(md, "## Tensor summary");
    let _ = writeln!(md);
    let _ = writeln!(md, "| Policy | Count |");
    let _ = writeln!(md, "| ------ | ----: |");

    let mut policy_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for tensor in &manifest.tensors {
        let policy_name = match tensor.quant_policy {
            QuantPolicy::PassthroughF32Router => "PassthroughF32Router",
            QuantPolicy::PassthroughF32Norm => "PassthroughF32Norm",
            QuantPolicy::CandidateSaaqEmbedding => "CandidateSaaqEmbedding",
            QuantPolicy::WrapExistingInt8Expert => "WrapExistingInt8Expert",
            QuantPolicy::WrapExistingInt8Unknown => "WrapExistingInt8Unknown",
            QuantPolicy::UnknownPassthroughOrWarn => "UnknownPassthroughOrWarn",
        };
        *policy_counts.entry(policy_name).or_insert(0) += 1;
    }
    for (policy, count) in &policy_counts {
        let _ = writeln!(md, "| `{}` | {} |", policy, count);
    }
}

pub fn write_pilot_selection_plan_json(plan: &PilotSelectionPlan, out: &Path) -> Result<()> {
    write_pretty_json(plan, "serialize pilot selection plan to json", out)
}

pub fn write_route_preservation_report_json(
    report_doc: &RoutePreservationReport,
    out: &Path,
) -> Result<()> {
    write_pretty_json(
        report_doc,
        "serialize route preservation report to json",
        out,
    )
}

pub fn render_pilot_selection_plan_markdown(plan: &PilotSelectionPlan) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# xai-dissect Grok-1 pilot selection plan");
    let _ = writeln!(md);
    let _ = writeln!(md, "- **model_family**: `{}`", plan.model_family);
    let _ = writeln!(md, "- **checkpoint**: `{}`", plan.checkpoint_path.display());
    let _ = writeln!(md, "- **baseline**: `{}`", plan.baseline);
    let _ = writeln!(md, "- **schema_version**: {}", plan.schema_version);
    let _ = writeln!(md);
    let _ = writeln!(md, "## Selected blocks");
    let _ = writeln!(md);
    let _ = writeln!(md, "| Block | Label | Rationale |");
    let _ = writeln!(md, "| ----: | ----- | --------- |");
    for block in &plan.selected_blocks {
        let _ = writeln!(
            md,
            "| {} | `{}` | {} |",
            block.block_index, block.label, block.rationale
        );
    }
    let _ = writeln!(md);
    let _ = writeln!(md, "## Modes");
    let _ = writeln!(md);
    for mode in &plan.modes {
        let _ = writeln!(md, "- `{}`", mode.label());
    }
    let _ = writeln!(md);
    let _ = writeln!(md, "## Protection rules");
    let _ = writeln!(md);
    for rule in &plan.protection_rules {
        let _ = writeln!(md, "- {}", rule);
    }
    let _ = writeln!(md);
    let _ = writeln!(md, "## Expected comparison artifacts");
    let _ = writeln!(md);
    for artifact in &plan.comparison_artifacts {
        let _ = writeln!(md, "- `{}`", artifact);
    }
    let _ = writeln!(md);
    let _ = writeln!(md, "## Notes");
    let _ = writeln!(md);
    for note in &plan.notes {
        let _ = writeln!(md, "- {}", note);
    }
    md
}

pub fn write_pilot_selection_plan_markdown(plan: &PilotSelectionPlan, out: &Path) -> Result<()> {
    write_text(&render_pilot_selection_plan_markdown(plan), out)
}

pub fn render_route_preservation_markdown(report_doc: &RoutePreservationReport) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# xai-dissect Grok-1 route-preservation report");
    let _ = writeln!(md);
    let _ = writeln!(md, "- **model_family**: `{}`", report_doc.model_family);
    let _ = writeln!(
        md,
        "- **checkpoint**: `{}`",
        report_doc.checkpoint_path.display()
    );
    let _ = writeln!(md, "- **baseline**: `{}`", report_doc.baseline);
    let _ = writeln!(md, "- **schema_version**: {}", report_doc.schema_version);
    let _ = writeln!(md);

    render_route_metric_section(&mut md, "Router metrics", &report_doc.router_metrics);
    render_route_metric_section(&mut md, "Block metrics", &report_doc.block_metrics);
    render_route_metric_section(&mut md, "Weight metrics", &report_doc.weight_metrics);
    render_route_metric_section(&mut md, "Model metrics", &report_doc.model_metrics);

    let _ = writeln!(md);
    let _ = writeln!(md, "## Notes");
    let _ = writeln!(md);
    for note in &report_doc.notes {
        let _ = writeln!(md, "- {}", note);
    }
    md
}

pub fn write_route_preservation_markdown(
    report_doc: &RoutePreservationReport,
    out: &Path,
) -> Result<()> {
    write_text(&render_route_preservation_markdown(report_doc), out)
}

fn render_route_metric_section(md: &mut String, title: &str, items: &[RouteMetricStatus]) {
    let _ = writeln!(md);
    let _ = writeln!(md, "## {title}");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Metric | Scope | Status | Threshold | Observed | Detail |"
    );
    let _ = writeln!(
        md,
        "| ------ | ----- | ------ | --------- | -------- | ------ |"
    );
    for item in items {
        let _ = writeln!(
            md,
            "| `{}` | {} | {} | {} | {} | {} |",
            item.name,
            item.scope,
            metric_status_label(item.status),
            item.threshold.as_deref().unwrap_or("-"),
            item.observed.as_deref().unwrap_or("-"),
            item.detail
        );
    }
}

fn metric_status_label(status: crate::schema::MetricStatus) -> &'static str {
    match status {
        crate::schema::MetricStatus::Pass => "pass",
        crate::schema::MetricStatus::Fail => "fail",
        crate::schema::MetricStatus::Unknown => "unknown",
    }
}
