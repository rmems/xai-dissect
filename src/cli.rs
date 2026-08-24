// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Shared CLI helpers for inventory-backed subcommands.
//!
//! Keeps clap flag names, help text, and artifact paths unchanged while
//! collapsing the repeated resolve → inventory → analyze → write loop.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::Args;

use xai_dissect::experts::build_expert_atlas;
use xai_dissect::exports::{self, OutputBundle};
use xai_dissect::inventory::{InventoryConfig, build_inventory};
use xai_dissect::planning::{
    build_grok1_pilot_selection_plan, build_grok1_planning_artifacts,
    build_grok1_route_preservation_report,
};
use xai_dissect::report;
use xai_dissect::routing::build_routing_report;
use xai_dissect::schema::{
    ExpertAtlas, ModelInventory, PilotSelectionPlan, QuantPlan, RoutePreservationReport,
    RoutingReport, SaaqReadinessReport, StatsProfileReport, TensorInfo, TensorKind,
};
use xai_dissect::stats::{StatsConfig, build_saaq_readiness_report, build_stats_report};

const FULL_TREE_DIRS: &str = "{reports,exports,manifests}";
const PLANNING_TREE_DIRS: &str = "{reports,manifests}";

#[derive(Args, Debug, Clone)]
pub struct OutputTreeArgs {
    /// If set, also write artifacts into
    /// `<root>/{reports,exports,manifests}/<checkpoint-slug>/...`.
    #[arg(long)]
    pub output_root: Option<PathBuf>,
    /// Optional override for the checkpoint slug used under the unified
    /// output tree. If unset, the slug is inferred from the checkpoint path.
    #[arg(long, requires = "output_root")]
    pub checkpoint_slug: Option<String>,
}

#[derive(Args, Debug, Clone)]
pub struct CheckpointScanArgs {
    /// Checkpoint directory (e.g. `/path/to/grok-1/ckpt-0`).
    pub path: PathBuf,
    /// Filename prefix filter.
    #[arg(long, default_value = "tensor")]
    pub prefix: String,
    /// Only process the first N shards (sorted by filename).
    #[arg(long)]
    pub limit: Option<usize>,
}

#[derive(Args, Debug, Clone)]
pub struct ModelFamilyArg {
    /// Model family tag written into the export header. Only `grok-1`
    /// is officially supported today.
    #[arg(long, default_value = "grok-1")]
    pub family: String,
}

#[derive(Args, Debug, Clone)]
pub struct PlanningFamilyArg {
    /// Model family tag written into the export header.
    #[arg(long, default_value = "grok-1")]
    pub family: String,
}

#[derive(Args, Debug, Clone)]
pub struct SampleValuesArg {
    /// Maximum sampled values per tensor.
    #[arg(long, default_value_t = 65_536)]
    pub sample_values: usize,
}

pub fn run_inventory(
    scan: &CheckpointScanArgs,
    family: &str,
    json_out: Option<&Path>,
    md_out: Option<&Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    let inv = run_inventory_command(scan, family, None)?;
    print_console_summary(&inv);
    write_json_and_markdown(
        &inv,
        json_out,
        md_out,
        "JSON inventory",
        "Markdown report",
        report::write_json,
        report::write_markdown,
        report::render_markdown,
    )?;
    write_output_tree(
        output_tree,
        "inventory bundle",
        FULL_TREE_DIRS,
        |root, slug| exports::write_inventory_bundle(&inv, root, slug),
    )
}

pub fn run_experts(
    scan: &CheckpointScanArgs,
    family: &str,
    json_out: Option<&Path>,
    md_out: Option<&Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    let inv = run_inventory_command(scan, family, None)?;
    let atlas = build_expert_atlas(&inv);
    print_expert_console_summary(&atlas);
    write_json_and_markdown(
        &atlas,
        json_out,
        md_out,
        "JSON expert atlas",
        "Markdown expert atlas",
        report::write_expert_json,
        report::write_expert_markdown,
        report::render_expert_markdown,
    )?;
    write_output_tree(
        output_tree,
        "expert bundle",
        FULL_TREE_DIRS,
        |root, slug| exports::write_expert_bundle(&atlas, root, slug),
    )
}

pub fn run_routing_report(
    scan: &CheckpointScanArgs,
    family: &str,
    json_out: Option<&Path>,
    md_out: Option<&Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    let inv = run_inventory_command(scan, family, None)?;
    let report_doc = build_routing_report(&inv);
    print_routing_console_summary(&report_doc);
    write_json_and_markdown(
        &report_doc,
        json_out,
        md_out,
        "JSON routing report",
        "Markdown routing report",
        report::write_routing_json,
        report::write_routing_markdown,
        report::render_routing_markdown,
    )?;
    write_output_tree(
        output_tree,
        "routing bundle",
        FULL_TREE_DIRS,
        |root, slug| exports::write_routing_bundle(&report_doc, root, slug),
    )
}

pub fn run_stats(
    scan: &CheckpointScanArgs,
    family: &str,
    sample_values: usize,
    json_out: Option<&Path>,
    md_out: Option<&Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    let inv = run_inventory_command(scan, family, None)?;
    let report_doc = build_stats_report(&inv, &stats_config(sample_values))?;
    print_stats_console_summary(&report_doc);
    write_json_and_markdown(
        &report_doc,
        json_out,
        md_out,
        "JSON stats report",
        "Markdown stats report",
        report::write_stats_json,
        report::write_stats_markdown,
        report::render_stats_markdown,
    )?;
    write_output_tree(output_tree, "stats bundle", FULL_TREE_DIRS, |root, slug| {
        exports::write_stats_bundle(&report_doc, root, slug)
    })
}

pub fn run_saaq_readiness(
    scan: &CheckpointScanArgs,
    family: &str,
    sample_values: usize,
    json_out: Option<&Path>,
    md_out: Option<&Path>,
    manifest_out: Option<&Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    let inv = run_inventory_command(scan, family, None)?;
    let stats = build_stats_report(&inv, &stats_config(sample_values))?;
    let readiness = build_saaq_readiness_report(&inv, &stats);
    print_saaq_console_summary(&readiness);
    write_json_and_markdown(
        &readiness,
        json_out,
        md_out,
        "JSON SAAQ-readiness report",
        "Markdown SAAQ-readiness report",
        report::write_saaq_readiness_json,
        report::write_saaq_readiness_markdown,
        report::render_saaq_readiness_markdown,
    )?;
    write_optional_file(manifest_out, "candidate manifest", |p| {
        report::write_candidate_manifest_json(&readiness.manifest, p)
    })?;
    write_output_tree(output_tree, "saaq bundle", FULL_TREE_DIRS, |root, slug| {
        exports::write_saaq_bundle(&readiness, root, slug)
    })
}

pub fn run_pilot_plan(
    scan: &CheckpointScanArgs,
    family: &str,
    json_out: Option<&Path>,
    md_out: Option<&Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    let inv = run_inventory_command(scan, family, Some("pilot-plan"))?;
    let plan = build_grok1_pilot_selection_plan(&inv)?;
    print_pilot_plan_console_summary(&plan);
    write_json_and_markdown(
        &plan,
        json_out,
        md_out,
        "JSON pilot selection plan",
        "Markdown pilot selection plan",
        report::write_pilot_selection_plan_json,
        report::write_pilot_selection_plan_markdown,
        report::render_pilot_selection_plan_markdown,
    )?;
    write_output_tree(
        output_tree,
        "pilot-plan bundle",
        PLANNING_TREE_DIRS,
        |root, slug| exports::write_pilot_plan_bundle(&plan, root, slug),
    )
}

pub fn run_route_preservation(
    scan: &CheckpointScanArgs,
    family: &str,
    json_out: Option<&Path>,
    md_out: Option<&Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    let inv = run_inventory_command(scan, family, Some("route-preservation"))?;
    let report_doc = build_grok1_route_preservation_report(&inv)?;
    print_route_preservation_console_summary(&report_doc);
    write_json_and_markdown(
        &report_doc,
        json_out,
        md_out,
        "JSON route-preservation report",
        "Markdown route-preservation report",
        report::write_route_preservation_report_json,
        report::write_route_preservation_markdown,
        report::render_route_preservation_markdown,
    )?;
    write_output_tree(
        output_tree,
        "route-preservation bundle",
        PLANNING_TREE_DIRS,
        |root, slug| exports::write_route_preservation_bundle(&report_doc, root, slug),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_quant_plan(
    scan: &CheckpointScanArgs,
    family: &str,
    sample_values: usize,
    json_out: Option<&Path>,
    md_out: Option<&Path>,
    conversion_manifest_out: Option<&Path>,
    conversion_manifest_md_out: Option<&Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    let inv = run_inventory_command(scan, family, Some("quant-plan"))?;
    let atlas = build_expert_atlas(&inv);
    let routing = build_routing_report(&inv);
    let stats = build_stats_report(&inv, &stats_config(sample_values))?;
    let readiness = build_saaq_readiness_report(&inv, &stats);
    let (conversion_manifest, quant_plan) =
        build_grok1_planning_artifacts(&inv, &atlas, &routing, &readiness)?;

    print_quant_plan_console_summary(&quant_plan, &conversion_manifest);
    write_optional_file(conversion_manifest_out, "conversion manifest", |p| {
        report::write_conversion_manifest_json(&conversion_manifest, p)
    })?;
    write_optional_file(
        conversion_manifest_md_out,
        "conversion manifest Markdown",
        |p| report::write_conversion_manifest_markdown(&conversion_manifest, p),
    )?;
    write_json_and_markdown(
        &quant_plan,
        json_out,
        md_out,
        "JSON quant plan",
        "Markdown quant plan",
        report::write_quant_plan_json,
        report::write_quant_plan_markdown,
        report::render_quant_plan_markdown,
    )?;
    write_output_tree(
        output_tree,
        "quant-plan bundle",
        FULL_TREE_DIRS,
        |root, slug| {
            exports::write_quant_plan_bundle(&conversion_manifest, &quant_plan, root, slug)
        },
    )
}

pub(crate) fn validate_complete_inventory_scope(
    command: &str,
    prefix: &str,
    limit: Option<usize>,
) -> Result<()> {
    if prefix != "tensor" {
        bail!(
            "{command} requires a complete Grok-1 inventory; `--prefix {}` is not supported",
            prefix
        );
    }
    if let Some(limit) = limit {
        bail!("{command} requires a complete Grok-1 inventory; `--limit {limit}` is not supported");
    }
    Ok(())
}

fn run_inventory_command(
    scan: &CheckpointScanArgs,
    family: &str,
    complete_command: Option<&str>,
) -> Result<ModelInventory> {
    if let Some(command) = complete_command {
        validate_complete_inventory_scope(command, &scan.prefix, scan.limit)?;
    }
    let cfg = InventoryConfig {
        prefix: scan.prefix.clone(),
        limit: scan.limit,
        model_family: family.to_string(),
    };
    build_inventory(&scan.path, &cfg)
}

fn stats_config(sample_values: usize) -> StatsConfig {
    StatsConfig {
        max_sample_values: sample_values,
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn write_json_and_markdown<T>(
    doc: &T,
    json_out: Option<&Path>,
    md_out: Option<&Path>,
    json_label: &str,
    md_label: &str,
    write_json: fn(&T, &Path) -> Result<()>,
    write_md: fn(&T, &Path) -> Result<()>,
    render_md: fn(&T) -> String,
) -> Result<()> {
    write_optional_file(json_out, json_label, |p| write_json(doc, p))?;
    if let Some(path) = md_out {
        write_md(doc, path)?;
        eprintln!("wrote {md_label} -> {}", path.display());
    } else {
        println!();
        println!("{}", render_md(doc));
    }
    Ok(())
}

fn write_optional_file(
    path: Option<&Path>,
    label: &str,
    write: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    if let Some(path) = path {
        write(path)?;
        eprintln!("wrote {label} -> {}", path.display());
    }
    Ok(())
}

fn write_output_tree(
    output_tree: &OutputTreeArgs,
    label: &str,
    dirs: &str,
    write: impl FnOnce(&Path, Option<&str>) -> Result<OutputBundle>,
) -> Result<()> {
    if let Some(root) = output_tree.output_root.as_deref() {
        let bundle = write(root, output_tree.checkpoint_slug.as_deref())?;
        eprintln!(
            "wrote {label} -> {}/{dirs}/{}/...",
            root.display(),
            bundle.checkpoint_slug
        );
    }
    Ok(())
}

fn print_quant_plan_console_summary(
    quant_plan: &QuantPlan,
    conversion_manifest: &xai_dissect::schema::ConversionManifest,
) {
    eprintln!(
        "checkpoint: {}  baseline: {}  tensors: {}",
        quant_plan.checkpoint_path.display(),
        quant_plan.baseline,
        conversion_manifest.tensors.len(),
    );
    eprintln!(
        "keep_fp32: {}  pilot_quantize: {}  defer: {}",
        quant_plan.keep_fp32.len(),
        quant_plan.pilot_quantize.len(),
        quant_plan.defer.len(),
    );
    if !conversion_manifest.warnings.is_empty() {
        eprintln!(
            "warn: conversion manifest emitted {} warning categories",
            conversion_manifest.warnings.len()
        );
    }
}

fn print_pilot_plan_console_summary(plan: &PilotSelectionPlan) {
    eprintln!(
        "checkpoint: {}  baseline: {}  selected_blocks: {}  modes: {}",
        plan.checkpoint_path.display(),
        plan.baseline,
        plan.selected_blocks.len(),
        plan.modes.len(),
    );
}

fn print_route_preservation_console_summary(report_doc: &RoutePreservationReport) {
    eprintln!(
        "model_family: {}  baseline: {}  router_metrics: {}  block_metrics: {}",
        report_doc.model_family,
        report_doc.baseline,
        report_doc.router_metrics.len(),
        report_doc.block_metrics.len(),
    );
}

fn print_console_summary(inv: &ModelInventory) {
    eprintln!(
        "checkpoint: {}  shards: {}  tensors: {}",
        inv.checkpoint_path.display(),
        inv.shard_count,
        inv.totals.tensors,
    );
    let hp = &inv.inferred;
    eprintln!(
        "inferred:  vocab={:?}  d_model={:?}  n_experts={:?}  d_ff={:?}  n_blocks={:?}",
        hp.vocab_size, hp.d_model, hp.n_experts, hp.d_ff, hp.n_blocks,
    );

    let has_embedding = inv
        .tensors
        .iter()
        .any(|t: &TensorInfo| matches!(t.kind, TensorKind::TokenEmbedding));
    if !has_embedding {
        eprintln!("warn: no TokenEmbedding tensor classified; hyperparameters may be off");
    }
    let unknown = inv
        .tensors
        .iter()
        .filter(|t| matches!(t.kind, TensorKind::Unknown { .. }))
        .count();
    if unknown > 0 {
        eprintln!("warn: {unknown} tensors classified as Unknown");
    }
}

fn print_expert_console_summary(atlas: &ExpertAtlas) {
    eprintln!(
        "checkpoint: {}  blocks: {}  expected_experts_per_block: {:?}",
        atlas.checkpoint_path.display(),
        atlas.relevant_block_count,
        atlas.expected_experts_per_block,
    );
    eprintln!(
        "naming_checks: {}  anomalies: {}",
        atlas
            .naming_checks
            .iter()
            .filter(|check| check.passed)
            .count(),
        atlas.anomalies.len(),
    );
    if !atlas.anomalies.is_empty() {
        eprintln!(
            "warn: expert atlas contains {} anomalies",
            atlas.anomalies.len()
        );
    }
}

fn print_routing_console_summary(report_doc: &RoutingReport) {
    eprintln!(
        "checkpoint: {}  routing_blocks: {}  candidates: {}",
        report_doc.checkpoint_path.display(),
        report_doc.relevant_block_count,
        report_doc.candidate_tensors.len(),
    );
    eprintln!(
        "expected_experts_per_router: {:?}  critical_blocks: {}  anomalies: {}",
        report_doc.expected_experts_per_router,
        report_doc.likely_routing_critical_blocks.len(),
        report_doc.anomalies.len(),
    );
    if !report_doc.anomalies.is_empty() {
        eprintln!(
            "warn: routing report contains {} anomalies",
            report_doc.anomalies.len()
        );
    }
}

fn print_stats_console_summary(report_doc: &StatsProfileReport) {
    eprintln!(
        "checkpoint: {}  tensors: {}  layers: {}",
        report_doc.checkpoint_path.display(),
        report_doc.tensors.len(),
        report_doc.layers.len(),
    );
    eprintln!(
        "sample_values_per_tensor: {}  mean_rms: {:.6}  mean_variance: {:.6}",
        report_doc.sampling.max_sample_values,
        report_doc.norm_summary.mean_rms,
        report_doc.variance_summary.mean_variance,
    );
}

fn print_saaq_console_summary(report_doc: &SaaqReadinessReport) {
    eprintln!(
        "checkpoint: {}  candidate_targets: {}  routing_critical: {}",
        report_doc.checkpoint_path.display(),
        report_doc.candidate_targets.len(),
        report_doc.routing_critical_tensors.len(),
    );
    if let Some(top) = report_doc.candidate_targets.first() {
        eprintln!(
            "top_candidate: {}  readiness: {:.3}  risk: {:.3}",
            top.structural_name, top.readiness_score, top.risk_score,
        );
    }
}
