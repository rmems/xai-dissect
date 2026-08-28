// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Console summaries printed after each subcommand.
//!
//! Split out of `cli` so the command-dispatch module stays focused on the
//! resolve -> inventory -> analyze -> write loop. Text is unchanged.

use xai_dissect::schema::{
    ExpertAtlas, ModelInventory, PilotSelectionPlan, QuantPlan, RoutePreservationReport,
    RoutingReport, SaaqReadinessReport, StatsProfileReport, TensorInfo, TensorKind,
};

pub(super) fn print_quant_plan_console_summary(
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

pub(super) fn print_pilot_plan_console_summary(plan: &PilotSelectionPlan) {
    eprintln!(
        "checkpoint: {}  baseline: {}  selected_blocks: {}  modes: {}",
        plan.checkpoint_path.display(),
        plan.baseline,
        plan.selected_blocks.len(),
        plan.modes.len(),
    );
}

pub(super) fn print_route_preservation_console_summary(report_doc: &RoutePreservationReport) {
    eprintln!(
        "model_family: {}  baseline: {}  router_metrics: {}  block_metrics: {}",
        report_doc.model_family,
        report_doc.baseline,
        report_doc.router_metrics.len(),
        report_doc.block_metrics.len(),
    );
}

pub(super) fn print_console_summary(inv: &ModelInventory) {
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

pub(super) fn print_expert_console_summary(atlas: &ExpertAtlas) {
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

pub(super) fn print_routing_console_summary(report_doc: &RoutingReport) {
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

pub(super) fn print_stats_console_summary(report_doc: &StatsProfileReport) {
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

pub(super) fn print_saaq_console_summary(report_doc: &SaaqReadinessReport) {
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
