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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use xai_dissect::schema::{
        CandidateTensorManifest, ConversionManifest, ExpertAtlas, ExpertIssue, ExpertIssueCategory,
        ExpertIssueSeverity, ExpertNamingCheck, Grok1CoverageCounts, InferredHyperparams,
        InventoryTotals, MetricStatus, ModelInventory, PilotQuantizationMode, PilotSelectionPlan,
        QuantPlan, RouteMetricStatus, RoutePreservationReport, RoutingIssue, RoutingIssueCategory,
        RoutingIssueSeverity, RoutingReport, SaaqCandidate, SaaqDisposition, SaaqReadinessReport,
        SaaqRegionClass, StatsProfileReport, StatsSamplingConfig, TensorDType, TensorInfo,
        TensorKind, TensorRole, TensorShape, TensorStats,
    };

    use super::{
        print_console_summary, print_expert_console_summary, print_pilot_plan_console_summary,
        print_quant_plan_console_summary, print_route_preservation_console_summary,
        print_routing_console_summary, print_saaq_console_summary, print_stats_console_summary,
    };

    fn ckpt() -> PathBuf {
        PathBuf::from("/tmp/hex-fixture")
    }

    fn sample_tensor(kind: TensorKind) -> TensorInfo {
        TensorInfo {
            shard_path: PathBuf::from("/tmp/hex-fixture/tensor0000.pkl"),
            shard_ordinal: 0,
            in_shard_index: 0,
            role: TensorRole::Tensor,
            dtype: TensorDType::F32,
            shape: TensorShape::new(vec![2, 4]),
            offset: 0,
            nbytes: 32,
            kind,
            block_index: None,
            block_slot: None,
        }
    }

    fn inventory_with(tensors: Vec<TensorInfo>) -> ModelInventory {
        ModelInventory {
            model_family: "grok-1".into(),
            checkpoint_path: ckpt(),
            shard_count: 1,
            inferred: InferredHyperparams {
                vocab_size: Some(2),
                d_model: Some(4),
                n_experts: Some(8),
                d_ff: Some(6),
                n_blocks: Some(1),
            },
            totals: InventoryTotals {
                tensors: tensors.len() as u64,
                ..Default::default()
            },
            tensors,
            blocks: Vec::new(),
            schema_version: 2,
        }
    }

    fn empty_expert_atlas(anomalies: Vec<ExpertIssue>) -> ExpertAtlas {
        ExpertAtlas {
            model_family: "grok-1".into(),
            checkpoint_path: ckpt(),
            shard_count: 1,
            inferred: InferredHyperparams::default(),
            relevant_block_count: 0,
            expected_experts_per_block: Some(8),
            blocks: Vec::new(),
            naming_patterns: Vec::new(),
            naming_checks: vec![
                ExpertNamingCheck {
                    check: "expert_count_consistent".into(),
                    passed: true,
                    detail: "ok".into(),
                },
                ExpertNamingCheck {
                    check: "family_count".into(),
                    passed: false,
                    detail: "missing".into(),
                },
            ],
            anomalies,
            schema_version: 1,
        }
    }

    fn empty_routing(anomalies: Vec<RoutingIssue>) -> RoutingReport {
        RoutingReport {
            model_family: "grok-1".into(),
            checkpoint_path: ckpt(),
            shard_count: 1,
            inferred: InferredHyperparams::default(),
            relevant_block_count: 0,
            expected_experts_per_router: Some(8),
            candidate_tensors: Vec::new(),
            blocks: Vec::new(),
            orientation_summaries: Vec::new(),
            likely_routing_critical_blocks: Vec::new(),
            grok_layout_notes: Vec::new(),
            anomalies,
            schema_version: 1,
        }
    }

    fn empty_stats() -> StatsProfileReport {
        StatsProfileReport {
            model_family: "grok-1".into(),
            checkpoint_path: ckpt(),
            shard_count: 1,
            inferred: InferredHyperparams::default(),
            sampling: StatsSamplingConfig {
                max_sample_values: 64,
                f32_near_zero_abs: 1e-3,
                i8_near_zero_abs: 1,
            },
            tensors: vec![TensorStats {
                shard_ordinal: 0,
                in_shard_index: 0,
                block_index: None,
                block_slot: None,
                structural_name: "embedding.slot_00.token_embedding".into(),
                role: TensorRole::Tensor,
                dtype: TensorDType::F32,
                shape: TensorShape::new(vec![2, 4]),
                kind_label: "token_embedding".into(),
                sampled: true,
                total_values: 8,
                sample_values: 8,
                total_nbytes: 32,
                mean: 0.0,
                variance: 0.25,
                stddev: 0.5,
                min: -1.0,
                max: 1.0,
                max_abs: 1.0,
                l1_norm: 1.0,
                l2_norm: 1.0,
                rms: 0.5,
                zero_fraction: 0.0,
                near_zero_fraction: 0.0,
                positive_fraction: 0.5,
                negative_fraction: 0.5,
                outlier_fraction: 0.0,
                peak_to_rms: 2.0,
                distribution_label: "dense_balanced".into(),
            }],
            layers: Vec::new(),
            norm_summary: xai_dissect::schema::NormSummary {
                mean_rms: 0.5,
                max_rms: None,
                max_l2: None,
                top_rms: Vec::new(),
                top_l2: Vec::new(),
            },
            variance_summary: xai_dissect::schema::VarianceSummary {
                mean_variance: 0.25,
                max_variance: None,
                min_variance: None,
                top_variance: Vec::new(),
                lowest_variance: Vec::new(),
            },
            outlier_summary: xai_dissect::schema::OutlierSummary {
                mean_outlier_fraction: 0.0,
                most_outlier_heavy: Vec::new(),
                highest_peak_to_rms: Vec::new(),
            },
            schema_version: 1,
        }
    }

    fn sample_candidate() -> SaaqCandidate {
        SaaqCandidate {
            rank: 1,
            shard_ordinal: 0,
            in_shard_index: 0,
            block_index: None,
            block_slot: None,
            structural_name: "embedding.slot_00.token_embedding".into(),
            kind_label: "token_embedding".into(),
            dtype: TensorDType::F32,
            shape: TensorShape::new(vec![2, 4]),
            region_class: SaaqRegionClass::EmbeddingHeavy,
            disposition: SaaqDisposition::ObserveOnly,
            readiness_score: 0.1,
            opportunity_score: 0.2,
            risk_score: 0.3,
            reasons: vec!["embedding".into()],
        }
    }

    fn empty_saaq(candidates: Vec<SaaqCandidate>) -> SaaqReadinessReport {
        SaaqReadinessReport {
            model_family: "grok-1".into(),
            checkpoint_path: ckpt(),
            shard_count: 1,
            inferred: InferredHyperparams::default(),
            candidate_targets: candidates.clone(),
            quantization_candidates: candidates.clone(),
            routing_critical_tensors: Vec::new(),
            precision_sensitive_tensors: Vec::new(),
            deferred_tensors: Vec::new(),
            risky_tensors: Vec::new(),
            layer_readiness: Vec::new(),
            notes: Vec::new(),
            manifest: CandidateTensorManifest {
                model_family: "grok-1".into(),
                checkpoint_path: ckpt(),
                candidates,
                schema_version: 1,
            },
            schema_version: 2,
        }
    }

    fn empty_quant_plan() -> QuantPlan {
        QuantPlan {
            model_family: "grok-1".into(),
            checkpoint_path: ckpt(),
            baseline: "grok1-map-v1-clean".into(),
            required_validation: Grok1CoverageCounts::default(),
            discovered_validation: Grok1CoverageCounts::default(),
            keep_fp32: vec!["router".into()],
            pilot_quantize: vec!["moe_expert.down".into()],
            defer: vec!["token_embedding".into()],
            notes: Vec::new(),
            schema_version: 1,
        }
    }

    fn empty_conversion(warnings: Vec<String>) -> ConversionManifest {
        ConversionManifest {
            model_family: "grok-1".into(),
            checkpoint_path: ckpt(),
            baseline_profile: "grok1-map-v1-clean".into(),
            required_validation: Grok1CoverageCounts::default(),
            discovered_validation: Grok1CoverageCounts::default(),
            relevant_block_count: 0,
            expected_experts_per_block: None,
            expert_tensor_families_per_block: None,
            router_orientation: None,
            router_shape: None,
            tensors: Vec::new(),
            warnings,
            schema_version: 1,
        }
    }

    #[test]
    fn print_console_summary_covers_embedding_and_unknown_warns() {
        print_console_summary(&inventory_with(vec![sample_tensor(
            TensorKind::TokenEmbedding,
        )]));
        print_console_summary(&inventory_with(Vec::new()));
        print_console_summary(&inventory_with(vec![sample_tensor(TensorKind::Unknown {
            reason: "unclassified".into(),
        })]));
    }

    #[test]
    fn print_expert_console_summary_covers_anomaly_warn() {
        print_expert_console_summary(&empty_expert_atlas(Vec::new()));
        print_expert_console_summary(&empty_expert_atlas(vec![ExpertIssue {
            severity: ExpertIssueSeverity::Warning,
            category: ExpertIssueCategory::LayoutAnomaly,
            block_index: Some(0),
            tensor: None,
            message: "irregular expert layout".into(),
        }]));
    }

    #[test]
    fn print_routing_console_summary_covers_anomaly_warn() {
        print_routing_console_summary(&empty_routing(Vec::new()));
        print_routing_console_summary(&empty_routing(vec![RoutingIssue {
            severity: RoutingIssueSeverity::Warning,
            category: RoutingIssueCategory::MissingCandidate,
            block_index: Some(0),
            tensor: None,
            message: "no router".into(),
        }]));
    }

    #[test]
    fn print_stats_console_summary_prints_sample_means() {
        print_stats_console_summary(&empty_stats());
    }

    #[test]
    fn print_saaq_console_summary_covers_empty_and_top_candidate() {
        print_saaq_console_summary(&empty_saaq(Vec::new()));
        print_saaq_console_summary(&empty_saaq(vec![sample_candidate()]));
    }

    #[test]
    fn print_quant_plan_console_summary_covers_warning_categories() {
        print_quant_plan_console_summary(&empty_quant_plan(), &empty_conversion(Vec::new()));
        print_quant_plan_console_summary(
            &empty_quant_plan(),
            &empty_conversion(vec!["hash collision".into()]),
        );
    }

    #[test]
    fn print_pilot_plan_console_summary_prints_counts() {
        print_pilot_plan_console_summary(&PilotSelectionPlan {
            model_family: "grok-1".into(),
            checkpoint_path: ckpt(),
            baseline: "grok1-map-v1-clean".into(),
            required_validation: Grok1CoverageCounts::default(),
            selected_blocks: Vec::new(),
            modes: vec![PilotQuantizationMode::AttentionOnly],
            protection_rules: Vec::new(),
            comparison_artifacts: Vec::new(),
            notes: Vec::new(),
            schema_version: 1,
        });
    }

    #[test]
    fn print_route_preservation_console_summary_prints_metric_counts() {
        print_route_preservation_console_summary(&RoutePreservationReport {
            model_family: "grok-1".into(),
            checkpoint_path: ckpt(),
            baseline: "grok1-map-v1-clean".into(),
            required_validation: Grok1CoverageCounts::default(),
            summary: Vec::new(),
            router_metrics: vec![RouteMetricStatus {
                name: "router_top1_agreement".into(),
                scope: "router_behavior".into(),
                status: MetricStatus::Unknown,
                threshold: None,
                observed: None,
                detail: "unset".into(),
            }],
            block_metrics: Vec::new(),
            weight_metrics: Vec::new(),
            model_metrics: Vec::new(),
            notes: Vec::new(),
            schema_version: 1,
        });
    }
}
