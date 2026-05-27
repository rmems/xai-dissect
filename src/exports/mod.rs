// SPDX-License-Identifier: GPL-3.0-only
//
// Unified output-tree planning and bundle writers. This layer keeps path
// conventions and manifest generation out of the analyzers themselves.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::inventory;
use crate::report;
use crate::schema::{
    CheckpointInventoryBlockSnapshot, CheckpointInventorySnapshot, ConversionManifest, ExpertAtlas,
    FindingsSeverity, FindingsSummary, FindingsSummaryItem, ModelInventory, PilotSelectionPlan,
    QuantPlan, RoutePreservationReport, RoutingCriticalTensor, RoutingCriticalTensorManifest,
    RoutingReport, SaaqReadinessReport, StatsProfileReport,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputLayout {
    pub checkpoint_slug: String,
    pub reports_dir: PathBuf,
    pub exports_dir: PathBuf,
    pub manifests_dir: PathBuf,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OutputBundle {
    pub checkpoint_slug: String,
    pub written_paths: Vec<PathBuf>,
}

pub fn prepare_output_layout(
    root: &Path,
    checkpoint_path: &Path,
    slug_override: Option<&str>,
) -> Result<OutputLayout> {
    let checkpoint_slug = resolve_checkpoint_slug(checkpoint_path, slug_override)?;
    let reports_dir = root.join("reports").join(&checkpoint_slug);
    let exports_dir = root.join("exports").join(&checkpoint_slug);
    let manifests_dir = root.join("manifests").join(&checkpoint_slug);

    fs::create_dir_all(&reports_dir)
        .with_context(|| format!("create {}", reports_dir.display()))?;
    fs::create_dir_all(&exports_dir)
        .with_context(|| format!("create {}", exports_dir.display()))?;
    fs::create_dir_all(&manifests_dir)
        .with_context(|| format!("create {}", manifests_dir.display()))?;

    Ok(OutputLayout {
        checkpoint_slug,
        reports_dir,
        exports_dir,
        manifests_dir,
    })
}

pub fn resolve_checkpoint_slug(
    checkpoint_path: &Path,
    slug_override: Option<&str>,
) -> Result<String> {
    if let Some(slug_override) = slug_override {
        let slug = sanitize_slug_component(slug_override);
        if slug.is_empty() {
            bail!(
                "checkpoint slug override {:?} resolved to an empty value after sanitization; provide an override containing retained ASCII alphanumeric characters",
                slug_override
            );
        }
        return Ok(slug);
    }

    let mut parts = slug_parts(checkpoint_path);

    if parts.is_empty() {
        if let Ok(canonical) = checkpoint_path.canonicalize() {
            parts = slug_parts(&canonical);
        }
    }

    if parts.is_empty() {
        parts.push("checkpoint".to_string());
    }

    let tail = if parts.len() >= 2 {
        parts[parts.len() - 2..].to_vec()
    } else {
        vec![parts[parts.len() - 1].clone()]
    };

    Ok(tail.join("__"))
}

fn slug_parts(checkpoint_path: &Path) -> Vec<String> {
    checkpoint_path
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .map(sanitize_slug_component)
        .filter(|part| !part.is_empty())
        .collect()
}

pub fn write_inventory_bundle(
    inv: &ModelInventory,
    root: &Path,
    slug_override: Option<&str>,
) -> Result<OutputBundle> {
    let coverage = if inventory::should_validate_grok1_coverage(inv) {
        Some(inventory::validate_grok1_complete_manifest(inv)?)
    } else {
        None
    };
    let layout = prepare_output_layout(root, &inv.checkpoint_path, slug_override)?;
    let mut bundle = OutputBundle {
        checkpoint_slug: layout.checkpoint_slug.clone(),
        written_paths: Vec::new(),
    };

    let json_path = layout.exports_dir.join("inventory.json");
    report::write_json(inv, &json_path)?;
    bundle.written_paths.push(json_path);

    let md_path = layout.reports_dir.join("inventory.md");
    report::write_markdown(inv, &md_path)?;
    bundle.written_paths.push(md_path);

    let findings = build_inventory_findings_summary(inv, &layout.checkpoint_slug);
    let findings_path = layout.exports_dir.join("inventory-findings.json");
    report::write_findings_summary_json(&findings, &findings_path)?;
    bundle.written_paths.push(findings_path);

    let snapshot = build_inventory_snapshot(inv);
    let manifest_path = layout
        .manifests_dir
        .join("checkpoint-inventory-snapshot.json");
    report::write_inventory_snapshot_manifest_json(&snapshot, &manifest_path)?;
    bundle.written_paths.push(manifest_path);

    if let Some(coverage) = coverage {
        let coverage_path = layout.manifests_dir.join("grok1-coverage.json");
        report::write_grok1_coverage_manifest_json(&coverage, &coverage_path)?;
        bundle.written_paths.push(coverage_path);
    }

    Ok(bundle)
}

pub fn write_expert_bundle(
    atlas: &ExpertAtlas,
    root: &Path,
    slug_override: Option<&str>,
) -> Result<OutputBundle> {
    let layout = prepare_output_layout(root, &atlas.checkpoint_path, slug_override)?;
    let mut bundle = OutputBundle {
        checkpoint_slug: layout.checkpoint_slug.clone(),
        written_paths: Vec::new(),
    };

    let json_path = layout.exports_dir.join("experts.json");
    report::write_expert_json(atlas, &json_path)?;
    bundle.written_paths.push(json_path);

    let md_path = layout.reports_dir.join("experts.md");
    report::write_expert_markdown(atlas, &md_path)?;
    bundle.written_paths.push(md_path);

    let findings = build_expert_findings_summary(atlas, &layout.checkpoint_slug);
    let findings_path = layout.exports_dir.join("experts-findings.json");
    report::write_findings_summary_json(&findings, &findings_path)?;
    bundle.written_paths.push(findings_path);

    Ok(bundle)
}

pub fn write_routing_bundle(
    report_doc: &RoutingReport,
    root: &Path,
    slug_override: Option<&str>,
) -> Result<OutputBundle> {
    let layout = prepare_output_layout(root, &report_doc.checkpoint_path, slug_override)?;
    let mut bundle = OutputBundle {
        checkpoint_slug: layout.checkpoint_slug.clone(),
        written_paths: Vec::new(),
    };

    let json_path = layout.exports_dir.join("routing-report.json");
    report::write_routing_json(report_doc, &json_path)?;
    bundle.written_paths.push(json_path);

    let md_path = layout.reports_dir.join("routing-report.md");
    report::write_routing_markdown(report_doc, &md_path)?;
    bundle.written_paths.push(md_path);

    let findings = build_routing_findings_summary(report_doc, &layout.checkpoint_slug);
    let findings_path = layout.exports_dir.join("routing-report-findings.json");
    report::write_findings_summary_json(&findings, &findings_path)?;
    bundle.written_paths.push(findings_path);

    let manifest = build_routing_critical_manifest(report_doc);
    let manifest_path = layout.manifests_dir.join("routing-critical-tensors.json");
    report::write_routing_critical_manifest_json(&manifest, &manifest_path)?;
    bundle.written_paths.push(manifest_path);

    Ok(bundle)
}

pub fn write_stats_bundle(
    report_doc: &StatsProfileReport,
    root: &Path,
    slug_override: Option<&str>,
) -> Result<OutputBundle> {
    let layout = prepare_output_layout(root, &report_doc.checkpoint_path, slug_override)?;
    let mut bundle = OutputBundle {
        checkpoint_slug: layout.checkpoint_slug.clone(),
        written_paths: Vec::new(),
    };

    let json_path = layout.exports_dir.join("stats.json");
    report::write_stats_json(report_doc, &json_path)?;
    bundle.written_paths.push(json_path);

    let md_path = layout.reports_dir.join("stats.md");
    report::write_stats_markdown(report_doc, &md_path)?;
    bundle.written_paths.push(md_path);

    let findings = build_stats_findings_summary(report_doc, &layout.checkpoint_slug);
    let findings_path = layout.exports_dir.join("stats-findings.json");
    report::write_findings_summary_json(&findings, &findings_path)?;
    bundle.written_paths.push(findings_path);

    Ok(bundle)
}

pub fn write_saaq_bundle(
    report_doc: &SaaqReadinessReport,
    root: &Path,
    slug_override: Option<&str>,
) -> Result<OutputBundle> {
    let layout = prepare_output_layout(root, &report_doc.checkpoint_path, slug_override)?;
    let mut bundle = OutputBundle {
        checkpoint_slug: layout.checkpoint_slug.clone(),
        written_paths: Vec::new(),
    };

    let json_path = layout.exports_dir.join("saaq-readiness.json");
    report::write_saaq_readiness_json(report_doc, &json_path)?;
    bundle.written_paths.push(json_path);

    let md_path = layout.reports_dir.join("saaq-readiness.md");
    report::write_saaq_readiness_markdown(report_doc, &md_path)?;
    bundle.written_paths.push(md_path);

    let findings = build_saaq_findings_summary(report_doc, &layout.checkpoint_slug);
    let findings_path = layout.exports_dir.join("saaq-readiness-findings.json");
    report::write_findings_summary_json(&findings, &findings_path)?;
    bundle.written_paths.push(findings_path);

    let manifest_path = layout.manifests_dir.join("candidate-saaq-targets.json");
    report::write_candidate_manifest_json(&report_doc.manifest, &manifest_path)?;
    bundle.written_paths.push(manifest_path);

    Ok(bundle)
}

pub fn write_quant_plan_bundle(
    conversion_manifest: &ConversionManifest,
    quant_plan: &QuantPlan,
    root: &Path,
    slug_override: Option<&str>,
) -> Result<OutputBundle> {
    let layout = prepare_output_layout(root, &quant_plan.checkpoint_path, slug_override)?;
    let mut bundle = OutputBundle {
        checkpoint_slug: layout.checkpoint_slug.clone(),
        written_paths: Vec::new(),
    };

    let conversion_path = layout.manifests_dir.join("conversion-manifest.json");
    report::write_conversion_manifest_json(conversion_manifest, &conversion_path)?;
    bundle.written_paths.push(conversion_path);

    let plan_path = layout.manifests_dir.join("quant-plan.json");
    report::write_quant_plan_json(quant_plan, &plan_path)?;
    bundle.written_paths.push(plan_path);

    let md_path = layout.reports_dir.join("quant-plan.md");
    report::write_quant_plan_markdown(quant_plan, &md_path)?;
    bundle.written_paths.push(md_path);

    Ok(bundle)
}

pub fn write_pilot_plan_bundle(
    plan: &PilotSelectionPlan,
    root: &Path,
    slug_override: Option<&str>,
) -> Result<OutputBundle> {
    let layout = prepare_output_layout(root, &plan.checkpoint_path, slug_override)?;
    let mut bundle = OutputBundle {
        checkpoint_slug: layout.checkpoint_slug.clone(),
        written_paths: Vec::new(),
    };

    let json_path = layout.manifests_dir.join("pilot-selection-plan.json");
    report::write_pilot_selection_plan_json(plan, &json_path)?;
    bundle.written_paths.push(json_path);

    let md_path = layout.reports_dir.join("pilot-selection-plan.md");
    report::write_pilot_selection_plan_markdown(plan, &md_path)?;
    bundle.written_paths.push(md_path);

    Ok(bundle)
}

pub fn write_route_preservation_bundle(
    report_doc: &RoutePreservationReport,
    root: &Path,
    slug_override: Option<&str>,
) -> Result<OutputBundle> {
    let layout = prepare_output_layout(root, &report_doc.checkpoint_path, slug_override)?;
    let mut bundle = OutputBundle {
        checkpoint_slug: layout.checkpoint_slug.clone(),
        written_paths: Vec::new(),
    };

    let json_path = layout.manifests_dir.join("route-preservation-report.json");
    report::write_route_preservation_report_json(report_doc, &json_path)?;
    bundle.written_paths.push(json_path);

    let md_path = layout.reports_dir.join("route-preservation-report.md");
    report::write_route_preservation_markdown(report_doc, &md_path)?;
    bundle.written_paths.push(md_path);

    Ok(bundle)
}

pub fn build_inventory_snapshot(inv: &ModelInventory) -> CheckpointInventorySnapshot {
    CheckpointInventorySnapshot {
        model_family: inv.model_family.clone(),
        checkpoint_path: inv.checkpoint_path.clone(),
        shard_count: inv.shard_count,
        inferred: inv.inferred.clone(),
        total_tensors: inv.totals.tensors,
        total_nbytes: inv.totals.total_nbytes,
        blocks: inv
            .blocks
            .iter()
            .map(|block| CheckpointInventoryBlockSnapshot {
                label: block.label.clone(),
                block_index: block.block_index,
                shard_range: block.shard_range,
                tensor_count: block.tensor_count,
                total_nbytes: block.total_nbytes,
                kind_labels: block
                    .kinds
                    .iter()
                    .map(|kind| kind.kind_label.clone())
                    .collect(),
            })
            .collect(),
        schema_version: inv.schema_version,
    }
}

pub fn build_routing_critical_manifest(
    report_doc: &RoutingReport,
) -> RoutingCriticalTensorManifest {
    let reasons = report_doc
        .likely_routing_critical_blocks
        .iter()
        .map(|block| (block.block_index, block.reason.clone()))
        .collect::<BTreeMap<_, _>>();

    RoutingCriticalTensorManifest {
        model_family: report_doc.model_family.clone(),
        checkpoint_path: report_doc.checkpoint_path.clone(),
        tensors: report_doc
            .candidate_tensors
            .iter()
            .map(|tensor| RoutingCriticalTensor {
                shard_ordinal: tensor.shard_ordinal,
                in_shard_index: tensor.in_shard_index,
                block_index: tensor.block_index,
                block_slot: tensor.block_slot,
                structural_name: tensor.structural_name.clone(),
                kind_label: tensor.kind_label.clone(),
                orientation: tensor.orientation,
                linked_expert_count: tensor.linked_expert_count,
                total_nbytes: tensor.gate_metrics.total_nbytes,
                criticality_reason: reasons.get(&tensor.block_index).cloned(),
            })
            .collect(),
        schema_version: report_doc.schema_version,
    }
}

pub fn build_inventory_findings_summary(
    inv: &ModelInventory,
    checkpoint_slug: &str,
) -> FindingsSummary {
    let unknown_tensors = inv
        .tensors
        .iter()
        .filter(|tensor| matches!(tensor.kind, crate::schema::TensorKind::Unknown { .. }))
        .count();

    let mut findings = vec![
        info_finding(
            "checkpoint_scale",
            format!(
                "{} shards, {} tensors, {} bytes",
                inv.shard_count, inv.totals.tensors, inv.totals.total_nbytes
            ),
        ),
        info_finding(
            "block_structure",
            format!(
                "{} block summaries, inferred d_model {}, inferred experts {}",
                inv.blocks.len(),
                fmt_opt(inv.inferred.d_model),
                fmt_opt(inv.inferred.n_experts)
            ),
        ),
    ];

    if inv.totals.quant_tensors > 0 {
        findings.push(info_finding(
            "quantization_layout",
            format!(
                "{} quantized tensors alongside {} f32 tensors",
                inv.totals.quant_tensors, inv.totals.f32_tensors
            ),
        ));
    }

    if unknown_tensors > 0 {
        findings.push(warn_finding(
            "unknown_tensor_kinds",
            format!("{unknown_tensors} tensors remained structurally unclassified"),
        ));
    }

    FindingsSummary {
        analysis: "inventory".into(),
        model_family: inv.model_family.clone(),
        checkpoint_path: inv.checkpoint_path.clone(),
        checkpoint_slug: checkpoint_slug.to_string(),
        headline: format!(
            "{} tensors across {} shards",
            inv.totals.tensors, inv.shard_count
        ),
        findings,
        schema_version: inv.schema_version,
    }
}

pub fn build_expert_findings_summary(
    atlas: &ExpertAtlas,
    checkpoint_slug: &str,
) -> FindingsSummary {
    let expert_counts = atlas
        .blocks
        .iter()
        .filter_map(|block| block.expert_count)
        .collect::<BTreeSet<_>>();
    let failed_checks = atlas
        .naming_checks
        .iter()
        .filter(|check| !check.passed)
        .count();

    let mut findings = vec![info_finding(
        "expert_blocks",
        format!(
            "{} relevant blocks, expected experts per block {}",
            atlas.relevant_block_count,
            fmt_opt(atlas.expected_experts_per_block)
        ),
    )];

    if expert_counts.len() <= 1 {
        findings.push(info_finding(
            "expert_count_consistency",
            format!(
                "expert count is structurally consistent at {}",
                fmt_opt(expert_counts.iter().next().copied())
            ),
        ));
    } else {
        findings.push(warn_finding(
            "expert_count_consistency",
            format!(
                "observed multiple expert counts across blocks: {}",
                expert_counts
                    .iter()
                    .map(|count| count.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }

    if failed_checks > 0 {
        findings.push(warn_finding(
            "naming_checks",
            format!("{failed_checks} expert naming consistency checks failed"),
        ));
    }

    if !atlas.anomalies.is_empty() {
        findings.push(warn_finding(
            "expert_anomalies",
            format!("{} expert anomalies detected", atlas.anomalies.len()),
        ));
    }

    FindingsSummary {
        analysis: "experts".into(),
        model_family: atlas.model_family.clone(),
        checkpoint_path: atlas.checkpoint_path.clone(),
        checkpoint_slug: checkpoint_slug.to_string(),
        headline: format!("{} relevant expert blocks", atlas.relevant_block_count),
        findings,
        schema_version: atlas.schema_version,
    }
}

pub fn build_routing_findings_summary(
    report_doc: &RoutingReport,
    checkpoint_slug: &str,
) -> FindingsSummary {
    let mut findings = vec![
        info_finding(
            "routing_candidates",
            format!(
                "{} routing candidates across {} relevant blocks",
                report_doc.candidate_tensors.len(),
                report_doc.relevant_block_count
            ),
        ),
        info_finding(
            "expert_linkage",
            format!(
                "expected experts per router {}",
                fmt_opt(report_doc.expected_experts_per_router)
            ),
        ),
    ];

    if !report_doc.orientation_summaries.is_empty() {
        findings.push(info_finding(
            "orientations",
            report_doc
                .orientation_summaries
                .iter()
                .map(|summary| format!("{} x{}", summary.orientation.label(), summary.count))
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }

    if !report_doc.likely_routing_critical_blocks.is_empty() {
        findings.push(info_finding(
            "routing_critical_blocks",
            format!(
                "{} blocks marked routing-critical",
                report_doc.likely_routing_critical_blocks.len()
            ),
        ));
    }

    if !report_doc.anomalies.is_empty() {
        findings.push(warn_finding(
            "routing_anomalies",
            format!("{} routing anomalies detected", report_doc.anomalies.len()),
        ));
    }

    FindingsSummary {
        analysis: "routing-report".into(),
        model_family: report_doc.model_family.clone(),
        checkpoint_path: report_doc.checkpoint_path.clone(),
        checkpoint_slug: checkpoint_slug.to_string(),
        headline: format!(
            "{} routing candidates discovered",
            report_doc.candidate_tensors.len()
        ),
        findings,
        schema_version: report_doc.schema_version,
    }
}

pub fn build_stats_findings_summary(
    report_doc: &StatsProfileReport,
    checkpoint_slug: &str,
) -> FindingsSummary {
    let mut findings = vec![info_finding(
        "sampling",
        format!(
            "{} tensors profiled with up to {} sampled values each",
            report_doc.tensors.len(),
            report_doc.sampling.max_sample_values
        ),
    )];

    if let Some(tensor) = &report_doc.norm_summary.max_rms {
        findings.push(info_finding(
            "highest_rms",
            format!(
                "highest RMS: {} ({:.6})",
                tensor.structural_name, tensor.value
            ),
        ));
    }
    if let Some(tensor) = &report_doc.variance_summary.max_variance {
        findings.push(info_finding(
            "highest_variance",
            format!(
                "highest variance: {} ({:.6})",
                tensor.structural_name, tensor.value
            ),
        ));
    }
    if let Some(tensor) = report_doc.outlier_summary.most_outlier_heavy.first() {
        findings.push(info_finding(
            "outlier_heavy",
            format!(
                "most outlier-heavy tensor: {} ({:.6})",
                tensor.structural_name, tensor.value
            ),
        ));
    }

    FindingsSummary {
        analysis: "stats".into(),
        model_family: report_doc.model_family.clone(),
        checkpoint_path: report_doc.checkpoint_path.clone(),
        checkpoint_slug: checkpoint_slug.to_string(),
        headline: format!("{} tensors profiled", report_doc.tensors.len()),
        findings,
        schema_version: report_doc.schema_version,
    }
}

pub fn build_saaq_findings_summary(
    report_doc: &SaaqReadinessReport,
    checkpoint_slug: &str,
) -> FindingsSummary {
    let mut findings = vec![
        info_finding(
            "candidate_targets",
            format!(
                "{} candidate targets, {} routing-critical tensors",
                report_doc.candidate_targets.len(),
                report_doc.routing_critical_tensors.len()
            ),
        ),
        info_finding(
            "layer_readiness",
            format!("{} layer readiness rows", report_doc.layer_readiness.len()),
        ),
    ];

    if let Some(candidate) = report_doc.candidate_targets.first() {
        findings.push(info_finding(
            "top_candidate",
            format!(
                "top candidate {} with readiness {:.3} and risk {:.3}",
                candidate.structural_name, candidate.readiness_score, candidate.risk_score
            ),
        ));
    } else {
        findings.push(warn_finding(
            "top_candidate",
            "no candidate tensors were promoted for the current checkpoint slice".into(),
        ));
    }

    if let Some(candidate) = report_doc.risky_tensors.first() {
        findings.push(warn_finding(
            "highest_risk",
            format!(
                "highest-risk tensor {} at {:.3}",
                candidate.structural_name, candidate.risk_score
            ),
        ));
    }

    FindingsSummary {
        analysis: "saaq-readiness".into(),
        model_family: report_doc.model_family.clone(),
        checkpoint_path: report_doc.checkpoint_path.clone(),
        checkpoint_slug: checkpoint_slug.to_string(),
        headline: format!(
            "{} SAAQ candidates ranked",
            report_doc.candidate_targets.len()
        ),
        findings,
        schema_version: report_doc.schema_version,
    }
}

fn sanitize_slug_component(input: &str) -> String {
    let mut slug = String::new();
    let mut last_dash = false;

    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            slug.push('-');
            last_dash = true;
        }
    }

    slug.trim_matches('-').to_string()
}

fn info_finding(category: &str, detail: String) -> FindingsSummaryItem {
    FindingsSummaryItem {
        severity: FindingsSeverity::Info,
        category: category.to_string(),
        detail,
    }
}

fn warn_finding(category: &str, detail: String) -> FindingsSummaryItem {
    FindingsSummaryItem {
        severity: FindingsSeverity::Warning,
        category: category.to_string(),
        detail,
    }
}

fn fmt_opt(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::inventory::SCHEMA_VERSION;
    use crate::schema::{
        BlockSummary, InferredHyperparams, InventoryTotals, KindCount, MoeProjection,
        QuantizedAttentionWidth, RoutingBlockReport, RoutingCriticalBlock, RoutingGateMetrics,
        RoutingOrientation, RoutingOrientationSummary, RoutingTensorLocator, RoutingTensorRef,
        ShardRange, TensorDType, TensorInfo, TensorKind, TensorRole, TensorShape,
    };

    #[test]
    fn resolves_checkpoint_slug_from_parent_and_leaf() {
        let path = Path::new("/tmp/grok-1-official/ckpt-0");
        let slug = resolve_checkpoint_slug(path, None).expect("slug");
        assert_eq!(slug, "grok-1-official__ckpt-0");
    }

    #[test]
    fn respects_checkpoint_slug_override() {
        let path = Path::new("/tmp/grok-1-official/ckpt-0");
        let slug = resolve_checkpoint_slug(path, Some("Milestone Release / CKPT-0"))
            .expect("override slug");
        assert_eq!(slug, "milestone-release-ckpt-0");
    }

    #[test]
    fn resolves_checkpoint_slug_from_current_dir() {
        let slug = resolve_checkpoint_slug(Path::new("."), None).expect("slug");

        assert!(!slug.is_empty());
        assert_ne!(slug, ".");
    }

    #[test]
    fn inventory_bundle_writes_standard_tree() {
        let root = unique_test_root("inventory_bundle");
        let inv = sample_inventory();

        let bundle = write_inventory_bundle(&inv, &root, None).expect("write inventory bundle");

        assert_eq!(bundle.checkpoint_slug, "grok-1-official__ckpt-0");
        assert!(
            root.join("reports/grok-1-official__ckpt-0/inventory.md")
                .exists()
        );
        assert!(
            root.join("exports/grok-1-official__ckpt-0/inventory.json")
                .exists()
        );
        assert!(
            root.join("exports/grok-1-official__ckpt-0/inventory-findings.json")
                .exists()
        );
        assert!(
            root.join("manifests/grok-1-official__ckpt-0/checkpoint-inventory-snapshot.json")
                .exists()
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn inventory_bundle_validates_complete_grok1_before_writing_outputs() {
        let root = unique_test_root("inventory_bundle_validation_error");
        let mut inv = complete_grok1_inventory();
        inv.inferred.vocab_size = Some(123);

        let err = write_inventory_bundle(&inv, &root, None).unwrap_err();

        assert!(format!("{err:#}").contains("inferred vocab_size"));
        assert!(!root.join("exports").exists());
        assert!(!root.join("reports").exists());
        assert!(!root.join("manifests").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn inventory_bundle_writes_repacked_grok1_coverage_manifest() {
        let root = unique_test_root("inventory_bundle_repacked_grok1");
        let mut inv = complete_grok1_inventory();
        inv.shard_count = 42;

        let bundle = write_inventory_bundle(&inv, &root, None).expect("write inventory bundle");

        assert!(
            bundle
                .written_paths
                .iter()
                .any(|path| path.file_name().and_then(|name| name.to_str())
                    == Some("grok1-coverage.json"))
        );
        assert!(
            root.join("manifests/grok-1-official__ckpt-0/grok1-coverage.json")
                .exists()
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn routing_manifest_carries_block_reason() {
        let report_doc = sample_routing_report();
        let manifest = build_routing_critical_manifest(&report_doc);

        assert_eq!(manifest.tensors.len(), 1);
        assert_eq!(
            manifest.tensors[0].criticality_reason.as_deref(),
            Some("primary router present")
        );
    }

    fn unique_test_root(prefix: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("xai-dissect-{prefix}-{stamp}"))
    }

    fn sample_inventory() -> ModelInventory {
        ModelInventory {
            model_family: "grok-1".into(),
            checkpoint_path: PathBuf::from("/tmp/grok-1-official/ckpt-0"),
            shard_count: 2,
            inferred: InferredHyperparams {
                d_model: Some(6144),
                n_experts: Some(8),
                n_blocks: Some(1),
                ..Default::default()
            },
            tensors: Vec::new(),
            blocks: vec![BlockSummary {
                block_index: Some(0),
                label: "block_000".into(),
                shard_range: Some(ShardRange {
                    start: 0,
                    end_inclusive: 1,
                }),
                tensor_count: 3,
                total_nbytes: 1024,
                dtypes: vec![TensorDType::F32, TensorDType::I8],
                kinds: vec![KindCount {
                    kind_label: "router".into(),
                    count: 1,
                    nbytes: 128,
                }],
            }],
            totals: InventoryTotals {
                tensors: 3,
                quant_tensors: 1,
                f32_tensors: 2,
                i8_tensors: 1,
                total_nbytes: 1024,
                total_elements: 256,
            },
            schema_version: 1,
        }
    }

    fn complete_grok1_inventory() -> ModelInventory {
        let mut tensors = vec![
            tensor(
                0,
                0,
                None,
                None,
                TensorKind::TokenEmbedding,
                TensorRole::Tensor,
                TensorDType::F32,
                vec![131_072, 6_144],
            ),
            tensor(
                1,
                0,
                None,
                None,
                TensorKind::FinalNorm,
                TensorRole::Tensor,
                TensorDType::F32,
                vec![6_144],
            ),
        ];

        for block in 0..64u32 {
            for slot in 0..12u32 {
                let shard = 2 + block * 12 + slot;
                let (kind, role, dtype, shape) = match slot {
                    0 | 2 => (
                        TensorKind::MoeExpertProjection {
                            projection: if slot == 0 {
                                MoeProjection::Gate
                            } else {
                                MoeProjection::Up
                            },
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 6_144, 32_768],
                    ),
                    1 => (
                        TensorKind::MoeExpertProjection {
                            projection: MoeProjection::Down,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 32_768, 6_144],
                    ),
                    3 | 6 => (
                        TensorKind::QuantizedAttentionProjection {
                            width: QuantizedAttentionWidth::Narrow,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![6_144, 1_024],
                    ),
                    4 | 5 => (
                        TensorKind::QuantizedAttentionProjection {
                            width: QuantizedAttentionWidth::ModelWidth,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![6_144, 6_144],
                    ),
                    7..=10 => (
                        TensorKind::BlockNorm,
                        TensorRole::Tensor,
                        TensorDType::F32,
                        vec![6_144],
                    ),
                    11 => (
                        TensorKind::Router,
                        TensorRole::Tensor,
                        TensorDType::F32,
                        vec![6_144, 8],
                    ),
                    _ => unreachable!(),
                };
                tensors.push(tensor(
                    shard,
                    0,
                    Some(block),
                    Some(slot),
                    kind,
                    role,
                    dtype,
                    shape,
                ));
            }
        }

        let blocks = crate::inventory::summarize_blocks(&tensors);
        let totals = inventory_totals(&tensors);
        ModelInventory {
            model_family: "grok-1".to_string(),
            checkpoint_path: PathBuf::from("/tmp/grok-1-official/ckpt-0"),
            shard_count: 770,
            inferred: InferredHyperparams {
                vocab_size: Some(131_072),
                d_model: Some(6_144),
                n_experts: Some(8),
                d_ff: Some(32_768),
                n_blocks: Some(64),
            },
            tensors,
            blocks,
            totals,
            schema_version: SCHEMA_VERSION,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn tensor(
        shard_ordinal: u32,
        in_shard_index: u32,
        block_index: Option<u32>,
        block_slot: Option<u32>,
        kind: TensorKind,
        role: TensorRole,
        dtype: TensorDType,
        shape: Vec<u64>,
    ) -> TensorInfo {
        TensorInfo {
            shard_path: PathBuf::from(format!(
                "/tmp/grok-1-official/ckpt-0/tensor{shard_ordinal:05}_000"
            )),
            shard_ordinal,
            in_shard_index,
            role,
            dtype,
            shape: TensorShape::new(shape.clone()),
            offset: 0,
            nbytes: dtype.itemsize() as u64 * shape.iter().product::<u64>(),
            kind,
            block_index,
            block_slot,
        }
    }

    fn inventory_totals(tensors: &[TensorInfo]) -> InventoryTotals {
        let mut out = InventoryTotals {
            tensors: tensors.len() as u64,
            ..Default::default()
        };
        for tensor in tensors {
            out.total_nbytes += tensor.nbytes;
            out.total_elements += tensor.shape.numel();
            match tensor.dtype {
                TensorDType::F32 => out.f32_tensors += 1,
                TensorDType::I8 => out.i8_tensors += 1,
            }
            match tensor.role {
                TensorRole::QuantWeight | TensorRole::QuantScales => out.quant_tensors += 1,
                TensorRole::Tensor => {}
            }
        }
        out
    }

    fn sample_routing_report() -> RoutingReport {
        RoutingReport {
            model_family: "grok-1".into(),
            checkpoint_path: PathBuf::from("/tmp/grok-1-official/ckpt-0"),
            shard_count: 2,
            inferred: InferredHyperparams {
                d_model: Some(6144),
                n_experts: Some(8),
                ..Default::default()
            },
            relevant_block_count: 1,
            expected_experts_per_router: Some(8),
            candidate_tensors: vec![RoutingTensorRef {
                shard_ordinal: 1,
                in_shard_index: 0,
                block_index: Some(0),
                block_slot: Some(4),
                role: TensorRole::Tensor,
                dtype: TensorDType::F32,
                shape: TensorShape::new(vec![6144, 8]),
                kind_label: "router".into(),
                orientation: RoutingOrientation::DModelToExperts,
                expert_axis: Some(1),
                linked_expert_count: Some(8),
                matches_inferred_expert_count: true,
                structural_name: "block_000.slot_04.router".into(),
                gate_metrics: RoutingGateMetrics {
                    total_elements: 49_152,
                    total_nbytes: 196_608,
                    input_width: Some(6144),
                    output_width: Some(8),
                    expert_count: Some(8),
                    logits_per_input: Some(8),
                },
            }],
            blocks: vec![RoutingBlockReport {
                block_index: Some(0),
                label: "block_000".into(),
                shard_range: Some(ShardRange {
                    start: 0,
                    end_inclusive: 1,
                }),
                local_expert_count: Some(8),
                primary_candidate: Some(RoutingTensorLocator {
                    shard_ordinal: 1,
                    in_shard_index: 0,
                    block_slot: Some(4),
                }),
                candidates: Vec::new(),
            }],
            orientation_summaries: vec![RoutingOrientationSummary {
                orientation: RoutingOrientation::DModelToExperts,
                count: 1,
                observed_shapes: vec![TensorShape::new(vec![6144, 8])],
                observed_blocks: 1,
            }],
            likely_routing_critical_blocks: vec![RoutingCriticalBlock {
                block_index: Some(0),
                label: "block_000".into(),
                reason: "primary router present".into(),
                primary_candidate: Some(RoutingTensorLocator {
                    shard_ordinal: 1,
                    in_shard_index: 0,
                    block_slot: Some(4),
                }),
            }],
            grok_layout_notes: Vec::new(),
            anomalies: Vec::new(),
            schema_version: 1,
        }
    }
}
