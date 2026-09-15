//! Branch coverage for report renderers moved out of `src/report/mod.rs`.
//!
//! Snapshot tests already cover the happy-path documents. These cases hit
//! empty tables, alternate enum arms, and writers that the export bundles
//! do not call, without changing any golden snapshots.

mod support;

use std::fs;
use std::path::PathBuf;

use xai_dissect::report::{
    render_conversion_manifest_markdown, render_expert_markdown, render_markdown,
    render_quant_plan_markdown, render_route_preservation_markdown, render_routing_markdown,
    render_saaq_readiness_markdown, render_stats_markdown, write_conversion_manifest_markdown,
    write_findings_summary_json,
};
use xai_dissect::schema::{
    ExpertIssue, ExpertIssueCategory, ExpertIssueSeverity, ExpertTensorLocator, FindingsSeverity,
    FindingsSummary, FindingsSummaryItem, MetricStatus, QuantPolicy, RouteMetricStatus,
    RoutingIssue, RoutingIssueCategory, RoutingIssueSeverity, RoutingTensorLocator,
    SaaqDisposition, SaaqLayerReadiness, SaaqRegionClass,
};

use support::{
    sample_conversion_manifest, sample_expert_atlas, sample_inventory, sample_quant_plan,
    sample_route_preservation_report, sample_routing_report, sample_saaq_readiness,
    sample_stats_profile, unique_temp_root,
};

#[test]
fn inventory_markdown_covers_missing_shard_range_and_empty_kinds() {
    let mut inv = sample_inventory();
    inv.blocks[0].shard_range = None;
    inv.blocks[0].kinds.clear();
    inv.inferred = Default::default();
    inv.totals.total_nbytes = 1024u64.pow(4);
    inv.blocks[1].block_index = Some(1);
    inv.blocks[1].total_nbytes = 1024u64.pow(3);
    for tensor in &mut inv.tensors {
        if tensor.block_index == Some(0) {
            tensor.block_index = Some(1);
        }
    }
    let md = render_markdown(&inv);
    assert!(md.contains("| embedding | - | - |"));
    assert!(md.contains("| vocab_size | - |"));
    assert!(md.contains("1.00 TiB"));
    assert!(md.contains("1.00 GiB"));
    assert!(!md.contains("## Exemplar block"));
}

#[test]
fn expert_markdown_covers_empty_lists_failed_checks_and_issues() {
    let mut atlas = sample_expert_atlas();
    atlas.blocks[0].tensors.clear();
    atlas.blocks[0].experts[0].tensors.clear();
    atlas.naming_patterns[0].observed_shapes.clear();
    atlas.naming_patterns[0].block_slots.clear();
    atlas.naming_checks[0].passed = false;
    atlas.anomalies = vec![
        ExpertIssue {
            severity: ExpertIssueSeverity::Warning,
            category: ExpertIssueCategory::MissingOrIrregularTensor,
            block_index: Some(0),
            tensor: Some(ExpertTensorLocator {
                shard_ordinal: 1,
                in_shard_index: 0,
                block_slot: Some(3),
            }),
            message: "missing family".into(),
        },
        ExpertIssue {
            severity: ExpertIssueSeverity::Error,
            category: ExpertIssueCategory::LayoutAnomaly,
            block_index: None,
            tensor: Some(ExpertTensorLocator {
                shard_ordinal: 2,
                in_shard_index: 1,
                block_slot: None,
            }),
            message: "layout".into(),
        },
        ExpertIssue {
            severity: ExpertIssueSeverity::Warning,
            category: ExpertIssueCategory::NamingConsistency,
            block_index: Some(1),
            tensor: None,
            message: "naming".into(),
        },
    ];
    let md = render_expert_markdown(&atlas);
    assert!(md.contains("| - |"));
    assert!(md.contains("| fail |"));
    assert!(md.contains("missing family"));
    assert!(md.contains("shard 2 idx 1 slot ?"));
    assert!(md.contains("| 1 | warning | - | naming |"));
    assert!(md.contains("| error |"));

    atlas.blocks.clear();
    let empty = render_expert_markdown(&atlas);
    assert!(!empty.contains("## Exemplar block"));
}

#[test]
fn routing_markdown_covers_empty_shapes_unmatched_experts_and_issues() {
    let mut report = sample_routing_report();
    report.orientation_summaries[0].observed_shapes.clear();
    report.candidate_tensors[0].matches_inferred_expert_count = false;
    report.anomalies = vec![
        RoutingIssue {
            severity: RoutingIssueSeverity::Warning,
            category: RoutingIssueCategory::ShapeSummary,
            block_index: Some(0),
            tensor: Some(RoutingTensorLocator {
                shard_ordinal: 1,
                in_shard_index: 0,
                block_slot: Some(0),
            }),
            message: "shape note".into(),
        },
        RoutingIssue {
            severity: RoutingIssueSeverity::Error,
            category: RoutingIssueCategory::ExpertCountLinkage,
            block_index: None,
            tensor: Some(RoutingTensorLocator {
                shard_ordinal: 3,
                in_shard_index: 2,
                block_slot: None,
            }),
            message: "linkage".into(),
        },
        RoutingIssue {
            severity: RoutingIssueSeverity::Warning,
            category: RoutingIssueCategory::LayoutNote,
            block_index: Some(4),
            tensor: None,
            message: "layout".into(),
        },
        RoutingIssue {
            severity: RoutingIssueSeverity::Error,
            category: RoutingIssueCategory::MissingCandidate,
            block_index: Some(5),
            tensor: None,
            message: "missing router".into(),
        },
    ];
    let md = render_routing_markdown(&report);
    assert!(md.contains("| no |"));
    assert!(md.contains("shape_summary"));
    assert!(md.contains("expert_count_linkage"));
    assert!(md.contains("layout_note"));
    assert!(md.contains("missing_candidate"));
    assert!(md.contains("shard 3 idx 2 slot ?"));
    assert!(md.contains("missing router"));

    report.candidate_tensors[0].block_index = None;
    report.candidate_tensors[0].block_slot = None;
    report.candidate_tensors[0].linked_expert_count = None;
    report.candidate_tensors[0].gate_metrics.total_nbytes = 1024 * 1024;
    report.candidate_tensors[0].gate_metrics.input_width = None;
    report.blocks[0].block_index = None;
    report.blocks[0].local_expert_count = None;
    report.blocks[0].primary_candidate = None;
    report.likely_routing_critical_blocks.clear();
    report.grok_layout_notes.clear();
    let empty = render_routing_markdown(&report);
    assert!(empty.contains("| - | - |"));
    assert!(empty.contains("1.00 MiB"));
    assert!(empty.contains("## Likely routing-critical blocks"));
    assert!(empty.contains("## Grok-specific layout notes"));
    assert_eq!(empty.matches("None detected.").count(), 2);
}

#[test]
fn stats_markdown_covers_empty_ranked_tables() {
    let mut report = sample_stats_profile();
    report.norm_summary.top_rms.clear();
    let md = render_stats_markdown(&report);
    assert!(md.contains("None detected."));
}

#[test]
fn saaq_markdown_covers_empty_deferred_and_remaining_labels() {
    let mut report = sample_saaq_readiness();
    report.deferred_tensors.clear();
    report.quantization_candidates[0].region_class = SaaqRegionClass::AlreadyCompressed;
    report.quantization_candidates[0].disposition = SaaqDisposition::ObserveOnly;
    let extra = report.quantization_candidates[0].clone();
    for (name, region, disposition) in [
        (
            "blk.0.attn.qkv.weight",
            SaaqRegionClass::Unknown,
            SaaqDisposition::ObserveOnly,
        ),
        (
            "blk.0.moe.router.weight",
            SaaqRegionClass::RoutingCritical,
            SaaqDisposition::AvoidForNow,
        ),
        (
            "blk.0.moe.expert.0.mlp.linear_1.weight",
            SaaqRegionClass::PotentialCompressionTarget,
            SaaqDisposition::Candidate,
        ),
        (
            "token_embd.weight",
            SaaqRegionClass::EmbeddingHeavy,
            SaaqDisposition::ObserveOnly,
        ),
    ] {
        let mut tensor = extra.clone();
        tensor.structural_name = name.into();
        tensor.region_class = region;
        tensor.disposition = disposition;
        report.quantization_candidates.push(tensor);
    }
    report.risky_tensors[0].region_class = SaaqRegionClass::Unknown;
    report.layer_readiness.push(SaaqLayerReadiness {
        block_index: None,
        label: "embedding".into(),
        routing_critical: false,
        candidate_target_count: 0,
        mean_readiness_score: 0.0,
        max_risk_score: 0.0,
    });
    let md = render_saaq_readiness_markdown(&report);
    assert!(md.contains("## Deferred tensors"));
    assert!(md.contains("None detected."));
    assert!(md.contains("already_compressed"));
    assert!(md.contains("observe_only"));
    assert!(md.contains("unknown"));
    assert!(md.contains("routing_critical"));
    assert!(md.contains("avoid_for_now"));
    assert!(md.contains("potential_target"));
    assert!(md.contains("embedding_heavy"));
    assert!(md.contains("| embedding | - | no |"));

    report.routing_critical_tensors.clear();
    report.precision_sensitive_tensors.clear();
    report.risky_tensors.clear();
    let empty_tables = render_saaq_readiness_markdown(&report);
    assert!(empty_tables.contains("## Routing-critical tensors"));
    assert!(empty_tables.contains("## Precision-sensitive tensors"));
    assert!(empty_tables.contains("## Highest-risk tensors"));
    assert_eq!(
        empty_tables.matches("None detected.").count(),
        4,
        "deferred, routing-critical, precision-sensitive, and risky tables should all be empty"
    );
}

#[test]
fn quant_plan_markdown_covers_empty_notes() {
    let mut plan = sample_quant_plan();
    plan.notes.clear();
    let md = render_quant_plan_markdown(&plan);
    assert!(md.contains("## Notes"));
    assert!(md.contains("None."));
}

#[test]
fn conversion_manifest_markdown_covers_policies_warnings_and_optional_router() {
    let mut manifest = sample_conversion_manifest();
    let template = manifest.tensors[0].clone();
    for (kind, policy) in [
        ("moe_expert.gate", QuantPolicy::WrapExistingInt8Expert),
        ("unknown_i8", QuantPolicy::WrapExistingInt8Unknown),
        ("unknown_f32", QuantPolicy::UnknownPassthroughOrWarn),
    ] {
        let mut tensor = template.clone();
        tensor.kind = kind.into();
        tensor.quant_policy = policy;
        tensor.structural_name = format!("extra.{kind}");
        manifest.tensors.push(tensor);
    }
    manifest.warnings.push("review unknown tensors".into());
    let md = render_conversion_manifest_markdown(&manifest);
    assert!(md.contains("`WrapExistingInt8Expert`"));
    assert!(md.contains("`WrapExistingInt8Unknown`"));
    assert!(md.contains("`UnknownPassthroughOrWarn`"));
    assert!(md.contains("review unknown tensors"));
    assert!(md.contains("**router_shape**"));

    manifest.router_shape = None;
    manifest.router_orientation = None;
    manifest.warnings.clear();
    let empty = render_conversion_manifest_markdown(&manifest);
    assert!(!empty.contains("**router_shape**"));
    assert!(empty.contains("None."));

    let root = unique_temp_root("conversion-md");
    let path = root.join("reports").join("conversion-manifest.md");
    write_conversion_manifest_markdown(&manifest, &path).expect("write conversion markdown");
    let on_disk = fs::read_to_string(&path).expect("read conversion markdown");
    assert_eq!(on_disk, empty);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn route_preservation_markdown_covers_pass_and_fail_status() {
    let mut report = sample_route_preservation_report();
    report.router_metrics[0].status = MetricStatus::Pass;
    report.router_metrics[0].observed = Some("99.2%".into());
    report.block_metrics = vec![RouteMetricStatus {
        name: "block_rms_delta".into(),
        scope: "block".into(),
        status: MetricStatus::Fail,
        threshold: None,
        observed: None,
        detail: "exceeded".into(),
    }];
    let md = render_route_preservation_markdown(&report);
    assert!(md.contains("| pass |"));
    assert!(md.contains("| fail |"));
}

#[test]
fn findings_summary_json_writes_when_path_has_no_parent() {
    let summary = FindingsSummary {
        analysis: "inventory".into(),
        model_family: "grok-1".into(),
        checkpoint_path: PathBuf::from("/fixtures/grok-1-official/ckpt-0"),
        checkpoint_slug: "grok-1-official__ckpt-0".into(),
        headline: "ok".into(),
        findings: vec![FindingsSummaryItem {
            severity: FindingsSeverity::Info,
            category: "test".into(),
            detail: "n/a".into(),
        }],
        schema_version: 1,
    };
    let name = format!(
        "xai-dissect-findings-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    );
    write_findings_summary_json(&summary, name.as_ref()).expect("write relative findings json");
    let body = fs::read_to_string(&name).expect("read findings json");
    let _ = fs::remove_file(&name);
    assert!(body.contains("\"analysis\": \"inventory\""));
}
