mod support;

use xai_dissect::report::{
    render_pilot_selection_plan_markdown, render_route_preservation_markdown,
};

#[test]
fn pilot_selection_plan_lists_expected_blocks_and_modes() {
    let plan = xai_dissect::schema::PilotSelectionPlan {
        model_family: "grok-1".into(),
        checkpoint_path: "/fixtures/grok-1-official/ckpt-0".into(),
        baseline: "grok1-map-v1-clean".into(),
        required_validation: support::sample_quant_plan().required_validation,
        selected_blocks: vec![
            xai_dissect::schema::PilotBlockSelection {
                block_index: 0,
                label: "block_000".into(),
                rationale: "early baseline".into(),
            },
            xai_dissect::schema::PilotBlockSelection {
                block_index: 63,
                label: "block_063".into(),
                rationale: "late-layer / high peak-to-rms router region".into(),
            },
        ],
        modes: vec![
            xai_dissect::schema::PilotQuantizationMode::AttentionOnly,
            xai_dissect::schema::PilotQuantizationMode::ExpertOnly,
            xai_dissect::schema::PilotQuantizationMode::AttentionPlusExpert,
        ],
        protection_rules: vec!["router tensors must remain untouched".into()],
        comparison_artifacts: vec!["pilot-selection-plan.json".into()],
        notes: vec!["planning only".into()],
        schema_version: 1,
    };
    let md = render_pilot_selection_plan_markdown(&plan);
    assert!(md.contains("block_000"));
    assert!(md.contains("block_063"));
    assert!(md.contains("attention_only"));
    assert!(md.contains("expert_only"));
    assert!(md.contains("attention_plus_expert"));
}

#[test]
fn route_preservation_markdown_shows_router_thresholds() {
    let report = xai_dissect::schema::RoutePreservationReport {
        model_family: "grok-1".into(),
        checkpoint_path: "/fixtures/grok-1-official/ckpt-0".into(),
        baseline: "grok1-map-v1-clean".into(),
        required_validation: support::sample_quant_plan().required_validation,
        summary: vec![],
        router_metrics: vec![xai_dissect::schema::RouteMetricStatus {
            name: "router_top1_agreement".into(),
            scope: "router_behavior".into(),
            status: xai_dissect::schema::MetricStatus::Unknown,
            threshold: Some(">= 99.0%".into()),
            observed: None,
            detail: "pending downstream pilot evidence".into(),
        }],
        block_metrics: vec![],
        weight_metrics: vec![],
        notes: vec!["planning only".into()],
        schema_version: 1,
    };
    let md = render_route_preservation_markdown(&report);
    assert!(md.contains("router_top1_agreement"));
    assert!(md.contains(">= 99.0%"));
    assert!(md.contains("unknown"));
}
