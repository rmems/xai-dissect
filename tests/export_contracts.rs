mod support;

use std::fs;

use xai_dissect::exports;

use support::{
    assert_snapshot_sections, bundle_sections, sample_checkpoint_slug, sample_conversion_manifest,
    sample_expert_atlas, sample_inventory, sample_pilot_selection_plan, sample_quant_plan,
    sample_route_preservation_report, sample_routing_report, sample_saaq_readiness,
    sample_stats_profile, unique_temp_root,
};

#[test]
fn inventory_bundle_matches_snapshot() {
    let root = unique_temp_root("inventory-bundle");
    let bundle = exports::write_inventory_bundle(&sample_inventory(), &root, None)
        .expect("write inventory bundle");
    assert_eq!(bundle.checkpoint_slug, sample_checkpoint_slug());
    let sections = bundle_sections(&root, &bundle);
    assert_snapshot_sections("tests/fixtures/exports/inventory.snap", &sections);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn expert_bundle_matches_snapshot() {
    let root = unique_temp_root("expert-bundle");
    let bundle = exports::write_expert_bundle(&sample_expert_atlas(), &root, None)
        .expect("write expert bundle");
    assert_eq!(bundle.checkpoint_slug, sample_checkpoint_slug());
    let sections = bundle_sections(&root, &bundle);
    assert_snapshot_sections("tests/fixtures/exports/experts.snap", &sections);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn routing_bundle_matches_snapshot() {
    let root = unique_temp_root("routing-bundle");
    let bundle = exports::write_routing_bundle(&sample_routing_report(), &root, None)
        .expect("write routing bundle");
    assert_eq!(bundle.checkpoint_slug, sample_checkpoint_slug());
    let sections = bundle_sections(&root, &bundle);
    assert_snapshot_sections("tests/fixtures/exports/routing-report.snap", &sections);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn stats_bundle_matches_snapshot() {
    let root = unique_temp_root("stats-bundle");
    let bundle = exports::write_stats_bundle(&sample_stats_profile(), &root, None)
        .expect("write stats bundle");
    assert_eq!(bundle.checkpoint_slug, sample_checkpoint_slug());
    let sections = bundle_sections(&root, &bundle);
    assert_snapshot_sections("tests/fixtures/exports/stats.snap", &sections);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn saaq_bundle_matches_snapshot() {
    let root = unique_temp_root("saaq-bundle");
    let bundle = exports::write_saaq_bundle(&sample_saaq_readiness(), &root, None)
        .expect("write saaq bundle");
    assert_eq!(bundle.checkpoint_slug, sample_checkpoint_slug());
    let sections = bundle_sections(&root, &bundle);
    assert_snapshot_sections("tests/fixtures/exports/saaq-readiness.snap", &sections);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn quant_plan_bundle_matches_snapshot() {
    let root = unique_temp_root("quant-plan-bundle");
    let bundle = exports::write_quant_plan_bundle(
        &sample_conversion_manifest(),
        &sample_quant_plan(),
        &root,
        None,
    )
    .expect("write quant-plan bundle");
    assert_eq!(bundle.checkpoint_slug, sample_checkpoint_slug());
    let sections = bundle_sections(&root, &bundle);
    assert_snapshot_sections("tests/fixtures/exports/quant-plan.snap", &sections);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn pilot_plan_bundle_matches_snapshot() {
    let root = unique_temp_root("pilot-plan-bundle");
    let bundle = exports::write_pilot_plan_bundle(&sample_pilot_selection_plan(), &root, None)
        .expect("write pilot-plan bundle");
    assert_eq!(bundle.checkpoint_slug, sample_checkpoint_slug());
    let sections = bundle_sections(&root, &bundle);
    assert_snapshot_sections("tests/fixtures/exports/pilot-plan.snap", &sections);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn route_preservation_bundle_matches_snapshot() {
    let root = unique_temp_root("route-preservation-bundle");
    let bundle =
        exports::write_route_preservation_bundle(&sample_route_preservation_report(), &root, None)
            .expect("write route-preservation bundle");
    assert_eq!(bundle.checkpoint_slug, sample_checkpoint_slug());
    let sections = bundle_sections(&root, &bundle);
    assert_snapshot_sections("tests/fixtures/exports/route-preservation.snap", &sections);
    let _ = fs::remove_dir_all(root);
}
