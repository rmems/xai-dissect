mod support;

use std::fs;

use xai_dissect::exports;

use support::{
    assert_snapshot_sections, bundle_sections, sample_checkpoint_slug, sample_conversion_manifest,
    sample_expert_atlas, sample_inventory, sample_pilot_selection_plan, sample_quant_plan,
    sample_route_preservation_report, sample_routing_report, sample_saaq_readiness,
    sample_stats_profile, unique_temp_root, unmapped_repacked_grok1_inventory,
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

/// Regression: `saaq-readiness.json` must be readable by the type that defines
/// its contract. Before the `SaaqReadinessReportWire` shim, `candidate_targets`
/// and `quantization_candidates` both routed into one field via `serde(alias)`,
/// so every emitted v2 document failed to deserialize with
/// `duplicate field quantization_candidates`. Nothing caught it because no test
/// ever deserialized this type.
#[test]
fn saaq_readiness_round_trips_through_its_own_schema() {
    let report = sample_saaq_readiness();
    let json = serde_json::to_string_pretty(&report).expect("serialize saaq readiness");

    // Both keys are on the wire for pre-v2 consumers.
    let raw: serde_json::Value = serde_json::from_str(&json).expect("parse as generic json");
    assert!(
        raw.get("candidate_targets").is_some() && raw.get("quantization_candidates").is_some(),
        "v2 documents must keep emitting both keys"
    );

    let parsed: xai_dissect::schema::SaaqReadinessReport =
        serde_json::from_str(&json).expect("emitted saaq-readiness.json must deserialize");

    assert_eq!(
        parsed.quantization_candidates.len(),
        report.quantization_candidates.len()
    );
    assert_eq!(
        parsed.candidate_targets.len(),
        parsed.quantization_candidates.len(),
        "the legacy mirror must stay equal to the canonical field"
    );
    assert_eq!(parsed.schema_version, report.schema_version);
}

/// A legacy v1 document carries only `candidate_targets`. It must still load,
/// with the canonical `quantization_candidates` backfilled from it.
#[test]
fn saaq_readiness_reads_legacy_v1_candidate_targets_only() {
    let report = sample_saaq_readiness();
    let mut raw: serde_json::Value =
        serde_json::to_value(&report).expect("serialize saaq readiness");
    let obj = raw.as_object_mut().expect("report is a json object");
    obj.remove("quantization_candidates");
    obj.insert("schema_version".into(), serde_json::json!(1));

    let parsed: xai_dissect::schema::SaaqReadinessReport =
        serde_json::from_value(raw).expect("legacy v1 saaq-readiness.json must deserialize");

    assert_eq!(parsed.schema_version, 1);
    assert_eq!(
        parsed.quantization_candidates.len(),
        report.candidate_targets.len(),
        "v1 candidate_targets must backfill quantization_candidates"
    );
    assert_eq!(
        serde_json::to_value(&parsed.candidate_targets).expect("serialize mirror"),
        serde_json::to_value(&parsed.quantization_candidates).expect("serialize canonical"),
        "the legacy mirror must hold the same candidates as the canonical field"
    );
}

/// An explicitly empty `quantization_candidates: []` array must take precedence
/// over a populated legacy `candidate_targets` array.
#[test]
fn saaq_readiness_prefers_explicitly_empty_canonical_candidates() {
    let report = sample_saaq_readiness();
    let mut raw: serde_json::Value =
        serde_json::to_value(&report).expect("serialize saaq readiness");
    let obj = raw.as_object_mut().expect("report is a json object");
    assert!(
        !obj.get("candidate_targets")
            .and_then(|value| value.as_array())
            .expect("legacy mirror")
            .is_empty(),
        "fixture candidate_targets must be non-empty so this test actually guards precedence"
    );
    obj.insert(
        "quantization_candidates".into(),
        serde_json::Value::Array(Vec::new()),
    );

    let parsed: xai_dissect::schema::SaaqReadinessReport =
        serde_json::from_value(raw).expect("deserialization with empty canonical list succeeds");

    assert!(
        parsed.quantization_candidates.is_empty(),
        "canonical empty list must be preserved"
    );
    assert!(
        parsed.candidate_targets.is_empty(),
        "legacy mirror must reflect canonical empty list"
    );
}

/// Explicit JSON `null` on either candidate key is rejected. Absence is a
/// missing key, not null — otherwise a corrupted v2 document could fall
/// back to the legacy mirror.
#[test]
fn saaq_readiness_rejects_explicit_null_candidate_keys() {
    let report = sample_saaq_readiness();
    for key in ["quantization_candidates", "candidate_targets"] {
        let mut raw: serde_json::Value =
            serde_json::to_value(&report).expect("serialize saaq readiness");
        raw.as_object_mut()
            .expect("report is a json object")
            .insert(key.into(), serde_json::Value::Null);
        let err = serde_json::from_value::<xai_dissect::schema::SaaqReadinessReport>(raw)
            .expect_err("explicit null must fail closed");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid type: null") || msg.contains("explicit null"),
            "{key} null must be a hard error, got: {msg}"
        );
    }
}

/// Rewriting an inventory bundle for an unmapped repack must remove any stale
/// `grok1-coverage.json` left behind from a previous canonical run.
#[test]
fn inventory_bundle_removes_stale_coverage_manifest_on_unmapped_repack() {
    let root = unique_temp_root("inventory-stale-coverage-cleanup");
    let _ = fs::remove_dir_all(&root);

    let inv = unmapped_repacked_grok1_inventory();
    let slug = sample_checkpoint_slug();
    let coverage_dir = root.join("manifests").join(slug);
    fs::create_dir_all(&coverage_dir).expect("create manifests dir");
    let coverage_path = coverage_dir.join("grok1-coverage.json");
    fs::write(&coverage_path, b"{\"validation\":\"stale\"}").expect("write dummy stale manifest");
    assert!(coverage_path.exists(), "stale manifest exists prior to run");

    let bundle = exports::write_inventory_bundle(&inv, &root, None)
        .expect("write inventory bundle skipping coverage");
    assert!(
        !coverage_path.exists(),
        "stale grok1-coverage.json must be removed when coverage is skipped"
    );
    assert!(
        bundle
            .removed_paths
            .iter()
            .any(|path| path == &coverage_path),
        "OutputBundle must record the stale coverage removal"
    );
    assert!(
        bundle
            .written_paths
            .iter()
            .any(|path| path.file_name().and_then(|name| name.to_str()) == Some("inventory.json")),
        "skipped-coverage rewrite must still publish inventory artifacts"
    );
    assert!(
        root.join("exports")
            .join(slug)
            .join("inventory.json")
            .is_file(),
        "exports/<slug>/inventory.json must exist after stale coverage delete"
    );
    assert!(
        !bundle
            .written_paths
            .iter()
            .any(|path| path.file_name().and_then(|name| name.to_str())
                == Some("grok1-coverage.json")),
        "unmapped repack must not emit a coverage manifest"
    );

    let _ = fs::remove_dir_all(&root);
}
