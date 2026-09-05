// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail};
use tempfile::TempDir;
use xai_dissect::exports::OutputBundle;
use xai_dissect::report;

use super::{
    CheckpointScanArgs, OutputTreeArgs, run_experts, run_inventory, run_pilot_plan, run_quant_plan,
    run_route_preservation, run_routing_report, run_saaq_readiness, run_stats,
    validate_complete_inventory_scope, write_json_and_markdown, write_optional_file,
    write_output_tree,
};

fn unique_temp_root(prefix: &str) -> TempDir {
    tempfile::Builder::new()
        .prefix(&format!("xai-dissect-cli-{prefix}-"))
        .tempdir()
        .expect("create unique temp dir")
}

fn decode_hex_fixture() -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/parser/single_f32_tensor.pkl.hex");
    let hex = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read fixture {}: {err}", path.display()));
    let hex = hex
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    assert_eq!(hex.len() % 2, 0, "fixture must have an even hex length");

    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut i = 0;
    while i < hex.len() {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16).expect("hex byte");
        out.push(byte);
        i += 2;
    }
    out
}

fn write_hex_checkpoint(prefix: &str) -> TempDir {
    let root = unique_temp_root(prefix);
    fs::write(root.path().join("tensor0000.pkl"), decode_hex_fixture()).expect("write shard");
    root
}

fn scan_for(path: PathBuf) -> CheckpointScanArgs {
    CheckpointScanArgs {
        path,
        prefix: "tensor".into(),
        limit: None,
    }
}

fn write_text(doc: &String, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, doc)?;
    Ok(())
}

fn render_text(doc: &String) -> String {
    format!("rendered:{doc}")
}

#[test]
fn validate_complete_inventory_scope_accepts_default_prefix_without_limit() {
    validate_complete_inventory_scope("pilot-plan", "tensor", None).expect("default scope");
}

#[test]
fn validate_complete_inventory_scope_rejects_non_default_prefix() {
    let err = validate_complete_inventory_scope("quant-plan", "tensor-shard", None).unwrap_err();
    assert!(format!("{err:#}").contains("--prefix tensor-shard"));
}

#[test]
fn validate_complete_inventory_scope_rejects_limit() {
    let err = validate_complete_inventory_scope("quant-plan", "tensor", Some(4)).unwrap_err();
    assert!(format!("{err:#}").contains("--limit 4"));
}

#[test]
fn write_optional_file_skips_when_path_unset() {
    write_optional_file(None, "optional", |_| {
        bail!("writer must not run when path is unset")
    })
    .expect("skip unset path");
}

#[test]
fn write_optional_file_writes_when_path_set() {
    let root = unique_temp_root("optional-file");
    let path = root.path().join("artifact.txt");
    write_optional_file(Some(&path), "optional", |p| {
        fs::write(p, "payload")?;
        Ok(())
    })
    .expect("write optional file");
    assert_eq!(fs::read_to_string(&path).expect("read"), "payload");
}

#[test]
fn write_json_and_markdown_writes_both_files() {
    let root = unique_temp_root("json-md");
    let json = root.path().join("doc.json");
    let md = root.path().join("doc.md");
    let doc = String::from("body");
    write_json_and_markdown(
        &doc,
        Some(&json),
        Some(&md),
        "JSON doc",
        "Markdown doc",
        write_text,
        write_text,
        render_text,
    )
    .expect("write json and markdown");
    assert_eq!(fs::read_to_string(&json).expect("json"), "body");
    assert_eq!(fs::read_to_string(&md).expect("md"), "body");
}

#[test]
fn write_json_and_markdown_prints_when_md_unset() {
    let root = unique_temp_root("json-stdout");
    let json = root.path().join("doc.json");
    let doc = String::from("body");
    write_json_and_markdown(
        &doc,
        Some(&json),
        None,
        "JSON doc",
        "Markdown doc",
        write_text,
        write_text,
        render_text,
    )
    .expect("print markdown when unset");
    assert_eq!(fs::read_to_string(&json).expect("json"), "body");
}

#[test]
fn write_output_tree_skips_when_root_unset() {
    write_output_tree(
        &OutputTreeArgs {
            output_root: None,
            checkpoint_slug: None,
        },
        "bundle",
        "{reports}",
        |_, _| bail!("writer must not run when output_root is unset"),
    )
    .expect("skip unset output root");
}

#[test]
fn write_output_tree_writes_when_root_set() {
    let root = unique_temp_root("output-tree");
    let root_path = root.path().to_path_buf();
    let called = AtomicBool::new(false);
    write_output_tree(
        &OutputTreeArgs {
            output_root: Some(root_path.clone()),
            checkpoint_slug: Some("hex-fixture".into()),
        },
        "bundle",
        "{reports,exports}",
        |out_root, slug| {
            called.store(true, Ordering::Relaxed);
            assert_eq!(out_root, root_path.as_path());
            assert_eq!(slug, Some("hex-fixture"));
            Ok(OutputBundle {
                checkpoint_slug: slug.unwrap_or("inferred").to_string(),
                written_paths: vec![out_root.join("marker")],
            })
        },
    )
    .expect("write output tree");
    assert!(called.load(Ordering::Relaxed));
}

#[test]
fn run_inventory_writes_json_markdown_and_output_tree() {
    let ckpt = write_hex_checkpoint("run-inventory");
    let out = unique_temp_root("run-inventory-out");
    let json = out.path().join("inventory.json");
    let md = out.path().join("inventory.md");
    let tree = out.path().join("tree");
    run_inventory(
        &scan_for(ckpt.path().to_path_buf()),
        "grok-1",
        Some(&json),
        Some(&md),
        &OutputTreeArgs {
            output_root: Some(tree.clone()),
            checkpoint_slug: Some("hex-fixture".into()),
        },
    )
    .expect("run inventory");
    assert!(json.is_file(), "json inventory");
    assert!(md.is_file(), "markdown inventory");
    assert!(
        tree.join("exports/hex-fixture/inventory.json").is_file(),
        "output-tree inventory"
    );
}

#[test]
fn run_inventory_prints_markdown_when_md_unset() {
    let ckpt = write_hex_checkpoint("run-inventory-stdout");
    run_inventory(
        &scan_for(ckpt.path().to_path_buf()),
        "grok-1",
        None,
        None,
        &OutputTreeArgs {
            output_root: None,
            checkpoint_slug: None,
        },
    )
    .expect("run inventory without files");
}

#[test]
fn run_experts_writes_json_and_markdown() {
    let ckpt = write_hex_checkpoint("run-experts");
    let out = unique_temp_root("run-experts-out");
    let json = out.path().join("experts.json");
    let md = out.path().join("experts.md");
    let tree = out.path().join("tree");
    run_experts(
        &scan_for(ckpt.path().to_path_buf()),
        "grok-1",
        Some(&json),
        Some(&md),
        &OutputTreeArgs {
            output_root: Some(tree.clone()),
            checkpoint_slug: Some("hex-fixture".into()),
        },
    )
    .expect("run experts");
    assert!(json.is_file());
    assert!(md.is_file());
    assert!(tree.join("exports/hex-fixture/experts.json").is_file());
}

#[test]
fn run_stats_and_saaq_readiness_against_hex_fixture() {
    let ckpt = write_hex_checkpoint("run-stats");
    let out = unique_temp_root("run-stats-out");
    let scan = scan_for(ckpt.path().to_path_buf());
    let tree = OutputTreeArgs {
        output_root: Some(out.path().join("tree")),
        checkpoint_slug: Some("hex-fixture".into()),
    };
    run_stats(
        &scan,
        "grok-1",
        64,
        Some(&out.path().join("stats.json")),
        Some(&out.path().join("stats.md")),
        &tree,
    )
    .expect("run stats");
    run_saaq_readiness(
        &scan,
        "grok-1",
        64,
        Some(&out.path().join("saaq.json")),
        Some(&out.path().join("saaq.md")),
        Some(&out.path().join("candidates.json")),
        &tree,
    )
    .expect("run saaq-readiness");
    assert!(out.path().join("stats.json").is_file());
    assert!(out.path().join("saaq.json").is_file());
    assert!(out.path().join("candidates.json").is_file());
    assert!(
        out.path()
            .join("tree/exports/hex-fixture/stats.json")
            .is_file()
    );
    assert!(
        out.path()
            .join("tree/exports/hex-fixture/saaq-readiness.json")
            .is_file()
    );
}

#[test]
fn run_routing_report_against_hex_fixture() {
    let ckpt = write_hex_checkpoint("run-routing");
    let out = unique_temp_root("run-routing-out");
    run_routing_report(
        &scan_for(ckpt.path().to_path_buf()),
        "grok-1",
        Some(&out.path().join("routing.json")),
        Some(&out.path().join("routing.md")),
        &OutputTreeArgs {
            output_root: Some(out.path().join("tree")),
            checkpoint_slug: Some("hex-fixture".into()),
        },
    )
    .expect("run routing-report");
    assert!(out.path().join("routing.json").is_file());
}

#[test]
fn planning_commands_reject_incomplete_hex_fixture() {
    let ckpt = write_hex_checkpoint("run-planning");
    let scan = scan_for(ckpt.path().to_path_buf());
    let no_tree = OutputTreeArgs {
        output_root: None,
        checkpoint_slug: None,
    };
    let pilot = run_pilot_plan(&scan, "grok-1", None, None, &no_tree).unwrap_err();
    assert!(
        format!("{pilot:#}").contains("Grok-1") || format!("{pilot:#}").contains("coverage"),
        "{pilot:#}"
    );
    let route = run_route_preservation(&scan, "grok-1", None, None, &no_tree).unwrap_err();
    assert!(
        format!("{route:#}").contains("Grok-1") || format!("{route:#}").contains("coverage"),
        "{route:#}"
    );
    let quant = run_quant_plan(&scan, "grok-1", 64, None, None, None, None, &no_tree).unwrap_err();
    assert!(
        format!("{quant:#}").contains("Grok-1") || format!("{quant:#}").contains("coverage"),
        "{quant:#}"
    );
}

#[test]
fn planning_commands_reject_filtered_scan_before_inventory() {
    let ckpt = write_hex_checkpoint("run-planning-filter");
    let mut scan = scan_for(ckpt.path().to_path_buf());
    scan.prefix = "tensor-shard".into();
    let no_tree = OutputTreeArgs {
        output_root: None,
        checkpoint_slug: None,
    };
    let err = run_pilot_plan(&scan, "grok-1", None, None, &no_tree).unwrap_err();
    assert!(format!("{err:#}").contains("--prefix tensor-shard"));
}

#[test]
fn write_json_and_markdown_uses_real_inventory_renderers() {
    let ckpt = write_hex_checkpoint("inventory-renderers");
    let inv = xai_dissect::inventory::build_inventory(
        ckpt.path(),
        &xai_dissect::inventory::InventoryConfig {
            prefix: "tensor".into(),
            limit: None,
            model_family: "grok-1".into(),
        },
    )
    .expect("build inventory");
    let out = unique_temp_root("inventory-renderers-out");
    write_json_and_markdown(
        &inv,
        Some(&out.path().join("inv.json")),
        Some(&out.path().join("inv.md")),
        "JSON inventory",
        "Markdown report",
        report::write_json,
        report::write_markdown,
        report::render_markdown,
    )
    .expect("write inventory via helpers");
    assert!(out.path().join("inv.json").is_file());
}
