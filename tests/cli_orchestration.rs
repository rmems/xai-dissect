mod support;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use support::unique_temp_root;

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

fn write_hex_checkpoint(prefix: &str) -> PathBuf {
    let root = unique_temp_root(prefix);
    fs::create_dir_all(&root).expect("create checkpoint dir");
    fs::write(root.join("tensor0000.pkl"), decode_hex_fixture()).expect("write shard");
    root
}

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_xai-dissect"))
        .args(args)
        .output()
        .expect("run xai-dissect")
}

fn assert_cli_ok(label: &str, args: &[&str]) {
    let output = run_cli(args);
    assert!(
        output.status.success(),
        "{label} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_cli_fails(label: &str, args: &[&str]) {
    let output = run_cli(args);
    assert!(
        !output.status.success(),
        "{label} should reject the incomplete hex fixture"
    );
}

fn hex_cli_paths(prefix: &str) -> (PathBuf, PathBuf) {
    let ckpt = write_hex_checkpoint(prefix);
    let out = unique_temp_root(&format!("{prefix}-out"));
    fs::create_dir_all(&out).expect("create output dir");
    (ckpt, out)
}

fn utf8(path: &Path) -> &str {
    path.to_str().expect("utf8 path")
}

fn assert_json_md(label: &str, ckpt: &str, json: &Path, md: &Path) {
    assert_cli_ok(
        label,
        &[label, ckpt, "--json", utf8(json), "--md", utf8(md)],
    );
}

fn run_inventory_tree(ckpt: &str, out: &Path) {
    let json = out.join("inventory.json");
    let md = out.join("inventory.md");
    let tree = out.join("tree");
    assert_cli_ok(
        "inventory",
        &[
            "inventory",
            ckpt,
            "--json",
            utf8(&json),
            "--md",
            utf8(&md),
            "--output-root",
            utf8(&tree),
            "--checkpoint-slug",
            "hex-fixture",
        ],
    );
    assert!(json.is_file());
}

#[test]
fn inventory_experts_and_stats_succeed_against_hex_fixture() {
    let (ckpt, out) = hex_cli_paths("cli-orch-inv");
    let ckpt_s = utf8(&ckpt);
    run_inventory_tree(ckpt_s, &out);
    assert_json_md(
        "experts",
        ckpt_s,
        &out.join("experts.json"),
        &out.join("experts.md"),
    );
    assert_cli_ok(
        "stats",
        &[
            "stats",
            ckpt_s,
            "--sample-values",
            "64",
            "--json",
            utf8(&out.join("stats.json")),
            "--md",
            utf8(&out.join("stats.md")),
        ],
    );
    let _ = fs::remove_dir_all(ckpt);
    let _ = fs::remove_dir_all(out);
}

#[test]
fn routing_saaq_and_dissect_succeed_against_hex_fixture() {
    let (ckpt, out) = hex_cli_paths("cli-orch-route");
    let ckpt_s = utf8(&ckpt);
    assert_json_md(
        "routing-report",
        ckpt_s,
        &out.join("routing.json"),
        &out.join("routing.md"),
    );
    assert_cli_ok(
        "saaq-readiness",
        &[
            "saaq-readiness",
            ckpt_s,
            "--sample-values",
            "64",
            "--json",
            utf8(&out.join("saaq.json")),
            "--md",
            utf8(&out.join("saaq.md")),
            "--manifest",
            utf8(&out.join("candidates.json")),
        ],
    );
    assert_cli_ok("dissect", &["dissect", ckpt_s]);
    let _ = fs::remove_dir_all(ckpt);
    let _ = fs::remove_dir_all(out);
}

#[test]
fn planning_commands_fail_closed_against_hex_fixture() {
    let ckpt = write_hex_checkpoint("cli-orch-plan");
    let ckpt_s = utf8(&ckpt);
    assert_cli_fails("pilot-plan", &["pilot-plan", ckpt_s]);
    assert_cli_fails("route-preservation", &["route-preservation", ckpt_s]);
    assert_cli_fails("quant-plan", &["quant-plan", ckpt_s]);
    let _ = fs::remove_dir_all(ckpt);
}
