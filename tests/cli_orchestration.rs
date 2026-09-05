mod support;

use std::fs;
use std::path::Path;
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

fn write_hex_checkpoint(prefix: &str) -> std::path::PathBuf {
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

#[test]
fn analysis_commands_run_against_hex_fixture() {
    let ckpt = write_hex_checkpoint("cli-orch-ckpt");
    let out = unique_temp_root("cli-orch-out");
    fs::create_dir_all(&out).expect("create output dir");
    let ckpt_s = ckpt.to_str().expect("utf8 checkpoint");
    let json = out.join("inventory.json");
    let md = out.join("inventory.md");
    let tree = out.join("tree");
    let experts_json = out.join("experts.json");
    let experts_md = out.join("experts.md");
    let stats_json = out.join("stats.json");
    let stats_md = out.join("stats.md");

    let inventory = run_cli(&[
        "inventory",
        ckpt_s,
        "--json",
        json.to_str().expect("utf8 json"),
        "--md",
        md.to_str().expect("utf8 md"),
        "--output-root",
        tree.to_str().expect("utf8 tree"),
        "--checkpoint-slug",
        "hex-fixture",
    ]);
    assert!(
        inventory.status.success(),
        "inventory failed: {}",
        String::from_utf8_lossy(&inventory.stderr)
    );
    assert!(json.is_file());

    let experts = run_cli(&[
        "experts",
        ckpt_s,
        "--json",
        experts_json.to_str().expect("utf8 experts json"),
        "--md",
        experts_md.to_str().expect("utf8 experts md"),
    ]);
    assert!(
        experts.status.success(),
        "experts failed: {}",
        String::from_utf8_lossy(&experts.stderr)
    );

    let stats = run_cli(&[
        "stats",
        ckpt_s,
        "--sample-values",
        "64",
        "--json",
        stats_json.to_str().expect("utf8 stats json"),
        "--md",
        stats_md.to_str().expect("utf8 stats md"),
    ]);
    assert!(
        stats.status.success(),
        "stats failed: {}",
        String::from_utf8_lossy(&stats.stderr)
    );

    let routing = run_cli(&[
        "routing-report",
        ckpt_s,
        "--json",
        out.join("routing.json")
            .to_str()
            .expect("utf8 routing json"),
        "--md",
        out.join("routing.md").to_str().expect("utf8 routing md"),
    ]);
    assert!(
        routing.status.success(),
        "routing-report failed: {}",
        String::from_utf8_lossy(&routing.stderr)
    );

    let saaq = run_cli(&[
        "saaq-readiness",
        ckpt_s,
        "--sample-values",
        "64",
        "--json",
        out.join("saaq.json").to_str().expect("utf8 saaq json"),
        "--md",
        out.join("saaq.md").to_str().expect("utf8 saaq md"),
        "--manifest",
        out.join("candidates.json")
            .to_str()
            .expect("utf8 candidates"),
    ]);
    assert!(
        saaq.status.success(),
        "saaq-readiness failed: {}",
        String::from_utf8_lossy(&saaq.stderr)
    );

    let dissect = run_cli(&["dissect", ckpt_s]);
    assert!(
        dissect.status.success(),
        "dissect failed: {}",
        String::from_utf8_lossy(&dissect.stderr)
    );

    let pilot = run_cli(&["pilot-plan", ckpt_s]);
    assert!(
        !pilot.status.success(),
        "pilot-plan should reject the incomplete hex fixture"
    );
    let route = run_cli(&["route-preservation", ckpt_s]);
    assert!(
        !route.status.success(),
        "route-preservation should reject the incomplete hex fixture"
    );
    let quant = run_cli(&["quant-plan", ckpt_s]);
    assert!(
        !quant.status.success(),
        "quant-plan should reject the incomplete hex fixture"
    );

    let _ = fs::remove_dir_all(ckpt);
    let _ = fs::remove_dir_all(out);
}
