use std::process::Command;

#[test]
fn top_level_help_lists_current_commands() {
    let stdout = run_help(&["--help"]);
    for command in [
        "inventory",
        "experts",
        "routing-report",
        "stats",
        "saaq-readiness",
        "pilot-plan",
        "route-preservation",
        "quant-plan",
    ] {
        assert!(
            stdout.contains(command),
            "top-level help is missing command {command}"
        );
    }
}

#[test]
fn analysis_commands_expose_output_tree_options() {
    for command in [
        "inventory",
        "experts",
        "routing-report",
        "stats",
        "saaq-readiness",
        "pilot-plan",
        "route-preservation",
        "quant-plan",
    ] {
        let stdout = run_help(&[command, "--help"]);
        assert!(
            stdout.contains("--output-root <OUTPUT_ROOT>"),
            "{command} help is missing --output-root"
        );
        assert!(
            stdout.contains("--checkpoint-slug <CHECKPOINT_SLUG>"),
            "{command} help is missing --checkpoint-slug"
        );
    }
}

#[test]
fn dissect_help_stays_parser_only() {
    let stdout = run_help(&["dissect", "--help"]);
    assert!(!stdout.contains("--output-root"));
    assert!(!stdout.contains("--checkpoint-slug"));
}

#[test]
fn family_flag_rejects_unsupported_family() {
    // An unknown --family must be a hard CLI error. Accepting it would stamp a
    // foreign model_family label onto Grok-1-layout artifacts and disable the
    // coverage gate (should_validate_grok1_coverage keys on model_family).
    for command in [
        "inventory",
        "experts",
        "routing-report",
        "stats",
        "saaq-readiness",
        "pilot-plan",
        "route-preservation",
        "quant-plan",
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_xai-dissect"))
            .args([command, "/tmp/ckpt", "--family", "grok-2"])
            .output()
            .expect("run xai-dissect");
        assert!(
            !output.status.success(),
            "{command} accepted unsupported --family grok-2"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("invalid value"),
            "{command} rejected --family for the wrong reason: {stderr}"
        );
    }
}

fn run_help(args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_xai-dissect"))
        .args(args)
        .output()
        .expect("run xai-dissect");
    assert!(
        output.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("stdout utf8")
}
