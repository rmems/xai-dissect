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
