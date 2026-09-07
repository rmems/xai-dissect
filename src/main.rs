// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! xai-dissect CLI. Thin wrapper over the `xai_dissect` library crate.
//!
//! ## Command map
//!
//! | Command | Driven by | Produces |
//! |---------|----------|---------|
//! | `dissect` | `parser::dissect_shard` | Raw tensor table (no classification) |
//! | `inventory` | `inventory::build_inventory` | Full tensor catalog + coverage manifest |
//! | `experts` | `experts::build_expert_atlas` | Expert atlas |
//! | `routing-report` | `routing::build_routing_report` | Routing structure + critical-tensors manifest |
//! | `stats` | `stats::build_stats_report` | Offline tensor statistics |
//! | `saaq-readiness` | `stats::build_saaq_readiness_report` | SAAQ candidate manifest |
//! | `pilot-plan` | `planning::build_grok1_pilot_selection_plan` | Pilot block selection plan |
//! | `route-preservation` | `planning::build_grok1_route_preservation_report` | Route preservation gate report |
//! | `quant-plan` | `planning::build_grok1_planning_artifacts` | Conversion manifest + quant plan |
//!
//! ## OutputTreeArgs semantics
//! `--output-root` enables the unified output tree (`reports/`, `exports/`,
//! `manifests/`). `--checkpoint-slug` overrides the inferred slug and is
//! required when `--output-root` is set and a custom name is needed.
//!
//! ## What this module does NOT do
//! - It does **not** execute model inference
//! - It does **not** mutate checkpoint files
//! - It does **not** implement quantization kernels

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use comfy_table::{Cell, ContentArrangement, Table, presets::UTF8_FULL};

mod cli;
mod observability;

/// Test helpers shared by this module's tests and `cli::tests`.
#[cfg(test)]
mod test_support;

use cli::{CheckpointScanArgs, ModelFamilyArg, OutputTreeArgs, PlanningFamilyArg, SampleValuesArg};
use xai_dissect::parser;

#[derive(Parser, Debug)]
#[command(
    name = "xai-dissect",
    version,
    about = "Static structural analysis of Grok-family open-weight checkpoints"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Parse each shard and print a per-shard tensor table. Raw parser
    /// output only; no classification, no grouping.
    Dissect {
        /// Directory containing `tensor*` shard files (non-recursive).
        path: PathBuf,
        /// Only process the first N shards (sorted by filename).
        #[arg(long)]
        limit: Option<usize>,
        /// Filename prefix filter.
        #[arg(long, default_value = "tensor")]
        prefix: String,
    },
    /// Build a full inventory of a checkpoint directory: parse, classify,
    /// group by block, and optionally export JSON and Markdown.
    Inventory {
        #[command(flatten)]
        scan: CheckpointScanArgs,
        #[command(flatten)]
        family: ModelFamilyArg,
        /// If set, write the full inventory as pretty JSON to this path.
        #[arg(long)]
        json: Option<PathBuf>,
        /// If set, write the Markdown summary to this path. If unset, the
        /// Markdown summary is printed to stdout.
        #[arg(long)]
        md: Option<PathBuf>,
        #[command(flatten)]
        output_tree: OutputTreeArgs,
    },
    /// Build an expert-level atlas of a checkpoint directory: discover
    /// expert-stacked tensors, map blocks to expert counts, and optionally
    /// export JSON and Markdown.
    Experts {
        #[command(flatten)]
        scan: CheckpointScanArgs,
        #[command(flatten)]
        family: ModelFamilyArg,
        /// If set, write the full expert atlas as pretty JSON to this path.
        #[arg(long)]
        json: Option<PathBuf>,
        /// If set, write the expert atlas Markdown report to this path. If
        /// unset, the Markdown report is printed to stdout.
        #[arg(long)]
        md: Option<PathBuf>,
        #[command(flatten)]
        output_tree: OutputTreeArgs,
    },
    /// Build a routing-structure report for a checkpoint directory:
    /// identify likely router tensors, summarize their geometry, and
    /// optionally export JSON and Markdown.
    RoutingReport {
        #[command(flatten)]
        scan: CheckpointScanArgs,
        #[command(flatten)]
        family: ModelFamilyArg,
        /// If set, write the full routing report as pretty JSON to this path.
        #[arg(long)]
        json: Option<PathBuf>,
        /// If set, write the routing Markdown report to this path. If
        /// unset, the Markdown report is printed to stdout.
        #[arg(long)]
        md: Option<PathBuf>,
        #[command(flatten)]
        output_tree: OutputTreeArgs,
    },
    /// Profile tensor payload statistics for offline analysis.
    Stats {
        #[command(flatten)]
        scan: CheckpointScanArgs,
        #[command(flatten)]
        family: ModelFamilyArg,
        #[command(flatten)]
        sample: SampleValuesArg,
        /// If set, write the stats profile as pretty JSON to this path.
        #[arg(long)]
        json: Option<PathBuf>,
        /// If set, write the stats Markdown report to this path. If unset,
        /// the Markdown report is printed to stdout.
        #[arg(long)]
        md: Option<PathBuf>,
        #[command(flatten)]
        output_tree: OutputTreeArgs,
    },
    /// Rank likely SAAQ experiment targets without applying SAAQ itself.
    SaaqReadiness {
        #[command(flatten)]
        scan: CheckpointScanArgs,
        #[command(flatten)]
        family: ModelFamilyArg,
        #[command(flatten)]
        sample: SampleValuesArg,
        /// If set, write the SAAQ-readiness report as pretty JSON.
        #[arg(long)]
        json: Option<PathBuf>,
        /// If set, write the SAAQ-readiness Markdown report. If unset, the
        /// Markdown report is printed to stdout.
        #[arg(long)]
        md: Option<PathBuf>,
        /// If set, write the machine-readable candidate manifest as pretty JSON.
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[command(flatten)]
        output_tree: OutputTreeArgs,
    },
    /// Emit a planning-side Grok-1 pilot selection plan.
    PilotPlan {
        #[command(flatten)]
        scan: CheckpointScanArgs,
        #[command(flatten)]
        family: PlanningFamilyArg,
        /// If set, write the pilot-selection plan JSON to this path.
        #[arg(long)]
        json: Option<PathBuf>,
        /// If set, write the pilot-selection Markdown report to this path.
        #[arg(long)]
        md: Option<PathBuf>,
        #[command(flatten)]
        output_tree: OutputTreeArgs,
    },
    /// Emit a planning-side Grok-1 route-preservation gate report.
    RoutePreservation {
        #[command(flatten)]
        scan: CheckpointScanArgs,
        #[command(flatten)]
        family: PlanningFamilyArg,
        /// If set, write the route-preservation report JSON to this path.
        #[arg(long)]
        json: Option<PathBuf>,
        /// If set, write the route-preservation Markdown report to this path.
        #[arg(long)]
        md: Option<PathBuf>,
        #[command(flatten)]
        output_tree: OutputTreeArgs,
    },
    /// Emit deterministic Grok-1 conversion and quant-planning artifacts.
    QuantPlan {
        #[command(flatten)]
        scan: CheckpointScanArgs,
        #[command(flatten)]
        family: ModelFamilyArg,
        /// Maximum sampled values per tensor when computing readiness.
        #[arg(long, default_value_t = 65_536)]
        sample_values: usize,
        /// If set, write the deterministic quant-plan JSON to this path.
        #[arg(long)]
        json: Option<PathBuf>,
        /// If set, write the quant-plan Markdown summary to this path. If
        /// unset, the Markdown report is printed to stdout.
        #[arg(long)]
        md: Option<PathBuf>,
        /// If set, write the conversion-ready manifest JSON to this path.
        #[arg(long)]
        conversion_manifest: Option<PathBuf>,
        /// If set, write the conversion-ready manifest Markdown summary to
        /// this path.
        #[arg(long)]
        conversion_manifest_md: Option<PathBuf>,
        #[command(flatten)]
        output_tree: OutputTreeArgs,
    },
}

struct CommandFields {
    limit: Option<usize>,
    prefix: Option<String>,
    family: Option<String>,
    sample_values: Option<usize>,
}

impl Command {
    fn name(&self) -> &'static str {
        match self {
            Command::Dissect { .. } => "dissect",
            Command::Inventory { .. } => "inventory",
            Command::Experts { .. } => "experts",
            Command::RoutingReport { .. } => "routing-report",
            Command::Stats { .. } => "stats",
            Command::SaaqReadiness { .. } => "saaq-readiness",
            Command::PilotPlan { .. } => "pilot-plan",
            Command::RoutePreservation { .. } => "route-preservation",
            Command::QuantPlan { .. } => "quant-plan",
        }
    }

    fn fields(&self) -> CommandFields {
        match self {
            Command::Dissect { limit, prefix, .. } => CommandFields {
                limit: *limit,
                prefix: Some(prefix.clone()),
                family: None,
                sample_values: None,
            },
            Command::Inventory { scan, family, .. }
            | Command::Experts { scan, family, .. }
            | Command::RoutingReport { scan, family, .. } => CommandFields {
                limit: scan.limit,
                prefix: Some(scan.prefix.clone()),
                family: Some(family.family.clone()),
                sample_values: None,
            },
            Command::Stats {
                scan,
                family,
                sample,
                ..
            }
            | Command::SaaqReadiness {
                scan,
                family,
                sample,
                ..
            } => CommandFields {
                limit: scan.limit,
                prefix: Some(scan.prefix.clone()),
                family: Some(family.family.clone()),
                sample_values: Some(sample.sample_values),
            },
            Command::PilotPlan { scan, family, .. }
            | Command::RoutePreservation { scan, family, .. } => CommandFields {
                limit: scan.limit,
                prefix: Some(scan.prefix.clone()),
                family: Some(family.family.clone()),
                sample_values: None,
            },
            Command::QuantPlan {
                scan,
                family,
                sample_values,
                ..
            } => CommandFields {
                limit: scan.limit,
                prefix: Some(scan.prefix.clone()),
                family: Some(family.family.clone()),
                sample_values: Some(*sample_values),
            },
        }
    }
}

fn main() -> Result<()> {
    // Keep guard alive for the whole process so panics/events flush on drop.
    let _sentry_guard = observability::init_sentry();
    observability::init_tracing();
    let cli = Cli::parse();
    let command = cli.command.name();
    let fields = cli.command.fields();
    let run_id = observability::run_id();
    let git_sha = observability::git_sha();

    let span = tracing::info_span!(
        "command",
        repo = "xai-dissect",
        command,
        run_id,
        git_sha,
        limit = ?fields.limit,
        prefix = fields.prefix.as_deref().unwrap_or(""),
        family = fields.family.as_deref().unwrap_or(""),
        sample_values = ?fields.sample_values,
    );
    let _enter = span.enter();

    tracing::info!(event = "command_start", "command_start");
    sentry::configure_scope(|scope| {
        scope.set_tag("command", command);
    });

    let started = Instant::now();
    let result = match cli.command {
        Command::Dissect {
            path,
            limit,
            prefix,
        } => run_dissect(&path, limit, &prefix),
        Command::Inventory {
            scan,
            family,
            json,
            md,
            output_tree,
        } => cli::run_inventory(
            &scan,
            &family.family,
            json.as_deref(),
            md.as_deref(),
            &output_tree,
        ),
        Command::Experts {
            scan,
            family,
            json,
            md,
            output_tree,
        } => cli::run_experts(
            &scan,
            &family.family,
            json.as_deref(),
            md.as_deref(),
            &output_tree,
        ),
        Command::RoutingReport {
            scan,
            family,
            json,
            md,
            output_tree,
        } => cli::run_routing_report(
            &scan,
            &family.family,
            json.as_deref(),
            md.as_deref(),
            &output_tree,
        ),
        Command::Stats {
            scan,
            family,
            sample,
            json,
            md,
            output_tree,
        } => cli::run_stats(
            &scan,
            &family.family,
            sample.sample_values,
            json.as_deref(),
            md.as_deref(),
            &output_tree,
        ),
        Command::SaaqReadiness {
            scan,
            family,
            sample,
            json,
            md,
            manifest,
            output_tree,
        } => cli::run_saaq_readiness(
            &scan,
            &family.family,
            sample.sample_values,
            json.as_deref(),
            md.as_deref(),
            manifest.as_deref(),
            &output_tree,
        ),
        Command::PilotPlan {
            scan,
            family,
            json,
            md,
            output_tree,
        } => cli::run_pilot_plan(
            &scan,
            &family.family,
            json.as_deref(),
            md.as_deref(),
            &output_tree,
        ),
        Command::RoutePreservation {
            scan,
            family,
            json,
            md,
            output_tree,
        } => cli::run_route_preservation(
            &scan,
            &family.family,
            json.as_deref(),
            md.as_deref(),
            &output_tree,
        ),
        Command::QuantPlan {
            scan,
            family,
            sample_values,
            json,
            md,
            conversion_manifest,
            conversion_manifest_md,
            output_tree,
        } => cli::run_quant_plan(
            &scan,
            &family.family,
            sample_values,
            json.as_deref(),
            md.as_deref(),
            conversion_manifest.as_deref(),
            conversion_manifest_md.as_deref(),
            &output_tree,
        ),
    };

    let latency_ms = started.elapsed().as_millis() as u64;
    let error_category = observability::error_category(result.as_ref().err());
    tracing::info!(
        event = "command_finish",
        latency_ms,
        success = result.is_ok(),
        error_category,
        "command_finish"
    );

    if let Err(ref error) = result {
        observability::capture_error(error);
    }

    result
}

// --- `dissect` -------------------------------------------------------------

fn run_dissect(path: &std::path::Path, limit: Option<usize>, prefix: &str) -> Result<()> {
    let md = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if !md.is_dir() {
        bail!("{} is not a directory", path.display());
    }

    let mut shards = Vec::new();
    for entry in std::fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
        let entry = entry.with_context(|| format!("read entry in {}", path.display()))?;
        let p = entry.path();
        if p.is_file()
            && p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(prefix))
                .unwrap_or(false)
        {
            shards.push(p);
        }
    }
    shards.sort();
    if shards.is_empty() {
        bail!(
            "no shards found under {} with prefix '{}'",
            path.display(),
            prefix
        );
    }
    if let Some(n) = limit {
        shards.truncate(n);
    }

    for shard in &shards {
        match parser::dissect_shard(shard) {
            Ok(entries) => {
                println!("\n{}", shard.display());
                if entries.is_empty() {
                    println!("  (no tensors found)");
                    continue;
                }
                let mut table = Table::new();
                table
                    .load_preset(UTF8_FULL)
                    .set_content_arrangement(ContentArrangement::Dynamic)
                    .set_header(vec!["Idx", "Role", "Dtype", "Shape", "Offset", "Nbytes"]);
                for (i, e) in entries.iter().enumerate() {
                    table.add_row(vec![
                        Cell::new(i),
                        Cell::new(e.role.label()),
                        Cell::new(e.dtype.label()),
                        Cell::new(e.shape.render()),
                        Cell::new(format!("{:#x}", e.offset)),
                        Cell::new(e.nbytes),
                    ]);
                }
                println!("{table}");
            }
            Err(e) => eprintln!("warn: {}: {:#}", shard.display(), e),
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use crate::cli::{
        CheckpointScanArgs, ModelFamilyArg, OutputTreeArgs, PlanningFamilyArg, SampleValuesArg,
        validate_complete_inventory_scope,
    };

    use super::{Command, run_dissect};
    use crate::test_support::write_hex_checkpoint;

    fn dummy_scan() -> CheckpointScanArgs {
        CheckpointScanArgs {
            path: PathBuf::from("/tmp/ckpt"),
            prefix: "tensor".into(),
            limit: None,
        }
    }

    fn dummy_tree() -> OutputTreeArgs {
        OutputTreeArgs {
            output_root: None,
            checkpoint_slug: None,
        }
    }

    fn command_fixtures() -> (
        CheckpointScanArgs,
        ModelFamilyArg,
        PlanningFamilyArg,
        SampleValuesArg,
        OutputTreeArgs,
    ) {
        (
            dummy_scan(),
            ModelFamilyArg {
                family: "grok-1".into(),
            },
            PlanningFamilyArg {
                family: "grok-1".into(),
            },
            SampleValuesArg { sample_values: 64 },
            dummy_tree(),
        )
    }

    fn inspect_commands(
        scan: &CheckpointScanArgs,
        family: &ModelFamilyArg,
        tree: &OutputTreeArgs,
    ) -> [Command; 4] {
        [
            Command::Dissect {
                path: PathBuf::from("/tmp/ckpt"),
                limit: Some(1),
                prefix: "tensor".into(),
            },
            Command::Inventory {
                scan: scan.clone(),
                family: family.clone(),
                json: None,
                md: None,
                output_tree: tree.clone(),
            },
            Command::Experts {
                scan: scan.clone(),
                family: family.clone(),
                json: None,
                md: None,
                output_tree: tree.clone(),
            },
            Command::RoutingReport {
                scan: scan.clone(),
                family: family.clone(),
                json: None,
                md: None,
                output_tree: tree.clone(),
            },
        ]
    }

    fn profile_commands(
        scan: &CheckpointScanArgs,
        family: &ModelFamilyArg,
        sample: SampleValuesArg,
        tree: &OutputTreeArgs,
    ) -> [Command; 2] {
        [
            Command::Stats {
                scan: scan.clone(),
                family: family.clone(),
                sample: sample.clone(),
                json: None,
                md: None,
                output_tree: tree.clone(),
            },
            Command::SaaqReadiness {
                scan: scan.clone(),
                family: family.clone(),
                sample,
                json: None,
                md: None,
                manifest: None,
                output_tree: tree.clone(),
            },
        ]
    }

    fn planning_commands(
        scan: CheckpointScanArgs,
        family: ModelFamilyArg,
        planning: PlanningFamilyArg,
        tree: OutputTreeArgs,
    ) -> [Command; 3] {
        [
            Command::PilotPlan {
                scan: scan.clone(),
                family: planning.clone(),
                json: None,
                md: None,
                output_tree: tree.clone(),
            },
            Command::RoutePreservation {
                scan: scan.clone(),
                family: planning,
                json: None,
                md: None,
                output_tree: tree.clone(),
            },
            Command::QuantPlan {
                scan,
                family,
                sample_values: 64,
                json: None,
                md: None,
                conversion_manifest: None,
                conversion_manifest_md: None,
                output_tree: tree,
            },
        ]
    }

    fn assert_every_command_name_and_prefix(variants: &[Command]) {
        let names: Vec<_> = variants.iter().map(Command::name).collect();
        assert_eq!(
            names,
            [
                "dissect",
                "inventory",
                "experts",
                "routing-report",
                "stats",
                "saaq-readiness",
                "pilot-plan",
                "route-preservation",
                "quant-plan",
            ]
        );
        for command in variants {
            assert_eq!(command.fields().prefix.as_deref(), Some("tensor"));
        }
    }

    #[test]
    fn complete_inventory_scope_accepts_default_prefix_without_limit() {
        validate_complete_inventory_scope("quant-plan", "tensor", None).expect("default scope");
    }

    #[test]
    fn quant_plan_rejects_non_default_prefix() {
        let err =
            validate_complete_inventory_scope("quant-plan", "tensor-shard", None).unwrap_err();
        assert!(format!("{err:#}").contains("--prefix tensor-shard"));
    }

    #[test]
    fn quant_plan_rejects_limited_inventory() {
        let err = validate_complete_inventory_scope("quant-plan", "tensor", Some(4)).unwrap_err();
        assert!(format!("{err:#}").contains("--limit 4"));
    }

    #[test]
    fn command_name_and_fields_cover_every_variant() {
        let (scan, family, planning, sample, tree) = command_fixtures();
        let inspect = inspect_commands(&scan, &family, &tree);
        let profile = profile_commands(&scan, &family, sample, &tree);
        let planning = planning_commands(scan, family, planning, tree);
        let mut variants = Vec::with_capacity(9);
        variants.extend(inspect);
        variants.extend(profile);
        variants.extend(planning);
        assert_every_command_name_and_prefix(&variants);
    }

    /// The shared single-shard checkpoint plus a deliberately unreadable
    /// second shard, so `run_dissect` exercises its warn-and-continue path.
    fn write_dissect_hex_fixture() -> tempfile::TempDir {
        let root = write_hex_checkpoint("dissect");
        fs::write(root.path().join("tensor0001.pkl"), b"not-a-pickle")
            .expect("write garbage shard");
        root
    }

    #[test]
    fn dissect_hex_fixture_prints_tensor_table() {
        let root = write_dissect_hex_fixture();
        run_dissect(root.path(), Some(1), "tensor").expect("dissect hex fixture");
        run_dissect(root.path(), None, "tensor").expect("dissect includes unreadable shard warn");
        let empty = run_dissect(root.path(), Some(0), "missing").unwrap_err();
        assert!(format!("{empty:#}").contains("no shards found"));
        let not_dir = run_dissect(&root.path().join("tensor0000.pkl"), None, "tensor").unwrap_err();
        assert!(format!("{not_dir:#}").contains("is not a directory"));
    }
}
