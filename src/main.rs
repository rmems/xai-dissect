// SPDX-License-Identifier: GPL-3.0-only
//
//! xai-dissect CLI. Thin wrapper over the `xai_dissect` library crate.
//!
//! ## Command map
//!
//! | Command | Driven by | Produces |
//! |---------|----------|---------|
//! | `dissect` | `parser::parse_single_shard` | Raw tensor table (no classification) |
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
use clap::{Args, Parser, Subcommand};
use comfy_table::{Cell, ContentArrangement, Table, presets::UTF8_FULL};

mod observability;

use xai_dissect::experts::build_expert_atlas;
use xai_dissect::exports;
use xai_dissect::inventory::{InventoryConfig, build_inventory};
use xai_dissect::parser;
use xai_dissect::planning::{
    build_grok1_pilot_selection_plan, build_grok1_planning_artifacts,
    build_grok1_route_preservation_report,
};
use xai_dissect::report;
use xai_dissect::routing::build_routing_report;
use xai_dissect::schema::{
    ExpertAtlas, ModelInventory, PilotSelectionPlan, RoutePreservationReport, RoutingReport,
    SaaqReadinessReport, StatsProfileReport, TensorInfo,
};
use xai_dissect::stats::{StatsConfig, build_saaq_readiness_report, build_stats_report};

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

#[derive(Args, Debug, Clone, Default)]
struct OutputTreeArgs {
    /// If set, also write artifacts into
    /// `<root>/{reports,exports,manifests}/<checkpoint-slug>/...`.
    #[arg(long)]
    output_root: Option<PathBuf>,
    /// Optional override for the checkpoint slug used under the unified
    /// output tree. If unset, the slug is inferred from the checkpoint path.
    #[arg(long, requires = "output_root")]
    checkpoint_slug: Option<String>,
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
        /// Checkpoint directory (e.g. `/path/to/grok-1/ckpt-0`).
        path: PathBuf,
        /// Filename prefix filter.
        #[arg(long, default_value = "tensor")]
        prefix: String,
        /// Only process the first N shards (sorted by filename).
        #[arg(long)]
        limit: Option<usize>,
        /// Model family tag written into the export header. Only `grok-1`
        /// is officially supported today.
        #[arg(long, default_value = "grok-1")]
        family: String,
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
        /// Checkpoint directory (e.g. `/path/to/grok-1/ckpt-0`).
        path: PathBuf,
        /// Filename prefix filter.
        #[arg(long, default_value = "tensor")]
        prefix: String,
        /// Only process the first N shards (sorted by filename).
        #[arg(long)]
        limit: Option<usize>,
        /// Model family tag written into the export header. Only `grok-1`
        /// is officially supported today.
        #[arg(long, default_value = "grok-1")]
        family: String,
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
        /// Checkpoint directory (e.g. `/path/to/grok-1/ckpt-0`).
        path: PathBuf,
        /// Filename prefix filter.
        #[arg(long, default_value = "tensor")]
        prefix: String,
        /// Only process the first N shards (sorted by filename).
        #[arg(long)]
        limit: Option<usize>,
        /// Model family tag written into the export header. Only `grok-1`
        /// is officially supported today.
        #[arg(long, default_value = "grok-1")]
        family: String,
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
        /// Checkpoint directory (e.g. `/path/to/grok-1/ckpt-0`).
        path: PathBuf,
        /// Filename prefix filter.
        #[arg(long, default_value = "tensor")]
        prefix: String,
        /// Only process the first N shards (sorted by filename).
        #[arg(long)]
        limit: Option<usize>,
        /// Model family tag written into the export header. Only `grok-1`
        /// is officially supported today.
        #[arg(long, default_value = "grok-1")]
        family: String,
        /// Maximum sampled values per tensor.
        #[arg(long, default_value_t = 65_536)]
        sample_values: usize,
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
        /// Checkpoint directory (e.g. `/path/to/grok-1/ckpt-0`).
        path: PathBuf,
        /// Filename prefix filter.
        #[arg(long, default_value = "tensor")]
        prefix: String,
        /// Only process the first N shards (sorted by filename).
        #[arg(long)]
        limit: Option<usize>,
        /// Model family tag written into the export header. Only `grok-1`
        /// is officially supported today.
        #[arg(long, default_value = "grok-1")]
        family: String,
        /// Maximum sampled values per tensor.
        #[arg(long, default_value_t = 65_536)]
        sample_values: usize,
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
        /// Checkpoint directory (e.g. `/path/to/grok-1/ckpt-0`).
        path: PathBuf,
        /// Filename prefix filter.
        #[arg(long, default_value = "tensor")]
        prefix: String,
        /// Only process the first N shards (sorted by filename).
        #[arg(long)]
        limit: Option<usize>,
        /// Model family tag written into the export header.
        #[arg(long, default_value = "grok-1")]
        family: String,
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
        /// Checkpoint directory (e.g. `/path/to/grok-1/ckpt-0`).
        path: PathBuf,
        /// Filename prefix filter.
        #[arg(long, default_value = "tensor")]
        prefix: String,
        /// Only process the first N shards (sorted by filename).
        #[arg(long)]
        limit: Option<usize>,
        /// Model family tag written into the export header.
        #[arg(long, default_value = "grok-1")]
        family: String,
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
        /// Checkpoint directory (e.g. `/path/to/grok-1/ckpt-0`).
        path: PathBuf,
        /// Filename prefix filter.
        #[arg(long, default_value = "tensor")]
        prefix: String,
        /// Only process the first N shards (sorted by filename).
        #[arg(long)]
        limit: Option<usize>,
        /// Model family tag written into the export header. Only `grok-1`
        /// is officially supported today.
        #[arg(long, default_value = "grok-1")]
        family: String,
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
            Command::Inventory {
                limit,
                prefix,
                family,
                ..
            }
            | Command::Experts {
                limit,
                prefix,
                family,
                ..
            }
            | Command::RoutingReport {
                limit,
                prefix,
                family,
                ..
            } => CommandFields {
                limit: *limit,
                prefix: Some(prefix.clone()),
                family: Some(family.clone()),
                sample_values: None,
            },
            Command::Stats {
                limit,
                prefix,
                family,
                sample_values,
                ..
            }
            | Command::SaaqReadiness {
                limit,
                prefix,
                family,
                sample_values,
                ..
            }
            | Command::QuantPlan {
                limit,
                prefix,
                family,
                sample_values,
                ..
            } => CommandFields {
                limit: *limit,
                prefix: Some(prefix.clone()),
                family: Some(family.clone()),
                sample_values: Some(*sample_values),
            },
            Command::PilotPlan {
                limit,
                prefix,
                family,
                ..
            }
            | Command::RoutePreservation {
                limit,
                prefix,
                family,
                ..
            } => CommandFields {
                limit: *limit,
                prefix: Some(prefix.clone()),
                family: Some(family.clone()),
                sample_values: None,
            },
        }
    }
}

fn main() -> Result<()> {
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

    let started = Instant::now();
    let result = match cli.command {
        Command::Dissect {
            path,
            limit,
            prefix,
        } => run_dissect(&path, limit, &prefix),
        Command::Inventory {
            path,
            prefix,
            limit,
            family,
            json,
            md,
            output_tree,
        } => run_inventory(
            &path,
            &prefix,
            limit,
            &family,
            json.as_deref(),
            md.as_deref(),
            &output_tree,
        ),
        Command::Experts {
            path,
            prefix,
            limit,
            family,
            json,
            md,
            output_tree,
        } => run_experts(
            &path,
            &prefix,
            limit,
            &family,
            json.as_deref(),
            md.as_deref(),
            &output_tree,
        ),
        Command::RoutingReport {
            path,
            prefix,
            limit,
            family,
            json,
            md,
            output_tree,
        } => run_routing_report(
            &path,
            &prefix,
            limit,
            &family,
            json.as_deref(),
            md.as_deref(),
            &output_tree,
        ),
        Command::Stats {
            path,
            prefix,
            limit,
            family,
            sample_values,
            json,
            md,
            output_tree,
        } => run_stats(
            &path,
            &prefix,
            limit,
            &family,
            sample_values,
            json.as_deref(),
            md.as_deref(),
            &output_tree,
        ),
        Command::SaaqReadiness {
            path,
            prefix,
            limit,
            family,
            sample_values,
            json,
            md,
            manifest,
            output_tree,
        } => run_saaq_readiness(
            &path,
            &prefix,
            limit,
            &family,
            sample_values,
            json.as_deref(),
            md.as_deref(),
            manifest.as_deref(),
            &output_tree,
        ),
        Command::PilotPlan {
            path,
            prefix,
            limit,
            family,
            json,
            md,
            output_tree,
        } => run_pilot_plan(
            &path,
            &prefix,
            limit,
            &family,
            json.as_deref(),
            md.as_deref(),
            &output_tree,
        ),
        Command::RoutePreservation {
            path,
            prefix,
            limit,
            family,
            json,
            md,
            output_tree,
        } => run_route_preservation(
            &path,
            &prefix,
            limit,
            &family,
            json.as_deref(),
            md.as_deref(),
            &output_tree,
        ),
        Command::QuantPlan {
            path,
            prefix,
            limit,
            family,
            sample_values,
            json,
            md,
            conversion_manifest,
            conversion_manifest_md,
            output_tree,
        } => run_quant_plan(
            &path,
            &prefix,
            limit,
            &family,
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

// --- `inventory` -----------------------------------------------------------

fn run_inventory(
    path: &std::path::Path,
    prefix: &str,
    limit: Option<usize>,
    family: &str,
    json_out: Option<&std::path::Path>,
    md_out: Option<&std::path::Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    let cfg = InventoryConfig {
        prefix: prefix.to_string(),
        limit,
        model_family: family.to_string(),
    };
    let inv = build_inventory(path, &cfg)?;

    // Always print a compact console summary.
    print_console_summary(&inv);

    if let Some(p) = json_out {
        report::write_json(&inv, p)?;
        eprintln!("wrote JSON inventory -> {}", p.display());
    }
    if let Some(p) = md_out {
        report::write_markdown(&inv, p)?;
        eprintln!("wrote Markdown report -> {}", p.display());
    } else {
        // If no Markdown file was requested, print the Markdown report to
        // stdout so `cargo run -- inventory <path>` is useful on its own.
        println!();
        println!("{}", report::render_markdown(&inv));
    }
    if let Some(root) = output_tree.output_root.as_deref() {
        let bundle =
            exports::write_inventory_bundle(&inv, root, output_tree.checkpoint_slug.as_deref())?;
        print_output_bundle("inventory bundle", root, &bundle);
    }

    Ok(())
}

// --- `experts` -------------------------------------------------------------

fn run_experts(
    path: &std::path::Path,
    prefix: &str,
    limit: Option<usize>,
    family: &str,
    json_out: Option<&std::path::Path>,
    md_out: Option<&std::path::Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    let cfg = InventoryConfig {
        prefix: prefix.to_string(),
        limit,
        model_family: family.to_string(),
    };
    let inv = build_inventory(path, &cfg)?;
    let atlas = build_expert_atlas(&inv);

    print_expert_console_summary(&atlas);

    if let Some(p) = json_out {
        report::write_expert_json(&atlas, p)?;
        eprintln!("wrote JSON expert atlas -> {}", p.display());
    }
    if let Some(p) = md_out {
        report::write_expert_markdown(&atlas, p)?;
        eprintln!("wrote Markdown expert atlas -> {}", p.display());
    } else {
        println!();
        println!("{}", report::render_expert_markdown(&atlas));
    }
    if let Some(root) = output_tree.output_root.as_deref() {
        let bundle =
            exports::write_expert_bundle(&atlas, root, output_tree.checkpoint_slug.as_deref())?;
        print_output_bundle("expert bundle", root, &bundle);
    }

    Ok(())
}

// --- `routing-report` -----------------------------------------------------

fn run_routing_report(
    path: &std::path::Path,
    prefix: &str,
    limit: Option<usize>,
    family: &str,
    json_out: Option<&std::path::Path>,
    md_out: Option<&std::path::Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    let cfg = InventoryConfig {
        prefix: prefix.to_string(),
        limit,
        model_family: family.to_string(),
    };
    let inv = build_inventory(path, &cfg)?;
    let report_doc = build_routing_report(&inv);

    print_routing_console_summary(&report_doc);

    if let Some(p) = json_out {
        report::write_routing_json(&report_doc, p)?;
        eprintln!("wrote JSON routing report -> {}", p.display());
    }
    if let Some(p) = md_out {
        report::write_routing_markdown(&report_doc, p)?;
        eprintln!("wrote Markdown routing report -> {}", p.display());
    } else {
        println!();
        println!("{}", report::render_routing_markdown(&report_doc));
    }
    if let Some(root) = output_tree.output_root.as_deref() {
        let bundle = exports::write_routing_bundle(
            &report_doc,
            root,
            output_tree.checkpoint_slug.as_deref(),
        )?;
        print_output_bundle("routing bundle", root, &bundle);
    }

    Ok(())
}

// --- `stats` --------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_stats(
    path: &std::path::Path,
    prefix: &str,
    limit: Option<usize>,
    family: &str,
    sample_values: usize,
    json_out: Option<&std::path::Path>,
    md_out: Option<&std::path::Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    let cfg = InventoryConfig {
        prefix: prefix.to_string(),
        limit,
        model_family: family.to_string(),
    };
    let inv = build_inventory(path, &cfg)?;
    let stats_cfg = StatsConfig {
        max_sample_values: sample_values,
        ..Default::default()
    };
    let report_doc = build_stats_report(&inv, &stats_cfg)?;

    print_stats_console_summary(&report_doc);

    if let Some(p) = json_out {
        report::write_stats_json(&report_doc, p)?;
        eprintln!("wrote JSON stats report -> {}", p.display());
    }
    if let Some(p) = md_out {
        report::write_stats_markdown(&report_doc, p)?;
        eprintln!("wrote Markdown stats report -> {}", p.display());
    } else {
        println!();
        println!("{}", report::render_stats_markdown(&report_doc));
    }
    if let Some(root) = output_tree.output_root.as_deref() {
        let bundle =
            exports::write_stats_bundle(&report_doc, root, output_tree.checkpoint_slug.as_deref())?;
        print_output_bundle("stats bundle", root, &bundle);
    }

    Ok(())
}

// --- `saaq-readiness` -----------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_saaq_readiness(
    path: &std::path::Path,
    prefix: &str,
    limit: Option<usize>,
    family: &str,
    sample_values: usize,
    json_out: Option<&std::path::Path>,
    md_out: Option<&std::path::Path>,
    manifest_out: Option<&std::path::Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    let cfg = InventoryConfig {
        prefix: prefix.to_string(),
        limit,
        model_family: family.to_string(),
    };
    let inv = build_inventory(path, &cfg)?;
    let stats_cfg = StatsConfig {
        max_sample_values: sample_values,
        ..Default::default()
    };
    let stats = build_stats_report(&inv, &stats_cfg)?;
    let readiness = build_saaq_readiness_report(&inv, &stats);

    print_saaq_console_summary(&readiness);

    if let Some(p) = json_out {
        report::write_saaq_readiness_json(&readiness, p)?;
        eprintln!("wrote JSON SAAQ-readiness report -> {}", p.display());
    }
    if let Some(p) = md_out {
        report::write_saaq_readiness_markdown(&readiness, p)?;
        eprintln!("wrote Markdown SAAQ-readiness report -> {}", p.display());
    } else {
        println!();
        println!("{}", report::render_saaq_readiness_markdown(&readiness));
    }
    if let Some(p) = manifest_out {
        report::write_candidate_manifest_json(&readiness.manifest, p)?;
        eprintln!("wrote candidate manifest -> {}", p.display());
    }
    if let Some(root) = output_tree.output_root.as_deref() {
        let bundle =
            exports::write_saaq_bundle(&readiness, root, output_tree.checkpoint_slug.as_deref())?;
        print_output_bundle("saaq bundle", root, &bundle);
    }

    Ok(())
}

// --- `pilot-plan` ---------------------------------------------------------

fn run_pilot_plan(
    path: &std::path::Path,
    prefix: &str,
    limit: Option<usize>,
    family: &str,
    json_out: Option<&std::path::Path>,
    md_out: Option<&std::path::Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    validate_complete_inventory_scope("pilot-plan", prefix, limit)?;
    let cfg = InventoryConfig {
        prefix: prefix.to_string(),
        limit,
        model_family: family.to_string(),
    };
    let inv = build_inventory(path, &cfg)?;
    let plan = build_grok1_pilot_selection_plan(&inv)?;
    print_pilot_plan_console_summary(&plan);
    if let Some(p) = json_out {
        report::write_pilot_selection_plan_json(&plan, p)?;
        eprintln!("wrote JSON pilot selection plan -> {}", p.display());
    }
    if let Some(p) = md_out {
        report::write_pilot_selection_plan_markdown(&plan, p)?;
        eprintln!("wrote Markdown pilot selection plan -> {}", p.display());
    } else {
        println!();
        println!("{}", report::render_pilot_selection_plan_markdown(&plan));
    }
    if let Some(root) = output_tree.output_root.as_deref() {
        let bundle =
            exports::write_pilot_plan_bundle(&plan, root, output_tree.checkpoint_slug.as_deref())?;
        eprintln!(
            "wrote pilot-plan bundle -> {}/{{reports,manifests}}/{}/...",
            root.display(),
            bundle.checkpoint_slug
        );
    }
    Ok(())
}

// --- `route-preservation` --------------------------------------------------

fn run_route_preservation(
    path: &std::path::Path,
    prefix: &str,
    limit: Option<usize>,
    family: &str,
    json_out: Option<&std::path::Path>,
    md_out: Option<&std::path::Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    validate_complete_inventory_scope("route-preservation", prefix, limit)?;
    let cfg = InventoryConfig {
        prefix: prefix.to_string(),
        limit,
        model_family: family.to_string(),
    };
    let inv = build_inventory(path, &cfg)?;
    let report_doc = build_grok1_route_preservation_report(&inv)?;
    print_route_preservation_console_summary(&report_doc);
    if let Some(p) = json_out {
        report::write_route_preservation_report_json(&report_doc, p)?;
        eprintln!("wrote JSON route-preservation report -> {}", p.display());
    }
    if let Some(p) = md_out {
        report::write_route_preservation_markdown(&report_doc, p)?;
        eprintln!(
            "wrote Markdown route-preservation report -> {}",
            p.display()
        );
    } else {
        println!();
        println!(
            "{}",
            report::render_route_preservation_markdown(&report_doc)
        );
    }
    if let Some(root) = output_tree.output_root.as_deref() {
        let bundle = exports::write_route_preservation_bundle(
            &report_doc,
            root,
            output_tree.checkpoint_slug.as_deref(),
        )?;
        eprintln!(
            "wrote route-preservation bundle -> {}/{{reports,manifests}}/{}/...",
            root.display(),
            bundle.checkpoint_slug
        );
    }
    Ok(())
}

// --- `quant-plan` ---------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_quant_plan(
    path: &std::path::Path,
    prefix: &str,
    limit: Option<usize>,
    family: &str,
    sample_values: usize,
    json_out: Option<&std::path::Path>,
    md_out: Option<&std::path::Path>,
    conversion_manifest_out: Option<&std::path::Path>,
    conversion_manifest_md_out: Option<&std::path::Path>,
    output_tree: &OutputTreeArgs,
) -> Result<()> {
    validate_complete_inventory_scope("quant-plan", prefix, limit)?;
    let cfg = InventoryConfig {
        prefix: prefix.to_string(),
        limit,
        model_family: family.to_string(),
    };
    let inv = build_inventory(path, &cfg)?;
    let atlas = build_expert_atlas(&inv);
    let routing = build_routing_report(&inv);
    let stats_cfg = StatsConfig {
        max_sample_values: sample_values,
        ..Default::default()
    };
    let stats = build_stats_report(&inv, &stats_cfg)?;
    let readiness = build_saaq_readiness_report(&inv, &stats);
    let (conversion_manifest, quant_plan) =
        build_grok1_planning_artifacts(&inv, &atlas, &routing, &readiness)?;

    print_quant_plan_console_summary(&quant_plan, &conversion_manifest);

    if let Some(p) = conversion_manifest_out {
        report::write_conversion_manifest_json(&conversion_manifest, p)?;
        eprintln!("wrote conversion manifest -> {}", p.display());
    }
    if let Some(p) = conversion_manifest_md_out {
        report::write_conversion_manifest_markdown(&conversion_manifest, p)?;
        eprintln!("wrote conversion manifest Markdown -> {}", p.display());
    }
    if let Some(p) = json_out {
        report::write_quant_plan_json(&quant_plan, p)?;
        eprintln!("wrote JSON quant plan -> {}", p.display());
    }
    if let Some(p) = md_out {
        report::write_quant_plan_markdown(&quant_plan, p)?;
        eprintln!("wrote Markdown quant plan -> {}", p.display());
    } else {
        println!();
        println!("{}", report::render_quant_plan_markdown(&quant_plan));
    }
    if let Some(root) = output_tree.output_root.as_deref() {
        let bundle = exports::write_quant_plan_bundle(
            &conversion_manifest,
            &quant_plan,
            root,
            output_tree.checkpoint_slug.as_deref(),
        )?;
        print_output_bundle("quant-plan bundle", root, &bundle);
    }

    Ok(())
}

fn validate_complete_inventory_scope(
    command: &str,
    prefix: &str,
    limit: Option<usize>,
) -> Result<()> {
    if prefix != "tensor" {
        bail!(
            "{command} requires a complete Grok-1 inventory; `--prefix {}` is not supported",
            prefix
        );
    }
    if let Some(limit) = limit {
        bail!("{command} requires a complete Grok-1 inventory; `--limit {limit}` is not supported");
    }
    Ok(())
}

fn print_quant_plan_console_summary(
    quant_plan: &xai_dissect::schema::QuantPlan,
    conversion_manifest: &xai_dissect::schema::ConversionManifest,
) {
    eprintln!(
        "checkpoint: {}  baseline: {}  tensors: {}",
        quant_plan.checkpoint_path.display(),
        quant_plan.baseline,
        conversion_manifest.tensors.len(),
    );
    eprintln!(
        "keep_fp32: {}  pilot_quantize: {}  defer: {}",
        quant_plan.keep_fp32.len(),
        quant_plan.pilot_quantize.len(),
        quant_plan.defer.len(),
    );
    if !conversion_manifest.warnings.is_empty() {
        eprintln!(
            "warn: conversion manifest emitted {} warning categories",
            conversion_manifest.warnings.len()
        );
    }
}

fn print_pilot_plan_console_summary(plan: &PilotSelectionPlan) {
    eprintln!(
        "checkpoint: {}  baseline: {}  selected_blocks: {}  modes: {}",
        plan.checkpoint_path.display(),
        plan.baseline,
        plan.selected_blocks.len(),
        plan.modes.len(),
    );
}

fn print_route_preservation_console_summary(report_doc: &RoutePreservationReport) {
    eprintln!(
        "model_family: {}  baseline: {}  router_metrics: {}  block_metrics: {}",
        report_doc.model_family,
        report_doc.baseline,
        report_doc.router_metrics.len(),
        report_doc.block_metrics.len(),
    );
}

fn print_output_bundle(label: &str, root: &std::path::Path, bundle: &exports::OutputBundle) {
    eprintln!(
        "wrote {label} -> {}/{{reports,exports,manifests}}/{}/...",
        root.display(),
        bundle.checkpoint_slug
    );
}

#[cfg(test)]
mod tests {
    use super::validate_complete_inventory_scope;

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
}

fn print_console_summary(inv: &ModelInventory) {
    eprintln!(
        "checkpoint: {}  shards: {}  tensors: {}",
        inv.checkpoint_path.display(),
        inv.shard_count,
        inv.totals.tensors,
    );
    let hp = &inv.inferred;
    eprintln!(
        "inferred:  vocab={:?}  d_model={:?}  n_experts={:?}  d_ff={:?}  n_blocks={:?}",
        hp.vocab_size, hp.d_model, hp.n_experts, hp.d_ff, hp.n_blocks,
    );

    // Warn if the embedding tensor is missing or there are Unknown records.
    let has_embedding = inv
        .tensors
        .iter()
        .any(|t: &TensorInfo| matches!(t.kind, xai_dissect::schema::TensorKind::TokenEmbedding));
    if !has_embedding {
        eprintln!("warn: no TokenEmbedding tensor classified; hyperparameters may be off");
    }
    let unknown = inv
        .tensors
        .iter()
        .filter(|t| matches!(t.kind, xai_dissect::schema::TensorKind::Unknown { .. }))
        .count();
    if unknown > 0 {
        eprintln!("warn: {} tensors classified as Unknown", unknown);
    }
}

fn print_expert_console_summary(atlas: &ExpertAtlas) {
    eprintln!(
        "checkpoint: {}  blocks: {}  expected_experts_per_block: {:?}",
        atlas.checkpoint_path.display(),
        atlas.relevant_block_count,
        atlas.expected_experts_per_block,
    );
    eprintln!(
        "naming_checks: {}  anomalies: {}",
        atlas
            .naming_checks
            .iter()
            .filter(|check| check.passed)
            .count(),
        atlas.anomalies.len(),
    );
    if !atlas.anomalies.is_empty() {
        eprintln!(
            "warn: expert atlas contains {} anomalies",
            atlas.anomalies.len()
        );
    }
}

fn print_routing_console_summary(report_doc: &RoutingReport) {
    eprintln!(
        "checkpoint: {}  routing_blocks: {}  candidates: {}",
        report_doc.checkpoint_path.display(),
        report_doc.relevant_block_count,
        report_doc.candidate_tensors.len(),
    );
    eprintln!(
        "expected_experts_per_router: {:?}  critical_blocks: {}  anomalies: {}",
        report_doc.expected_experts_per_router,
        report_doc.likely_routing_critical_blocks.len(),
        report_doc.anomalies.len(),
    );
    if !report_doc.anomalies.is_empty() {
        eprintln!(
            "warn: routing report contains {} anomalies",
            report_doc.anomalies.len()
        );
    }
}

fn print_stats_console_summary(report_doc: &StatsProfileReport) {
    eprintln!(
        "checkpoint: {}  tensors: {}  layers: {}",
        report_doc.checkpoint_path.display(),
        report_doc.tensors.len(),
        report_doc.layers.len(),
    );
    eprintln!(
        "sample_values_per_tensor: {}  mean_rms: {:.6}  mean_variance: {:.6}",
        report_doc.sampling.max_sample_values,
        report_doc.norm_summary.mean_rms,
        report_doc.variance_summary.mean_variance,
    );
}

fn print_saaq_console_summary(report_doc: &SaaqReadinessReport) {
    eprintln!(
        "checkpoint: {}  candidate_targets: {}  routing_critical: {}",
        report_doc.checkpoint_path.display(),
        report_doc.candidate_targets.len(),
        report_doc.routing_critical_tensors.len(),
    );
    if let Some(top) = report_doc.candidate_targets.first() {
        eprintln!(
            "top_candidate: {}  readiness: {:.3}  risk: {:.3}",
            top.structural_name, top.readiness_score, top.risk_score,
        );
    }
}
