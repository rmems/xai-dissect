// SPDX-License-Identifier: GPL-3.0-only
//
// xai-dissect CLI. Thin wrapper over the library crate. Two subcommands:
//
//   dissect    - legacy per-shard byte-table view (parser output only).
//   inventory  - full checkpoint cartography with JSON / Markdown export.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use comfy_table::{Cell, ContentArrangement, Table, presets::UTF8_FULL};

use xai_dissect::experts::build_expert_atlas;
use xai_dissect::inventory::{InventoryConfig, build_inventory};
use xai_dissect::parser;
use xai_dissect::report;
use xai_dissect::routing::build_routing_report;
use xai_dissect::schema::{ExpertAtlas, ModelInventory, RoutingReport, TensorInfo};

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
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
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
        } => run_inventory(
            &path,
            &prefix,
            limit,
            &family,
            json.as_deref(),
            md.as_deref(),
        ),
        Command::Experts {
            path,
            prefix,
            limit,
            family,
            json,
            md,
        } => run_experts(
            &path,
            &prefix,
            limit,
            &family,
            json.as_deref(),
            md.as_deref(),
        ),
        Command::RoutingReport {
            path,
            prefix,
            limit,
            family,
            json,
            md,
        } => run_routing_report(
            &path,
            &prefix,
            limit,
            &family,
            json.as_deref(),
            md.as_deref(),
        ),
    }
}

// --- `dissect` -------------------------------------------------------------

fn run_dissect(path: &std::path::Path, limit: Option<usize>, prefix: &str) -> Result<()> {
    let md = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if !md.is_dir() {
        bail!("{} is not a directory", path.display());
    }

    let mut shards: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with(prefix))
                    .unwrap_or(false)
        })
        .collect();
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

    Ok(())
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
