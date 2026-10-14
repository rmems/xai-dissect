// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Parsed CLI surface for `xai-dissect`: the `Cli`/`Command` clap types plus
//! the subcommand dispatch. Argument shapes and routing only — every command
//! implementation lives in the sibling `cli` helpers.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use super::{
    CheckpointScanArgs, ModelFamilyArg, OutputTreeArgs, PlanningFamilyArg, SampleValuesArg,
};

#[derive(Parser, Debug)]
#[command(
    name = "xai-dissect",
    version,
    about = "Static structural analysis of Grok-family open-weight checkpoints"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
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

pub(crate) struct CommandFields {
    pub(crate) limit: Option<usize>,
    pub(crate) prefix: Option<String>,
    pub(crate) family: Option<String>,
    pub(crate) sample_values: Option<usize>,
}

impl Command {
    pub fn name(&self) -> &'static str {
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

    pub fn fields(&self) -> CommandFields {
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

impl Command {
    /// Execute the parsed subcommand. Pure dispatch: every arm delegates
    /// to a `cli` helper, so no command logic lives in this module.
    pub fn run(self) -> Result<()> {
        match self {
            Command::Dissect {
                path,
                limit,
                prefix,
            } => super::dissect::run_dissect(&path, limit, &prefix),
            Command::Inventory {
                scan,
                family,
                json,
                md,
                output_tree,
            } => super::run_inventory(
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
            } => super::run_experts(
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
            } => super::run_routing_report(
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
            } => super::run_stats(
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
            } => super::run_saaq_readiness(
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
            } => super::run_pilot_plan(
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
            } => super::run_route_preservation(
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
            } => super::run_quant_plan(
                &scan,
                &family.family,
                sample_values,
                json.as_deref(),
                md.as_deref(),
                conversion_manifest.as_deref(),
                conversion_manifest_md.as_deref(),
                &output_tree,
            ),
        }
    }
}
