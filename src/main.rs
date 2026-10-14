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
//! Maintainer routing (handlers, shared flags, test touchpoints):
//! `docs/cli-routing.md`. User-facing walkthrough: `docs/run-examples.md`.
//!
//! ## What this module does NOT do
//! - It does **not** execute model inference
//! - It does **not** mutate checkpoint files
//! - It does **not** implement quantization kernels

use std::time::Instant;

use anyhow::Result;
use clap::Parser;

mod cli;
mod observability;

/// Test helpers shared by the binary crate's test modules.
#[cfg(test)]
mod test_support;

use cli::Cli;

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
    let result = cli.command.run();

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
