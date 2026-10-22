// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! One-command adapters between `Command` variants and the `cli::run_*`
//! handlers. Kept separate so each adapter stays under Codacy's method-length
//! limit.

use std::path::PathBuf;

use anyhow::Result;

use super::{
    CheckpointScanArgs, ModelFamilyArg, OutputTreeArgs, PlanningFamilyArg, SampleValuesArg,
};

pub(super) fn run_inventory_cmd(
    scan: CheckpointScanArgs,
    family: ModelFamilyArg,
    json: Option<PathBuf>,
    md: Option<PathBuf>,
    output_tree: OutputTreeArgs,
) -> Result<()> {
    super::run_inventory(
        &scan,
        &family.family,
        json.as_deref(),
        md.as_deref(),
        &output_tree,
    )
}

pub(super) fn run_experts_cmd(
    scan: CheckpointScanArgs,
    family: ModelFamilyArg,
    json: Option<PathBuf>,
    md: Option<PathBuf>,
    output_tree: OutputTreeArgs,
) -> Result<()> {
    super::run_experts(
        &scan,
        &family.family,
        json.as_deref(),
        md.as_deref(),
        &output_tree,
    )
}

pub(super) fn run_routing_cmd(
    scan: CheckpointScanArgs,
    family: ModelFamilyArg,
    json: Option<PathBuf>,
    md: Option<PathBuf>,
    output_tree: OutputTreeArgs,
) -> Result<()> {
    super::run_routing_report(
        &scan,
        &family.family,
        json.as_deref(),
        md.as_deref(),
        &output_tree,
    )
}

pub(super) fn run_stats_cmd(
    scan: CheckpointScanArgs,
    family: ModelFamilyArg,
    sample: SampleValuesArg,
    json: Option<PathBuf>,
    md: Option<PathBuf>,
    output_tree: OutputTreeArgs,
) -> Result<()> {
    super::run_stats(
        &scan,
        &family.family,
        sample.sample_values,
        json.as_deref(),
        md.as_deref(),
        &output_tree,
    )
}

pub(super) fn run_saaq_cmd(
    scan: CheckpointScanArgs,
    family: ModelFamilyArg,
    sample: SampleValuesArg,
    json: Option<PathBuf>,
    md: Option<PathBuf>,
    manifest: Option<PathBuf>,
    output_tree: OutputTreeArgs,
) -> Result<()> {
    super::run_saaq_readiness(
        &scan,
        &family.family,
        sample.sample_values,
        json.as_deref(),
        md.as_deref(),
        manifest.as_deref(),
        &output_tree,
    )
}

pub(super) fn run_pilot_cmd(
    scan: CheckpointScanArgs,
    family: PlanningFamilyArg,
    json: Option<PathBuf>,
    md: Option<PathBuf>,
    output_tree: OutputTreeArgs,
) -> Result<()> {
    super::run_pilot_plan(
        &scan,
        &family.family,
        json.as_deref(),
        md.as_deref(),
        &output_tree,
    )
}

pub(super) fn run_route_cmd(
    scan: CheckpointScanArgs,
    family: PlanningFamilyArg,
    json: Option<PathBuf>,
    md: Option<PathBuf>,
    output_tree: OutputTreeArgs,
) -> Result<()> {
    super::run_route_preservation(
        &scan,
        &family.family,
        json.as_deref(),
        md.as_deref(),
        &output_tree,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_quant_cmd(
    scan: CheckpointScanArgs,
    family: ModelFamilyArg,
    sample_values: usize,
    json: Option<PathBuf>,
    md: Option<PathBuf>,
    conversion_manifest: Option<PathBuf>,
    conversion_manifest_md: Option<PathBuf>,
    output_tree: OutputTreeArgs,
) -> Result<()> {
    super::run_quant_plan(
        &scan,
        &family.family,
        sample_values,
        json.as_deref(),
        md.as_deref(),
        conversion_manifest.as_deref(),
        conversion_manifest_md.as_deref(),
        &output_tree,
    )
}
