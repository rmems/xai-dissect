// SPDX-License-Identifier: GPL-3.0-only
//
// Export layer: stable serialization of `ModelInventory` to JSON and a
// human-readable Markdown summary. These two formats are the public
// integration surface for sibling repositories.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::schema::{
    BlockSummary, CandidateTensorManifest, ExpertAtlas, ExpertIssueCategory, ModelInventory,
    RoutingIssueCategory, RoutingReport, SaaqDisposition, SaaqReadinessReport, StatsProfileReport,
};

/// Write the full inventory as pretty-printed JSON. The JSON layout is the
/// `ModelInventory` struct rendered via serde; its schema version is carried
/// in the `schema_version` field.
pub fn write_json(inv: &ModelInventory, out: &Path) -> Result<()> {
    let s = serde_json::to_string_pretty(inv).context("serialize inventory to json")?;
    fs::write(out, s).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

/// Write the full expert atlas as pretty-printed JSON.
pub fn write_expert_json(atlas: &ExpertAtlas, out: &Path) -> Result<()> {
    let s = serde_json::to_string_pretty(atlas).context("serialize expert atlas to json")?;
    fs::write(out, s).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

/// Write the full routing report as pretty-printed JSON.
pub fn write_routing_json(report_doc: &RoutingReport, out: &Path) -> Result<()> {
    let s = serde_json::to_string_pretty(report_doc).context("serialize routing report to json")?;
    fs::write(out, s).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

/// Write the full stats profile as pretty-printed JSON.
pub fn write_stats_json(report_doc: &StatsProfileReport, out: &Path) -> Result<()> {
    let s = serde_json::to_string_pretty(report_doc).context("serialize stats report to json")?;
    fs::write(out, s).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

/// Write the full SAAQ-readiness report as pretty-printed JSON.
pub fn write_saaq_readiness_json(report_doc: &SaaqReadinessReport, out: &Path) -> Result<()> {
    let s = serde_json::to_string_pretty(report_doc)
        .context("serialize saaq-readiness report to json")?;
    fs::write(out, s).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

/// Write the candidate manifest as pretty-printed JSON.
pub fn write_candidate_manifest_json(manifest: &CandidateTensorManifest, out: &Path) -> Result<()> {
    let s =
        serde_json::to_string_pretty(manifest).context("serialize candidate manifest to json")?;
    fs::write(out, s).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

/// Render a Markdown summary report for humans. Intentionally small and
/// text-only; no plots, no HTML, no colors.
pub fn render_markdown(inv: &ModelInventory) -> String {
    let mut md = String::new();

    let _ = writeln!(md, "# xai-dissect inventory");
    let _ = writeln!(md);
    let _ = writeln!(md, "- **model_family**: `{}`", inv.model_family);
    let _ = writeln!(md, "- **checkpoint**: `{}`", inv.checkpoint_path.display());
    let _ = writeln!(md, "- **shards**: {}", inv.shard_count);
    let _ = writeln!(md, "- **schema_version**: {}", inv.schema_version);

    // Inferred hyperparameters.
    let _ = writeln!(md);
    let _ = writeln!(md, "## Inferred hyperparameters");
    let _ = writeln!(md);
    let _ = writeln!(md, "| Field | Value |");
    let _ = writeln!(md, "| ----- | ----- |");
    let hp = &inv.inferred;
    let _ = writeln!(md, "| vocab_size | {} |", fmt_opt(hp.vocab_size));
    let _ = writeln!(md, "| d_model | {} |", fmt_opt(hp.d_model));
    let _ = writeln!(md, "| n_experts | {} |", fmt_opt(hp.n_experts));
    let _ = writeln!(md, "| d_ff | {} |", fmt_opt(hp.d_ff));
    let _ = writeln!(md, "| n_blocks | {} |", fmt_opt_u32(hp.n_blocks));

    // Totals.
    let _ = writeln!(md);
    let _ = writeln!(md, "## Totals");
    let _ = writeln!(md);
    let _ = writeln!(md, "| Metric | Value |");
    let _ = writeln!(md, "| ------ | ----- |");
    let t = &inv.totals;
    let _ = writeln!(md, "| tensors | {} |", t.tensors);
    let _ = writeln!(md, "| f32 tensors | {} |", t.f32_tensors);
    let _ = writeln!(md, "| int8 tensors | {} |", t.i8_tensors);
    let _ = writeln!(md, "| quant tensors | {} |", t.quant_tensors);
    let _ = writeln!(md, "| total elements | {} |", t.total_elements);
    let _ = writeln!(
        md,
        "| total bytes | {} ({}) |",
        t.total_nbytes,
        human_bytes(t.total_nbytes)
    );

    // Kind breakdown (across the whole inventory).
    let _ = writeln!(md);
    let _ = writeln!(md, "## Tensor kinds");
    let _ = writeln!(md);
    let _ = writeln!(md, "| Kind | Count | Bytes |");
    let _ = writeln!(md, "| ---- | ----: | ----: |");
    let mut agg: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for ti in &inv.tensors {
        let e = agg.entry(ti.kind.short_label()).or_insert((0, 0));
        e.0 += 1;
        e.1 += ti.nbytes;
    }
    for (k, (c, n)) in &agg {
        let _ = writeln!(md, "| {} | {} | {} ({}) |", k, c, n, human_bytes(*n));
    }

    // Block summary table (compact).
    let _ = writeln!(md);
    let _ = writeln!(md, "## Blocks");
    let _ = writeln!(md);
    let _ = writeln!(md, "| Label | Block | Shards | Tensors | Bytes | Kinds |");
    let _ = writeln!(md, "| ----- | ----: | ------ | ------: | ----: | ----- |");
    for b in &inv.blocks {
        let shards = match b.shard_range {
            Some(r) => format!("{}..={}", r.start, r.end_inclusive),
            None => "-".to_string(),
        };
        let kinds = render_kinds(b);
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} ({}) | {} |",
            b.label,
            fmt_opt_u32(b.block_index),
            shards,
            b.tensor_count,
            b.total_nbytes,
            human_bytes(b.total_nbytes),
            kinds
        );
    }

    // Exemplar block: dump the tensors of the first block-indexed summary
    // so the reader can see the per-block layout at a glance.
    if let Some(exemplar) = inv.blocks.iter().find(|b| b.block_index == Some(0)) {
        let _ = writeln!(md);
        let _ = writeln!(md, "## Exemplar block (`{}`)", exemplar.label);
        let _ = writeln!(md);
        let _ = writeln!(
            md,
            "| Shard | In-shard | Role | Dtype | Shape | Kind | Slot |"
        );
        let _ = writeln!(
            md,
            "| ----: | -------: | ---- | ----- | ----- | ---- | ---: |"
        );
        for ti in inv.tensors.iter().filter(|t| t.block_index == Some(0)) {
            let _ = writeln!(
                md,
                "| {} | {} | {} | {} | `{}` | {} | {} |",
                ti.shard_ordinal,
                ti.in_shard_index,
                ti.role.label(),
                ti.dtype.label(),
                ti.shape.render(),
                ti.kind.short_label(),
                fmt_opt_u32(ti.block_slot),
            );
        }
    }

    md
}

/// Write the Markdown summary to `out`.
pub fn write_markdown(inv: &ModelInventory, out: &Path) -> Result<()> {
    let s = render_markdown(inv);
    fs::write(out, s).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

/// Render a Markdown summary report for humans from an expert atlas.
pub fn render_expert_markdown(atlas: &ExpertAtlas) -> String {
    let mut md = String::new();

    let _ = writeln!(md, "# xai-dissect expert atlas");
    let _ = writeln!(md);
    let _ = writeln!(md, "- **model_family**: `{}`", atlas.model_family);
    let _ = writeln!(
        md,
        "- **checkpoint**: `{}`",
        atlas.checkpoint_path.display()
    );
    let _ = writeln!(md, "- **shards**: {}", atlas.shard_count);
    let _ = writeln!(md, "- **relevant_blocks**: {}", atlas.relevant_block_count);
    let _ = writeln!(
        md,
        "- **expected_experts_per_block**: {}",
        fmt_opt(atlas.expected_experts_per_block)
    );
    let _ = writeln!(md, "- **schema_version**: {}", atlas.schema_version);

    let _ = writeln!(md);
    let _ = writeln!(md, "## Expert counts by block");
    let _ = writeln!(md);
    let _ = writeln!(md, "| Block | Experts | Expert tensors | Slots | Shapes |");
    let _ = writeln!(md, "| ----: | ------: | -------------: | ----- | ------ |");
    for block in &atlas.blocks {
        let slots = if block.tensors.is_empty() {
            "-".to_string()
        } else {
            block
                .tensors
                .iter()
                .map(|tensor| {
                    tensor
                        .block_slot
                        .map(|slot| slot.to_string())
                        .unwrap_or_else(|| "?".to_string())
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        let shapes = if block.tensors.is_empty() {
            "-".to_string()
        } else {
            block
                .tensors
                .iter()
                .map(|tensor| format!("{} {}", tensor.family_label, tensor.shape.render()))
                .collect::<Vec<_>>()
                .join("<br>")
        };
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} |",
            block.block_index,
            fmt_opt(block.expert_count),
            block.tensors.len(),
            slots,
            shapes
        );
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Tensor naming patterns");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Family | Pattern | Projection | Slots | Shapes | Blocks |"
    );
    let _ = writeln!(
        md,
        "| ------ | ------- | ---------- | ----- | ------ | -----: |"
    );
    for pattern in &atlas.naming_patterns {
        let shapes = if pattern.observed_shapes.is_empty() {
            "-".to_string()
        } else {
            pattern
                .observed_shapes
                .iter()
                .map(|shape| shape.render())
                .collect::<Vec<_>>()
                .join("<br>")
        };
        let slots = if pattern.block_slots.is_empty() {
            "-".to_string()
        } else {
            pattern
                .block_slots
                .iter()
                .map(|slot| slot.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let _ = writeln!(
            md,
            "| {} | `{}` | {} | {} | {} | {} |",
            pattern.family_label,
            pattern.pattern,
            pattern.projection.label(),
            slots,
            shapes,
            pattern.observed_blocks
        );
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Naming consistency checks");
    let _ = writeln!(md);
    let _ = writeln!(md, "| Check | Result | Detail |");
    let _ = writeln!(md, "| ----- | ------ | ------ |");
    for check in &atlas.naming_checks {
        let _ = writeln!(
            md,
            "| {} | {} | {} |",
            check.check,
            if check.passed { "pass" } else { "fail" },
            check.detail
        );
    }

    render_issue_section(
        &mut md,
        "Missing or irregular expert tensors",
        atlas,
        ExpertIssueCategory::MissingOrIrregularTensor,
    );
    render_issue_section(
        &mut md,
        "Layout anomalies",
        atlas,
        ExpertIssueCategory::LayoutAnomaly,
    );

    if let Some(block) = atlas.blocks.first() {
        let _ = writeln!(md);
        let _ = writeln!(md, "## Exemplar block (`block_{:03}`)", block.block_index);
        let _ = writeln!(md);
        let _ = writeln!(md, "| Expert | Tensor associations |");
        let _ = writeln!(md, "| -----: | ------------------- |");
        for expert in &block.experts {
            let associations = if expert.tensors.is_empty() {
                "-".to_string()
            } else {
                expert
                    .tensors
                    .iter()
                    .map(|tensor| {
                        format!(
                            "`{}` {} `{}`",
                            tensor.structural_name,
                            tensor.projection.label(),
                            tensor.slice_shape.render()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("<br>")
            };
            let _ = writeln!(md, "| {} | {} |", expert.expert_index, associations);
        }
    }

    md
}

/// Write the expert atlas Markdown summary to `out`.
pub fn write_expert_markdown(atlas: &ExpertAtlas, out: &Path) -> Result<()> {
    let s = render_expert_markdown(atlas);
    fs::write(out, s).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

/// Render a Markdown summary report for humans from a routing report.
pub fn render_routing_markdown(report_doc: &RoutingReport) -> String {
    let mut md = String::new();

    let _ = writeln!(md, "# xai-dissect routing report");
    let _ = writeln!(md);
    let _ = writeln!(md, "- **model_family**: `{}`", report_doc.model_family);
    let _ = writeln!(
        md,
        "- **checkpoint**: `{}`",
        report_doc.checkpoint_path.display()
    );
    let _ = writeln!(md, "- **shards**: {}", report_doc.shard_count);
    let _ = writeln!(
        md,
        "- **relevant_blocks**: {}",
        report_doc.relevant_block_count
    );
    let _ = writeln!(
        md,
        "- **expected_experts_per_router**: {}",
        fmt_opt(report_doc.expected_experts_per_router)
    );
    let _ = writeln!(md, "- **schema_version**: {}", report_doc.schema_version);

    let _ = writeln!(md);
    let _ = writeln!(md, "## Candidate routing tensors");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Block | Slot | Shape | Orientation | Experts | Kind | Structural name |"
    );
    let _ = writeln!(
        md,
        "| ----: | ---: | ----- | ----------- | ------: | ---- | --------------- |"
    );
    for tensor in &report_doc.candidate_tensors {
        let _ = writeln!(
            md,
            "| {} | {} | `{}` | {} | {} | {} | `{}` |",
            tensor
                .block_index
                .map(|index| index.to_string())
                .unwrap_or_else(|| "-".to_string()),
            tensor
                .block_slot
                .map(|slot| slot.to_string())
                .unwrap_or_else(|| "-".to_string()),
            tensor.shape.render(),
            tensor.orientation.label(),
            fmt_opt(tensor.linked_expert_count),
            tensor.kind_label,
            tensor.structural_name
        );
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Shape and orientation summaries");
    let _ = writeln!(md);
    let _ = writeln!(md, "| Orientation | Count | Blocks | Shapes |");
    let _ = writeln!(md, "| ----------- | ----: | -----: | ------ |");
    for summary in &report_doc.orientation_summaries {
        let shapes = if summary.observed_shapes.is_empty() {
            "-".to_string()
        } else {
            summary
                .observed_shapes
                .iter()
                .map(|shape| shape.render())
                .collect::<Vec<_>>()
                .join("<br>")
        };
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} |",
            summary.orientation.label(),
            summary.count,
            summary.observed_blocks,
            shapes
        );
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Layer-by-layer routing metadata");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Label | Block | Local experts | Primary candidate | Candidate count |"
    );
    let _ = writeln!(
        md,
        "| ----- | ----: | ------------: | ----------------- | --------------: |"
    );
    for block in &report_doc.blocks {
        let primary = block
            .primary_candidate
            .as_ref()
            .map(|locator| {
                format!(
                    "shard {} idx {} slot {}",
                    locator.shard_ordinal,
                    locator.in_shard_index,
                    locator
                        .block_slot
                        .map(|slot| slot.to_string())
                        .unwrap_or_else(|| "-".to_string())
                )
            })
            .unwrap_or_else(|| "-".to_string());
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} |",
            block.label,
            block
                .block_index
                .map(|index| index.to_string())
                .unwrap_or_else(|| "-".to_string()),
            fmt_opt(block.local_expert_count),
            primary,
            block.candidates.len()
        );
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Gate tensor structural metrics");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Structural name | Input width | Output width | Logits/input | Bytes |"
    );
    let _ = writeln!(
        md,
        "| --------------- | ----------: | -----------: | -----------: | ----: |"
    );
    for tensor in &report_doc.candidate_tensors {
        let _ = writeln!(
            md,
            "| `{}` | {} | {} | {} | {} ({}) |",
            tensor.structural_name,
            fmt_opt(tensor.gate_metrics.input_width),
            fmt_opt(tensor.gate_metrics.output_width),
            fmt_opt(tensor.gate_metrics.logits_per_input),
            tensor.gate_metrics.total_nbytes,
            human_bytes(tensor.gate_metrics.total_nbytes)
        );
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Expert count linkage");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Structural name | Linked experts | Matches inferred experts |"
    );
    let _ = writeln!(
        md,
        "| --------------- | -------------: | ----------------------- |"
    );
    for tensor in &report_doc.candidate_tensors {
        let _ = writeln!(
            md,
            "| `{}` | {} | {} |",
            tensor.structural_name,
            fmt_opt(tensor.linked_expert_count),
            if tensor.matches_inferred_expert_count {
                "yes"
            } else {
                "no"
            }
        );
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Likely routing-critical blocks");
    let _ = writeln!(md);
    if report_doc.likely_routing_critical_blocks.is_empty() {
        let _ = writeln!(md, "None detected.");
    } else {
        let _ = writeln!(md, "| Block | Label | Reason |");
        let _ = writeln!(md, "| ----: | ----- | ------ |");
        for block in &report_doc.likely_routing_critical_blocks {
            let _ = writeln!(
                md,
                "| {} | {} | {} |",
                block
                    .block_index
                    .map(|index| index.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                block.label,
                block.reason
            );
        }
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Grok-specific layout notes");
    let _ = writeln!(md);
    if report_doc.grok_layout_notes.is_empty() {
        let _ = writeln!(md, "None detected.");
    } else {
        for note in &report_doc.grok_layout_notes {
            let _ = writeln!(md, "- {}", note);
        }
    }

    render_routing_issue_section(&mut md, "Routing anomalies", report_doc, None);
    render_routing_issue_section(
        &mut md,
        "Missing routing candidates",
        report_doc,
        Some(RoutingIssueCategory::MissingCandidate),
    );

    md
}

/// Write the routing Markdown summary to `out`.
pub fn write_routing_markdown(report_doc: &RoutingReport, out: &Path) -> Result<()> {
    let s = render_routing_markdown(report_doc);
    fs::write(out, s).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

/// Render a Markdown summary report for humans from a stats profile.
pub fn render_stats_markdown(report_doc: &StatsProfileReport) -> String {
    let mut md = String::new();

    let _ = writeln!(md, "# xai-dissect stats report");
    let _ = writeln!(md);
    let _ = writeln!(md, "- **model_family**: `{}`", report_doc.model_family);
    let _ = writeln!(
        md,
        "- **checkpoint**: `{}`",
        report_doc.checkpoint_path.display()
    );
    let _ = writeln!(md, "- **shards**: {}", report_doc.shard_count);
    let _ = writeln!(
        md,
        "- **sample_values_per_tensor**: {}",
        report_doc.sampling.max_sample_values
    );
    let _ = writeln!(md, "- **schema_version**: {}", report_doc.schema_version);

    let _ = writeln!(md);
    let _ = writeln!(md, "## Norm summary");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "- **mean_rms**: {:.6}",
        report_doc.norm_summary.mean_rms
    );
    render_ranked_table(&mut md, "Top RMS tensors", &report_doc.norm_summary.top_rms);
    render_ranked_table(&mut md, "Top L2 tensors", &report_doc.norm_summary.top_l2);

    let _ = writeln!(md);
    let _ = writeln!(md, "## Variance summary");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "- **mean_variance**: {:.6}",
        report_doc.variance_summary.mean_variance
    );
    render_ranked_table(
        &mut md,
        "Top variance tensors",
        &report_doc.variance_summary.top_variance,
    );
    render_ranked_table(
        &mut md,
        "Lowest variance tensors",
        &report_doc.variance_summary.lowest_variance,
    );

    let _ = writeln!(md);
    let _ = writeln!(md, "## Outlier summary");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "- **mean_outlier_fraction**: {:.6}",
        report_doc.outlier_summary.mean_outlier_fraction
    );
    render_ranked_table(
        &mut md,
        "Most outlier-heavy tensors",
        &report_doc.outlier_summary.most_outlier_heavy,
    );
    render_ranked_table(
        &mut md,
        "Highest peak-to-RMS tensors",
        &report_doc.outlier_summary.highest_peak_to_rms,
    );

    let _ = writeln!(md);
    let _ = writeln!(md, "## Per-layer metrics");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Label | Block | Tensors | Bytes | Mean RMS | Mean variance | Mean outlier frac | Routing tensors | Candidate-like tensors |"
    );
    let _ = writeln!(
        md,
        "| ----- | ----: | ------: | ----: | -------: | ------------: | ----------------: | --------------: | ---------------------: |"
    );
    for layer in &report_doc.layers {
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} ({}) | {:.6} | {:.6} | {:.6} | {} | {} |",
            layer.label,
            fmt_opt_u32(layer.block_index),
            layer.tensor_count,
            layer.total_nbytes,
            human_bytes(layer.total_nbytes),
            layer.mean_rms,
            layer.mean_variance,
            layer.mean_outlier_fraction,
            layer.routing_tensor_count,
            layer.compressible_candidate_count
        );
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Per-tensor metrics");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Tensor | Kind | Dtype | Shape | RMS | Variance | Zero frac | Near-zero frac | Outlier frac | Distribution |"
    );
    let _ = writeln!(
        md,
        "| ------ | ---- | ----- | ----- | ---: | -------: | --------: | -------------: | -----------: | ------------ |"
    );
    for tensor in &report_doc.tensors {
        let _ = writeln!(
            md,
            "| `{}` | {} | {} | `{}` | {:.6} | {:.6} | {:.4} | {:.4} | {:.4} | {} |",
            tensor.structural_name,
            tensor.kind_label,
            tensor.dtype.label(),
            tensor.shape.render(),
            tensor.rms,
            tensor.variance,
            tensor.zero_fraction,
            tensor.near_zero_fraction,
            tensor.outlier_fraction,
            tensor.distribution_label
        );
    }

    md
}

/// Write the stats Markdown summary to `out`.
pub fn write_stats_markdown(report_doc: &StatsProfileReport, out: &Path) -> Result<()> {
    let s = render_stats_markdown(report_doc);
    fs::write(out, s).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

/// Render a Markdown summary report for humans from a SAAQ-readiness report.
pub fn render_saaq_readiness_markdown(report_doc: &SaaqReadinessReport) -> String {
    let mut md = String::new();

    let _ = writeln!(md, "# xai-dissect SAAQ-readiness report");
    let _ = writeln!(md);
    let _ = writeln!(md, "- **model_family**: `{}`", report_doc.model_family);
    let _ = writeln!(
        md,
        "- **checkpoint**: `{}`",
        report_doc.checkpoint_path.display()
    );
    let _ = writeln!(md, "- **shards**: {}", report_doc.shard_count);
    let _ = writeln!(
        md,
        "- **candidate_targets**: {}",
        report_doc.candidate_targets.len()
    );
    let _ = writeln!(
        md,
        "- **routing_critical_tensors**: {}",
        report_doc.routing_critical_tensors.len()
    );
    let _ = writeln!(md, "- **schema_version**: {}", report_doc.schema_version);

    let _ = writeln!(md);
    let _ = writeln!(md, "## Candidate target tensors");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Rank | Tensor | Kind | Region | Readiness | Opportunity | Risk | Disposition |"
    );
    let _ = writeln!(
        md,
        "| ---: | ------ | ---- | ------ | --------: | ----------: | ---: | ----------- |"
    );
    for candidate in &report_doc.candidate_targets {
        let _ = writeln!(
            md,
            "| {} | `{}` | {} | {} | {:.3} | {:.3} | {:.3} | {} |",
            candidate.rank,
            candidate.structural_name,
            candidate.kind_label,
            saaq_region_label(candidate.region_class),
            candidate.readiness_score,
            candidate.opportunity_score,
            candidate.risk_score,
            saaq_disposition_label(candidate.disposition)
        );
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Routing-critical tensors");
    let _ = writeln!(md);
    if report_doc.routing_critical_tensors.is_empty() {
        let _ = writeln!(md, "None detected.");
    } else {
        let _ = writeln!(md, "| Tensor | Readiness | Risk | Reasons |");
        let _ = writeln!(md, "| ------ | --------: | ---: | ------- |");
        for candidate in &report_doc.routing_critical_tensors {
            let _ = writeln!(
                md,
                "| `{}` | {:.3} | {:.3} | {} |",
                candidate.structural_name,
                candidate.readiness_score,
                candidate.risk_score,
                candidate.reasons.join("<br>")
            );
        }
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Highest-risk tensors");
    let _ = writeln!(md);
    if report_doc.risky_tensors.is_empty() {
        let _ = writeln!(md, "None detected.");
    } else {
        let _ = writeln!(md, "| Tensor | Region | Risk | Reasons |");
        let _ = writeln!(md, "| ------ | ------ | ---: | ------- |");
        for candidate in &report_doc.risky_tensors {
            let _ = writeln!(
                md,
                "| `{}` | {} | {:.3} | {} |",
                candidate.structural_name,
                saaq_region_label(candidate.region_class),
                candidate.risk_score,
                candidate.reasons.join("<br>")
            );
        }
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Layer readiness");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Label | Block | Routing critical | Candidate targets | Mean readiness | Max risk |"
    );
    let _ = writeln!(
        md,
        "| ----- | ----: | ---------------- | ----------------: | -------------: | -------: |"
    );
    for layer in &report_doc.layer_readiness {
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {:.3} | {:.3} |",
            layer.label,
            fmt_opt_u32(layer.block_index),
            if layer.routing_critical { "yes" } else { "no" },
            layer.candidate_target_count,
            layer.mean_readiness_score,
            layer.max_risk_score
        );
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "## Notes");
    let _ = writeln!(md);
    if report_doc.notes.is_empty() {
        let _ = writeln!(md, "None.");
    } else {
        for note in &report_doc.notes {
            let _ = writeln!(md, "- {}", note);
        }
    }

    md
}

/// Write the SAAQ-readiness Markdown summary to `out`.
pub fn write_saaq_readiness_markdown(report_doc: &SaaqReadinessReport, out: &Path) -> Result<()> {
    let s = render_saaq_readiness_markdown(report_doc);
    fs::write(out, s).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

// --- Helpers ---------------------------------------------------------------

fn render_kinds(b: &BlockSummary) -> String {
    if b.kinds.is_empty() {
        return "-".to_string();
    }
    b.kinds
        .iter()
        .map(|k| format!("{}x{}", k.count, k.kind_label))
        .collect::<Vec<_>>()
        .join(", ")
}

fn fmt_opt(v: Option<u64>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "-".to_string(),
    }
}

fn fmt_opt_u32(v: Option<u32>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "-".to_string(),
    }
}

fn human_bytes(n: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0usize;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{} {}", n, UNITS[0])
    } else {
        format!("{:.2} {}", v, UNITS[u])
    }
}

fn render_issue_section(
    md: &mut String,
    title: &str,
    atlas: &ExpertAtlas,
    category: ExpertIssueCategory,
) {
    let issues = atlas
        .anomalies
        .iter()
        .filter(|issue| issue.category == category)
        .collect::<Vec<_>>();

    let _ = writeln!(md);
    let _ = writeln!(md, "## {}", title);
    let _ = writeln!(md);

    if issues.is_empty() {
        let _ = writeln!(md, "None detected.");
        return;
    }

    let _ = writeln!(md, "| Block | Severity | Tensor | Message |");
    let _ = writeln!(md, "| ----: | -------- | ------ | ------- |");
    for issue in issues {
        let tensor = issue
            .tensor
            .as_ref()
            .map(|tensor| {
                format!(
                    "shard {} idx {} slot {}",
                    tensor.shard_ordinal,
                    tensor.in_shard_index,
                    tensor
                        .block_slot
                        .map(|slot| slot.to_string())
                        .unwrap_or_else(|| "?".to_string())
                )
            })
            .unwrap_or_else(|| "-".to_string());
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} |",
            issue
                .block_index
                .map(|index| index.to_string())
                .unwrap_or_else(|| "-".to_string()),
            match issue.severity {
                crate::schema::ExpertIssueSeverity::Warning => "warning",
                crate::schema::ExpertIssueSeverity::Error => "error",
            },
            tensor,
            issue.message
        );
    }
}

fn render_routing_issue_section(
    md: &mut String,
    title: &str,
    report_doc: &RoutingReport,
    category: Option<RoutingIssueCategory>,
) {
    let issues = report_doc
        .anomalies
        .iter()
        .filter(|issue| {
            category
                .map(|category| issue.category == category)
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();

    let _ = writeln!(md);
    let _ = writeln!(md, "## {}", title);
    let _ = writeln!(md);

    if issues.is_empty() {
        let _ = writeln!(md, "None detected.");
        return;
    }

    let _ = writeln!(md, "| Block | Severity | Category | Tensor | Message |");
    let _ = writeln!(md, "| ----: | -------- | -------- | ------ | ------- |");
    for issue in issues {
        let tensor = issue
            .tensor
            .as_ref()
            .map(|tensor| {
                format!(
                    "shard {} idx {} slot {}",
                    tensor.shard_ordinal,
                    tensor.in_shard_index,
                    tensor
                        .block_slot
                        .map(|slot| slot.to_string())
                        .unwrap_or_else(|| "?".to_string())
                )
            })
            .unwrap_or_else(|| "-".to_string());
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} |",
            issue
                .block_index
                .map(|index| index.to_string())
                .unwrap_or_else(|| "-".to_string()),
            match issue.severity {
                crate::schema::RoutingIssueSeverity::Warning => "warning",
                crate::schema::RoutingIssueSeverity::Error => "error",
            },
            match issue.category {
                crate::schema::RoutingIssueCategory::MissingCandidate => "missing_candidate",
                crate::schema::RoutingIssueCategory::ShapeSummary => "shape_summary",
                crate::schema::RoutingIssueCategory::ExpertCountLinkage => "expert_count_linkage",
                crate::schema::RoutingIssueCategory::LayoutNote => "layout_note",
            },
            tensor,
            issue.message
        );
    }
}

fn render_ranked_table(md: &mut String, title: &str, rows: &[crate::schema::RankedTensorStat]) {
    let _ = writeln!(md);
    let _ = writeln!(md, "### {}", title);
    let _ = writeln!(md);
    if rows.is_empty() {
        let _ = writeln!(md, "None detected.");
        return;
    }
    let _ = writeln!(md, "| Tensor | Kind | Block | Value |");
    let _ = writeln!(md, "| ------ | ---- | ----: | ----: |");
    for row in rows {
        let _ = writeln!(
            md,
            "| `{}` | {} | {} | {:.6} |",
            row.structural_name,
            row.kind_label,
            fmt_opt_u32(row.block_index),
            row.value
        );
    }
}

fn saaq_region_label(region: crate::schema::SaaqRegionClass) -> &'static str {
    match region {
        crate::schema::SaaqRegionClass::RoutingCritical => "routing_critical",
        crate::schema::SaaqRegionClass::NormalizationSensitive => "normalization_sensitive",
        crate::schema::SaaqRegionClass::AlreadyCompressed => "already_compressed",
        crate::schema::SaaqRegionClass::PotentialCompressionTarget => "potential_target",
        crate::schema::SaaqRegionClass::EmbeddingHeavy => "embedding_heavy",
        crate::schema::SaaqRegionClass::Unknown => "unknown",
    }
}

fn saaq_disposition_label(disposition: SaaqDisposition) -> &'static str {
    match disposition {
        SaaqDisposition::Candidate => "candidate",
        SaaqDisposition::ObserveOnly => "observe_only",
        SaaqDisposition::AvoidForNow => "avoid_for_now",
    }
}
