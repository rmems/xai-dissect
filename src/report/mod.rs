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

use crate::schema::{BlockSummary, ExpertAtlas, ExpertIssueCategory, ModelInventory};

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
