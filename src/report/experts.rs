// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Expert-atlas Markdown/JSON writers.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;

use super::common::{fmt_opt, format_tensor_locator, write_pretty_json, write_text};
use crate::schema::{ExpertAtlas, ExpertIssueCategory};

/// Write the full expert atlas as pretty-printed JSON.
pub fn write_expert_json(atlas: &ExpertAtlas, out: &Path) -> Result<()> {
    write_pretty_json(atlas, "serialize expert atlas to json", out)
}

/// Render a Markdown summary report for humans from an expert atlas.
pub fn render_expert_markdown(atlas: &ExpertAtlas) -> String {
    let mut md = String::new();
    render_expert_header(&mut md, atlas);
    render_expert_counts(&mut md, atlas);
    render_naming_patterns(&mut md, atlas);
    render_naming_checks(&mut md, atlas);
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
    render_issue_section(
        &mut md,
        "Naming consistency issues",
        atlas,
        ExpertIssueCategory::NamingConsistency,
    );
    render_expert_exemplar(&mut md, atlas);
    md
}

/// Write the expert atlas Markdown summary to `out`.
pub fn write_expert_markdown(atlas: &ExpertAtlas, out: &Path) -> Result<()> {
    let s = render_expert_markdown(atlas);
    write_text(&s, out)
}

fn render_expert_header(md: &mut String, atlas: &ExpertAtlas) {
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
}

fn render_expert_counts(md: &mut String, atlas: &ExpertAtlas) {
    let _ = writeln!(md);
    let _ = writeln!(md, "## Expert counts by block");
    let _ = writeln!(md);
    let _ = writeln!(md, "| Block | Experts | Expert tensors | Slots | Shapes |");
    let _ = writeln!(md, "| ----: | ------: | -------------: | ----- | ------ |");
    for block in &atlas.blocks {
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} |",
            block.block_index,
            fmt_opt(block.expert_count),
            block.tensors.len(),
            expert_block_slots(&block.tensors),
            expert_block_shapes(&block.tensors)
        );
    }
}

fn expert_block_slots(tensors: &[crate::schema::ExpertTensorRef]) -> String {
    if tensors.is_empty() {
        return "-".to_string();
    }
    tensors
        .iter()
        .map(|tensor| {
            tensor
                .block_slot
                .map(|slot| slot.to_string())
                .unwrap_or_else(|| "?".to_string())
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn expert_block_shapes(tensors: &[crate::schema::ExpertTensorRef]) -> String {
    if tensors.is_empty() {
        return "-".to_string();
    }
    tensors
        .iter()
        .map(|tensor| format!("{} {}", tensor.family_label, tensor.shape.render()))
        .collect::<Vec<_>>()
        .join("<br>")
}

fn render_naming_patterns(md: &mut String, atlas: &ExpertAtlas) {
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
}

fn render_naming_checks(md: &mut String, atlas: &ExpertAtlas) {
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
}

fn render_expert_exemplar(md: &mut String, atlas: &ExpertAtlas) {
    let Some(block) = atlas.blocks.first() else {
        return;
    };
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

    write_expert_issue_table(md, &issues);
}

fn write_expert_issue_table(md: &mut String, issues: &[&crate::schema::ExpertIssue]) {
    let _ = writeln!(md, "| Block | Severity | Tensor | Message |");
    let _ = writeln!(md, "| ----: | -------- | ------ | ------- |");
    for issue in issues {
        let tensor = issue
            .tensor
            .as_ref()
            .map(|tensor| {
                format_tensor_locator(
                    tensor.shard_ordinal,
                    tensor.in_shard_index,
                    tensor.block_slot,
                    "?",
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
