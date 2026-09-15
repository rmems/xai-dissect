// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Routing-report Markdown/JSON writers (`routing-report.md`, routing-critical manifest).

use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;

use crate::schema::{RoutingCriticalTensorManifest, RoutingIssueCategory, RoutingReport};

use super::common::{fmt_opt, human_bytes, write_pretty_json, write_text};

/// Write the full routing report as pretty-printed JSON.
pub fn write_routing_json(report_doc: &RoutingReport, out: &Path) -> Result<()> {
    write_pretty_json(report_doc, "serialize routing report to json", out)
}

/// Write the routing-critical tensor manifest as pretty-printed JSON.
pub fn write_routing_critical_manifest_json(
    manifest: &RoutingCriticalTensorManifest,
    out: &Path,
) -> Result<()> {
    write_pretty_json(manifest, "serialize routing-critical manifest to json", out)
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
    write_text(&s, out)
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
