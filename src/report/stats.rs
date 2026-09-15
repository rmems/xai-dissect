// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Stats-profile Markdown/JSON writers.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;

use super::common::{fmt_opt_u32, human_bytes, write_pretty_json, write_text};
use crate::schema::StatsProfileReport;

/// Write the full stats profile as pretty-printed JSON.
pub fn write_stats_json(report_doc: &StatsProfileReport, out: &Path) -> Result<()> {
    write_pretty_json(report_doc, "serialize stats report to json", out)
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
    write_text(&s, out)
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
