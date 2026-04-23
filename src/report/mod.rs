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

use crate::schema::{BlockSummary, ModelInventory};

/// Write the full inventory as pretty-printed JSON. The JSON layout is the
/// `ModelInventory` struct rendered via serde; its schema version is carried
/// in the `schema_version` field.
pub fn write_json(inv: &ModelInventory, out: &Path) -> Result<()> {
    let s = serde_json::to_string_pretty(inv).context("serialize inventory to json")?;
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
    let _ = writeln!(md, "| total bytes | {} ({}) |", t.total_nbytes, human_bytes(t.total_nbytes));

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
        let _ = writeln!(md, "| Shard | In-shard | Role | Dtype | Shape | Kind | Slot |");
        let _ = writeln!(md, "| ----: | -------: | ---- | ----- | ----- | ---- | ---: |");
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
