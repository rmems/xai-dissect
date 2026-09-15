// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Inventory Markdown/JSON writers and snapshot/coverage manifests.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;

use super::common::{fmt_opt, fmt_opt_u32, human_bytes, write_pretty_json, write_text};
use crate::schema::{
    BlockSummary, CheckpointInventorySnapshot, Grok1CoverageManifest, ModelInventory,
};

/// Write the full inventory as pretty-printed JSON. The JSON layout is the
/// `ModelInventory` struct rendered via serde; its schema version is carried
/// in the `schema_version` field.
pub fn write_json(inv: &ModelInventory, out: &Path) -> Result<()> {
    write_pretty_json(inv, "serialize inventory to json", out)
}

/// Write the inventory snapshot manifest as pretty-printed JSON.
pub fn write_inventory_snapshot_manifest_json(
    manifest: &CheckpointInventorySnapshot,
    out: &Path,
) -> Result<()> {
    write_pretty_json(
        manifest,
        "serialize inventory snapshot manifest to json",
        out,
    )
}

/// Write the strict Grok-1 coverage manifest as pretty-printed JSON.
pub fn write_grok1_coverage_manifest_json(
    manifest: &Grok1CoverageManifest,
    out: &Path,
) -> Result<()> {
    write_pretty_json(manifest, "serialize grok-1 coverage manifest", out)
}

/// Render a Markdown summary report for humans. Intentionally small and
/// text-only; no plots, no HTML, no colors.
pub fn render_markdown(inv: &ModelInventory) -> String {
    let mut md = String::new();
    render_inventory_preamble(&mut md, inv);
    render_inventory_kinds(&mut md, inv);
    render_inventory_blocks(&mut md, inv);
    render_inventory_exemplar(&mut md, inv);
    md
}

/// Write the Markdown summary to `out`.
pub fn write_markdown(inv: &ModelInventory, out: &Path) -> Result<()> {
    let s = render_markdown(inv);
    write_text(&s, out)
}

fn render_inventory_preamble(md: &mut String, inv: &ModelInventory) {
    let _ = writeln!(md, "# xai-dissect inventory");
    let _ = writeln!(md);
    let _ = writeln!(md, "- **model_family**: `{}`", inv.model_family);
    let _ = writeln!(md, "- **checkpoint**: `{}`", inv.checkpoint_path.display());
    let _ = writeln!(md, "- **shards**: {}", inv.shard_count);
    let _ = writeln!(md, "- **schema_version**: {}", inv.schema_version);

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
}

fn render_inventory_kinds(md: &mut String, inv: &ModelInventory) {
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
}

fn render_inventory_blocks(md: &mut String, inv: &ModelInventory) {
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
}

fn render_inventory_exemplar(md: &mut String, inv: &ModelInventory) {
    let Some(exemplar) = inv.blocks.iter().find(|b| b.block_index == Some(0)) else {
        return;
    };
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
