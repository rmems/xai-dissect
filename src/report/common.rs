// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Shared JSON/text writers and formatting helpers for report artifacts.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::schema::FindingsSummary;

/// Write the compact findings summary as pretty-printed JSON.
pub fn write_findings_summary_json(summary: &FindingsSummary, out: &Path) -> Result<()> {
    write_pretty_json(summary, "serialize findings summary to json", out)
}

pub(super) fn write_pretty_json<T: Serialize>(value: &T, context: &str, out: &Path) -> Result<()> {
    let s = serde_json::to_string_pretty(value).with_context(|| context.to_string())?;
    write_text(&s, out)
}

pub(super) fn write_text(s: &str, out: &Path) -> Result<()> {
    ensure_parent_dir(out)?;
    fs::write(out, s).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

fn ensure_parent_dir(out: &Path) -> Result<()> {
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    Ok(())
}

pub(super) fn fmt_opt(v: Option<u64>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "-".to_string(),
    }
}

pub(super) fn fmt_opt_u32(v: Option<u32>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "-".to_string(),
    }
}

pub(super) fn format_tensor_locator(
    shard_ordinal: u32,
    in_shard_index: u32,
    block_slot: Option<u32>,
    missing_slot: &str,
) -> String {
    format!(
        "shard {} idx {} slot {}",
        shard_ordinal,
        in_shard_index,
        block_slot
            .map(|slot| slot.to_string())
            .unwrap_or_else(|| missing_slot.to_string())
    )
}

pub(super) fn render_bullet_section(md: &mut String, title: &str, items: &[String], empty: &str) {
    let _ = writeln!(md);
    let _ = writeln!(md, "## {title}");
    let _ = writeln!(md);
    if items.is_empty() {
        let _ = writeln!(md, "{empty}");
    } else {
        for item in items {
            let _ = writeln!(md, "- {item}");
        }
    }
}

pub(super) fn render_notes_section(md: &mut String, notes: &[String], empty: &str) {
    render_bullet_section(md, "Notes", notes, empty);
}

pub(super) fn human_bytes(n: u64) -> String {
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
