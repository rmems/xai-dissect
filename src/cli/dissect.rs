// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! The `dissect` subcommand: parse each shard of a checkpoint directory and
//! print a per-shard tensor table. Raw parser output only — no
//! classification, no grouping.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use comfy_table::{Cell, ContentArrangement, Table, presets::UTF8_FULL};

use xai_dissect::parser::{self, RawTensor};

pub(crate) fn run_dissect(path: &Path, limit: Option<usize>, prefix: &str) -> Result<()> {
    let shards = collect_shards(path, limit, prefix)?;
    for shard in &shards {
        print_shard(shard);
    }
    Ok(())
}

fn collect_shards(path: &Path, limit: Option<usize>, prefix: &str) -> Result<Vec<PathBuf>> {
    ensure_directory(path)?;
    let mut shards = matching_files(path, prefix)?;
    shards.sort();
    require_shards(&shards, path, prefix)?;
    if let Some(n) = limit {
        shards.truncate(n);
    }
    Ok(shards)
}

fn ensure_directory(path: &Path) -> Result<()> {
    let metadata = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if !metadata.is_dir() {
        bail!("{} is not a directory", path.display());
    }
    Ok(())
}

fn matching_files(path: &Path, prefix: &str) -> Result<Vec<PathBuf>> {
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
        let entry = entry.with_context(|| format!("read entry in {}", path.display()))?;
        let candidate = entry.path();
        if file_matches_prefix(&candidate, prefix) {
            shards.push(candidate);
        }
    }
    Ok(shards)
}

fn file_matches_prefix(path: &Path, prefix: &str) -> bool {
    path.is_file()
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(prefix))
}

fn require_shards(shards: &[PathBuf], path: &Path, prefix: &str) -> Result<()> {
    if shards.is_empty() {
        bail!(
            "no shards found under {} with prefix '{}'",
            path.display(),
            prefix
        );
    }
    Ok(())
}

fn print_shard(shard: &Path) {
    match parser::dissect_shard(shard) {
        Ok(entries) => print_entries(shard, &entries),
        Err(error) => eprintln!("warn: {}: {:#}", shard.display(), error),
    }
}

fn print_entries(shard: &Path, entries: &[RawTensor]) {
    println!("\n{}", shard.display());
    if entries.is_empty() {
        println!("  (no tensors found)");
        return;
    }
    println!("{}", tensor_table(entries));
}

fn tensor_table(entries: &[RawTensor]) -> Table {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Idx", "Role", "Dtype", "Shape", "Offset", "Nbytes"]);
    for (index, entry) in entries.iter().enumerate() {
        table.add_row(tensor_row(index, entry));
    }
    table
}

fn tensor_row(index: usize, entry: &RawTensor) -> Vec<Cell> {
    vec![
        Cell::new(index),
        Cell::new(entry.role.label()),
        Cell::new(entry.dtype.label()),
        Cell::new(entry.shape.render()),
        Cell::new(format!("{:#x}", entry.offset)),
        Cell::new(entry.nbytes),
    ]
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::test_support::write_hex_checkpoint;

    use super::run_dissect;

    fn write_dissect_hex_fixture() -> tempfile::TempDir {
        let root = write_hex_checkpoint("dissect");
        fs::write(root.path().join("tensor0001.pkl"), b"not-a-pickle")
            .expect("write garbage shard");
        root
    }

    #[test]
    fn dissect_hex_fixture_prints_tensor_table() {
        let root = write_dissect_hex_fixture();
        run_dissect(root.path(), Some(1), "tensor").expect("dissect hex fixture");
        run_dissect(root.path(), None, "tensor").expect("dissect includes unreadable shard warn");
        let empty = run_dissect(root.path(), Some(0), "missing").unwrap_err();
        assert!(format!("{empty:#}").contains("no shards found"));
        let not_dir = run_dissect(&root.path().join("tensor0000.pkl"), None, "tensor").unwrap_err();
        assert!(format!("{not_dir:#}").contains("is not a directory"));
    }
}
