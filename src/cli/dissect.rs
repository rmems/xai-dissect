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
    let failures: Vec<String> = shards
        .iter()
        .filter_map(|shard| shard_failure(shard))
        .collect();
    ensure_one_shard_parsed(path, shards.len(), &failures)
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

/// `None` when the shard parsed. The returned string is the warning detail
/// for a shard that did not.
fn shard_failure(shard: &Path) -> Option<String> {
    match parser::dissect_shard(shard) {
        Ok(entries) => {
            print_entries(shard, &entries);
            None
        }
        Err(error) => Some(warn_unreadable(shard, &format!("{error:#}"))),
    }
}

fn warn_unreadable(shard: &Path, detail: &str) -> String {
    eprintln!("warn: {}: {detail}", shard.display());
    format!("{}: {detail}", shard.display())
}

/// An empty selection (`--limit 0`) still succeeds. Total failure is only
/// when every shard that was actually selected failed to parse. A mix of
/// readable and unreadable shards warns and continues.
fn ensure_one_shard_parsed(path: &Path, selected: usize, failures: &[String]) -> Result<()> {
    if selected == 0 || failures.len() != selected {
        return Ok(());
    }
    let noun = if selected == 1 { "shard" } else { "shards" };
    bail!(
        "every selected shard failed to parse under {} ({selected} {noun}): {}",
        path.display(),
        failures.join("; ")
    )
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

    use crate::test_support::{unique_temp_dir, write_hex_checkpoint};

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

    #[test]
    fn dissect_errors_when_every_selected_shard_fails() {
        let root = unique_temp_dir("dissect-all-corrupt");
        fs::write(root.path().join("tensor0000.pkl"), b"not-a-pickle").expect("write shard");
        fs::write(root.path().join("tensor0001.pkl"), b"also-garbage").expect("write shard");

        let err = run_dissect(root.path(), None, "tensor").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("every selected shard failed to parse"),
            "{msg}"
        );
        assert!(msg.contains("2 shards"), "{msg}");
        assert!(msg.contains("tensor0000.pkl"), "{msg}");
        assert!(msg.contains("tensor0001.pkl"), "{msg}");
        assert!(msg.contains("not a pickle protocol 4"), "{msg}");
    }

    #[test]
    fn dissect_errors_when_limit_selects_only_corrupt_shards() {
        let root = write_hex_checkpoint("dissect-limit-corrupt");
        let valid = fs::read(root.path().join("tensor0000.pkl")).expect("read valid shard");
        fs::write(root.path().join("tensor0000.pkl"), b"not-a-pickle").expect("overwrite shard");
        fs::write(root.path().join("tensor0001.pkl"), valid).expect("write later valid shard");

        let err = run_dissect(root.path(), Some(1), "tensor").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("every selected shard failed to parse"),
            "{msg}"
        );
        assert!(msg.contains("1 shard"), "{msg}");
        assert!(msg.contains("tensor0000.pkl"), "{msg}");
        assert!(!msg.contains("tensor0001.pkl"), "{msg}");

        run_dissect(root.path(), None, "tensor").expect("later valid shard still parses");
    }

    #[test]
    fn dissect_limit_zero_selects_nothing_and_succeeds() {
        let root = write_hex_checkpoint("dissect-limit-zero");
        run_dissect(root.path(), Some(0), "tensor").expect("limit 0 selects no shards");
    }
}
