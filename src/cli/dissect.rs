// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! The `dissect` subcommand: parse each shard of a checkpoint directory and
//! print a per-shard tensor table. Raw parser output only — no
//! classification, no grouping.

use anyhow::{Context, Result, bail};
use comfy_table::{Cell, ContentArrangement, Table, presets::UTF8_FULL};

use xai_dissect::parser;

pub(crate) fn run_dissect(
    path: &std::path::Path,
    limit: Option<usize>,
    prefix: &str,
) -> Result<()> {
    let md = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if !md.is_dir() {
        bail!("{} is not a directory", path.display());
    }

    let mut shards = Vec::new();
    for entry in std::fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
        let entry = entry.with_context(|| format!("read entry in {}", path.display()))?;
        let p = entry.path();
        if p.is_file()
            && p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(prefix))
                .unwrap_or(false)
        {
            shards.push(p);
        }
    }
    shards.sort();
    if shards.is_empty() {
        bail!(
            "no shards found under {} with prefix '{}'",
            path.display(),
            prefix
        );
    }
    if let Some(n) = limit {
        shards.truncate(n);
    }

    for shard in &shards {
        match parser::dissect_shard(shard) {
            Ok(entries) => {
                println!("\n{}", shard.display());
                if entries.is_empty() {
                    println!("  (no tensors found)");
                    continue;
                }
                let mut table = Table::new();
                table
                    .load_preset(UTF8_FULL)
                    .set_content_arrangement(ContentArrangement::Dynamic)
                    .set_header(vec!["Idx", "Role", "Dtype", "Shape", "Offset", "Nbytes"]);
                for (i, e) in entries.iter().enumerate() {
                    table.add_row(vec![
                        Cell::new(i),
                        Cell::new(e.role.label()),
                        Cell::new(e.dtype.label()),
                        Cell::new(e.shape.render()),
                        Cell::new(format!("{:#x}", e.offset)),
                        Cell::new(e.nbytes),
                    ]);
                }
                println!("{table}");
            }
            Err(e) => eprintln!("warn: {}: {:#}", shard.display(), e),
        }
    }

    Ok(())
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
