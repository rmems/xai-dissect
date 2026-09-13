// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Test helpers shared by the binary crate's two test modules —
//! `main::tests` and `cli::tests`. Compiled only under `cfg(test)`.
//!
//! The integration tests under `tests/` cannot reach into a binary crate, so
//! they keep their own copy in `tests/support/mod.rs`. Two homes is the floor
//! here, not an oversight; it was four.

use std::fs;
use std::path::Path;

use tempfile::TempDir;

/// The one parser fixture the CLI tests build a checkpoint from.
pub(crate) const PARSER_HEX_FIXTURE: &str = "tests/fixtures/parser/single_f32_tensor.pkl.hex";

/// Decode a whitespace-tolerant hex fixture, relative to the crate root.
pub(crate) fn decode_hex_fixture(rel_path: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel_path);
    let hex = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read fixture {}: {err}", path.display()));
    let hex = hex
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    assert_eq!(hex.len() % 2, 0, "fixture must have an even hex length");

    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut i = 0;
    while i < hex.len() {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16).expect("hex byte");
        out.push(byte);
        i += 2;
    }
    out
}

/// A uniquely-named temp dir that is removed when the returned handle drops.
pub(crate) fn unique_temp_dir(prefix: &str) -> TempDir {
    tempfile::Builder::new()
        .prefix(&format!("xai-dissect-{prefix}-"))
        .tempdir()
        .expect("create unique temp dir")
}

/// A fresh single-shard checkpoint directory holding [`PARSER_HEX_FIXTURE`].
pub(crate) fn write_hex_checkpoint(prefix: &str) -> TempDir {
    let root = unique_temp_dir(prefix);
    fs::write(
        root.path().join("tensor0000.pkl"),
        decode_hex_fixture(PARSER_HEX_FIXTURE),
    )
    .expect("write shard");
    root
}
