// SPDX-License-Identifier: GPL-3.0-only
//
// xai-dissect: static structural analysis of Grok-family open weights.
// Copyright (C) 2026 xai-dissect contributors.
//
// Library entry point. The binary (src/main.rs) is a thin CLI on top of
// this crate. External consumers should favor the CLI and exported artifact
// schema first; the in-process Rust API is intentionally smaller and still
// secondary to the repository's parser/analysis workflow.

pub mod experts;
pub mod exports;
pub mod inventory;
pub mod parser;
pub mod report;
pub mod routing;
pub mod schema;
pub mod stats;
