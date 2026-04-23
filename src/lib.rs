// SPDX-License-Identifier: GPL-3.0-only
//
// xai-dissect: static structural analysis of Grok-family open weights.
// Copyright (C) 2026 xai-dissect contributors.
//
// Library entry point. The binary (src/main.rs) is a thin CLI on top of
// this crate; external consumers should depend on the library and the
// stable export schema produced by `xai_dissect::report`.

pub mod experts;
pub mod inventory;
pub mod parser;
pub mod report;
pub mod routing;
pub mod schema;
pub mod stats;
