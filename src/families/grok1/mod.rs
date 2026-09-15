// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Grok-1 family profile: canonical layout constants and the
//! grok1-map-v1-clean coverage validator.
//!
//! Generic inventory can still parse partial or non-Grok-1 scans. Full
//! Grok-1 exports fail closed by calling [`should_validate_grok1_coverage`]
//! and [`validate_grok1_complete_manifest`] on this path.

mod coverage;

pub use coverage::{
    GROK1_COVERAGE_SCHEMA_VERSION, should_validate_grok1_coverage, validate_grok1_complete_manifest,
};

pub(crate) use coverage::{
    GROK1_BLOCK_SLOTS, GROK1_D_FF, GROK1_D_MODEL, GROK1_EXPECTED_BLOCKS, GROK1_EXPECTED_VOCAB_SIZE,
    GROK1_N_EXPERTS,
};
