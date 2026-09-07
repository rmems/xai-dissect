// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Shared canonical-Grok-1 test fixtures. Compiled only under `cfg(test)`.
//!
//! `exports`, `planning`, and `grok1_coverage` each need "a complete,
//! canonical Grok-1 ckpt-0 inventory" as a starting point, and each grew its
//! own builder for it. The three copies agreed on the layout but not on how
//! they spelled it: `grok1_coverage` built its fixture from the named
//! `GROK1_*` constants, while `exports` and `planning` hardcoded `6_144`,
//! `8`, `32_768` and `131_072` as literals. A change to a canonical constant
//! therefore updated one fixture and silently desynchronized the other two —
//! the tests kept passing against a layout production no longer emits.
//!
//! One builder, every dimension derived from the constants in
//! [`super::grok1_coverage`].
//!
//! ## What this fixture is and is not
//!
//! It is a **positive control**: the canonical 770-tensor layout, built to
//! satisfy [`super::validate_grok1_complete_manifest`]. A test asserting that
//! the validator accepts it is a smoke test, not a discriminating one — that
//! was already true of all three predecessors, which hardcoded the same
//! numbers the validator checks.
//!
//! The discriminating tests are the ones that *mutate* what this returns
//! (wrong shape, cleared block mapping, unknown tensor kind, non-canonical
//! shard count) and assert the validator rejects the result. Those keep their
//! full power. The slot → shape mapping below is written out independently of
//! the production `GROK1_SLOT_SPECS` table for the same reason: so the
//! validator is still checked against a second statement of the layout rather
//! than against itself.

use std::path::{Path, PathBuf};

use crate::schema::{
    InferredHyperparams, ModelInventory, MoeProjection, QuantizedAttentionWidth, TensorDType,
    TensorInfo, TensorKind, TensorRole, TensorShape,
};

use super::grok1_coverage::{
    GROK1_BLOCK_SLOTS, GROK1_D_FF, GROK1_D_MODEL, GROK1_EXPECTED_BLOCKS, GROK1_EXPECTED_VOCAB_SIZE,
    GROK1_N_EXPERTS,
};
use super::{SCHEMA_VERSION, compute_totals, summarize_blocks};

/// Attention narrow width. Not a `GROK1_*` constant in its own right —
/// production spells it inline inside `GROK1_ATTENTION_NARROW_SHAPE`.
const ATTENTION_NARROW_WIDTH: u64 = 1_024;

/// The checkpoint path `exports` and `planning` fixtures have always used.
/// `exports` derives an output slug from it, so it is load-bearing there.
pub(crate) const CANONICAL_CHECKPOINT: &str = "/tmp/grok-1-official/ckpt-0";

const EXPERT_UP_OR_GATE_SHAPE: [u64; 3] = [GROK1_N_EXPERTS, GROK1_D_MODEL, GROK1_D_FF];
const EXPERT_DOWN_SHAPE: [u64; 3] = [GROK1_N_EXPERTS, GROK1_D_FF, GROK1_D_MODEL];
const ATTENTION_NARROW_SHAPE: [u64; 2] = [GROK1_D_MODEL, ATTENTION_NARROW_WIDTH];
const ATTENTION_WIDE_SHAPE: [u64; 2] = [GROK1_D_MODEL, GROK1_D_MODEL];
const BLOCK_NORM_SHAPE: [u64; 1] = [GROK1_D_MODEL];
const ROUTER_SHAPE: [u64; 2] = [GROK1_D_MODEL, GROK1_N_EXPERTS];
const TOKEN_EMBEDDING_SHAPE: [u64; 2] = [GROK1_EXPECTED_VOCAB_SIZE, GROK1_D_MODEL];

/// The canonical 12-slot block layout, as `(kind, shape)`.
///
/// Role and dtype are not stored: they follow from the kind, so
/// [`role_and_dtype`] states that rule once instead of repeating it on all
/// fourteen tensors a block and its singletons produce.
///
/// The kinds are spelled out in full rather than built by `const fn` helpers.
/// A `const fn` reachable only from a `static` initializer is const-evaluated
/// and never runs, so llvm-cov reports it as uncovered — real dead weight in
/// the patch-coverage number for code that cannot be executed.
static CANONICAL_BLOCK_SLOTS: [(TensorKind, &[u64]); GROK1_BLOCK_SLOTS as usize] = [
    (
        TensorKind::MoeExpertProjection {
            projection: MoeProjection::Gate,
        },
        &EXPERT_UP_OR_GATE_SHAPE,
    ),
    (
        TensorKind::MoeExpertProjection {
            projection: MoeProjection::Down,
        },
        &EXPERT_DOWN_SHAPE,
    ),
    (
        TensorKind::MoeExpertProjection {
            projection: MoeProjection::Up,
        },
        &EXPERT_UP_OR_GATE_SHAPE,
    ),
    (
        TensorKind::QuantizedAttentionProjection {
            width: QuantizedAttentionWidth::Narrow,
        },
        &ATTENTION_NARROW_SHAPE,
    ),
    (
        TensorKind::QuantizedAttentionProjection {
            width: QuantizedAttentionWidth::ModelWidth,
        },
        &ATTENTION_WIDE_SHAPE,
    ),
    (
        TensorKind::QuantizedAttentionProjection {
            width: QuantizedAttentionWidth::ModelWidth,
        },
        &ATTENTION_WIDE_SHAPE,
    ),
    (
        TensorKind::QuantizedAttentionProjection {
            width: QuantizedAttentionWidth::Narrow,
        },
        &ATTENTION_NARROW_SHAPE,
    ),
    (TensorKind::BlockNorm, &BLOCK_NORM_SHAPE),
    (TensorKind::BlockNorm, &BLOCK_NORM_SHAPE),
    (TensorKind::BlockNorm, &BLOCK_NORM_SHAPE),
    (TensorKind::BlockNorm, &BLOCK_NORM_SHAPE),
    (TensorKind::Router, &ROUTER_SHAPE),
];

/// Role and dtype follow from the kind in the canonical layout: every
/// quantized projection is an `i8` `quant_weight`, and every norm, router and
/// embedding is a bare `f32` tensor.
fn role_and_dtype(kind: &TensorKind) -> (TensorRole, TensorDType) {
    match kind {
        TensorKind::MoeExpertProjection { .. }
        | TensorKind::QuantizedAttentionProjection { .. } => {
            (TensorRole::QuantWeight, TensorDType::I8)
        }
        _ => (TensorRole::Tensor, TensorDType::F32),
    }
}

/// A complete, canonical Grok-1 ckpt-0 inventory at [`CANONICAL_CHECKPOINT`]:
/// 1 token embedding + 1 final norm + 64 blocks × 12 slots = 770 tensors.
pub(crate) fn canonical_grok1_inventory() -> ModelInventory {
    canonical_grok1_inventory_at(Path::new(CANONICAL_CHECKPOINT))
}

/// [`canonical_grok1_inventory`] rooted at an explicit checkpoint path, for
/// callers whose assertions depend on the derived slug or shard paths.
pub(crate) fn canonical_grok1_inventory_at(checkpoint_path: &Path) -> ModelInventory {
    let mut inv = ModelInventory {
        model_family: "grok-1".to_string(),
        checkpoint_path: checkpoint_path.to_path_buf(),
        shard_count: 770,
        inferred: canonical_hyperparams(),
        tensors: canonical_tensors(checkpoint_path),
        blocks: Vec::new(),
        totals: Default::default(),
        schema_version: SCHEMA_VERSION,
    };
    refresh_derived_fields(&mut inv);
    inv
}

fn canonical_hyperparams() -> InferredHyperparams {
    InferredHyperparams {
        vocab_size: Some(GROK1_EXPECTED_VOCAB_SIZE),
        d_model: Some(GROK1_D_MODEL),
        n_experts: Some(GROK1_N_EXPERTS),
        d_ff: Some(GROK1_D_FF),
        n_blocks: Some(GROK1_EXPECTED_BLOCKS),
    }
}

/// The 770 canonical tensors: an embedding singleton, a final-norm singleton,
/// then 64 blocks of the 12-slot layout.
fn canonical_tensors(checkpoint_path: &Path) -> Vec<TensorInfo> {
    let mut tensors = vec![
        block_tensor(
            checkpoint_path,
            0,
            None,
            None,
            TensorKind::TokenEmbedding,
            &TOKEN_EMBEDDING_SHAPE,
        ),
        block_tensor(
            checkpoint_path,
            1,
            None,
            None,
            TensorKind::FinalNorm,
            &BLOCK_NORM_SHAPE,
        ),
    ];

    for block in 0..GROK1_EXPECTED_BLOCKS {
        for slot in 0..GROK1_BLOCK_SLOTS {
            let (kind, shape) = &CANONICAL_BLOCK_SLOTS[slot as usize];
            tensors.push(block_tensor(
                checkpoint_path,
                2 + block * GROK1_BLOCK_SLOTS + slot,
                Some(block),
                Some(slot),
                kind.clone(),
                shape,
            ));
        }
    }

    tensors
}

/// One canonical tensor, with role and dtype derived from its kind.
fn block_tensor(
    checkpoint_path: &Path,
    shard_ordinal: u32,
    block_index: Option<u32>,
    block_slot: Option<u32>,
    kind: TensorKind,
    shape: &[u64],
) -> TensorInfo {
    let (role, dtype) = role_and_dtype(&kind);
    fixture_tensor(
        checkpoint_path,
        shard_ordinal,
        block_index,
        block_slot,
        kind,
        role,
        dtype,
        shape.to_vec(),
    )
}

/// Recompute `blocks` and `totals` after a test has mutated `tensors`.
///
/// Both come from the production helpers, so a fixture never carries totals
/// that disagree with what the real inventory pipeline would have produced —
/// `exports` and `planning` previously each kept a private re-implementation
/// of `compute_totals`, which meant those tests could not have caught a bug
/// in it.
pub(crate) fn refresh_derived_fields(inv: &mut ModelInventory) {
    inv.blocks = summarize_blocks(&inv.tensors);
    inv.totals = compute_totals(&inv.tensors);
}

/// Build one fixture tensor. `nbytes` is derived from dtype and shape rather
/// than passed in, so a fixture can never carry a byte count that contradicts
/// its own geometry.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the TensorInfo fields a fixture must vary; grouping them \
              into a struct would only move the same list one level out"
)]
pub(crate) fn fixture_tensor(
    checkpoint_path: &Path,
    shard_ordinal: u32,
    block_index: Option<u32>,
    block_slot: Option<u32>,
    kind: TensorKind,
    role: TensorRole,
    dtype: TensorDType,
    shape: Vec<u64>,
) -> TensorInfo {
    TensorInfo {
        shard_path: PathBuf::from(format!(
            "{}/tensor{shard_ordinal:05}_000",
            checkpoint_path.display()
        )),
        shard_ordinal,
        in_shard_index: 0,
        role,
        dtype,
        shape: TensorShape::new(shape.clone()),
        offset: 0,
        nbytes: dtype.itemsize() as u64 * shape.iter().product::<u64>(),
        kind,
        block_index,
        block_slot,
    }
}
