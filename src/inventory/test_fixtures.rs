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
        fixture_tensor(
            checkpoint_path,
            0,
            None,
            None,
            TensorKind::TokenEmbedding,
            TensorRole::Tensor,
            TensorDType::F32,
            vec![GROK1_EXPECTED_VOCAB_SIZE, GROK1_D_MODEL],
        ),
        fixture_tensor(
            checkpoint_path,
            1,
            None,
            None,
            TensorKind::FinalNorm,
            TensorRole::Tensor,
            TensorDType::F32,
            vec![GROK1_D_MODEL],
        ),
    ];

    for block in 0..GROK1_EXPECTED_BLOCKS {
        for slot in 0..GROK1_BLOCK_SLOTS {
            let spec = &CANONICAL_BLOCK_SLOTS[slot as usize];
            tensors.push(fixture_tensor(
                checkpoint_path,
                2 + block * GROK1_BLOCK_SLOTS + slot,
                Some(block),
                Some(slot),
                spec.kind.clone(),
                spec.role,
                spec.dtype,
                spec.shape.to_vec(),
            ));
        }
    }

    tensors
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

/// One block slot's canonical signature.
struct SlotSignature {
    kind: TensorKind,
    role: TensorRole,
    dtype: TensorDType,
    shape: &'static [u64],
}

const EXPERT_UP_OR_GATE_SHAPE: [u64; 3] = [GROK1_N_EXPERTS, GROK1_D_MODEL, GROK1_D_FF];
const EXPERT_DOWN_SHAPE: [u64; 3] = [GROK1_N_EXPERTS, GROK1_D_FF, GROK1_D_MODEL];
const ATTENTION_NARROW_SHAPE: [u64; 2] = [GROK1_D_MODEL, ATTENTION_NARROW_WIDTH];
const ATTENTION_WIDE_SHAPE: [u64; 2] = [GROK1_D_MODEL, GROK1_D_MODEL];
const BLOCK_NORM_SHAPE: [u64; 1] = [GROK1_D_MODEL];
const ROUTER_SHAPE: [u64; 2] = [GROK1_D_MODEL, GROK1_N_EXPERTS];

const fn moe_slot(projection: MoeProjection, shape: &'static [u64]) -> SlotSignature {
    SlotSignature {
        kind: TensorKind::MoeExpertProjection { projection },
        role: TensorRole::QuantWeight,
        dtype: TensorDType::I8,
        shape,
    }
}

const fn attention_slot(width: QuantizedAttentionWidth, shape: &'static [u64]) -> SlotSignature {
    SlotSignature {
        kind: TensorKind::QuantizedAttentionProjection { width },
        role: TensorRole::QuantWeight,
        dtype: TensorDType::I8,
        shape,
    }
}

const fn f32_slot(kind: TensorKind, shape: &'static [u64]) -> SlotSignature {
    SlotSignature {
        kind,
        role: TensorRole::Tensor,
        dtype: TensorDType::F32,
        shape,
    }
}

/// The canonical 12-slot block layout.
///
/// Written out independently of production's `GROK1_SLOT_SPECS` on purpose —
/// see the module docs. It is a table there and a table here so the two read
/// side by side, but they remain two statements of the layout, not one.
static CANONICAL_BLOCK_SLOTS: [SlotSignature; GROK1_BLOCK_SLOTS as usize] = [
    moe_slot(MoeProjection::Gate, &EXPERT_UP_OR_GATE_SHAPE),
    moe_slot(MoeProjection::Down, &EXPERT_DOWN_SHAPE),
    moe_slot(MoeProjection::Up, &EXPERT_UP_OR_GATE_SHAPE),
    attention_slot(QuantizedAttentionWidth::Narrow, &ATTENTION_NARROW_SHAPE),
    attention_slot(QuantizedAttentionWidth::ModelWidth, &ATTENTION_WIDE_SHAPE),
    attention_slot(QuantizedAttentionWidth::ModelWidth, &ATTENTION_WIDE_SHAPE),
    attention_slot(QuantizedAttentionWidth::Narrow, &ATTENTION_NARROW_SHAPE),
    f32_slot(TensorKind::BlockNorm, &BLOCK_NORM_SHAPE),
    f32_slot(TensorKind::BlockNorm, &BLOCK_NORM_SHAPE),
    f32_slot(TensorKind::BlockNorm, &BLOCK_NORM_SHAPE),
    f32_slot(TensorKind::BlockNorm, &BLOCK_NORM_SHAPE),
    f32_slot(TensorKind::Router, &ROUTER_SHAPE),
];

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
