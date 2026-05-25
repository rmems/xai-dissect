// SPDX-License-Identifier: GPL-3.0-only
//
// Strict Grok-1 coverage validation. This module is intentionally separate
// from the generic inventory builder so parser/classification logic remains
// usable for partial scans while full Grok-1 exports fail closed.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};

use crate::schema::{
    Grok1CoverageCounts, Grok1CoverageManifest, Grok1UnknownSlot, ModelInventory, MoeProjection,
    QuantizedAttentionWidth, TensorDType, TensorInfo, TensorKind, TensorRole,
};

use super::SCHEMA_VERSION;

pub const GROK1_COVERAGE_SCHEMA_VERSION: u32 = 1;

const GROK1_EXPECTED_BLOCKS: u32 = 64;
const GROK1_EXPECTED_TENSORS: u64 = 770;
const GROK1_EXPECTED_ROUTERS: u64 = 64;
const GROK1_EXPECTED_EXPERT_FAMILIES: u64 = 64 * 3;
const GROK1_EXPECTED_VOCAB_SIZE: u64 = 131_072;
const GROK1_D_MODEL: u64 = 6_144;
const GROK1_D_FF: u64 = 32_768;
const GROK1_N_EXPERTS: u64 = 8;
const GROK1_BLOCK_SLOTS: u32 = 12;
const GROK1_EXPERT_UP_OR_GATE_SHAPE: [u64; 3] = [GROK1_N_EXPERTS, GROK1_D_MODEL, GROK1_D_FF];
const GROK1_EXPERT_DOWN_SHAPE: [u64; 3] = [GROK1_N_EXPERTS, GROK1_D_FF, GROK1_D_MODEL];
const GROK1_ATTENTION_NARROW_SHAPE: [u64; 2] = [GROK1_D_MODEL, 1_024];
const GROK1_ATTENTION_MODEL_WIDTH_SHAPE: [u64; 2] = [GROK1_D_MODEL, GROK1_D_MODEL];
const GROK1_BLOCK_NORM_SHAPE: [u64; 1] = [GROK1_D_MODEL];
const GROK1_ROUTER_SHAPE: [u64; 2] = [GROK1_D_MODEL, GROK1_N_EXPERTS];

/// Decide whether an inventory is complete enough to require strict Grok-1
/// coverage validation before export. This intentionally does not depend on
/// shard count, because completeness is a tensor/layout property and repacked
/// checkpoints can preserve all 770 tensors with a different file count.
pub fn should_validate_grok1_coverage(inv: &ModelInventory) -> bool {
    inv.model_family == "grok-1" && inv.tensors.len() as u64 >= GROK1_EXPECTED_TENSORS
}

/// Validate a complete Grok-1 inventory and emit a deterministic coverage
/// manifest. Complete Grok-1 manifests fail closed on missing blocks/tensors,
/// duplicate structural names, unexpected layouts, unknown tensors, or
/// incomplete hyperparameter metadata.
pub fn validate_grok1_complete_manifest(inv: &ModelInventory) -> Result<Grok1CoverageManifest> {
    if inv.model_family != "grok-1" {
        bail!("Grok-1 coverage validation requires model_family 'grok-1'");
    }

    let expected = Grok1CoverageCounts {
        blocks: GROK1_EXPECTED_BLOCKS,
        tensors: GROK1_EXPECTED_TENSORS,
        routers: GROK1_EXPECTED_ROUTERS,
        expert_families: GROK1_EXPECTED_EXPERT_FAMILIES,
        unknown_tensors: 0,
    };
    let mut errors = Vec::new();
    validate_grok1_metadata(inv, &mut errors);
    if !errors.is_empty() {
        return fail_grok1_coverage(errors);
    }

    let index = CoverageIndex::build(inv);
    let discovered = grok1_discovered_counts(inv, &index);
    compare_count("blocks", discovered.blocks, expected.blocks, &mut errors);
    compare_count("tensors", discovered.tensors, expected.tensors, &mut errors);
    compare_count("routers", discovered.routers, expected.routers, &mut errors);
    compare_count(
        "expert_families",
        discovered.expert_families,
        expected.expert_families,
        &mut errors,
    );
    compare_count(
        "unknown_tensors",
        discovered.unknown_tensors,
        expected.unknown_tensors,
        &mut errors,
    );

    validate_unique_source_keys(inv, &mut errors);
    validate_unique_structural_names(inv, &mut errors);
    validate_grok1_blocks(inv, &index, &mut errors);
    validate_grok1_expected_slots(&index, &mut errors);

    if !errors.is_empty() {
        return fail_grok1_coverage(errors);
    }

    let unknown_slots = grok1_unknown_slots(inv);
    let checksum = grok1_checksum(inv, &discovered, &unknown_slots);
    Ok(Grok1CoverageManifest {
        model_family: inv.model_family.clone(),
        schema_version: inv.schema_version,
        coverage_schema_version: GROK1_COVERAGE_SCHEMA_VERSION,
        validation: "pass".to_string(),
        checksum,
        expected,
        discovered,
        unknown_slots,
    })
}

fn fail_grok1_coverage<T>(errors: Vec<String>) -> Result<T> {
    bail!(
        "Grok-1 complete manifest validation failed: {}",
        errors.join("; ")
    )
}

struct CoverageIndex<'a> {
    by_block: BTreeMap<u32, Vec<&'a TensorInfo>>,
    by_slot: BTreeMap<(u32, u32), Vec<&'a TensorInfo>>,
}

impl<'a> CoverageIndex<'a> {
    fn build(inv: &'a ModelInventory) -> Self {
        let mut by_block: BTreeMap<u32, Vec<&TensorInfo>> = BTreeMap::new();
        let mut by_slot: BTreeMap<(u32, u32), Vec<&TensorInfo>> = BTreeMap::new();
        for tensor in &inv.tensors {
            if let Some(block_index) = tensor.block_index {
                by_block.entry(block_index).or_default().push(tensor);
                if let Some(slot) = tensor.block_slot {
                    by_slot.entry((block_index, slot)).or_default().push(tensor);
                }
            }
        }
        Self { by_block, by_slot }
    }
}

fn validate_grok1_metadata(inv: &ModelInventory, errors: &mut Vec<String>) {
    if inv.schema_version != SCHEMA_VERSION {
        errors.push(format!(
            "schema_version {} != expected {}",
            inv.schema_version, SCHEMA_VERSION
        ));
    }
    if inv.inferred.vocab_size != Some(GROK1_EXPECTED_VOCAB_SIZE) {
        errors.push(format!(
            "inferred vocab_size {:?} != expected {}",
            inv.inferred.vocab_size, GROK1_EXPECTED_VOCAB_SIZE
        ));
    }
    if inv.inferred.n_blocks != Some(GROK1_EXPECTED_BLOCKS) {
        errors.push(format!(
            "inferred n_blocks {:?} != expected {}",
            inv.inferred.n_blocks, GROK1_EXPECTED_BLOCKS
        ));
    }
    if inv.inferred.d_model != Some(GROK1_D_MODEL) {
        errors.push(format!(
            "inferred d_model {:?} != expected {}",
            inv.inferred.d_model, GROK1_D_MODEL
        ));
    }
    if inv.inferred.d_ff != Some(GROK1_D_FF) {
        errors.push(format!(
            "inferred d_ff {:?} != expected {}",
            inv.inferred.d_ff, GROK1_D_FF
        ));
    }
    if inv.inferred.n_experts != Some(GROK1_N_EXPERTS) {
        errors.push(format!(
            "inferred n_experts {:?} != expected {}",
            inv.inferred.n_experts, GROK1_N_EXPERTS
        ));
    }
}

fn grok1_discovered_counts(inv: &ModelInventory, index: &CoverageIndex<'_>) -> Grok1CoverageCounts {
    Grok1CoverageCounts {
        blocks: index.by_block.len() as u32,
        tensors: inv.tensors.len() as u64,
        routers: inv
            .tensors
            .iter()
            .filter(|tensor| matches!(tensor.kind, TensorKind::Router))
            .count() as u64,
        expert_families: inv
            .tensors
            .iter()
            .filter(|tensor| matches!(tensor.kind, TensorKind::MoeExpertProjection { .. }))
            .count() as u64,
        unknown_tensors: inv
            .tensors
            .iter()
            .filter(|tensor| matches!(tensor.kind, TensorKind::Unknown { .. }))
            .count() as u64,
    }
}

fn grok1_unknown_slots(inv: &ModelInventory) -> Vec<Grok1UnknownSlot> {
    let mut out = inv
        .tensors
        .iter()
        .filter_map(|tensor| match &tensor.kind {
            TensorKind::Unknown { reason } => Some(Grok1UnknownSlot {
                structural_name: grok1_structural_name(tensor),
                block_index: tensor.block_index,
                block_slot: tensor.block_slot,
                shape: tensor.shape.clone(),
                reason: reason.clone(),
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| a.structural_name.cmp(&b.structural_name));
    out
}

fn compare_count<T>(label: &str, actual: T, expected: T, errors: &mut Vec<String>)
where
    T: Copy + Eq + std::fmt::Display,
{
    if actual != expected {
        errors.push(format!("{label} {actual} != expected {expected}"));
    }
}

fn validate_grok1_blocks(
    inv: &ModelInventory,
    index: &CoverageIndex<'_>,
    errors: &mut Vec<String>,
) {
    for block_index in 0..GROK1_EXPECTED_BLOCKS {
        match index.by_block.get(&block_index) {
            Some(tensors) if tensors.len() == GROK1_BLOCK_SLOTS as usize => {}
            Some(tensors) => errors.push(format!(
                "block_{block_index:03} has {} tensors, expected {}",
                tensors.len(),
                GROK1_BLOCK_SLOTS
            )),
            None => errors.push(format!("missing block_{block_index:03}")),
        }
    }

    for block_index in index.by_block.keys() {
        if *block_index >= GROK1_EXPECTED_BLOCKS {
            errors.push(format!("unexpected block_{block_index:03}"));
        }
    }

    for tensor in &inv.tensors {
        if tensor.block_index.is_none()
            && !matches!(
                tensor.kind,
                TensorKind::TokenEmbedding | TensorKind::FinalNorm
            )
        {
            errors.push(format!(
                "unexpected unassigned tensor {}",
                grok1_structural_name(tensor)
            ));
        }
    }
}

fn validate_unique_structural_names(inv: &ModelInventory, errors: &mut Vec<String>) {
    let mut seen = BTreeSet::new();
    for tensor in &inv.tensors {
        let name = grok1_structural_name(tensor);
        if !seen.insert(name.clone()) {
            errors.push(format!("duplicate structural name {name}"));
        }
    }
}

fn validate_unique_source_keys(inv: &ModelInventory, errors: &mut Vec<String>) {
    let mut seen = BTreeSet::new();
    for tensor in &inv.tensors {
        let key = (tensor.shard_ordinal, tensor.in_shard_index);
        if !seen.insert(key) {
            errors.push(format!(
                "duplicate tensor key shard={} in_shard={}",
                tensor.shard_ordinal, tensor.in_shard_index
            ));
        }
    }
}

fn validate_grok1_expected_slots(index: &CoverageIndex<'_>, errors: &mut Vec<String>) {
    for block_index in 0..GROK1_EXPECTED_BLOCKS {
        for slot in 0..GROK1_BLOCK_SLOTS {
            let tensors = index
                .by_slot
                .get(&(block_index, slot))
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if tensors.is_empty() {
                errors.push(format!("missing block_{block_index:03}.slot_{slot:02}"));
            }
            if tensors.len() > 1 {
                errors.push(format!(
                    "duplicate block_{block_index:03}.slot_{slot:02} entries: {}",
                    tensors.len()
                ));
            }
            if let Some(tensor) = tensors.first() {
                validate_grok1_slot_signature(tensor, errors);
            }
        }
    }
}

fn validate_grok1_slot_signature(tensor: &TensorInfo, errors: &mut Vec<String>) {
    let Some(slot) = tensor.block_slot else {
        return;
    };
    let name = grok1_structural_name(tensor);
    let Some(spec) = grok1_slot_spec(slot) else {
        errors.push(format!("unexpected block slot {slot} at {name}"));
        return;
    };
    validate_tensor_signature(
        tensor,
        &name,
        spec.role,
        spec.dtype,
        spec.shape,
        |kind| spec.kind.matches(kind),
        errors,
    );
}

#[derive(Clone, Copy)]
struct Grok1SlotSpec {
    slot: u32,
    role: TensorRole,
    dtype: TensorDType,
    shape: &'static [u64],
    kind: Grok1ExpectedKind,
}

#[derive(Clone, Copy)]
enum Grok1ExpectedKind {
    MoeExpertProjection(MoeProjection),
    AttentionNarrow,
    AttentionModelWidth,
    BlockNorm,
    Router,
}

impl Grok1ExpectedKind {
    fn matches(self, kind: &TensorKind) -> bool {
        match self {
            Self::MoeExpertProjection(expected) => matches!(
                kind,
                TensorKind::MoeExpertProjection { projection } if *projection == expected
            ),
            Self::AttentionNarrow => matches!(
                kind,
                TensorKind::QuantizedAttentionProjection {
                    width: QuantizedAttentionWidth::Narrow
                }
            ),
            Self::AttentionModelWidth => matches!(
                kind,
                TensorKind::QuantizedAttentionProjection {
                    width: QuantizedAttentionWidth::ModelWidth
                }
            ),
            Self::BlockNorm => matches!(kind, TensorKind::BlockNorm),
            Self::Router => matches!(kind, TensorKind::Router),
        }
    }
}

static GROK1_SLOT_SPECS: [Grok1SlotSpec; GROK1_BLOCK_SLOTS as usize] = [
    Grok1SlotSpec {
        slot: 0,
        role: TensorRole::QuantWeight,
        dtype: TensorDType::I8,
        shape: &GROK1_EXPERT_UP_OR_GATE_SHAPE,
        kind: Grok1ExpectedKind::MoeExpertProjection(MoeProjection::Gate),
    },
    Grok1SlotSpec {
        slot: 1,
        role: TensorRole::QuantWeight,
        dtype: TensorDType::I8,
        shape: &GROK1_EXPERT_DOWN_SHAPE,
        kind: Grok1ExpectedKind::MoeExpertProjection(MoeProjection::Down),
    },
    Grok1SlotSpec {
        slot: 2,
        role: TensorRole::QuantWeight,
        dtype: TensorDType::I8,
        shape: &GROK1_EXPERT_UP_OR_GATE_SHAPE,
        kind: Grok1ExpectedKind::MoeExpertProjection(MoeProjection::Up),
    },
    Grok1SlotSpec {
        slot: 3,
        role: TensorRole::QuantWeight,
        dtype: TensorDType::I8,
        shape: &GROK1_ATTENTION_NARROW_SHAPE,
        kind: Grok1ExpectedKind::AttentionNarrow,
    },
    Grok1SlotSpec {
        slot: 4,
        role: TensorRole::QuantWeight,
        dtype: TensorDType::I8,
        shape: &GROK1_ATTENTION_MODEL_WIDTH_SHAPE,
        kind: Grok1ExpectedKind::AttentionModelWidth,
    },
    Grok1SlotSpec {
        slot: 5,
        role: TensorRole::QuantWeight,
        dtype: TensorDType::I8,
        shape: &GROK1_ATTENTION_MODEL_WIDTH_SHAPE,
        kind: Grok1ExpectedKind::AttentionModelWidth,
    },
    Grok1SlotSpec {
        slot: 6,
        role: TensorRole::QuantWeight,
        dtype: TensorDType::I8,
        shape: &GROK1_ATTENTION_NARROW_SHAPE,
        kind: Grok1ExpectedKind::AttentionNarrow,
    },
    Grok1SlotSpec {
        slot: 7,
        role: TensorRole::Tensor,
        dtype: TensorDType::F32,
        shape: &GROK1_BLOCK_NORM_SHAPE,
        kind: Grok1ExpectedKind::BlockNorm,
    },
    Grok1SlotSpec {
        slot: 8,
        role: TensorRole::Tensor,
        dtype: TensorDType::F32,
        shape: &GROK1_BLOCK_NORM_SHAPE,
        kind: Grok1ExpectedKind::BlockNorm,
    },
    Grok1SlotSpec {
        slot: 9,
        role: TensorRole::Tensor,
        dtype: TensorDType::F32,
        shape: &GROK1_BLOCK_NORM_SHAPE,
        kind: Grok1ExpectedKind::BlockNorm,
    },
    Grok1SlotSpec {
        slot: 10,
        role: TensorRole::Tensor,
        dtype: TensorDType::F32,
        shape: &GROK1_BLOCK_NORM_SHAPE,
        kind: Grok1ExpectedKind::BlockNorm,
    },
    Grok1SlotSpec {
        slot: 11,
        role: TensorRole::Tensor,
        dtype: TensorDType::F32,
        shape: &GROK1_ROUTER_SHAPE,
        kind: Grok1ExpectedKind::Router,
    },
];

fn grok1_slot_spec(slot: u32) -> Option<&'static Grok1SlotSpec> {
    GROK1_SLOT_SPECS.iter().find(|spec| spec.slot == slot)
}

fn validate_tensor_signature(
    tensor: &TensorInfo,
    name: &str,
    role: TensorRole,
    dtype: TensorDType,
    shape: &[u64],
    kind_matches: impl FnOnce(&TensorKind) -> bool,
    errors: &mut Vec<String>,
) {
    let actual = || {
        format!(
            "role={} dtype={} shape={} kind={}",
            tensor.role.label(),
            tensor.dtype.label(),
            tensor.shape.render(),
            tensor.kind.short_label()
        )
    };

    if tensor.role != role {
        errors.push(format!("unexpected tensor role at {name}: {}", actual()));
        return;
    }
    if tensor.dtype != dtype {
        errors.push(format!("unexpected tensor dtype at {name}: {}", actual()));
        return;
    }
    if tensor.shape.dims() != shape {
        errors.push(format!("unexpected tensor shape at {name}: {}", actual()));
        return;
    }
    if !kind_matches(&tensor.kind) {
        errors.push(format!("unexpected tensor kind at {name}: {}", actual()));
    }
}

fn grok1_structural_name(tensor: &TensorInfo) -> String {
    match (tensor.block_index, tensor.block_slot) {
        (Some(block), Some(slot)) => {
            format!(
                "block_{block:03}.slot_{slot:02}.{}",
                tensor.kind.short_label()
            )
        }
        (Some(block), None) => format!(
            "block_{block:03}.slot_unknown.{}",
            tensor.kind.short_label()
        ),
        (None, _) if matches!(tensor.kind, TensorKind::TokenEmbedding) => "embedding".to_string(),
        (None, _) if matches!(tensor.kind, TensorKind::FinalNorm) => "final_norm".to_string(),
        (None, Some(slot)) => format!("unassigned.slot_{slot:02}.{}", tensor.kind.short_label()),
        (None, None) => format!("unassigned.{}", tensor.kind.short_label()),
    }
}

fn grok1_checksum(
    inv: &ModelInventory,
    discovered: &Grok1CoverageCounts,
    unknown_slots: &[Grok1UnknownSlot],
) -> String {
    let mut lines = vec![
        format!("model_family={}", inv.model_family),
        format!("schema_version={}", inv.schema_version),
        format!("blocks={}", discovered.blocks),
        format!("tensors={}", discovered.tensors),
        format!("routers={}", discovered.routers),
        format!("expert_families={}", discovered.expert_families),
        format!("unknown_tensors={}", discovered.unknown_tensors),
    ];

    let mut tensor_lines = inv
        .tensors
        .iter()
        .map(|tensor| {
            format!(
                "tensor|{}|{}|{}|{}|{}|{}|{}",
                grok1_structural_name(tensor),
                tensor.shard_ordinal,
                tensor.in_shard_index,
                tensor.role.label(),
                tensor.dtype.label(),
                tensor.shape.render(),
                tensor.kind.short_label()
            )
        })
        .collect::<Vec<_>>();
    tensor_lines.sort();
    lines.extend(tensor_lines);
    for unknown in unknown_slots {
        lines.push(format!(
            "unknown|{}|{}",
            unknown.structural_name, unknown.reason
        ));
    }

    stable_fnv1a64(lines.join("\n").as_bytes())
}

fn stable_fnv1a64(bytes: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{hash:016x}")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::schema::{InferredHyperparams, InventoryTotals, MoeProjection, TensorShape};

    use super::super::{compute_totals, summarize_blocks};
    use super::*;

    #[test]
    fn validates_complete_grok1_coverage_manifest() {
        let inv = complete_grok1_inventory();

        let manifest = validate_grok1_complete_manifest(&inv).expect("valid manifest");

        assert_eq!(manifest.expected.tensors, 770);
        assert_eq!(manifest.discovered.tensors, 770);
        assert_eq!(manifest.expected.blocks, 64);
        assert_eq!(manifest.discovered.blocks, 64);
        assert_eq!(manifest.expected.routers, 64);
        assert_eq!(manifest.discovered.routers, 64);
        assert_eq!(manifest.expected.expert_families, 192);
        assert_eq!(manifest.discovered.expert_families, 192);
        assert!(manifest.unknown_slots.is_empty());
        assert_eq!(manifest.checksum, "fnv1a64:de5a1c978121c62c");
    }

    #[test]
    fn validates_repacked_grok1_inventory_by_tensor_count_not_shard_count() {
        let mut inv = complete_grok1_inventory();
        inv.shard_count = 42;

        assert!(should_validate_grok1_coverage(&inv));
        validate_grok1_complete_manifest(&inv).expect("repacked manifest remains complete");
    }

    #[test]
    fn skips_strict_coverage_for_intentionally_truncated_grok1_inventory() {
        let mut inv = complete_grok1_inventory();
        inv.tensors.pop();
        refresh_inventory_derived_fields(&mut inv);

        assert!(!should_validate_grok1_coverage(&inv));
    }

    #[test]
    fn grok1_coverage_fails_on_missing_block() {
        let mut inv = complete_grok1_inventory();
        inv.tensors.retain(|tensor| tensor.block_index != Some(10));
        refresh_inventory_derived_fields(&mut inv);

        let err = validate_grok1_complete_manifest(&inv).unwrap_err();

        assert!(format!("{err:#}").contains("missing block_010"));
    }

    #[test]
    fn grok1_coverage_fails_on_missing_tensor() {
        let mut inv = complete_grok1_inventory();
        inv.tensors
            .retain(|tensor| !(tensor.block_index == Some(7) && tensor.block_slot == Some(4)));
        refresh_inventory_derived_fields(&mut inv);

        let err = validate_grok1_complete_manifest(&inv).unwrap_err();

        assert!(format!("{err:#}").contains("missing block_007.slot_04"));
    }

    #[test]
    fn grok1_coverage_fails_on_duplicate_tensor_key() {
        let mut inv = complete_grok1_inventory();
        let duplicate = inv
            .tensors
            .iter()
            .find(|tensor| tensor.block_index == Some(3) && tensor.block_slot == Some(5))
            .expect("source tensor")
            .clone();
        inv.tensors.push(duplicate);
        refresh_inventory_derived_fields(&mut inv);

        let err = validate_grok1_complete_manifest(&inv).unwrap_err();

        assert!(format!("{err:#}").contains("duplicate structural name"));
        assert!(format!("{err:#}").contains("block_003.slot_05"));
    }

    #[test]
    fn grok1_coverage_fails_on_unexpected_key_layout() {
        let mut inv = complete_grok1_inventory();
        let tensor = inv
            .tensors
            .iter_mut()
            .find(|tensor| tensor.block_index == Some(4) && tensor.block_slot == Some(3))
            .expect("slot tensor");
        tensor.kind = TensorKind::Unknown {
            reason: "synthetic unexpected slot layout".to_string(),
        };
        refresh_inventory_derived_fields(&mut inv);

        let err = validate_grok1_complete_manifest(&inv).unwrap_err();

        assert!(format!("{err:#}").contains("unknown_tensors 1 != expected 0"));
        assert!(format!("{err:#}").contains("unexpected tensor kind at block_004.slot_03"));
    }

    #[test]
    fn grok1_coverage_fails_on_wrong_norm_slot_kind() {
        let mut inv = complete_grok1_inventory();
        let tensor = inv
            .tensors
            .iter_mut()
            .find(|tensor| tensor.block_index == Some(5) && tensor.block_slot == Some(7))
            .expect("slot tensor");
        tensor.kind = TensorKind::MoeScales;
        refresh_inventory_derived_fields(&mut inv);

        let err = validate_grok1_complete_manifest(&inv).unwrap_err();

        assert!(format!("{err:#}").contains("unexpected tensor kind at block_005.slot_07"));
    }

    #[test]
    fn grok1_coverage_fails_on_wrong_norm_slot_shape() {
        let mut inv = complete_grok1_inventory();
        let tensor = inv
            .tensors
            .iter_mut()
            .find(|tensor| tensor.block_index == Some(5) && tensor.block_slot == Some(9))
            .expect("slot tensor");
        tensor.shape = TensorShape::new(vec![GROK1_D_FF]);
        refresh_inventory_derived_fields(&mut inv);

        let err = validate_grok1_complete_manifest(&inv).unwrap_err();

        assert!(format!("{err:#}").contains("unexpected tensor shape at block_005.slot_09"));
    }

    #[test]
    fn grok1_coverage_fails_on_unresolved_moe_projection_slot() {
        let mut inv = complete_grok1_inventory();
        let tensor = inv
            .tensors
            .iter_mut()
            .find(|tensor| tensor.block_index == Some(5) && tensor.block_slot == Some(0))
            .expect("slot tensor");
        tensor.kind = TensorKind::MoeExpertProjection {
            projection: MoeProjection::Unresolved,
        };
        refresh_inventory_derived_fields(&mut inv);

        let err = validate_grok1_complete_manifest(&inv).unwrap_err();

        assert!(format!("{err:#}").contains("unexpected tensor kind at block_005.slot_00"));
    }

    #[test]
    fn grok1_coverage_fails_on_swapped_moe_projection_slot() {
        let mut inv = complete_grok1_inventory();
        let tensor = inv
            .tensors
            .iter_mut()
            .find(|tensor| tensor.block_index == Some(5) && tensor.block_slot == Some(2))
            .expect("slot tensor");
        tensor.kind = TensorKind::MoeExpertProjection {
            projection: MoeProjection::Gate,
        };
        refresh_inventory_derived_fields(&mut inv);

        let err = validate_grok1_complete_manifest(&inv).unwrap_err();

        assert!(format!("{err:#}").contains("unexpected tensor kind at block_005.slot_02"));
    }

    #[test]
    fn grok1_coverage_fails_on_wrong_vocab_size() {
        let mut inv = complete_grok1_inventory();
        inv.inferred.vocab_size = Some(131_071);

        let err = validate_grok1_complete_manifest(&inv).unwrap_err();

        assert!(format!("{err:#}").contains("inferred vocab_size"));
    }

    #[test]
    fn grok1_coverage_checksum_is_reproducible() {
        let inv = complete_grok1_inventory();

        let first = validate_grok1_complete_manifest(&inv)
            .expect("first manifest")
            .checksum;
        let second = validate_grok1_complete_manifest(&inv)
            .expect("second manifest")
            .checksum;

        assert_eq!(first, second);
    }

    fn complete_grok1_inventory() -> ModelInventory {
        let mut tensors = vec![
            complete_tensor(
                0,
                0,
                None,
                None,
                TensorKind::TokenEmbedding,
                TensorRole::Tensor,
                TensorDType::F32,
                vec![GROK1_EXPECTED_VOCAB_SIZE, GROK1_D_MODEL],
            ),
            complete_tensor(
                1,
                0,
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
                let shard = 2 + block * GROK1_BLOCK_SLOTS + slot;
                let (kind, role, dtype, shape) = match slot {
                    0 | 2 => (
                        TensorKind::MoeExpertProjection {
                            projection: if slot == 0 {
                                MoeProjection::Gate
                            } else {
                                MoeProjection::Up
                            },
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![GROK1_N_EXPERTS, GROK1_D_MODEL, GROK1_D_FF],
                    ),
                    1 => (
                        TensorKind::MoeExpertProjection {
                            projection: MoeProjection::Down,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![GROK1_N_EXPERTS, GROK1_D_FF, GROK1_D_MODEL],
                    ),
                    3 | 6 => (
                        TensorKind::QuantizedAttentionProjection {
                            width: QuantizedAttentionWidth::Narrow,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![GROK1_D_MODEL, 1_024],
                    ),
                    4 | 5 => (
                        TensorKind::QuantizedAttentionProjection {
                            width: QuantizedAttentionWidth::ModelWidth,
                        },
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![GROK1_D_MODEL, GROK1_D_MODEL],
                    ),
                    7..=10 => (
                        TensorKind::BlockNorm,
                        TensorRole::Tensor,
                        TensorDType::F32,
                        vec![GROK1_D_MODEL],
                    ),
                    11 => (
                        TensorKind::Router,
                        TensorRole::Tensor,
                        TensorDType::F32,
                        vec![GROK1_D_MODEL, GROK1_N_EXPERTS],
                    ),
                    _ => unreachable!(),
                };
                tensors.push(complete_tensor(
                    shard,
                    0,
                    Some(block),
                    Some(slot),
                    kind,
                    role,
                    dtype,
                    shape,
                ));
            }
        }

        let mut inv = ModelInventory {
            model_family: "grok-1".to_string(),
            checkpoint_path: PathBuf::from("/tmp/grok-1/ckpt-0"),
            shard_count: 770,
            inferred: InferredHyperparams {
                vocab_size: Some(GROK1_EXPECTED_VOCAB_SIZE),
                d_model: Some(GROK1_D_MODEL),
                n_experts: Some(GROK1_N_EXPERTS),
                d_ff: Some(GROK1_D_FF),
                n_blocks: Some(GROK1_EXPECTED_BLOCKS),
            },
            tensors,
            blocks: Vec::new(),
            totals: InventoryTotals::default(),
            schema_version: SCHEMA_VERSION,
        };
        refresh_inventory_derived_fields(&mut inv);
        inv
    }

    fn refresh_inventory_derived_fields(inv: &mut ModelInventory) {
        inv.blocks = summarize_blocks(&inv.tensors);
        inv.totals = compute_totals(&inv.tensors);
    }

    #[allow(clippy::too_many_arguments)]
    fn complete_tensor(
        shard_ordinal: u32,
        in_shard_index: u32,
        block_index: Option<u32>,
        block_slot: Option<u32>,
        kind: TensorKind,
        role: TensorRole,
        dtype: TensorDType,
        shape: Vec<u64>,
    ) -> TensorInfo {
        TensorInfo {
            shard_path: PathBuf::from(format!("/tmp/grok-1/ckpt-0/tensor{shard_ordinal:05}_000")),
            shard_ordinal,
            in_shard_index,
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
}
