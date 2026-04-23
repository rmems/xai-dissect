// SPDX-License-Identifier: GPL-3.0-only
//
// Expert-level structural analysis derived from `ModelInventory`. This layer
// maps block-local MoE tensor families, associates each family with every
// expert index, and reports missing or irregular layout patterns.

use std::collections::BTreeMap;

use crate::schema::{
    ExpertAtlas, ExpertBlock, ExpertIssue, ExpertIssueCategory, ExpertIssueSeverity,
    ExpertNamingCheck, ExpertNamingPattern, ExpertSlice, ExpertSliceTensor, ExpertTensorRef,
    InferredHyperparams, ModelInventory, MoeProjection, ShardRange, TensorDType, TensorInfo,
    TensorKind, TensorRole, TensorShape,
};

pub const EXPERT_ATLAS_SCHEMA_VERSION: u32 = 1;

#[derive(Clone)]
struct Candidate<'a> {
    tensor: &'a TensorInfo,
    projection: MoeProjection,
    expert_count: Option<u64>,
    family_label: String,
    structural_name: String,
}

/// Build an expert atlas from an inventory. This pass never touches tensor
/// bodies; it operates solely on the normalized inventory records.
pub fn build_expert_atlas(inv: &ModelInventory) -> ExpertAtlas {
    let mut by_block: BTreeMap<u32, Vec<&TensorInfo>> = BTreeMap::new();
    for tensor in &inv.tensors {
        if let Some(block_index) = tensor.block_index {
            by_block.entry(block_index).or_default().push(tensor);
        }
    }

    let mut blocks: Vec<ExpertBlock> = Vec::new();
    let mut anomalies: Vec<ExpertIssue> = Vec::new();

    for (block_index, mut members) in by_block {
        members.sort_by_key(|t| {
            (
                t.block_slot.unwrap_or(u32::MAX),
                t.shard_ordinal,
                t.in_shard_index,
            )
        });
        let candidates = expert_candidates(
            block_index,
            &members,
            inv.inferred.n_experts,
            inv.inferred.d_model,
        );
        let block_expert_count = block_expert_count(block_index, &candidates, &mut anomalies);

        if candidates.is_empty() {
            anomalies.push(ExpertIssue {
                severity: ExpertIssueSeverity::Error,
                category: ExpertIssueCategory::MissingOrIrregularTensor,
                block_index: Some(block_index),
                tensor: None,
                message: "no expert-stacked tensors detected in this block".to_string(),
            });
        }

        let slots: Vec<u32> = candidates
            .iter()
            .filter_map(|c| c.tensor.block_slot)
            .collect();
        if !slots.is_empty() && !is_contiguous(&slots) {
            anomalies.push(ExpertIssue {
                severity: ExpertIssueSeverity::Warning,
                category: ExpertIssueCategory::LayoutAnomaly,
                block_index: Some(block_index),
                tensor: None,
                message: format!("expert tensor slots are not contiguous: {:?}", slots),
            });
        }

        let tensors = candidates
            .iter()
            .map(|candidate| ExpertTensorRef {
                shard_ordinal: candidate.tensor.shard_ordinal,
                in_shard_index: candidate.tensor.in_shard_index,
                block_slot: candidate.tensor.block_slot,
                role: candidate.tensor.role,
                dtype: candidate.tensor.dtype,
                shape: candidate.tensor.shape.clone(),
                kind_label: candidate.tensor.kind.short_label(),
                projection: candidate.projection,
                expert_axis: candidate.expert_count.map(|_| 0),
                expert_count: candidate.expert_count,
                family_label: candidate.family_label.clone(),
                structural_name: candidate.structural_name.clone(),
            })
            .collect::<Vec<_>>();

        let experts = build_expert_slices(block_index, block_expert_count, &candidates);

        blocks.push(ExpertBlock {
            block_index,
            shard_range: shard_range(&members),
            expert_count: block_expert_count,
            tensors,
            experts,
        });
    }

    let expected_experts_per_block =
        mode_u64(blocks.iter().filter_map(|b| b.expert_count).collect());
    let canonical_layout = canonical_layout_signature(&blocks);

    for block in &blocks {
        if let Some(expected) = expected_experts_per_block {
            if block.expert_count != Some(expected) {
                anomalies.push(ExpertIssue {
                    severity: ExpertIssueSeverity::Warning,
                    category: ExpertIssueCategory::NamingConsistency,
                    block_index: Some(block.block_index),
                    tensor: None,
                    message: format!(
                        "expert count {:?} does not match checkpoint-wide expectation {}",
                        block.expert_count, expected
                    ),
                });
            }
        }

        if let Some(expected_layout) = canonical_layout.as_ref() {
            let layout = layout_signature(block);
            if &layout != expected_layout {
                anomalies.push(ExpertIssue {
                    severity: ExpertIssueSeverity::Warning,
                    category: ExpertIssueCategory::LayoutAnomaly,
                    block_index: Some(block.block_index),
                    tensor: None,
                    message: format!("expert layout differs from canonical pattern: {:?}", layout),
                });
            }
        }
    }

    let naming_patterns = build_naming_patterns(&blocks);
    let naming_checks = build_naming_checks(&blocks, expected_experts_per_block, canonical_layout);

    ExpertAtlas {
        model_family: inv.model_family.clone(),
        checkpoint_path: inv.checkpoint_path.clone(),
        shard_count: inv.shard_count,
        inferred: InferredHyperparams {
            vocab_size: inv.inferred.vocab_size,
            d_model: inv.inferred.d_model,
            n_experts: inv.inferred.n_experts,
            d_ff: inv.inferred.d_ff,
            n_blocks: inv.inferred.n_blocks,
        },
        relevant_block_count: blocks.len() as u32,
        expected_experts_per_block,
        blocks,
        naming_patterns,
        naming_checks,
        anomalies,
        schema_version: EXPERT_ATLAS_SCHEMA_VERSION,
    }
}

fn expert_candidates<'a>(
    block_index: u32,
    members: &[&'a TensorInfo],
    expected_experts: Option<u64>,
    d_model: Option<u64>,
) -> Vec<Candidate<'a>> {
    let mut candidates = Vec::new();
    let mut ordinal = 0u32;

    for tensor in members {
        let Some(projection) = infer_expert_projection(tensor, expected_experts, d_model) else {
            continue;
        };
        let expert_count = tensor.shape.dims().first().copied();
        let family_label = match tensor.block_slot {
            Some(slot) => format!("expert_slot_{slot:02}"),
            None => {
                let label = format!("expert_tensor_{ordinal:02}");
                ordinal += 1;
                label
            }
        };
        let structural_name = format!("block_{block_index:03}.{}", family_label);
        candidates.push(Candidate {
            tensor,
            projection,
            expert_count,
            family_label,
            structural_name,
        });
    }

    candidates
}

fn infer_expert_projection(
    tensor: &TensorInfo,
    expected_experts: Option<u64>,
    d_model: Option<u64>,
) -> Option<MoeProjection> {
    match &tensor.kind {
        TensorKind::MoeExpertProjection { projection } => return Some(*projection),
        TensorKind::MoeScales => {
            if tensor.shape.dims().first().copied() == expected_experts {
                return Some(MoeProjection::Unresolved);
            }
        }
        _ => {}
    }

    if tensor.role != TensorRole::QuantWeight {
        return None;
    }
    if tensor.dtype != TensorDType::I8 {
        return None;
    }
    if tensor.shape.rank() != 3 {
        return None;
    }

    let dims = tensor.shape.dims();
    if let Some(expected) = expected_experts {
        if dims[0] != expected {
            return None;
        }
    }

    if let Some(dm) = d_model {
        if dims[2] == dm {
            return Some(MoeProjection::Down);
        }
        if dims[1] == dm {
            return Some(MoeProjection::Unresolved);
        }
    }

    Some(MoeProjection::Unresolved)
}

fn block_expert_count(
    block_index: u32,
    candidates: &[Candidate<'_>],
    anomalies: &mut Vec<ExpertIssue>,
) -> Option<u64> {
    let counts = candidates
        .iter()
        .filter_map(|c| c.expert_count)
        .collect::<Vec<_>>();
    let expected = mode_u64(counts.clone());

    if counts.is_empty() {
        return None;
    }

    if counts.iter().any(|count| Some(*count) != expected) {
        anomalies.push(ExpertIssue {
            severity: ExpertIssueSeverity::Warning,
            category: ExpertIssueCategory::MissingOrIrregularTensor,
            block_index: Some(block_index),
            tensor: None,
            message: format!("expert tensors disagree on expert count: {:?}", counts),
        });
    }

    expected
}

fn build_expert_slices(
    block_index: u32,
    expert_count: Option<u64>,
    candidates: &[Candidate<'_>],
) -> Vec<ExpertSlice> {
    let Some(expert_count) = expert_count else {
        return Vec::new();
    };

    let mut experts = Vec::new();
    for expert_index in 0..expert_count {
        let tensors = candidates
            .iter()
            .filter(|candidate| candidate.expert_count == Some(expert_count))
            .map(|candidate| ExpertSliceTensor {
                family_label: candidate.family_label.clone(),
                structural_name: format!(
                    "block_{block_index:03}.{}.expert_{expert_index:02}",
                    candidate.family_label
                ),
                source_shard_ordinal: candidate.tensor.shard_ordinal,
                source_in_shard_index: candidate.tensor.in_shard_index,
                source_block_slot: candidate.tensor.block_slot,
                projection: candidate.projection,
                dtype: candidate.tensor.dtype,
                slice_shape: TensorShape::new(candidate.tensor.shape.dims()[1..].to_vec()),
            })
            .collect::<Vec<_>>();

        experts.push(ExpertSlice {
            expert_index: expert_index as u32,
            tensors,
        });
    }

    experts
}

fn shard_range(members: &[&TensorInfo]) -> Option<ShardRange> {
    if members.is_empty() {
        return None;
    }
    let mut start = u32::MAX;
    let mut end_inclusive = 0u32;
    for tensor in members {
        start = start.min(tensor.shard_ordinal);
        end_inclusive = end_inclusive.max(tensor.shard_ordinal);
    }
    Some(ShardRange {
        start,
        end_inclusive,
    })
}

fn is_contiguous(values: &[u32]) -> bool {
    values
        .windows(2)
        .all(|window| window[1] == window[0].saturating_add(1))
}

fn canonical_layout_signature(blocks: &[ExpertBlock]) -> Option<Vec<String>> {
    let signatures = blocks.iter().map(layout_signature).collect::<Vec<_>>();
    mode_vec_string(signatures)
}

fn layout_signature(block: &ExpertBlock) -> Vec<String> {
    block
        .tensors
        .iter()
        .map(|tensor| {
            format!(
                "{}|{}|{}|{}",
                tensor.family_label,
                tensor.projection.label(),
                tensor.dtype.label(),
                tensor.shape.render()
            )
        })
        .collect()
}

fn build_naming_patterns(blocks: &[ExpertBlock]) -> Vec<ExpertNamingPattern> {
    #[derive(Default)]
    struct PatternAcc {
        projection: Option<MoeProjection>,
        block_slots: Vec<u32>,
        shapes: Vec<TensorShape>,
        blocks: u32,
    }

    let mut patterns: BTreeMap<String, PatternAcc> = BTreeMap::new();
    for block in blocks {
        let mut seen_in_block: Vec<String> = Vec::new();
        for tensor in &block.tensors {
            let entry = patterns.entry(tensor.family_label.clone()).or_default();
            entry.projection.get_or_insert(tensor.projection);
            if let Some(slot) = tensor.block_slot {
                if !entry.block_slots.contains(&slot) {
                    entry.block_slots.push(slot);
                }
            }
            if !entry.shapes.contains(&tensor.shape) {
                entry.shapes.push(tensor.shape.clone());
            }
            if !seen_in_block.contains(&tensor.family_label) {
                entry.blocks += 1;
                seen_in_block.push(tensor.family_label.clone());
            }
        }
    }

    patterns
        .into_iter()
        .map(|(family_label, mut acc)| {
            acc.block_slots.sort_unstable();
            ExpertNamingPattern {
                pattern: format!("block_{{block}}.{}.expert_{{expert}}", family_label),
                family_label,
                projection: acc.projection.unwrap_or(MoeProjection::Unresolved),
                block_slots: acc.block_slots,
                observed_shapes: acc.shapes,
                observed_blocks: acc.blocks,
            }
        })
        .collect()
}

fn build_naming_checks(
    blocks: &[ExpertBlock],
    expected_experts_per_block: Option<u64>,
    canonical_layout: Option<Vec<String>>,
) -> Vec<ExpertNamingCheck> {
    let family_counts = blocks
        .iter()
        .map(|b| b.tensors.len() as u64)
        .collect::<Vec<_>>();
    let expected_family_count = mode_u64(family_counts.clone());

    let expert_count_consistent = expected_experts_per_block.is_some()
        && blocks
            .iter()
            .all(|block| block.expert_count == expected_experts_per_block);
    let family_count_consistent = expected_family_count.is_some()
        && family_counts
            .iter()
            .all(|count| Some(*count) == expected_family_count);
    let layout_consistent = if let Some(canonical) = canonical_layout.as_ref() {
        blocks
            .iter()
            .all(|block| layout_signature(block) == *canonical)
    } else {
        false
    };

    vec![
        ExpertNamingCheck {
            check: "expert_blocks_present".to_string(),
            passed: !blocks.is_empty(),
            detail: if blocks.is_empty() {
                "no block-local expert tensors were discovered".to_string()
            } else {
                format!("discovered expert tensors in {} blocks", blocks.len())
            },
        },
        ExpertNamingCheck {
            check: "expert_count_consistent".to_string(),
            passed: expert_count_consistent,
            detail: match expected_experts_per_block {
                Some(expected) => format!("expected {} experts per block", expected),
                None => "could not derive a checkpoint-wide expert count".to_string(),
            },
        },
        ExpertNamingCheck {
            check: "expert_family_count_consistent".to_string(),
            passed: family_count_consistent,
            detail: match expected_family_count {
                Some(expected) => format!("expected {} expert tensor families per block", expected),
                None => "could not derive a canonical expert-family count".to_string(),
            },
        },
        ExpertNamingCheck {
            check: "expert_layout_pattern_consistent".to_string(),
            passed: layout_consistent,
            detail: canonical_layout
                .map(|layout| format!("canonical inferred pattern: {:?}", layout))
                .unwrap_or_else(|| "could not derive a canonical expert layout".to_string()),
        },
    ]
}

fn mode_u64(values: Vec<u64>) -> Option<u64> {
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for value in values {
        *counts.entry(value).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(value, count)| (*count, *value))
        .map(|(value, _)| value)
}

fn mode_vec_string(values: Vec<Vec<String>>) -> Option<Vec<String>> {
    let mut counts: BTreeMap<Vec<String>, usize> = BTreeMap::new();
    for value in values {
        *counts.entry(value).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(value, count)| (*count, value.len()))
        .map(|(value, _)| value)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::schema::{
        InferredHyperparams, InventoryTotals, ModelInventory, TensorDType, TensorInfo, TensorKind,
        TensorRole, TensorShape,
    };

    use super::build_expert_atlas;

    #[test]
    fn atlas_discovers_expert_blocks_and_slices() {
        let inv = inventory_with_blocks(vec![
            block(
                0,
                vec![
                    tensor(
                        1,
                        0,
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 6144, 32768],
                        TensorKind::MoeExpertProjection {
                            projection: crate::schema::MoeProjection::Unresolved,
                        },
                    ),
                    tensor(
                        2,
                        1,
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 32768, 6144],
                        TensorKind::MoeExpertProjection {
                            projection: crate::schema::MoeProjection::Down,
                        },
                    ),
                    tensor(
                        3,
                        2,
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 6144, 32768],
                        TensorKind::MoeExpertProjection {
                            projection: crate::schema::MoeProjection::Unresolved,
                        },
                    ),
                ],
            ),
            block(
                1,
                vec![
                    tensor(
                        13,
                        0,
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 6144, 32768],
                        TensorKind::MoeExpertProjection {
                            projection: crate::schema::MoeProjection::Unresolved,
                        },
                    ),
                    tensor(
                        14,
                        1,
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 32768, 6144],
                        TensorKind::MoeExpertProjection {
                            projection: crate::schema::MoeProjection::Down,
                        },
                    ),
                    tensor(
                        15,
                        2,
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 6144, 32768],
                        TensorKind::MoeExpertProjection {
                            projection: crate::schema::MoeProjection::Unresolved,
                        },
                    ),
                ],
            ),
        ]);

        let atlas = build_expert_atlas(&inv);

        assert_eq!(atlas.relevant_block_count, 2);
        assert_eq!(atlas.expected_experts_per_block, Some(8));
        assert!(atlas.anomalies.is_empty());
        assert_eq!(atlas.blocks[0].experts.len(), 8);
        assert_eq!(atlas.blocks[0].experts[0].tensors.len(), 3);
        assert!(atlas.naming_checks.iter().all(|check| check.passed));
    }

    #[test]
    fn atlas_flags_missing_expert_family() {
        let inv = inventory_with_blocks(vec![
            block(
                0,
                vec![
                    tensor(
                        1,
                        0,
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 6144, 32768],
                        TensorKind::MoeExpertProjection {
                            projection: crate::schema::MoeProjection::Unresolved,
                        },
                    ),
                    tensor(
                        2,
                        1,
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 32768, 6144],
                        TensorKind::MoeExpertProjection {
                            projection: crate::schema::MoeProjection::Down,
                        },
                    ),
                    tensor(
                        3,
                        2,
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 6144, 32768],
                        TensorKind::MoeExpertProjection {
                            projection: crate::schema::MoeProjection::Unresolved,
                        },
                    ),
                ],
            ),
            block(
                1,
                vec![
                    tensor(
                        13,
                        0,
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 6144, 32768],
                        TensorKind::MoeExpertProjection {
                            projection: crate::schema::MoeProjection::Unresolved,
                        },
                    ),
                    tensor(
                        14,
                        1,
                        TensorRole::QuantWeight,
                        TensorDType::I8,
                        vec![8, 32768, 6144],
                        TensorKind::MoeExpertProjection {
                            projection: crate::schema::MoeProjection::Down,
                        },
                    ),
                ],
            ),
        ]);

        let atlas = build_expert_atlas(&inv);

        assert!(
            atlas
                .anomalies
                .iter()
                .any(|issue| issue.message.contains("canonical pattern"))
        );
        assert!(
            atlas
                .naming_checks
                .iter()
                .any(|check| check.check == "expert_family_count_consistent" && !check.passed)
        );
    }

    fn inventory_with_blocks(blocks: Vec<Vec<TensorInfo>>) -> ModelInventory {
        let tensors = blocks.into_iter().flatten().collect::<Vec<_>>();
        ModelInventory {
            model_family: "grok-1".to_string(),
            checkpoint_path: PathBuf::from("/tmp/grok-1"),
            shard_count: tensors.len() as u32,
            inferred: InferredHyperparams {
                vocab_size: Some(131072),
                d_model: Some(6144),
                n_experts: Some(8),
                d_ff: Some(32768),
                n_blocks: Some(2),
            },
            tensors,
            blocks: Vec::new(),
            totals: InventoryTotals::default(),
            schema_version: 1,
        }
    }

    fn block(block_index: u32, tensors: Vec<TensorInfo>) -> Vec<TensorInfo> {
        tensors
            .into_iter()
            .enumerate()
            .map(|(slot, mut tensor)| {
                tensor.block_index = Some(block_index);
                tensor.block_slot = Some(slot as u32);
                tensor
            })
            .collect()
    }

    fn tensor(
        shard_ordinal: u32,
        in_shard_index: u32,
        role: TensorRole,
        dtype: TensorDType,
        shape: Vec<u64>,
        kind: TensorKind,
    ) -> TensorInfo {
        TensorInfo {
            shard_path: PathBuf::from(format!("/tmp/tensor{shard_ordinal:05}_000")),
            shard_ordinal,
            in_shard_index,
            role,
            dtype,
            shape: TensorShape::new(shape),
            offset: 0,
            nbytes: 0,
            kind,
            block_index: None,
            block_slot: None,
        }
    }
}
